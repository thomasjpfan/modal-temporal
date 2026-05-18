import os
import sys
import asyncio
import inspect
import modal
from functools import wraps
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, Any, Callable, ParamSpec, TypeVar, Sequence
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    Worker,
    Interceptor,
    ActivityInboundInterceptor,
    ExecuteActivityInput,
)
from temporalio.client import Client
from async_lru import alru_cache

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

HEARTBEAT_INTERVAL_SECONDS = 2.0


async def heartbeat_loop(
    handle, activity_name: str, activity_task: asyncio.Task
) -> None:
    """Temporal keeps a heartbeat to make sure the activity is running."""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            await handle.heartbeat()
            print(f"[external worker] heartbeat sent for {activity_name}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[external worker] heartbeat failed: {e}")
            # Heartbeat failure usually means Temporal cancelled or expired the activity.
            activity_task.cancel()
            return


async def run_activity(fn: Callable, args: Any, client: Client, task_token: bytes):
    if inspect.iscoroutinefunction(fn):
        coro = fn(*args)
    else:
        coro = asyncio.to_thread(fn, *args)
    return await run_activity_with_temporal(coro, fn.__name__, client, task_token)


async def run_activity_with_temporal(
    coro: Coroutine,
    activity_name: str,
    client: Client,
    task_token: bytes,
):
    handle = client.get_async_activity_handle(task_token=task_token)
    activity_task = asyncio.create_task(coro)
    hb_task = asyncio.create_task(heartbeat_loop(handle, activity_name, activity_task))

    try:
        result = await activity_task
        await handle.complete(result)
        print(f"[external worker] completed {activity_name} -> {result!r}")
        return result
    except asyncio.CancelledError:
        await handle.report_cancellation()
        return
    except Exception as e:
        await handle.fail(ApplicationError(str(e)))
        print(f"[external worker] failed {activity_name}: {e}")
    finally:
        hb_task.cancel()


@alru_cache(maxsize=1)
async def get_temporal_client() -> Client:
    return await Client.connect(
        os.environ["TEMPORAL_SERVER"], namespace=os.environ["TEMPORAL_NAMESPACE"]
    )


@dataclass(frozen=True)
class _Runner:
    modal_name: str  # name the handler is deployed under in the Modal app
    is_class: bool


# Temporal activity_type -> how the dispatcher reaches its Modal handler.
# Populated at import time, read by DispatchActivityInterceptor. There is no
# naming convention: each activity records its handler here explicitly.
REGISTRY: dict[str, _Runner] = {}


def modal_activity(
    app: modal.App, **modal_opts: Any
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register a function as a Temporal activity AND build + record the Modal
    function that runs it. Returns the Temporal activity to pass to the Worker."""

    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        temporal_activity: Callable[P, R] = activity.defn(f)

        @wraps(f)
        async def runner(task_token: bytes, /, args: Any):
            client = await get_temporal_client()
            return await run_activity(f, args, client, task_token)

        modal_name = f"{f.__name__}_runner"
        runner.__name__ = runner.__qualname__ = modal_name
        # Modal references a non-serialized function by f"{module}:{qualname}"
        # and re-imports that module in the container. @wraps put the activity's
        # module on `runner`; bind it there as a real global so the lookup
        # resolves. The decorator re-runs on the remote import and re-binds it.
        setattr(sys.modules[f.__module__], modal_name, runner)
        app.function(**modal_opts)(runner)

        REGISTRY[f.__name__] = _Runner(modal_name, is_class=False)
        return temporal_activity

    return decorate


def modal_activity_cls(
    app: modal.App, **modal_opts: Any
) -> Callable[[type[T]], type[T]]:
    """Class analogue of @modal_activity. Decorate the activity class itself: a
    runner @app.cls() is generated from its __init__ params and public async
    methods, bound into the activity's module so Modal can reference it by
    symbol (no serialized=True; the decorator re-runs on the remote import and
    re-binds it), and recorded in REGISTRY. Returns the class unchanged.

    Assumes bare @activity.defn on the methods => activity name == method
    __name__ (same convention as @modal_activity)."""

    def decorate(cls: type[T]) -> type[T]:
        # Dynamically add `modal.parameters`
        anns: dict[str, Any] = {}
        ns: dict[str, Any] = {}
        param_names: list[str] = []
        for p in inspect.signature(cls.__init__).parameters.values():
            if p.name == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            param_names.append(p.name)
            anns[p.name] = (
                p.annotation if p.annotation is not inspect.Parameter.empty else Any
            )
            ns[p.name] = (
                modal.parameter()
                if p.default is inspect.Parameter.empty
                else modal.parameter(default=p.default)
            )
        ns["__annotations__"] = anns

        # Dynamically add a `modal.enter()`
        async def _start(self):
            self._activity = cls(**{n: getattr(self, n) for n in param_names})

        ns["start"] = modal.enter()(_start)

        # Dynamically add methods from activities
        def make_method(method_name: str):
            async def _run(self, task_token: bytes, args: Any):
                client = await get_temporal_client()
                return await run_activity(
                    getattr(self._activity, method_name), args, client, task_token
                )

            return modal.method()(_run)

        methods = [
            n
            for n, fn in vars(cls).items()
            if not n.startswith("_") and inspect.iscoroutinefunction(fn)
        ]
        for n in methods:
            ns[n] = make_method(n)

        runner_name = f"{cls.__name__}Runner"
        runner_cls = type(runner_name, (), ns)
        # Same trick as @modal_activity: make the generated runner a real
        # f"{module}:{qualname}" symbol in the activity's module so Modal's
        # by-reference lookup resolves. type() set __module__ to this module;
        # point it at the activity's module instead.
        runner_cls.__module__ = cls.__module__
        runner_cls.__qualname__ = runner_name
        setattr(sys.modules[cls.__module__], runner_name, runner_cls)
        app.cls(**modal_opts)(runner_cls)

        for n in methods:
            REGISTRY[n] = _Runner(runner_name, is_class=True)
        return cls

    return decorate


@alru_cache()
async def get_modal_function(app_name: str, func_name: str) -> modal.Function:
    return await modal.Function.from_name(app_name, func_name).hydrate.aio()


@alru_cache()
async def get_modal_cls(app_name: str, cls_name: str) -> modal.Cls:
    return await modal.Cls.from_name(app_name, cls_name).hydrate.aio()


class DispatchActivityInterceptor(ActivityInboundInterceptor):
    def __init__(self, next: ActivityInboundInterceptor, app_name: str) -> None:
        super().__init__(next)
        self._app_name = app_name

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        info = activity.info()
        task_token = info.task_token
        args = list(input.args)

        entry = REGISTRY[info.activity_type]

        if entry.is_class:
            # input.fn is a bound method, e.g. SayHello().run. The instance's
            # attributes map to the Modal Cls parameters.
            instance = input.fn.__self__
            modal_cls = await get_modal_cls(self._app_name, entry.modal_name)
            handle = getattr(modal_cls(**vars(instance)), input.fn.__name__)
        else:
            handle = await get_modal_function(self._app_name, entry.modal_name)

        print(f"[dispatcher] {info.activity_type} -> {entry.modal_name} args={args}")
        await handle.spawn.aio(task_token, args)
        activity.raise_complete_async()


class DispatchInterceptor(Interceptor):
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def intercept_activity(
        self, next: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return DispatchActivityInterceptor(next, self._app_name)


async def run_dispatcher(
    app_name: str,
    *,
    task_queue: str,
    workflows: Sequence[type],
    activities: Sequence[Callable],
    max_workers: int = 4,
    **worker_kwargs: Any,
) -> None:
    """Run the Temporal worker that dispatches activities to Modal.

    Owns only the dispatcher plumbing: the Temporal client, the
    DispatchInterceptor, and a thread pool so sync activities can run. Extra
    keyword args are forwarded to temporalio's Worker, so callers can set any
    Worker parameter. Special-cased so the plumbing is not lost:
    - `interceptors` you pass are appended after the DispatchInterceptor.
    - `activity_executor`, if given, replaces the default thread pool.
    """
    client = await get_temporal_client()
    interceptors = [
        DispatchInterceptor(app_name),
        *worker_kwargs.pop("interceptors", []),
    ]
    worker_kwargs.setdefault(
        "activity_executor", ThreadPoolExecutor(max_workers=max_workers)
    )
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=list(workflows),
        activities=list(activities),
        interceptors=interceptors,
        **worker_kwargs,
    )
    print(f"[dispatcher] worker on {task_queue!r}; activities run on Modal")
    await worker.run()
