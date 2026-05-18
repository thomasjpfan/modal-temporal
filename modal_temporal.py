import os
import sys
import asyncio
import inspect
import modal
from functools import wraps
from dataclasses import dataclass
from typing import Coroutine, Any, Callable, ParamSpec, TypeVar
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    Interceptor,
    ActivityInboundInterceptor,
    ExecuteActivityInput,
)
from temporalio.client import Client
from async_lru import alru_cache

P = ParamSpec("P")
R = TypeVar("R")

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


def modal_activity(app: modal.App, **modal_opts):
    """Register a function as a Temporal activity AND build + record the Modal
    function that runs it. Returns the Temporal activity to pass to the Worker."""

    def decorate(f: Callable) -> Callable:
        temporal_activity = activity.defn(f)

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


def register_modal_cls(activity_method: Callable, runner_cls: type) -> None:
    """Explicitly pair a class-based activity method (e.g. SayHello.run) with
    its @app.cls() runner. Keyed by the Temporal activity name so it matches
    activity.info().activity_type at dispatch time."""
    defn = activity._Definition.from_callable(activity_method)
    if defn is None:
        raise ValueError(f"{activity_method!r} is not an @activity.defn")
    REGISTRY[defn.name] = _Runner(runner_cls.__name__, is_class=True)


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
