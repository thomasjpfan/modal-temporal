import os
import asyncio
import inspect
import modal
from functools import wraps
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


def get_name(activity_name: str) -> str:
    return f"{activity_name}_runner"


def get_cls_name(class_name: str) -> str:
    return f"{class_name}Runner"


def modal_activity(f: Callable):
    @wraps(f)
    async def wrapper(task_token: bytes, /, args: Any):
        client = await get_temporal_client()
        return await run_activity(f, args, client, task_token)

    func_name = get_name(f.__name__)
    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    return wrapper


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
        task_token = activity.info().task_token
        args = list(input.args)

        if inspect.ismethod(input.fn):
            # Class-based: input.fn is a bound method, e.g. SayHello().run.
            # The instance's attributes map to the Modal Cls parameters.
            instance = input.fn.__self__
            cls_name = type(instance).__name__
            method_name = input.fn.__name__
            params = vars(instance)
            modal_cls = await get_modal_cls(self._app_name, get_cls_name(cls_name))
            method = getattr(modal_cls(**params), method_name)
            print(
                f"[dispatcher] class activity={cls_name}.{method_name} "
                f"params={params} args={args} -> external worker"
            )
            await method.spawn.aio(task_token, args)
        else:
            activity_name = input.fn.__name__
            modal_func = await get_modal_function(
                self._app_name, get_name(activity_name)
            )
            print(
                f"[dispatcher] activity={activity_name} args={args} -> external worker"
            )
            await modal_func.spawn.aio(task_token, args)

        activity.raise_complete_async()


class DispatchInterceptor(Interceptor):
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def intercept_activity(
        self, next: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return DispatchActivityInterceptor(next, self._app_name)
