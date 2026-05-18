import re
import os
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import modal
from temporalio import activity, workflow
from async_lru import alru_cache
from temporalio.client import Client
from temporalio.worker import Worker

from modal_temporal import (
    DispatchInterceptor,
    get_temporal_client,
    modal_activity,
    modal_activity_cls,
)

APP_NAME = "temporal-testing"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("temporalio==1.27.0", "async-lru==2.3.0")
    .add_local_python_source("modal_temporal")
)


env: dict[str, str] = {
    "TEMPORAL_SERVER": os.environ["TEMPORAL_SERVER"],
    "TEMPORAL_NAMESPACE": os.environ["TEMPORAL_NAMESPACE"],
}


@modal_activity(app, env=env, image=image, cpu=0.5)
async def greet(name: str) -> str:
    return f"Hello {name}"


@modal_activity(app, env=env, image=image, cpu=1)
def word_count(text: str) -> int:
    return len(re.findall(r"\b[a-zA-Z]+\b", text))


@modal_activity(app, env=env, image=image)
async def add_two(value: int) -> int:
    return value + 2


# Class based activity
@modal_activity_cls(app, env=env, image=image)
class SayHello:
    def __init__(self, greeting: str = "Hello again"):
        self.greeting = greeting

    @activity.defn
    async def run(self, name: str) -> str:
        return f"{self.greeting}, {name}!"


@workflow.defn
class SayHelloWorkflow:
    @workflow.run
    async def run(self, name: str) -> int:
        result = await workflow.execute_activity(
            greet, name, schedule_to_close_timeout=timedelta(seconds=30)
        )
        more_result = await workflow.execute_activity_method(
            SayHello.run, result, schedule_to_close_timeout=timedelta(seconds=30)
        )
        count = await workflow.execute_activity(
            word_count, more_result, schedule_to_close_timeout=timedelta(seconds=30)
        )
        return await workflow.execute_activity(
            add_two, count, schedule_to_close_timeout=timedelta(seconds=30)
        )


@app.function(schedule=modal.Period(minutes=30), timeout=30 * 60, image=image, env=env)
async def queuer():
    """Pulls task from Temporal's task queue and immediately places it on Modal input queue.

    This function does not actually run the Temporal activity and should not take many resourc.es

    An alternative is to run this queuer on a machine external to Modal.
    """
    client = await get_temporal_client()
    greeting_activity = SayHello(greeting="You are great")
    worker = Worker(
        client,
        task_queue="my-task-queue",
        workflows=[SayHelloWorkflow],
        activities=[greet, word_count, add_two, greeting_activity.run],
        interceptors=[DispatchInterceptor(APP_NAME)],
        # Add a thread pool executor so we can run sync activities
        activity_executor=ThreadPoolExecutor(max_workers=4),
    )
    print("Dispatcher worker started. Activities will be completed by a Modal function")
    await worker.run()


if __name__ == "__main__":
    # Deploy and trigger the initial queuer
    with modal.enable_output():
        app.deploy()
        func = modal.Function.from_name("temporal-testing", "queuer")
        func.spawn()
