import re
import os
from datetime import timedelta

import modal
from temporalio import workflow

from modal_temporal import (
    modal_activity,
    modal_activity_cls,
    modal_activity_method,
    run_dispatcher,
)

APP_NAME = "temporal-testing"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .uv_pip_install("temporalio==1.27.0", "async-lru==2.3.0")
    .add_local_python_source("modal_temporal")
)


env: dict[str, str | None] = {
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
    greeting: str = modal.parameter()

    @modal_activity_method
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

    This function does not actually run the Temporal activity and should not take many resources.

    An alternative is to run this queuer on a machine external to Modal.
    """
    await run_dispatcher(
        APP_NAME,
        task_queue="my-task-queue",
        workflows=[SayHelloWorkflow],
        activities=[greet, word_count, add_two, SayHello(greeting="You are great").run],
    )


if __name__ == "__main__":
    # Deploy and trigger the initial queuer
    with modal.enable_output():
        app.deploy()
        func = modal.Function.from_name("temporal-testing", "queuer")
        func.spawn()
