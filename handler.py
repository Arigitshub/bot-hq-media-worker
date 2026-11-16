"""Runpod queue endpoint entry point. Credentials belong to the app, not models."""
from runner import execute
from policy import failure


def handler(job):
    if not isinstance(job, dict) or "input" not in job:
        return failure("invalid_job", "A Runpod input object is required.")
    return execute(job["input"])


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda _: 1})
