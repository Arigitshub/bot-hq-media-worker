"""Optional per-modality queue entrypoint with exact Runpod host-cache mapping."""
import os
from pathlib import Path

from policy import MODELS, RequestError, failure, validate
from runner import execute

HOST_CACHE = Path("/runpod-volume/huggingface-cache/hub")


def link_host_snapshot(kind, target_cache, host_cache=HOST_CACHE):
    """Link only the approved immutable revision; never select another snapshot."""
    model = MODELS[kind]
    folder = "models--" + model["id"].replace("/", "--")
    source = host_cache / folder / "snapshots" / model["revision"]
    if not source.is_dir():
        return False
    sentinel = "model_index.json" if kind in ("image", "video") else "config.json"
    if not (source / sentinel).is_file():
        return False
    destination = target_cache / folder / "snapshots" / model["revision"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        return destination.resolve() == source.resolve()
    if destination.exists():
        # A separately downloaded cache must not silently replace the host cache.
        return False
    destination.symlink_to(source, target_is_directory=True)
    return True


def handler(job):
    if not isinstance(job, dict) or "input" not in job:
        return failure("invalid_job", "A Runpod input object is required.")
    try:
        value = validate(job["input"])
    except RequestError as error:
        return failure(error.code, str(error))
    configured = os.environ.get("BOT_HQ_KIND", "all")
    if configured not in ("all", "image", "video", "speech", "transcribe"):
        return failure("worker_configuration", "The worker modality is not configured.")
    if configured != "all" and value["kind"] != configured:
        return failure("unsupported_kind", "This endpoint does not serve that media type.")
    if os.environ.get("BOT_HQ_REQUIRE_HOST_CACHE", "0") == "1":
        try:
            target = Path(os.environ.get("HF_HUB_CACHE", "/cache/huggingface/hub"))
            if not link_host_snapshot(value["kind"], target):
                return failure("model_cache_missing", "The pinned model cache is not ready. No download was started.")
        except OSError:
            return failure("model_cache_unavailable", "The pinned model cache could not be accessed.")
    return execute(value)


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda _: 1})
