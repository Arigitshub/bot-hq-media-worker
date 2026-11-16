"""Bounded child-process execution shared by Runpod and private HTTP transports."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading

from policy import MAX_JSON_BYTES, MAX_SECONDS, RequestError, failure, validate

_gate = threading.Lock()


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=3)


def execute(value, *, timeout=MAX_SECONDS, launcher=subprocess.Popen):
    try:
        value = validate(value)
    except RequestError as error:
        return failure(error.code, str(error))
    if not _gate.acquire(blocking=False):
        return failure("worker_busy", "One media request is already running. No request was queued.")
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="bot-hq-job-") as directory:
            path = Path(directory)
            request_file, result_file = path / "request.json", path / "result.json"
            request_file.write_text(json.dumps(value), encoding="utf-8")
            os.chmod(request_file, 0o600)
            # No provider credentials or unrelated application secrets reach a model process.
            allowed_env = ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                           "NVIDIA_DRIVER_CAPABILITIES", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_DISABLE_TELEMETRY",
                           "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
            environment = {name: os.environ[name] for name in allowed_env if name in os.environ}
            environment.setdefault("LANG", "C.UTF-8")
            environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            environment["OMP_NUM_THREADS"] = "2"
            environment["BOT_HQ_PARENT_PID"] = str(os.getpid())
            process = launcher([sys.executable, str(Path(__file__).with_name("runtime.py")), str(request_file), str(result_file)],
                               cwd=str(Path(__file__).parent), env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=(os.name == "posix"))
            try:
                exit_code = process.wait(timeout=min(MAX_SECONDS, max(1, timeout)))
            except subprocess.TimeoutExpired:
                stop_process(process)
                return failure("generation_timeout", "The media job reached its time limit and was stopped. No automatic retry was made.")
            if exit_code != 0 or not result_file.is_file():
                return failure("worker_failed", "The isolated model process ended without a result.")
            if not 1 <= result_file.stat().st_size <= MAX_JSON_BYTES:
                return failure("result_size", "The model result exceeded the response limit.")
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                return failure("invalid_result", "The model returned an invalid result.")
            return result
    except Exception:
        return failure("worker_failed", "The isolated model process could not complete the request.")
    finally:
        if process is not None and process.poll() is None:
            stop_process(process)
        _gate.release()
