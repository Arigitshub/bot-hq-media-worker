"""Offline contracts and isolation checks. No model imports, downloads or GPU use."""
import base64
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy
import runner
import runtime
from http_server import JobStore, PrivateHttpServer, handler_type


class InputTests(unittest.TestCase):
    def test_fixed_generation_options(self):
        self.assertEqual(policy.validate({"kind": "image", "prompt": " a cat "}), {"kind": "image", "prompt": "a cat", "seed": 42})
        for extra in ({"width": 4096}, {"model": "arbitrary/model"}, {"url": "http://internal"}, {"steps": 900}):
            with self.assertRaises(policy.RequestError):
                policy.validate({"kind": "image", "prompt": "cat", **extra})

    def test_bounds_and_strict_types(self):
        for value in (None, [], {"kind": []}, {"kind": "shell"}, {"kind": "image", "prompt": " "},
                      {"kind": "image", "prompt": "x" * 1201}, {"kind": "speech", "text": "x" * 601},
                      {"kind": "video", "prompt": "cat", "seed": True}, {"kind": "video", "prompt": "cat", "seed": -1}):
            with self.assertRaises(policy.RequestError):
                policy.validate(value)

    def test_audio_validation(self):
        data = {"kind": "transcribe", "audio_mime": "audio/wav", "audio_base64": base64.b64encode(b"RIFFtest").decode()}
        self.assertEqual(policy.validate(data), data)
        for changed in ({"audio_base64": "data:audio/wav;base64,abcd"}, {"audio_base64": "!invalid"},
                        {"audio_mime": "text/html"}, {"audio_base64": "a" * (8 * 1024 * 1024 + 4)}):
            with self.assertRaises(policy.RequestError):
                policy.validate({**data, **changed})

    def test_pinned_models(self):
        for kind in ("image", "video", "speech", "transcribe"):
            self.assertRegex(policy.MODELS[kind]["revision"], r"^[a-f0-9]{40}$")
        self.assertLessEqual(policy.MODELS["video"]["frames"] / policy.MODELS["video"]["fps"], 4)

    def test_artifact_hash_and_limits(self):
        result = policy.artifact_result("image", b"example", "image/png")
        self.assertEqual(result["sha256"], hashlib.sha256(b"example").hexdigest())
        self.assertEqual(base64.b64decode(result["artifact_base64"]), b"example")
        for content, mime in ((b"", "image/png"), (b"x", "text/html"), (b"a" * (policy.MAX_ARTIFACT_BYTES + 1), "image/png")):
            with self.assertRaises(policy.RequestError):
                policy.artifact_result("image", content, mime)


class FakeProcess:
    pid = 987654321

    def __init__(self, args, **kwargs):
        self.args, self.kwargs, self.returncode = args, kwargs, None
        Path(args[-1]).write_text(json.dumps({"ok": True, "kind": "image"}))

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode


class RunnerTests(unittest.TestCase):
    def test_invalid_input_never_launches(self):
        with patch("runner.subprocess.Popen") as launch:
            result = runner.execute({"kind": "shell"}, launcher=launch)
            self.assertFalse(result["ok"])
            launch.assert_not_called()

    def test_private_model_process_excludes_secrets(self):
        seen = []
        def launch(args, **kwargs):
            process = FakeProcess(args, **kwargs)
            seen.append(process)
            return process
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "never-forward", "MEDIA_TOKEN": "never-forward", "OPENAI_API_KEY": "never-forward"}):
            self.assertTrue(runner.execute({"kind": "image", "prompt": "cat"}, launcher=launch)["ok"])
        self.assertEqual(len(seen), 1)
        environment = seen[0].kwargs["env"]
        self.assertFalse({"RUNPOD_API_KEY", "MEDIA_TOKEN", "OPENAI_API_KEY"} & environment.keys())
        self.assertFalse(Path(seen[0].args[-1]).exists(), "Per-job request/result directory must be removed")

    def test_one_concurrent_job_no_queue(self):
        runner._gate.acquire()
        try:
            result = runner.execute({"kind": "image", "prompt": "cat"})
            self.assertEqual(result["error"]["code"], "worker_busy")
        finally:
            runner._gate.release()

    def test_timeout_stops_once_and_releases_slot(self):
        class TimeoutProcess(FakeProcess):
            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired(self.args, timeout)
        def stopped(process):
            process.returncode = -15
        with patch("runner.stop_process", side_effect=stopped) as stop:
            result = runner.execute({"kind": "image", "prompt": "cat"}, launcher=TimeoutProcess, timeout=2)
            self.assertEqual(result["error"]["code"], "generation_timeout")
            self.assertEqual(stop.call_count, 1)
        self.assertTrue(runner._gate.acquire(blocking=False))
        runner._gate.release()


class DecoderTests(unittest.TestCase):
    def test_fixed_decoder_and_duration(self):
        value = {"audio_mime": "audio/webm", "audio_base64": base64.b64encode(b"fake-media").decode()}
        seen = []
        def decode(args, **kwargs):
            seen.append((args, kwargs))
            with wave.open(args[-1], "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\0\0" * 16000)
            return type("Result", (), {"returncode": 0})()
        with tempfile.TemporaryDirectory() as directory, patch("runtime.subprocess.run", side_effect=decode):
            _, seconds = runtime.normalized_audio(value, directory)
        self.assertEqual(seconds, 1)
        args, kwargs = seen[0]
        self.assertEqual(args[args.index("-f") + 1], "matroska")
        self.assertEqual(args[args.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertEqual(args[args.index("-t") + 1], "61")
        self.assertEqual(kwargs["timeout"], 20)
        self.assertNotIn("shell", kwargs)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def execute(value):
            self.calls.append(value)
            return {"ok": True, "kind": value["kind"]}
        self.server = PrivateHttpServer(("127.0.0.1", 0), handler_type("t" * 48, ["image", "speech"], execute))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, value=None, token="t" * 48):
        req = urllib.request.Request(self.base + path, data=None if value is None else json.dumps(value).encode(),
                                     headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_authentication_and_metadata_are_not_inference(self):
        self.assertEqual(self.request("/health", token="wrong")[0], 401)
        status, data = self.request("/health")
        self.assertEqual(status, 200)
        self.assertFalse(data["inference_verified"])
        self.assertEqual(self.calls, [])

    def test_capability_boundary_and_authorized_execution(self):
        self.assertEqual(self.request("/generate", {"kind": "video", "prompt": "cat"})[0], 400)
        self.assertEqual(self.request("/generate", {"kind": "image", "prompt": "cat"}, token="wrong")[0], 401)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.request("/generate", {"kind": "image", "prompt": "cat"})[0], 200)
        self.assertEqual(len(self.calls), 1)

    def test_async_submission_and_authenticated_result(self):
        status, job = self.request("/jobs", {"kind": "image", "prompt": "cat"})
        self.assertEqual(status, 202)
        self.assertRegex(job["id"], r"^media_[a-f0-9]{32}$")
        self.assertEqual(self.request("/jobs/" + job["id"], token="wrong")[0], 401)
        for _ in range(20):
            _, result = self.request("/jobs/" + job["id"])
            if result["status"] != "running":
                break
            time.sleep(0.01)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["output"]["ok"])


class AsyncTests(unittest.TestCase):
    def test_idempotent_admission_and_no_backlog(self):
        release, began = threading.Event(), threading.Event()
        calls = []
        def execute(value):
            calls.append(value)
            began.set()
            release.wait(timeout=2)
            return {"ok": True, "kind": value["kind"]}
        jobs = JobStore(execute)
        value = {"kind": "image", "prompt": "cat"}
        try:
            first = jobs.submit(value, "same-request-123")
            self.assertTrue(began.wait(timeout=1))
            same = jobs.submit(value, "same-request-123")
            self.assertEqual(first["id"], same["id"])
            with self.assertRaisesRegex(policy.RequestError, "different input"):
                jobs.submit({"kind": "image", "prompt": "dog"}, "same-request-123")
            with self.assertRaisesRegex(policy.RequestError, "already running"):
                jobs.submit(value, "other-request-123")
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
        for _ in range(100):
            result = jobs.get(first["id"])
            if result["status"] != "running":
                break
            time.sleep(0.01)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(jobs.get("missing"))

    def test_failed_model_is_not_reported_completed(self):
        jobs = JobStore(lambda _: {"ok": False, "error": {"code": "generation_failed"}})
        record = jobs.submit({"kind": "speech", "text": "hello"})
        for _ in range(100):
            result = jobs.get(record["id"])
            if result["status"] != "running":
                break
            time.sleep(0.01)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["output"]["ok"])


if __name__ == "__main__":
    unittest.main()
