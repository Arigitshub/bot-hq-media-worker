"""Private authenticated HTTP alternative for a GPU Pod or cloud CPU speech host.

Expose through HTTPS only. This is a backend service, never a public browser API.
"""
import argparse
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import threading
import time
import uuid
import re

from policy import MAX_JSON_BYTES, MODELS, RequestError, failure, validate
from runner import execute


class PrivateHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.connections = threading.BoundedSemaphore(16)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class JobStore:
    """Small authenticated backend queue: one active job, no pending backlog."""
    def __init__(self, executor):
        self.executor, self.lock, self.jobs, self.keys, self.active = executor, threading.Lock(), {}, {}, False

    def _prune(self):
        cutoff = time.monotonic() - 1800
        for job_id, record in list(self.jobs.items()):
            if record["status"] != "running" and (record["created"] < cutoff or len(self.jobs) >= 20):
                self.jobs.pop(job_id)
                if record.get("key"):
                    self.keys.pop(record["key"], None)

    def public(self, record):
        return {key: record[key] for key in ("id", "status", "output") if key in record}

    def submit(self, value, key=None):
        value = validate(value)
        if key is not None and (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", key)):
            raise RequestError("invalid_idempotency_key", "Use a stable 8-100 character request identifier.")
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        with self.lock:
            self._prune()
            if key and key in self.keys:
                record = self.jobs[self.keys[key]]
                if record["digest"] != digest:
                    raise RequestError("idempotency_conflict", "That request identifier was already used for different input.")
                return self.public(record)
            if self.active:
                raise RequestError("worker_busy", "One media request is already running. No request was queued.")
            job_id = "media_" + uuid.uuid4().hex
            record = {"id": job_id, "status": "running", "created": time.monotonic(), "digest": digest, "key": key}
            self.jobs[job_id] = record
            if key:
                self.keys[key] = job_id
            self.active = True
            thread = threading.Thread(target=self._work, args=(job_id, value), daemon=True)
            thread.start()
            return self.public(record)

    def _work(self, job_id, value):
        try:
            output = self.executor(value)
        except Exception:
            output = failure("worker_failed", "The media job failed without a result.")
        with self.lock:
            record = self.jobs[job_id]
            record["output"] = output
            record["status"] = "completed" if output.get("ok") else "failed"
            self.active = False

    def get(self, job_id):
        with self.lock:
            self._prune()
            record = self.jobs.get(job_id)
            return self.public(record) if record else None


def handler_type(token, kinds, executor=execute):
    jobs = JobStore(executor)
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass

        def json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def auth(self):
            entries = self.headers.get_all("Authorization", [])
            return len(entries) == 1 and hmac.compare_digest(entries[0].encode(), ("Bearer " + token).encode())

        def do_GET(self):
            if not self.auth():
                return self.json(401, failure("unauthorized", "Authentication required."))
            if re.fullmatch(r"/jobs/media_[a-f0-9]{32}", self.path):
                record = jobs.get(self.path.split("/")[-1])
                return self.json(200, record) if record else self.json(404, failure("job_not_found", "Job not found or expired."))
            if self.path != "/health":
                return self.json(404, failure("not_found", "Route not found."))
            return self.json(200, {"ok": True, "kind": "configuration", "models": {k: MODELS[k]["id"] for k in kinds},
                                   "inference_verified": False, "max_concurrent_jobs": 1, "async_jobs": True})

        def do_POST(self):
            if not self.auth():
                return self.json(401, failure("unauthorized", "Authentication required."))
            if self.path not in ("/generate", "/jobs"):
                return self.json(404, failure("not_found", "Route not found."))
            try:
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) != 1:
                    return self.json(400, failure("invalid_framing", "A fixed request body is required."))
                length = int(self.headers["Content-Length"])
                if not 1 <= length <= MAX_JSON_BYTES:
                    return self.json(413, failure("request_size", "The request is too large."))
                if self.headers.get_content_type() != "application/json":
                    return self.json(415, failure("content_type", "Send JSON content."))
                body = self.rfile.read(length)
                if len(body) != length:
                    return self.json(400, failure("incomplete_request", "The request was incomplete."))
                value = json.loads(body)
                if not isinstance(value, dict) or value.get("kind") not in kinds:
                    return self.json(400, failure("unsupported_kind", "This service does not provide the requested capability."))
            except (ValueError, UnicodeError):
                return self.json(400, failure("invalid_json", "Invalid JSON request."))
            if self.path == "/jobs":
                try:
                    record = jobs.submit(value, self.headers.get("Idempotency-Key"))
                    return self.json(202, record)
                except RequestError as error:
                    code = 429 if error.code == "worker_busy" else (409 if error.code == "idempotency_conflict" else 400)
                    return self.json(code, failure(error.code, str(error)))
            result = executor(value)
            code = 200 if result.get("ok") else (429 if result.get("error", {}).get("code") == "worker_busy" else 422)
            return self.json(code, result)
    return Handler


def read_token(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 512:
            raise ValueError("Use a regular private token file.")
        if os.name == "posix" and info.st_mode & 0o077:
            raise ValueError("Token file must be readable only by its service owner.")
        token = handle.read().strip()
    if not 32 <= len(token) <= 256 or any(c.isspace() for c in token):
        raise ValueError("Token must contain 32-256 non-whitespace characters.")
    return token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "0.0.0.0"])
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--kinds", default="image,video,transcribe,speech")
    args = parser.parse_args()
    kinds = args.kinds.split(",")
    if not kinds or not set(kinds) <= {"image", "video", "transcribe", "speech"} or not 1024 <= args.port <= 65535:
        parser.error("Invalid service capabilities or port.")
    token = read_token(args.token_file)
    with PrivateHttpServer((args.host, args.port), handler_type(token, kinds)) as server:
        server.serve_forever()
