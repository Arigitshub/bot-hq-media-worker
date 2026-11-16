"""CPU-only loopback service; per-job OS limits apply before model imports."""
import argparse
import os
from pathlib import Path
import subprocess

from http_server import PrivateHttpServer, handler_type, read_token
from runner import execute


def limited_process(args, **kwargs):
    # Only an administrator-installed fixed runtime command reaches this launcher.
    # Address-space and CPU-time limits belong to each disposable model process,
    # not the long-running HTTP server. One logical CPU leaves room for the app.
    cpu = min(os.sched_getaffinity(0))
    command = ["/usr/bin/prlimit", "--as=4294967296", "--cpu=120:125", "--nproc=128", "--nofile=128", "--",
               "/usr/bin/taskset", "--cpu-list", str(cpu), *args]
    return subprocess.Popen(command, **kwargs)


def execute_cpu(value):
    return execute(value, timeout=300, launcher=limited_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "posix" or os.getuid() == 0:
        parser.error("Run this service as its unprivileged Linux cloud account.")
    token = read_token(args.token_file)
    with PrivateHttpServer(("127.0.0.1", 8766), handler_type(token, ("speech", "transcribe"), executor=execute_cpu)) as server:
        server.serve_forever()
