"""Verify CPU voice launches are limited before importing model dependencies."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice_server


class VoiceLimitTests(unittest.TestCase):
    def test_fixed_os_limits_and_cpu_affinity_wrap_the_child(self):
        process = Mock()
        args = ["/opt/venv/bin/python", "/opt/worker/runtime.py", "/tmp/input.json", "/tmp/output.json"]
        with patch.object(voice_server.os, "sched_getaffinity", return_value={5, 7}, create=True), \
                patch.object(voice_server.subprocess, "Popen", return_value=process) as launch:
            self.assertIs(voice_server.limited_process(args, env={"SAFE": "1"}, start_new_session=True), process)
        command = launch.call_args.args[0]
        self.assertEqual(command[:10], ["/usr/bin/prlimit", "--as=4294967296", "--cpu=120:125", "--nproc=128", "--nofile=128", "--",
                                       "/usr/bin/taskset", "--cpu-list", "5", args[0]])
        self.assertEqual(command[-len(args):], args)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
