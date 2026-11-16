"""No cloud calls: cache selection and endpoint admission contracts only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serverless_entry
from policy import MODELS


class ServerlessTests(unittest.TestCase):
    def test_modality_mismatch_never_launches(self):
        with patch.dict(os.environ, {"BOT_HQ_KIND": "image"}), patch.object(serverless_entry, "execute") as execute:
            result = serverless_entry.handler({"input": {"kind": "video", "prompt": "boat"}})
        self.assertEqual(result["error"]["code"], "unsupported_kind")
        execute.assert_not_called()

    def test_missing_cache_fails_without_download(self):
        with patch.dict(os.environ, {"BOT_HQ_KIND": "image", "BOT_HQ_REQUIRE_HOST_CACHE": "1"}), \
                patch.object(serverless_entry, "link_host_snapshot", return_value=False), \
                patch.object(serverless_entry, "execute") as execute:
            result = serverless_entry.handler({"input": {"kind": "image", "prompt": "boat"}})
        self.assertEqual(result["error"]["code"], "model_cache_missing")
        execute.assert_not_called()

    @unittest.skipIf(os.name == "nt", "Actual symlink creation is tested in the Linux cloud build")
    def test_links_exact_revision_without_copying_or_selecting_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = "models--" + MODELS["image"]["id"].replace("/", "--")
            revision = MODELS["image"]["revision"]
            snapshots = root / "host" / folder / "snapshots"
            wrong = snapshots / ("0" * 40)
            wrong.mkdir(parents=True)
            (wrong / "model_index.json").write_text("{}")
            self.assertFalse(serverless_entry.link_host_snapshot("image", root / "local", root / "host"))
            correct = snapshots / revision
            correct.mkdir()
            (correct / "model_index.json").write_text("{}")
            self.assertTrue(serverless_entry.link_host_snapshot("image", root / "local", root / "host"))
            target = root / "local" / folder / "snapshots" / revision
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), correct.resolve())
            self.assertTrue(serverless_entry.link_host_snapshot("image", root / "local", root / "host"))


if __name__ == "__main__":
    unittest.main()
