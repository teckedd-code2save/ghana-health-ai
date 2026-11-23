import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from download_r7_base_local import BASE_SHA, MODEL, REVISION, command, run, safe_line


class DownloadContract(unittest.TestCase):
    def test_only_pinned_public_weights_with_resume_and_hash(self):
        args = command("aria2c", Path("/tmp/base-transfer"))
        self.assertEqual(args[-1], f"https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors")
        for flag in ("--continue=true", "--always-resume=true", "--allow-overwrite=false",
                     "--auto-file-renaming=false", "--enable-rpc=false", "--checksum=sha-256=" + BASE_SHA):
            self.assertIn(flag, args)

    def test_signed_urls_not_logged(self):
        self.assertEqual(safe_line("failed https://example.test/file?signature=private next"),
                         "failed [transfer URL omitted] next")

    def test_existing_wrong_final_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch("shutil.which", return_value="aria2c"), \
                patch("subprocess.Popen") as launch:
            root = Path(directory)
            (root / "afrique-base-bf16").mkdir()
            final = root / "afrique-base-bf16/model.safetensors"
            final.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "Wrong completed"):
                run(root, "aria2c")
            self.assertEqual(final.read_bytes(), b"wrong")
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
