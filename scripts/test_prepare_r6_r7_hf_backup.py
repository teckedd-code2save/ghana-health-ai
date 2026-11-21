import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from prepare_r6_r7_hf_backup import BASE_SHA, MODEL, REVISION, sha, upload, verify_package
from check_afrique_training_contract import ASSET_HASHES


class BackupBoundary(unittest.TestCase):
    def test_upload_refuses_public_repository(self):
        api = Mock()
        api.whoami.return_value = {"name": "teckedd", "isPro": False}
        api.model_info.return_value.private = False
        with tempfile.TemporaryDirectory() as d, \
                patch("prepare_r6_r7_hf_backup.verify_package", return_value={"variant": "r6", "files": {}}), \
                patch("huggingface_hub.get_token", return_value=True), \
                patch("huggingface_hub.HfApi", return_value=api):
            with self.assertRaisesRegex(ValueError, "public repository"):
                upload(Path(d), Path(d) / "receipt.json")
            api.upload_folder.assert_not_called()
            api.create_repo.assert_not_called()
            self.assertFalse((Path(d) / "receipt.json").exists())

    def package(self, folder):
        files = {"adapter_model.safetensors": "weights", "adapter_config.json": "config", "README.md": "card",
                 **{"tokenizer/" + name: name for name in ASSET_HASHES}}
        for name, value in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        variant = ("unused", sha(folder / "adapter_model.safetensors"), sha(folder / "adapter_config.json"))
        identity = {"variant": "r6", "base_model": MODEL, "base_revision": REVISION,
                    "base_weights_sha256": BASE_SHA, "adapter_sha256": variant[1],
                    "adapter_config_sha256": variant[2], "corpus_included": False, "base_weights_included": False}
        (folder / "SOURCE_IDENTITY.json").write_text(json.dumps(identity))
        files["SOURCE_IDENTITY.json"] = ""
        self.remanifest(folder, files)
        return files, variant, {name: sha(folder / "tokenizer" / name) for name in ASSET_HASHES}

    def remanifest(self, folder, files):
        (folder / "CHECKSUMS.json").write_text(json.dumps({"version": 1, "variant": "r6", "files": {
            name: {"bytes": (folder / name).stat().st_size, "sha256": sha(folder / name)} for name in files}}))

    def test_valid_exact_package_and_weight_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            _, variant, assets = self.package(folder)
            with patch.dict("prepare_r6_r7_hf_backup.VARIANTS", {"r6": variant}), patch.dict(ASSET_HASHES, assets):
                self.assertEqual(verify_package(folder)["variant"], "r6")
                (folder / "adapter_model.safetensors").write_text("changed")
                with self.assertRaisesRegex(ValueError, "Changed backup file"):
                    verify_package(folder)

    def test_regenerated_manifest_does_not_hide_tokenizer_change(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            files, variant, assets = self.package(folder)
            with patch.dict("prepare_r6_r7_hf_backup.VARIANTS", {"r6": variant}), patch.dict(ASSET_HASHES, assets):
                (folder / "tokenizer/tokenizer.json").write_text("wrong tokenizer")
                self.remanifest(folder, files)
                with self.assertRaisesRegex(ValueError, "Changed pinned tokenizer"):
                    verify_package(folder)

    def test_wrong_base_revision_even_with_matching_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            files, variant, assets = self.package(folder)
            with patch.dict("prepare_r6_r7_hf_backup.VARIANTS", {"r6": variant}), patch.dict(ASSET_HASHES, assets):
                path = folder / "SOURCE_IDENTITY.json"
                identity = json.loads(path.read_text())
                identity["base_revision"] = "main"
                path.write_text(json.dumps(identity))
                self.remanifest(folder, files)
                with self.assertRaisesRegex(ValueError, "Changed source identity"):
                    verify_package(folder)

    def test_unknown_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "CHECKSUMS.json").write_text(json.dumps({"version": 1, "variant": "other", "files": {}}))
            with self.assertRaisesRegex(ValueError, "Unknown package"):
                verify_package(p)

    def test_corpus_cannot_be_added_to_the_upload_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "CHECKSUMS.json").write_text(json.dumps({"version": 1, "variant": "r7", "files": {"train.jsonl": {}}}))
            with self.assertRaisesRegex(ValueError, "Non-checkpoint content"):
                verify_package(p)


if __name__ == "__main__":
    unittest.main()
