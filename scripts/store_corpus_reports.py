"""Store small release reports privately; never archive the raw corpus or audio."""
import argparse
import hashlib
import json
from pathlib import Path
import modal
from corpus_release import save, sha

VOLUME = "ghana-health-understand-train"
FILES = ("manifest.json", "SEALED.json", "inventory.json", "DATASET_CARD.md", "HANDOFF.md",
         "reports/coverage.json", "reports/collections.json", "reports/readiness.json",
         "reports/verification.json", "reports/source-selection.json", "reports/research-reference.json",
         "annotation/work-plan.json", "training/data-config.json")


def store(release):
    seal = json.loads((release / "SEALED.json").read_text())
    if sha(release / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Changed release manifest")
    if json.loads((release / "reports/verification.json").read_text())["status"] != "passed":
        raise ValueError("Only verified release reports can be stored")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    prefix = f"/corpus-release-reports/{release.name}/{seal['manifest_sha256'][:12]}"
    expected, pending = {}, []
    for name in FILES:
        path = release / name
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Report exceeds small-artifact limit: " + name)
        expected[name] = sha(path)
        try:
            remote = hashlib.sha256()
            for chunk in volume.read_file(prefix + "/" + name): remote.update(chunk)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pending.append(name)
        else:
            if remote.hexdigest() != expected[name]: raise ValueError("Remote artifact differs; never overwrite")
    if pending:
        with volume.batch_upload(force=False) as upload:
            for name in pending: upload.put_file(release / name, prefix + "/" + name)
    for name, checksum in expected.items():
        actual = hashlib.sha256()
        for chunk in volume.read_file(prefix + "/" + name): actual.update(chunk)
        if actual.hexdigest() != checksum: raise ValueError("Remote verification failed: " + name)
    receipt = {"provider": "modal", "volume": VOLUME, "path": prefix,
               "payload_class": "Small dataset reports, counts, source references and checksums only",
               "raw_corpus_uploaded": False, "training_rows_uploaded": False,
               "audio_uploaded": False, "public": False, "remote_sha256_verified": expected}
    save(release.with_suffix(".storage.json"), receipt)
    print(json.dumps({"verified_files": len(expected), "private_volume": VOLUME, "path": prefix}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    store(parser.parse_args().release.resolve())
