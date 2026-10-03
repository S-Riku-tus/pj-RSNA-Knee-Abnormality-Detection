"""Verify a manually downloaded cache export without decoding MRI or training."""

import argparse
import json
from pathlib import Path

from rsna_knee.contracts import ID, preprocess_fingerprint, read_csv, sha256, source_hashes

try:
    from .export_paths import comparable_path
except ImportError:  # Running directly as python scripts/verify_cache_export.py.
    from export_paths import comparable_path


def verify(root):
    root = Path(root).resolve()
    record = json.loads((root / "export.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("complete") is not True:
        raise ValueError("A complete, supported export is required; a ten-study debug cache cannot train all folds")
    inventory = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "export.json" and "__pycache__" not in path.parts
    }
    if inventory != record["files"].keys():
        raise ValueError("Export file inventory differs from the recorded hash list")
    for name, metadata in record["files"].items():
        path = (root / name).resolve()
        comparable_path(path).relative_to(comparable_path(root))
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Transferred file hash/size mismatch: {name}")
    if record["source_sha256"] != source_hashes():
        raise ValueError("Repository source differs from cache export; use its recorded commit before the first run")
    cache = root / "train-v1"
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    if metadata["split"] != "train" or metadata["fingerprint"] != preprocess_fingerprint(record["config"]):
        raise ValueError("Cache split or preprocessing differs from the recorded config/source")
    for name, key in (("train.csv", "studies_csv_sha256"), ("train_series.csv", "series_csv_sha256")):
        if sha256(root / "raw" / name) != metadata[key]:
            raise ValueError(f"Cache originates from a different {name}")
    _, rows = read_csv(root / "raw" / "train.csv", (ID,))
    expected = {row[ID] for row in rows}
    actual = {path.stem for path in cache.glob("*.npz")}
    if len(expected) != len(rows) or actual != expected or record["studies_total"] != len(rows):
        raise ValueError("Exported cache study IDs/count do not match train.csv")
    if record["studies_requested"] != len(rows):
        raise ValueError("Export did not request the full training set")
    return {"valid": True, "studies": len(rows), "files_verified": len(record["files"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    print(json.dumps(verify(parser.parse_args().export_dir)))
