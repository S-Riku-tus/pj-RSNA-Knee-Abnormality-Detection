"""Verify a manually downloaded cache export without decoding MRI or training."""

import argparse
import hashlib
import json
from pathlib import Path

from rsna_knee.contracts import ID, index_studies, load_config, preprocess_fingerprint, read_csv, sha256, source_hashes

try:
    from .download_cache_output import safe_path, validate_export
except ImportError:  # Running directly as python scripts/verify_cache_export.py.
    from download_cache_output import safe_path, validate_export


def _normalized_hash(path):
    """Hash recorded Python text without importing or executing export code."""
    implementation = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(implementation.encode()).hexdigest()


def _stored_fingerprint(config, imaging_hash):
    # Schema-1 caches use this recorded pixel-preprocessing contract. Do not
    # execute their copy of contracts.py to compute its fingerprint.
    payload = {"settings": config["preprocess"], "implementation": imaging_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify(root, config=None, require_source_match=False):
    """Check immutable export integrity separately from current pixel compatibility.

    config may be a mapping or a config JSON path; None uses the export's saved
    config. Model and logging changes may reuse intact compatible pixels, while
    require_source_match restores the strict whole-repository comparison.
    """
    root = Path(root).resolve()
    record = json.loads(safe_path(root, "export.json").read_text(encoding="utf-8"))
    files, _ = validate_export(record, root)
    inventory = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "export.json" and "__pycache__" not in path.parts
    }
    if inventory != files.keys():
        raise ValueError("Export file inventory differs from the recorded hash list")
    for name, metadata in files.items():
        path = safe_path(root, name)
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Transferred file hash/size mismatch: {name}")

    stored_sources = {
        path.name: _normalized_hash(path) for path in sorted(safe_path(root, "code/src/rsna_knee").glob("*.py"))
    }
    if not stored_sources or "imaging.py" not in stored_sources or stored_sources != record.get("source_sha256"):
        raise ValueError("Stored export source differs from its recorded source hashes")
    saved_config = json.loads(safe_path(root, "code/configs/baseline.json").read_text(encoding="utf-8-sig"))
    if saved_config != record.get("config"):
        raise ValueError("Stored config differs from the recorded export config")
    stored_fingerprint = _stored_fingerprint(saved_config, stored_sources["imaging.py"])

    cache = root / "train-v1"
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("split") != "train"
        or metadata.get("preprocess") != saved_config["preprocess"]
        or metadata.get("fingerprint") != stored_fingerprint
        or record.get("cache", {}).get("fingerprint") != stored_fingerprint
    ):
        raise ValueError("Cache split or preprocessing differs from the recorded config/source")
    for name, key in (("train.csv", "studies_csv_sha256"), ("train_series.csv", "series_csv_sha256")):
        if sha256(root / "raw" / name) != metadata[key]:
            raise ValueError(f"Cache originates from a different {name}")
    _, rows = read_csv(root / "raw" / "train.csv", (ID,))
    expected = set(index_studies(rows))
    actual = {path.stem for path in cache.glob("*.npz")}
    if not expected or actual != expected or type(record.get("studies_total")) is not int:
        raise ValueError("Exported cache study IDs/count do not match train.csv")
    if record["studies_total"] != len(rows):
        raise ValueError("Exported cache study IDs/count do not match train.csv")
    if type(record.get("studies_requested")) is not int or record["studies_requested"] != len(rows):
        raise ValueError("Export did not request the full training set")

    current_sources = source_hashes()
    source_differences = sorted(
        name
        for name in current_sources.keys() | stored_sources.keys()
        if current_sources.get(name) != stored_sources.get(name)
    )
    selected_config = saved_config if config is None else config
    if isinstance(selected_config, (str, Path)):
        selected_config = load_config(selected_config)
    current_fingerprint = preprocess_fingerprint(selected_config)
    if (
        current_sources.get("imaging.py") != stored_sources["imaging.py"]
        or selected_config["preprocess"] != saved_config["preprocess"]
        or current_fingerprint != stored_fingerprint
    ):
        raise ValueError("Current pixel preprocessing is incompatible with this export; use a new cache")
    if require_source_match and source_differences:
        raise ValueError("Repository source differs from cache export and strict source matching was requested")
    return {
        "valid": True,
        "studies": len(rows),
        "files_verified": len(files),
        "export_integrity": True,
        "source_integrity": True,
        "preprocess_compatible": True,
        "preprocess_fingerprint": stored_fingerprint,
        "source_matches_repository": not source_differences,
        "source_differences": source_differences,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--config", type=Path, help="Current training config; otherwise compare the saved config")
    parser.add_argument("--require-source-match", action="store_true", help="Require all source files to match")
    args = parser.parse_args()
    print(json.dumps(verify(args.export_dir, config=args.config, require_source_match=args.require_source_match)))
