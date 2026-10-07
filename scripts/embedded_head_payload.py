"""Restore a small audited local head bundle; standard library only, no downloads."""

import base64
import binascii
import hashlib
from pathlib import Path


def materialize_embedded_head(payload, destination):
    """Validate every byte before creating a fresh output directory or loading a model."""
    expected_names = {"best.pt", "config.json", "checkpoint-provenance.json"}
    if not isinstance(payload, dict) or set(payload) != expected_names:
        raise ValueError("Embedded head file allowlist differs from the audited bundle")
    decoded = {}
    for name, item in payload.items():
        if not isinstance(item, dict) or set(item) != {"base64", "bytes", "sha256"}:
            raise ValueError(f"Invalid embedded head metadata: {name}")
        if type(item["bytes"]) is not int or not 0 < item["bytes"] < 1_000_000:
            raise ValueError(f"Invalid embedded head size: {name}")
        try:
            raw = base64.b64decode(item["base64"], validate=True)
        except (binascii.Error, ValueError, TypeError) as error:
            raise ValueError(f"Invalid embedded head base64: {name}") from error
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Embedded head bytes/hash mismatch: {name}")
        decoded[name] = raw
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Embedded head requires a fresh kernel/output directory")
    destination.mkdir()
    for name, raw in decoded.items():
        with (destination / name).open("xb") as handle:
            handle.write(raw)
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != payload[name]["sha256"]:
            raise ValueError(f"Embedded head on-disk verification failed: {name}")
    return {
        "mode": "embedded_notebook_bytes",
        "checkpoint_path": str(destination / "best.pt"),
        "files": {
            name: {"bytes": payload[name]["bytes"], "sha256": payload[name]["sha256"]}
            for name in sorted(payload)
        },
        "head_dataset_required": False,
        "downloaded": False,
    }
