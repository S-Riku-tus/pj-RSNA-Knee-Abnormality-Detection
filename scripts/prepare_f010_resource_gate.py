"""Build an F010 resource gate from completed new-source label-free QA receipts.

No weights, pixels or labels are opened. This does not start fitting, save a
Kaggle Notebook, upload any file, or submit to the competition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.contracts import sha256
from rsna_knee.f010_last4_training import (
    ENGINEERING_LINEAGE,
    RESOURCE_GATE_SCHEMA,
    native_source_hashes,
    validate_native_config,
    validate_resource_gate,
)
from rsna_knee.feature_contracts import json_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixel16-review", type=Path, required=True)
    parser.add_argument("--processed-qa", type=Path, required=True)
    parser.add_argument("--artificial144-qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh F010 resource-gate file")
    inputs = {
        "pixel16": args.pixel16_review,
        "processed_QA": args.processed_qa,
        "artificial144_QA": args.artificial144_qa,
    }
    evidence = {}
    for name, path in inputs.items():
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.is_file() or resolved.is_symlink():
            raise ValueError("F010 receipts must be regular files inside this workspace")
        evidence[name] = {"path": resolved.relative_to(ROOT).as_posix(), "sha256": sha256(resolved)}
    configs = {}
    for blocks, name in ((0, "f010-dinov2-native-frozen-QA-only.json"), (4, "f010-dinov2-native-last4-late-cosine.json")):
        value = validate_native_config(json.loads((ROOT / "configs/experiments" / name).read_text(encoding="utf8")))
        configs[str(blocks)] = json_hash(value)
    gate = {
        "schema_version": RESOURCE_GATE_SCHEMA,
        "source_sha256": native_source_hashes(),
        "CLI_sha256": sha256(ROOT / "scripts/train_f010_last4_fast16.py"),
        "engineering_lineage": ENGINEERING_LINEAGE,
        "approved_for_real_fit": True,
        "config_semantic_sha256": configs,
        "evidence": evidence,
        "approval_scope": "Local F010 source/resource qualification only; no automatic fitting or external authority",
    }
    temporary = args.output.with_suffix(args.output.suffix + ".candidate")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(gate, stream, indent=2)
        stream.write("\n")
    validate_resource_gate(temporary)
    # No existing file may be replaced after a concurrent writer appeared.
    with args.output.open("x", encoding="utf8", newline="\n") as stream:
        stream.write(temporary.read_text(encoding="utf8"))
    temporary.unlink()
    print(json.dumps({"status": "F010_NEW_SOURCE_RESOURCE_GATE_PASS", "path": str(args.output), "sha256": sha256(args.output), "fitting_started": False}))


if __name__ == "__main__":
    main()
