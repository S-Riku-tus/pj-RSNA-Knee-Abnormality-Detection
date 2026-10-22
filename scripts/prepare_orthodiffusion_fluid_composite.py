"""Seal qualified parent/FD feature files only; never trains or accesses Kaggle."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee.orthodiff_fluid_features import seal_composite_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-bundle", "child-qualification", "records", "summary", "status", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-parent-bundle-sha256", required=True)
    parser.add_argument("--expected-child-qualification-sha256", required=True)
    args = parser.parse_args()
    value = seal_composite_bundle(args.parent_bundle, args.expected_parent_bundle_sha256,
        args.child_qualification, args.expected_child_qualification_sha256, args.records, args.summary, args.status, args.output)
    print(json.dumps({"output": str(args.output), **value}))


if __name__ == "__main__":
    main()
