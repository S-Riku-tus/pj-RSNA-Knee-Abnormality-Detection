"""Seal a manually specified actual-qualified regional1024 feature bundle."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.orthodiff_regional_features import seal_regional_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-bundle", "qualification", "records", "summary", "status", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-qualification-sha256", required=True)
    args = parser.parse_args()
    result = seal_regional_bundle(args.parent_bundle, args.qualification, args.expected_qualification_sha256,
                                  args.records, args.summary, args.status, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
