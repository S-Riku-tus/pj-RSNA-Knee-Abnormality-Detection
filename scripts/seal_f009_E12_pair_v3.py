"""Manual local F009 amended pair sealer; no uploads, GPU or automatic jobs."""
import argparse
import json
from pathlib import Path

from rsna_knee.f009_pair_stage_v3 import seal_f009_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "SOURCE_PREPARATION_ONLY_AMENDED_PAIR_NOT_CONSTRUCTED",
                          "actual_stage_constructed": False, "weights_or_run_opened": False}))
        return
    if args.spec is None or args.output is None or not args.expected_spec_sha256:
        parser.error("--execute requires --spec --expected-spec-sha256 --output")
    seal_f009_pair(args.spec, args.expected_spec_sha256, args.output)


if __name__ == "__main__":
    main()
