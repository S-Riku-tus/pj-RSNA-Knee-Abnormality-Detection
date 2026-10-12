"""Explicit local native392 full-index validation; no download/decode/fit/upload."""

import argparse
import json
from pathlib import Path

from rsna_knee.contracts import sha256
from rsna_knee.native_pixel_shards import build_full_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-glob", default="shards/**/NATIVE392_PIXEL_SHARD_SUMMARY.json")
    args = parser.parse_args()
    package = json.loads((args.package / "package.json").read_text(encoding="utf-8"))
    if (
        sha256(args.package / "plan.json") != package["plan_file_sha256"]
        or sha256(args.package / "native392-code.zip") != package["execution_binding"]["code_zip_sha256"]
    ):
        raise ValueError("Immutable package code/plan hashes changed")
    plan = json.loads((args.package / "plan.json").read_text(encoding="utf-8"))
    summaries = sorted(args.cache_root.glob(args.summary_glob))
    result = build_full_index(plan, package, summaries, args.output, cache_root=args.cache_root)
    print(json.dumps({"complete": result["complete"], **result["verified"], "shards": len(result["shards"])}))


if __name__ == "__main__":
    main()
