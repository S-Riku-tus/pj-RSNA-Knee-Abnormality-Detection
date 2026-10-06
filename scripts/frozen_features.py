"""Offline DINOv2 feature cache, CUDA head fitting and Report-free predictions; never uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.contracts import sha256
from rsna_knee.feature_contracts import load_feature_config, source_tree_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect-encoder", help="Hash locally supplied assets; does not download or declare eligibility"
    )
    inspect.add_argument("--repo", type=Path, required=True)
    inspect.add_argument("--weights", type=Path, required=True)
    extract = sub.add_parser("extract")
    for key in ("root", "output", "config", "provenance", "repo", "weights"):
        extract.add_argument(f"--{key}", type=Path, required=True)
    extract.add_argument("--split", choices=("train", "test"), required=True)
    extract.add_argument("--studies-csv", type=Path)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--shard-count", type=int, default=1)
    merge = sub.add_parser("merge")
    merge.add_argument("--cache", type=Path, action="append", required=True)
    for key in ("studies-csv", "output", "config"):
        merge.add_argument(f"--{key}", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--cache", type=Path, required=True)
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--allow-limited", action="store_true")
    train = sub.add_parser("train")
    for key in ("manifest", "cache", "output", "config"):
        train.add_argument(f"--{key}", type=Path, required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--fold-audit", type=Path, required=True)
    predict = sub.add_parser("predict")
    for key in ("test-csv", "cache", "checkpoint", "output"):
        predict.add_argument(f"--{key}", type=Path, required=True)
    predict.add_argument("--sample-csv", type=Path)
    predict.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "inspect-encoder":
        result = {"source_tree_sha256": source_tree_hash(args.repo), "weights_sha256": sha256(args.weights)}
    elif args.command == "extract":
        from rsna_knee.feature_imaging import extract_cache

        metadata = extract_cache(
            args.root,
            args.split,
            args.output,
            load_feature_config(args.config),
            json.loads(args.provenance.read_text(encoding="utf-8-sig")),
            args.repo,
            args.weights,
            studies_csv=args.studies_csv,
            limit=args.limit,
            device=args.device,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        result = {key: metadata[key] for key in ("complete", "requested", "elapsed_seconds", "total_payload_bytes")}
    elif args.command == "merge":
        from rsna_knee.feature_imaging import merge_feature_shards

        result = merge_feature_shards(args.cache, args.studies_csv, args.output, load_feature_config(args.config))
        result = {key: result[key] for key in ("complete", "requested", "total_payload_bytes")}
    elif args.command == "verify":
        from rsna_knee.feature_imaging import verify_feature_cache

        metadata, ids = verify_feature_cache(
            args.cache, load_feature_config(args.config), allow_limited=args.allow_limited
        )
        result = {"valid": True, "studies": len(ids), "fingerprint": metadata["fingerprint"]}
    elif args.command == "train":
        from rsna_knee.feature_runtime import train_feature_head

        record = train_feature_head(
            args.manifest,
            args.cache,
            args.output,
            load_feature_config(args.config),
            args.fold,
            fold_audit_path=args.fold_audit,
        )
        result = {key: record[key] for key in ("status", "fold", "selected", "elapsed_seconds")}
    else:
        from rsna_knee.feature_runtime import predict_feature_cache

        result = predict_feature_cache(
            args.test_csv, args.cache, args.checkpoint, args.output, args.device, args.sample_csv
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
