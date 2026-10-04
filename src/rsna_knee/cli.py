from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from .contracts import dump_json, index_studies, load_config, read_csv, validate_submission


def main(argv=None):
    parser = argparse.ArgumentParser(description="RSNA Knee: local metadata tools and GPU pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect runtime only; no downloads or model allocation")
    p = sub.add_parser("audit", help="Audit CSVs without decoding DICOM")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("label-template")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--weak-labels", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--groups", help="CSV with StudyInstanceUID,group_id for every study")
    p.add_argument("--config", default="configs/baseline.json")
    p = sub.add_parser("validate-submission")
    p.add_argument("--submission", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--sample-submission")
    p = sub.add_parser("evaluate")
    p.add_argument("--truth", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("cache", help="GPU terminal only: decode supplied DICOM into versioned cache")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--limit", type=int, help="Debug only; process first N studies")
    p = sub.add_parser("train", help="CUDA-only training; no implicit CPU training")
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default="configs/baseline.json")
    p.add_argument("--fold", type=int, default=0)
    p = sub.add_parser(
        "diagnose-train", help="CUDA-only memorization diagnostic on training partition; no gold evaluation"
    )
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--studies", type=int, default=16)
    p = sub.add_parser("predict")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = {
                "python": sys.version.split()[0],
                "modules": {
                    name: importlib.util.find_spec(name) is not None
                    for name in ("numpy", "pydicom", "PIL", "torch", "torchvision")
                },
                "training_policy": "CUDA required; this command does not train or download",
            }
            if result["modules"]["torch"]:
                import torch

                result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        elif args.command == "audit":
            from .prepare import audit

            result = audit(args.data_root, args.output)
        elif args.command == "label-template":
            from .prepare import label_template

            result = label_template(args.train_csv, args.output)
        elif args.command == "prepare":
            from .prepare import prepare

            config = load_config(args.config)
            result = prepare(
                args.train_csv,
                args.weak_labels,
                args.provenance,
                args.output_dir,
                n_folds=config["n_folds"],
                seed=config["seed"],
                groups_path=args.groups,
            )
        elif args.command == "validate-submission":
            result = validate_submission(args.submission, args.test_csv, args.sample_submission)
        elif args.command == "evaluate":
            from .metrics import evaluate_rows

            _, truth = read_csv(args.truth)
            _, predictions = read_csv(args.predictions)
            result = evaluate_rows(index_studies(truth), index_studies(predictions))
            dump_json(args.output, result)
        elif args.command == "cache":
            from .imaging import build_cache

            result = build_cache(args.data_root, args.split, args.cache_dir, load_config(args.config), args.limit)
        elif args.command in ("train", "diagnose-train"):
            from .runtime import train

            result = train(
                args.manifest_dir,
                args.cache_dir,
                args.run_dir,
                load_config(args.config),
                args.fold,
                diagnostic_studies=args.studies if args.command == "diagnose-train" else None,
            )
        elif args.command == "predict":
            from .runtime import predict

            result = predict(args.test_csv, args.cache_dir, args.checkpoint, args.output, args.device)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
