"""Dedicated dense regional4x4 last2 CLI using existing native392 caches.

QA uses synthetic pixels and existing generic assets only. There are no automatic
downloads, uploads, Notebook executions or competition submissions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Set this before any imported module can initialize CUDA/cuBLAS.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise ValueError("Dedicated regional execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from rsna_knee.feature_contracts import new_directory
from rsna_knee.regional_training_fast16 import (
    configure_regional_determinism,
    load_regional_config,
    predict_regional,
    preflight_regional_training,
    train_regional,
)
from rsna_knee.runtime import ensure_cuda


def add_assets(parser, *, provenance=True):
    for name in ("repo", "weights", "provenance") if provenance else ("repo", "weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    train = sub.add_parser("train")
    for mode in (preflight, train):
        for name in ("manifest", "index", "config", "resource-gate", "provenance"):
            mode.add_argument(f"--{name}", type=Path, required=True)
        mode.add_argument("--fold", choices=(0, 1), type=int, required=True)
    add_assets(train, provenance=False)
    train.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict")
    for name in ("root", "checkpoint", "output", "workspace"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    add_assets(predict)
    qa = sub.add_parser("qa")
    for name in ("config", "output"):
        qa.add_argument(f"--{name}", type=Path, required=True)
    add_assets(qa)
    args = parser.parse_args()
    configure_regional_determinism()
    ensure_cuda()  # All config/provenance/cache/teacher/model reads and mutations follow this guard.
    if args.command == "qa":
        from check_regional_training_fast16 import run

        return run(args)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if args.command in ("preflight", "train"):
        config = load_regional_config(args.config)
        if args.command == "preflight":
            training, validation, store, _ = preflight_regional_training(
                args.manifest, args.index, config, args.fold, args.resource_gate, provenance
            )
            print(
                json.dumps(
                    {
                        "ready": True,
                        "train_studies": len(training),
                        "valid_studies": len(validation),
                        "cache_studies": len(store.studies),
                        "fitting_performed": False,
                    }
                )
            )
            return
        return train_regional(
            args.manifest,
            args.index,
            args.output,
            config,
            args.fold,
            args.repo,
            args.weights,
            provenance,
            args.resource_gate,
        )
    if args.command == "predict":
        if not Path("/kaggle/input").is_dir():
            raise ValueError("Regional MRI inference belongs on Kaggle")
        from bounded_feature_reader import make_bounded_reader

        workspace = new_directory(args.workspace)
        return predict_regional(
            args.root,
            args.checkpoint,
            args.output,
            args.repo,
            args.weights,
            provenance,
            reader=make_bounded_reader(workspace),
        )


if __name__ == "__main__":
    main()
