"""CUDA-only fitting of a small head; immutable exports and Report-free inference."""

from __future__ import annotations

import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .contracts import (
    ID,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    probability,
    read_csv,
    sha256,
    source_hashes,
    validate_submission,
    write_csv,
)
from .feature_contracts import (
    head_implementation_hash,
    new_directory,
    selection_structure_hash,
    validate_feature_config,
    validate_fold_audit,
)
from .feature_imaging import load_features, verify_feature_cache
from .feature_model import FrozenFeatureHead
from .metrics import evaluate_rows
from .runtime import check_split, ensure_cuda, masked_bce


class FeatureDataset(Dataset):
    def __init__(self, rows, cache_dir, config, metadata):
        self.rows, self.root, self.config, self.metadata = rows, cache_dir, config, metadata

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        arrays = load_features(self.root, row[ID], self.config, self.metadata["fingerprint"])
        values = [probability(row.get(target, ""), missing=True) for target in TARGETS]
        observed = torch.tensor([value is not None for value in values], dtype=torch.bool)
        # Zero is only a tensor placeholder; masked_bce excludes every unobserved cell.
        targets = torch.tensor([value if value is not None else 0 for value in values], dtype=torch.float32)
        return {name: torch.from_numpy(value) for name, value in arrays.items()}, targets, observed, row[ID]


def on_device(arrays, device):
    return {name: value.to(device) for name, value in arrays.items()}


@torch.inference_mode()
def infer_features(model, loader, device):
    model.eval()
    result, loss_sum, count = [], 0.0, 0
    for arrays, targets, observed, keys in loader:
        logits = model(**on_device(arrays, device)).float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite feature-head output")
        if observed.any():
            loss_sum += float(masked_bce(logits, targets, observed)) * int(observed.sum())
            count += int(observed.sum())
        for key, values in zip(keys, logits.sigmoid().tolist()):
            result.append({ID: key, **{target: str(score) for target, score in zip(TARGETS, values)}})
    return result, loss_sum / count if count else None


def train_feature_head(manifest_dir, cache_dir, output, config, fold, *, fold_audit_path=None):
    ensure_cuda()  # Before all data reads and output mutations, same guard as the legacy CLI.
    validate_feature_config(config)
    manifest_dir = Path(manifest_dir)
    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"] != config["seed"] or manifest["n_folds"] != config["n_folds"]:
        raise ValueError("Fixed split seed/folds mismatch")
    if manifest["label_provenance"].get("gold_used_for_tuning") is not False:
        raise ValueError("Gold-tuned weak label source cannot initialize this independent experiment")
    fold_audit_path = Path(fold_audit_path) if fold_audit_path else manifest_dir / "fold-audit-v2.json"
    fold_audit = validate_fold_audit(manifest_dir, fold_audit_path, manifest)
    _, weak = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "source", "group_id", "fold"))
    _, gold = read_csv(manifest_dir / "gold.csv", (ID, "source", "group_id"))
    training, validation = check_split(weak, gold, fold, config["n_folds"])
    metadata, ids = verify_feature_cache(cache_dir, config)
    if metadata["split"] != "train" or metadata["studies_csv_sha256"] != manifest["inputs_sha256"]["train"]:
        raise ValueError("Feature cache and weak manifest have different source train CSV")
    if ids.keys() != index_studies(weak).keys() or metadata.get(
        "selection_structure_sha256"
    ) != selection_structure_hash(weak):
        raise ValueError("Training cache must exactly cover the fixed weak manifest (no pilot/gold)")
    for row in weak:
        if not any(probability(row[target], missing=True) is not None for target in TARGETS):
            raise ValueError("Weak manifest contains a fully unobserved study")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = config["train"]
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        FeatureDataset(training, cache_dir, config, metadata),
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        generator=generator,
    )
    valid_loader = DataLoader(
        FeatureDataset(validation, cache_dir, config, metadata),
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
        generator=torch.Generator().manual_seed(config["seed"] + 1),
    )
    model = FrozenFeatureHead(config).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    out = new_directory(output)
    dump_json(out / "config.json", config)
    record = {
        "schema_version": "frozen_feature_run_v1",
        "status": "running",
        "seed": config["seed"],
        "fold": fold,
        "reason": config["reason"],
        "feature_contract": metadata["contract"],
        "feature_fingerprint": metadata["fingerprint"],
        "export_sha256": sha256(Path(cache_dir) / "export.json"),
        "manifest_sha256": {name: sha256(manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv")},
        "manifest": manifest,
        "fold_audit_sha256": sha256(fold_audit_path),
        "fold_audit": fold_audit,
        "source_sha256": source_hashes(),
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
        "gpu": torch.cuda.get_device_name(0),
        "train_studies": len(training),
        "valid_studies": len(validation),
        "selection_data": "weak_validation",
        "gold_evaluated": False,
        "gold_used_for_training_or_selection": False,
        "encoder_trainable": False,
        "selection_rule": settings["selection"],
        "history": [],
        "patient_independence_verified": False,
        "public_lb": None,
        "note": "Generic encoder pretraining overlap is not independently excluded. No public competition checkpoint used.",
    }
    dump_json(out / "run.json", record)
    start, selected = time.perf_counter(), None
    truth = index_studies(validation)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        loss_sum, count, steps = 0.0, 0, 0
        for arrays, targets, observed, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(**on_device(arrays, "cuda"))
            loss = masked_bce(logits, targets.cuda(), observed.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(observed.sum())
            count += int(observed.sum())
            steps += 1
        predictions, bce = infer_features(model, valid_loader, "cuda")
        metrics = evaluate_rows(truth, index_studies(predictions))
        auc = metrics["macro_auc_12"]
        prefix = f"epoch_{epoch:03d}"
        write_csv(out / f"{prefix}_weak.csv", SUBMISSION_COLUMNS, predictions)
        item = {
            "epoch": epoch,
            "train_bce": loss_sum / count,
            "weak_bce": bce,
            "weak_macro_auc_12": auc,
            "optimizer_steps": steps,
            "elapsed_seconds": time.perf_counter() - start,
            "metrics": metrics,
            "prediction_sha256": sha256(out / f"{prefix}_weak.csv"),
        }
        record["history"].append(item)
        # AUC is selectable only if every target has both observed binary classes.
        eligible = settings["selection"] == "bce" or auc is not None
        key = (-bce,) if settings["selection"] == "bce" else ((auc, -bce) if auc is not None else None)
        checkpoint = {
            "schema_version": "frozen_feature_head_v1",
            "head_implementation_sha256": head_implementation_hash(),
            "model": model.state_dict(),
            "config": config,
            "targets": list(TARGETS),
            "fold": fold,
            "epoch": epoch,
            "feature_contract": metadata["contract"],
            "feature_fingerprint": metadata["fingerprint"],
            "selection_data": "weak_validation",
            "resume_supported": False,
        }
        torch.save(checkpoint, out / "last.pt")
        if eligible and (selected is None or key > selected):
            selected = key
            torch.save(checkpoint, out / "best.pt")
            write_csv(out / "weak_valid_predictions.csv", SUBMISSION_COLUMNS, predictions)
            record["selected"] = {**item, "checkpoint_sha256": sha256(out / "best.pt")}
        dump_json(out / "run.json", record)
        print(f"Head epoch {epoch}/{settings['epochs']}: weak AUC12={auc}, BCE={bce:.6f}", flush=True)
    if selected is None:
        raise ValueError("No epoch has a defined 12-target weak AUC; do not silently change selection rule")
    record.update(status="complete", elapsed_seconds=time.perf_counter() - start)
    dump_json(out / "run.json", record)
    return record


def load_head_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "frozen_feature_head_v1" or checkpoint.get("targets") != list(TARGETS):
        raise ValueError("Unsupported feature-head checkpoint/target order")
    validate_feature_config(checkpoint["config"])
    if checkpoint.get("head_implementation_sha256") != head_implementation_hash():
        raise ValueError("Head implementation differs from trained checkpoint")
    if checkpoint.get("selection_data") != "weak_validation":
        raise ValueError("Checkpoint was not selected on weak validation")
    return checkpoint


def predict_feature_cache(test_csv, cache_dir, checkpoint_path, output_csv, device="cuda", sample_csv=None):
    if Path(output_csv).exists():
        raise ValueError("Use a new prediction path")
    checkpoint = load_head_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    metadata, ids = verify_feature_cache(cache_dir, config)
    if metadata["split"] != "test" or metadata["studies_csv_sha256"] != sha256(test_csv):
        raise ValueError("Test cache/source CSV mismatch")
    if (
        metadata["contract"] != checkpoint["feature_contract"]
        or metadata["fingerprint"] != checkpoint["feature_fingerprint"]
    ):
        raise ValueError("Encoder, weights or preprocessing differ from the trained feature head")
    _, rows = read_csv(test_csv, (ID,))  # No Report column required or consumed.
    if index_studies(rows).keys() != ids.keys():
        raise ValueError("Feature cache and test study IDs differ")
    # Discard all non-ID columns, even if a caller supplies labels or Report.
    rows = [{ID: row[ID]} for row in rows]
    model = FrozenFeatureHead(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    loader = DataLoader(FeatureDataset(rows, cache_dir, config, metadata), batch_size=config["train"]["batch_size"])
    predictions, _ = infer_features(model, loader, device)
    write_csv(output_csv, SUBMISSION_COLUMNS, predictions)
    result = validate_submission(output_csv, test_csv, sample_csv)
    dump_json(
        Path(output_csv).with_suffix(".receipt.json"),
        {
            **result,
            "checkpoint_sha256": sha256(checkpoint_path),
            "feature_fingerprint": metadata["fingerprint"],
            "submission_sha256": sha256(output_csv),
            "note": "Contract-valid output; no claim of Public score or hidden-test runtime",
        },
    )
    return result
