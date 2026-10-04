"""GPU training and portable inference; imported only by the relevant CLI commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .contracts import (
    ID,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    preprocess_fingerprint,
    probability,
    read_csv,
    sha256,
    source_hashes,
    validate_submission,
    write_csv,
)
from .diagnostics import prediction_diagnostics
from .imaging import load_cached
from .inputs import normalize_images
from .metrics import evaluate_rows
from .model import KneeMIL


class StudyDataset(Dataset):
    def __init__(self, rows, cache_dir, config):
        self.rows, self.cache_dir, self.config = rows, cache_dir, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images, mask = load_cached(self.cache_dir, row[ID], self.config)
        values = [probability(row.get(t, ""), missing=True) for t in TARGETS]
        targets = torch.tensor([v if v is not None else 0 for v in values], dtype=torch.float32)
        observed = torch.tensor([v is not None for v in values], dtype=torch.bool)
        return (
            normalize_images(images, self.config),
            torch.from_numpy(mask),
            targets,
            observed,
            row[ID],
        )


def masked_bce(logits, targets, observed):
    if not observed.any():
        raise ValueError("No observed targets in batch")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * observed).sum() / observed.sum()


def ensure_cuda():
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU unavailable. Training must run on the GPU terminal, not this laptop.")


def check_cache_origin(cache_dir, config, split, studies_hash=None):
    metadata = json.loads((Path(cache_dir) / "cache.json").read_text(encoding="utf-8"))
    if metadata["fingerprint"] != preprocess_fingerprint(config) or metadata["split"] != split:
        raise ValueError("Cache split/config mismatch")
    if studies_hash and metadata["studies_csv_sha256"] != studies_hash:
        raise ValueError("Cache and labels originate from different train.csv files")
    return metadata


def check_split(weak, gold, fold, n_folds):
    weak_ids, gold_ids = index_studies(weak), index_studies(gold)
    if weak_ids.keys() & gold_ids.keys():
        raise ValueError("Official holdout leaked into weak training manifest")
    if {r["group_id"] for r in weak} & {r["group_id"] for r in gold}:
        raise ValueError("Gold group leaked into weak manifest")
    if not 0 <= fold < n_folds:
        raise ValueError("Fold outside configured range")
    if any(r["source"] != "report_weak" or not 0 <= int(r["fold"]) < n_folds for r in weak):
        raise ValueError("Invalid weak manifest source/fold")
    train_rows = [r for r in weak if int(r["fold"]) != fold]
    valid_rows = [r for r in weak if int(r["fold"]) == fold]
    if not train_rows or not valid_rows:
        raise ValueError("Empty train/validation partition")
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in valid_rows}:
        raise ValueError("Group leaked across folds")
    return train_rows, valid_rows


@torch.inference_mode()
def infer_loader(model, loader, device, *, return_details=False):
    model.eval()
    rows, loss_sum, observations = [], 0.0, 0
    target_sums = [0.0] * len(TARGETS)
    target_counts = [0] * len(TARGETS)
    for images, mask, targets, observed, keys in loader:
        logits = model(images.to(device), mask.to(device))
        probabilities = logits.sigmoid().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite model output")
        if observed.any():
            losses = F.binary_cross_entropy_with_logits(logits.cpu(), targets, reduction="none")
            loss_sum += float((losses * observed).sum())
            observations += int(observed.sum())
            if return_details:
                for index, (total, count) in enumerate(
                    zip((losses * observed).sum(0).tolist(), observed.sum(0).tolist())
                ):
                    target_sums[index] += total
                    target_counts[index] += count
        for key, scores in zip(keys, probabilities.tolist()):
            rows.append({ID: key, **dict(zip(TARGETS, scores))})
    mean_loss = loss_sum / observations if observations else None
    if not return_details:
        return rows, mean_loss
    per_target = {
        target: {"observed": count, "masked_bce": total / count if count else None}
        for target, total, count in zip(TARGETS, target_sums, target_counts)
    }
    defined = [value["masked_bce"] for value in per_target.values() if value["masked_bce"] is not None]
    return (
        rows,
        mean_loss,
        {
            "cell_mean_bce": mean_loss,
            "observed_target_mean_bce": sum(defined) / len(defined) if defined else None,
            "observed_cells": observations,
            "per_target": per_target,
        },
    )


@contextmanager
def preserve_training_state(model):
    """Extra diagnostics must not alter RNG, dropout or mixed BN train/eval modes."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for module, mode in modes:
            module.training = mode


def diagnostic_inference(model, loader, device):
    generators = {}
    for candidate in (getattr(loader, "generator", None), getattr(getattr(loader, "sampler", None), "generator", None)):
        if candidate is not None:
            generators[id(candidate)] = (candidate, candidate.get_state())
    try:
        with preserve_training_state(model):
            return infer_loader(model, loader, device, return_details=True)
    finally:
        for generator, state in generators.values():
            generator.set_state(state)


def fixed_subset(rows, seed, count):
    if type(count) is not int or count < 0:
        raise ValueError("Diagnostic subset size must be a nonnegative integer")
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row[ID]}".encode()).hexdigest())[:count]


def diagnostic_settings(config):
    settings = config.get("diagnostics", {})
    if type(settings.get("enabled", False)) is not bool or type(settings.get("audit_gold", True)) is not bool:
        raise ValueError("Diagnostic enabled/audit_gold flags must be booleans")
    count = settings.get("train_eval_studies", 0)
    if type(count) is not int or count < 0:
        raise ValueError("train_eval_studies must be a nonnegative integer")
    return settings


def train(manifest_dir, cache_dir, run_dir, config, fold, *, diagnostic_studies=None):
    ensure_cuda()  # Fails before run files, cache reads or model allocation on a CPU terminal.
    manifest_dir, out = Path(manifest_dir), Path(run_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty run directory; do not overwrite an experiment")
    metadata = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    if metadata["seed"] != config["seed"] or metadata["n_folds"] != config["n_folds"]:
        raise ValueError("Config seed/folds differ from the prepared manifest")
    _, weak = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "fold", "group_id", "source"))
    _, gold = read_csv(manifest_dir / "gold.csv", (ID, *TARGETS, "group_id", "source"))
    training, validation = check_split(weak, gold, fold, config["n_folds"])
    diagnostics = diagnostic_settings(config)
    diagnostic_mode = diagnostic_studies is not None
    if diagnostic_mode:
        if type(diagnostic_studies) is not int or diagnostic_studies < 1:
            raise ValueError("diagnostic_studies must be a positive integer")
        training = fixed_subset(training, config["seed"], diagnostic_studies)
        validation = training
    audit_gold = diagnostics.get("audit_gold", True) and not diagnostic_mode
    log_diagnostics = diagnostics.get("enabled", False) or diagnostic_mode
    cache_meta = check_cache_origin(cache_dir, config, "train", metadata["inputs_sha256"]["train"])
    # Preflight every study, so missing caches fail before the expensive run begins.
    for row in [*training, *validation, *(gold if audit_gold else [])]:
        load_cached(cache_dir, row[ID], config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = config["train"]
    generator = torch.Generator().manual_seed(config["seed"])

    def loader(rows, shuffle=False):
        return DataLoader(
            StudyDataset(rows, cache_dir, config),
            batch_size=settings["batch_size"],
            shuffle=shuffle,
            num_workers=settings["num_workers"],
            generator=generator,
        )

    training_loader, valid_loader = loader(training, True), loader(validation)
    train_eval_rows = fixed_subset(training, config["seed"], diagnostics.get("train_eval_studies", 0))
    train_eval_loader = None
    if log_diagnostics and train_eval_rows and not diagnostic_mode:
        train_eval_loader = DataLoader(
            StudyDataset(train_eval_rows, cache_dir, config),
            batch_size=settings["batch_size"],
            shuffle=False,
            num_workers=settings["num_workers"],
            generator=torch.Generator().manual_seed(config["seed"] + 1),
        )
    model_settings = config["model"]
    model = KneeMIL(model_settings["dropout"], bn_running_stats=model_settings.get("bn_running_stats", "update"))
    initialization = {"name": "random", "note": "no pretrained weights"}
    if model_settings.get("initialization", "random") == "imagenet1k_v1":
        initialization = model.initialize_encoder(
            model_settings["pretrained_path"], model_settings["pretrained_sha256"]
        )
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=settings["amp"])
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "config.json", config)
    dump_json(
        out / "run.json",
        {
            "fold": fold,
            "manifest": metadata,
            "cache": cache_meta,
            "source_sha256": source_hashes(),
            "manifest_sha256": {
                name: sha256(manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv")
            },
            "runtime": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "pydicom", "Pillow")
            },
            "gpu": torch.cuda.get_device_name(0),
            "initialization": initialization,
            "initial_head_sha256": hashlib.sha256(
                b"".join(
                    value.detach().cpu().numpy().tobytes()
                    for name, value in model.state_dict().items()
                    if not name.startswith("encoder.")
                )
            ).hexdigest(),
            "train_studies": len(training),
            "valid_studies": len(validation),
            "diagnostic_mode": diagnostic_mode,
            "gold_audit_enabled": audit_gold,
            "checkpoint_selection": (
                "minimum selected training-subset BCE; memorization diagnostic, not holdout evaluation"
                if diagnostic_mode
                else "minimum weak validation masked BCE; gold never used for selection"
            ),
        },
    )
    if train_eval_rows or diagnostic_mode:
        selected = training if diagnostic_mode else train_eval_rows
        write_csv(out / "diagnostic_subset.csv", (ID,), [{ID: row[ID]} for row in selected])
        dump_json(
            out / "diagnostic_selection.json",
            {
                "studies": len(selected),
                "seed": config["seed"],
                "fold": fold,
                "subset_csv_sha256": sha256(out / "diagnostic_subset.csv"),
                "note": "Training partition only; not an independent validation set",
            },
        )
    history, best, start_time = [], float("inf"), time.perf_counter()
    extra_eval_seconds = 0.0
    if diagnostic_mode:
        initial_predictions, _, details = diagnostic_inference(model, valid_loader, "cuda")
        dump_json(
            out / "initial_train_metrics.json",
            prediction_diagnostics(index_studies(validation), index_studies(initial_predictions), details),
        )
    for epoch in range(settings["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accumulation = settings["accumulation_steps"]
        for step, (images, mask, targets, observed, _) in enumerate(training_loader):
            group_start = step // accumulation * accumulation
            group_size = min(accumulation, len(training_loader) - group_start)
            with torch.autocast(device_type="cuda", enabled=settings["amp"]):
                logits = model(images.cuda(), mask.cuda())
                loss = masked_bce(logits, targets.cuda(), observed.cuda())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(training_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if log_diagnostics:
            predictions, valid_loss, details = infer_loader(model, valid_loader, "cuda", return_details=True)
        else:
            predictions, valid_loss = infer_loader(model, valid_loader, "cuda")
        if valid_loss is None or not np.isfinite(valid_loss):
            raise ValueError("Invalid weak validation loss")
        record = {
            "epoch": epoch + 1,
            "train_batch_mean_bce": sum(losses) / len(losses),
            "diagnostic_train_masked_bce" if diagnostic_mode else "weak_valid_masked_bce": valid_loss,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        if log_diagnostics:
            epoch_dir = out / "epochs" / f"{epoch + 1:03d}"
            write_csv(epoch_dir / "predictions.csv", SUBMISSION_COLUMNS, predictions)
            metrics = prediction_diagnostics(index_studies(validation), index_studies(predictions), details)
            dump_json(epoch_dir / "metrics.json", metrics)
            record["diagnostic_train_macro_auc_12" if diagnostic_mode else "weak_valid_macro_auc_12"] = metrics[
                "macro_auc_12"
            ]
            record["defined_auc_classes"] = metrics["defined_classes"]
            record[
                "diagnostic_train_observed_target_mean_bce"
                if diagnostic_mode
                else "weak_valid_observed_target_mean_bce"
            ] = metrics["observed_target_mean_bce"]
            if train_eval_loader is not None:
                eval_started = time.perf_counter()
                train_predictions, train_loss, train_details = diagnostic_inference(model, train_eval_loader, "cuda")
                extra_eval_seconds += time.perf_counter() - eval_started
                train_metrics = prediction_diagnostics(
                    index_studies(train_eval_rows), index_studies(train_predictions), train_details
                )
                dump_json(epoch_dir / "train_eval_metrics.json", train_metrics)
                write_csv(epoch_dir / "train_eval_predictions.csv", SUBMISSION_COLUMNS, train_predictions)
                record["train_eval_cell_mean_bce"] = train_loss
                record["train_eval_observed_target_mean_bce"] = train_metrics["observed_target_mean_bce"]
            record["extra_train_eval_seconds"] = extra_eval_seconds
            record["elapsed_seconds"] = time.perf_counter() - start_time
            record["elapsed_excluding_extra_train_eval_seconds"] = record["elapsed_seconds"] - extra_eval_seconds
        history.append(record)
        dump_json(out / "history.json", history)
        print(json.dumps(record), flush=True)
        if valid_loss < best:
            best = valid_loss
            torch.save(
                {
                    "schema_version": 1,
                    "model": model.state_dict(),
                    "config": config,
                    "targets": list(TARGETS),
                    "epoch": epoch + 1,
                    "fold": fold,
                    "preprocess_fingerprint": preprocess_fingerprint(config),
                    "input_normalization": model_settings.get("input_normalization", "legacy"),
                },
                out / "best.pt",
            )
            write_csv(
                out / ("diagnostic_train_predictions.csv" if diagnostic_mode else "weak_valid_predictions.csv"),
                SUBMISSION_COLUMNS,
                predictions,
            )
    if gold and audit_gold:
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        predictions, _ = infer_loader(model, loader(gold), "cuda")
        write_csv(out / "gold_predictions.csv", SUBMISSION_COLUMNS, predictions)
        scores = evaluate_rows(index_studies(gold), index_studies(predictions))
        scores["independence"] = metadata["gold_independence"]
        dump_json(out / "gold_metrics.json", scores)
    return {
        "run_dir": str(out),
        "best_diagnostic_train_bce" if diagnostic_mode else "best_weak_valid_bce": best,
        "checkpoint": str(out / "best.pt"),
        "diagnostic_mode": diagnostic_mode,
    }


def predict(test_csv, cache_dir, checkpoint_path, output, device="cuda"):
    if device == "cuda":
        ensure_cuda()
    start = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("targets") != list(TARGETS):
        raise ValueError("Unsupported checkpoint or target order")
    config = checkpoint["config"]
    if checkpoint.get("input_normalization", "legacy") != config["model"].get("input_normalization", "legacy"):
        raise ValueError("Checkpoint input normalization metadata is inconsistent")
    if checkpoint["preprocess_fingerprint"] != preprocess_fingerprint(config):
        raise ValueError("Checkpoint preprocessing metadata is inconsistent")
    cache_meta = check_cache_origin(cache_dir, config, "test")
    if cache_meta["studies_csv_sha256"] != sha256(test_csv):
        raise ValueError("Test cache originates from a different test.csv")
    _, rows = read_csv(test_csv, (ID,))
    index_studies(rows)
    # Construct inputs from IDs only: reports and test labels are never consumed.
    inputs = [{ID: r[ID]} for r in rows]
    loader = DataLoader(StudyDataset(inputs, cache_dir, config), batch_size=1, shuffle=False, num_workers=0)
    model = KneeMIL(config["model"]["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    predictions, _ = infer_loader(model, loader, device)
    write_csv(output, SUBMISSION_COLUMNS, predictions)
    validation = validate_submission(output, test_csv)
    elapsed = time.perf_counter() - start
    result = {
        **validation,
        "seconds": elapsed,
        "seconds_per_study": elapsed / len(rows),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_sha256": source_hashes(),
        "device": device,
    }
    dump_json(Path(output).with_suffix(".meta.json"), result)
    return result
