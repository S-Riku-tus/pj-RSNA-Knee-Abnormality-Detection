"""GPU training and portable inference; imported only by the relevant CLI commands."""

from __future__ import annotations

import importlib.metadata
import json
import random
import time
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
from .imaging import load_cached
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
            torch.from_numpy(images.astype(np.float32) / 127.5 - 1),
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
def infer_loader(model, loader, device):
    model.eval()
    rows, loss_sum, observations = [], 0.0, 0
    for images, mask, targets, observed, keys in loader:
        logits = model(images.to(device), mask.to(device))
        probabilities = logits.sigmoid().cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite model output")
        if observed.any():
            losses = F.binary_cross_entropy_with_logits(logits.cpu(), targets, reduction="none")
            loss_sum += float((losses * observed).sum())
            observations += int(observed.sum())
        for key, scores in zip(keys, probabilities.tolist()):
            rows.append({ID: key, **dict(zip(TARGETS, scores))})
    return rows, loss_sum / observations if observations else None


def train(manifest_dir, cache_dir, run_dir, config, fold):
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
    cache_meta = check_cache_origin(cache_dir, config, "train", metadata["inputs_sha256"]["train"])
    # Preflight every study, so missing caches fail before the expensive run begins.
    for row in [*training, *validation, *gold]:
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
    model = KneeMIL(config["model"]["dropout"]).to("cuda")
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
            "initialization": "random; no pretrained weights",
            "checkpoint_selection": "minimum weak validation masked BCE; gold never used for selection",
        },
    )
    history, best, start_time = [], float("inf"), time.perf_counter()
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
        predictions, valid_loss = infer_loader(model, valid_loader, "cuda")
        if valid_loss is None or not np.isfinite(valid_loss):
            raise ValueError("Invalid weak validation loss")
        record = {
            "epoch": epoch + 1,
            "train_batch_mean_bce": sum(losses) / len(losses),
            "weak_valid_masked_bce": valid_loss,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
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
                },
                out / "best.pt",
            )
            write_csv(out / "weak_valid_predictions.csv", SUBMISSION_COLUMNS, predictions)
    if gold:
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        predictions, _ = infer_loader(model, loader(gold), "cuda")
        write_csv(out / "gold_predictions.csv", SUBMISSION_COLUMNS, predictions)
        scores = evaluate_rows(index_studies(gold), index_studies(predictions))
        scores["independence"] = metadata["gold_independence"]
        dump_json(out / "gold_metrics.json", scores)
    return {"run_dir": str(out), "best_weak_valid_bce": best, "checkpoint": str(out / "best.pt")}


def predict(test_csv, cache_dir, checkpoint_path, output, device="cuda"):
    if device == "cuda":
        ensure_cuda()
    start = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("targets") != list(TARGETS):
        raise ValueError("Unsupported checkpoint or target order")
    config = checkpoint["config"]
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
