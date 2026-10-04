"""Model-only checkpoint selection and audit records; no training resume state."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch

from .contracts import TARGETS, dump_json, preprocess_fingerprint, sha256


def selection_criterion(config, *, diagnostic_mode=False):
    criterion = config["train"].get("checkpoint_selection", "bce")
    if criterion not in ("bce", "auc"):
        raise ValueError("checkpoint_selection must be bce or auc")
    return "bce" if diagnostic_mode else criterion


def write_progress(out, epoch, epochs, completed_microbatches, total_microbatches, counter, elapsed_seconds, status):
    """Write read-only observations of the current run without inspecting the model."""
    dump_json(
        Path(out) / "progress.json",
        {
            "epoch": epoch,
            "epochs": epochs,
            "completed_microbatches": completed_microbatches,
            "total_microbatches": total_microbatches,
            "cumulative_optimizer_steps": counter.steps,
            "cumulative_optimizer_step_opportunities": counter.opportunities,
            "cumulative_amp_skipped_steps": counter.skipped,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
        },
    )


class OptimizerStepCounter:
    """Count actual optimizer calls and opportunities; AMP skips are excluded from steps."""

    def __init__(self, optimizer):
        self.steps = 0
        self.opportunities = 0
        self._handle = optimizer.register_step_post_hook(self._completed_step)

    def _completed_step(self, optimizer, args, kwargs):
        self.steps += 1

    def opportunity(self):
        self.opportunities += 1

    @property
    def skipped(self):
        return self.opportunities - self.steps

    def close(self):
        self._handle.remove()


class CheckpointTracker:
    """Save inference-compatible weights and connect each to its epoch predictions."""

    def __init__(self, out, config, fold, *, diagnostic_mode=False):
        self.out = Path(out)
        self.config = config
        self.fold = fold
        self.diagnostic_mode = diagnostic_mode
        self.criterion = selection_criterion(config, diagnostic_mode=diagnostic_mode)
        self.milestones = config["train"].get("checkpoint_milestones", [5, 10, 15, 20])
        if not isinstance(self.milestones, list) or any(
            type(epoch) is not int or epoch < 1 for epoch in self.milestones
        ):
            raise ValueError("checkpoint_milestones must be a list of positive epoch integers")
        if len(self.milestones) != len(set(self.milestones)):
            raise ValueError("checkpoint_milestones must not contain duplicate epochs")
        self.index = {
            "schema_version": 1,
            "checkpoint_selection": self.criterion,
            "selection_data": "training_diagnostic" if diagnostic_mode else "weak_validation",
            "resume_supported": False,
            "note": "Model weights only for inference; optimizer, scaler and RNG states are absent. Cannot resume training.",
            "auc_rule": "All 12 AUCs must be defined; maximize macro AUC, ties minimum masked BCE then earliest epoch.",
            "checkpoints": {},
            "selected": None,
        }

    def _save(self, name, model, record):
        path = self.out / f"{name}.pt"
        torch.save(
            {
                "schema_version": 1,
                "model": model.state_dict(),
                "config": self.config,
                "targets": list(TARGETS),
                "epoch": record["epoch"],
                "fold": self.fold,
                "preprocess_fingerprint": preprocess_fingerprint(self.config),
                "input_normalization": self.config["model"].get("input_normalization", "legacy"),
                "resume_supported": False,
                "selection_data": self.index["selection_data"],
                "metrics": {key: record[key] for key in ("masked_bce", "macro_auc_12")},
                "predictions_sha256": record["predictions_sha256"],
                "optimizer_steps": record["optimizer_steps"],
                "optimizer_step_opportunities": record["optimizer_step_opportunities"],
            },
            path,
        )
        self.index["checkpoints"][name] = {
            **record,
            "checkpoint": path.name,
            "checkpoint_sha256": sha256(path),
        }

    def update(self, model, epoch, bce, auc, predictions_csv, metrics_json, *, steps, opportunities):
        if not math.isfinite(bce):
            raise ValueError("Checkpoint masked BCE must be finite")
        # The caller supplies macro_auc_12 only when every class has a defined AUC.
        if auc is not None and (not math.isfinite(auc) or not 0 <= auc <= 1):
            auc = None
        predictions_csv, metrics_json = Path(predictions_csv), Path(metrics_json)
        record = {
            "epoch": epoch,
            "masked_bce": bce,
            "macro_auc_12": auc,
            "predictions_csv": predictions_csv.relative_to(self.out).as_posix(),
            "predictions_sha256": sha256(predictions_csv),
            "metrics_json": metrics_json.relative_to(self.out).as_posix(),
            "metrics_sha256": sha256(metrics_json),
            "optimizer_steps": steps,
            "optimizer_step_opportunities": opportunities,
        }
        checkpoints = self.index["checkpoints"]
        best_bce = checkpoints.get("best_bce")
        if best_bce is None or bce < best_bce["masked_bce"]:
            self._save("best_bce", model, record)
        best_auc = checkpoints.get("best_auc")
        if auc is not None and (
            best_auc is None
            or auc > best_auc["macro_auc_12"]
            or (auc == best_auc["macro_auc_12"] and bce < best_auc["masked_bce"])
        ):
            self._save("best_auc", model, record)
        self._save("last", model, record)
        if epoch in self.milestones:
            self._save(f"epoch_{epoch:03d}", model, record)
        selected = checkpoints.get(f"best_{self.criterion}")
        if selected is not None and (
            self.index["selected"] is None or selected["epoch"] != self.index["selected"]["epoch"]
        ):
            shutil.copyfile(self.out / selected["checkpoint"], self.out / "best.pt")
            csv_name = "diagnostic_train_predictions.csv" if self.diagnostic_mode else "weak_valid_predictions.csv"
            shutil.copyfile(self.out / selected["predictions_csv"], self.out / csv_name)
            self.index["selected"] = {
                **selected,
                "checkpoint": "best.pt",
                "predictions_csv": csv_name,
                "source_checkpoint": selected["checkpoint"],
                "source_predictions_csv": selected["predictions_csv"],
                "criterion": self.criterion,
            }
        dump_json(self.out / "checkpoint_index.json", self.index)

    @property
    def selected(self):
        selected = self.index["selected"]
        if selected is None:
            raise ValueError("No checkpoint has a defined macro AUC over all 12 targets; AUC selection cannot complete")
        return selected
