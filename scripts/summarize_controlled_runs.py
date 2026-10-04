"""Audit three completed A/B/C runs using CSV/JSON only; never select a winner."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from rsna_knee.contracts import (
    ID,
    SUBMISSION_COLUMNS,
    TARGETS,
    index_studies,
    probability,
    read_csv,
    sha256,
    source_hashes,
)
from rsna_knee.diagnostics import prediction_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "v1-vmohitrao-research"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, (dict, list)), f"Invalid JSON structure: {Path(path).name}")
    return value


def finite(value, name, *, lower=None, upper=None):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite or invalid {name}")
    require(lower is None or value >= lower, f"{name} below its allowed range")
    require(upper is None or value <= upper, f"{name} above its allowed range")
    return value


def match(expected, actual, name):
    """Compare aggregates without including row identifiers or values in errors."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Invalid {name} structure")
        for key, value in expected.items():
            require(key in actual, f"Missing {name}.{key}")
            match(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        finite(actual, name)
        require(math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12), f"Inconsistent {name}")
    else:
        require(expected == actual, f"Inconsistent {name}")


def predictions(path, expected_ids):
    fields, rows = read_csv(path)
    require(fields == list(SUBMISSION_COLUMNS), "Prediction CSV must contain ID and the ordered 12 targets")
    indexed = index_studies(rows)
    require(indexed.keys() == expected_ids, "Prediction CSV study set differs from validation manifest")
    for row in indexed.values():
        for target in TARGETS:
            probability(row[target])
    return indexed


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def manifest_truth(manifest_dir, fold):
    metadata = read_json(manifest_dir / "manifest.json")
    _, weak_rows = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "fold", "group_id", "source"))
    _, gold_rows = read_csv(manifest_dir / "gold.csv", (ID, "group_id", "source"))
    weak, gold = index_studies(weak_rows), index_studies(gold_rows)
    require(not weak.keys() & gold.keys(), "Gold studies overlap weak manifest")
    require(
        not {r["group_id"] for r in weak_rows} & {r["group_id"] for r in gold_rows}, "Gold groups overlap weak manifest"
    )
    require(type(fold) is int and 0 <= fold < metadata["n_folds"], "Invalid validation fold")
    require(all(r["source"] == "report_weak" for r in weak_rows), "Unexpected weak manifest source")
    require(all(0 <= int(r["fold"]) < metadata["n_folds"] for r in weak_rows), "Invalid manifest fold assignment")
    training = {key: row for key, row in weak.items() if int(row["fold"]) != fold}
    truth = {key: row for key, row in weak.items() if int(row["fold"]) == fold}
    require(bool(training) and bool(truth), "Empty training or validation partition")
    require(
        not {r["group_id"] for r in training.values()} & {r["group_id"] for r in truth.values()}, "Groups overlap folds"
    )
    for row in weak.values():
        for target in TARGETS:
            probability(row[target], missing=True)
    hashes = {name: sha256(manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv")}
    return metadata, training, truth, hashes


def audit_run(path, condition, manifest_dir):
    execution = read_json(path / "execution.json")
    require(execution.get("status") == "completed", f"Run {condition} is not completed")
    config, metadata, history = (read_json(path / name) for name in ("config.json", "run.json", "history.json"))
    require(metadata.get("diagnostic_mode") is False, "Memorization diagnostics cannot be compared as holdout runs")
    require(metadata.get("gold_audit_enabled") is False, "Controlled comparison must disable gold audit")
    require(config.get("diagnostics", {}).get("audit_gold") is False, "Config must explicitly disable gold audit")
    require(config.get("diagnostics", {}).get("enabled") is True, "Epoch diagnostics are required")
    require(
        not (path / "gold_predictions.csv").exists() and not (path / "gold_metrics.json").exists(),
        "Unexpected gold audit artifacts",
    )
    require(
        metadata.get("checkpoint_selection") == "minimum weak validation masked BCE; gold never used for selection",
        "Unexpected checkpoint selection rule",
    )
    require(execution.get("config") == config, "Launch and saved config differ")
    require(execution.get("fold") == metadata.get("fold"), "Launch and run fold differ")
    require(execution.get("diagnostic_studies") is None, "Launch was a training-subset diagnostic")
    require(
        execution.get("source_sha256") == metadata.get("source_sha256"), "Source changed before run metadata was saved"
    )
    finish_source_recorded = "source_sha256_at_finish" in execution
    if finish_source_recorded:
        require(
            execution["source_sha256_at_finish"] == metadata.get("source_sha256"),
            "Source changed before completion",
        )
    require(metadata.get("source_sha256") == source_hashes(), "Current source differs from the audited run source")
    source_config = Path(execution["config_path"])
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    source_config_hash = sha256(source_config)
    require(source_config_hash == execution.get("config_file_sha256"), "Original config file changed")
    require(read_json(source_config) == config, "Original and saved config contents differ")
    manifest, training, truth, manifest_hashes = manifest_truth(manifest_dir, metadata["fold"])
    require(metadata["manifest"] == manifest, "Run manifest metadata differs from the supplied manifest")
    require(metadata["manifest_sha256"] == manifest_hashes, "Run manifest files differ from supplied manifest hashes")
    require(
        config["seed"] == manifest["seed"] and config["n_folds"] == manifest["n_folds"],
        "Config and manifest seed/folds differ",
    )
    require(
        metadata["train_studies"] == len(training) and metadata["valid_studies"] == len(truth),
        "Run partition counts differ",
    )
    require(
        metadata["cache"]["studies_csv_sha256"] == manifest["inputs_sha256"]["train"],
        "Cache source differs from label source",
    )
    require(metadata["cache"]["preprocess"] == config["preprocess"], "Cache preprocessing differs from config")
    subset_fields, subset_rows = read_csv(path / "diagnostic_subset.csv")
    require(subset_fields == [ID], "Invalid diagnostic subset header")
    subset = index_studies(subset_rows)
    require(subset.keys() <= training.keys(), "Diagnostic subset contains non-training studies")
    require(
        len(subset) == min(config["diagnostics"]["train_eval_studies"], len(training)), "Diagnostic subset size differs"
    )
    selection = read_json(path / "diagnostic_selection.json")
    subset_hash = sha256(path / "diagnostic_subset.csv")
    require(selection["subset_csv_sha256"] == subset_hash, "Diagnostic subset hash mismatch")
    require(
        selection["studies"] == len(subset)
        and selection["seed"] == config["seed"]
        and selection["fold"] == metadata["fold"],
        "Diagnostic selection metadata differs",
    )
    require(
        isinstance(history, list) and len(history) == config["train"]["epochs"] and bool(history),
        "Run history is incomplete",
    )
    require([r["epoch"] for r in history] == list(range(1, len(history) + 1)), "Epoch history is not contiguous")
    epochs, saved_metrics, previous_elapsed = [], {}, -1.0
    for item in history:
        epoch = item["epoch"]
        directory = path / "epochs" / f"{epoch:03d}"
        metrics = read_json(directory / "metrics.json")
        estimated = predictions(directory / "predictions.csv", truth.keys())
        match(prediction_diagnostics(truth, estimated), metrics, "epoch metrics")
        bce = finite(item["weak_valid_masked_bce"], "weak BCE", lower=0)
        match(bce, metrics["cell_mean_bce"], "history/metrics BCE")
        match(item["weak_valid_macro_auc_12"], metrics["macro_auc_12"], "history/metrics AUC")
        match(item["defined_auc_classes"], metrics["defined_classes"], "history/metrics defined classes")
        observed = 0
        target_losses = []
        for target in TARGETS:
            cell = metrics["per_target"][target]
            observed += cell["observed"]
            if cell["observed"]:
                target_losses.append(finite(cell["masked_bce"], "target BCE", lower=0))
            else:
                require(cell["masked_bce"] is None, "Unobserved target has a BCE")
        require(metrics["observed_cells"] == observed, "Observed-cell count differs from manifest labels")
        target_mean = sum(target_losses) / len(target_losses) if target_losses else None
        match(target_mean, metrics["observed_target_mean_bce"], "target-mean BCE")
        elapsed = finite(item["elapsed_seconds"], "epoch elapsed", lower=0)
        require(elapsed >= previous_elapsed, "Epoch elapsed time decreased")
        previous_elapsed = elapsed
        saved_metrics[epoch] = metrics
        epochs.append(
            {
                "epoch": epoch,
                "weak_valid_bce": bce,
                "weak_macro_auc_12": metrics["macro_auc_12"],
                "available_class_macro_auc": metrics["available_class_macro_auc"],
                "defined_classes": metrics["defined_classes"],
                "observed_target_mean_bce": target_mean,
                "observed_cells": observed,
                "elapsed_seconds": elapsed,
                "extra_train_eval_seconds": item.get("extra_train_eval_seconds"),
                "metrics_sha256": sha256(directory / "metrics.json"),
                "predictions_sha256": sha256(directory / "predictions.csv"),
            }
        )
    best = min(epochs, key=lambda row: row["weak_valid_bce"])
    match(best["weak_valid_bce"], execution["result"]["best_weak_valid_bce"], "execution best BCE")
    best_predictions = path / "weak_valid_predictions.csv"
    predictions(best_predictions, truth.keys())
    require(
        sha256(best_predictions) == best["predictions_sha256"],
        "Best predictions do not match the first minimum-BCE epoch",
    )
    require((path / "best.pt").is_file(), "Best checkpoint is missing")
    elapsed = finite(execution["elapsed_seconds"], "execution elapsed", lower=0)
    peak = finite(execution["peak_allocated_bytes"], "peak allocated memory", lower=0)
    initialization = config["model"].get("initialization", "random")
    require(metadata["initialization"]["name"] == initialization, "Initialization record differs from config")
    if initialization == "imagenet1k_v1":
        require(
            metadata["initialization"]["sha256"] == config["model"]["pretrained_sha256"],
            "Pretrained hash differs from config",
        )
    return {
        "condition": condition,
        "run_dir": str(path),
        "status": "completed",
        "config": config,
        "original_config_file_sha256": source_config_hash,
        "saved_config_file_sha256": sha256(path / "config.json"),
        "config_contents_match": True,
        "inputs_sha256": manifest["inputs_sha256"],
        "manifest_sha256": manifest_hashes,
        "source_sha256": metadata["source_sha256"],
        "cache_fingerprint": metadata["cache"]["fingerprint"],
        "seed": config["seed"],
        "fold": metadata["fold"],
        "initial_head_sha256": metadata["initial_head_sha256"],
        "diagnostic_subset_sha256": subset_hash,
        "diagnostic_subset_studies": len(subset),
        "train_studies": len(training),
        "valid_studies": len(truth),
        "runtime": metadata["runtime"],
        "gpu": metadata["gpu"],
        "gold_audit_enabled": False,
        "launch_run_and_current_source_match": True,
        "finish_source_hash_recorded": finish_source_recorded,
        "finish_source_matches_start": True if finish_source_recorded else None,
        "git_head_at_start": execution.get("git_head_at_start"),
        "execution_seconds": elapsed,
        "elapsed_scope": execution.get("elapsed_scope"),
        "peak_torch_allocated_bytes": peak,
        "epochs": epochs,
        "best_epoch_by_weak_bce": best["epoch"],
        "best_weak_valid_bce": best["weak_valid_bce"],
        "best_metrics": saved_metrics[best["epoch"]],
        "best_predictions_csv_sha256": sha256(best_predictions),
        "checkpoint_sha256": sha256(path / "best.pt"),
        "checkpoint_bytes": (path / "best.pt").stat().st_size,
        "gold_macro_auc_12": None,
        "public_lb": None,
    }


def summarize(runs, output, manifest_dir=DEFAULT_MANIFEST):
    output, manifest_dir = Path(output), Path(manifest_dir)
    require(not output.exists(), "Summary output already exists; preserve previous records")
    require(len(runs) == 3, "Specify exactly three runs in A, B, C order")
    paths = [Path(path).resolve() for path in runs]
    require(len(set(paths)) == 3, "A/B/C must use three distinct run directories")
    results = [audit_run(path, condition, manifest_dir) for path, condition in zip(paths, "ABC")]
    expected_modes = (("random", "legacy"), ("random", "imagenet"), ("imagenet1k_v1", "imagenet"))
    for result, (initialization, normalization) in zip(results, expected_modes):
        model = result["config"]["model"]
        require(
            model.get("initialization", "random") == initialization
            and model.get("input_normalization", "legacy") == normalization,
            "Run conditions must be A=random/legacy, B=random/ImageNet, C=pretrained/ImageNet",
        )
    ab = differences(results[0]["config"], results[1]["config"])
    bc = differences(results[1]["config"], results[2]["config"])
    require(ab == ["model.input_normalization"], "A/B must differ only in input normalization")
    require(
        set(bc) == {"model.initialization", "model.pretrained_path", "model.pretrained_sha256"},
        "B/C must differ only in encoder initialization and its path/hash",
    )
    shared_fields = (
        "inputs_sha256",
        "manifest_sha256",
        "source_sha256",
        "cache_fingerprint",
        "seed",
        "fold",
        "initial_head_sha256",
        "diagnostic_subset_sha256",
        "runtime",
        "gpu",
    )
    shared = {}
    for key in shared_fields:
        require(all(result[key] == results[0][key] for result in results[1:]), f"A/B/C shared {key} differs")
        shared[key] = copy.deepcopy(results[0][key])
    report = {
        "schema_version": 1,
        "scope": "Local research comparison; no automated model selection or Kaggle submission",
        "validation": {
            "valid": True,
            "gold_audit_disabled": True,
            "source_records_and_current_source_match": True,
            "prediction_study_sets_and_observation_counts_match_manifest": True,
            "best_selected_by_weak_bce": True,
            "single_factor_changes": {"A_to_B": ab, "B_to_C": bc},
        },
        "shared": shared,
        "runs": results,
        "selected_winner": None,
        "gold_macro_auc_12": None,
        "public_lb": None,
        "limitations": [
            "Weak metrics evaluate report-derived labels; they do not prove image-label or public performance",
            "Current source and launch/run records agree; this is not a continuous filesystem audit",
            "AUC values are diagnostic; the first minimum-BCE epoch remains the checkpoint selection rule",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, type=Path, help="Completed A, B, C directories in that order"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    try:
        result = summarize(arguments.runs, arguments.output, arguments.manifest_dir)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        json.dumps(
            {"output": str(arguments.output), "valid": result["validation"]["valid"], "runs": len(result["runs"])}
        )
    )
