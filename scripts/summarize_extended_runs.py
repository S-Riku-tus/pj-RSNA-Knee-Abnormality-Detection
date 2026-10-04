"""Audit paired fold-1 and extended BN runs with CSV/JSON only; never submit or pick a winner."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

from rsna_knee.contracts import ID, TARGETS, index_studies, read_csv, sha256, source_hashes
from rsna_knee.diagnostics import prediction_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "v1-vmohitrao-research"
DEFAULT_LEGACY_C = ROOT / "artifacts" / "runs" / "e005-pretrained-imagenet"
SPEC = importlib.util.spec_from_file_location(
    "controlled_summary_helpers", ROOT / "scripts/summarize_controlled_runs.py"
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
require, read_json, finite, match = HELPERS.require, HELPERS.read_json, HELPERS.finite, HELPERS.match
differences, predictions, manifest_truth = HELPERS.differences, HELPERS.predictions, HELPERS.manifest_truth

SELECTION_RULES = {
    "bce": "minimum weak validation masked BCE; gold never used for selection",
    "auc": (
        "maximum weak validation macro AUC over all 12 targets; "
        "ties minimum masked BCE then earliest epoch; gold never used for selection"
    ),
}
COUNTERS = ("optimizer_steps", "optimizer_step_opportunities", "amp_skipped_steps")
MILESTONES = (5, 10, 15, 20)
CONDITIONS = ("B_fold1", "C_fold1", "C20_bn_update", "C20_bn_freeze")


def comparison_config(config):
    """Normalize only explicit checkpoint defaults when comparing saved configurations."""
    value = copy.deepcopy(config)
    value["train"].setdefault("checkpoint_selection", "bce")
    value["train"].setdefault("checkpoint_milestones", list(MILESTONES))
    return value


def integer(value, name, *, lower=0):
    require(type(value) is int and value >= lower, f"Invalid {name}")
    return value


def best_bce(epochs):
    return min(epochs, key=lambda item: (item["weak_valid_bce"], item["epoch"]))


def best_auc(epochs):
    defined = [item for item in epochs if item["weak_macro_auc_12"] is not None]
    return (
        min(defined, key=lambda item: (-item["weak_macro_auc_12"], item["weak_valid_bce"], item["epoch"]))
        if defined
        else None
    )


def metric_summary(metrics):
    """Emit only named aggregate fields, keeping arbitrary JSON text and private rows out."""
    targets = {}
    for target in TARGETS:
        cell = metrics["per_target"][target]
        targets[target] = {
            name: cell[name]
            for name in (
                "auc",
                "observed",
                "binary_observed",
                "positive",
                "negative",
                "missing",
                "soft_excluded",
                "masked_bce",
            )
        }
    auxiliary = [cell["auc"] for target, cell in targets.items() if target != "Synovitis"]
    defined = [value for value in auxiliary if value is not None]
    return {
        "macro_auc_12": metrics["macro_auc_12"],
        "available_class_macro_auc": metrics["available_class_macro_auc"],
        "defined_classes": metrics["defined_classes"],
        "macro_auc_11_without_synovitis": sum(defined) / 11 if len(defined) == 11 else None,
        "auxiliary_defined_classes": len(defined),
        "observed_cells": metrics["observed_cells"],
        "cell_mean_bce": metrics["cell_mean_bce"],
        "observed_target_mean_bce": metrics["observed_target_mean_bce"],
        "per_target": targets,
    }


def audited_metrics(metrics_path, predictions_path, truth):
    metrics = read_json(metrics_path)
    estimated = predictions(predictions_path, truth.keys())
    match(prediction_diagnostics(truth, estimated), metrics, "epoch metrics")
    bce = finite(metrics["cell_mean_bce"], "weak BCE", lower=0)
    losses, weighted_losses, observed = [], [], 0
    for target in TARGETS:
        cell = metrics["per_target"][target]
        count = integer(cell["observed"], "observed cells")
        observed += count
        if count:
            loss = finite(cell["masked_bce"], "target BCE", lower=0)
            losses.append(loss)
            weighted_losses.append(loss * count)
        else:
            require(cell["masked_bce"] is None, "Unobserved target has a BCE")
    match(sum(losses) / len(losses) if losses else None, metrics["observed_target_mean_bce"], "target-mean BCE")
    require(observed == metrics["observed_cells"] and observed > 0, "Observed-cell count differs from manifest")
    require(
        math.isclose(sum(weighted_losses) / observed, bce, rel_tol=1e-6, abs_tol=1e-7),
        "Per-target weighted BCE differs from canonical BCE",
    )
    return metrics


def audit_epoch(path, item, truth, cumulative, previous_elapsed, train_truth):
    epoch = integer(item["epoch"], "epoch", lower=1)
    directory = path / "epochs" / f"{epoch:03d}"
    metrics = audited_metrics(directory / "metrics.json", directory / "predictions.csv", truth)
    bce = finite(item["weak_valid_masked_bce"], "weak BCE", lower=0)
    match(bce, metrics["cell_mean_bce"], "history/metrics BCE")
    match(item["weak_valid_macro_auc_12"], metrics["macro_auc_12"], "history/metrics AUC")
    match(item["defined_auc_classes"], metrics["defined_classes"], "history/metrics defined classes")
    train_eval = None
    if train_truth:
        train_metrics_path, train_predictions_path = (
            directory / "train_eval_metrics.json",
            directory / "train_eval_predictions.csv",
        )
        train_metrics = audited_metrics(train_metrics_path, train_predictions_path, train_truth)
        match(item["train_eval_cell_mean_bce"], train_metrics["cell_mean_bce"], "train-eval history/metrics BCE")
        match(
            item["train_eval_observed_target_mean_bce"],
            train_metrics["observed_target_mean_bce"],
            "train-eval target-mean BCE",
        )
        train_eval = {
            "studies": len(train_truth),
            "metrics": metric_summary(train_metrics),
            "metrics_sha256": sha256(train_metrics_path),
            "predictions_sha256": sha256(train_predictions_path),
            "scope": "Fixed training examples in eval mode; not holdout performance or train-mode loss",
        }
    elapsed = finite(item["elapsed_seconds"], "epoch elapsed", lower=0)
    require(elapsed >= previous_elapsed, "Epoch elapsed time decreased")
    counters = {}
    for name in COUNTERS:
        value = integer(item[name], name)
        cumulative[name] += value
        require(
            integer(item[f"cumulative_{name}"], f"cumulative {name}") == cumulative[name],
            f"Inconsistent cumulative {name}",
        )
        counters[name] = value
        counters[f"cumulative_{name}"] = cumulative[name]
    require(
        counters["optimizer_steps"] + counters["amp_skipped_steps"] == counters["optimizer_step_opportunities"],
        "Actual optimizer steps and AMP skips do not equal update opportunities",
    )
    return {
        "epoch": epoch,
        "weak_valid_bce": bce,
        "weak_macro_auc_12": metrics["macro_auc_12"],
        "metrics": metric_summary(metrics),
        "elapsed_seconds": elapsed,
        "extra_train_eval_seconds": item.get("extra_train_eval_seconds"),
        "train_eval": train_eval,
        "metrics_sha256": sha256(directory / "metrics.json"),
        "predictions_sha256": sha256(directory / "predictions.csv"),
        **counters,
    }


def audit_checkpoint(path, entry, epoch, name, *, selected=False, criterion=None):
    require(isinstance(entry, dict), f"Missing checkpoint index entry {name}")
    checkpoint_name = "best.pt" if selected else f"{name}.pt"
    epoch_dir = f"epochs/{epoch['epoch']:03d}"
    require(entry["checkpoint"] == checkpoint_name, f"Unexpected checkpoint path for {name}")
    require(entry["metrics_json"] == f"{epoch_dir}/metrics.json", "Checkpoint metrics path differs")
    expected_predictions = "weak_valid_predictions.csv" if selected else f"{epoch_dir}/predictions.csv"
    require(entry["predictions_csv"] == expected_predictions, "Checkpoint prediction path differs")
    match(epoch["epoch"], entry["epoch"], "checkpoint epoch")
    match(epoch["weak_valid_bce"], entry["masked_bce"], "checkpoint BCE")
    match(epoch["weak_macro_auc_12"], entry["macro_auc_12"], "checkpoint AUC")
    match(epoch["cumulative_optimizer_steps"], entry["optimizer_steps"], "checkpoint optimizer steps")
    match(
        epoch["cumulative_optimizer_step_opportunities"],
        entry["optimizer_step_opportunities"],
        "checkpoint opportunities",
    )
    require(entry["metrics_sha256"] == epoch["metrics_sha256"], "Checkpoint epoch metrics hash differs")
    require(entry["predictions_sha256"] == epoch["predictions_sha256"], "Checkpoint epoch predictions hash differs")
    require(sha256(path / expected_predictions) == entry["predictions_sha256"], "Saved checkpoint predictions changed")
    checkpoint = path / checkpoint_name
    require(checkpoint.is_file(), "Checkpoint file is missing")
    require(sha256(checkpoint) == entry["checkpoint_sha256"], "Checkpoint file hash differs from its index")
    if selected:
        require(entry["criterion"] == criterion, "Selected checkpoint criterion differs")
        require(entry["source_checkpoint"] == f"best_{criterion}.pt", "Selected checkpoint source differs")
        require(
            entry["source_predictions_csv"] == f"{epoch_dir}/predictions.csv",
            "Selected epoch prediction source differs",
        )
        require(
            sha256(path / entry["source_checkpoint"]) == entry["checkpoint_sha256"],
            "Selected weights differ from source",
        )
    fields = (
        "checkpoint",
        "checkpoint_sha256",
        "epoch",
        "masked_bce",
        "macro_auc_12",
        "predictions_csv",
        "predictions_sha256",
        "metrics_json",
        "metrics_sha256",
        "optimizer_steps",
        "optimizer_step_opportunities",
    )
    if selected:
        fields += ("criterion", "source_checkpoint", "source_predictions_csv")
    return {**{field: entry[field] for field in fields}, "checkpoint_bytes": checkpoint.stat().st_size}


def audit_run(path, condition, manifest_dir):
    execution = read_json(path / "execution.json")
    require(execution.get("status") == "completed", f"Run {condition} is not completed")
    config, metadata, history = (read_json(path / name) for name in ("config.json", "run.json", "history.json"))
    require(metadata.get("diagnostic_mode") is False, "Training diagnostics cannot be compared as holdout runs")
    require(metadata.get("gold_audit_enabled") is False, "Extended comparison must disable gold audit")
    require(config.get("diagnostics", {}).get("audit_gold") is False, "Config must explicitly disable gold audit")
    require(config.get("diagnostics", {}).get("enabled") is True, "Epoch diagnostics are required")
    require(
        not (path / "gold_predictions.csv").exists() and not (path / "gold_metrics.json").exists(),
        "Unexpected gold audit artifacts",
    )
    criterion = config["train"].get("checkpoint_selection", "bce")
    require(criterion in SELECTION_RULES, "Unknown checkpoint selection criterion")
    require(metadata.get("checkpoint_selection") == SELECTION_RULES[criterion], "Unexpected checkpoint selection rule")
    require(metadata.get("checkpoint_selection_criterion") == criterion, "Checkpoint criterion metadata differs")
    require(metadata.get("checkpoint_milestones") == list(MILESTONES), "Planned checkpoint milestones differ")
    require(metadata.get("checkpoint_resume_supported") is False, "Checkpoint metadata claims resume support")
    require(execution.get("config") == config, "Launch and saved config differ")
    require(execution.get("fold") == metadata.get("fold"), "Launch and run fold differ")
    require(execution.get("diagnostic_studies") is None, "Launch was a training-subset diagnostic")
    require(
        execution.get("source_sha256") == metadata.get("source_sha256"), "Source changed before run metadata was saved"
    )
    require(
        execution.get("source_sha256_at_finish") == metadata.get("source_sha256"),
        "Source changed or finish hash missing",
    )
    require(metadata.get("source_sha256") == source_hashes(), "Current source differs from the audited run source")
    original_config = Path(execution["config_path"])
    if not original_config.is_absolute():
        original_config = ROOT / original_config
    require(sha256(original_config) == execution.get("config_file_sha256"), "Original config file changed")
    require(read_json(original_config) == config, "Original and saved config contents differ")
    manifest, training, truth, hashes = manifest_truth(manifest_dir, metadata["fold"])
    require(metadata["manifest"] == manifest, "Run manifest metadata differs")
    require(metadata["manifest_sha256"] == hashes, "Run manifest files differ from supplied manifest hashes")
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
    fields, subset_rows = read_csv(path / "diagnostic_subset.csv")
    require(fields == [ID], "Invalid diagnostic subset header")
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
    require([item["epoch"] for item in history] == list(range(1, len(history) + 1)), "Epoch history is not contiguous")
    cumulative, epochs, previous_elapsed = dict.fromkeys(COUNTERS, 0), [], -1.0
    batch_size = integer(config["train"]["batch_size"], "batch size", lower=1)
    accumulation = integer(config["train"]["accumulation_steps"], "gradient accumulation", lower=1)
    expected_opportunities = math.ceil(math.ceil(len(training) / batch_size) / accumulation)
    for item in history:
        row = audit_epoch(path, item, truth, cumulative, previous_elapsed, {key: training[key] for key in subset})
        require(
            row["optimizer_step_opportunities"] == expected_opportunities,
            "Update opportunities differ from training partition",
        )
        epochs.append(row)
        previous_elapsed = row["elapsed_seconds"]
    bce, auc = best_bce(epochs), best_auc(epochs)
    selected = bce if criterion == "bce" else auc
    require(selected is not None, "AUC selection has no epoch with all 12 AUCs defined")
    index = read_json(path / "checkpoint_index.json")
    require(index.get("schema_version") == 1, "Unsupported checkpoint index schema")
    require(index.get("selection_data") == "weak_validation", "Checkpoint selection data is not weak validation")
    require(index.get("checkpoint_selection") == criterion, "Checkpoint index criterion differs")
    require(index.get("resume_supported") is False, "Model-only checkpoint must not claim resume support")
    expected = {"best_bce": bce, "last": epochs[-1]}
    if auc is not None:
        expected["best_auc"] = auc
    for milestone in MILESTONES:
        if milestone <= len(epochs):
            expected[f"epoch_{milestone:03d}"] = epochs[milestone - 1]
    require(set(index["checkpoints"]) == set(expected), "Checkpoint index entries differ from required checkpoints")
    checkpoints = {
        name: audit_checkpoint(path, index["checkpoints"][name], row, name) for name, row in expected.items()
    }
    selected_checkpoint = audit_checkpoint(
        path, index["selected"], selected, "selected", selected=True, criterion=criterion
    )
    match(bce["weak_valid_bce"], execution["result"]["best_weak_valid_bce"], "execution best BCE")
    for key, expected_value in (
        ("selected_epoch", selected["epoch"]),
        ("selected_masked_bce", selected["weak_valid_bce"]),
        ("selected_macro_auc_12", selected["weak_macro_auc_12"]),
        ("checkpoint_selection", criterion),
        *((name, cumulative[name]) for name in COUNTERS),
    ):
        match(expected_value, execution["result"][key], f"execution {key}")
    initialization = config["model"].get("initialization", "random")
    require(metadata["initialization"]["name"] == initialization, "Initialization record differs from config")
    if initialization == "imagenet1k_v1":
        require(
            metadata["initialization"]["sha256"] == config["model"]["pretrained_sha256"],
            "Pretrained hash differs from config",
        )
    milestones = []
    for milestone in MILESTONES:
        if milestone <= len(epochs):
            subset_epochs = epochs[:milestone]
            milestones.append(
                {
                    "epoch": milestone,
                    "at_epoch": copy.deepcopy(epochs[milestone - 1]),
                    "best_bce_through_epoch": copy.deepcopy(best_bce(subset_epochs)),
                    "best_auc_through_epoch": copy.deepcopy(best_auc(subset_epochs)),
                }
            )
    return {
        "condition": condition,
        "run_dir": str(path),
        "config": config,
        "original_config_file_sha256": sha256(original_config),
        "saved_config_file_sha256": sha256(path / "config.json"),
        "inputs_sha256": manifest["inputs_sha256"],
        "manifest_sha256": hashes,
        "source_sha256": metadata["source_sha256"],
        "cache_fingerprint": metadata["cache"]["fingerprint"],
        "seed": config["seed"],
        "fold": metadata["fold"],
        "initial_head_sha256": metadata["initial_head_sha256"],
        "diagnostic_subset_sha256": subset_hash,
        "train_studies": len(training),
        "valid_studies": len(truth),
        "runtime": metadata["runtime"],
        "gpu": metadata["gpu"],
        "git_head_at_start": execution.get("git_head_at_start"),
        "execution_seconds": finite(execution["elapsed_seconds"], "execution elapsed", lower=0),
        "elapsed_scope": execution.get("elapsed_scope"),
        "peak_torch_allocated_bytes": finite(execution["peak_allocated_bytes"], "peak allocated memory", lower=0),
        "epochs": epochs,
        "milestones": milestones,
        "best_epoch_by_weak_bce": bce["epoch"],
        "best_epoch_by_weak_auc": auc["epoch"] if auc else None,
        "selected_criterion": criterion,
        "selected_epoch": selected["epoch"],
        "selected_metrics": selected["metrics"],
        "checkpoint_index_sha256": sha256(path / "checkpoint_index.json"),
        "checkpoints": checkpoints,
        "selected_checkpoint": selected_checkpoint,
        "actual_optimizer_steps": cumulative["optimizer_steps"],
        "optimizer_step_opportunities": cumulative["optimizer_step_opportunities"],
        "amp_skipped_steps": cumulative["amp_skipped_steps"],
        "gold_macro_auc_12": None,
        "public_lb": None,
    }


def metric_difference(left, right):
    def delta(a, b):
        return b - a if a is not None and b is not None else None

    for target in TARGETS:
        counts = ("observed", "binary_observed", "positive", "negative", "missing", "soft_excluded")
        require(
            all(left["per_target"][target][name] == right["per_target"][target][name] for name in counts),
            "Paired metric observation counts differ",
        )
    return {
        "macro_auc_12": delta(left["macro_auc_12"], right["macro_auc_12"]),
        "macro_auc_11_without_synovitis": delta(
            left["macro_auc_11_without_synovitis"], right["macro_auc_11_without_synovitis"]
        ),
        "cell_mean_bce": delta(left["cell_mean_bce"], right["cell_mean_bce"]),
        "per_target": {
            target: {
                name: delta(left["per_target"][target][name], right["per_target"][target][name])
                for name in ("auc", "masked_bce")
            }
            for target in TARGETS
        },
    }


def paired_comparison(left, right):
    require(left["fold"] == right["fold"], "Paired comparison folds differ")
    return {
        "left": left["condition"],
        "right": right["condition"],
        "difference_direction": "right minus left",
        "selected_epoch_difference": metric_difference(left["selected_metrics"], right["selected_metrics"]),
        "best_bce_epoch_difference": metric_difference(
            left["epochs"][left["best_epoch_by_weak_bce"] - 1]["metrics"],
            right["epochs"][right["best_epoch_by_weak_bce"] - 1]["metrics"],
        ),
        "best_auc_epoch_difference": metric_difference(
            left["epochs"][left["best_epoch_by_weak_auc"] - 1]["metrics"],
            right["epochs"][right["best_epoch_by_weak_auc"] - 1]["metrics"],
        )
        if left["best_epoch_by_weak_auc"] and right["best_epoch_by_weak_auc"]
        else None,
        "same_epoch_milestones": [
            {
                "epoch": epoch,
                "difference": metric_difference(
                    left["epochs"][epoch - 1]["metrics"], right["epochs"][epoch - 1]["metrics"]
                ),
            }
            for epoch in MILESTONES
            if epoch <= min(len(left["epochs"]), len(right["epochs"]))
        ],
    }


def legacy_first_five(legacy_path, extended, manifest_dir):
    """Audit historical CSVs as a trajectory reference, without reusing its old checkpoint."""
    legacy_path = Path(legacy_path).resolve()
    execution = read_json(legacy_path / "execution.json")
    config, metadata, history = (read_json(legacy_path / name) for name in ("config.json", "run.json", "history.json"))
    require(execution.get("status") == "completed", "Legacy C reference is not completed")
    require(
        metadata.get("gold_audit_enabled") is False and metadata.get("diagnostic_mode") is False,
        "Invalid legacy C scope",
    )
    require(config["diagnostics"]["audit_gold"] is False, "Legacy C gold audit must be disabled")
    require(metadata["fold"] == extended["fold"] == 0, "Legacy C trajectory fold differs")
    require(metadata["source_sha256"] == execution["source_sha256"], "Legacy launch/run source differs")
    if "source_sha256_at_finish" in execution:
        require(execution["source_sha256_at_finish"] == metadata["source_sha256"], "Legacy finish source differs")
    require(len(history) == config["train"]["epochs"] == 5, "Legacy C must contain five complete epochs")
    require([item["epoch"] for item in history] == [1, 2, 3, 4, 5], "Legacy C epochs are not contiguous")
    require(execution["config"] == config, "Legacy launch and saved config differ")
    require(
        set(differences(comparison_config(config), comparison_config(extended["config"])))
        == {"train.epochs", "train.checkpoint_selection"},
        "Legacy C/new C must differ only in epochs and declared selection",
    )
    manifest, _, truth, hashes = manifest_truth(manifest_dir, 0)
    require(metadata["manifest"] == manifest and metadata["manifest_sha256"] == hashes, "Legacy manifest differs")
    for name in ("initial_head_sha256", "runtime", "gpu"):
        require(metadata[name] == extended[name], f"Legacy/new {name} differs")
    require(metadata["cache"]["fingerprint"] == extended["cache_fingerprint"], "Legacy/new cache differs")
    comparisons = []
    for item, new in zip(history, extended["epochs"][:5]):
        directory = legacy_path / "epochs" / f"{item['epoch']:03d}"
        metrics = read_json(directory / "metrics.json")
        estimated = predictions(directory / "predictions.csv", truth.keys())
        match(prediction_diagnostics(truth, estimated), metrics, "legacy epoch metrics")
        match(item["weak_valid_masked_bce"], metrics["cell_mean_bce"], "legacy history/metrics BCE")
        match(item["weak_valid_macro_auc_12"], metrics["macro_auc_12"], "legacy history/metrics AUC")
        require(item["weak_valid_masked_bce"] == new["weak_valid_bce"], "First-five BCE changed from legacy C")
        require(item["weak_valid_macro_auc_12"] == new["weak_macro_auc_12"], "First-five AUC changed from legacy C")
        require(
            sha256(directory / "predictions.csv") == new["predictions_sha256"],
            "First-five predictions changed from legacy C",
        )
        comparisons.append(
            {
                "epoch": item["epoch"],
                "masked_bce_exact_match": True,
                "macro_auc_12_exact_match": True,
                "predictions_sha256": new["predictions_sha256"],
                "legacy_metrics_sha256": sha256(directory / "metrics.json"),
                "new_metrics_sha256": new["metrics_sha256"],
            }
        )
    return {
        "run_dir": str(legacy_path),
        "scope": "Historical weak trajectory reference only; old checkpoint is not selected or re-adopted",
        "legacy_recorded_source_sha256": metadata["source_sha256"],
        "new_recorded_source_sha256": extended["source_sha256"],
        "source_changes": differences(metadata["source_sha256"], extended["source_sha256"]),
        "historical_source_matches_current": metadata["source_sha256"] == source_hashes(),
        "legacy_current_source_match_required": False,
        "legacy_execution_sha256": sha256(legacy_path / "execution.json"),
        "legacy_config_sha256": sha256(legacy_path / "config.json"),
        "legacy_history_sha256": sha256(legacy_path / "history.json"),
        "first_five_epochs_exact_match": True,
        "epochs": comparisons,
    }


def summarize(runs, output, manifest_dir=DEFAULT_MANIFEST, legacy_c_run=None):
    output, manifest_dir = Path(output), Path(manifest_dir)
    require(not output.exists(), "Summary output already exists; preserve previous records")
    require(len(runs) == 4, "Specify B fold1, C fold1, C20 BN update, C20 BN freeze runs in that order")
    paths = [Path(path).resolve() for path in runs]
    require(len(set(paths)) == 4, "Use four distinct run directories")
    results = [audit_run(path, condition, manifest_dir) for path, condition in zip(paths, CONDITIONS)]
    for result, fold, epochs, initialization, criterion, batch_norm in zip(
        results,
        (1, 1, 0, 0),
        (5, 5, 20, 20),
        ("random", "imagenet1k_v1", "imagenet1k_v1", "imagenet1k_v1"),
        ("bce", "bce", "auc", "auc"),
        ("update", "update", "update", "freeze"),
    ):
        model = result["config"]["model"]
        require(
            result["fold"] == fold and len(result["epochs"]) == epochs,
            "Unexpected fold or epoch count for planned comparison",
        )
        require(
            model.get("initialization", "random") == initialization
            and model.get("input_normalization", "legacy") == "imagenet",
            "Unexpected initialization or normalization for planned comparison",
        )
        require(result["selected_criterion"] == criterion, "Unexpected predeclared checkpoint selection criterion")
        require(model.get("bn_running_stats", "update") == batch_norm, "Unexpected BN statistics mode")
    bc = differences(results[0]["config"], results[1]["config"])
    bn = differences(results[2]["config"], results[3]["config"])
    require(
        set(bc) == {"model.initialization", "model.pretrained_path", "model.pretrained_sha256"},
        "Fold1 B/C must differ only in encoder initialization",
    )
    require(bn == ["model.bn_running_stats"], "C20 BN comparison must differ only in BN statistics mode")
    c_short_long = differences(comparison_config(results[1]["config"]), comparison_config(results[2]["config"]))
    require(
        set(c_short_long) == {"train.epochs", "train.checkpoint_selection"},
        "C settings across folds must differ only in epoch count and declared selection",
    )
    shared = {}
    for name in (
        "inputs_sha256",
        "manifest_sha256",
        "source_sha256",
        "cache_fingerprint",
        "seed",
        "initial_head_sha256",
        "runtime",
        "gpu",
    ):
        require(all(result[name] == results[0][name] for result in results[1:]), f"Shared {name} differs")
        shared[name] = copy.deepcopy(results[0][name])
    for left, right in ((results[0], results[1]), (results[2], results[3])):
        require(
            left["diagnostic_subset_sha256"] == right["diagnostic_subset_sha256"],
            "Paired diagnostic subset hashes differ",
        )
    legacy = legacy_first_five(legacy_c_run, results[2], manifest_dir) if legacy_c_run is not None else None
    report = {
        "schema_version": 1,
        "scope": "Local weak-label comparison; no automatic winner selection or external upload/submission",
        "validation": {
            "valid": True,
            "gold_audit_disabled": True,
            "source_launch_finish_and_current_match": True,
            "all_epoch_predictions_and_observation_counts_match_manifest": True,
            "checkpoint_hashes_and_predeclared_selection_audited": True,
            "actual_optimizer_steps_and_amp_skips_audited": True,
            "single_factor_changes": {"fold1_B_to_C": bc, "C20_bn_update_to_freeze": bn},
        },
        "shared": shared,
        "runs": results,
        "comparisons": {
            "fold1_pretraining": paired_comparison(*results[:2]),
            "fold0_bn_statistics": paired_comparison(*results[2:]),
        },
        "legacy_C_first_five": legacy,
        "selected_winner": None,
        "gold_macro_auc_12": None,
        "public_lb": None,
        "limitations": [
            "Weak labels are report-derived; weak AUC does not establish image-label or public performance",
            "Supplied groups and reports are separated; patient independence is unverified",
            "Epoch selection uses the same validation fold; selected-epoch comparisons are descriptive",
            "11-target AUC excludes Synovitis only as a diagnostic; undefined classes never change denominators silently",
            "Model weights are hashed but not deserialized; checkpoint internal metadata needs a separate torch check",
            "Model-only last checkpoint does not contain the full optimizer/scaler/RNG state for exact resume",
            "Current and recorded source hashes agree; this is not a continuous filesystem audit",
            "Fold0 and fold1 use different validation sets and are not paired against each other",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, required=True, type=Path, metavar="RUN")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--legacy-c-run", type=Path, default=DEFAULT_LEGACY_C, help="Historical five-epoch C trajectory reference"
    )
    arguments = parser.parse_args()
    try:
        result = summarize(arguments.runs, arguments.output, arguments.manifest_dir, arguments.legacy_c_run)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        json.dumps(
            {"output": str(arguments.output), "valid": result["validation"]["valid"], "runs": len(result["runs"])}
        )
    )
