"""CSV-only paired group bootstrap of two weak predictions; no gold, images, weights or GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from rsna_knee.contracts import ID, TARGETS, index_studies, probability, read_csv, sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "controlled_bootstrap_helpers", ROOT / "scripts/summarize_controlled_runs.py"
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)
require, read_json, predictions = HELPERS.require, HELPERS.read_json, HELPERS.predictions


def weak_truth(manifest_dir, fold):
    """Check the weak partition without opening or hashing gold.csv."""
    manifest_dir = Path(manifest_dir)
    manifest = read_json(manifest_dir / "manifest.json")
    require(type(fold) is int and 0 <= fold < manifest["n_folds"], "Invalid validation fold")
    _, rows = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "fold", "group_id", "source"))
    weak = index_studies(rows)
    groups = {}
    for row in weak.values():
        require(row["source"] == "report_weak", "Unexpected weak manifest source")
        group = row["group_id"]
        require(bool(group) and group.strip() == group, "Invalid supplied group identifier")
        assigned_fold = int(row["fold"])
        require(0 <= assigned_fold < manifest["n_folds"], "Invalid weak fold assignment")
        require(group not in groups or groups[group] == assigned_fold, "Supplied group crosses folds")
        groups[group] = assigned_fold
        values = [probability(row[target], missing=True, binary=True) for target in TARGETS]
        require(any(value is not None for value in values), "Weak study has no observed targets")
    truth = {key: row for key, row in weak.items() if int(row["fold"]) == fold}
    require(bool(truth), "Empty weak validation fold")
    return manifest, truth


def score_buckets(truth, predicted, group_index, target):
    """Sort once, retaining group membership and exact prediction ties."""
    buckets = defaultdict(lambda: [[], []])
    for key, row in truth.items():
        label = probability(row[target], missing=True, binary=True)
        if label is not None:
            score = probability(predicted[key][target])
            buckets[score][int(label)].append(group_index[row["group_id"]])
    return [buckets[score] for score in sorted(buckets)]


def weighted_auc(buckets, weights):
    """Weighted Mann-Whitney AUC; ties contribute half of their positive/negative pairs."""
    positives = negatives = numerator = 0.0
    for negative_groups, positive_groups in buckets:
        negative = sum(weights[index] for index in negative_groups)
        positive = sum(weights[index] for index in positive_groups)
        numerator += positive * (negatives + 0.5 * negative)
        positives += positive
        negatives += negative
    return numerator / (positives * negatives) if positives and negatives else None


def macro_difference(indices, aucs):
    return (
        sum(aucs[index][1] - aucs[index][0] for index in indices) / len(indices)
        if all(aucs[index][0] is not None and aucs[index][1] is not None for index in indices)
        else None
    )


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def interval(values, repeats):
    return {
        "valid_resamples": len(values),
        "undefined_resamples": repeats - len(values),
        "percentile_95_interval": [percentile(values, 0.025), percentile(values, 0.975)] if values else None,
    }


def publish_json(output, serialized):
    """Publish a complete file atomically without replacing another writer's output."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError("Bootstrap output already exists; preserve previous records") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bootstrap(truth, left, right, *, seed, repeats, rng="numpy_pcg64"):
    require(
        type(seed) is int and type(repeats) is int and repeats > 0, "Seed must be integer and repeats positive integer"
    )
    require(bool(truth) and truth.keys() == left.keys() == right.keys(), "Paired prediction study sets differ")
    groups = sorted({row["group_id"] for row in truth.values()})
    group_index = {name: index for index, name in enumerate(groups)}
    data = [[score_buckets(truth, predicted, group_index, target) for predicted in (left, right)] for target in TARGETS]
    unit_weights = [1] * len(groups)
    observed = [[weighted_auc(buckets, unit_weights) for buckets in pair] for pair in data]
    auxiliary = [index for index, target in enumerate(TARGETS) if target != "Synovitis"]
    samples_12, samples_11 = [], []
    target_samples = [[] for _ in TARGETS]
    require(rng in ("numpy_pcg64", "python_random"), "Unknown random generator")
    if rng == "numpy_pcg64":
        try:
            import numpy as np
        except ImportError as error:
            raise ValueError("NumPy unavailable; choose --rng python_random explicitly") from error
        generator = np.random.default_rng(seed)
        probabilities = np.full(len(groups), 1 / len(groups))
    else:
        generator = random.Random(seed)
    for _ in range(repeats):
        if rng == "numpy_pcg64":
            weights = generator.multinomial(len(groups), probabilities).tolist()
        else:
            weights = [0] * len(groups)
            for _ in range(len(groups)):
                weights[generator.randrange(len(groups))] += 1
        aucs = [[weighted_auc(buckets, weights) for buckets in pair] for pair in data]
        for indices, samples in ((range(12), samples_12), (auxiliary, samples_11)):
            value = macro_difference(indices, aucs)
            if value is not None:
                samples.append(value)
        for index, (a, b) in enumerate(aucs):
            if a is not None and b is not None:
                target_samples[index].append(b - a)
    per_target = {}
    for index, target in enumerate(TARGETS):
        labels = [probability(row[target], missing=True, binary=True) for row in truth.values()]
        binary = [value for value in labels if value is not None]
        a, b = observed[index]
        per_target[target] = {
            "left_auc": a,
            "right_auc": b,
            "difference": b - a if a is not None and b is not None else None,
            "observed": len(binary),
            "positive": int(sum(binary)),
            "negative": len(binary) - int(sum(binary)),
            "missing": len(labels) - len(binary),
            **interval(target_samples[index], repeats),
        }
    return {
        "studies": len(truth),
        "supplied_groups": len(groups),
        "observed_cells": sum(cell["observed"] for cell in per_target.values()),
        "seed": seed,
        "requested_resamples": repeats,
        "left_macro_auc_12": sum(pair[0] for pair in observed) / 12
        if all(pair[0] is not None for pair in observed)
        else None,
        "right_macro_auc_12": sum(pair[1] for pair in observed) / 12
        if all(pair[1] is not None for pair in observed)
        else None,
        "left_macro_auc_11_without_synovitis": sum(observed[index][0] for index in auxiliary) / 11
        if all(observed[index][0] is not None for index in auxiliary)
        else None,
        "right_macro_auc_11_without_synovitis": sum(observed[index][1] for index in auxiliary) / 11
        if all(observed[index][1] is not None for index in auxiliary)
        else None,
        "macro_auc_12_difference": {"point": macro_difference(range(12), observed), **interval(samples_12, repeats)},
        "macro_auc_11_without_synovitis_difference": {
            "point": macro_difference(auxiliary, observed),
            **interval(samples_11, repeats),
        },
        "per_target": per_target,
    }


def compare(
    manifest_dir,
    fold,
    left_csv,
    right_csv,
    output,
    *,
    seed=20261004,
    repeats=3000,
    left_epoch=None,
    right_epoch=None,
    rng="numpy_pcg64",
):
    output, manifest_dir = Path(output), Path(manifest_dir)
    require(not output.exists(), "Bootstrap output already exists; preserve previous records")
    for epoch in (left_epoch, right_epoch):
        require(epoch is None or (type(epoch) is int and epoch > 0), "Optional epoch must be a positive integer")
    input_paths = {
        "manifest_json": manifest_dir / "manifest.json",
        "weak_csv": manifest_dir / "weak.csv",
        "left_predictions_csv": Path(left_csv),
        "right_predictions_csv": Path(right_csv),
    }
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    manifest, truth = weak_truth(manifest_dir, fold)
    left, right = predictions(left_csv, truth.keys()), predictions(right_csv, truth.keys())
    values = bootstrap(truth, left, right, seed=seed, repeats=repeats, rng=rng)
    mask = [
        (key, [probability(row[target], missing=True) is not None for target in TARGETS])
        for key, row in sorted(truth.items())
    ]
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    require(
        input_hashes == {name: sha256(path) for name, path in input_paths.items()},
        "Bootstrap inputs changed during comparison; no output published",
    )
    report = {
        "schema_version": 1,
        "scope": "Descriptive paired comparison of fixed weak prediction CSVs; no model selection or submission",
        "validation": {
            "valid": True,
            "same_validation_study_set": True,
            "shared_observation_mask": True,
            "binary_observed_weak_truth_only": True,
            "finite_probability_scores": True,
            "supplied_groups_do_not_cross_folds": True,
            "gold_read": False,
        },
        "fold": fold,
        "left_epoch_user_supplied": left_epoch,
        "right_epoch_user_supplied": right_epoch,
        "difference_direction": "right minus left",
        "inputs_sha256": {
            **input_hashes,
            "observation_mask": hashlib.sha256(
                json.dumps(mask, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "manifest_inputs_sha256": manifest["inputs_sha256"],
        "algorithm": {
            "resampling_unit": "supplied group_id; retain all member studies with the same weight",
            "group_order": "lexicographically sorted supplied group_id",
            "draws_per_resample": values["supplied_groups"],
            "replacement": True,
            "pairing": "same group multiplicities for left and right predictions in every draw",
            "rng": "NumPy default_rng PCG64; one multinomial(K, full(K, 1/K)) draw per resample"
            if rng == "numpy_pcg64"
            else "Python random.Random (Mersenne Twister); randrange draws; differs from the prior strategy-audit RNG",
            "auc": "weighted Mann-Whitney ranks using original score order; equal scores contribute 0.5",
            "interval": "percentile 2.5/97.5; linear interpolation at (n-1)*fraction; defined draws only",
            "macro_auc_12_denominator": 12,
            "auxiliary_denominator": 11,
            "auxiliary_excluded_target": "Synovitis",
            "undefined_policy": "macro undefined if any required target lacks either class; count and exclude that draw",
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy_version_installed": numpy_version,
            "numpy_used": rng == "numpy_pcg64",
        },
        **values,
        "gold_macro_auc_12": None,
        "public_lb": None,
        "limitations": [
            "Patient independence is unverified; supplied group independence is assumed for this bootstrap",
            "Weak labels are report-derived; these intervals do not establish image-label or Kaggle performance",
            "Checkpoint, epoch and configuration selection on the same fold are outside interval coverage",
            "Intervals use defined draws only; an interval conditional on definition is not evidence that all draws are valid",
            "Gold is not read; gold exclusion and report linkage rely on the separate fixed-manifest audit",
            "Optional epoch numbers are supplied by the caller; checkpoint/index selection is audited separately",
            "Input hashes are checked before and after computation; this is not continuous filesystem monitoring and "
            "does not detect an input changed and restored between checks",
        ],
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    publish_json(output, serialized)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--left-epoch", type=int)
    parser.add_argument("--right-epoch", type=int)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument("--rng", choices=("numpy_pcg64", "python_random"), default="numpy_pcg64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(
            args.manifest_dir,
            args.fold,
            args.left_predictions,
            args.right_predictions,
            args.output,
            seed=args.seed,
            repeats=args.repeats,
            left_epoch=args.left_epoch,
            right_epoch=args.right_epoch,
            rng=args.rng,
        )
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "valid": True,
                "macro_auc_12_difference": report["macro_auc_12_difference"],
                "macro_auc_11_without_synovitis_difference": report["macro_auc_11_without_synovitis_difference"],
            }
        )
    )
