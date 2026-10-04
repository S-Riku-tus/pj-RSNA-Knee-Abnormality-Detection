"""Probability diagnostics without implicit binarization or model selection."""

from __future__ import annotations

import math

from .contracts import probability
from .metrics import evaluate_rows


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "min": min(values),
        "max": max(values),
    }


def prediction_diagnostics(truth, predictions, loss_details=None):
    result = evaluate_rows(truth, predictions)
    for target, metrics in result["per_target"].items():
        all_scores, positive_scores, negative_scores = [], [], []
        observed = 0
        for key, row in truth.items():
            score = probability(predictions[key][target])
            label = probability(row.get(target, ""), missing=True)
            all_scores.append(score)
            observed += label is not None
            if label == 1:
                positive_scores.append(score)
            elif label == 0:
                negative_scores.append(score)
        metrics.update(
            observed=observed,
            negative=len(negative_scores),
            missing=len(truth) - observed,
            predictions=distribution(all_scores),
            positive_predictions=distribution(positive_scores),
            negative_predictions=distribution(negative_scores),
        )
        if loss_details is not None:
            metrics["masked_bce"] = loss_details["per_target"][target]["masked_bce"]
    if loss_details is not None:
        result["cell_mean_bce"] = loss_details["cell_mean_bce"]
        result["observed_target_mean_bce"] = loss_details["observed_target_mean_bce"]
        result["observed_cells"] = loss_details["observed_cells"]
    return result
