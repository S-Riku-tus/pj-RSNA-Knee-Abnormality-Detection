"""AUC with explicit undefined-class handling; no hidden change to the macro denominator."""

from __future__ import annotations

from .contracts import TARGETS, probability


def binary_auc(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC needs equal, nonempty arrays")
    pairs = sorted((probability(str(s)), probability(str(y), binary=True)) for y, s in zip(labels, scores))
    positives = sum(y for _, y in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_rows(truth, predictions):
    if truth.keys() != predictions.keys() or not truth:
        raise ValueError("Evaluation IDs must match and be nonempty")
    result = {}
    for target in TARGETS:
        labels, scores, soft = [], [], 0
        for key, row in truth.items():
            p = probability(predictions[key][target])
            y = probability(row[target], missing=True)
            if y in (0, 1):
                labels.append(y)
                scores.append(p)
            elif y is not None:
                soft += 1
        auc = binary_auc(labels, scores) if labels else None
        result[target] = {"auc": auc, "binary_observed": len(labels), "positive": sum(labels), "soft_excluded": soft}
    values = [v["auc"] for v in result.values() if v["auc"] is not None]
    return {
        "per_target": result,
        "macro_auc_12": sum(values) / 12 if len(values) == 12 else None,
        "available_class_macro_auc": sum(values) / len(values) if values else None,
        "defined_classes": len(values),
        "note": "Soft targets are excluded from ROC-AUC; use masked BCE for weak-label validation.",
    }
