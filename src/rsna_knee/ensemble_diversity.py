"""Same-support ensemble diagnostics using only CSV values and the standard library.

Pair transition accounting extends the frozen 20261012 pair-utility analysis.
No threshold search, weight fit, model loading or cross-fold pooling is performed.
"""

from __future__ import annotations

import math
import struct

from .metrics import binary_auc


def correlation(left, right):
    """Pearson correlation, explicitly undefined for constant or short vectors."""
    if len(left) != len(right) or not left:
        raise ValueError("Equal nonempty vectors required")
    if not all(math.isfinite(x) for x in (*left, *right)):
        raise ValueError("Finite vectors required")
    a, b = math.fsum(left) / len(left), math.fsum(right) / len(right)
    x, y = [v - a for v in left], [v - b for v in right]
    denominator = math.sqrt(math.fsum(v * v for v in x) * math.fsum(v * v for v in y))
    return max(-1.0, min(1.0, math.fsum(p * q for p, q in zip(x, y)) / denominator)) if denominator else None


def ranks(values):
    """Average ranks for exact ties, retaining input order."""
    result, ordered, start = [0.0] * len(values), sorted(range(len(values)), key=values.__getitem__), 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fixed_mean(left, right):
    return f32(f32(f32(left) + f32(right)) * 0.5)


def pair_transitions(labels, left, right):
    """3x3 incorrect/tie/correct transition table and exact half-unit AUC delta."""
    if len(labels) != len(left) or len(labels) != len(right) or not labels:
        raise ValueError("Equal nonempty supports required")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Binary support required")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (*left, *right)):
        raise ValueError("Finite probabilities required")
    positives = [i for i, y in enumerate(labels) if y == 1]
    negatives = [i for i, y in enumerate(labels) if y == 0]
    cells = [[0] * 3 for _ in range(3)]
    for positive in positives:
        for negative in negatives:
            a = (left[positive] > left[negative]) - (left[positive] < left[negative]) + 1
            b = (right[positive] > right[negative]) - (right[positive] < right[negative]) + 1
            cells[a][b] += 1
    total = len(positives) * len(negatives)
    net = sum((j - i) * cells[i][j] for i in range(3) for j in range(3))
    return {
        "pairs": total,
        "states": ["incorrect", "tie", "correct"],
        "transition_counts": cells,
        "both_incorrect": cells[0][0],
        "left_incorrect_right_correct": cells[0][2],
        "left_correct_right_incorrect": cells[2][0],
        "improved": sum(cells[i][j] for i in range(3) for j in range(i + 1, 3)),
        "worsened": sum(cells[i][j] for i in range(3) for j in range(i)),
        "net_half_units": net,
        "auc_delta": net / (2 * total) if total else None,
    }


def compare_target(labels, left, right):
    """Binary-only paired diagnostics. Missing/soft cells must be excluded upstream."""
    if len(labels) != len(left) or len(labels) != len(right):
        raise ValueError("Identical support required")
    if not labels:
        return {"binary_observed": 0, "positive": 0, "negative": 0, "auc": None,
                "rank_correlation": None, "residual_correlation": None, "pair_transitions": None,
                "mean50_transitions_from_left": None, "mean50_transitions_from_right": None}
    transitions = pair_transitions(labels, left, right)
    mean = [fixed_mean(a, b) for a, b in zip(left, right)]
    au = {"left": binary_auc(labels, left), "right": binary_auc(labels, right),
          "fixed_FP32_probability_mean50": binary_auc(labels, mean)}
    return {
        "binary_observed": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
        "auc": au,
        "rank_correlation": correlation(ranks(left), ranks(right)),
        "residual_correlation": correlation([p - y for p, y in zip(left, labels)],
                                            [p - y for p, y in zip(right, labels)]),
        "pair_transitions": transitions,
        "mean50_transitions_from_left": pair_transitions(labels, left, mean),
        "mean50_transitions_from_right": pair_transitions(labels, right, mean),
    }
