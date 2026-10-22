"""Parent-preserving fixed20 CSV arithmetic for a future innerC candidate.

No parent selection, fitting or deployment qualification is performed here.
An actual controller must bind the chosen parent, model, UID order and runtime.
Only feature-success cells in supported fixed10 columns can be replaced.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
FIXED10 = tuple(name for name in TARGETS if name not in ("MCL", "Synovitis"))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def probability(value):
    require(type(value) in (str, int, float) and not isinstance(value, bool), "Probability cell must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid probability cell") from error
    require(math.isfinite(number) and 0 <= number <= 1, "Finite probability in[0,1] required")
    return number


def rank_fp32(values):
    """Average-tie percentile ranks; quantize probabilities before identifying ties."""
    values = np.asarray(values, dtype=np.float32)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Finite nonempty rank vector required")
    order = sorted(range(len(values)), key=lambda i: float(values[i]))
    out = np.empty(len(values), dtype=np.float32)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = np.float32((start + 1 + end) / (2 * len(values)))
        out[order[start:end]] = rank
        start = end
    return out


def overlay_rows(parent_rows, head_by_uid, *, supported_targets):
    """Return cloned rows, changing only active cells; original strings survive.

    ``head_by_uid`` contains exactly the successful feature UID subset with12
    ordered probabilities. The controller must never insert zero fake features.
    ``supported_targets`` comes from the bound model's qualified-head metadata.
    Neither this arithmetic function nor its receipt grants candidate approval.
    """
    require(isinstance(parent_rows, (list, tuple)) and len(parent_rows) > 0
            and isinstance(head_by_uid, Mapping), "Nonempty parent rows and explicit success mapping required")
    supported = set(supported_targets)
    require(supported <= set(TARGETS), "Unknown supported target")
    fields = {ID, *TARGETS}
    uids, parents = [], {}
    for row in parent_rows:
        require(isinstance(row, Mapping) and set(row) == fields, "Parent needs exactly ordered UID and12 target columns")
        uid = row[ID]
        require(isinstance(uid, str) and bool(uid) and uid.strip() == uid and uid not in parents,
                "Unique nonempty aligned parent UID required")
        parents[uid] = [probability(row[name]) for name in TARGETS]
        uids.append(uid)
    require(set(head_by_uid) <= set(uids), "Feature success UID absent from parent")
    selected = [i for i, uid in enumerate(uids) if uid in head_by_uid]
    for uid in head_by_uid:
        values = head_by_uid[uid]
        require(isinstance(values, (list, tuple)) and len(values) == 12, "Twelve ordered head probabilities required")
        for value in values:
            probability(value)
    rows = [dict(row) for row in parent_rows]
    active = np.zeros((len(rows), 12), dtype=bool)
    details = {}
    for target_index, name in enumerate(TARGETS):
        info = {"N_valid": len(selected), "active": False}
        if name not in FIXED10 or name not in supported:
            details[name] = {**info, "reason": "unsupported_or_excluded_auxiliary"}
            continue
        values = np.asarray([probability(head_by_uid[uids[i]][target_index]) for i in selected], dtype=np.float32)
        if len(values) < 2 or np.all(values == values[0]):
            details[name] = {**info, "reason": "N_valid_lt2_or_constant_FP32"}
            continue
        ranks = rank_fp32(values)
        parent = np.asarray([parents[uids[i]][target_index] for i in selected], dtype=np.float32)
        left = np.multiply(np.float32(.8), parent, dtype=np.float32)
        right = np.multiply(np.float32(.2), ranks, dtype=np.float32)
        blend = np.add(left, right, dtype=np.float32)
        require(np.isfinite(blend).all() and np.all((blend >= 0) & (blend <= 1)), "Invalid overlay probabilities")
        for i, value in zip(selected, blend):
            rows[i][name] = format(float(value), ".9g")
            active[i, target_index] = True
        details[name] = {**info, "active": True, "distinct_FP32": len(set(float(x) for x in values))}
    for i, original in enumerate(parent_rows):
        require(rows[i][ID] == original[ID], "UID/order changed")
        for j, name in enumerate(TARGETS):
            if not active[i, j]:
                require(rows[i][name] == original[name] and type(rows[i][name]) is type(original[name]),
                        "Inactive parent cell changed")
    receipt = {"schema_version": "orthodiff_innerC_fixed20_parent_preserving_arithmetic_v1",
               "rows": len(rows), "feature_successes": len(selected), "feature_failures": len(rows) - len(selected),
               "active_cells": int(active.sum()), "per_target": details,
               "formula": "FP32(.8)*FP32(parent_success) + FP32(.2)*FP32(average_tie_rank(head_FP32)); separate FP32 products/add",
               "parent_or_output_reranked": False, "inactive_original_cells_preserved": True,
               "public_submission_contract_qualified": False}
    return rows, receipt
