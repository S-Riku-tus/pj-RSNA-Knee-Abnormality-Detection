"""Train-only monotone teacher calibration; finite binary observations required."""
from __future__ import annotations

import math


def fit(scores, labels):
    def require(ok):
        if not ok:
            raise ValueError('Finite paired binary observations required')
    require(len(scores) == len(labels) and len(scores) > 1)
    pairs = sorted((float(x), float(y)) for x, y in zip(scores, labels))
    require(all(math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and y in (0., 1.) for x, y in pairs))
    xs, counts, totals = [], [], []
    for x, y in pairs:
        if xs and x == xs[-1]:
            counts[-1] += 1
            totals[-1] += y
        else:
            xs.append(x)
            counts.append(1)
            totals.append(y)
    blocks = []
    for i, (n, total) in enumerate(zip(counts, totals)):
        blocks.append([i, i+1, n, total])
        while len(blocks) > 1 and blocks[-2][3] * blocks[-1][2] > blocks[-1][3] * blocks[-2][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    values = [None] * len(xs)
    for start, stop, n, total in blocks:
        values[start:stop] = [total/n] * (stop-start)
    return {'scores': xs, 'probabilities': values, 'counts': counts, 'positive_totals': totals,
            'n_observed_training': len(pairs), 'n_blocks': len(blocks)}


def apply(curve, scores):
    import numpy as np
    scores = np.asarray(scores, np.float64)
    if not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError('Finite teacher probabilities required; missing remains masked')
    return np.clip(np.interp(scores, curve['scores'], curve['probabilities']), .001, .999)
