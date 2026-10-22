"""CPU-only helpers for fixed-feature teacher diagnostics; no Torch/CUDA/network."""
from __future__ import annotations

import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pool_planes(features, mask, planes):
    import numpy as np
    require(features.ndim == 3 and features.shape[-1] == 384 and mask.shape == features.shape[:2]
            and planes.shape == (features.shape[0],), 'Exact frozen feature dimensions required')
    require(mask.dtype == np.bool_ and np.issubdtype(planes.dtype, np.integer)
            and np.all((0 <= planes) & (planes <= 3)), 'Exact plane and image-mask contracts required')
    require(mask.any() and np.isfinite(features[mask]).all(), 'At least one finite observed image window required')
    output = np.zeros(1540, np.float64)
    for plane in range(4):
        observed = mask & (planes[:, None] == plane)
        if observed.any():
            output[plane * 384:(plane + 1) * 384] = features[observed].mean(axis=0, dtype=np.float64)
            output[1536 + plane] = 1
    return output


def normalization(train):
    import numpy as np
    require(train.ndim == 2 and len(train) > 1 and np.isfinite(train).all(), 'Finite training feature matrix required')
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-8] = 1
    return mean, scale


def ridge_fit(x, y, alpha=.10):
    """Mean squared error + alpha L2, unregularized intercept; no missing fill."""
    import numpy as np
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    require(x.ndim == y.ndim == 2 and len(x) == len(y) and len(x) > 1
            and np.isfinite(x).all() and np.isfinite(y).all() and 0 < alpha < 1,
            'Finite observed training rows and fixed positive regularization required')
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    centered = x - xmean
    gram = centered.T @ centered / len(x)
    gram.flat[::gram.shape[0] + 1] += alpha
    rhs = centered.T @ (y - ymean) / len(x)
    beta = np.linalg.solve(gram, rhs)
    bias = ymean - xmean @ beta
    require(np.isfinite(beta).all() and np.isfinite(bias).all(), 'Finite closed-form head required')
    return beta, bias


def rank_scores(values):
    require(values and all(math.isfinite(float(v)) for v in values), 'Finite nonempty linear scores required')
    ordered = sorted((float(value), i) for i, value in enumerate(values))
    result, start, n = [None] * len(values), 0, len(values)
    while start < n:
        end = start + 1
        while end < n and ordered[end][0] == ordered[start][0]:
            end += 1
        for _, i in ordered[start:end]:
            result[i] = (start + 1 + end) / (2 * n)
        start = end
    return result
