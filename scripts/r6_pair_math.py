"""Label-free reconstruction of frozen R6 TTA/postprocessor counterfactuals."""
from __future__ import annotations

import math


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sigmoid(value):
    require(math.isfinite(float(value)), 'Finite logit required')
    value = float(value)
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def state_permutation(labels, layout, class_perm, bin_perm):
    """Equivalent flat mapping to frozen worker's pre-softmax class/bin swap."""
    blocks = {row['class']: row for row in layout['columns'] if 'categories' in row}
    bins = [row for row in layout['columns'] if 'binary' in row]
    width = max([row['stop'] for row in blocks.values()] + [row['column'] + 1 for row in bins])
    result = list(range(width))
    for i, label in enumerate(labels):
        left, right = blocks[label], blocks[labels[class_perm[i]]]
        require(left['categories'] == right['categories'], 'Swapped state category order must match')
        require(left['stop'] - left['start'] == right['stop'] - right['start'], 'State widths match')
        result[left['start']:left['stop']] = list(range(right['start'], right['stop']))
    require(len(bins) == len(bin_perm), 'Binary permutation width differs')
    for i, row in enumerate(bins):
        result[row['column']] = bins[bin_perm[i]]['column']
    require(sorted(result) == list(range(width)), 'Full bijective state mapping required')
    mcl = [i for row in layout['columns'] if row['class'] == 'MCL'
           for i in (range(row['start'], row['stop']) if 'categories' in row else [row['column']])]
    return result, mcl


def mix(record, meta, weight):
    import numpy as np
    require(weight in (0., .25, .50), 'Preregistered mirror weights only')
    require(all(len(record[k]) == 1 for k in ('z_plain', 'z_mirror_raw', 'state_plain', 'state_mirror_raw')),
            'Exactly one declared heldout model required')
    plain, mirrored = [np.asarray(record[k][0], np.float32) for k in ('z_plain', 'z_mirror_raw')]
    sp, sm = [np.asarray(record[k][0], np.float32) for k in ('state_plain', 'state_mirror_raw')]
    require(plain.shape == mirrored.shape == (len(meta['labels']),) and sp.shape == sm.shape == (len(meta['state_perm']),), 'Exact output shapes required')
    require(all(np.isfinite(v).all() for v in (plain, mirrored, sp, sm)), 'Finite model outputs required')
    require(all(np.all((v >= 0) & (v <= 1)) for v in (sp, sm)), 'State probabilities required')
    if weight == .5:
        z = np.float32(.5) * (plain + mirrored[meta['class_perm']])
        state = np.float32(.5) * (sp + sm[meta['state_perm']])
    else:
        z = np.float32(1 - weight) * plain + np.float32(weight) * mirrored[meta['class_perm']]
        state = np.float32(1 - weight) * sp + np.float32(weight) * sm[meta['state_perm']]
    z[meta['mcl_index']] = plain[meta['mcl_index']]
    state[meta['mcl_state_cols']] = sp[meta['mcl_state_cols']]
    return z.astype(np.float64), state.astype(np.float64)


def postprocess(z, state, fluid, meta, calibrate=True):
    import numpy as np
    probabilities = [sigmoid(v) for v in z]
    if not calibrate:
        return probabilities
    for target, co in meta['fluid'].items():
        index = meta['labels'].index(target)
        value = fluid.get(co['volume_column'])
        absent = value is None or not math.isfinite(float(value)) or float(value) < 0
        lv = 0. if absent else math.log1p(float(value))
        probabilities[index] = sigmoid(co['b0'] + co['b_z'] * z[index] + co['b_lv'] * lv + co['b_miss'] * int(absent))
    for target, co in meta['state'].items():
        index = meta['labels'].index(target)
        values = []
        for feature, column in zip(co['features'], co['columns']):
            if feature == 'z':
                values.append(float(z[index]))
            elif feature in ('lv', 'miss'):
                value = fluid.get(co['volume_column'])
                absent = value is None or not math.isfinite(float(value)) or float(value) < 0
                values.append((0. if absent else math.log1p(float(value))) if feature == 'lv' else float(absent))
            else:
                require(type(column) is int and 0 <= column < len(state), 'Valid state index required')
                values.append(float(state[column]))
        probabilities[index] = sigmoid(co['b0'] + float(np.dot(np.asarray(values, np.float64), np.asarray(co['coef'], np.float64))))
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities), 'Finite final probabilities required')
    return probabilities


def replay(record, fluid, meta, weight=.50, calibrate=True):
    z, state = mix(record, meta, weight)
    return postprocess(z, state, fluid, meta, calibrate)
