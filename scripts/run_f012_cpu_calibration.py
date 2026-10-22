"""Fixed CPU F012 diagnostic, preserving the independent existing GPU job."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cpu_teacher_screen import normalization, rank_scores, require, ridge_fit
from isotonic_teacher import apply, fit
from run_f011_cpu_teacher_screen import ID, TARGETS, missing, pin, save

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'artifacts/research/20261020-F012-CPU-train-calibration-prospective-v1/policy.json'
POLICY_SHA = 'f51a0288c9f138860c44cbb764add7dfee8c79fcf09f7169de95063c26d092f6'


def main():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np

    sys.path.insert(0, str(ROOT / 'src'))
    from rsna_knee.metrics import evaluate_rows
    require(pin(POLICY)['sha256'] == POLICY_SHA, 'Frozen calibration policy differs')
    policy = json.loads(POLICY.read_bytes())
    inputs = {key: ROOT / value['path'] for key, value in policy['inputs'].items()}
    for key, path in inputs.items():
        require(pin(path) == {'bytes': policy['inputs'][key]['bytes'], 'sha256': policy['inputs'][key]['SHA']}, 'Frozen input differs')
    out = ROOT / 'artifacts/runs/f012-cpu-train-calibration-fold0-20261020-v1'
    require(not out.exists(), 'Fresh run directory required')
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'scripts/isotonic_teacher.py', ROOT/'scripts/cpu_teacher_screen.py',
               ROOT/'scripts/run_f011_cpu_teacher_screen.py', ROOT/'src/rsna_knee/metrics.py']
    save(out/'start.json', {'at_UTC': datetime.now(timezone.utc).isoformat(), 'policy_SHA': POLICY_SHA,
                          'source_pins': {p.as_posix(): pin(p) for p in sources}, 'python': sys.version,
                          'numpy': np.__version__, 'platform': platform.platform(), 'threads': 1,
                          'Torch_GPU_MRI_Gold_API_calls': 0, 'automatic_followup': False})
    (out/'executed-sources').mkdir(exist_ok=False)
    for path in sources:
        (out/'executed-sources'/path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    parent = json.loads(inputs['F011_completion'].read_bytes())
    ids = json.loads(inputs['UIDs'].read_bytes())
    require(len(ids) == len(set(ids)) == 4207 and parent['status'] == 'complete', 'Exact complete parent cohort')
    def mapping(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        result = {r[ID]: r for r in rows}
        require(len(result) == len(rows), 'Duplicate source UID')
        return result
    weak, teacher = mapping(inputs['weak']), mapping(inputs['teacher'])
    require(set(ids) == set(weak) and set(ids).issubset(teacher), 'Exact teacher/weak cohort')
    groups = {}
    for row in weak.values():
        require(groups.setdefault(row['group_id'], row['fold']) == row['fold'], 'Supplied/report group crosses folds')
    folds = np.array([int(weak[uid]['fold']) for uid in ids])
    train, valid = folds != 0, folds == 0
    require(int(train.sum()) == 3386 and int(valid.sum()) == 821, 'Original fold0 split')
    old = np.array([[np.nan if missing(weak[uid][t]) else float(weak[uid][t]) for t in TARGETS] for uid in ids])
    # Numeric teacher scores outside the4207 cohort are never converted.
    soft = np.array([[np.nan if missing(teacher[uid][t]) else float(teacher[uid][t]) for t in TARGETS] for uid in ids])
    require(np.all(np.isnan(old) | np.isin(old, [0, 1])) and np.all(np.isnan(soft) | np.isfinite(soft) & (soft >= 0) & (soft <= 1)), 'Binary/missing weak and probability/missing teacher')
    matrix = np.load(inputs['features'], allow_pickle=False)
    require(matrix.shape == (4207, 1540) and np.isfinite(matrix).all(), 'Frozen feature dimensions')
    mean, scale = normalization(matrix[train])
    xtrain, xvalid = (matrix[train]-mean)/scale, (matrix[valid]-mean)/scale
    control_path = inputs['F011_completion'].parent/'C-raw-scores.npy'
    require(pin(control_path) == parent['output_pins'][control_path.name], 'Fixed C predictions differ')
    control = np.load(control_path, allow_pickle=False)
    result = np.empty((821, 12), np.float64)
    betas, biases, curves = np.empty((1540, 12), np.float64), np.empty(12), {}
    max_c_error = 0.
    for col, target in enumerate(TARGETS):
        common = np.isfinite(old[train, col]) & np.isfinite(soft[train, col])
        curve = fit(soft[train, col][common].tolist(), old[train, col][common].tolist())
        curves[target] = curve
        finite = np.isfinite(soft[train, col])
        targets = soft[train, col][finite]
        calibrated = apply(curve, targets)
        beta, bias = ridge_fit(xtrain[finite], np.stack([targets, calibrated], axis=1))
        predictions = xvalid @ beta + bias
        max_c_error = max(max_c_error, float(np.max(np.abs(predictions[:, 0]-control[:, col]))))
        result[:, col], betas[:, col], biases[col] = predictions[:, 1], beta[:, 1], bias[1]
        print('F012_FIT', target, 'observed_calibration', int(common.sum()), 'finite_support', int(finite.sum()), flush=True)
    require(max_c_error <= 1e-9 and np.isfinite(result).all(), 'Identical C control reproduction and finite D required')
    valid_ids = [uid for uid, flag in zip(ids, valid) if flag]
    ranked = np.array([rank_scores(result[:, j].tolist()) for j in range(12)]).T
    by = {uid: {t: repr(float(ranked[i, j])) for j, t in enumerate(TARGETS)} for i, uid in enumerate(valid_ids)}
    metrics = evaluate_rows({uid: weak[uid] for uid in valid_ids}, by)
    primary = sum(metrics['per_target'][t]['auc'] for t in TARGETS if t not in ('MCL', 'Synovitis'))/10
    require(math.isfinite(primary), 'Defined fixed10 metric')
    a, c = [parent['fixed10_weak_mean_AUC'][arm] for arm in 'AC']
    require('torch' not in sys.modules, 'No Torch import allowed')
    with (out/'D-weak-rank.csv').open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[ID, *TARGETS])
        writer.writeheader()
        writer.writerows({ID: uid, **by[uid]} for uid in valid_ids)
    np.save(out/'D-raw-scores.npy', result, allow_pickle=False)
    np.savez(out/'D-head.npz', beta=betas, bias=biases, mean=mean, scale=scale)
    save(out/'train-only-calibration-curves.json', curves)
    for key, path in inputs.items():
        require(pin(path)['sha256'] == policy['inputs'][key]['SHA'], 'Input changed during fit')
    completed = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'complete': True, 'policy_SHA': POLICY_SHA,
                 'fixed10_weak_AUC': {'A_prior': a, 'C_prior': c, 'D': primary}, 'D_minus_A': primary-a, 'D_minus_C': primary-c,
                 'D_promising_for_manual_fold1_review': primary > a and primary > c, 'metrics': metrics,
                 'C_reproduction_max_abs': max_c_error, 'seconds': time.monotonic()-started,
                 'own_Gold_fit_select': 0, 'GPU_Torch_MRI_calls': 0, 'auto_followup_Save_Submit': 0,
                 'limits': 'Repeated weakholdout; upstreamJEV knownGold-informed; no Public effect or patient independence claim.',
                 'output_pins': {p.name: pin(p) for p in out.iterdir() if p.is_file()}}
    save(out/'completion.json', completed)
    print('F012_COMPLETE', json.dumps({k: completed[k] for k in ('fixed10_weak_AUC', 'D_minus_A', 'D_minus_C', 'D_promising_for_manual_fold1_review', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
