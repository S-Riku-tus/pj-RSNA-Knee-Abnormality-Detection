"""Fixed F011 CPU teacher screen on cached embeddings. Never trains an image encoder."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cpu_teacher_screen import normalization, pool_planes, rank_scores, require, ridge_fit

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'artifacts/research/20261020-F011-CPU-teacher-screen-policy-v1/policy.json'
POLICY_SHA = 'e0ce21d5bb6eb4ccd967d570dac3a2329c20a2d36fa9a984c740cacb4af139e9'
ID = 'StudyInstanceUID'
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def save(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def missing(value):
    return value.strip().lower() in ('', 'na', 'nan', 'null')


def main():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    sys.path.insert(0, str(ROOT / 'src'))
    from rsna_knee.metrics import evaluate_rows
    require(pin(POLICY)['sha256'] == POLICY_SHA, 'Frozen prefit policy differs')
    policy = json.loads(POLICY.read_text(encoding='utf8'))
    out = ROOT / policy['run']
    require(not out.exists(), 'New F011 run required; no overwrite')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = {key: ROOT / policy[key] for key in ('feature_export', 'JEV')}
    inputs['weak'] = ROOT / policy['weak_manifest'] / 'weak.csv'
    for key in ('feature_export', 'JEV', 'weak'):
        require(pin(inputs[key])['sha256'] == policy[key + '_SHA'], 'Pinned input differs: ' + key)
    source = {name: pin(Path(__file__).parent / name) for name in ['cpu_teacher_screen.py', Path(__file__).name]}
    save(out / 'start.json', {'at_UTC': datetime.now(timezone.utc).isoformat(), 'status': 'started',
         'policy_SHA': POLICY_SHA, 'source': source, 'input_pins': {k: pin(p) for k, p in inputs.items()},
         'Python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
         'thread_env': {k: os.environ[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
         'Torch_CUDA_encoder_training': 0, 'own_Gold_fit_select': 0})
    try:
        weak, groups = {}, {}
        with inputs['weak'].open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == [ID, *TARGETS, 'source', 'group_id', 'fold'], 'Exact weak columns required')
            for row in reader:
                require(row[ID] not in weak and row['source'] == 'report_weak' and row['group_id'], 'Unique report-weak rows required')
                require(groups.setdefault(row['group_id'], row['fold']) == row['fold'], 'Report/suppliedgroup crosses folds')
                require(all(missing(row[t]) or row[t] in ('0', '0.0', '1', '1.0') for t in TARGETS), 'Binary or masked weak teacher only')
                weak[row[ID]] = row
        require(len(weak) == 4207, 'Exact4207 original nonGold weak cohort')
        ids = sorted(weak)
        teacher = {}
        skipped = 0
        with inputs['JEV'].open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == [ID, *TARGETS], 'Exact JEV label order')
            for row in reader:
                if row[ID] not in weak:
                    skipped += 1
                    continue
                require(row[ID] not in teacher, 'No duplicate JEV UID')
                teacher[row[ID]] = [np.nan if missing(row[t]) else float(row[t]) for t in TARGETS]
                require(all(np.isnan(v) or np.isfinite(v) and 0 <= v <= 1 for v in teacher[row[ID]]), 'Soft score or missing only')
        require(set(teacher) == set(ids), 'JEV covers exact existing cohort')
        export = json.loads(inputs['feature_export'].read_text())
        require(export['complete'] and not export['limited'] and export['requested'] == 4207
                and export['fingerprint'] == policy['feature_fingerprint'], 'Complete pinned generic feature export required')
        files = export['files']
        require(set(files) == {uid + '.npz' for uid in ids}, 'Exact feature UID set required')
        export.pop('coverage', None)
        matrix = np.empty((4207, 1540), np.float64)
        folder = inputs['feature_export'].parent
        for i, uid in enumerate(ids):
            path = folder / (uid + '.npz')
            require(not path.is_symlink() and pin(path) == files[path.name], 'Frozen feature bytes differ')
            with np.load(path, allow_pickle=False) as data:
                require(str(data['fingerprint'].item()) == policy['feature_fingerprint'], 'Feature encoder/preprocessing fingerprint differs')
                matrix[i] = pool_planes(data['features'], data['mask'], data['planes'])
            if (i + 1) % 1000 == 0:
                print('F011_POOL', i + 1, '/', len(ids), flush=True)
        del export, files
        folds = np.array([int(weak[uid]['fold']) for uid in ids])
        train, valid = folds != 0, folds == 0
        require(int(train.sum()) == 3386 and int(valid.sum()) == 821, 'Exact fixed fold0 split')
        old = np.array([[np.nan if missing(weak[uid][t]) else float(weak[uid][t]) for t in TARGETS] for uid in ids])
        soft = np.array([teacher[uid] for uid in ids], np.float64)
        mean, scale = normalization(matrix[train])
        xtrain = (matrix[train] - mean) / scale
        xvalid = (matrix[valid] - mean) / scale
        np.save(out / 'pooled-features.npy', matrix, allow_pickle=False)
        save(out / 'pooled-UIDs.json', ids)
        del matrix, teacher
        gc.collect()
        predictions = {arm: np.empty((821, 12), np.float64) for arm in 'ABC'}
        betas = {arm: np.empty((1540, 12), np.float64) for arm in 'ABC'}
        biases = {arm: np.empty(12, np.float64) for arm in 'ABC'}
        supports = {}
        for col, target in enumerate(TARGETS):
            observed = np.isfinite(old[train, col]) & np.isfinite(soft[train, col])
            require(observed.sum() > 1, 'Enough observed common target supervision required')
            y = np.stack([old[train, col][observed], soft[train, col][observed]], axis=1)
            beta, bias = ridge_fit(xtrain[observed], y)
            common_pred = xvalid @ beta + bias
            for i, arm in enumerate('AB'):
                predictions[arm][:, col], betas[arm][:, col], biases[arm][col] = common_pred[:, i], beta[:, i], bias[i]
            full = np.isfinite(soft[train, col])
            beta, bias = ridge_fit(xtrain[full], soft[train, col][full])
            predictions['C'][:, col] = (xvalid @ beta + bias).ravel()
            betas['C'][:, col], biases['C'][col] = beta.ravel(), bias[0]
            supports[target] = {'A_B_common_train': int(observed.sum()), 'C_finite_soft_train': int(full.sum()),
                                'validation_observed_old': int(np.isfinite(old[valid, col]).sum())}
            print('F011_FIT', target, json.dumps(supports[target]), flush=True)
        valid_ids = [uid for uid, flag in zip(ids, valid) if flag]
        truth = {uid: weak[uid] for uid in valid_ids}
        metrics = {}
        for arm in 'ABC':
            require(np.isfinite(predictions[arm]).all(), 'Finite head predictions required')
            ranked = np.array([rank_scores(predictions[arm][:, i].tolist()) for i in range(12)]).T
            by = {uid: {t: repr(float(ranked[i, j])) for j, t in enumerate(TARGETS)} for i, uid in enumerate(valid_ids)}
            metrics[arm] = evaluate_rows(truth, by)
            with (out / (arm + '-weak-rank.csv')).open('x', encoding='utf8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[ID, *TARGETS])
                writer.writeheader()
                writer.writerows({ID: uid, **by[uid]} for uid in valid_ids)
            np.savez(out / (arm + '-head.npz'), beta=betas[arm], bias=biases[arm], mean=mean, scale=scale)
            np.save(out / (arm + '-raw-scores.npy'), predictions[arm], allow_pickle=False)
        primary = {arm: sum(metrics[arm]['per_target'][t]['auc'] for t in TARGETS if t not in ('MCL', 'Synovitis')) / 10 for arm in 'ABC'}
        require(all(math.isfinite(v) for v in primary.values()), 'Defined fixed10 primary values required')
        require('torch' not in sys.modules, 'No Torch import permitted')
        for key, path in inputs.items():
            require(pin(path)['sha256'] == policy[key + '_SHA'], 'Input changed during CPU screen')
        result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'policy_SHA': POLICY_SHA,
                  'fixed10_weak_mean_AUC': primary, 'B_minus_A_values_only': primary['B'] - primary['A'],
                  'C_minus_B_support_only': primary['C'] - primary['B'], 'C_minus_A_total': primary['C'] - primary['A'],
                  'C_promising_for_manual_fixed_fold1_review': primary['C'] > primary['A'] and primary['C'] > primary['B'],
                  'train_UIDs': 3386, 'valid_UIDs': 821, 'targets': list(TARGETS), 'metrics': metrics, 'supports': supports,
                  'skipped_JEV_rows_before_numeric_conversion': skipped, 'seconds': time.monotonic() - started,
                  'feature_dim': 1540, 'head': 'ridge alpha.10/intercept, train-only standardization; all fixed',
                  'own_Gold_fit_select': 0, 'GPU_Torch_encoder_training': 0, 'automatic_fold1_or_Submit_or_upload': False,
                  'public_gain_established': False, 'upstream_JEV_Gold_selection': 'KNOWN and disclosed; generator notfullyclosed',
                  'limits': 'Repeated weakholdout, no patient-independence or Goldfree-upstream claim; not a Public effect estimate.',
                  'output_pins': {p.name: pin(p) for p in out.iterdir() if p.is_file()}}
        save(out / 'completion.json', result)
        print('F011_COMPLETE', json.dumps({k: result[k] for k in ('fixed10_weak_mean_AUC', 'B_minus_A_values_only', 'C_minus_B_support_only', 'C_minus_A_total', 'C_promising_for_manual_fixed_fold1_review', 'seconds')}), flush=True)
    except Exception as exc:
        save(out / 'failure.json', {'at_UTC': datetime.now(timezone.utc).isoformat(), 'status': 'failed', 'type': type(exc).__name__,
                                   'message': str(exc), 'GPU_Torch_calls': 0, 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    main()
