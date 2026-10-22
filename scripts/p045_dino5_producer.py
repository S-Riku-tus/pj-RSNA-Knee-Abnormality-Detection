"""Kaggle-only DINO5 component. Frozen source/config literals are set by builder.

Import inert: local preparation never imports torch, opens MRI or loads weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

P045_CONFIG = None
P045_FROZEN = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def run(config, frozen):
    require(Path('/kaggle/working').is_dir(), 'Kaggle working directory required')
    out = Path('/kaggle/working/P045/dino')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    roots = [p for p in [Path('/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights'),
                         Path('/kaggle/input/knee-mri-fold-weights')] if (p / 'm_f0.pt').is_file()]
    require(len(roots) == 1, 'One exact DINO weights root required')
    root = roots[0]
    for name, expected in config['weights'].items():
        require(pin(root / name) == expected, 'DINO checkpoint SHA differs: ' + name)
    wheel = root / 'timm-1.0.22-py3-none-any.whl'
    require(pin(wheel) == config['timm_wheel'], 'Official timm wheel differs')
    env = out / 'timm-env'
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                             '--disable-pip-version-check', '--target', str(env), str(wheel)],
                            capture_output=True, text=True, check=False)
    (out / 'timm-install.stdout.txt').write_text(result.stdout, encoding='utf8')
    (out / 'timm-install.stderr.txt').write_text(result.stderr, encoding='utf8')
    require(result.returncode == 0, 'Pinned timm offline install failed')
    sys.path.insert(0, str(env))
    sys.path.insert(1, '/kaggle/working/_own_env')
    import cv2
    import numpy as np
    import pandas as pd
    import pydicom
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    require(torch.cuda.is_available(), 'CUDA required; CPU production inference forbidden')
    require(timm.__version__ == '1.0.22', 'Exact timm version required')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ns = {'__name__': 'p045_frozen_DINO5', 'torch': torch, 'nn': nn, 'F': F, 'np': np,
          'pd': pd, 'pydicom': pydicom, 'cv2': cv2, 'timm': timm,
          'N_SLICE': 16, 'SIZE': 336, 'N_SLOT': 6, 'CROP_MM': 130.0,
          'SLICE_BAND': (.12, .88), 'INTENSITY': 'slice', 'AMP_PREF': 'fp16',
          'SLOTS': [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)],
          'LABELS': config['labels'], 'MICRO': 1}
    for name in ['models', 'preprocess', 'predict']:
        text = frozen[name]['text']
        require(hashlib.sha256(text.encode()).hexdigest() == frozen[name]['sha256'], 'Frozen component source differs')
        exec(compile(text, '<p045-frozen-' + name + '>', 'exec'), ns)
    models, members, configurations = [], {}, []
    for fold in range(5):
        name = 'm_f%d.pt' % fold
        z = torch.load(root / name, map_location='cpu', weights_only=True)
        cfg, state = z['cfg'], z['state_dict']
        require(z['fold'] == fold, 'Exact checkpoint fold identity required')
        labels = cfg.get('labels')
        if not isinstance(labels, (tuple, list)) or list(labels) != config['labels']:
            (out / 'label-order-diagnostic.json').write_text(json.dumps({
                'type': type(labels).__name__, 'value': labels if type(labels) in (list, tuple, str, int) else None,
                'expected': config['labels']}, indent=2) + '\n')
            raise ValueError('Checkpoint label order is not the declared target order')
        require(all(cfg.get(k, default) == value for k, (value, default) in config['architecture'].items()),
                'Exact fixed DINO architecture required')
        enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0,
                                in_chans=16, img_size=336)
        enc = ns['ViTSlotToken'](enc, ns['N_SLOT_TYPES'])
        model = ns['Net'](enc, 'token', 0, 'xcodex', stem='native', n_slice=16)
        expected = model.state_dict()
        mismatch = {'missing': sorted(set(expected) - set(state)),
                    'extra': sorted(set(state) - set(expected)),
                    'shapes': [k for k in set(expected).intersection(state) if expected[k].shape != state[k].shape]}
        if any(mismatch.values()):
            (out / 'state-diagnostic.json').write_text(json.dumps(mismatch, indent=2) + '\n')
            raise ValueError('Full strict state identity required; random encoder fallback forbidden')
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
        configurations.append(cfg)
        members[name] = {**config['weights'][name], 'fold': fold, 'state_keys': len(state),
                         'cfg': cfg, 'strict_all_keys_shapes': True}
        del z, state, expected
    ns.update(models=models, CFG=configurations[0], DEV='cpu', AMP_DT=torch.float32, AMP_ON=False)
    # Deterministic artificial valid/empty study pair; no MRI or random input.
    fixture = np.zeros((2, 6, 16, 336, 336), np.uint8)
    fixture[0, 0] = (np.arange(16 * 336 * 336).reshape(16, 336, 336) % 251 + 1).astype(np.uint8)
    fixture_mask = np.zeros((2, 6), np.uint8)
    fixture_mask[0, 0] = 16
    cpu = ns['predict'](fixture, fixture_mask)
    require(cpu.shape == (5, 2, 12) and np.isfinite(cpu[:, 0]).all() and np.isnan(cpu[:, 1]).all(),
            'Artificial CPU forward/empty-mask contract failed')
    ns.update(models=[m.to('cuda').eval() for m in models], DEV='cuda')
    ns['AMP_DT'], ns['AMP_ON'] = ns['amp_for']('cuda')
    gpu = ns['predict'](fixture, fixture_mask)
    require(gpu.shape == (5, 2, 12) and np.isfinite(gpu[:, 0]).all() and np.isnan(gpu[:, 1]).all(),
            'Artificial CUDA forward/empty-mask contract failed')
    qa = {'complete': True, 'artificial_only': True, 'CPU_finite_shape': True, 'CUDA_finite_shape': True,
          'empty_mask_NaN_refused': True, 'models': 5, 'shape': [5, 2, 12],
          'CPU_FP32_CUDA_FP16_max_abs_report_only': float(np.max(np.abs(cpu[:, 0] - gpu[:, 0]))),
          'CUDA_precision': str(ns['AMP_DT']), 'GPU': torch.cuda.get_device_name(0),
          'labels_MRI_optimizer_reads': 0}
    (out / 'artificial-QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    (out / 'model-manifest.json').write_text(json.dumps(members, indent=2) + '\n')
    print('P045_DINO5_ARTIFICIAL_CPU_CUDA_PASS', json.dumps(qa), flush=True)
    del fixture, cpu, gpu
    # Only now may competition test metadata and MRI be opened.
    roots = [p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
                         Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p / 'test.csv').is_file()]
    require(len(roots) == 1, 'Exact test competition root required')
    comp = roots[0]
    study_frame = pd.read_csv(comp / 'test.csv', dtype={'StudyInstanceUID': str})
    sample = pd.read_csv(comp / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
    ids = study_frame.StudyInstanceUID.tolist()
    require(ids and len(set(ids)) == len(ids) and set(ids) == set(sample.StudyInstanceUID), 'Exact test UID contract required')
    require(sample.columns.tolist() == ['StudyInstanceUID', *config['labels']], 'Exact submission column order required')
    series = pd.read_csv(comp / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    by = {uid: g.to_dict('records') for uid, g in series[series.StudyInstanceUID.isin(ids)].groupby('StudyInstanceUID')}
    series_root = comp / 'test_series'
    require(series_root.is_dir(), 'Test MRI path required; train fallback forbidden')
    ns['SERIES_ROOT'] = series_root
    probabilities = np.full((5, len(ids), 12), np.nan, np.float32)
    slot_counts, failed_tasks = [0] * len(ids), []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(ids), 8):
            block = ids[start:start + 8]
            images = np.zeros((len(block), 6, 16, 336, 336), np.uint8)
            masks = np.zeros((len(block), 6), np.uint8)
            tasks = {executor.submit(ns['build_study'], (i, uid, by.get(uid, []))): i for i, uid in enumerate(block)}
            for task in as_completed(tasks):
                i = tasks[task]
                try:
                    index, image, mask = task.result()
                    require(index == i and image.shape == images[i].shape and mask.shape == (6,), 'Preprocess shape/index contract')
                    # Original helper marks intended slices even if every decode failed.
                    # Suppress only all-zero slots, keeping that failure outside rank pools.
                    mask[image.reshape(6, -1).max(axis=1) == 0] = 0
                    images[i], masks[i] = image, mask
                    slot_counts[start + i] = int((mask > 0).sum())
                except Exception as exc:
                    failed_tasks.append({'index': start + i, 'type': type(exc).__name__})
            probabilities[:, start:start + len(block)] = ns['predict'](images, masks)
            print('P045_DINO5_PROGRESS', start + len(block), '/', len(ids), 'seconds', round(time.monotonic() - started, 1), flush=True)
            del images, masks
    success = np.isfinite(probabilities).all(axis=(0, 2)) & (np.asarray(slot_counts) > 0)
    require(success.any(), 'At least one decoded and finite fivefold study required')
    for fold in range(5):
        with (out / ('fold-%d.csv' % fold)).open('x', encoding='utf8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['StudyInstanceUID', *config['labels']])
            writer.writerows([uid, *[repr(float(v)) for v in probabilities[fold, i]]] if success[i]
                             else [uid, *['0.5'] * 12] for i, uid in enumerate(ids))
    with (out / 'mask.csv').open('x', encoding='utf8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', 'success'])
        writer.writerows([uid, int(success[i])] for i, uid in enumerate(ids))
    receipt = {'complete': True, 'folds': 5, 'rows': len(ids), 'success_rows': int(success.sum()),
               'failed_rows': int((~success).sum()), 'models_strict_keys': True, 'label_order_verified': True,
               'artificial_CPU_CUDA_pass': True, 'decoded_slot_counts': slot_counts, 'failed_tasks': failed_tasks,
               'seconds': time.monotonic() - started, 'weights': config['weights'], 'timm': timm.__version__,
               'torch': torch.__version__, 'precision': 'FP16 autocast CUDA; FP32 CPU artificial only',
               'model_manifest': pin(out / 'model-manifest.json'), 'artificial_QA': pin(out / 'artificial-QA.json'),
               'upstream_Gold_history': 'UNKNOWN', 'own_Gold_training_selection': 0,
               'Report_or_train_or_Gold_csv_reads': 0, 'Save_Submit_upload': 0}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('P045_DINO5_PREDICTIONS_COMPLETE', json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    run(P045_CONFIG, P045_FROZEN)
