"""Kaggle-only, label-free paired R6 validation inference. No remote API writes."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def require(ok, text):
    if not ok:
        raise ValueError(text)


def save(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_predictions(path, labels):
    with path.open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['StudyInstanceUID', *labels], 'Exact prediction columns required')
        rows = list(reader)
    result = {}
    for row in rows:
        uid = row['StudyInstanceUID']
        require(uid not in result, 'No duplicate prediction UID')
        values = [float(row[t]) for t in labels]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Finite probabilities required')
        result[uid] = values
    return result


ARTIFICIAL_WORKER = '''import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).parent), __CODE_PATH__]
import torch
import compact_infer as CI
class ArtificialStudies:
    def __init__(self, data, split, uids, *args, **kwargs): self.uids = uids
    def __len__(self): return len(self.uids)
    def __getitem__(self, index):
        im = (torch.arange(6*3*384*384, dtype=torch.int32).reshape(6,3,384,384) % 251).float() / 255
        meta = torch.zeros(6,9)
        meta[:,0] = torch.linspace(-1,1,6)
        meta[:,3] = 1.
        meta[:,6] = torch.tensor([0,0,1,1,2,2])
        meta[:,7] = torch.tensor([0,0,1,1,2,2])
        meta[:,8] = 20.
        return {'uid':self.uids[index], 'error':None, 'images':im, 'meta':meta,
                'seconds':[0.,0.,0.], 'side':None, 'anchor':None, 'fluid':None, 'fluid_s':0.}
CI.CStudies = ArtificialStudies
CI.main()
'''


def run_probe(stage, selection_text, selection_sha, policy_sha, math_source, math_sha):
    require(Path('/kaggle/working').is_dir(), 'Kaggle runtime required')
    require(hashlib.sha256(selection_text.encode()).hexdigest() == selection_sha, 'Frozen selection differs')
    require(hashlib.sha256(math_source.encode()).hexdigest() == math_sha, 'Frozen arithmetic differs')
    rows = list(csv.DictReader(io.StringIO(selection_text)))
    ids = [r['StudyInstanceUID'] for r in rows]
    require(len(rows) == 1664 and len(set(ids)) == len(ids), 'Exact1664 selected UIDs required')
    grouped = {}
    for row in rows:
        uid = row['StudyInstanceUID']
        expected = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4
        require(expected == int(row['R6_fold']) and row['weak_fold'] in ('0', '1'), 'Declared heldout model routing required')
        pair = (row['weak_fold'], row['R6_fold'])
        require(grouped.setdefault(row['group_token'], pair) == pair, 'Selected group crosses banks/models')
    source, code, bundle, data, tmp = [Path(stage[k]) for k in ('SRC', 'CODE', 'BUNDLE', 'DATA', 'TMP')]
    manifest = stage['manifest']
    require(manifest['bundle_id'] == 'fe055406fad6a12f' and manifest['run_config']['folds'] == 4
            and manifest['gold_in_training'] == 0 and manifest['backbone'] == 'convnext_small', 'Frozen public R6 bundle only')
    require(hashlib.md5((source / 'compact_train.py').read_bytes()).hexdigest() == manifest['trainer_md5'] == '8622009e3db0414124d5614ed252d7a2', 'Exact public fold source required')
    root = Path('/kaggle/working/R6-paired-weak-probe')
    root.mkdir(exist_ok=False)
    (root / 'selection.csv').write_text(selection_text, encoding='utf8')
    math_ns = {'__name__': 'r6_pair_math'}
    exec(compile(math_source, '<frozen-r6-paired-math>', 'exec'), math_ns)
    sys.path[:0] = [str(source), str(code)]
    CI, CT = importlib.import_module('compact_infer'), importlib.import_module('compact_train')
    labels = list(CT.LABELS)
    require(labels == ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
                       'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'], 'Exact12 label order')
    spec = json.loads((bundle / manifest['state_spec']).read_text())
    layout = CI.state_layout(spec, labels)
    bin_names = [(r['binary'], r['class']) for r in layout['columns'] if 'binary' in r]
    perm, bperm = CI.mirror_perm(labels, spec, bin_names)
    state_perm, mcl_cols = math_ns['state_permutation'](labels, layout, perm, bperm)
    fluid = CI.fluid_stacker_load(str(bundle / manifest['fluid_stacker']), 1)
    state = CI.state_stacker_load(str(bundle / manifest['state_stacker']), labels, layout,
                                  hashlib.md5((bundle / manifest['state_spec']).read_bytes()).hexdigest())
    meta = {'labels': labels, 'class_perm': list(map(int, perm)), 'state_perm': state_perm,
            'mcl_index': labels.index('MCL'), 'mcl_state_cols': mcl_cols,
            'fluid': {target: {k: (str(co[k]) if k == 'volume_column' else float(co[k]))
                              for k in ('b0', 'b_z', 'b_lv', 'b_miss', 'volume_column')}
                      for target, co in fluid['classes'].items()},
            'state': {target: {'features': list(co['features']), 'columns': [int(v) if v is not None else None for v in co['columns']],
                              'coef': list(map(float, co['coef'])), 'b0': float(co['b0']), 'volume_column': co['volume_column']}
                      for target, co in state['classes'].items()}}
    require(set(meta['fluid']) == {'Effusion'} and set(meta['state']) == {"Baker's"}, 'Exactly original two target postprocessors')
    save(root / 'math-meta.json', meta)
    folders = []
    for fold in range(4):
        folder = tmp / ('heldout-model-f%d' % fold)
        folder.mkdir(exist_ok=False)
        name = 'E_lrprior_s0_f%d.pt' % fold
        require(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == manifest['sha256'][name], 'Exact heldout model SHA')
        (folder / name).symlink_to(bundle / name)
        folders.append(folder)
    deadline = stage['NB_T0'] + 8.0 * 3600
    worker = source / 'compact_infer.py'
    common = list(stage['WORKER_ARGS'])
    require('--flip-tta' in common and '--models2' not in common, 'Original single-group mirrored R6 configuration')
    env0 = dict(os.environ, PYTHONPATH=str(source) + os.pathsep + str(code), PYTHONDONTWRITEBYTECODE='1')

    def execute(output, assignment, fold, gpu, *, artificial=False, mirrored=True):
        output.mkdir(parents=True, exist_ok=False)
        assignment_path = output / 'assignment.json'
        assignment_path.write_text(json.dumps(assignment))
        args = list(common)
        if not mirrored:
            args.remove('--flip-tta')
        if artificial and '--compile' in args:
            args.remove('--compile')
        program = source / 'r6_artificial_worker.py' if artificial else worker
        command = [sys.executable, '-u', str(program), '--data', str(data), '--split', 'train',
                   '--code', str(code), '--models', str(folders[fold]), '--arm', 'E_lrprior',
                   '--output', str(output), '--tmp', str(tmp / ('pair-tmp-' + output.name)),
                   '--assignment', str(assignment_path), '--amp', 'auto', '--workers', '0' if artificial else '2',
                   '--deadline', str(deadline), '--dump-logits', str(output / 'raw.json'), '--dump-state', *args]
        remaining = deadline - time.time()
        require(remaining > 120, 'Global inference deadline reached; no partial qualification')
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(gpu))
        with (output / 'stdout.txt').open('w', encoding='utf8') as stdout, (output / 'stderr.txt').open('w', encoding='utf8') as stderr:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False, timeout=remaining + 90)
        require(result.returncode == 0, 'Frozen worker failed: ' + output.name)
        receipt = json.loads((output / 'receipt.json').read_text())
        require(receipt['models'] == ['E_lrprior_s0_f%d.pt' % fold], 'One exact heldout fold model must participate')
        require(not receipt['timed_out'], 'Incomplete timed-out worker cannot qualify')
        require(all(not receipt[k]['stack_fallback'] for k in ('state_stacker', 'fluid_stacker')), 'Postprocessor fallback cannot qualify')
        return receipt

    (source / 'r6_artificial_worker.py').write_text(ARTIFICIAL_WORKER.replace('__CODE_PATH__', repr(str(code))))
    fake = '1.2.840.20261020.777777'
    require(fake not in ids, 'Synthetic-only UID required')
    qa_on, qa_off = root / 'QA-on', root / 'QA-off'
    on_receipt = execute(qa_on, [fake], 0, 0, artificial=True)
    off_receipt = execute(qa_off, [fake], 0, 0, artificial=True, mirrored=False)
    require(on_receipt['scored'] == off_receipt['scored'] == 1 and not on_receipt['fallback'] and not off_receipt['fallback'], 'Synthetic GPU forward must succeed')
    on_raw = json.loads((qa_on / 'raw.json').read_text())[fake]
    direct_on, direct_off = [read_predictions(p / 'predictions.csv', labels)[fake] for p in (qa_on, qa_off)]
    computed_on = math_ns['replay'](on_raw, {}, meta, .5)
    computed_off = math_ns['replay'](on_raw, {}, meta, 0.)
    err_on = max(abs(a - b) for a, b in zip(computed_on, direct_on))
    err_off = max(abs(a - b) for a, b in zip(computed_off, direct_off))
    require(max(err_on, err_off) <= 2e-6, 'Artificial directON/OFF versus replay discrepancy')
    save(root / 'artificial-QA.json', {'complete': True, 'artificial_only': True, 'models': 1, 'fold': 0,
         'rows': 1, 'labels': 12, 'direct_ON_replay_max_abs': err_on, 'direct_OFF_replay_max_abs': err_off,
         'MRI_Report_labels_Gold_reads': 0, 'source': 'same frozen worker, only synthetic Dataset replacement'})
    print('R6_PAIR_ARTIFICIAL_GPU_AND_REPLAY_PASS', err_on, err_off, flush=True)
    started = time.time()

    def gpu_queue(gpu, fold_ids):
        result = {}
        for fold in fold_ids:
            assignment = [r['StudyInstanceUID'] for r in rows if int(r['R6_fold']) == fold]
            print('R6_PAIR_FOLD_START', fold, 'GPU', gpu, 'rows', len(assignment), flush=True)
            result[fold] = execute(root / ('fold-%d' % fold), assignment, fold, gpu)
            print('R6_PAIR_FOLD_COMPLETE', fold, result[fold]['scored'], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(gpu_queue, 0, [0, 2]), executor.submit(gpu_queue, 1, [1, 3])]
        receipts = {}
        for future in futures:
            receipts.update(future.result())
    raw_all, fluid_all, originals = {}, {}, {}
    for fold in range(4):
        output = root / ('fold-%d' % fold)
        raw = json.loads((output / 'raw.json').read_text())
        expected = {r['StudyInstanceUID'] for r in rows if int(r['R6_fold']) == fold}
        require(set(raw).issubset(expected) and len(raw) == receipts[fold]['scored'], 'Raw success UID routing differs')
        require(not set(raw).intersection(receipts[fold]['fallback']), 'Fallback cannot enter paired success set')
        require(not set(raw).intersection(raw_all), 'Unique fold-routed output UIDs')
        raw_all.update(raw)
        originals.update(read_predictions(output / 'predictions.csv', labels))
        with (output / 'fluid_stacker.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                fluid_all[row['StudyInstanceUID']] = {k: None if row[k] == '' else float(row[k])
                                                       for k in ('eff_sag_p99_open', 'bak_sag_p99_open')}
    variants = {name: {} for name in ('baseline', 'tta_off', 'tta_quarter', 'no_calibration')}
    maximum_error, success = 0., set(raw_all).intersection(originals)
    for uid in sorted(success):
        record, volume = raw_all[uid], fluid_all.get(uid, {})
        baseline = math_ns['replay'](record, volume, meta, .5)
        error = max(abs(a - b) for a, b in zip(baseline, originals[uid]))
        maximum_error = max(maximum_error, error)
        require(error <= 2e-6, 'Actual original ON replay differs')
        variants['baseline'][uid] = originals[uid]
        variants['tta_off'][uid] = math_ns['replay'](record, volume, meta, 0.)
        variants['tta_quarter'][uid] = math_ns['replay'](record, volume, meta, .25)
        variants['no_calibration'][uid] = math_ns['replay'](record, volume, meta, .5, False)
        require(all(abs(values[uid][meta['mcl_index']] - originals[uid][meta['mcl_index']]) <= 2e-6 for values in variants.values()), 'MCL plain-only negative control changed')
    require(success.issubset(ids), 'Only frozen selected UIDs may be scored')
    for name, predictions in variants.items():
        with (root / (name + '.csv')).open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['StudyInstanceUID', *labels])
            writer.writerows([uid, *['%.9g' % value for value in predictions.get(uid, [.5] * 12)]] for uid in ids)
    with (root / 'success-mask.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', 'success'])
        writer.writerows([uid, int(uid in success)] for uid in ids)
    with gzip.open(root / 'raw-states.json.gz', 'wt', encoding='utf8') as stream:
        json.dump({'records': raw_all, 'fluid': fluid_all}, stream)
    counts = {bank: sum(r['weak_fold'] == bank for r in rows) for bank in ('0', '1')}
    successes = {bank: sum(r['weak_fold'] == bank and r['StudyInstanceUID'] in success for r in rows) for bank in counts}
    complete = all(successes[k] / counts[k] >= .98 for k in counts)
    save(root / 'completion.json', {'complete': complete, 'policy_SHA': policy_sha, 'selection_SHA': selection_sha,
         'rows': len(ids), 'counts_by_bank': counts, 'successes_by_bank': successes, 'all4_singlefold_receipts': receipts,
         'ON_replay_max_abs': maximum_error, 'artificial_QA_passed': True, 'seconds_real_inference': time.time() - started,
         'raw_state_Gold_truth_or_Report_inputs': 0, 'own_Gold_fit_select': 0, 'new_training': 0,
         'no_clean_whole_parent_OOF_claim': True, 'patient_independence_verified': False,
         'upstream_Gold_and_auxiliary_exposure_inherited': True, 'Submit_Save_upload': 0})
    print('R6_PAIR_PROBE_COMPLETE', complete, 'success', len(success), '/', len(ids), flush=True)
    require(complete, 'Paired coverage below preregistered98%; no selection permitted')
    return {'complete': complete, 'rows': len(ids), 'success': len(success)}
