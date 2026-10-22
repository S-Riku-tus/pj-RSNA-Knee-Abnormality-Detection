"""Observe one weak-selected R6 change in exact P033; no remote writes."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def save(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def artificial_qa(ns, config, output):
    """Both originalON and selected fourfold use only deterministic fake windows."""
    source, code, bundle = [Path(ns[k]) for k in ('SRC', 'CODE', 'BUNDLE')]
    original = config['original_worker']
    require(hashlib.sha256(original.encode()).hexdigest() == config['original_worker_SHA'], 'Frozen reference source required')
    reference = source/'compact_infer_reference.py'
    reference.write_text(original, encoding='utf8')
    template = config['artificial_worker']
    require(template.count('import compact_infer as CI') == 1, 'Exact artificial-only Dataset wrapper')
    (source/'p047_fake_reference.py').write_text(template.replace('import compact_infer as CI', 'import compact_infer_reference as CI').replace('__CODE_PATH__', repr(str(code))), encoding='utf8')
    (source/'p047_fake_selected.py').write_text(template.replace('__CODE_PATH__', repr(str(code))), encoding='utf8')
    qa = output/'QA'
    qa.mkdir(exist_ok=False)
    fake = '1.2.840.20261020.777778'
    assignment = qa/'assignment.json'
    assignment.write_text(json.dumps([fake]), encoding='utf8')
    args = list(ns['WORKER_ARGS'])
    if '--compile' in args:
        args.remove('--compile')
    expected = config['descriptor']
    require(args[args.index('--crop')+1] == str(expected['crop']), 'Selected crop must match policy')
    require(('--flip-tta' in args) == (expected['mirror_weight'] != 0), 'Selected mirror switch differs')
    require(('--fluid-stacker' in args) == expected['calibration'] and ('--state-stacker' in args) == expected['calibration'], 'Selected postprocessor flags differ')
    reference_args = list(args)
    if '--flip-tta' not in reference_args:
        reference_args.append('--flip-tta')
    if '--fluid-stacker' not in reference_args:
        reference_args += ['--fluid-stacker', str(bundle/ns['manifest']['fluid_stacker'])]
    if '--state-stacker' not in reference_args:
        reference_args += ['--state-stacker', str(bundle/ns['manifest']['state_stacker'])]
    deadline = time.time()+900
    receipts = {}
    for kind, extra in [('reference', reference_args), ('selected', args)]:
        folder = qa/kind
        folder.mkdir(exist_ok=False)
        cmd = [sys.executable, '-u', str(source/('p047_fake_'+kind+'.py')), '--data', str(ns['DATA']),
               '--split', 'test', '--code', str(code), '--models', str(bundle), '--arm', 'E_lrprior',
               '--assignment', str(assignment), '--output', str(folder), '--tmp', str(qa/('tmp-'+kind)),
               '--workers', '0', '--amp', 'auto', '--deadline', str(deadline), '--dump-logits', str(folder/'raw.json'), '--dump-state', *extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(source)+os.pathsep+str(code), PYTHONDONTWRITEBYTECODE='1')
        with (folder/'stdout.txt').open('w', encoding='utf8') as stdout, (folder/'stderr.txt').open('w', encoding='utf8') as stderr:
            result = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, timeout=600, check=False)
        require(result.returncode == 0, 'Fourfold artificial worker failed: '+kind)
        receipt = json.loads((folder/'receipt.json').read_bytes())
        require(receipt['models'] == ns['manifest']['models'] and receipt['scored'] == 1 and not receipt['fallback'] and not receipt['timed_out'], 'All4 exact models and one synthetic success required')
        require(receipt['crop'] == expected['crop'], 'Actual synthetic native crop differs')
        require(receipt['sha256']['compact_infer.py'] == (config['original_worker_SHA'] if kind == 'reference' else expected['worker_SHA']), 'Actual synthetic worker source differs')
        require(receipt['trainer_md5'] == ns['manifest']['trainer_md5'], 'Exact public trainer import required')
        if expected['calibration'] or kind == 'reference':
            require(all(not receipt[k]['stack_fallback'] for k in ('fluid_stacker', 'state_stacker')), 'No synthetic postprocessor fallback')
        receipts[kind] = receipt
    # Only label-free arithmetic in the controller; GPU contexts have exited.
    import importlib

    import numpy as np
    sys.path[:0] = [str(source), str(code)]
    CI = importlib.import_module('compact_infer_reference')
    CT = importlib.import_module('compact_train')
    labels = list(CT.LABELS)
    require(labels == config['labels'], 'Exact12 label order')
    spec_path = bundle/ns['manifest']['state_spec']
    spec = json.loads(spec_path.read_bytes())
    layout = CI.state_layout(spec, labels)
    p, b = CI.mirror_perm(labels, spec, [(r['binary'], r['class']) for r in layout['columns'] if 'binary' in r])
    helpers = {'__name__': 'p047_frozen_math'}
    require(hashlib.sha256(config['math_source'].encode()).hexdigest() == config['math_SHA'], 'Frozen replay math required')
    exec(compile(config['math_source'], '<p047-frozen-replay>', 'exec'), helpers)
    state_perm, mcl_cols = helpers['state_permutation'](labels, layout, p, b)
    fluid = CI.fluid_stacker_load(str(bundle/ns['manifest']['fluid_stacker']), 1)
    state = CI.state_stacker_load(str(bundle/ns['manifest']['state_stacker']), labels, layout, hashlib.md5(spec_path.read_bytes()).hexdigest())
    meta = {'labels': labels, 'class_perm': p, 'state_perm': state_perm, 'mcl_index': labels.index('MCL'),
            'mcl_state_cols': mcl_cols, 'fluid': fluid['classes'], 'state': state['classes']}
    raw = json.loads((qa/'reference/raw.json').read_bytes())[fake]
    require(all(len(raw[k]) == 4 for k in ('z_plain', 'z_mirror_raw', 'state_plain', 'state_mirror_raw')), 'All4 reference logits/states required')
    def expected_values(weight, calibrated):
        pairs = [helpers['mix']({key: [raw[key][i]] for key in ('z_plain', 'z_mirror_raw', 'state_plain', 'state_mirror_raw')}, meta, weight) for i in range(4)]
        z = np.stack([pair[0] for pair in pairs]).astype(np.float32).mean(0, dtype=np.float32).astype(np.float64)
        st = np.stack([pair[1] for pair in pairs]).mean(0, dtype=np.float64)
        return helpers['postprocess'](z, st, {}, meta, calibrated)
    errors = {}
    for kind, weight, calibrated in [('reference', .5, True), ('selected', expected['mirror_weight'], expected['calibration'])]:
        with (qa/kind/'predictions.csv').open(newline='', encoding='utf8') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ['StudyInstanceUID', *labels], 'Synthetic CSV contract')
            rows = list(reader)
        require(len(rows) == 1 and rows[0]['StudyInstanceUID'] == fake, 'Exactly one fake study')
        actual = [float(rows[0][t]) for t in labels]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in actual), 'Finite synthetic probabilities')
        errors[kind] = max(abs(a-b) for a, b in zip(actual, expected_values(weight, calibrated)))
        require(errors[kind] <= 2e-6, 'Fourfold direct/replay artificial mismatch')
    save(qa/'result.json', {'complete': True, 'artificial_only': True, 'all4_models': True, 'labels': labels,
                          'maximum_abs': errors, 'MRI_Gold_Report_reads': 0, 'descriptor': expected,
                          'reference_worker_SHA': config['original_worker_SHA'], 'selected_worker_SHA': expected['worker_SHA']})
    print('P047_R6_ARTIFICIAL_FOURFOLD_GPU_REPLAY_PASS', errors, flush=True)
    return errors


class Tee:
    def __init__(self, visible, captured):
        self.visible, self.captured = visible, captured

    def write(self, value):
        self.captured.write(value)
        return self.visible.write(value)

    def flush(self):
        self.visible.flush()


def execute(cells, shell, inventory, config, *, work='/kaggle/working'):
    require(config['descriptor']['choice'] in ('tta_off', 'tta_quarter', 'no_calibration', 'crop320'), 'Only preregistered R6 factor permitted')
    require(len(cells) == len(inventory) == 8, 'Eight frozen candidate scientific cells required')
    require(all({'bytes': len(s.encode()), 'sha256': hashlib.sha256(s.encode()).hexdigest()} == item for s, item in zip(cells, inventory)), 'Candidate science transport differs')
    root = Path(work)/'P047'
    root.mkdir(exist_ok=False)
    receipt = {'complete': False, 'candidate': 'P047', 'selected_variant': config['descriptor']['choice'],
               'selection_result_SHA': config['selection_result_SHA'], 'science_inventory': inventory,
               'completed_cells': [], 'own_Gold_fit_select': 0, 'Save_Submit_upload': 0}
    try:
        for index, cell in enumerate(cells):
            capture = io.StringIO()
            started = time.monotonic()
            with contextlib.redirect_stdout(Tee(sys.stdout, capture)):
                result = shell.run_cell(cell, store_history=False)
            (root/('cell-%02d.stdout.txt'%index)).write_text(capture.getvalue(), encoding='utf8')
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise error
            receipt['completed_cells'].append({'index': index, 'seconds': time.monotonic()-started, 'sha256': inventory[index]['sha256']})
            if index == 5:
                receipt['artificial_fourfold_GPU'] = artificial_qa(shell.user_ns, config, root)
        ns = shell.user_ns
        observation = {'__name__': 'p047_parent_observation'}
        require(hashlib.sha256(config['observer_source'].encode()).hexdigest() == config['observer_SHA'], 'Frozen parent observation required')
        exec(compile(config['observer_source'], '<p047-parent-contract-observer>', 'exec'), observation)
        parent, failed = observation['parent_observation'](ns, Path(work), root)
        require(parent and any(not value for value in failed), 'At least one complete three-branch study required')
        shutil.copyfile(Path(work)/'submission.csv', root/'final-submission.csv')
        shutil.copyfile(Path(work)/'submission_receipt.json', root/'R6-submission-receipt.json')
        receipt.update(complete=True, rows=len(parent), final=pin(Path(work)/'submission.csv'), parent3_success=sum(not value for value in failed),
                       scientific_factor=config['descriptor'], no_new_fusion=True, Public_gain_verified=False)
    except Exception as exc:
        receipt['error_type'] = type(exc).__name__
        raise
    finally:
        save(root/'execution.json', receipt)
    print('P047_SINGLE_R6_FACTOR_COMPLETE', json.dumps(receipt), flush=True)
    return receipt
