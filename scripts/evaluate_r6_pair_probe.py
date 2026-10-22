"""Score frozen paired R6 outputs against local weak labels; never Gold or APIs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import prepare_r6_pair_probe as B
from r6_pair_math import replay, require

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rsna_knee.contracts import TARGETS  # noqa: E402 -- standalone CLI adds this repository's src
from rsna_knee.metrics import evaluate_rows  # noqa: E402

VARIANTS = ('tta_off', 'tta_quarter', 'no_calibration')
FIXED10 = tuple(t for t in TARGETS if t not in ('MCL', 'Synovitis'))


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    body = path.read_bytes()
    return {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}


def rows(path, fields=None):
    with path.open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(fields is None or reader.fieldnames == fields, 'Exact CSV columns required')
        values = list(reader)
    result = {r['StudyInstanceUID']: r for r in values}
    require(len(values) == len(result), 'Duplicate UID rejected')
    return result


def choose(deltas):
    """Both banks strictly positive; maximize worse bank, exact tie frozen order."""
    require(set(deltas) == set(VARIANTS), 'Exact preregistered variants')
    for values in deltas.values():
        require(set(values) == {'0', '1'} and all(math.isfinite(v) for v in values.values()), 'Two finite bank deltas required')
    eligible = [name for name in VARIANTS if min(deltas[name].values()) > 0]
    return max(eligible, key=lambda name: (min(deltas[name].values()), -VARIANTS.index(name))) if eligible else None


def evaluate(folder, output):
    import numpy as np

    from rsna_knee.orthodiff_weak_head import paired_group_bootstrap
    folder, output = Path(folder), Path(output)
    require(not output.exists() and not output.is_symlink(), 'Fresh scoring output directory required')
    policy = read(B.POLICY)
    require(pin(B.POLICY)['sha256'] == B.POLICY_SHA, 'Preregistered policy differs')
    selection_path = ROOT / policy['selection']['path']
    weak_path = ROOT / policy['weak_input']['path']
    require(pin(selection_path)['sha256'] == B.SELECTION_SHA and pin(weak_path) == {k: policy['weak_input'][k] for k in ('bytes', 'sha256')}, 'Frozen selection/weak bytes differ')
    require((folder / 'selection.csv').read_bytes() == selection_path.read_bytes(), 'Runtime selection must match frozen exact CSV')
    selection = rows(selection_path)
    complete, qa, meta = [read(folder / name) for name in ('completion.json', 'artificial-QA.json', 'math-meta.json')]
    require(complete['complete'] and complete['policy_SHA'] == B.POLICY_SHA and complete['selection_SHA'] == B.SELECTION_SHA
            and complete['rows'] == 1664 and complete['artificial_QA_passed'] and complete['own_Gold_fit_select'] == 0, 'Complete frozen probe required')
    require(qa['complete'] and qa['artificial_only'] and qa['MRI_Report_labels_Gold_reads'] == 0
            and max(qa['direct_ON_replay_max_abs'], qa['direct_OFF_replay_max_abs']) <= 2e-6, 'Direct synthetic GPU parity required')
    require(meta['labels'] == list(TARGETS), 'Exact label order required')
    manifest = read(ROOT / policy['public_manifest']['path'])
    for fold in range(4):
        receipt = read(folder / ('fold-%d/receipt.json' % fold))
        require(receipt == complete['all4_singlefold_receipts'][str(fold)], 'Exact completed worker receipt required')
        expected = [uid for uid, row in selection.items() if int(row['R6_fold']) == fold]
        require(read(folder / ('fold-%d/assignment.json' % fold)) == expected, 'Exact singlefold routing required')
        require(receipt['models'] == ['E_lrprior_s0_f%d.pt' % fold] and receipt['studies'] == len(expected)
                and receipt['group2'] is None and not receipt['timed_out'] and receipt['trainer_md5'] == policy['training_code_LF_MD5'], 'Single heldout declared checkpoint required')
        require(receipt['sha256']['compact_train.py'] == policy['training_code_normalized_LF_SHA']
                and receipt['sha256']['compact_infer.py'] == '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d', 'Actual worker/trainer source differs')
        for item in ('state_stacker', 'fluid_stacker'):
            require(not receipt[item]['stack_fallback'] and receipt[item]['sha256'] == manifest['sha256'][manifest[item]], 'Exact postprocessor without fallback required')
        require(receipt['anchor_weights_sha256'] == manifest['anchor_weights_sha256']
                and receipt['side_check']['model_sha256'] == manifest['sha256'][manifest['side_model']], 'Auxiliary weights differ')
    mask = rows(folder / 'success-mask.csv', ['StudyInstanceUID', 'success'])
    require(mask.keys() == selection.keys() and all(r['success'] in ('0', '1') for r in mask.values()), 'Exact success mask required')
    success = {uid for uid, row in mask.items() if row['success'] == '1'}
    counts = {bank: sum(row['weak_fold'] == bank for row in selection.values()) for bank in ('0', '1')}
    successes = {bank: sum(uid in success and row['weak_fold'] == bank for uid, row in selection.items()) for bank in counts}
    require(all(successes[bank] / counts[bank] >= .98 for bank in counts), '98% coverage required independently in both banks')
    require(counts == complete['counts_by_bank'] and successes == complete['successes_by_bank'], 'Independent coverage count differs')
    predictions = {name: rows(folder / (name + '.csv'), ['StudyInstanceUID', *TARGETS]) for name in ('baseline', *VARIANTS)}
    with gzip.open(folder / 'raw-states.json.gz', 'rt', encoding='utf8') as stream:
        raw = json.load(stream)
    require(set(raw['records']) == success and all(p.keys() == selection.keys() for p in predictions.values()), 'Exact shared UID set required')
    max_error = 0.
    for uid in selection:
        for name, weight, calibration in [('baseline', .5, True), ('tta_off', 0., True), ('tta_quarter', .25, True), ('no_calibration', .5, False)]:
            values = [float(predictions[name][uid][t]) for t in TARGETS]
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Finite probabilities required')
            if uid not in success:
                require(values == [.5] * 12, 'Failed UID remains explicitly masked placeholder')
                continue
            expected = replay(raw['records'][uid], raw['fluid'].get(uid, {}), meta, weight, calibration)
            error = max(abs(a-b) for a, b in zip(values, expected))
            require(error <= (2e-6 if name == 'baseline' else 6e-10), 'Local reconstructed prediction differs')
            max_error = max(max_error, error)
    # No numeric labels are accessed until all transport/engineering checks above pass.
    weak = rows(weak_path)
    metrics, deltas, intervals = {}, {name: {} for name in VARIANTS}, {}
    for bank in ('0', '1'):
        ids = [uid for uid, row in selection.items() if row['weak_fold'] == bank and uid in success]
        require(all(str(weak[uid]['fold']) == bank for uid in ids), 'Original weak bank differs')
        truth = {uid: weak[uid] for uid in ids}
        metrics[bank] = {name: evaluate_rows(truth, {uid: predictions[name][uid] for uid in ids}) for name in predictions}
        primary = {name: sum(result['per_target'][t]['auc'] for t in FIXED10) / 10 for name, result in metrics[bank].items()}
        require(all(math.isfinite(value) for value in primary.values()), 'All fixed10 AUCs must be defined')
        for name in VARIANTS:
            deltas[name][bank] = primary[name] - primary['baseline']
        metrics[bank]['fixed10'] = primary
        y = np.array([[np.nan if weak[uid][t].strip().lower() in ('', 'nan', 'null', 'na') else float(weak[uid][t]) for t in TARGETS] for uid in ids])
        scores = {name: np.array([[float(predictions[name][uid][t]) for t in TARGETS] for uid in ids]) for name in predictions}
        intervals[bank] = {name: paired_group_bootstrap(y, scores['baseline'], scores[name], [selection[uid]['group_token'] for uid in ids]) for name in VARIANTS}
    selected = choose(deltas)
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'complete': True, 'policy_SHA': B.POLICY_SHA,
              'counts': counts, 'successes': successes, 'metrics': metrics, 'fixed10_deltas': deltas,
              'selected_variant': selected, 'bootstrap_report_only': intervals, 'local_reconstruction_max_abs': max_error,
              'own_Gold_fit_select': 0, 'automatic_Save_Submit_upload': 0, 'Public_gain_established': False,
              'limitations': 'Repeated report weak validation; upstream model membership/Gold/auxiliary exposure not independently certified; no whole-parent OOF or patient independence claim.',
              'input_pins': {p.name: pin(p) for p in folder.iterdir() if p.is_file()},
              'weak_input_SHA': policy['weak_input']['sha256'],
              'source_pins': {p.name: pin(p) for p in [Path(__file__), ROOT/'scripts/r6_pair_math.py', ROOT/'src/rsna_knee/metrics.py', ROOT/'src/rsna_knee/orthodiff_weak_head.py']}}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.folder, args.output)
    print(json.dumps({k: result[k] for k in ('complete', 'fixed10_deltas', 'selected_variant')}))
