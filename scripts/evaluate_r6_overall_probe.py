"""Prospective common-cohort selection across three R6 replay factors and crop320."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import evaluate_r6_pair_probe as E
import prepare_r6_crop320_probe as C
from r6_pair_math import replay, require

CHOICES = ('tta_off', 'tta_quarter', 'no_calibration', 'crop320')


class PendingProbe(ValueError):
    pass


def choose(deltas):
    require(set(deltas) == set(CHOICES), 'Exact four preregistered choices')
    require(all(set(v) == {'0', '1'} and all(math.isfinite(x) for x in v.values()) for v in deltas.values()), 'Two finite bank deltas')
    eligible = [name for name in CHOICES if min(deltas[name].values()) > 0]
    return max(eligible, key=lambda name: (min(deltas[name].values()), -CHOICES.index(name))) if eligible else None


def common_cohort(selection, left, right):
    common = set(left).intersection(right)
    counts = {bank: sum(r['weak_fold'] == bank for r in selection.values()) for bank in ('0', '1')}
    paired = {bank: sum(uid in common and r['weak_fold'] == bank for uid, r in selection.items()) for bank in counts}
    require(all(paired[bank]/counts[bank] >= .98 for bank in counts), 'Common paired coverage below98%')
    return common, counts, paired


def audit(poll, crop):
    poll = Path(poll)
    summary = E.read(poll/'summary.json')
    if summary['status_typed'] not in ('COMPLETE', 'ERROR'):
        raise PendingProbe('Probe pending; do not score or select')
    require(summary['status_typed'] == 'COMPLETE' and not summary['errors'] and summary['complete_expected_download_set'], 'Complete error-free named download set required')
    require(summary['kernelId'] == (139794653 if crop == 352 else 139798784) and summary['version'] == 1, 'Exact owned probe V1 required')
    reconciliation = E.read(poll.parent/'accepted-reconciliation.json')
    require(reconciliation['kernelId'] == summary['kernelId'] and reconciliation['saved_version'] == 1
            and reconciliation['all_checks_passed'] and all(reconciliation['checks'].values()), 'Saved source/Input/settings reconciliation required')
    for item in summary['downloaded']:
        name = PurePosixPath(item['name'])
        require(not name.is_absolute() and '..' not in name.parts and ':' not in item['name'] and '\\' not in item['name'], 'Safe artifact paths required')
        p = poll/'files'/Path(*name.parts)
        require(not p.is_symlink() and E.pin(p) == {'bytes': item['bytes'], 'sha256': item['SHA']}, 'Downloaded bytes differ')
    policy = E.read(E.B.POLICY)
    require(E.pin(E.B.POLICY)['sha256'] == E.B.POLICY_SHA and E.pin(C.POLICY)['sha256'] == C.POLICY_SHA, 'Both preregistrations pinned')
    folder = poll/'files/R6-paired-weak-probe'
    selection_path = E.ROOT/policy['selection']['path']
    require(hashlib.sha256((folder/'selection.csv').read_bytes()).hexdigest() == E.B.SELECTION_SHA
            and (folder/'selection.csv').read_bytes() == selection_path.read_bytes(), 'Exact1664 selection required')
    selection = E.rows(selection_path)
    complete, qa, meta = [E.read(folder/name) for name in ('completion.json', 'artificial-QA.json', 'math-meta.json')]
    require(complete['complete'] and complete['rows'] == len(selection) == 1664
            and complete['policy_SHA'] == (E.B.POLICY_SHA if crop == 352 else C.POLICY_SHA)
            and complete['selection_SHA'] == E.B.SELECTION_SHA and complete['own_Gold_fit_select'] == 0, 'Complete correctly bound probe')
    require(qa['complete'] and qa['artificial_only'] and qa['MRI_Report_labels_Gold_reads'] == 0
            and max(qa['direct_ON_replay_max_abs'], qa['direct_OFF_replay_max_abs']) <= 2e-6, 'Synthetic GPU/directOFF replay before MRI required')
    require(meta['labels'] == list(E.TARGETS), 'Exact12 label order')
    manifest = E.read(E.ROOT/policy['public_manifest']['path'])
    for fold in range(4):
        receipt = E.read(folder/('fold-%d/receipt.json'%fold))
        require(receipt == complete['all4_singlefold_receipts'][str(fold)], 'Exact completed receipt')
        expected = [uid for uid, row in selection.items() if int(row['R6_fold']) == fold]
        require(E.read(folder/('fold-%d/assignment.json'%fold)) == expected and receipt['studies'] == len(expected), 'Exact fold UID assignment')
        require(receipt['models'] == ['E_lrprior_s0_f%d.pt'%fold] and receipt['group2'] is None
                and not receipt['timed_out'] and receipt['crop'] == crop and receipt['backbone'] == 'convnext_small'
                and receipt['views'] == 'v4' and receipt['phys_mm'] == 3.5, 'Exact single-factor declared heldout model/config')
        require(receipt['trainer_md5'] == policy['training_code_LF_MD5']
                and receipt['sha256']['compact_train.py'] == policy['training_code_normalized_LF_SHA']
                and receipt['sha256']['compact_infer.py'] == '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d', 'Exact imported/actual worker/trainer')
        for key in ('fluid_stacker', 'state_stacker'):
            require(not receipt[key]['stack_fallback'] and receipt[key]['sha256'] == manifest['sha256'][manifest[key]], 'Exact calibrated postprocessors without fallback')
        require(receipt['anchor_weights_sha256'] == manifest['anchor_weights_sha256']
                and receipt['side_check']['model_sha256'] == manifest['sha256'][manifest['side_model']], 'Exact auxiliary weights')
    mask = E.rows(folder/'success-mask.csv', ['StudyInstanceUID', 'success'])
    require(mask.keys() == selection.keys() and all(r['success'] in ('0', '1') for r in mask.values()), 'Exact success mask')
    success = {uid for uid, row in mask.items() if row['success'] == '1'}
    _, counts, successes = common_cohort(selection, success, success)
    require(counts == complete['counts_by_bank'] and successes == complete['successes_by_bank'], 'Independent coverage counts match')
    names = ('baseline', *E.VARIANTS) if crop == 352 else ('baseline',)
    predictions = {name: E.rows(folder/(name+'.csv'), ['StudyInstanceUID', *E.TARGETS]) for name in names}
    require(all(values.keys() == selection.keys() for values in predictions.values()), 'Exact all-arm UID sets')
    with gzip.open(folder/'raw-states.json.gz', 'rt', encoding='utf8') as stream:
        raw = json.load(stream)
    require(set(raw['records']) == success, 'Exact successful raw UID set')
    settings = {'baseline': (.5, True), 'tta_off': (0., True), 'tta_quarter': (.25, True), 'no_calibration': (.5, False)}
    maximum = 0.
    for uid in selection:
        for name in names:
            values = [float(predictions[name][uid][t]) for t in E.TARGETS]
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Finite probabilities')
            if uid not in success:
                require(values == [.5]*12, 'Explicit masked placeholder required')
                continue
            weight, calibrated = settings[name]
            expected = replay(raw['records'][uid], raw['fluid'].get(uid, {}), meta, weight, calibrated)
            error = max(abs(a-b) for a, b in zip(values, expected))
            require(error <= (2e-6 if name == 'baseline' else 6e-10), 'Local replay qualification differs')
            maximum = max(maximum, error)
    return {'folder': folder, 'selection': selection, 'success': success, 'predictions': predictions,
            'meta': meta, 'fluid': raw['fluid'], 'replay_max_abs': maximum}


def evaluate(base_poll, crop_poll, output):
    import numpy as np

    from rsna_knee.orthodiff_weak_head import paired_group_bootstrap
    output = Path(output)
    require(not output.exists(), 'Fresh overall scoring output required')
    base = audit(base_poll, 352)
    try:
        crop = audit(crop_poll, 320)
        require(base['meta'] == crop['meta'], 'Identical non-image mathematics required')
        common, counts, paired = common_cohort(base['selection'], base['success'], crop['success'])
        require(all(base['fluid'].get(uid, {}) == crop['fluid'].get(uid, {}) for uid in common), 'Uncropped fluid measurements changed; crop-only isolation not established')
    except PendingProbe:
        raise
    except ValueError as exc:
        original = E.evaluate(base['folder'], output/'original-three-fallback')
        result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'complete': True, 'mode': 'original_three_only_crop_engineering_HOLD',
                  'crop_HOLD': str(exc), 'original_result': original, 'selected_variant': original['selected_variant'],
                  'own_Gold_fit_select': 0, 'automatic_Save_Submit': 0}
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
        return result
    policy = E.read(E.B.POLICY)
    weak_path = E.ROOT/policy['weak_input']['path']
    require(E.pin(weak_path) == {k: policy['weak_input'][k] for k in ('bytes', 'sha256')}, 'Frozen weak source differs')
    # Only now access numeric weak labels, after complete source/transport/image-pair QA.
    weak = E.rows(weak_path)
    selection = base['selection']
    predictions = dict(base['predictions'], crop320=crop['predictions']['baseline'])
    metrics, primary, deltas, intervals = {}, {}, {name: {} for name in CHOICES}, {}
    for bank in ('0', '1'):
        ids = [uid for uid, row in selection.items() if row['weak_fold'] == bank and uid in common]
        require(all(str(weak[uid]['fold']) == bank for uid in ids), 'Original weak bank required')
        truth = {uid: weak[uid] for uid in ids}
        metrics[bank] = {name: E.evaluate_rows(truth, {uid: rows[uid] for uid in ids}) for name, rows in predictions.items()}
        require(all(m['per_target'][t]['auc'] is not None for m in metrics[bank].values() for t in E.FIXED10), 'All fixed10 AUCs must be defined')
        primary[bank] = {name: sum(m['per_target'][t]['auc'] for t in E.FIXED10)/10 for name, m in metrics[bank].items()}
        for name in CHOICES:
            deltas[name][bank] = primary[bank][name]-primary[bank]['baseline']
        y = np.array([[np.nan if weak[uid][t].strip().lower() in ('', 'nan', 'null', 'na') else float(weak[uid][t]) for t in E.TARGETS] for uid in ids])
        scores = {name: np.array([[float(rows[uid][t]) for t in E.TARGETS] for uid in ids]) for name, rows in predictions.items()}
        intervals[bank] = {name: paired_group_bootstrap(y, scores['baseline'], scores[name], [selection[uid]['group_token'] for uid in ids]) for name in CHOICES}
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'complete': True, 'mode': 'four_choices_exact_common_cohort',
              'policy_SHA': C.POLICY_SHA, 'original_policy_SHA': E.B.POLICY_SHA, 'counts': counts, 'paired_successes': paired,
              'fixed10': primary, 'fixed10_deltas': deltas, 'selected_variant': choose(deltas), 'metrics': metrics,
              'bootstrap_report_only': intervals, 'original_replay_max_abs': base['replay_max_abs'], 'crop_replay_max_abs': crop['replay_max_abs'],
              'uncropped_fluid_exact_on_all_pairs': True, 'crop_combined_replay_variants_evaluated': False,
              'own_Gold_fit_select': 0, 'automatic_Save_Submit': 0, 'Public_gain_established': False,
              'upstream_and_patient_independence_limits': 'Declared fold exclusion only; upstream membership/Gold/auxiliary exposure not independently certified; repeated weakvalidation, no whole-parent OOF.',
              'polls': {'352': str(base_poll), '320': str(crop_poll)},
              'source_pins': {p.as_posix(): E.pin(p) for p in [Path(__file__), Path(E.__file__), E.ROOT/'scripts/r6_pair_math.py', E.ROOT/'src/rsna_knee/metrics.py', E.ROOT/'src/rsna_knee/orthodiff_weak_head.py']}}
    require('torch' not in sys.modules, 'No local Torch import permitted')
    output.mkdir(parents=True, exist_ok=False)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-poll', type=Path, required=True)
    parser.add_argument('--crop-poll', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.base_poll, args.crop_poll, args.output)
    print(json.dumps({k: result.get(k) for k in ('complete', 'mode', 'fixed10_deltas', 'selected_variant')}))
