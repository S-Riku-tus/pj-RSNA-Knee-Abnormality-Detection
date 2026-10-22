"""Retrospective fixedE12 weak diagnosis; original F009 engineering HOLD unchanged."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'artifacts/research/20261020-F013-F009-matched-counter-diagnostic-v1'
POLICY_SHA = 'd4556584f5792be8faf0f5748c78e668ff86488976a394930d841083dca8cd54'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def matched_counters(control, candidate):
    totals = {}
    for name, history in [('control', control), ('candidate', candidate)]:
        require(len(history) == 12 and [row['epoch'] for row in history] == list(range(1, 13)), 'Exact twelve epochs in order')
        for row in history:
            skip = int(row['epoch'] == 10)
            wanted = {'optimizer_steps': 208-skip, 'optimizer_step_opportunities': 208, 'amp_skipped_steps': skip}
            require(all(type(row.get(k)) is int and row[k] == value for k, value in wanted.items()), 'Only exact known matched E10skip1 pattern allowed')
        totals[name] = {k: sum(row[k] for row in history) for k in wanted}
    require(totals['control'] == totals['candidate'], 'Matched actual update counters required')
    return totals


def main():
    require(not (HERE/'result.json').exists(), 'Fresh diagnostic output required')
    policy_path = HERE/'policy.json'
    require(hashlib.sha256(policy_path.read_bytes()).hexdigest() == POLICY_SHA, 'Frozen new-scope policy differs')
    policy = json.loads(policy_path.read_bytes())
    for item in policy['pins'].values():
        p = ROOT/item['path']
        require(p.stat().st_size == item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], 'Original frozen source/result/HOLD differs')
    source = ROOT/policy['pins']['old_guard']['path']
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('f013_unchanged_f009_guard_and_math', source)
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    f0, guard0 = [policy['pins'][k] for k in ('fold0_result', 'fold0_guard')]
    guard, runs, old_policy = G.guard_inputs(ROOT, G.BINDINGS, ROOT/f0['path'], f0['sha256'], ROOT/guard0['path'], guard0['sha256'])
    counters = matched_counters(runs['control']['history'], runs['candidate']['history'])
    require(not guard['machine_engineering_passed'] and guard['AMP_skipped_steps'] == {'control':1, 'candidate':1}, 'Original zero-skip engineering failure must remain visible')
    configs = [copy.deepcopy(runs[arm]['config']) for arm in ('control', 'candidate')]
    for config in configs:
        config.pop('reason', None)
    augmentation = configs[1].pop('augmentation')
    require(augmentation == {'schema_version':'native392_train_study_gamma_v1', 'minimum':.85, 'maximum':1.15}
            and 'augmentation' not in configs[0] and configs[0] == configs[1], 'Only declared train-only gamma factor')
    require(runs['control']['encoder_provenance'] == runs['candidate']['encoder_provenance'], 'Same generic encoder provenance')
    module = G.load_math(ROOT)
    result = G.compare(ROOT, guard, runs, old_policy, module)
    require(result['candidate_epoch'] == result['control_epoch'] == 12 and result['joint_candidate_screen_passed'] is False
            and result['fold1_machine_engineering_passed'] is False, 'Original result must stay HOLD')
    _, truth = module.B.weak_truth(ROOT/G.MANIFEST_REL, 1)
    require(len(truth) == 891, 'Exactly original validation891')
    sys.path.insert(0, str(ROOT/'src'))
    from rsna_knee.metrics import evaluate_rows
    metrics = {}
    for arm in ('control', 'candidate'):
        folder = Path(guard[arm+'_run'])
        _, rows = module.read_csv(folder/'epoch_012_weak.csv', (module.ID, *G.TARGETS))
        predictions = module.index_studies(rows)
        metrics[arm] = evaluate_rows(truth, predictions)
        require(metrics[arm] == runs[arm]['history'][-1]['metrics'], 'Prediction CSV versus original writer metrics differs')
        for target in G.TARGETS:
            expected = metrics[arm]['per_target'][target]['auc']
            observed = result['per_target'][target][arm+'_auc']
            require(observed is expected is None or expected is not None and observed is not None and abs(observed-expected) < 1e-12, 'Independent AUC formulas differ')
    require('torch' not in sys.modules, 'No model deserialization or GPU import allowed')
    for item in policy['pins'].values():
        require(G.file_pin(ROOT/item['path']) == {k:item[k] for k in ('bytes','sha256')}, 'Original frozen evidence changed during diagnosis')
    require(hashlib.sha256(policy_path.read_bytes()).hexdigest() == POLICY_SHA, 'New policy unchanged')
    fixed10 = {arm: sum(metrics[arm]['per_target'][t]['auc'] for t in G.FIXED10)/10 for arm in metrics}
    report = {'at_UTC':datetime.now(timezone.utc).isoformat(), 'complete':True, 'status':'NEW_SCOPE_EXPLORATORY_FIXED_E12_DIAGNOSTIC_ONLY',
              'policy_SHA':POLICY_SHA, 'source':G.file_pin(Path(__file__)), 'unchanged_guard_source':policy['pins']['old_guard'],
              'original_HOLD_untouched':True, 'original_engineering_passed':False, 'matched_counters':counters,
              'eligible_for_release_or_Submit_from_this_diagnostic':False, 'fold':1, 'UIDs':891, 'train_studies':3316,
              'epoch':12, 'fixed10_means':fixed10, 'fold1_fixed10_delta':result['fixed10_delta'],
              'fold0_existing_fixed10_delta':guard['fold0_positive_sha_bound']['fixed10_delta'],
              'bothfold_weak_deltas_positive_exploratory_only':result['fixed10_delta'] > 0 and guard['fold0_positive_sha_bound']['fixed10_delta'] > 0,
              'raw_original_result':result, 'separate_writer_metric_checks':metrics, 'input_guards':guard,
              'own_Gold_fit_select':0, 'model_deserialization_GPU_training_API_Save_Submit':0,
              'limitations':'Known post-hoc matched AMP pattern; repeated weak holdouts, upstreamgeneric pretraining and patient independence unverified; no originalgate pass or Publicaccuracy claim.'}
    with (HERE/'result.json').open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:report[k] for k in ('complete','fixed10_means','fold1_fixed10_delta','fold0_existing_fixed10_delta','bothfold_weak_deltas_positive_exploratory_only','eligible_for_release_or_Submit_from_this_diagnostic')}))


if __name__ == '__main__':
    main()
