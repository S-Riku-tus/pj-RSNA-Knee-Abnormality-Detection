"""F016新scopeの固定E12両foldとF013家族/R6代理を監査。通信/Torchなし。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import check_f014_fixed12_weak as C

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'artifacts/research/20261022-F016-tiny-fresh-twofold-v1/policy.json'
SCOPE_SHA = '9d2c5ebba900ebdb274ef57452e90b2ff069d87cda21bf9bfbe2692b44f05f19'
FAMILY = 'artifacts/research/20261020-P048-native50-paired-weak-prospective-v1/policy.json'
FAMILY_SHA = '0f24b3625e4d555795cadfe4409ba81087d2c5dcdec5c5f17775993173c1842f'
R6_POLL = ROOT / 'artifacts/research/20261020-R6-pair-probe-root-execution-v1/poll-20261020T122939651701Z'


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def mean2(a, b):
    return f32(f32(f32(a) + f32(b)) * .5)


def rank(values):
    # 既存proxyと同じ保存値のtie。順位比をFP32化。native家族値は既にFP32。
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[begin]] == values[order[end]]:
            end += 1
        value = f32((begin + end + 1) / (2 * len(values)))
        for i in order[begin:end]:
            result[i] = value
        begin = end
    return result


def family_rows(left, right):
    C.require(C.unique_ids(left) == C.unique_ids(right), '家族の同一除外fold/UID順')
    return [{C.ID: a[C.ID], **{t: str(mean2(float(a[t]), float(b[t]))) for t in C.TARGETS}}
            for a, b in zip(left, right)]


def audit(output):
    output = Path(output).resolve()
    C.require(not output.exists() and output.is_relative_to((ROOT / 'artifacts/research').resolve()), '新規研究出力')
    snapshot = C.Snapshot(ROOT)
    scope_raw = SCOPE.read_bytes()
    C.require(hashlib.sha256(scope_raw).hexdigest() == SCOPE_SHA, '精度前のF016予算/採否policy')
    scope = json.loads(scope_raw)
    policy = copy.deepcopy(C.load_policy(snapshot))
    # 原auditorの旧runと順序policyは不変。このscope内の候補ディレクトリだけを明示的に差替え。
    policy['candidate']['directories'] = {str(f): scope['runs'][f] for f in (0, 1)}
    metadata, fold_audit, input_hashes = C.verify_inputs(snapshot, policy)
    runs = {str(f): {role: C.verify_run(snapshot, policy, f, role, fold_audit, input_hashes)
                     for role in ('control', 'candidate')} for f in (0, 1)}
    family_raw = snapshot.raw(FAMILY)
    C.require(hashlib.sha256(family_raw).hexdigest() == FAMILY_SHA, '固定F013/REG対照policy')
    family = json.loads(family_raw)
    for ref in family['pins'].values():
        snapshot.pin(ref['path'], {k: ref[k] for k in ('bytes', 'sha256')})
    for fold in (0, 1):
        for role in ('candidate', 'REG'):
            ref = family['inputs'][str(fold)][role]
            for name in ('run', 'prediction', 'checkpoint'):
                snapshot.pin(ref[name]['path'], {k: ref[name][k] for k in ('bytes', 'sha256')})
    # 新規両foldの全12epoch、source、counter、旧対照hashを満たしてから教師数値を読む。
    header, cells = C.csv_cells(snapshot.raw(policy['manifest_directory'] + '/weak.csv'))
    weak = [dict(zip(header, row)) for row in cells]
    C.require([{k: r[k] for k in (C.ID, 'source', 'group_id', 'fold')} for r in weak] == metadata, '弱教師metadata不変')
    for row in weak:
        C.require(any(C.probability(row[t], missing=True) is not None for t in C.TARGETS), '全欠損の学習対象拒否')
        for target in C.TARGETS:
            C.probability(row[target], missing=True)
    sys.path.insert(0, str(ROOT / 'src'))
    import evaluate_r6_overall_probe as O
    bank = O.audit(R6_POLL, 352)
    proxy_ref = 'artifacts/research/20261022-native-r6-proxy-v1/root-reproduction-v1/proxy.json'
    raw = snapshot.raw(proxy_ref)
    C.require(hashlib.sha256(raw).hexdigest() == 'e37f99f675cead2b05e5dbadf6dc4bfa727dfc5d4b4d7b338e866bbd48b6d049',
              '既存代理診断の正本')
    previous_proxy = json.loads(raw)
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'schema': 'F016_fixedE12_current_family_proxy_audit_v1',
              'scope_SHA': SCOPE_SHA, 'fixed_epoch': 12, 'alpha': .1, 'family_weight': .5, 'folds': {},
              '旧F014順序policy通過の主張': False, 'Gold数値利用': False, 'serving資格': False, 'Public改善証明': False}
    for fold in (0, 1):
        key = str(fold)
        C.verify_teacher_coverage(weak, fold, fold_audit)
        truth = [r for r in weak if r['fold'] == key]
        ids = C.unique_ids(truth)
        predictions, evaluated = {}, {}
        for role, run in runs[key].items():
            predictions[role] = C.prediction_rows(snapshot.raw(run['directory'] + '/epoch_012_weak.csv'), ids)
            evaluated[role] = C.evaluate_weak(truth, predictions[role])
            C.verify_writer(evaluated[role], run['E12'], policy['tolerances'])
        for role, name in [('F013', 'candidate'), ('REG', 'REG')]:
            ref = family['inputs'][key][name]['prediction']
            predictions[role] = C.prediction_rows(snapshot.raw(ref['path'], {k: ref[k] for k in ('bytes', 'sha256')}), ids)
            evaluated[role] = C.evaluate_weak(truth, predictions[role])
        for role in ('control', 'candidate', 'F013'):
            name = role + '_REG50'
            predictions[name] = family_rows(predictions[role], predictions['REG'])
            evaluated[name] = C.evaluate_weak(truth, predictions[name])
        primary = C.primary_delta(evaluated['control'], evaluated['candidate'])
        family_delta = C.primary_delta(evaluated['F013_REG50'], evaluated['candidate_REG50'])
        baseline_family_delta = C.primary_delta(evaluated['control_REG50'], evaluated['candidate_REG50'])
        proxy_ids = sorted(u for u, row in bank['selection'].items() if row['weak_fold'] == key and u in bank['success'])
        C.require(len(proxy_ids) == (809 if fold == 0 else 855) and set(proxy_ids) <= set(ids), '固定809/855検査')
        truth_keyed = {r[C.ID]: r for r in truth}
        proxy = {}
        for role in ('F013_REG50', 'candidate_REG50'):
            predicted = {r[C.ID]: r for r in predictions[role]}
            values = {}
            for target in C.FIXED10:
                parent_rank = rank([float(bank['predictions']['baseline'][u][target]) for u in proxy_ids])
                native_rank = rank([float(predicted[u][target]) for u in proxy_ids])
                scores = [f32(f32(f32(.9) * a) + f32(f32(.1) * b)) for a, b in zip(parent_rank, native_rank)]
                observed = [i for i, u in enumerate(proxy_ids) if C.probability(truth_keyed[u][target], missing=True) in (0, 1)]
                labels = [C.probability(truth_keyed[proxy_ids[i]][target]) for i in observed]
                values[target] = C.binary_auc(labels, [scores[i] for i in observed])
            proxy[role] = {'fixed10': math.fsum(values.values()) / 10, 'per_target': values}
        proxy_delta = proxy['candidate_REG50']['fixed10'] - proxy['F013_REG50']['fixed10']
        C.require(proxy['F013_REG50']['fixed10'] == previous_proxy['banks'][key]['rank_proxy_auc']['FP32']['gamma50']['0.1'],
                  '元F013+REGの同一proxy算術を再現')
        result['folds'][key] = {'primary': primary, 'family_vs_F006_REG': baseline_family_delta,
                               'family_vs_actual_F013_REG': family_delta, 'proxy': proxy, 'proxy_delta': proxy_delta,
                               'all12': evaluated, 'paired_UIDs': len(ids), 'proxy_UIDs': len(proxy_ids),
                               '候補条件': primary['strictly_positive'] and family_delta['strictly_positive'] and proxy_delta >= 0,
                               '固定E12_checkpoint': {role: run['checkpoint'] for role, run in runs[key].items()},
                               '実更新記録': {role: run['counters'] for role, run in runs[key].items()}}
    snapshot.recheck()
    C.require('torch' not in sys.modules, 'CSV監査にTorchを持ち込まない')
    result['両fold候補条件'] = all(value['候補条件'] for value in result['folds'].values())
    result['accessed_artifacts'] = snapshot.accessed
    result['解釈'] = '再利用weak・単一除外fold R6代理であり、Raptor全親OOF/実2fold推論/Public増分/患者独立性の証明ではない。'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({'両fold候補条件': result['両fold候補条件']}, ensure_ascii=False))
