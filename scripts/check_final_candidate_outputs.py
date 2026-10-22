"""今回の可視出力をsource/Input/モデル指紋と独立CSV算術で監査。通信なし。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import struct
import zipfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
ID = 'StudyInstanceUID'
EXCLUDED = ('MCL', 'Synovitis')


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    reader = csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'), newline=''))
    require(tuple(reader.fieldnames or ()) == (ID, *TARGETS), '13列の順序')
    result = list(reader)
    require(result and len({r[ID] for r in result}) == len(result), '一意UID')
    require(all(set(r) == {ID, *TARGETS} and r[ID] for r in result), 'CSV構造')
    for row in result:
        require(all(math.isfinite(float(row[t])) and 0 <= float(row[t]) <= 1 for t in TARGETS), '有限確率')
    return result


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def ranks(values):
    return [Fraction(2 * sum(x < v for x in values) + sum(x == v for x in values) + 1,
                     2 * len(values)) for v in values]


def mean2(a, b):
    return [f32(f32(f32(x) + f32(y)) * .5) for x, y in zip(a, b)]


def source_cells(book):
    return [c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            for c in book['cells'] if c['cell_type'] == 'code']


def audit(candidate, source, poll, output):
    source, poll, output = map(Path, (source, poll, output))
    require(not output.exists(), '新規監査出力')
    checks = {}
    is_native = candidate in ('P049', 'P051', 'P053_CNN')
    is_cnn = candidate == 'P053_CNN'

    def check(label, value):
        checks[label] = bool(value)
        require(checks[label], label)

    summary = read(poll / 'summary.json')
    check('今回の固定V1がCOMPLETE', summary['status_typed'] == 'COMPLETE' and summary['version'] == 1 and not summary['errors'])
    for item in summary['downloaded']:
        path = poll / 'files' / item['name']
        check('取得hash:' + item['name'], item['HTTP'] == 200 and path.stat().st_size == item['bytes'] and sha(path) == item['SHA'])
    manifest = read(source / 'manifest.json')
    check('source_hash', sha(source / 'source.ipynb') == manifest['source_SHA'])
    saved = read(poll / 'saved-source.json')
    metadata = saved['metadata']
    payload = read(source / 'save-payload.json')
    check('保存した全科学セル', source_cells(json.loads(saved['blob']['source'])) == source_cells(read(source / 'source.ipynb')))
    check('本人kernel固定V1', metadata['author'] == 'rsraki' and metadata['id'] == summary['kernelId']
          and metadata['currentVersionNumber'] == 1 and metadata['ref'] == payload['slug'])
    for remote, requested in [('datasetDataSources', 'dataset_data_sources'), ('kernelDataSources', 'kernel_data_sources'),
                              ('competitionDataSources', 'competition_data_sources')]:
        expected = [v.rsplit('/', 1)[0] if '/' in v and v.rsplit('/', 1)[1].isdigit() else v for v in payload[requested]]
        check(remote + '_要求版と正規化ラベル', sorted(metadata[remote]) == sorted(expected))
    check('private_offline_T4_Docker', metadata['isPrivate'] is True and metadata['enableGpu'] is True
          and metadata['enableInternet'] is False and not metadata['enableTpu']
          and metadata['machineShape'] == payload['machine_shape'] and metadata['dockerImage'] == payload['docker_image'])
    files = poll / 'files'
    submitted_final = rows(files / 'submission.csv')
    native_final_path = files / ('P032/fixed20/submission.csv' if candidate == 'P051' else 'submission.csv')
    final = rows(native_final_path)
    order = [r[ID] for r in submitted_final]
    check('可視3検査の提出契約', len(order) == 3)
    blend, controller = read(files / 'blend_receipt.json'), read(files / 'submission_receipt.json')
    wr = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35}
    a224 = .35
    if candidate == 'P050':
        wr.update({'Lateral OA': .25, 'PF OA': .55, 'Synovitis': .75, "Baker's": .4})
    elif candidate == 'P052':
        a224 = .2
    elif candidate == 'P053':
        a224 = .5
    check('事前固定係数', blend['weights'] == {'raptor': .5, 'ours': .5, 'raptor_per_class': wr, 'a224': a224})
    check('親3枝の今回の成功', all(blend[k] == 0 for k in ('raptor_failed', 'raptor384_failed', 'raptor224_failed', 'ours_failed', 'both_failed'))
          and controller['status'] == 'PASS' and controller['rows'] == controller['scored'] == len(order)
          and controller['fallback_studies'] == 0 and controller['worker_exit_codes'] == {'0': 0, '1': 0}
          and controller['bundle_id'] == 'fe055406fad6a12f')
    parent = rows(files / 'P032/parent-submission.csv') if is_native else final
    arms = [rows(files / name) for name in ('raptor_raw.csv', 'raptor224_raw.csv', 'ours_sub.csv')]
    check('全CSVのUID順一致', all([r[ID] for r in arm] == order for arm in [parent, *arms]))
    for target in TARGETS:
        ranked = [[float(x) for x in ranks([float(r[target]) for r in arm])] for arm in arms]
        weight = wr.get(target, .5)
        expected = [f32(weight * ((1 - a224) * ranked[0][i] + a224 * ranked[1][i]) + (1 - weight) * ranked[2][i])
                    for i in range(len(order))]
        check('独立親FP32:' + target, [f32(float(r[target])) for r in parent] == expected)
    if is_native:
        n, a, e, w, projection, numerical = [read(files / name) for name in
            ('P032/native-result.json', 'P032/adapter-receipt.json', 'P032/execution-receipt.json',
             'P032/child-watchdog.json', 'P032/isolated-uid-projection/projection-receipt.json', 'P032/fixed20/receipt.json')]
        native = rows(files / 'P032/isolated-uid-projection/native-mean.csv')
        coverage = {'N': 3, 'S': 3, 'F_structural': 0, 'F_operational': 0}
        check('今回native全成功', n['coverage'] == a['coverage'] == projection['coverage'] == coverage
              and not n['structural_failures'] and not n['operational_failures'] and not n['failure_UID_union']
              and n['successful_UID_model_fallback_count'] == n['operational_UID_fallback_count'] == 0)
        check('今回native_完了・採用あり', n['complete'] and n['startup_complete']
              and n['F016_current_constructor_qualification' if is_cnn else 'F013_current_constructor_qualification']
              and a['complete'] and e['complete'] and e['native_complete'] and e['native_contribution_used'] and not e['all_native_disabled'])
        check('子process回収', w['complete'] and w['joined'] and not w['timeout'] and w['returncode'] == 0 and w['owned_process_group_empty'])
        check('native_receipt結合', e['native_result_sha256'] == a['native_result_sha256'] == sha(files / 'P032/native-result.json')
              and e['native_adapter_receipt_sha256'] == sha(files / 'P032/adapter-receipt.json') and e['child_watchdog'] == w)
        old = read(Path(__file__).resolve().parents[1] / 'artifacts/research/20261021-P048-v4-root-visible-execution-v1/poll-20261021T035729991215Z/files/P032/native-result.json')
        component = n['F016_current_model_startup_component' if is_cnn else 'F013_current_model_startup_component']['reconstruction_component']
        fingerprints = n['reconstruction_component']['model_fingerprints']
        check('4モデル全指紋の今回receipt結合', fingerprints == a['model_fingerprints'] == component['model_fingerprints'])
        if is_cnn:
            archive = source / 'runtime-sources.bin'
            check('CNN内包source archive', sha(archive) == manifest['runtime_archive']['sha256'] and archive.stat().st_size == manifest['runtime_archive']['bytes'])
            with zipfile.ZipFile(archive) as bundle:
                cpu_raw = bundle.read('F016-current-CPU-evidence.json')
            binding = manifest['F016_binding']
            cpu = json.loads(cpu_raw)
            check('実CNN_E12_CPU資格の固定', hashlib.sha256(cpu_raw).hexdigest() == binding['CPU_evidence_SHA']
                  and cpu['schema'] == 'F016_actual_fixedE12_CPU_qualification_v1' and cpu['passed'] is True
                  and cpu['manifest_SHA'] == binding['manifest_SHA'] and cpu['scope_SHA'] == binding['scope_SHA'])
            for fold in ('0', '1'):
                ref, actual = cpu['folds'][fold], fingerprints['CLS:' + fold]
                check('CNN厳密E12指紋:' + fold, ref['checks'] and all(v is True for v in ref['checks'].values())
                      and actual['model_class'] == 'cnn_f016_source16.convnext_model.ConvNeXtTripletMIL'
                      and actual['selected_epoch'] == 12 and actual['checkpoint_sha256'] == ref['checkpoint_SHA']
                      and actual['package_manifest_SHA'] == binding['manifest_SHA']
                      and component['full_CPU_state_sha']['CLS:' + fold] == ref['full_state_SHA'])
                check('REG指紋不変:' + fold, fingerprints['REG:' + fold] == old['reconstruction_component']['model_fingerprints']['REG:' + fold])
            check('CNN全tensorがMRI後も不変', n['all_current_tensors_unchanged_after_MRI'] is True
                  and n['F016_scope_SHA'] == binding['scope_SHA'] and n['F016_package_manifest_SHA'] == binding['manifest_SHA'])
            check('CNNを含む今回のconstructor記録', n['generic_factory_calls'] == component['generic_factory_calls'] == 3
                  and n['generic_factory_calls_by_role'] == component['generic_factory_calls_by_role']
                  == {'original_DINO': 1, 'F016_CNN': 2}
                  and n['model_classes'] == component['model_classes']
                  and n['model_classes']['CLS'] == 'cnn_f016_source16.convnext_model.ConvNeXtTripletMIL')
        else:
            check('4モデル全指紋が正式P048と同一', fingerprints == old['reconstruction_component']['model_fingerprints'])
        gpu = component['current_GPU_synthetic']
        gpu_keys = (('finite_all_four_current_models', 'padding_exact', 'allmissing_refused', 'no_gamma', 'original_consumer_FP32_exact')
                    if is_cnn else ('finite_all_four_current_models', 'padding_exact', 'new_CLS_gamma_disabled', 'original_consumer_independent_FP32_mean_exact'))
        check('今回の2T4_8モデル人工確認', n['GPU_storage_alias_pairs_checked'] == 12 and n['GPU_storage_alias_count'] == 0
              and gpu['complete'] and set(gpu['devices']) == {'cuda:0', 'cuda:1'}
              and all(all(v[k] is True for k in gpu_keys) for v in gpu['devices'].values()))
        check('native_UID順序', [r[ID] for r in native] == order == [r[ID] for r in n['probabilities_test_order']])
        for i, uid in enumerate(order):
            record = n['perUID_receipts'][uid]
            check('実4forward:' + uid, record['family_order'] == ['CLS', 'REG'] and record['fold_order'] == [0, 1]
                  and record['logical_full_fold_model_forward_calls'] == 4 and not record['fallback']
                  and record['native_SHA_verified_before_model'] and record['same_native_array_SHA_after_model'])
            means = []
            for family in ('CLS', 'REG'):
                child = record['families'][family]
                expected = mean2(*child['fold_probabilities'])
                check('独立fold平均:' + family + ':' + uid, child['fold_order'] == [0, 1]
                      and child['logical_full_fold_model_forward_calls'] == 2 and not child['fallback']
                      and child['array_sha256'] == record['array_sha256'] and record['family_mean_probabilities'][family] == expected)
                means.append(expected)
            expected = mean2(*means)
            check('独立家族平均:' + uid, expected == n['probabilities_test_order'][i]['probabilities']
                  == [float(native[i][t]) for t in TARGETS] and native[i][ID] == uid)
        check('論理forward総数', n['full_fold_model_forward_calls'] == projection['logical_full_fold_model_forward_calls'] == 12)
        for target in TARGETS:
            ranked = ranks([f32(float(row[target])) for row in native])
            expected = [parent[i][target] if target in EXCLUDED else
                        repr(f32(f32(f32(.9) * f32(float(parent[i][target]))) + f32(f32(.1) * f32(rank))))
                        for i, rank in enumerate(ranked)]
            check('独立最終文字列:' + target, [r[target] for r in final] == expected)
        check('数値契約receipt', e['numeric_blend'] == numerical and numerical['alpha_requested'] == .1
              and '0.9 * parent_s + 0.1 *' in numerical['formula'] and numerical['excluded_targets'] == list(EXCLUDED)
              and not numerical['arithmetic']['parent_reranked'] and not numerical['arithmetic']['output_reranked']
              and e['submission_csv_sha256'] == numerical['output']['sha256'] == sha(native_final_path)
              and native_final_path.read_bytes() == (files / 'P032/fixed20/submission.csv').read_bytes()
              and (files / 'P032/fixed20/receipt.sha256').read_text(encoding='ascii') == sha(files / 'P032/fixed20/receipt.json') + '  receipt.json\n')
        check('Report_Gold不使用・旧HOLD維持', not n['accuracy_evaluated'] and not n['teacher_Report_read'] and not n['own_Gold_fit_or_selection']
              and n['old_F009_HOLD_unchanged']
              and not n['old_CLS_receipt_certifies_new_CNN' if is_cnn else 'old_F006_CPU_receipt_certifies_new_CLS'])
        check('旧HOLD_file不変', sha(Path(__file__).resolve().parents[1] / 'experiments/F009-retry-fixed12-candidate-AMP-HOLD-v2-20261019.json')
              == '87effd8b40180a7554ddd32a1146c21e7c5919324427aacfe09f8a549d46926f')
        if candidate == 'P051':
            from p040_p041_factorial_runtime import READER_WEIGHTS
            reader_receipt = read(files / 'P051/execution.json')
            own = rows(files / '_own.csv')
            own_by = {row[ID]: row for row in own}
            check('Reader親は今回のnative10', native_final_path.read_bytes() == (files / 'P051/native10-parent/submission.csv').read_bytes()
                  == (files / '_public_stack.csv').read_bytes())
            check('Readerの今回の成功記録', reader_receipt['complete'] is True and reader_receipt['Reader_positive_completion'] is True
                  and reader_receipt['native_before_Reader'] == e and not reader_receipt['own_Gold_fit_select'])
            marker = (files / 'P051-reader.stdout.txt').read_text(encoding='utf8')
            check('Reader3本の正の完了marker', 'own leg: blended 3 checkpoints' in marker
                  and 'own leg FAILED' not in marker and 'Part B skipped' not in marker and set(own_by) == set(order))
            for target in TARGETS:
                left = [float(v) for v in ranks([float(row[target]) for row in final])]
                right = [float(v) for v in ranks([float(own_by[uid][target]) for uid in order])]
                weight = READER_WEIGHTS[target]
                expected = [float(v) for v in ranks([(1 - weight) * a + weight * b for a, b in zip(left, right)])]
                check('独立Reader_FP64_最終順位:' + target, [float(row[target]) for row in submitted_final] == expected)
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'candidate': candidate, 'passed': all(checks.values()),
              'checks': checks, 'kernelId': summary['kernelId'], 'version': 1, 'source_SHA': manifest['source_SHA'],
              'submission_SHA': sha(files / 'submission.csv'), 'visible_studies': len(order),
              'scope_JA': '今回の可視実行・source/Inputラベルと指定版・算術の確認。hidden全件完走やPublic改善の証明ではない。',
              'Input_version_limit_JA': 'Save要求は版固定。GetのInputラベルは正規化されるため実内容guardと区別。',
              'Save_Submit_upload_GPU_MRI_Gold': 0}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=('P049', 'P050', 'P051', 'P052', 'P053', 'P053_CNN'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--poll', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(**vars(args))
    print(json.dumps({k: result[k] for k in ('candidate', 'passed', 'kernelId', 'submission_SHA')}, ensure_ascii=False))
