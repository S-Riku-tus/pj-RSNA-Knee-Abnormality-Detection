"""実F016 E12/CPU資格後にP049のCLS2本だけをCNNへ替えるsourceを構築。通信なし。"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import io
import json
import lzma
import re
import zipfile
from pathlib import Path

import check_f016_fixed12 as A
import prepare_p049_native10 as B

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/kaggle/p049-f013-native10-20261022-v2'
PARENT_SHA = '465beec9a8ee9d7b013a1ca88545471ee749e6c7dfd970d140346b12f47ea5e0'


def check(value, message):
    if not value:
        raise ValueError(message)


def derive_native_sources(files):
    """実checkpointを開かずに、原routerの接続境界だけを検証する。"""
    child = files['p048_native_child.py'].decode()
    child = B.replace(child, 'import f013_p039_bridge\n    return f013_p039_bridge.execute(',
                      'import f016_cnn_serving_runtime\n    return f016_cnn_serving_runtime.execute(')
    child = B.replace(child, 'p048_f013_isolated_uid_adapter_v1', 'f016_cnn_isolated_uid_adapter_v1')
    extension = files['p048_parent_extension.py'].decode()
    extension = B.replace(extension, 'p048_f013_isolated_uid_adapter_v1', 'f016_cnn_isolated_uid_adapter_v1')
    extension = B.replace(extension, 'p049_F013_CLS_P039_REG_fixed10_execution_v1', 'p053_F016_CNN_P039_REG_fixed10_execution_v1')
    result = {}
    for name, text in [('p048_native_child.py', child), ('p048_parent_extension.py', extension)]:
        text = text.replace('F013_current_constructor_qualification', 'F016_current_constructor_qualification')
        text = text.replace('F013_bridge_policy_SHA', 'F016_scope_SHA')
        text = text.replace('F013_CLS_original_REG_native_rank', 'F016_CNN_original_REG_native_rank')
        if name == 'p048_parent_extension.py':
            text = B.replace(text, 'f14432a21fd6f19bcb776fadff374f7d2c88c8aee42ae63f50238192f5cb1540', A.SCOPE_SHA)
        compile(text, name, 'exec')
        result[name] = text.encode()
    return result


def prepare(package, cpu_evidence, dataset_slug, output):
    package, cpu_evidence, output = map(Path, (package, cpu_evidence, output))
    check(not output.exists() and not output.is_symlink() and re.fullmatch(r'f016-tiny-e12-[a-z0-9-]{5,35}', dataset_slug),
          '新規出力と本人F016 dataset名')
    manifest_raw = (package / 'package-manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    check(manifest['schema'] == 'F016_fixedE12_CNN_package_v1' and manifest['scope_SHA'] == A.SCOPE_SHA
          and manifest['alpha'] == .1 and manifest['family_weight'] == .5, '固定F016 package')
    cpu_raw = cpu_evidence.read_bytes()
    cpu = json.loads(cpu_raw)
    check(cpu['schema'] == 'F016_actual_fixedE12_CPU_qualification_v1' and cpu['passed'] is True
          and cpu['scope_SHA'] == A.SCOPE_SHA and cpu['manifest_SHA'] == B.pin(manifest_raw)['sha256']
          and cpu['audit_SHA'] == manifest['audit_SHA'] and set(cpu['folds']) == {'0', '1'}, '同じ実E12 packageのCPU資格')
    for fold in ('0', '1'):
        check(cpu['folds'][fold]['checkpoint_SHA'] == manifest['checkpoints'][fold]['sha256']
              and cpu['folds'][fold]['checks'] and all(v is True for v in cpu['folds'][fold]['checks'].values()), '両fold CPU全確認')
    raw = (PARENT / 'source.ipynb').read_bytes()
    check(B.pin(raw)['sha256'] == PARENT_SHA, '固定P049親')
    parent_manifest = json.loads((PARENT / 'manifest.json').read_bytes())
    archive = (PARENT / 'runtime-sources.bin').read_bytes()
    check(B.pin(archive) == parent_manifest['runtime_archive'], '親runtime全byte')
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        files = {n: bundle.read(n) for n in bundle.namelist()}
    check({n: B.pin(body) for n, body in files.items()} == parent_manifest['runtime_files'], '親内包全source')
    original = files.copy()
    files.update(derive_native_sources(files))
    binding = {'scope_SHA': A.SCOPE_SHA, 'fixed_epoch': 12, 'dataset_slug': dataset_slug,
               'manifest_SHA': B.pin(manifest_raw)['sha256'], 'audit_SHA': manifest['audit_SHA'],
               'CPU_evidence_SHA': B.pin(cpu_raw)['sha256'], 'alpha': .1, 'family_weight': .5}
    files['F016-package-binding.json'] = (json.dumps(binding, ensure_ascii=False, indent=2) + '\n').encode()
    files['F016-current-CPU-evidence.json'] = cpu_raw
    files['f016_cnn_serving_runtime.py'] = (ROOT / 'scripts/f016_cnn_serving_runtime.py').read_text(encoding='utf8').encode()
    check(files['P033-source.ipynb'] == original['P033-source.ipynb']
          and files['p032blend/fixed20_blend.py'] == original['p032blend/fixed20_blend.py']
          and files['p032blend/success20_blend.py'] == original['p032blend/success20_blend.py'], 'P033とnative10算術は変更しない')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for name, body in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_STORED, 0o100644 << 16
            bundle.writestr(info, body)
    new_archive = buffer.getvalue()
    packed = lzma.compress(new_archive, preset=6)
    inventory = {n: B.pin(body) for n, body in files.items()}
    book = json.loads(raw)
    setup = ast.parse(book['cells'][2]['source'])
    call = setup.body[-1].value
    call.args = [ast.parse(repr(v), mode='eval').body for v in
                 [base64.b85encode(packed).decode(), B.pin(packed), B.pin(new_archive), inventory, '/kaggle/working/P053_F016_RUNTIME']]
    book['cells'][2]['source'] = ast.unparse(ast.fix_missing_locations(setup)) + '\n'
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = cell['source'].replace('_p049', '_p053')
            compile(cell['source'], 'P053-F016-cell', 'exec')
    book['cells'][0]['source'] = ('# P053：P049 native10のCLS2本だけをF016 Tiny E12へ置換\n\n'
        '固定E12両fold・実F013家族対照/R6代理条件・実CPU復元を確認したpackageだけを受理。'
        '同REG・fold/家族50%・native10・前処理・失敗処理・P0338科学セルは同じ。現2T4の全8モデル人工資格はMRI前に実行。\n')
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + '\n').encode()
    check(len(raw) < 950000, 'source上限')
    payload = copy.deepcopy(json.loads((PARENT / 'save-payload.json').read_bytes()))
    payload.update(slug='rsraki/p053-f016-tiny-native10-20261022-v1', new_title='p053-f016-tiny-native10-20261022-v1', text=raw.decode())
    payload['dataset_data_sources'].append('rsraki/' + dataset_slug + '/1')
    result = {'schema': 'P053_F016_CNN_on_P049_native10_source_v1', 'source_SHA': B.pin(raw)['sha256'],
              'parent_source_SHA': PARENT_SHA, 'runtime_archive': B.pin(new_archive), 'runtime_files': inventory,
              'F016_binding': binding, 'P033_exact_eight_cell_SHA': parent_manifest['P033_exact_eight_cell_SHA'],
              'alpha': .1, 'family_weight': .5, 'actual_current_GPU_MRI_qualified': False,
              '外部upload_Save_Submit': 0, '比較範囲': 'F013 CLS2からCNN2への最終モデル置換。純粋なarchitecture単独効果とは呼ばない。'}
    output.mkdir(parents=True, exist_ok=False)
    for name, body in [('source.ipynb', raw), ('runtime-sources.bin', new_archive),
                       ('save-payload.json', (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()),
                       ('manifest.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())]:
        (output / name).write_bytes(body)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--cpu-evidence', type=Path, required=True)
    parser.add_argument('--dataset-slug', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False))
