"""採点済みP048 V4のnative比率だけを10%へ変更する。通信・Torchなし。"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
import lzma
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/kaggle/p048-f013-native20-20261021-v4'
PARENT_SHA = '05b8b300cc397ae61de643b7788601f3963da489fb366cb7530ddf6946b0ccca'
CHANGED = {'p032blend/fixed20_blend.py', 'p032blend/success20_blend.py', 'p048_parent_extension.py'}


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError('固定変更境界が異なる: ' + old)
    return source.replace(old, new)


def variant_files():
    raw = (PARENT / 'source.ipynb').read_bytes()
    if pin(raw)['sha256'] != PARENT_SHA:
        raise ValueError('正式P048のsource hashが異なる')
    manifest = json.loads((PARENT / 'manifest.json').read_bytes())
    archive = (PARENT / 'runtime-sources.bin').read_bytes()
    if pin(archive) != manifest['runtime_archive']:
        raise ValueError('正式P048のruntime hashが異なる')
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        original = {name: bundle.read(name) for name in bundle.namelist()}
    if {name: pin(body) for name, body in original.items()} != manifest['runtime_files']:
        raise ValueError('正式P048の内包runtimeが異なる')
    files = original.copy()
    for name in CHANGED:
        source = original[name].decode()
        if name.endswith('fixed20_blend.py'):
            for old, new, count in (
                ('f32(.8)', 'f32(.9)', 1), ('f32(.2)', 'f32(.1)', 1),
                ('0.8 * parent_s + 0.2 *', '0.9 * parent_s + 0.1 *', 1),
                ('alpha=0.2', 'alpha=0.1', 3), ('alpha=.2 (candidate)', 'alpha=.1 (candidate)', 1),
                ('alpha not in (0, 0.2)', 'alpha not in (0, 0.1)', 1),
                ('choices=(0.0, 0.2), default=0.2', 'choices=(0.0, 0.1), default=0.1', 1),
                ('fixed20_fullN_column_fallback_v1', 'P049_fixed10_fullN_column_fallback_v1', 1),
            ):
                source = replace(source, old, new, count)
        elif name.endswith('success20_blend.py'):
            source = replace(source, 'alpha=0.2', 'alpha=0.1', 2)
            source = replace(source, '0.8 * parent_s + 0.2 *', '0.9 * parent_s + 0.1 *')
            source = replace(source, 'fixed20_success_subset_cell_fallback_v1', 'P049_fixed10_success_subset_cell_fallback_v1')
        else:
            source = replace(source, 'numerical.get("alpha_requested") != 0.2', 'numerical.get("alpha_requested") != 0.1')
            source = replace(source, 'parent_factor, native_factor = (.8, .2)', 'parent_factor, native_factor = (.9, .1)')
            source = replace(source, 'alpha=0.2,', 'alpha=0.1,')
            source = replace(source, 'p048_F013_CLS_P039_REG_fixed20_execution_v1', 'p049_F013_CLS_P039_REG_fixed10_execution_v1')
            source = replace(source, 'no active fixed20 native cells', 'no active fixed10 native cells')
        compile(source, name, 'exec')
        files[name] = source.encode()
    if {name for name in files if files[name] != original[name]} != CHANGED:
        raise ValueError('比率以外のruntime変更')
    return json.loads(raw), files, original, manifest


def prepare(output, slug):
    output = Path(output)
    if output.exists() or output.is_symlink() or not re.fullmatch(r'rsraki/p049-[a-z0-9-]{5,43}', slug):
        raise ValueError('新規出力と本人P049 slugが必要')
    book, files, original, parent_manifest = variant_files()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for name, body in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, body)
    raw_archive = buffer.getvalue()
    packed = lzma.compress(raw_archive, preset=6)
    inventory = {name: pin(body) for name, body in files.items()}
    setup = ast.parse(book['cells'][2]['source'])
    # 元restorerの検証処理をそのまま維持し、archiveとinventoryだけを更新。
    call = setup.body[-1].value
    assert isinstance(call, ast.Call) and len(call.args) == 5
    values = [base64.b85encode(packed).decode(), pin(packed), pin(raw_archive), inventory,
              '/kaggle/working/P049_NATIVE_RUNTIME']
    call.args = [ast.parse(repr(value), mode='eval').body for value in values]
    book['cells'][2]['source'] = ast.unparse(ast.fix_missing_locations(setup)) + '\n'
    book['cells'][0]['source'] = (
        '# P049：P048 native20→10%だけを比較\n\n'
        '同じF013 CLS2・REG2・fold/家族50%・前処理・P033科学セル・Input/Docker。'
        '成功集合内tie順位、FP32別積和、MCL/Synovitisと失敗セルの親文字列を保持。'
        'weak代理診断はPublic改善や10%最適性の証明ではない。本人Gold選択0。\n')
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = cell['source'].replace('_p048', '_p049')
            compile(cell['source'], 'P049-cell', 'exec')
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + '\n').encode()
    if len(raw) >= 950000:
        raise ValueError('Notebook source上限超過')
    payload = copy.deepcopy(json.loads((PARENT / 'save-payload.json').read_bytes()))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=raw.decode())
    result = {'schema': 'P049_native10_only_source_v1', 'parent_source_SHA': PARENT_SHA,
              'source_SHA': pin(raw)['sha256'], 'native_alpha': 0.1, 'parent_factor': 0.9,
              'runtime_archive': pin(raw_archive), 'runtime_files': inventory,
              'changed_runtime_files': sorted(CHANGED),
              'unchanged_runtime_files': {name: pin(body) for name, body in original.items() if name not in CHANGED},
              'P033_exact_eight_cell_SHA': parent_manifest['P033_exact_eight_cell_SHA'],
              'all_Input_Docker_GPU_settings_unchanged': True, 'API_upload_Save_Submit': 0,
              'actual_current_GPU_MRI_qualified': False, 'own_Public_gain_verified': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, body in [('source.ipynb', raw), ('runtime-sources.bin', raw_archive),
                       ('save-payload.json', (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()),
                       ('manifest.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())]:
        (output / name).write_bytes(body)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.slug)
    print(json.dumps({key: result[key] for key in ('source_SHA', 'native_alpha', 'changed_runtime_files')}, ensure_ascii=False))
