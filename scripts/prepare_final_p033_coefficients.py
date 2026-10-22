"""P033の事前固定係数だけを変え、新規Notebook/payloadをローカル生成する。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/kaggle/p033-p029m-public958-weights-20261016-v1'
PARENT_SHA = '57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb'
VECTOR = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
REVERSE = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.25, 'PF OA': 0.55, 'Synovitis': 0.75, \"Baker's\": 0.4}"
SPECS = {
    'P050': ('p050-p033-vector-reverse-20261022-v1', 'P033所見別ベクトルの事前固定反対方向を一度比較', None),
    'P052': ('p052-p033-a224-low-20261022-v1', 'P033 A224=.35→.20だけを比較', .2),
    'P053': ('p053-p033-a224-high-20261022-v1', 'CNN予約枠を使わない場合のP033 A224=.35→.50比較', .5),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(cell):
    value = cell['source']
    return value if isinstance(value, str) else ''.join(value)


def prepare(candidate, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('新規runが必要')
    raw = (PARENT / 'source.ipynb').read_bytes()
    if sha(raw) != PARENT_SHA:
        raise ValueError('正式P033 sourceが異なる')
    book = json.loads(raw)
    original = [source(c) for c in book['cells'] if c['cell_type'] == 'code']
    slug, reason, a224 = SPECS[candidate]
    affected = []
    for cell in book['cells']:
        if cell['cell_type'] != 'code' or VECTOR not in source(cell):
            continue
        text = source(cell)
        assert text.count(VECTOR) == text.count('A224 = 0.35   #') == 1
        if candidate == 'P050':
            text = text.replace(VECTOR, REVERSE)
        else:
            text = text.replace('A224 = 0.35   #', f'A224 = {a224:.2f}   #')
            text = text.replace('A224 = 0.35 fixed before any run', f'A224 = {a224:.2f} fixed before any run')
        cell['source'] = text
        affected.append(True)
    updated = [source(c) for c in book['cells'] if c['cell_type'] == 'code']
    assert len(affected) == 1 and len(updated) == len(original) == 8
    differences = [i for i, (old, new) in enumerate(zip(original, updated)) if old != new]
    assert differences == [7]
    for text in updated:
        body = text.split('\n', 1)[1] if text.startswith('%%writefile ') else text
        compile(body, candidate + '-cell', 'exec')
    book['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source':
        f'# {candidate}：{reason}\n\n元P033の7科学セルと全モデル/Input/Dockerを保持。'
        '係数変更にはnative10案と同じweak根拠はなく、Public改善は未確認。本人Goldによる選択0。\n'})
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + '\n').encode()
    payload = copy.deepcopy(json.loads((PARENT / 'fixed-input-payload.json').read_bytes()))
    # 元資料に付随したローカル照合欄はSDK requestのfieldではない。
    payload.pop('source_sha256')
    payload.pop('source_file')
    payload.update(slug='rsraki/' + slug, new_title=slug, text=raw.decode())
    manifest = {'schema': 'final_P033_single_coefficient_factor_v1', 'candidate': candidate,
                'reason_JA': reason, 'parent_source_SHA': PARENT_SHA, 'source_SHA': sha(raw),
                'parent_code_SHA': [sha(s.encode()) for s in original],
                'candidate_code_SHA': [sha(s.encode()) for s in updated],
                'changed_code_cell_indices': differences, 'all_Input_Docker_GPU_settings_unchanged': True,
                'native_Reader_R7_added': False, 'API_upload_Save_Submit': 0,
                'actual_current_GPU_MRI_qualified': False, 'own_Public_gain_verified': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, body in [('source.ipynb', raw),
                       ('save-payload.json', (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()),
                       ('manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())]:
        (output / name).write_bytes(body)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', choices=SPECS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.candidate, args.output)
    print(json.dumps(result, ensure_ascii=False))
