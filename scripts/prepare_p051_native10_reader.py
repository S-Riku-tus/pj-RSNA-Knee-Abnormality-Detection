"""P049 native10へ、既存P041のReaderを最後にそのまま追加。ローカル生成のみ。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'artifacts/kaggle/p049-f013-native10-20261022-v2'
READER = ROOT / 'artifacts/kaggle/p041-p033-reader-20261019-v1'
NATIVE_SHA = '465beec9a8ee9d7b013a1ca88545471ee749e6c7dfd970d140346b12f47ea5e0'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code(book):
    return [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code']


def prepare(output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('新規出力が必要')
    raw = (NATIVE / 'source.ipynb').read_bytes()
    assert sha(raw) == NATIVE_SHA
    native = json.loads(raw)
    reader = json.loads((READER / 'source.ipynb').read_bytes())
    reader_manifest = json.loads((READER / 'manifest.json').read_bytes())
    assert sha((READER / 'source.ipynb').read_bytes()) == reader_manifest['source']['sha256']
    science = json.loads((READER / 'science-cells.json').read_bytes())
    overlay = science[-1]
    assert len(science) == 9 and sha(overlay.encode()) == reader_manifest['science_inventory'][-1]['sha256']
    # 既存preflightそのものを今回も実行。過去の資格receiptだけで通さない。
    preflight = code(reader)[0]
    wrapper = (
        'import contextlib as _p051_context,io as _p051_io,json as _p051_json\n'
        'from pathlib import Path as _p051_Path\n'
        "_p051_work=_p051_Path('/kaggle/working')\n"
        "_p051_out=_p051_work/'P051';_p051_out.mkdir(exist_ok=False)\n"
        "_p051_before=_factor_ns['snapshot'](_p051_work,_p051_out/'native10-parent')\n"
        "_p051_parent=_factor_ns['parent_evidence'](_p051_work,globals())\n"
        f'_p051_reader={overlay!r}\n'
        f'assert _factor_hash.sha256(_p051_reader.encode()).hexdigest()=={sha(overlay.encode())!r}\n'
        '_p051_stream=_p051_io.StringIO()\n'
        'with _p051_context.redirect_stdout(_p051_stream),_p051_context.redirect_stderr(_p051_stream):\n'
        '    _p051_run=get_ipython().run_cell(_p051_reader,store_history=False)\n'
        '_p051_log=_p051_stream.getvalue()\n'
        "(_p051_work/'P051-reader.stdout.txt').write_text(_p051_log,encoding='utf8')\n"
        'print(_p051_log)\n'
        '_p051_error=_p051_run.error_before_exec or _p051_run.error_in_exec\n'
        'if _p051_error is not None: raise _p051_error\n'
        "assert 'own leg: blended 3 checkpoints' in _p051_log and 'own leg FAILED' not in _p051_log and 'Part B skipped' not in _p051_log\n"
        "assert OWN_MODE=='end' and OWN_P33==0.0 and OWN_W_PER_TARGET==_factor_ns['READER_WEIGHTS']\n"
        "assert sorted(_p051_Path(p).name for p in _o_ck)==['cnxt_v0_fold0.pt','cnxt_v0_fold1.pt','cnxt_v0_fold2.pt']\n"
        "_p051_math=_factor_ns['verify_reader_final'](_p051_out/'native10-parent/submission.csv',_p051_work/'_own.csv',_p051_work/'submission.csv')\n"
        "assert _factor_ns['body_pin'](_p051_work/'_public_stack.csv')==_factor_ns['body_pin'](_p051_out/'native10-parent/submission.csv')\n"
        "_p051_receipt={'candidate':'P051','complete':True,'parent':'P049 native10','native_before_Reader':_p049_result,'Reader':_p051_math,'Reader_positive_completion':True,'own_Gold_fit_select':False}\n"
        "_factor_ns['save'](_p051_out/'execution.json',_p051_receipt)\n"
    )
    cells = [preflight, *code(native), wrapper]
    assert len(cells) == 6
    for text in cells:
        compile(text, 'P051-cell', 'exec')
    book = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source':
                      '# P051：native10の後に固定P041 Readerだけを追加\n\n'
                      'P049が公式.957以下なら優先から外す。係数探索・追加gamma・224反転なし。'
                      'Readerは親と最終を再順位化する既存end処理をそのまま使用。本人Gold選択0。\n'},
                     *[{'cell_type': 'code', 'metadata': {}, 'source': s, 'execution_count': None, 'outputs': []} for s in cells]],
            'metadata': copy.deepcopy(native['metadata']), 'nbformat': 4, 'nbformat_minor': 5}
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + '\n').encode()
    assert len(raw) < 950000
    payload = json.loads((NATIVE / 'save-payload.json').read_bytes())
    reader_payload = json.loads((READER / 'save-payload.json').read_bytes())
    payload.update(slug='rsraki/p051-native10-reader-20261022-v1', new_title='p051-native10-reader-20261022-v1', text=raw.decode())
    payload['dataset_data_sources'] = list(dict.fromkeys([*payload['dataset_data_sources'], *reader_payload['dataset_data_sources']]))
    payload['kernel_data_sources'] = list(dict.fromkeys([*payload['kernel_data_sources'], *reader_payload['kernel_data_sources']]))
    assert payload['docker_image'] == reader_payload['docker_image']
    manifest = {'schema': 'P051_exact_Reader_after_P049_source_v1', 'source_SHA': sha(raw), 'parent_source_SHA': NATIVE_SHA,
                'P049_four_cells_byte_exact': True, 'Reader_SHA': sha(overlay.encode()), 'P041_preflight_cell_SHA': sha(preflight.encode()),
                'alpha': .1, 'Reader': '元P041そのまま、end・3checkpoint・所見別係数・親/最終再順位化を固定',
                '候補順条件': 'P049公式.957以下なら優先から外す。未返却はゼロと扱わない。',
                'Save_Submit_upload': 0, 'current_GPU_MRI未確認': True, 'Public改善未確認': True}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb', raw), ('save-payload.json', (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode()),
                        ('manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())]:
        (output / name).write_bytes(value)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False))
