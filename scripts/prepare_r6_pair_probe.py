"""Build inert private Kaggle validation payload; no upload or submission calls."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path

import prepare_p040_p041_factorial as F

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'artifacts/research/20261020-R6-paired-weak-probe-prospective-v1/policy.json'
POLICY_SHA = '579b4bc490d27f46898cc75266478a59ddf73356b8952b8fa2f3a14e99945fa8'
SELECTION_SHA = '99f3217a869ecd3a3e91450996907e8045bb37d81ca071943237b4db28c5420b'


def prepare(output, slug):
    output = Path(output)
    F.require(not output.exists() and not output.is_symlink(), 'Fresh artifact directory required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own slug required')
    F.require(F.pin(POLICY)['sha256'] == POLICY_SHA, 'Prospective policy differs')
    policy = F.read(POLICY)
    selection = (ROOT / policy['selection']['path']).read_bytes()
    F.require(hashlib.sha256(selection).hexdigest() == SELECTION_SHA, 'Selection differs')
    F.require(F.pin(F.P033 / 'source.ipynb')['sha256'] == F.P033_SHA, 'Frozen parent differs')
    parent = F.read(F.P033 / 'source.ipynb')
    science = [F.text(c) for c in parent['cells'] if c['cell_type'] == 'code']
    F.require(len(science) == 8, 'Frozen eight-cell parent required')
    frozen = [science[0], science[5]]
    for value in frozen:
        ast.parse(value)
    runtime, arithmetic = [ (ROOT / ('scripts/' + name)).read_text(encoding='utf8')
                            for name in ('r6_pair_probe_runtime.py', 'r6_pair_math.py')]
    for value in (runtime, arithmetic):
        ast.parse(value)
    packed = base64.b85encode(zlib.compress(selection, 9)).decode()
    loader = ("import time\nNB_T0=time.time()\nimport hashlib,base64,zlib\n"
              f"_pair_runtime={runtime!r}\n"
              f"assert hashlib.sha256(_pair_runtime.encode()).hexdigest()=={hashlib.sha256(runtime.encode()).hexdigest()!r}\n"
              "_pair_ns={'__name__':'r6_pair_probe_runtime'}\n"
              "exec(compile(_pair_runtime,'<frozen-paired-runtime>','exec'),_pair_ns)\n")
    invoke = (f"_pair_selection=zlib.decompress(base64.b85decode({packed!r})).decode()\n"
              f"_pair_math={arithmetic!r}\n"
              f"_pair_result=_pair_ns['run_probe'](globals(),_pair_selection,{SELECTION_SHA!r},"
              f"{POLICY_SHA!r},_pair_math,{hashlib.sha256(arithmetic.encode()).hexdigest()!r})\n")
    notice = ('# R6 paired weak validation probe\n\n'
              'Validation only; never submit this notebook. No root submission.csv. Each selected study uses only '
              'its declared heldout R6 fold checkpoint. Two fixed weak banks, TTA weights 0/.25/.5 and original '
              'postprocessor removal are compared without fitting, labels, or Reports in this runtime. Synthetic '
              'GPU forward and direct OFF/replay parity precede MRI decode. Original source, preprocessing and '
              'model SHA guards are preserved. Actual upstream fold membership, auxiliary pretraining and '
              'Gold-informed selection are not independently certified. No patient independence or clean whole-model '
              'OOF claim. Labels and numeric scoring remain local. Own Gold fit/select=0.\n')
    book = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': notice}, F.code(loader),
                      *[F.code(s) for s in frozen], F.code(invoke),
                      *[copy.deepcopy(c) for c in parent['cells'] if c['cell_type'] == 'markdown']],
            'metadata': copy.deepcopy(parent['metadata']), 'nbformat': 4, 'nbformat_minor': 5}
    text = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    F.require(len(text.encode()) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(F.P033 / 'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=text,
                   dataset_data_sources=['heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1'])
    manifest = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'policy_SHA': POLICY_SHA,
                'selection_SHA': SELECTION_SHA, 'source_SHA': hashlib.sha256(text.encode()).hexdigest(),
                'frozen_original_cells': F.inventory(frozen), 'original_cell_indices': [0, 5],
                'runtime': F.pin(ROOT / 'scripts/r6_pair_probe_runtime.py'),
                'arithmetic': F.pin(ROOT / 'scripts/r6_pair_math.py'), 'probe_only_NEVER_formal_submit': True,
                'local_Torch_CUDA_MRI_Gold_network_Save_Submit': 0, 'source_bytes': len(text.encode()),
                'settings': {k: payload[k] for k in ('is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image')},
                'inputs': {k: payload[k] for k in ('dataset_data_sources', 'kernel_data_sources', 'competition_data_sources')}}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb', book), ('save-payload.json', payload), ('manifest.json', manifest)]:
        rendered = text if name == 'source.ipynb' else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
        (output / name).write_text(rendered, encoding='utf8', newline='\n')
    (output / 'runtime.py').write_text(runtime, encoding='utf8', newline='\n')
    (output / 'arithmetic.py').write_text(arithmetic, encoding='utf8', newline='\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slug', required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))
