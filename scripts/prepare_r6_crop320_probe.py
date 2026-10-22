"""Inert one-factor crop320 validation clone; no remote writes or data inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import prepare_p040_p041_factorial as F

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'artifacts/kaggle/r6-paired-weak-probe-20261020-v1'
PARENT_SHA = 'ce1848aadbffc1ecb00c966e39227c5da926b7461afa2f31f894b8c2012d1c5b'
OLD_POLICY_SHA = '579b4bc490d27f46898cc75266478a59ddf73356b8952b8fa2f3a14e99945fa8'
POLICY = ROOT/'artifacts/research/20261020-R6-crop320-paired-prospective-v1/policy.json'
POLICY_SHA = '2ef4aaf522c942f1dc61ed8b11d6f65a63d3a7b71f3ead5d52e709533b3bac4a'
CROP_ONLY = """assert manifest['crop']==352 and manifest['backbone']=='convnext_small'
assert WORKER_ARGS.count('--crop')==1
_pair_crop_index=WORKER_ARGS.index('--crop')+1
assert WORKER_ARGS[_pair_crop_index]=='352'
WORKER_ARGS=WORKER_ARGS.copy()
WORKER_ARGS[_pair_crop_index]='320'
print('R6_SINGLE_FACTOR_NATIVE_CROP_352_TO_320',flush=True)
"""


def prepare(output, slug):
    output = Path(output)
    F.require(not output.exists() and not output.is_symlink(), 'Fresh output required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own slug required')
    F.require(F.pin(PARENT/'source.ipynb')['sha256'] == PARENT_SHA and F.pin(POLICY)['sha256'] == POLICY_SHA,
              'Frozen parent and prospective policy required')
    book = copy.deepcopy(F.read(PARENT/'source.ipynb'))
    indices = [i for i, c in enumerate(book['cells']) if c['cell_type'] == 'code']
    F.require(len(indices) == 4, 'Frozen four-code-cell validation parent required')
    invoke = F.text(book['cells'][indices[-1]])
    F.require(invoke.count(OLD_POLICY_SHA) == 1, 'Only fixed provenance binding may change')
    book['cells'][indices[-1]]['source'] = invoke.replace(OLD_POLICY_SHA, POLICY_SHA)
    book['cells'].insert(indices[-1], F.code(CROP_ONLY))
    book['cells'][0]['source'] = ('# R6 crop320 paired weak validation probe\n\n'
        'Never formally submit this notebook. Only inference native center crop changes352 to320. '
        'The exact parent assets/worker/models/heldout routing, UID selection, preprocessing render384, '
        'mirrorTTA.50 and original postprocessors remain fixed. Synthetic GPU directON/OFF and replay '
        'must pass before MRI. Only the baseline320 result will be evaluated against baseline352; '
        'the inherited unused replay combinations will not be evaluated or selected. '
        'Original R6-paired-weak-probe internal output directory is retained. '
        'Own Gold fit/select=0; upstream membership/Gold/auxiliary exposure and patient independence '
        'remain unverified. No accuracy improvement claimed.\n')
    text = json.dumps(book, ensure_ascii=False, indent=1)+'\n'
    F.require(len(text.encode()) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(PARENT/'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=text)
    manifest = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'policy_SHA': POLICY_SHA,
                'parent_source_SHA': PARENT_SHA, 'source_SHA': hashlib.sha256(text.encode()).hexdigest(),
                'single_factor': '--crop352to320', 'neural_worker_source_model_SHAs_unchanged': True,
                'original_asset_and_runtime_cells_byte_exact': True, 'new_training': 0, 'own_Gold_fit_select': 0,
                'formal_Submit_allowed': False, 'local_Torch_GPU_MRI_network_Save_Submit': 0,
                'only_baseline320_evaluated': True, 'internal_output_folder': 'R6-paired-weak-probe',
                'inputs': {k: payload[k] for k in ('dataset_data_sources', 'kernel_data_sources', 'competition_data_sources')},
                'settings': {k: payload[k] for k in ('is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image')}}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb', book), ('save-payload.json', payload), ('manifest.json', manifest)]:
        (output/name).write_text(text if name == 'source.ipynb' else json.dumps(value, indent=2)+'\n', encoding='utf8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))
