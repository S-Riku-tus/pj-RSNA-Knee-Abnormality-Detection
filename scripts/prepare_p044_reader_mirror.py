"""Offline P041 -> P044: exact tested P036 mirror224 cell only; no remote writes."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import lzma
import re
from datetime import datetime, timezone
from pathlib import Path

import prepare_p040_p041_factorial as F

ROOT = Path(__file__).resolve().parents[1]
P041 = ROOT / 'artifacts/kaggle/p041-p033-reader-20261019-v1'
P036 = ROOT / 'artifacts/kaggle/p036-p033-224-mirror-20261017-v1'
POLICY = ROOT / 'artifacts/research/20261020-P044-reader-mirror-prospective-v1/prospective-policy.json'
P041_SHA = '0c06b802d34d6cee85b5156fb30ae1b20f7b09c747550639c40547e2ad1e5c7a'
P036_SHA = '44be198b41208edd3f01e9897deb76f44ae88bdf30ac8d728b50d17db87c31b9'
POLICY_SHA = 'bd0965958f2d80423273f3cabcf0ad1964172ef7546e0921f0418b41711b222b'
MIRROR_SHA = '5c2dfda1c2b538e4e2387b71d4e760768211ad69d7693797cc4254502e32abad'


def literal(tree, name):
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    F.require(len(nodes) == 1, 'Single literal assignment required: ' + name)
    return nodes[0]


def prepare(output, slug):
    output = Path(output)
    F.require(not output.exists() and not output.is_symlink(), 'Fresh P044 output required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own private slug required')
    for path, sha in [(P041 / 'source.ipynb', P041_SHA), (P036 / 'source.ipynb', P036_SHA), (POLICY, POLICY_SHA)]:
        F.require(F.pin(path)['sha256'] == sha, 'Pinned source/policy differs: ' + path.name)
    parent = F.read(P041 / 'source.ipynb')
    parent_manifest = F.read(P041 / 'manifest.json')
    originals = F.read(P041 / 'science-cells.json')
    F.require(len(originals) == 9 and F.inventory(originals) == parent_manifest['science_inventory'],
              'Exact nine scored P041 science cells required')
    mirror_book = F.read(P036 / 'source.ipynb')
    mirror = [F.text(c) for c in mirror_book['cells'] if c['cell_type'] == 'code']
    F.require(len(mirror) == 8 and [i for i in range(8) if mirror[i] != originals[i]] == [3],
              'P036 differs from P041 core only at224 source')
    F.require(hashlib.sha256(mirror[3].encode()).hexdigest() == MIRROR_SHA, 'Exact tested mirror cell required')
    science = list(originals)
    science[3] = mirror[3]
    F.require([i for i, (a, b) in enumerate(zip(originals, science)) if a != b] == [3],
              'Exactly one of nine science cells may differ')
    for cell in science:
        ast.parse(cell.split('\n', 1)[1] if cell.startswith('%%writefile ') else cell)
    runtime_path = ROOT / 'scripts/p044_reader_mirror_runtime.py'
    runtime = runtime_path.read_text(encoding='utf-8')
    ast.parse(runtime)
    loader = F.text(parent['cells'][1])
    old_runtime_sha = parent_manifest['runtime']['sha256']
    F.require(loader.count(old_runtime_sha) == 1, 'Original controller hash assertion required')
    loader = loader.replace(old_runtime_sha, hashlib.sha256(runtime.encode()).hexdigest())
    tree = ast.parse(loader)
    literal(tree, '_factor_runtime_source').value = ast.Constant(runtime)
    descriptor = {'candidate': 'P044', 'baseline': 'P041 own958 ref57402914',
                  'scientific_factor': 'Raptor224_anatomical_mirror_only',
                  'other8of9_science_cells_byte_exact': True, 'mirror_source': 'P036 own958 ref57291580',
                  'R7_included': False, 'Gaussian_pooling_included': False,
                  'Reader_positive_completion_required': True, 'own_Gold_fit_select': 0,
                  'artificial_mirror_CPU_CUDA_QA_required': True,
                  'upstream_Gold_selection_exposure_inherited': True}
    preflight = ast.literal_eval(literal(tree, '_factor_preflight_config').value)
    preflight['producer_descriptor'] = descriptor
    literal(tree, '_factor_preflight_config').value = ast.parse(repr(preflight), mode='eval').body
    loader = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    loader = loader.replace('p040_p041_factorial_runtime', 'p044_reader_mirror_runtime')
    loader = loader.replace('p041_inherited_P038_preflight', 'p044_inherited_P038_preflight')
    inherited = ast.literal_eval(literal(ast.parse(loader), '_factor_preflight_source').value)
    F.require(hashlib.sha256(inherited.encode()).hexdigest() == parent_manifest['inherited_preflight_source']['sha256'],
              'Inherited decoder/checkpoint preflight must remain byte-exact')
    inventory = F.inventory(science)
    config = {'candidate': 'P044', 'science_inventory': inventory, 'class_vector': F.OLD_VECTOR,
              'inherited_preflight': 'Exact frozen P038 CP313/Input/fixture guards; new actual pass required'}
    raw = json.dumps(science, ensure_ascii=False, separators=(',', ':')).encode()
    packed = lzma.compress(raw, preset=6)
    transport = (f"_factor_science=_factor_ns['restore_cells']({base64.b85encode(packed).decode()!r},"
                 f"{hashlib.sha256(packed).hexdigest()!r},{hashlib.sha256(raw).hexdigest()!r},{inventory!r})\n"
                 "_factor_mirror_QA=_factor_ns['mirror_artificial_qa'](_factor_science[3])\n"
                 f"_factor_execution=_factor_ns['execute_cells'](_factor_science,get_ipython(),{config!r})\n")
    notice = ('# P044: mirror224 only on scored P041 Reader baseline\n\n'
              'One of nine scientific cells changes: exact P036 anatomical-mirror224 TTA replaces plain224. '
              'Every other scientific cell, three Reader checkpoints, k16/end-mode weights, classvector/A224, '
              'seven Inputs and offlineT4Docker remain fixed. No R7, Gaussian pooling, label fitting or new ownGold use. '
              'Positive Reader/decoder completion and current-run arithmetic/source verification are required. '
              'Upstream Gold-selection exposure is inherited. No Public improvement is guaranteed.\n')
    credits = [copy.deepcopy(c) for c in F.read(F.P033 / 'source.ipynb')['cells'] if c['cell_type'] == 'markdown']
    notebook = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': notice}, F.code(loader), F.code(transport),
                           {'cell_type': 'markdown', 'metadata': {}, 'source': '## Inherited attribution and historical parent notes\n'},
                           *credits], 'metadata': copy.deepcopy(parent['metadata']), 'nbformat': 4, 'nbformat_minor': 5}
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1) + '\n').encode()
    F.require(len(serialized) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(P041 / 'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=serialized.decode())
    manifest = {'schema': 'P044_fixed_one_cell_mirror_on_Reader_source_v1',
                'at_UTC': datetime.now(timezone.utc).isoformat(), 'candidate': 'P044', 'stage': 'source_only',
                'source': {'bytes': len(serialized), 'sha256': hashlib.sha256(serialized).hexdigest()},
                'parent': {'source': F.pin(P041 / 'source.ipynb'), 'science_inventory': F.inventory(originals)},
                'mirror_donor': {'source': F.pin(P036 / 'source.ipynb'), 'cell_SHA': MIRROR_SHA},
                'policy': F.pin(POLICY), 'changed_scientific_indices': [3], 'science_inventory': inventory,
                'descriptor': descriptor, 'runtime': F.pin(runtime_path), 'guard_config': preflight,
                'inherited_preflight_source': parent_manifest['inherited_preflight_source'],
                'inputs': copy.deepcopy(parent_manifest['inputs']),
                'runtime_settings': {k: payload[k] for k in ['is_private','enable_gpu','enable_tpu','enable_internet',
                                                          'machine_shape','docker_image','kernel_execution_type']},
                'local_GPU_or_models_MRI_labels_fit_network_Save_Submit': 0,
                'own_Public': None, 'actual_visible_qualification': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb', serialized), ('science-cells.json', raw), ('p044-runtime.py', runtime.encode())]:
        (output / name).write_bytes(value)
    for name, value in [('manifest.json', manifest), ('save-payload.json', payload),
                        ('fixed-input-payload.json', {**{k:v for k,v in payload.items() if k != 'text'},
                                                     'source_sha256': manifest['source']['sha256']})]:
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    value = prepare(args.output_dir, args.slug)
    print(json.dumps({'candidate': 'P044', 'source': value['source'], 'inputs': value['inputs']['total']}))
