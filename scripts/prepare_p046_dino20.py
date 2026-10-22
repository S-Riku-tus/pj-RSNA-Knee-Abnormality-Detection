"""Inert single-factor P045->P046 clone. Never authenticates, uploads or submits."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import prepare_p040_p041_factorial as F

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/kaggle/p045-p033-dino5-20261020-v2'
PARENT_SHA = '2d0525ba7ee0af4bce4e936a481a2f34ab1d185832fa4b2c617b6c081bed35b4'
RUNTIME_SHA = 'cda716ca11e7a2c237375a019966379ca3899cd5fdb6d1018e4c69575e99b537'
POLICY = ROOT / 'artifacts/research/20261020-P046-DINO20-prospective-v1/policy.json'
POLICY_SHA = '5a211bd382877723b1d1824923d6e527f0da010388f78d850c2b7fe407a75f6b'


def runtime20(source):
    """Only two .10 constants and one .90 constant implement this factor."""
    tree = ast.parse(source)
    count = {.1: 0, .9: 0, 'description': 0}

    class Change(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is float and node.value in (.1, .9):
                count[node.value] += 1
                return ast.copy_location(ast.Constant(.2 if node.value == .1 else .8), node)
            if node.value == 'One fixed .10 ordering component; exact parent strings outside cohort.':
                count['description'] += 1
                return ast.copy_location(ast.Constant('One fixed .20 ordering component; exact parent strings outside cohort.'), node)
            return node

    changed = Change().visit(tree)
    F.require(count == {.1: 2, .9: 1, 'description': 1}, 'Exactly the fixed factor constants required')
    result = ast.unparse(ast.fix_missing_locations(changed)) + '\n'
    # Reverse only the registered changes to prove no other AST difference.
    reverse = result.replace('DINO_WEIGHT = 0.2', 'DINO_WEIGHT = 0.1').replace('0.8 * left[i]', '0.9 * left[i]')
    reverse = reverse.replace("config.get('weight') == 0.2", "config.get('weight') == 0.1")
    reverse = reverse.replace('One fixed .20 ordering component;', 'One fixed .10 ordering component;')
    F.require(ast.dump(ast.parse(reverse), include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False), 'Only global factor may change')
    return result


def prepare(output, slug):
    output = Path(output)
    F.require(not output.exists() and not output.is_symlink(), 'Fresh artifact directory required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own slug required')
    F.require(F.pin(POLICY)['sha256'] == POLICY_SHA and F.pin(PARENT/'source.ipynb')['sha256'] == PARENT_SHA,
              'Frozen parent/policy differs')
    F.require(F.pin(PARENT/'runtime.py')['sha256'] == RUNTIME_SHA, 'Frozen integer-tie runtime differs')
    old_runtime = (PARENT/'runtime.py').read_text(encoding='utf8')
    runtime = runtime20(old_runtime)
    runtime_sha = hashlib.sha256(runtime.encode()).hexdigest()
    book = copy.deepcopy(F.read(PARENT/'source.ipynb'))
    codes = [c for c in book['cells'] if c['cell_type'] == 'code']
    F.require(len(codes) == 2, 'Exact corrected parent loader/transport required')
    loader_tree = ast.parse(F.text(codes[0]))
    counts = {'runtime': 0, 'hash': 0, 'descriptor': 0}

    class Loader(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == old_runtime:
                counts['runtime'] += 1
                return ast.copy_location(ast.Constant(runtime), node)
            if node.value == RUNTIME_SHA:
                counts['hash'] += 1
                return ast.copy_location(ast.Constant(runtime_sha), node)
            if node.value == 'DINOv3 fivefold ordering .10':
                counts['descriptor'] += 1
                return ast.copy_location(ast.Constant('DINOv3 fivefold ordering .20'), node)
            return node

    loader = ast.unparse(ast.fix_missing_locations(Loader().visit(loader_tree))) + '\n'
    F.require(counts == {'runtime': 1, 'hash': 1, 'descriptor': 1}, 'Only runtime/hash/descriptive factor substitutions allowed')
    transport_tree = ast.parse(F.text(codes[1]))
    configs = [n for n in ast.walk(transport_tree) if isinstance(n, ast.Dict)
               and [k.value for k in n.keys if isinstance(k, ast.Constant)] == ['candidate', 'weight']]
    F.require(len(configs) == 1 and ast.literal_eval(configs[0]) == {'candidate': 'P045', 'weight': .1}, 'Exact inherited execution config required')
    configs[0].values[1] = ast.Constant(.2)
    transport = ast.unparse(ast.fix_missing_locations(transport_tree)) + '\n'
    codes[0]['source'], codes[1]['source'] = loader, transport
    book['cells'][0]['source'] = ('# P046: fixed 20% DINOv3 ordering addition\n\n'
        'Only the global component weight changes from P045 corrected V2 (.10 to .20; parent .90 to .80). '
        'All eight P033 science cells, integer tie-safe fivefold ordering, model/preprocessing/Inputs/Docker and '
        'failure-cell preservation remain fixed. The inherited P045 internal output directory, receipt schema '
        'and producer namespace are intentionally retained. External candidate identity is P046. '
        'This is an exploratory Public-guided comparison, not a validated gain. Own Gold fit/select=0; '
        'upstream DINO training/Gold history remains UNKNOWN. Never automatically submit.\n')
    text = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    F.require(len(text.encode()) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(PARENT/'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=text)
    parent_manifest = F.read(PARENT/'manifest.json')
    manifest = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'candidate': 'P046', 'policy_SHA': POLICY_SHA,
                'parent_source_SHA': PARENT_SHA, 'source_SHA': hashlib.sha256(text.encode()).hexdigest(),
                'runtime_SHA': runtime_sha, 'source_bytes': len(text.encode()), 'parent_weight': .8, 'DINO_weight': .2,
                'parent8_inventory': parent_manifest['science_inventory'], 'all8_science_byte_preserved': True,
                'producer_SHA': parent_manifest['producer']['sha256'], 'original_integer_tie_logic_preserved': True,
                'internal_output_namespace': 'P045', 'local_model_GPU_MRI_network_Save_Submit': 0,
                'conditional_eligibility': 'Only own officialP045 COMPLETE.958 and goalunmet; fresh review/qualification/quota',
                'inputs': {k: payload[k] for k in ('dataset_data_sources', 'kernel_data_sources', 'competition_data_sources')},
                'settings': {k: payload[k] for k in ('is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image')}}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb', book), ('save-payload.json', payload), ('manifest.json', manifest)]:
        (output/name).write_text(text if name == 'source.ipynb' else json.dumps(value, indent=2)+'\n', encoding='utf8', newline='\n')
    (output/'runtime.py').write_text(runtime, encoding='utf8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))
