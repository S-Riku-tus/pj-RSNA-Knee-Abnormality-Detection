"""Offline P045 builder; inert payloads only, no authentication or remote writes."""
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
PUBLIC = ROOT / 'artifacts/research/20261020-DINO-public-diversity-source-audit-v1/shenzai25-source-V4.ipynb'
POLICY = ROOT / 'artifacts/research/20261020-P045-DINO5-prospective-v1/policy.json'
POLICY_SHA = 'c59c79187d5e19c540a748f4ff01b8d756b47aba8ac65adc50d1f3f1cde6d6f0'
TIE_AMENDMENT = ROOT / 'artifacts/research/20261020-P045-DINO5-tie-safety-amendment-v1/amendment.json'
TIE_AMENDMENT_SHA = 'e26188e9c6b06eb5ad1ea5d1052a813294d5a53d38f7a84d3fb4b5bf75ff95f6'
PUBLIC_SHA = 'bda7cdfd0cae68fe666ae73119fb18ee99346ed2b346dcf4a6bf03b8f782ae44'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def functions(source, wanted):
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    F.require({n.name for n in found} == set(wanted), 'Every exact frozen function required')
    return '\n\n'.join(''.join(lines[min([n.lineno, *[d.lineno for d in n.decorator_list]]) - 1:n.end_lineno])
                        for n in found) + '\n'


def dino_program(policy):
    F.require(F.pin(PUBLIC)['sha256'] == PUBLIC_SHA, 'Frozen public source differs')
    book = F.read(PUBLIC)
    model = F.text(book['cells'][22]).split('\nmodels = []\n', 1)
    F.require(len(model) == 2, 'Exact public model-definition boundary required')
    frozen = {'models': model[0] + '\n',
              'preprocess': functions(F.text(book['cells'][21]),
                                      ['ordered_files', 'series_side', 'read_crop', 'window', 'render', 'build_study']),
              'predict': functions(F.text(book['cells'][23]), ['amp_for', '_norm_', '_micro', 'predict'])}
    for value in frozen.values():
        ast.parse(value)
    frozen = {name: {'text': value, 'sha256': hashlib.sha256(value.encode()).hexdigest()} for name, value in frozen.items()}
    architecture = {'backbone': ['vit_small_patch16_dinov3.lvd1689m', None], 'cond': ['token', None],
                    'img': [336, None], 'meta': ['none', None], 'pool': ['xcodex', None],
                    'norm': ['none', 'none'], 'n_meta': [0, 0], 'n_slice': [16, 16], 'stem': ['native', 'native']}
    config = {'weights': policy['public_weight_SHA'], 'labels': TARGETS, 'architecture': architecture,
              'timm_wheel': {k: policy['timm_wheel'][k] for k in ('bytes', 'sha256')}}
    template = ROOT / 'scripts/p045_dino5_producer.py'
    tree = ast.parse(template.read_text(encoding='utf8'))
    for name, value in [('P045_CONFIG', config), ('P045_FROZEN', frozen)]:
        node = F.assignment(ast.unparse(tree), name)[1]
        # Find the actual node in this tree; F.assignment parses a separate tree.
        candidates = [n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        F.require(len(candidates) == 1 and ast.literal_eval(node.value) is None, 'Single empty producer literal required')
        candidates[0].value = ast.parse(repr(value), mode='eval').body
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n', frozen, config


def prepare(output, slug):
    output = Path(output)
    F.require(not output.exists() and not output.is_symlink(), 'Fresh P045 output directory required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own slug required')
    F.require(F.pin(POLICY)['sha256'] == POLICY_SHA, 'Frozen prospective policy differs')
    F.require(F.pin(TIE_AMENDMENT)['sha256'] == TIE_AMENDMENT_SHA, 'Frozen integer-tie safety amendment required')
    F.require(F.pin(F.P033 / 'source.ipynb')['sha256'] == F.P033_SHA, 'Frozen P033 source differs')
    F.require(F.pin(P041 / 'source.ipynb')['sha256'] == '0c06b802d34d6cee85b5156fb30ae1b20f7b09c747550639c40547e2ad1e5c7a',
              'Frozen operational loader differs')
    policy, parent = F.read(POLICY), F.read(F.P033 / 'source.ipynb')
    science = [F.text(c) for c in parent['cells'] if c['cell_type'] == 'code']
    F.require(len(science) == 8, 'All eight original parent scientific cells required')
    for cell in science:
        ast.parse(cell.split('\n', 1)[1] if cell.startswith('%%writefile ') else cell)
    producer, component, component_config = dino_program(policy)
    runtime_path = ROOT / 'scripts/p045_dino5_runtime.py'
    runtime = runtime_path.read_text(encoding='utf8')
    ast.parse(runtime)
    # Reuse the scored Reader baseline's operational decoder/Input guard only.
    # Reader weights remain mounted and hashed; no Reader science is executed.
    old_loader = F.text(F.read(P041 / 'source.ipynb')['cells'][1])
    guard_source = ast.literal_eval(F.assignment(old_loader, '_factor_preflight_source')[1].value)
    guard_config = ast.literal_eval(F.assignment(old_loader, '_factor_preflight_config')[1].value)
    descriptor = {'candidate': 'P045', 'parent': 'P033 original8science', 'factor': 'DINOv3 fivefold ordering .10',
                  'Reader_science_executed': False, 'R7_science_executed': False, 'Gaussian_pooling': False,
                  'unused_Reader_Input_hashed_by_inherited_guard': True, 'own_Gold_fit_select': 0,
                  'DINO_upstream_train_Gold_history': 'UNKNOWN'}
    guard_config['producer_descriptor'] = descriptor
    loader = (f"_p045_runtime_source={runtime!r}\nimport hashlib as _p045_hash\n"
              f"assert _p045_hash.sha256(_p045_runtime_source.encode()).hexdigest()=={hashlib.sha256(runtime.encode()).hexdigest()!r}\n"
              "_p045_ns={'__name__':'p045_dino5_runtime'}\n"
              "exec(compile(_p045_runtime_source,'<p045-fixed-runtime>','exec'),_p045_ns)\n"
              f"_p045_guard_source={guard_source!r}\n"
              f"assert _p045_hash.sha256(_p045_guard_source.encode()).hexdigest()=={hashlib.sha256(guard_source.encode()).hexdigest()!r}\n"
              "_p045_guard_ns={'__name__':'p045_inherited_decoder_guard'}\n"
              "exec(compile(_p045_guard_source,'<p045-inherited-guard>','exec'),_p045_guard_ns)\n"
              f"_p045_guard=_p045_guard_ns['preflight']({guard_config!r})\n")
    inventory = F.inventory(science)
    raw = json.dumps(science, ensure_ascii=False, separators=(',', ':')).encode()
    packed = lzma.compress(raw, preset=6)
    transport = (f"_p045_science=_p045_ns['restore_cells']({base64.b85encode(packed).decode()!r},"
                 f"{hashlib.sha256(packed).hexdigest()!r},{hashlib.sha256(raw).hexdigest()!r},{inventory!r})\n"
                 f"_p045_dino_source={producer!r}\n"
                 f"_p045_execution=_p045_ns['execute_pipeline'](_p045_science,get_ipython(),{inventory!r},"
                 f"_p045_dino_source,{hashlib.sha256(producer.encode()).hexdigest()!r},{{'candidate':'P045','weight':.10}})\n")
    notice = ('# P045: fixed 10% public DINOv3 fivefold addition\n\n'
              'Original eight P033 scientific cells remain byte-exact. One new distinct DINOv3 component is added with '
              'global weight .10, using strict five-checkpoint SHA/key/label guards, pinned timm1.0.22, artificial CPU/CUDA '
              'forward before new MRI decode, and parent-cell-string rollback outside the successful cohort. '
              'No Reader/R7/Gaussian science or fitting. Upstream DINO training/Gold selection is UNKNOWN; no clean OOF '
              'or accuracy gain is claimed. Reader Inputs remain mounted only for the inherited operational guard.\n\n'
              'DINO source: [shenzai25 V4](https://www.kaggle.com/code/shenzai25/bend-the-knee-to-the-dinosaurs?scriptVersionId=358339637), '
              'fivefold weights: [Mattia Angeli V2](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights/versions/2), CC0.\n')
    credits = [copy.deepcopy(c) for c in parent['cells'] if c['cell_type'] == 'markdown']
    credits += [copy.deepcopy(F.read(PUBLIC)['cells'][0])]
    book = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': notice}, F.code(loader), F.code(transport),
                      {'cell_type': 'markdown', 'metadata': {}, 'source': '## Inherited attribution and historical source notes\n'}, *credits],
            'metadata': copy.deepcopy(parent['metadata']), 'nbformat': 4, 'nbformat_minor': 5}
    text = json.dumps(book, ensure_ascii=False, indent=1) + '\n'
    F.require(len(text.encode()) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(P041 / 'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=text)
    payload['dataset_data_sources'].append('mattiaangeli/knee-mri-fold-weights/2')
    manifest = {'schema': 'P045_fixed_DINO5_addition_source_v1', 'at_UTC': datetime.now(timezone.utc).isoformat(),
                'source': {'bytes': len(text.encode()), 'sha256': hashlib.sha256(text.encode()).hexdigest()},
                'parent': F.pin(F.P033 / 'source.ipynb'), 'science_inventory': inventory,
                'parent_all8_byte_exact': True, 'runtime': F.pin(runtime_path), 'producer_template': F.pin(ROOT / 'scripts/p045_dino5_producer.py'),
                'producer': {'bytes': len(producer.encode()), 'sha256': hashlib.sha256(producer.encode()).hexdigest()},
                'public_source': F.pin(PUBLIC), 'policy': F.pin(POLICY), 'tie_safety_amendment': F.pin(TIE_AMENDMENT),
                'component_function_SHA': {k: v['sha256'] for k, v in component.items()}, 'component_config': component_config,
                'inherited_guard': {'bytes': len(guard_source.encode()), 'sha256': hashlib.sha256(guard_source.encode()).hexdigest()},
                'guard_config': guard_config, 'descriptor': descriptor,
                'inputs': {k: payload[k] for k in ['dataset_data_sources', 'kernel_data_sources', 'competition_data_sources']},
                'settings': {k: payload[k] for k in ['is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image']},
                'local_model_MRI_Gold_fit_GPU_network_Save_Submit': 0, 'own_Public': None}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source.ipynb').write_text(text, encoding='utf8', newline='\n')
    for name, value in [('manifest.json', manifest), ('save-payload.json', payload), ('science-cells.json', science)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'dino-producer.py').write_text(producer, encoding='utf8', newline='\n')
    (output / 'runtime.py').write_text(runtime, encoding='utf8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.slug)
    print(json.dumps({'source': result['source'], 'parent8_byte_exact': True, 'remote_writes': 0}))
