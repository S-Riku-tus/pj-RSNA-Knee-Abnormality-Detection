"""Prepare the P029 R6 V1 stage fallback offline, without running generated code.

The generated stage notebook contains an anonymous public download routine for
later reviewed Kaggle execution. This CLI only reads pinned public metadata and
source text, compiles text, and writes a new local run. It does not authenticate,
download models, decode MRI, train, upload, or submit.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_p030_tta_control import compile_cell, embedded_sources, source_text

SOURCE_SHA = "449ce7786021dc17fa671783d86d6f5ae4e10638f013c93085eadaba847cbd02"
PARENT_MANIFEST_SHA = "3acd5dbdbbb04320d7708a0256746457500ab03a9fb2878c0c5d9ca1ef9a1cf5"
R6_SHA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
R6_SLUG = "rsna-knee-r6-frozen-fe055406"
ANCHOR = "    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))\n"
PUBLIC_EVIDENCE = {
    'SWA_current_V1': ('artifacts/research/20261015-p029-public-input-metadata-v1/SWA-view.json',
                       '1045b99e09a9893b8f9a92fbc9b6ced2a34de4c5cc2f8618893ccaa6831caedd'),
    'Ortho_current_V1': ('artifacts/research/20261015-p029-public-input-metadata-v1/Ortho-view.json',
                         '8981aaab59ab38e4c2513619be9f9b9d3d1d28b4eba1ce303d39732abf0216ae'),
    'R6_current_V2_not_pinned_V1_terms': (
        'artifacts/research/20261015-helios957-source-metadata-v1/r6-frozen-dataset-view.json',
        '16328c062580facd145c0121ff457febc8c3009ca1991d53be00b385d6d87edb'),
}

def validated_stage_slug(stage_slug):
    # A single lower-case Kaggle slug; no owner, separator, quote, or traversal.
    if not isinstance(stage_slug, str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{3,48}[a-z0-9])', stage_slug):
        raise ValueError('Supply a lower-case stage slug of 5-50 letters/digits/dashes')
    return stage_slug

def finder_branch(stage_slug):
    return "    cands += [INPUT / 'notebooks' / 'rsraki' / '" + validated_stage_slug(stage_slug) + "' / slug]\n"

STAGE_BODY = r'''
from pathlib import Path
import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

def validate_name(name):
    if (not isinstance(name, str) or name not in EXPECTED_SIZES
            or name in ('.', '..') or any(c in name for c in '/\\:')):
        raise ValueError('Only the nine pinned R6 V1 basenames are allowed')
    return name

def validate_manifest(raw):
    if len(raw) != EXPECTED_SIZES['manifest.json'] or hashlib.sha256(raw).hexdigest() != EXPECTED_SHA['manifest.json']:
        raise ValueError('R6 V1 raw manifest size/SHA mismatch; do not substitute latest')
    value = json.loads(raw)
    if (value.get('bundle_id') != 'fe055406fad6a12f' or value.get('models') != EXPECTED_MODELS
            or value.get('sha256') != {k: v for k, v in EXPECTED_SHA.items() if k != 'manifest.json'}):
        raise ValueError('R6 V1 bundle/four-fold/member SHA contract mismatch')
    return value

def public_https_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or '').lower()
    allowed = host in {'www.kaggle.com', 'kaggle.com', 'storage.googleapis.com'} or host.endswith('.googleusercontent.com')
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443) or not allowed:
        raise ValueError('Unexpected public download redirect')
    return url

class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        public_https_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)

def download_verified(opener, name, destination):
    validate_name(name)
    destination = Path(destination)
    if destination.name != name or destination.exists() or destination.is_symlink():
        raise FileExistsError('New pinned destination required')
    query = urllib.parse.urlencode({'datasetVersionNumber': 1, 'raw': 'true'})
    url = 'https://www.kaggle.com/api/v1/datasets/download/heliosli/' + R6_SLUG + '/' + name + '?' + query
    request = urllib.request.Request(public_https_url(url), headers={'Accept-Encoding': 'identity', 'User-Agent': 'P029-R6V1-metadata-pinned-stage/1'})
    count = 0
    sha = hashlib.sha256()
    try:
        with opener.open(request, timeout=120) as response, destination.open('xb') as output:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != EXPECTED_SIZES[name]:
                raise ValueError('Unexpected response size: ' + name)
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > EXPECTED_SIZES[name]:
                    raise ValueError('Pinned file size exceeded: ' + name)
                output.write(block)
                sha.update(block)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        # Do not print signed redirect URLs, cookies, tokens, or response bodies.
        raise RuntimeError('Anonymous public download failed for ' + name + ' (' + type(error).__name__ + '); no authentication fallback') from None
    if count != EXPECTED_SIZES[name] or sha.hexdigest() != EXPECTED_SHA[name]:
        raise ValueError('Pinned file size/SHA mismatch: ' + name)
    return {'name': name, 'bytes': count, 'sha256': sha.hexdigest()}

def make_temporary_directory(temporary):
    # /kaggle/temp is absent in some CPU images. Reject redirected parents, then
    # create the missing parent and the new child without reusing an old run.
    if temporary.parent.is_symlink() or temporary.parent.resolve() != Path('/kaggle/temp'):
        raise RuntimeError('Unexpected temporary parent path')
    temporary.mkdir(parents=True, exist_ok=False)

def main():
    # Source preparation does not call this function. Future execution is Kaggle CPU only.
    working = Path('/kaggle/working')
    if (Path.cwd().resolve() != working or not Path('/kaggle/input').is_dir()
            or working.is_symlink() or Path('/dev/nvidia0').exists()):
        raise RuntimeError('Run only as a fresh private Kaggle CPU batch with Internet ON')
    output = working / R6_SLUG
    temporary = Path('/kaggle/temp') / (STAGE_SLUG + '-unverified')
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError('Do not rerun/update the V1-only stage; use a fresh reviewed ref')
    make_temporary_directory(temporary)
    # No cookies, credential files, authentication helpers, proxy environment, or retries.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirect())
    evidence = [download_verified(opener, 'manifest.json', temporary / 'manifest.json')]
    validate_manifest((temporary / 'manifest.json').read_bytes())
    for name in sorted(EXPECTED_SIZES):
        if name != 'manifest.json':
            evidence.append(download_verified(opener, name, temporary / name))
    if {p.name for p in temporary.iterdir()} != set(EXPECTED_SIZES):
        raise ValueError('Unexpected stage members')
    # Nothing is copied into notebook outputs until all nine files have passed.
    shutil.copytree(temporary, output, symlinks=False)
    for name, size in EXPECTED_SIZES.items():
        path = output / name
        sha = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                sha.update(block)
        if path.stat().st_size != size or sha.hexdigest() != EXPECTED_SHA[name]:
            raise ValueError('Copied output failed verification: ' + name)
    provenance = dict(PROVENANCE)
    provenance.update({'verified_at_UTC': datetime.now(timezone.utc).isoformat(), 'files': evidence,
                       'all_nine_files_verified': True, 'model_deserialization': False,
                       'MRI_read_decode_training': False, 'submission': False})
    (working / 'stage-provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (working / 'UPSTREAM_CREDITS.txt').write_text(UPSTREAM_CREDITS, encoding='utf-8')
    (working / 'public-model-use-metadata.json').write_text(json.dumps(PUBLIC_MODEL_METADATA, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'stage': STAGE_SLUG, 'verified_files': len(evidence), 'bundle_id': 'fe055406fad6a12f', 'model_loads': 0, 'MRI': 0, 'keep_saved_version': 1}))

if __name__ == '__main__':
    main()
'''

GUARD_BODY = r'''
from pathlib import Path
import hashlib
import json

_p029_input = Path('/kaggle/input')
_p029_paths = [
    _p029_input / 'notebooks' / 'rsraki' / STAGE_SLUG / R6_SLUG,
    _p029_input / STAGE_SLUG / R6_SLUG,
]
_p029_found = [p for p in _p029_paths if p.is_dir()]
if len(_p029_found) != 1:
    raise RuntimeError('Attach exactly the reviewed private stage V1 output; do not attach the R6 dataset as well')
_p029_root = _p029_found[0]
if _p029_root.is_symlink() or not _p029_root.resolve().is_relative_to(_p029_input.resolve()):
    raise RuntimeError('Stage path escapes the Kaggle Input mount')
if {p.name for p in _p029_root.iterdir()} != set(EXPECTED_SIZES):
    raise RuntimeError('Stage output does not contain exactly the nine pinned files')
for _p029_name, _p029_size in EXPECTED_SIZES.items():
    _p029_file = _p029_root / _p029_name
    if (_p029_file.is_symlink() or not _p029_file.is_file()
            or not _p029_file.resolve().is_relative_to(_p029_root.resolve())
            or _p029_file.stat().st_size != _p029_size):
        raise RuntimeError('Invalid stage member: ' + _p029_name)
    _p029_sha = hashlib.sha256()
    with _p029_file.open('rb') as _p029_handle:
        for _p029_block in iter(lambda: _p029_handle.read(1024 * 1024), b''):
            _p029_sha.update(_p029_block)
    if _p029_sha.hexdigest() != EXPECTED_SHA[_p029_name]:
        raise RuntimeError('Stage member SHA mismatch: ' + _p029_name)
_p029_manifest = json.loads((_p029_root / 'manifest.json').read_bytes())
if (_p029_manifest.get('bundle_id') != 'fe055406fad6a12f'
        or _p029_manifest.get('models') != EXPECTED_MODELS):
    raise RuntimeError('R6 V1/four-fold manifest required')
print('P029 stage preflight: 9 exact members verified, R6 V1, 4 folds, no model loaded')
'''

def sha(body):
    return hashlib.sha256(body).hexdigest()

def read_pinned(relative, expected):
    body = (ROOT / relative).read_bytes()
    if sha(body) != expected:
        raise ValueError('Pinned source changed: ' + relative)
    return body, json.loads(body)

def notebook(cells, gpu, internet):
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}, 'kaggle': {'isGpuEnabled': gpu, 'isInternetEnabled': internet}}, 'nbformat': 4, 'nbformat_minor': 4}

def cell(kind, source):
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(outputs=[], execution_count=None)
    return value

def changed_finder(before, stage_slug):
    branch = finder_branch(stage_slug)
    tree = ast.parse(before)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_mount']
    if len(nodes) != 1 or before.count(ANCHOR) != 1:
        raise ValueError('One exact top-level mount finder required')
    after = before.replace(ANCHOR, ANCHOR + branch, 1)
    candidate = ast.parse(after)
    f = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == 'find_mount')
    added = [n for n in f.body if isinstance(n, ast.AugAssign) and ast.unparse(n) == ast.unparse(ast.parse(branch.strip()).body[0])]
    if len(added) != 1:
        raise ValueError('One fixed bounded notebook mount branch required')
    f.body.remove(added[0])
    if ast.dump(tree, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        raise ValueError('Change beyond mount discovery')
    if after.replace(branch, '', 1) != before:
        raise ValueError('Non-finder byte changes')
    return after

def prepare(output, stage_slug):
    output = Path(output)
    stage_slug = validated_stage_slug(stage_slug)
    branch = finder_branch(stage_slug)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve the previous backup preparation')
    source_bytes, parent = read_pinned('artifacts/kaggle/p029-public957-r6-v1-20261015/source.ipynb', SOURCE_SHA)
    _, parent_manifest = read_pinned('artifacts/kaggle/p029-public957-r6-v1-20261015/manifest.json', PARENT_MANIFEST_SHA)
    _, r6 = read_pinned('artifacts/research/20261015-r6-v1-manifest-only-v1/manifest.json', R6_SHA)
    sizes = parent_manifest['public_Input_metadata']['R6_V1_file_listing']
    hashes = dict(r6['sha256'], **{'manifest.json': R6_SHA})
    if set(sizes) != set(hashes) or len(sizes) != 9 or len(r6['models']) != 4:
        raise ValueError('Expected exact R6 V1 nine-file/four-fold contract')
    public = {}
    for label, (relative, expected_sha) in PUBLIC_EVIDENCE.items():
        raw, value = read_pinned(relative, expected_sha)
        public[label] = {'evidence_relative_path': relative, 'evidence_sha256': sha(raw),
                         'description': value.get('description', value.get('descriptionNullable')),
                         'license': value.get('licenseName', value.get('licenseNameNullable')),
                         'currentVersionNumber': value.get('currentVersionNumber')}
    credits = '\n\n'.join(source_text(parent['cells'][i]) for i in (0, 1))
    credits += '\n\nP029 fallback: private non-commercial competition use only; preserve all original credits and terms.\nSWA/OAI Gold58 selection exposure is inherited; no claim of Gold-clean pretraining or patient independence.\nR6 public Dataset license is Other; this private transport copy does not establish unconditional redistribution permission.\n'
    provenance = {'source_ref': 'heliosli/rsna-knee-r6-frozen-fe055406/1', 'dataset_id': 12524886,
                  'internal_dataset_version_id': 20611930, 'version': 1, 'bundle_id': 'fe055406fad6a12f',
                  'manifest_sha256': R6_SHA, 'stage_ref': 'rsraki/' + stage_slug,
                  'stage_must_remain_private': True, 'stage_must_remain_only_V1': True,
                  'parent_notebook_sha256': SOURCE_SHA, 'original_credits_preserved': True,
                  'no_MRI_reports_labels': True, 'raw_HTTP_endpoint_not_yet_executed_or_verified': True}
    constants = '\n'.join(name + ' = ' + repr(value) for name, value in {
        'STAGE_SLUG': stage_slug, 'R6_SLUG': R6_SLUG, 'EXPECTED_SIZES': sizes,
        'EXPECTED_SHA': hashes, 'EXPECTED_MODELS': r6['models']}.items()) + '\n'
    stage_source = constants + '\nPROVENANCE = ' + repr(provenance) + '\nUPSTREAM_CREDITS = ' + repr(credits) + '\nPUBLIC_MODEL_METADATA = ' + repr(public) + '\n' + STAGE_BODY
    guard = constants + GUARD_BODY
    compile(stage_source, 'unexecuted_stage', 'exec')
    compile(guard, 'unexecuted_mount_guard', 'exec')
    stage = notebook([cell('markdown', '# Unused P029 fallback: private CPU / Internet ON / V1 only\n\nThis notebook is source preparation only. It will download public R6 V1 weights only when separately run on Kaggle. It never loads models, reads MRI, trains, uploads, or submits. Save it once as V1 under the exact reviewed stage ref, then never update it. No credentials or authentication fallback.\n\nThe per-file public raw endpoint has not been tested; an unsupported response stops on size/SHA mismatch.\n'), cell('markdown', credits), cell('code', stage_source)], False, True)
    candidate = copy.deepcopy(parent)
    before = source_text(parent['cells'][6])
    after = changed_finder(before, stage_slug)
    candidate['cells'][6]['source'] = after.splitlines(keepends=True)
    for c in candidate['cells']:
        c.get('metadata', {}).pop('execution', None)
        c.get('metadata', {}).pop('papermill', None)
        if c['cell_type'] == 'code':
            c['outputs'], c['execution_count'] = [], None
    candidate.get('metadata', {}).pop('execution', None)
    candidate.get('metadata', {}).pop('papermill', None)
    # Attachments/settings must come from the new reviewed Save request, not the
    # author's historical notebook metadata.
    candidate['metadata']['kaggle'] = {'isGpuEnabled': True, 'isInternetEnabled': False}
    checks = []
    for i in range(2, 9):
        old, new = source_text(parent['cells'][i]), source_text(candidate['cells'][i])
        compile_cell(new, i)
        if i != 6 and old != new:
            raise ValueError('Other science cell changed')
        checks.append({'parent_cell': i, 'parent_sha256': sha(old.encode()), 'candidate_sha256': sha(new.encode()), 'same': old == new})
    modules = embedded_sources(before)
    if modules != embedded_sources(after):
        raise ValueError('Embedded scientific module changed')
    candidate['cells'].insert(0, cell('code', guard))
    notice = '# Unused P029 fallback mount adapter\n\nR6 V1 content is inherited through one immutable private stage Notebook output. The seven author code cells retain all numerical code, four folds and FLIP_TTA=True. Only one fixed mount-discovery candidate is added. The leading guard checks every content SHA before either Raptor loads. Author scores/outputs are not our results; all old outputs/counts were removed.\n\n'
    candidate['cells'].insert(0, cell('markdown', notice))
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in [('stage.ipynb', stage), ('inference-mount-adapter.ipynb', candidate)]:
        body = (json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        with (output / name).open('xb') as handle:
            handle.write(body)
        files[name] = {'bytes': len(body), 'sha256': sha(body)}
    manifest = {'schema_version': 'p029_unused_private_CPU_stage_backup_source_v1', 'prepared_at_UTC': datetime.now(timezone.utc).isoformat(),
                'status': 'UNUSED_SOURCE_PREPARATION_ONLY', 'parent_P029_immutable': True,
                'source_parent': {'bytes': len(source_bytes), 'sha256': SOURCE_SHA, 'manifest_sha256': PARENT_MANIFEST_SHA},
                'files': files, 'stage_provenance': provenance, 'source_file_sizes': sizes, 'source_file_sha256': hashes,
                'stage_settings_required': {'ref': 'rsraki/' + stage_slug, 'is_private': True, 'enable_gpu': False, 'enable_internet': True, 'saved_version': 1, 'never_save_again': True, 'input_sources': []},
                'inference_settings_required': {'is_private': True, 'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
                    'docker_image': 'gcr.io/kaggle-gpu-images/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
                    'competition_sources': ['rsna-knee-abnormality-detection'],
                    'dataset_sources': [i['ref'] for i in parent_manifest['fixed_inputs'][1:3]], 'kernel_sources': ['rsraki/' + stage_slug + '/1'],
                    'do_not_also_attach_R6_dataset': True},
                'comparison': {'author_code_cells': checks, 'only_source_edit': branch.strip(), 'finder_AST_only': True,
                    'all_other_code_bytes_same': True, 'four_R6_folds_and_flip_TTA_true_preserved': True,
                    'embedded_modules': {name: sha(text.encode()) for name, text in sorted(modules.items())},
                    'original_two_markdown_cells_and_credits_unchanged': True, 'historical_outputs_and_counts_removed': True,
                    'extra_guard': 'bounded exact stage mount, exact 9 sizes/SHA before model load; no model deserialization'},
                'public_model_metadata': public,
                'verification_boundary': {'local_network': False, 'credentials_read': False, 'model_bodies_downloaded': False,
                    'original_scientific_code_executed': False, 'MRI': False, 'train': False, 'upload': False, 'submit': False,
                    'future_remote_download_endpoint_verified': False, 'future_stage_completed': False, 'future_mount_GPU_forward': False,
                    'own_Public_score': None, 'success_probability_estimated': False},
                'manual_next_steps_only_if_P029_mount_guard_fails': [
                    'Confirm P029 result; keep this fallback unused if the exact R6 V1 Input mounted correctly.',
                    'Review code, credits and model-use conditions; create exact private CPU InternetON stage once, without competitions or source MRI.',
                    'Require stage COMPLETE, all_nine_files_verified receipt and saved version exactly1; record its internal version ID. Never update this stage.',
                    'Attach only its version1 output plus original OrthoV1/SWAV1/competition to new private inference ref; inspect actual saved settings/InputIDs.',
                    'Run a fresh InternetOFF T4 batch after review; require early exact SHA guard and genuine visible CSV validation before any separate manual submission.']}
    with (output / 'manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    return manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='A new local run directory; existing directories are rejected')
    parser.add_argument('--stage-slug', required=True, help='Explicit new rsraki private stage slug; save only one V1 remotely')
    args = parser.parse_args(argv)
    result = prepare(args.output_dir, args.stage_slug)
    print(json.dumps({'status': result['status'], 'files': result['files']}, indent=2))

if __name__ == '__main__':
    main()
