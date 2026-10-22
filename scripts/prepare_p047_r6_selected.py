"""Build an inert P047 payload only from a completed, positive frozen R6 selection."""
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

import evaluate_r6_overall_probe as O
import p047_r6_transform as T
import prepare_p040_p041_factorial as F
from r6_pair_probe_runtime import ARTIFICIAL_WORKER

ROOT = Path(__file__).resolve().parents[1]
P041 = ROOT/'artifacts/kaggle/p041-p033-reader-20261019-v1'


def selected(result):
    F.require(result.get('complete') is True and result.get('own_Gold_fit_select') == 0, 'Complete selection with no ownGold required')
    choice = result.get('selected_variant')
    F.require(choice in T.CHOICES, 'No positive candidate selected; no payload permitted')
    if result['mode'] == 'four_choices_exact_common_cohort':
        F.require(result['policy_SHA'] == O.C.POLICY_SHA and result['original_policy_SHA'] == O.E.B.POLICY_SHA,
                  'Frozen overall policy required')
        F.require(O.choose(result['fixed10_deltas']) == choice, 'Exact both-bank/minimum-delta selection required')
        F.require(all(result['paired_successes'][bank]/result['counts'][bank] >= .98 for bank in ('0', '1')),
                  'Common cohort98% both banks required')
        F.require(result['uncropped_fluid_exact_on_all_pairs'] and not result['crop_combined_replay_variants_evaluated'], 'Crop single-factor checks required')
    else:
        F.require(result['mode'] == 'original_three_only_crop_engineering_HOLD' and choice != 'crop320', 'Only declared fallback allowed')
        original = result['original_result']
        F.require(original['policy_SHA'] == O.E.B.POLICY_SHA and original['complete'] and original['own_Gold_fit_select'] == 0,
                  'Complete original selection required')
        F.require(O.E.choose(original['fixed10_deltas']) == choice, 'Original strict two-bank selector required')
        F.require(all(original['successes'][bank]/original['counts'][bank] >= .98 for bank in ('0', '1')), 'Original98% both banks required')
    return choice


def prepare(output, slug, selection_result):
    output, selection_result = Path(output), Path(selection_result).resolve()
    F.require(not output.exists() and not output.is_symlink(), 'Fresh artifact directory required')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}', slug), 'Own slug required')
    F.require(selection_result.is_relative_to((ROOT/'artifacts/research').resolve()) and selection_result.is_file()
              and not selection_result.is_symlink(), 'Existing local research result required')
    result = F.read(selection_result)
    choice = selected(result)
    F.require(F.pin(F.P033/'source.ipynb')['sha256'] == F.P033_SHA, 'Exact ownbest P033 required')
    F.require(F.pin(P041/'source.ipynb')['sha256'] == '0c06b802d34d6cee85b5156fb30ae1b20f7b09c747550639c40547e2ad1e5c7a', 'Exact operational guard required')
    F.require(F.pin(O.C.POLICY)['sha256'] == O.C.POLICY_SHA and F.pin(O.E.B.POLICY)['sha256'] == O.E.B.POLICY_SHA,
              'Both preregistrations unchanged')
    freeze_path = ROOT/'artifacts/research/20261020-R6-crop320-paired-prospective-v1/overall-selector-beforemetrics-freeze.json'
    freeze = F.read(freeze_path)
    F.require(freeze['before_any_new_probe_predictionCSV_or_numericlabel_evaluation'], 'Premetric selector freeze required')
    for name, item in freeze['source_pins'].items():
        F.require(F.pin(ROOT/name) == {'bytes': item['bytes'], 'sha256': item['SHA']}, 'Frozen selector/math implementation changed')
    parent = F.read(F.P033/'source.ipynb')
    science = [F.text(c) for c in parent['cells'] if c['cell_type']=='code']
    original = list(science)
    F.require(len(science) == 8, 'Exact8 parent cells required')
    science[5], descriptor = T.transform_stage(science[5], choice)
    for cell in science:
        ast.parse(cell.split('\n', 1)[1] if cell.startswith('%%writefile ') else cell)
    sources = ast.literal_eval(F.assignment(original[5], 'SOURCES')[1].value)
    runtime = (ROOT/'scripts/p047_r6_runtime.py').read_text(encoding='utf8')
    observer = (ROOT/'artifacts/kaggle/p045-p033-dino5-20261020-v2/runtime.py').read_text(encoding='utf8')
    F.require(hashlib.sha256(observer.encode()).hexdigest() == 'cda716ca11e7a2c237375a019966379ca3899cd5fdb6d1018e4c69575e99b537', 'Frozen parent contract observation required')
    arithmetic = (ROOT/'scripts/r6_pair_math.py').read_text(encoding='utf8')
    config = {'descriptor': descriptor, 'selection_result_SHA': F.pin(selection_result)['sha256'],
              'original_worker': sources['compact_infer.py'], 'original_worker_SHA': T.WORKER_SHA,
              'artificial_worker': ARTIFICIAL_WORKER, 'labels': list(O.E.TARGETS),
              'math_source': arithmetic, 'math_SHA': hashlib.sha256(arithmetic.encode()).hexdigest(),
              'observer_source': observer, 'observer_SHA': hashlib.sha256(observer.encode()).hexdigest()}
    old_loader = F.text(F.read(P041/'source.ipynb')['cells'][1])
    guard = ast.literal_eval(F.assignment(old_loader, '_factor_preflight_source')[1].value)
    guard_config = ast.literal_eval(F.assignment(old_loader, '_factor_preflight_config')[1].value)
    guard_config['producer_descriptor'] = {'candidate': 'P047', 'factor': choice, 'Reader_executed': False,
                                           'unused_Reader_Input_hashed': True, 'DINO_R7_Gaussian': False, 'own_Gold_fit_select': 0}
    loader = ('import hashlib\n'
              f'_p047_runtime={runtime!r}\n'
              f'assert hashlib.sha256(_p047_runtime.encode()).hexdigest()=={hashlib.sha256(runtime.encode()).hexdigest()!r}\n'
              "_p047_ns={'__name__':'p047_frozen_runtime'}\n"
              "exec(compile(_p047_runtime,'<p047-runtime>','exec'),_p047_ns)\n"
              f'_p047_guard_source={guard!r}\n'
              f'assert hashlib.sha256(_p047_guard_source.encode()).hexdigest()=={hashlib.sha256(guard.encode()).hexdigest()!r}\n'
              "_p047_guard_ns={'__name__':'p047_inherited_guard'}\n"
              "exec(compile(_p047_guard_source,'<p047-operational-guard>','exec'),_p047_guard_ns)\n"
              f"_p047_guard=_p047_guard_ns['preflight']({guard_config!r})\n")
    raw = json.dumps(science, ensure_ascii=False, separators=(',', ':')).encode()
    compressed = lzma.compress(raw, preset=6)
    invoke = ('import base64,lzma,json\n'
              f'_p047_packed=base64.b85decode({base64.b85encode(compressed).decode()!r})\n'
              f'assert hashlib.sha256(_p047_packed).hexdigest()=={hashlib.sha256(compressed).hexdigest()!r}\n'
              '_p047_raw=lzma.decompress(_p047_packed,memlimit=128*1024**2)\n'
              f'assert hashlib.sha256(_p047_raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n'
              f"_p047_result=_p047_ns['execute'](json.loads(_p047_raw),get_ipython(),{F.inventory(science)!r},{config!r})\n")
    notice = ('# P047: one weak-validated R6 factor on own0.958 parent\n\n'
              f'Selected factor: {choice}. Seven of the original eight P033 science cells remain byte exact; '
              'only R6 asset/setup cell changes as prescribed. Same fusion weights/A224=.35 and models. '
              'Fourfold synthetic originalON/selected direct GPU and replay must pass before real R6 test MRI. '
              'Decoder guard/7 Inputs inherited from P041; unused Reader data are hashed, Reader science is not run. '
              'No DINO/R7/Gaussian addition. Own Gold fit/select=0. Declared fold weak-validation has upstream '
              'Gold/auxiliary/membership uncertainty and does not prove a Public gain or patient independence.\n')
    book = {'cells': [{'cell_type':'markdown', 'metadata':{}, 'source':notice}, F.code(loader), F.code(invoke),
                      *[copy.deepcopy(c) for c in parent['cells'] if c['cell_type']=='markdown']],
            'metadata':copy.deepcopy(parent['metadata']), 'nbformat':4, 'nbformat_minor':5}
    text = json.dumps(book, ensure_ascii=False, indent=1)+'\n'
    F.require(len(text.encode()) < 950000, 'Source capacity exceeded')
    payload = copy.deepcopy(F.read(P041/'save-payload.json'))
    payload.update(slug=slug, new_title=slug.split('/')[1], text=text)
    manifest = {'at_UTC':datetime.now(timezone.utc).isoformat(), 'candidate':'P047', 'selected_variant':choice,
                'selection_result':{'path':selection_result.relative_to(ROOT).as_posix(), **F.pin(selection_result)},
                'source_SHA':hashlib.sha256(text.encode()).hexdigest(), 'source_bytes':len(text.encode()),
                'original_science':F.inventory(original), 'candidate_science':F.inventory(science),
                'seven_other_cells_byte_exact':all(original[i]==science[i] for i in range(8) if i!=5),
                'descriptor':descriptor, 'runtime':F.pin(ROOT/'scripts/p047_r6_runtime.py'),
                'original_policy_SHA':O.E.B.POLICY_SHA, 'overall_policy_SHA':O.C.POLICY_SHA,
                'decoder_guard_SHA':hashlib.sha256(guard.encode()).hexdigest(),
                'inputs':{k:payload[k] for k in ('dataset_data_sources','kernel_data_sources','competition_data_sources')},
                'settings':{k:payload[k] for k in ('is_private','enable_gpu','enable_internet','machine_shape','docker_image')},
                'local_model_MRI_GPU_Save_Submit':0, 'own_Public':None}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('source.ipynb',book),('save-payload.json',payload),('manifest.json',manifest),('science-cells.json',science)]:
        (output/name).write_text(text if name=='source.ipynb' else json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
    (output/'runtime.py').write_text(runtime,encoding='utf8',newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slug',required=True)
    parser.add_argument('--selection-result',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output,args.slug,args.selection_result),indent=2))
