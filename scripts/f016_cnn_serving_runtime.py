"""F016専用constructor接続。旧preview/旧CLS資格を変更せず、根拠不在は拒否。"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import f013_cpu_qualifier as C
from f013_p039_bridge import ROUTER_SHA, derive_execute

SCOPE_SHA = '9d2c5ebba900ebdb274ef57452e90b2ff069d87cda21bf9bfbe2692b44f05f19'
FAMILIES = ('CLS', 'REG')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare_package(runtime):
    C.require('torch' not in sys.modules, '原bootstrap前のTorch未importを維持')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    binding = json.loads((Path(runtime) / 'F016-package-binding.json').read_bytes())
    C.require(binding['scope_SHA'] == SCOPE_SHA and binding['fixed_epoch'] == 12
              and binding['alpha'] == .1 and binding['family_weight'] == .5, '固定F016資格・係数')
    slug = binding['dataset_slug']
    C.require(type(slug) is str and slug.startswith('f016-tiny-e12-') and '/' not in slug and '..' not in slug,
              '本人固定package名')
    root = Path('/kaggle/input')
    choices = [root / slug, *root.glob('datasets/rsraki/' + slug), *root.glob('datasets/rsraki/' + slug + '/*')]
    found = list(dict.fromkeys(p for p in choices if (p / 'package-manifest.json').is_file()))
    C.require(len(found) == 1, 'F016固定Inputが一意')
    mount = found[0]
    C.require(C.sha(mount / 'package-manifest.json') == binding['manifest_SHA'], '固定package manifest')
    manifest = json.loads((mount / 'package-manifest.json').read_bytes())
    C.require(manifest['schema'] == 'F016_fixedE12_CNN_package_v1' and manifest['scope_SHA'] == SCOPE_SHA
              and manifest['alpha'] == .1 and manifest['family_weight'] == .5, 'F016 schema')
    cpu_path = Path(runtime) / 'F016-current-CPU-evidence.json'
    C.require(C.sha(cpu_path) == binding['CPU_evidence_SHA'], '実E12の事前CPU証拠を固定')
    cpu = json.loads(cpu_path.read_bytes())
    C.require(cpu['schema'] == 'F016_actual_fixedE12_CPU_qualification_v1' and cpu['passed'] is True
              and cpu['scope_SHA'] == SCOPE_SHA and cpu['manifest_SHA'] == binding['manifest_SHA']
              and cpu['audit_SHA'] == manifest['audit_SHA'] and set(cpu['folds']) == {'0', '1'}, '同一packageの実CPU資格')
    for fold in ('0', '1'):
        C.require(cpu['folds'][fold]['checkpoint_SHA'] == manifest['checkpoints'][fold]['sha256']
                  and cpu['folds'][fold]['checks'] and all(v is True for v in cpu['folds'][fold]['checks'].values()),
                  '空でない固定E12 CPU全確認')
    archive = mount / 'f016-models.bin'
    C.require(archive.stat().st_size == manifest['archive']['bytes'] and C.sha(archive) == manifest['archive']['sha256'],
              '固定package全byte')
    package = Path(tempfile.mkdtemp(prefix='F016-package-parent-')) / 'verified'
    C.extract(archive, manifest['members'], package, exact=True)
    audit = json.loads((package / 'audit.json').read_bytes())
    C.require(C.sha(package / 'audit.json') == manifest['audit_SHA'] == binding['audit_SHA']
              and C.sha(package / 'scope-policy.json') == SCOPE_SHA and audit['両fold候補条件'] is True
              and audit['scope_SHA'] == SCOPE_SHA and audit['fixed_epoch'] == 12 and audit['Gold数値利用'] is False,
              '両foldの実家族対照・R6代理条件')
    C.require(sys.platform == 'linux' and platform.machine() == 'x86_64' and platform.python_implementation() == 'CPython'
              and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', '現Linux/cublas guard')
    C.require(not any(name == 'timm' or name.startswith('timm.') or name == 'cnn_f016_source16'
                      or name.startswith('cnn_f016_source16.') for name in sys.modules), '新規CNN namespace必須')
    vendor = package.parent / 'vendor'
    C.require(not vendor.exists(), '新規vendor')
    wheels = sorted((package / 'wheels').glob('*.whl'))
    C.require(len(wheels) == 3, '固定Linux wheel3本')
    completed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                               '--disable-pip-version-check', '--target', str(vendor), *map(str, wheels)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=False,
                              env={**os.environ, 'PIP_NO_INDEX': '1', 'HF_HUB_OFFLINE': '1'})
    C.require(completed.returncode == 0 and vendor.is_dir(), '固定wheelだけを隔離vendorへ導入')
    return package, vendor, manifest, binding


def new_pair(old, manifest, manifest_sha):
    pair = copy.deepcopy(old)
    cls = {'schema_version': 'F016_fixed_E12_CNN_pair_v1', 'bindings': {}}
    for fold in (0, 1):
        cls['bindings'][fold] = {'config': manifest['config'], 'checkpoint_sha256': manifest['checkpoints'][str(fold)]['sha256'],
                                  'selected_epoch': 12, 'fold': fold, 'selection': 'F016事前固定E12・本人Gold不使用',
                                  'producer_class': 'cnn_f016_source16.convnext_model.ConvNeXtTripletMIL'}
    cls['pair_fingerprint_sha256'] = object_sha(cls)
    pair['family_pairs']['CLS'] = cls
    pair['schema_version'] = 'F016_CNN_original_REG_pair_v1'
    pair['model_classes']['CLS'] = 'cnn_f016_source16.convnext_model.ConvNeXtTripletMIL'
    pair['experimental_factor'] = 'F013 CLS2本をF016 Tiny固定E12へ置換。同REG・fold/家族50%・native10を維持。'
    pair['original_P039_pair_SHA_reference'] = old['pair_fingerprint_sha256']
    if 'family_fold_manifest_sha256' in pair:
        pair['family_fold_manifest_sha256']['CLS'] = {f: manifest_sha for f in (0, 1)}
    pair.pop('pair_fingerprint_sha256', None)
    pair['pair_fingerprint_sha256'] = object_sha(pair)
    return pair


def initial(args, plan, determinism):
    return {'schema_version': 'F016_CNN_raw_component_v1', 'complete': False,
            'phase_contract_passed': False, 'startup_complete': False, 'determinism': determinism,
            'F016_scope_SHA': SCOPE_SHA, 'pair_fingerprint_sha256': plan['pair']['pair_fingerprint_sha256'],
            'input_receipt': plan['input_receipt'], 'baseline_csv_sha256': plan['baseline_csv_sha256'],
            'old_F009_HOLD_unchanged': True, 'old_CLS_receipt_certifies_new_CNN': False,
            'teacher_Report_read': False, 'own_Gold_fit_or_selection': False, 'accuracy_evaluated': False,
            'final_CSV_written': False, 'remaining_child_budget_seconds': args.budget_seconds}


class Mixed:
    def __init__(self, original, plan, package, vendor, manifest, binding):
        self.original, self.plan = original, plan
        self.package, self.vendor, self.manifest, self.binding = package, vendor, manifest, binding
        self.old_pair = copy.deepcopy(plan['pair'])

    def reconstruct_cpu_models(self, roots, pair, views, documents, permission, **kwargs):
        import torch

        core = kwargs['core']
        models, old = self.original.reconstruct_cpu_models(roots, self.old_pair, views, documents, permission, **kwargs)
        C.require(old['model_fingerprints'] == self.plan['values']['CPU_smoke']['model_fingerprints'],
                  '原REGと原constructorは原資格を通す')
        C.require(not any(n == 'timm' or n.startswith('timm.') or n == 'safetensors' or n.startswith('safetensors.')
                          or n == 'huggingface_hub' or n.startswith('huggingface_hub.')
                          for n in sys.modules), '旧parentのCNN依存moduleを流用しない')
        sys.path[:0] = [str(self.vendor), str(self.package)]
        module = importlib.import_module('cnn_f016_source16.convnext_model')
        C.require(Path(module.__file__).resolve() == (self.package / 'cnn_f016_source16/convnext_model.py').resolve(),
                  '新規exact CNN producer path')
        self.module = module
        expected = {k: copy.deepcopy(v) for k, v in old['model_fingerprints'].items() if k.startswith('REG:')}
        config, provenance = self.manifest['config'], self.manifest['provenance']
        for fold in (0, 1):
            path = self.package / f'fold{fold}/epoch_012.pt'
            checkpoint = module.load_checkpoint(path)
            C.require(C.sha(path) == self.manifest['checkpoints'][str(fold)]['sha256'] and checkpoint['fold'] == fold
                      and checkpoint['epoch'] == 12 and checkpoint['config'] == config
                      and checkpoint['encoder_provenance'] == provenance and checkpoint['gold_used_for_training_or_selection'] is False,
                      '実学習E12の厳密なCNN identity')
            with torch.random.fork_rng(devices=[]):
                model = module.create_model(self.package / 'generic.safetensors', config, provenance, device='cpu')
            for name, version in (('timm', '1.0.22'), ('safetensors', '0.6.2'), ('huggingface_hub', '0.34.4')):
                library = importlib.import_module(name)
                C.require(library.__version__ == version and Path(library.__file__).resolve().is_relative_to(self.vendor.resolve()),
                          '固定wheelの隔離vendorから実import')
            C.require(module.apply_checkpoint(model, checkpoint, provenance) is model, '完全generic初期化後にCNN selected stateを厳密復元')
            model.eval().requires_grad_(False)
            C.require(type(model) is module.ConvNeXtTripletMIL and module.prefix_digest(model) == checkpoint['frozen_prefix_sha256'],
                      'CNN classとprefix一致')
            for part, state in [('encoder_suffix', {k: v for k, v in model.encoder.state_dict().items() if k in module.suffix_keys(model)}),
                                ('head', model.head.state_dict())]:
                C.require(set(state) == set(checkpoint[part]) and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                          and torch.equal(v, checkpoint[part][k]) for k, v in state.items()), '独立selected tensor完全一致')
            expected['CLS:' + str(fold)] = {'model_class': type(model).__module__ + '.' + type(model).__name__,
                'selected_epoch': 12, 'checkpoint_sha256': C.sha(path), 'frozen_prefix_sha256': module.prefix_digest(model),
                'encoder_suffix_tensor_sha256': core.tensor_digest(checkpoint['encoder_suffix']),
                'head_tensor_sha256': core.tensor_digest(checkpoint['head']), 'package_manifest_SHA': self.binding['manifest_SHA']}
            models['CLS'][fold] = model
        self.classes = {f: {n: type(models[f][n]) for n in (0, 1)} for f in FAMILIES}
        checked = self.original.p.check_alias_pairs({f + ':' + str(n): models[f][n] for f in FAMILIES for n in (0, 1)},
                                                    lambda m: m.state_dict(), core.storage_aliases)
        C.require(checked == 6, 'CPU全6 storage組の非共有')
        self.full_states = {f + ':' + str(n): core.tensor_digest(models[f][n].state_dict()) for f in FAMILIES for n in (0, 1)}
        self.plan['F016_expected_fingerprints'] = expected
        self.component = {'schema_version': 'F016_CNN_current_CPU_component_v1', 'complete': True,
                          'model_fingerprints': expected, 'CPU_storage_alias_pairs_checked': checked, 'CPU_storage_alias_count': 0,
                          'generic_factory_calls': old['generic_factory_calls'] + 2,
                          'generic_factory_calls_by_role': {'original_DINO': old['generic_factory_calls'], 'F016_CNN': 2},
                          'model_classes': pair['model_classes'],
                          'original_P039_component_reference': old, 'original_reference_scope': 'REG構築の資格。新CNNの資格にはしない。',
                          'new_CNN_strict_roundtrip': True, 'full_CPU_state_sha': self.full_states}
        return models, self.component

    def verify_full_state(self, models, core):
        for family in FAMILIES:
            for device in ('cuda:0', 'cuda:1'):
                for fold in (0, 1):
                    C.require(core.tensor_digest(models[family][device][fold].state_dict()) == self.full_states[family + ':' + str(fold)],
                              '全prefix/suffix/headがCPU復元時から不変')

    def models_on_two_devices(self, cpu_models, core):
        import torch

        models = {f: core.models_on_two_devices(cpu_models[f]) for f in FAMILIES}
        for family in FAMILIES:
            for device in ('cuda:0', 'cuda:1'):
                for fold in (0, 1):
                    model = models[family][device][fold]
                    C.require(type(model) is self.classes[family][fold] and not model.training and not model.encoder.training
                              and all(str(v.device) == device and v.dtype == torch.float32 and not v.requires_grad for v in model.parameters()),
                              '現8モデルのexact class/device/FP32/frozen')
        checked = sum(self.original.p.check_alias_pairs({f + ':' + str(n): models[f][d][n] for f in FAMILIES for n in (0, 1)},
                      lambda m: m.state_dict(), core.storage_aliases) for d in ('cuda:0', 'cuda:1'))
        C.require(checked == 12, 'GPU全12 storage組の非共有')
        self.verify_full_state(models, core)
        self.component['current_GPU_synthetic'] = artificial(models, core, self.plan['pair'], self.original)
        self.verify_full_state(models, core)
        self.active_models, self.core = models, core
        return models, checked

    def consume_four(self, *args, **kwargs):
        return self.original.consume_four(*args, **kwargs)


def artificial(models, core, pair, original):
    import torch

    from rsna_knee.native_training_fast16 import tensor_study
    from rsna_knee.triplet_cache import array_hashes

    images = torch.zeros((3, 48, 3, 392, 392), dtype=torch.uint8)
    mask = torch.zeros((3, 48), dtype=torch.bool)
    for series, window in ((0, 0), (0, 20), (1, 13), (2, 25)):
        mask[series, window] = True
    images[mask] = (torch.arange(4 * 3 * 392 * 392).reshape(4, 3, 392, 392) % 251).to(torch.uint8)
    arrays = {'images': images.numpy(), 'mask': mask.numpy(),
              'positions': torch.linspace(0, 1, 48)[None, :].expand(3, 48).clone().numpy(),
              'planes': torch.tensor([0, 1, 2], dtype=torch.long).numpy(),
              'flags': torch.tensor([[1., 0.], [0., 1.], [1., 1.]]).numpy()}
    pixel = {'native_SHA_verified_before_model': True, 'valid_windows': 4, 'array_sha256': array_hashes(arrays)}
    CPU_rng = torch.get_rng_state().clone()
    GPU_rng = [torch.cuda.get_rng_state(i).clone() for i in (0, 1)]
    result = {}
    for device in ('cuda:0', 'cuda:1'):
        inputs = tensor_study(arrays, device=device)
        references = {}
        with torch.cuda.device(device), torch.inference_mode():
            for family in FAMILIES:
                probabilities = []
                for fold in (0, 1):
                    model = models[family][device][fold]
                    logits = model(**inputs).float()
                    C.require(logits.shape == (1, 12) and bool(torch.isfinite(logits).all()), '現モデルの有限12出力')
                    poisoned = dict(inputs, images=inputs['images'].clone())
                    poisoned['images'][~inputs['mask']] = 255
                    C.require(torch.equal(logits, model(**poisoned).float()), 'padding poison不変')
                    refused = False
                    try:
                        model(**dict(inputs, mask=torch.zeros_like(inputs['mask'])))
                    except ValueError:
                        refused = True
                    C.require(refused and getattr(model, 'study_gamma', None) is None, '全欠損拒否・推論gammaなし')
                    probabilities.append(logits.sigmoid()[0].cpu().tolist())
                references[family] = core.probability_mean_reference(*probabilities)
        predicted, receipts = {}, {}
        family_predictions, family_receipts = {f: {} for f in FAMILIES}, {f: {} for f in FAMILIES}
        original.consume_four(device, ('F016_artificial', arrays, pixel), models=models, predictions=predicted, receipts=receipts,
                              lock=threading.Lock(), core=core, pair=pair, family_predictions=family_predictions,
                              family_receipts=family_receipts, array_hashes=array_hashes)
        C.require(predicted['F016_artificial'] == core.probability_mean_reference(references['CLS'], references['REG']),
                  '原consume_fourとfold50→家族50の別計算が一致')
        result[device] = {'finite_all_four_current_models': True, 'padding_exact': True, 'allmissing_refused': True,
                          'no_gamma': True, 'original_consumer_FP32_exact': True, 'probabilities': predicted['F016_artificial']}
    C.require(torch.equal(CPU_rng, torch.get_rng_state()) and all(torch.equal(GPU_rng[i], torch.cuda.get_rng_state(i)) for i in (0, 1)),
              'CPU/CUDA RNGを推論で消費しない')
    return {'complete': True, 'devices': result, 'MRI_Report_Gold_reads': 0}


def execute(args, active, plan, router, *, task_source_sha256, started, runtime):
    import isolated_mixed_runtime as original
    from p032blend.operational_uid_recovery import execute_router_with_operational_recovery

    C.require(args.native_executor == 'isolated-uid' and task_source_sha256 == ROUTER_SHA, '原UID180 readerの固定')
    package, vendor, manifest, binding = prepare_package(runtime)
    mixed = Mixed(original, plan, package, vendor, manifest, binding)
    replacement = new_pair(plan['pair'], manifest, binding['manifest_SHA'])
    plan['pair'].clear()
    plan['pair'].update(replacement)
    plan.update(F016_mixed=mixed, F016_initial=initial)
    tree = derive_execute(Path(router.__file__).read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace('F013_', 'F016_')
    clone = ModuleType('_f016_current_router')
    clone.__dict__.update(router.__dict__)
    clone.__name__ = '_f016_current_router'
    exec(compile(tree, 'F016-current-execute', 'exec'), clone.__dict__)
    C.require(clone.prepare_hidden_task is router.prepare_hidden_task and clone.verify_hidden_payload is router.verify_hidden_payload,
              '原decoder callablesを維持')
    with tempfile.TemporaryDirectory(prefix='F016-operational-', dir=active.output.parent) as temporary:
        intermediate = copy.copy(active)
        intermediate.output = Path(temporary) / 'operational.json'
        result = execute_router_with_operational_recovery(clone, intermediate, plan, started,
                    task_source_sha256=task_source_sha256, uid_deadline_seconds=180.)
        C.require(mixed.component['current_GPU_synthetic']['complete'], '今回のGPU資格前にMRIへ進めない')
        mixed.verify_full_state(mixed.active_models, mixed.core)
        result.update(F016_current_constructor_qualification=True, F016_scope_SHA=SCOPE_SHA,
                      F016_package_manifest_SHA=binding['manifest_SHA'], old_F009_HOLD_unchanged=True,
                      all_current_tensors_unchanged_after_MRI=True,
                      generic_factory_calls=mixed.component['generic_factory_calls'],
                      generic_factory_calls_by_role=mixed.component['generic_factory_calls_by_role'],
                      model_classes=plan['pair']['model_classes'])
        if 'unchanged_model_startup_component' in result:
            result['F016_current_model_startup_component'] = result.pop('unchanged_model_startup_component')
            result.pop('unchanged_model_startup_component_scope', None)
        result['runtime_changes'].update(reader_source_bytes_changed=False, two_CLS_model_sources_changed=True,
                                         REG_model_sources_changed=False, success_four_fold_FP32_math_changed=False)
        result['legacy_assertions_scope'] = '原reader/REG/演算契約。新CNNはF016 constructor資格で検証。旧CLS資格は流用しない。'
        C.require(not active.output.exists(), '新規F016結果')
        active.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result
