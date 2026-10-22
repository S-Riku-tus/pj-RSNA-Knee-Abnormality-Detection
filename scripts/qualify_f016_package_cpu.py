"""実F016固定E12 packageの新namespace厳密CPU復元。GPU/MRI/ネットワークなし。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import f013_cpu_qualifier as C

SCOPE_SHA = '9d2c5ebba900ebdb274ef57452e90b2ff069d87cda21bf9bfbe2692b44f05f19'


def qualify(package, output, vendor):
    package, output, vendor = map(Path, (package, output, vendor))
    C.require(not output.exists(), '新規CPU資格出力')
    manifest_path = package / 'package-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    C.require(manifest['schema'] == 'F016_fixedE12_CNN_package_v1' and manifest['scope_SHA'] == SCOPE_SHA,
              '固定F016 package')
    archive = package / 'f016-models.bin'
    C.require(C.sha(archive) == manifest['archive']['sha256'] and archive.stat().st_size == manifest['archive']['bytes'],
              'package全byte')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    C.require('torch' not in sys.modules and not any(n == 'cnn_f016_source16' or n.startswith('cnn_f016_source16.') for n in sys.modules),
              '新規CPU process/namespace')
    parent = Path(tempfile.mkdtemp(prefix='F016-local-CPU-', dir=package.parent))
    restored = parent / 'verified'
    C.extract(archive, manifest['members'], restored, exact=True)
    audit = json.loads((restored / 'audit.json').read_bytes())
    C.require(audit['両fold候補条件'] is True and audit['scope_SHA'] == SCOPE_SHA
              and C.sha(restored / 'audit.json') == manifest['audit_SHA'], '事前採否条件を満たす実監査')
    sys.path[:0] = [str(vendor.resolve(strict=True)), str(restored)]
    import torch
    torch.set_num_threads(1)
    model_module = importlib.import_module('cnn_f016_source16.convnext_model')
    config, provenance = manifest['config'], manifest['provenance']
    images = torch.zeros((1, 3, 48, 3, 392, 392), dtype=torch.uint8)
    mask = torch.zeros((1, 3, 48), dtype=torch.bool)
    for series, window in ((0, 0), (0, 20), (1, 13), (2, 25)):
        mask[0, series, window] = True
    images[mask] = (torch.arange(4 * 3 * 392 * 392).reshape(4, 3, 392, 392) % 251).to(torch.uint8)
    batch = {'images': images, 'mask': mask, 'positions': torch.linspace(0, 1, 48)[None, None].expand(1, 3, 48).clone(),
             'planes': torch.tensor([[0, 1, 2]], dtype=torch.long), 'flags': torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])}
    result = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'schema': 'F016_actual_fixedE12_CPU_qualification_v1',
              'scope_SHA': SCOPE_SHA, 'manifest_SHA': C.sha(manifest_path), 'audit_SHA': manifest['audit_SHA'],
              'folds': {}, 'actual_GPU_MRI_qualified': False, 'Gold_labels_training_network': 0}
    for fold in (0, 1):
        checkpoint = model_module.load_checkpoint(restored / f'fold{fold}/epoch_012.pt')
        C.require(checkpoint['fold'] == fold and checkpoint['epoch'] == 12 and checkpoint['artificial'] is False
                  and checkpoint['config'] == config and checkpoint['encoder_provenance'] == provenance,
                  '人工/中間epochを実E12として受理しない')
        CPU_rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            model = model_module.create_model(restored / 'generic.safetensors', config, provenance, device='cpu')
        model_module.apply_checkpoint(model, checkpoint, provenance)
        model.eval().requires_grad_(False)
        checks = {'新規exact_class': type(model) is model_module.ConvNeXtTripletMIL,
                  'prefix一致': model_module.prefix_digest(model) == checkpoint['frozen_prefix_sha256']}
        for part, state in [('encoder_suffix', {k: v for k, v in model.encoder.state_dict().items() if k in model_module.suffix_keys(model)}),
                            ('head', model.head.state_dict())]:
            checks[part + '独立完全一致'] = set(state) == set(checkpoint[part]) and all(v.dtype == torch.float32
                and bool(torch.isfinite(v).all()) and torch.equal(v, checkpoint[part][k]) for k, v in state.items())
        before = model_module.tensor_digest(model.state_dict())
        images[~mask] = 0
        with torch.inference_mode():
            logits = model(**batch).float()
            checks['人工4窓有限12出力'] = logits.shape == (1, 12) and bool(torch.isfinite(logits).all())
            images[~mask] = 255
            checks['padding不変'] = torch.equal(logits, model(**batch).float())
            refused = False
            try:
                model(**dict(batch, mask=torch.zeros_like(mask)))
            except ValueError:
                refused = True
            checks['全欠損拒否'] = refused
        checks.update(全state不変=model_module.tensor_digest(model.state_dict()) == before,
                      CPU_RNG不変=torch.equal(CPU_rng, torch.get_rng_state()), CUDA未初期化=not torch.cuda.is_initialized())
        C.require(all(checks.values()), '実E12 CPU資格の不一致')
        result['folds'][str(fold)] = {'checks': checks, 'checkpoint_SHA': C.sha(restored / f'fold{fold}/epoch_012.pt'),
                                    'full_state_SHA': before, 'synthetic_probabilities': logits.sigmoid()[0].tolist()}
        del model, checkpoint, logits
    result['passed'] = True
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'passed': qualify(**vars(args))['passed']}, ensure_ascii=False))
