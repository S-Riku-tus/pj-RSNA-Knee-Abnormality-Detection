"""Kaggle CPU-only fixed E12 integrity/reconstruction, no MRI, fit or API writes."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

POLICY_SHA = '9b7d6e581be8aa6ac4f83fad910f1399488183d2e2ba6defdc29097723c2f5de'
GENERIC_TRANSPORT_SHA = 'b48ff9f89999a8198f0f03d1324bd8239018cfec42788e97c38d59fdab9e598a'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, inventory, destination, *, exact):
    """Positive hash inventory, no extraction helper or existing paths overwritten."""
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), 'Fresh extraction destination required')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)) == len({n.casefold() for n in names}), 'No duplicate/colliding ZIP paths')
        require(set(inventory) == set(names) if exact else set(inventory).issubset(names), 'Positive ZIP inventory differs')
        require(sum(v['bytes'] for v in inventory.values()) <= 256*1024**2, 'Bounded positive extraction size')
        chosen = [i for i in infos if i.filename in inventory]
        for info in chosen:
            p = PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename and ':' not in info.filename
                    and not info.is_dir() and info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 1
                    and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), 'Unsafe ZIP member refused')
            require(info.file_size == inventory[info.filename]['bytes'] and hashlib.sha256(z.read(info)).hexdigest() == inventory[info.filename]['sha256'], 'ZIP bytes/hash differ')
        destination.mkdir(parents=True, exist_ok=False)
        for info in chosen:
            target = destination/Path(*PurePosixPath(info.filename).parts)
            require(target.resolve().is_relative_to(destination.resolve()), 'Extraction escape refused')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(z.read(info))
            require(sha(target) == inventory[info.filename]['sha256'], 'Written extraction differs')


def child(root, fold, output):
    # This entry point is launched in an isolated CPU process for each encoder.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import importlib

    import torch
    torch.set_num_threads(1)
    root, output = Path(root), Path(output)
    policy = json.loads((root/'qualification-policy.json').read_bytes())
    require(sha(root/'qualification-policy.json') == POLICY_SHA, 'Fixed new qualification policy')
    ref = policy['fixed_checkpoints'][str(fold)]
    checkpoint_path = root/('fold%d/epoch_012.pt'%fold)
    require(checkpoint_path.stat().st_size == ref['bytes'] and sha(checkpoint_path) == ref['sha256'], 'Exact fixed E12 checkpoint bytes')
    sys.path.insert(0, str(root))
    trainer = importlib.import_module('source14.native_training_fast16')
    model_module = importlib.import_module('source14.finetune_model')
    require(trainer.__file__ == str(root/'source14/native_training_fast16.py') and trainer.NativeTripletMIL is model_module.NativeTripletMIL, 'Exact isolated producer class path')
    checkpoint = trainer.load_native_checkpoint(checkpoint_path, allow_artificial=False)
    require(checkpoint['fold'] == fold and checkpoint['epoch'] == 12 and checkpoint['config'] == ref['config']
            and checkpoint['encoder_provenance'] == ref['encoder_provenance'] and checkpoint['gold_used_for_training_or_selection'] is False, 'Fixed E12 source/config/provenance/ownGold identity')
    model = trainer.new_native_model(root/'generic/repo', root/'generic/dinov2_vits14_pretrain.pth', ref['encoder_provenance'], ref['config'], device='cpu')
    prefix = trainer.prefix_digest(model)
    require(prefix == checkpoint['frozen_prefix_sha256'], 'Generic prefix before apply')
    require(trainer.apply_native_checkpoint(model, checkpoint, ref['encoder_provenance']) is model, 'Strict apply identity')
    model.eval().requires_grad_(False)
    require(trainer.prefix_digest(model) == prefix, 'Frozen prefix after apply')
    suffix = {k:v for k,v in model.encoder.state_dict().items() if k in trainer.suffix_keys(model)}
    head = model.head.state_dict()
    tensor_counts = {}
    for part, values in [('encoder_suffix', suffix), ('head', head)]:
        require(set(values) == set(checkpoint[part]) and all(torch.equal(v, checkpoint[part][k]) and torch.isfinite(v).all() and v.dtype == torch.float32 for k,v in values.items()), 'Exact finite FP32 strict tensor roundtrip')
        tensor_counts[part] = len(values)
    require(not model.training and not model.encoder.training and all(p.device.type == 'cpu' and not p.requires_grad for p in model.parameters()), 'CPU eval/no gradients')
    state, generator = torch.get_rng_state().clone(), model._gamma_generator.get_state().clone()
    require(model.sample_training_gamma(1, 'cpu') is None and model._gamma_draws == 0
            and torch.equal(state, torch.get_rng_state()) and torch.equal(generator, model._gamma_generator.get_state()), 'Eval gamma must draw nothing')
    # FP32 normalized triplet -> quantized CLS -> exact unchanged aggregation head.
    im = (torch.arange(3*392*392).reshape(1,3,392,392) % 251).float()/255
    mean, std = [im.new_tensor(v)[None,:,None,None] for v in ([.485,.456,.406],[.229,.224,.225])]
    with torch.no_grad():
        feature = model.encode_normalized((im-mean)/std)
        require(feature.shape == (1,384) and torch.isfinite(feature).all(), 'Artificial native392 encoder output')
        features = feature.new_zeros((1,3,48,384))
        features[0,0,0] = feature[0]
        mask = torch.zeros(1,3,48,dtype=torch.bool)
        mask[0,0,0] = True
        positions, planes, flags = torch.zeros(1,3,48), torch.zeros(1,3,dtype=torch.long), torch.zeros(1,3,2)
        logits = model.head(features, mask, positions, planes, flags)
        require(logits.shape == (1,12) and torch.isfinite(logits).all(), 'Artificial12 output contract')
        poisoned = features.clone()
        poisoned[~mask] = float('nan')
        poisoned_positions = positions.clone()
        poisoned_positions[~mask] = float('nan')
        require(torch.equal(model.head(poisoned,mask,poisoned_positions,planes,flags), logits), 'Masked padding invariance')
        refused = False
        try:
            model.head(features,torch.zeros_like(mask),positions,planes,flags)
        except ValueError:
            refused = True
        require(refused, 'All-missing image study refused')
    require(not torch.cuda.is_initialized() and trainer.prefix_digest(model) == prefix and model._gamma_draws == 0, 'No CUDA/frozen prefix drift/eval gamma')
    result = {'complete':True,'fold':fold,'epoch':12,'checkpoint_SHA':ref['sha256'],'frozen_prefix_SHA':prefix,
              'strict_tensor_counts':tensor_counts,'finite_exact_suffix_head_roundtrip':True,'eval_gamma_disabled':True,
              'artificial_FP32_normalized392_encoder_and_head':True,'all12_finite':True,'masked_padding_exact':True,
              'allmissing_image_rejected':True,'Torch':torch.__version__,'CUDA_initialized':False,
              'MRI_Report_Gold_training_reads':0,'full_native_model_CUDA_forward_qualified':False,
              'original_F009_engineering_HOLD_unchanged':True,'qualified_for_formal_Submit':False}
    require(not output.exists(), 'Fresh CPU result')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('F013_CPU_FOLD_PASS',fold,flush=True)


def run(package_mount, generic_mount, expected_package_manifest_SHA, output):
    require(Path('/kaggle/working').is_dir(), 'Kaggle-only CPU qualification')
    package_mount, generic_mount, output = map(Path,(package_mount,generic_mount,output))
    require(not output.exists(), 'Fresh CPU qualifier output')
    manifest_path = package_mount/'package-manifest.json'
    require(sha(manifest_path) == expected_package_manifest_SHA, 'Pinned private package manifest')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['policy_SHA'] == POLICY_SHA and manifest['originalF009HOLD_untouched'] and not manifest['legacy_release_pass_claim'], 'Distinct standalone qualification scope')
    archive = package_mount/manifest['archive']['path']
    require(archive.stat().st_size == manifest['archive']['bytes'] and sha(archive) == manifest['archive']['sha256'], 'Pinned private package body')
    output.mkdir(exist_ok=False)
    root = Path('/tmp/f013-fixed-E12-CPU-qualification')
    extract(archive,manifest['members'],root,exact=True)
    transport_path = root/'generic-transport-manifest.json'
    require(sha(transport_path) == GENERIC_TRANSPORT_SHA and sha(generic_mount/'transport-manifest.json') == GENERIC_TRANSPORT_SHA, 'Pinned generic Input manifest')
    transport = json.loads(transport_path.read_bytes())
    generic_archive = generic_mount/transport['archive']['path']
    require(generic_archive.stat().st_size == transport['archive']['bytes'] and sha(generic_archive) == transport['archive']['sha256'], 'Exact generic Input archive')
    inventory = {k:v for k,v in transport['members'].items() if k.startswith('generic/')}
    require(len(inventory) == 162, 'Exact generic source/weight162 members')
    extracted = Path('/tmp/f013-generic-only')
    extract(generic_archive,inventory,extracted,exact=False)
    (root/'generic').symlink_to(extracted/'generic',target_is_directory=True)
    receipts = []
    for fold in (0,1):
        stdout, stderr = output/('fold%d.stdout.txt'%fold), output/('fold%d.stderr.txt'%fold)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with stdout.open('w') as out,stderr.open('w') as err:
            result = subprocess.run([sys.executable,'-u',__file__,'--child',str(fold),'--root',str(root),'--output',str(output/('fold%d.json'%fold))],stdout=out,stderr=err,env=env,timeout=1200,check=False)
        require(result.returncode == 0, 'CPU reconstruction child failed for fold%d'%fold)
        receipt = json.loads((output/('fold%d.json'%fold)).read_bytes())
        require(receipt['complete'] and receipt['fold'] == fold and not receipt['CUDA_initialized'], 'Each fold CPU evidence complete')
        receipts.append(receipt)
    completed = {'complete':True,'folds':receipts,'policy_SHA':POLICY_SHA,'package_manifest_SHA':expected_package_manifest_SHA,
                 'MRI_Report_Gold_newfit_Save_Submit':0,'GPU_forward_or_actualMRI_qualified':False,
                 'original_F009_HOLD_unchanged':True,'qualified_for_formal_Submit':False}
    (output/'completion.json').write_text(json.dumps(completed,indent=2)+'\n',encoding='utf8')
    print('F013_FIXED_E12_CPU_QUALIFICATION_PASS',flush=True)
    return completed


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',type=int,choices=(0,1),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    child(args.root,args.child,args.output)
