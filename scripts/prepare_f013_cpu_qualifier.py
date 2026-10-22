"""Offline CPU qualification notebook builder. No upload/Save/Submit calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import prepare_p040_p041_factorial as F
from f013_cpu_qualifier import POLICY_SHA

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/staging/f013-fixed-E12-qualification-20261020-v2'
DATASET='rsraki/f013-gamma-e12-qualification-20261020-v1/1'
GENERIC='rsraki/n392-aucf0-816d6df4187b4878a3ccd7234ffa14dd/1'


def prepare(output,slug):
    output=Path(output)
    F.require(not output.exists() and not output.is_symlink(),'Fresh output directory')
    F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}',slug),'Own slug')
    manifest=F.read(STAGE/'package-manifest.json')
    F.require(manifest['policy_SHA']==POLICY_SHA,'New qualification policy only')
    F.require(F.pin(STAGE/manifest['archive']['path'])=={k:manifest['archive'][k] for k in ('bytes','sha256')},'Private model package bytes differ')
    runtime=(ROOT/'scripts/f013_cpu_qualifier.py').read_text(encoding='utf8')
    runtime_sha=hashlib.sha256(runtime.encode()).hexdigest()
    manifest_sha=F.pin(STAGE/'package-manifest.json')['sha256']
    code=('import os,hashlib,importlib.util\nfrom pathlib import Path\n'
          "os.environ['CUDA_VISIBLE_DEVICES']=''\n"
          f'_f013_source={runtime!r}\n'
          f'assert hashlib.sha256(_f013_source.encode()).hexdigest()=={runtime_sha!r}\n'
          "_f013_program=Path('/kaggle/working/f013_cpu_qualifier.py')\n"
          "assert not _f013_program.exists()\n_f013_program.write_text(_f013_source,encoding='utf8')\n"
          "_f013_spec=importlib.util.spec_from_file_location('f013_standalone_qualifier',_f013_program)\n"
          '_f013_module=importlib.util.module_from_spec(_f013_spec)\n_f013_spec.loader.exec_module(_f013_module)\n'
          "def _f013_mount(slug,filename):\n"
          "    root=Path('/kaggle/input')\n"
          "    paths=[root/slug,*root.glob('datasets/rsraki/'+slug),*root.glob('datasets/rsraki/'+slug+'/*')]\n"
          "    found=list(dict.fromkeys(p for p in paths if (p/filename).is_file()))\n"
          "    assert len(found)==1\n    return found[0]\n"
          f"_f013_result=_f013_module.run(_f013_mount({DATASET.split('/')[1]!r},'package-manifest.json'),"
          f"_f013_mount({GENERIC.split('/')[1]!r},'transport-manifest.json'),{manifest_sha!r},Path('/kaggle/working/F013-CPU-qualification'))\n")
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':'# F013 standalone CPU integrity qualification\n\nOwn fixed E12 gamma checkpoints only. Original F009 engineering HOLD stays unchanged. Two private owned Inputs, offline CPU, no competition data/Reports/Gold/fit. Strict source14 reconstruction and finite tensors plus synthetic encoder/head checks; not full CUDA/MRI/Public qualification. Never formally submit this notebook.\n'},F.code(code)],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    text=json.dumps(book,indent=1)+'\n'
    payload={'slug':slug,'new_title':slug.split('/')[1],'text':text,'language':'python','kernel_type':'notebook',
             'dataset_data_sources':[DATASET,GENERIC],'kernel_data_sources':[],'competition_data_sources':[],
             'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
             'docker_image':'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f',
             'docker_image_pinning_type':'original','kernel_execution_type':'SAVE_AND_RUN_ALL'}
    receipt={'at_UTC':datetime.now(timezone.utc).isoformat(),'policy_SHA':POLICY_SHA,'package_manifest_SHA':manifest_sha,
             'source_SHA':hashlib.sha256(text.encode()).hexdigest(),'runtime_SHA':runtime_sha,
             'runtime':F.pin(ROOT/'scripts/f013_cpu_qualifier.py'),'only_private_CPU_no_competition_Input':True,
             'no_local_Torch_GPU_MRI_or_remote_writes':True,'private_dataset_ref':DATASET,'generic_ref':GENERIC,
             'original_HOLD_unchanged':True,'formal_Submit_allowed':False}
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('source.ipynb',book),('save-payload.json',payload),('manifest.json',receipt)]:
        (output/name).write_text(text if name=='source.ipynb' else json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
    (output/'runtime.py').write_text(runtime,encoding='utf8',newline='\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slug',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.slug),indent=2))
