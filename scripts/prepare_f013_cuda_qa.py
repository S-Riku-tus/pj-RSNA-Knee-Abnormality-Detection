"""Prepare inert two-Input artificial CUDA notebook, no uploads or remote calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import prepare_f013_cpu_qualifier as B
from f013_cuda_qa import GPU_POLICY_SHA

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'artifacts/research/20261020-F013-artificial-CUDA-prospective-v1/policy.json'


def prepare(output,slug):
    output=Path(output)
    B.F.require(not output.exists() and not output.is_symlink(),'Fresh output')
    B.F.require(re.fullmatch(r'rsraki/[a-z0-9][a-z0-9-]{4,49}',slug),'Own slug')
    B.F.require(B.F.pin(POLICY)['sha256']==GPU_POLICY_SHA,'Frozen artificialCUDA policy')
    policy=B.F.read(POLICY)
    for ref in policy['pins'].values():
        B.F.require(B.F.pin(ROOT/ref['path'])=={k:ref[k] for k in ('bytes','sha256')},'Prior CPU/source/policy proof differs')
    prior=B.F.read(ROOT/policy['pins']['CPU_evidence']['path'])
    B.F.require(prior['all_pass'] and all(prior['checks'].values()),'Prior exactbothfoldCPU proof')
    sources={name:(ROOT/'scripts'/name).read_text(encoding='utf8') for name in ('f013_cpu_qualifier.py','f013_cuda_qa.py')}
    manifest_SHA=B.F.pin(B.STAGE/'package-manifest.json')['sha256']
    loader='import os,sys,hashlib,importlib.util\nfrom pathlib import Path\n'
    for name,source in sources.items():
        loader+=(f'_f013_text={source!r}\n'
                 f'assert hashlib.sha256(_f013_text.encode()).hexdigest()=={hashlib.sha256(source.encode()).hexdigest()!r}\n'
                 f'_f013_path=Path("/kaggle/working")/{name!r}\n'
                 'assert not _f013_path.exists()\n_f013_path.write_text(_f013_text,encoding="utf8")\n')
    loader+='sys.path.insert(0,"/kaggle/working")\nimport f013_cuda_qa as _f013_qa\n'
    loader+=("def _f013_mount(slug,filename):\n"
             "    root=Path('/kaggle/input')\n"
             "    paths=[root/slug,*root.glob('datasets/rsraki/'+slug),*root.glob('datasets/rsraki/'+slug+'/*')]\n"
             "    found=list(dict.fromkeys(p for p in paths if (p/filename).is_file()))\n"
             "    assert len(found)==1\n    return found[0]\n"
             f"_f013_result=_f013_qa.run(_f013_mount({B.DATASET.split('/')[1]!r},'package-manifest.json'),_f013_mount({B.GENERIC.split('/')[1]!r},'transport-manifest.json'),{manifest_SHA!r},Path('/kaggle/working/F013-artificial-CUDA'))\n")
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':'# F013 fixed E12 artificial CUDA qualification\n\nOnly two own private Inputs. CPU strict roundtrip followed by full native392 artificial CUDA forwards for both fixed folds. No competition data, MRI, Report, Gold, optimizer, labels, or training. Prior F009 HOLD unchanged. This is not real MRI/ensemble/Public qualification; never submit this notebook.\n'},B.F.code(loader)],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    text=json.dumps(book,indent=1)+'\n'
    payload=B.F.read(ROOT/'artifacts/kaggle/f013-fixed-e12-cpu-qualification-20261020-v1/save-payload.json')
    payload.update(slug=slug,new_title=slug.split('/')[1],text=text,enable_gpu=True,machine_shape='NvidiaTeslaT4',session_timeout_seconds=1800)
    manifest={'at_UTC':datetime.now(timezone.utc).isoformat(),'policy_SHA':GPU_POLICY_SHA,'source_SHA':hashlib.sha256(text.encode()).hexdigest(),
              'sources':{name:{'bytes':len(body.encode()),'sha256':hashlib.sha256(body.encode()).hexdigest()} for name,body in sources.items()},
              'private_two_Inputs_only':payload['dataset_data_sources'],'session_timeout_seconds':1800,'actual_GPU_or_MRI_run':False,'Save_Submit_upload':0}
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('source.ipynb',book),('save-payload.json',payload),('manifest.json',manifest)]:
        (output/name).write_text(text if name=='source.ipynb' else json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slug',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.slug),indent=2))
