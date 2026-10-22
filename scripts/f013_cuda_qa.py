"""Artificial-only native392 CUDA qualification; no MRI, labels, training or APIs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import f013_cpu_qualifier as C

GPU_POLICY_SHA='6682d35c4c3e55a51e5e6c3f8b43883be5438f83874d173095556d1ffc73dbed'


def child(root,fold,output):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['XFORMERS_DISABLED']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    import importlib

    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root,output=Path(root),Path(output)
    policy=json.loads((root/'qualification-policy.json').read_bytes())
    C.require(C.sha(root/'qualification-policy.json')==C.POLICY_SHA,'Original qualification policy')
    ref=policy['fixed_checkpoints'][str(fold)]
    checkpoint_path=root/('fold%d/epoch_012.pt'%fold)
    C.require(C.sha(checkpoint_path)==ref['sha256'],'Exact fixed E12 model')
    C.require(torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4','One assigned actualT4 per isolated child')
    sys.path.insert(0,str(root))
    trainer=importlib.import_module('source14.native_training_fast16')
    model_module=importlib.import_module('source14.finetune_model')
    checkpoint=trainer.load_native_checkpoint(checkpoint_path,allow_artificial=False)
    C.require(checkpoint['epoch']==12 and checkpoint['fold']==fold and checkpoint['config']==ref['config']
              and checkpoint['encoder_provenance']==ref['encoder_provenance'],'Fixed metadata identity')
    model=trainer.new_native_model(root/'generic/repo',root/'generic/dinov2_vits14_pretrain.pth',ref['encoder_provenance'],ref['config'],device='cpu')
    C.require(type(model) is model_module.NativeTripletMIL,'Exact isolated native model class')
    prefix=trainer.prefix_digest(model)
    C.require(prefix==checkpoint['frozen_prefix_sha256'],'Generic frozen prefix')
    trainer.apply_native_checkpoint(model,checkpoint,ref['encoder_provenance'])
    model.eval().requires_grad_(False)
    images=torch.zeros((1,3,48,3,392,392),dtype=torch.uint8)
    mask=torch.zeros((1,3,48),dtype=torch.bool)
    for series,window in ((0,0),(0,20),(1,13),(2,25)):
        mask[0,series,window]=True
    valid=(torch.arange(4*3*392*392).reshape(4,3,392,392)%251).to(torch.uint8)
    images[mask]=valid
    positions=torch.linspace(0,1,48)[None,None,:].expand(1,3,48).clone()
    planes=torch.tensor([[0,1,2]],dtype=torch.long)
    flags=torch.tensor([[[1.,0.],[0.,1.],[1.,1.]]])
    with torch.no_grad():
        im=valid.float()/255
        mean,std=[im.new_tensor(v)[None,:,None,None] for v in ([.485,.456,.406],[.229,.224,.225])]
        features=im.new_zeros((1,3,48,384))
        features[mask]=model.encode_normalized((im-mean)/std)
        reference=model.head(features,mask,positions,planes,flags).sigmoid()
        C.require(reference.shape==(1,12) and torch.isfinite(reference).all(),'Artificial CPU reference12')
    device='cuda:0'
    model.to(device)
    inputs={name:value.to(device) for name,value in {'images':images,'mask':mask,'positions':positions,'planes':planes,'flags':flags}.items()}
    rng= torch.get_rng_state().clone()
    gamma=model._gamma_generator.get_state().clone()
    cuda_rng=torch.cuda.get_rng_state(0).clone()
    with torch.inference_mode():
        logits=model(**inputs).float()
        C.require(logits.shape==(1,12) and torch.isfinite(logits).all(),'Full original nativeCUDA12 finite')
        poisoned=dict(inputs)
        poisoned['images']=inputs['images'].clone()
        poisoned['images'][~inputs['mask']]=255
        C.require(torch.equal(logits,model(**poisoned).float()),'Fullnative masked pixel padding invariant')
        empty=dict(inputs,mask=torch.zeros_like(inputs['mask']))
        refused=False
        try:
            model(**empty)
        except ValueError:
            refused=True
        C.require(refused,'Allmaskmissing must fail beforeencoder')
        probabilities=logits.sigmoid().cpu()
        error=float(torch.max(torch.abs(probabilities-reference)))
        C.require(error<=.005,'Fixed CPUmanualFP32 versusfullCUDA probability tolerance .005')
    C.require(model._gamma_draws==0 and torch.equal(gamma,model._gamma_generator.get_state())
              and torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(0)),'Eval gamma/global random streams unchanged')
    C.require(trainer.prefix_digest(model)==prefix,'Frozen prefix unchanged afterCUDA')
    for part,values in [('encoder_suffix',{k:v for k,v in model.encoder.state_dict().items() if k in trainer.suffix_keys(model)}),('head',model.head.state_dict())]:
        C.require(set(values)==set(checkpoint[part]) and all(torch.equal(v.cpu(),checkpoint[part][k]) and torch.isfinite(v).all() for k,v in values.items()),'ActualCUDA selected tensors unchanged and finite')
    result={'complete':True,'fold':fold,'epoch':12,'checkpoint_SHA':ref['sha256'],'frozen_prefix_SHA':prefix,
            'class':'source14.finetune_model.NativeTripletMIL','GPU':torch.cuda.get_device_name(0),'Torch':torch.__version__,
            'full_native392_forward_finite12':True,'valid_triplets':4,'all3planes':True,'padding_pixel_0_vs255_logits_bitexact':True,
            'allmissing_rejected_beforeencoder':True,'evalgamma_and_CPU_CUDA_RNG_unchanged':True,'suffix_head_prefix_unchanged':True,
            'CPUmanual_FP32_reference_probability_maxabs':error,'fixed_probability_tolerance':.005,
            'MRI_Report_Gold_labels_training_reads':0,'original_F009_HOLD_unchanged':True,'realMRI_or_finalfusion_qualified':False}
    C.require(not output.exists(),'Fresh CUDA output')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('F013_ARTIFICIAL_FULL_NATIVE_CUDA_PASS',fold,error,flush=True)


def run(package,generic,manifest_SHA,output):
    output=Path(output)
    C.require(Path('/kaggle/working').is_dir() and not output.exists(),'Fresh Kaggle artificialCUDA output')
    output.mkdir(exist_ok=False)
    cpu=C.run(package,generic,manifest_SHA,output/'CPU-reference')
    C.require(cpu['complete'],'Current exactprivate Inputs CPU gate')
    receipts=[]
    for fold in (0,1):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(fold),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (output/('fold%d.stdout.txt'%fold)).open('w') as out,(output/('fold%d.stderr.txt'%fold)).open('w') as err:
            result=subprocess.run([sys.executable,'-u',__file__,'--fold',str(fold),'--root','/tmp/f013-fixed-E12-CPU-qualification','--output',str(output/('fold%d.json'%fold))],env=env,stdout=out,stderr=err,timeout=600,check=False)
        C.require(result.returncode==0,'Artificial CUDA child failed fold%d'%fold)
        receipt=json.loads((output/('fold%d.json'%fold)).read_bytes())
        C.require(receipt['complete'] and receipt['fold']==fold,'Both CUDA folds required')
        receipts.append(receipt)
    complete={'complete':True,'policy_SHA':GPU_POLICY_SHA,'qualification_policy_SHA':C.POLICY_SHA,
              'package_manifest_SHA':manifest_SHA,'folds':receipts,'artificial_only':True,'MRI_Report_Gold_labels_training_Save_Submit':0,
              'old_F009_HOLD_and_consumers_unchanged':True,'realMRI_fusion_Public_improvement_qualified':False}
    (output/'completion.json').write_text(json.dumps(complete,indent=2)+'\n',encoding='utf8')
    print('F013_BOTH_FIXED_E12_ARTIFICIAL_CUDA_COMPLETE',flush=True)
    return complete


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=(0,1),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    child(args.root,args.fold,args.output)
