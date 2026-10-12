"""Prepare one-factor paired configs and code-only handoff, never run training."""

import argparse
import copy
import json
import zipfile
from pathlib import Path

from rsna_knee.contracts import dump_json, sha256
from rsna_knee.native_training import validate_native_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {0: "f005-dinov2-native-frozen-control.json", 2: "f005-dinov2-native-last2.json"}


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve old handoff; use a new output directory")
    base = json.loads((ROOT / "configs/experiments/f002-dinov2-attention.json").read_text(encoding="utf-8"))
    configs = {}
    for blocks, name in CONFIGS.items():
        config = copy.deepcopy(base)
        config["reason"] = (
            "New native392 paired online frozen control. Match microbatch1/effective16, observed-cell accumulation, fixed f002 head/seed/folds/BCE selection to last2."
            if blocks == 0
            else "Change only generic DINO blocks10/11 trainability relative to native frozen control; encoder LR prespecified1e-5, head LR old3e-4; no weak/Public sweep."
        )
        config["online"] = {
            "schema_version": "native392_last2_training_v1",
            "trainable_blocks": blocks,
            "microbatch_studies": 1,
            "effective_batch_studies": 16,
            "window_chunk": 2,
            "checkpoint_last2": True,
            "encoder_learning_rate": 1e-5,
            "accumulation_denominator": "total_observed_labels_in_effective_batch",
            "folds": [0, 1],
            "amp_grad_scaler": True,
        }
        validate_native_config(config)
        path = ROOT / "configs/experiments" / name
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != config:
                raise ValueError("Previously prepared config changed")
        else:
            dump_json(path, config)
        configs[name] = config
    output.mkdir(parents=True)
    files = sorted((ROOT / "src/rsna_knee").glob("*.py")) + [
        ROOT / "scripts" / name
        for name in ("train_native_dinov2.py", "profile_dinov2_last2.py", "bounded_feature_reader.py")
    ]
    files += [ROOT / "configs/experiments" / name for name in CONFIGS.values()]
    files.append(ROOT / "configs/experiments/f002-dinov2-attention.json")
    with zipfile.ZipFile(output / "f005-native-code.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    common = "--manifest data/manifests/v1-vmohitrao-research --index data/native392/full-index.json --resource-gate data/native392/resource-gate.json"
    assets = "--repo GENERIC_ASSETS/encoder/dinov2-source --weights GENERIC_ASSETS/encoder/dinov2_vits14_pretrain.pth --provenance GENERIC_ASSETS/encoder/encoder-provenance.json"
    commands = [
        f".venv/Scripts/python.exe scripts/train_native_dinov2.py preflight {common} --config configs/experiments/{name} --fold {fold}"
        for name in CONFIGS.values()
        for fold in (0, 1)
    ]
    train_commands = [
        f".venv/Scripts/python.exe scripts/train_native_dinov2.py train {common} {assets} --config configs/experiments/{name} --fold {fold} --output artifacts/runs/f005-{'native-frozen-control' if blocks == 0 else 'native-last2'}-fold{fold}-NEW"
        for blocks, name in CONFIGS.items()
        for fold in (0, 1)
    ]
    gate = {
        "schema_version": "native392_training_resource_gate_v1",
        "native392_pixel16_complete": False,
        "native392_byte_QC_passed": False,
        "compression_measured": False,
        "direct_archive_read_speed_measured": False,
        "artificial_CUDA_gradient_and_roundtrip_passed": False,
        "note": "Template only; cannot enable training. Root binds actual passed evidence before writing a live gate.",
    }
    dump_json(output / "resource-gate-template.json", gate)
    metadata = {
        "status": "prepared_real_fitting_not_started",
        "one_model_factor": "trainable DINO blocks0 vs last2 only",
        "configs": {name: sha256(ROOT / "configs/experiments" / name) for name in configs},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in files},
        "code_zip_sha256": sha256(output / "f005-native-code.zip"),
        "preflight_commands_after_full_cache_gate": commands,
        "training_commands_not_executed": train_commands,
        "effective_batch_studies": 16,
        "microbatch_studies": 1,
        "head_lr": 0.0003,
        "encoder_lr": 1e-5,
        "epochs": 12,
        "folds": [0, 1],
        "seed": 20261002,
        "selection": "minimum weak validation masked BCE; ties earliest epoch",
        "cache_scope": "Full weak UID/group/fold coverage required; current16 pilot cannot start these fits",
        "checkpoint": "Strict generic provenance/prefix + suffix/head state; no resume; Report-free Kaggle native inference",
        "real_fit_started": False,
        "full_cache_downloaded": False,
        "automatic_upload_submit": False,
    }
    dump_json(output / "handoff.json", metadata)
    print(
        json.dumps(
            {
                "handoff": str(output.relative_to(ROOT)),
                "configs": list(configs),
                "code_zip_sha256": metadata["code_zip_sha256"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
