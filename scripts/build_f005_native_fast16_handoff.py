"""Code-only fast16 handoff. No data acquisition, upload, execution or fitting."""

import argparse
import copy
import json
import zipfile
from pathlib import Path

from rsna_knee.contracts import dump_json, sha256
from rsna_knee.feature_contracts import json_hash
from rsna_knee.native_training_fast16 import (
    ENGINEERING_LINEAGE,
    RESOURCE_GATE_SCHEMA,
    native_source_hashes,
    validate_native_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {0: "f005-dinov2-native-frozen-control-fast16.json", 2: "f005-dinov2-native-last2-fast16.json"}


def prepare(output):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "artifacts").resolve()) or output.exists():
        raise ValueError("Fresh handoff inside project artifacts required")
    configs = {
        blocks: validate_native_config(json.loads((ROOT / "configs/experiments" / name).read_text(encoding="utf-8")))
        for blocks, name in CONFIGS.items()
    }
    left, right = (copy.deepcopy(configs[blocks]) for blocks in (0, 2))
    for config in (left, right):
        config.pop("reason")
        config["online"].pop("trainable_blocks")
    if left != right or any(configs[blocks]["online"]["trainable_blocks"] != blocks for blocks in configs):
        raise ValueError("Paired arms must have identical engineering/training conditions")
    output.mkdir(parents=True)
    files = sorted((ROOT / "src/rsna_knee").glob("*.py")) + [
        ROOT / "scripts" / name for name in (
            "train_native_dinov2_fast16.py", "profile_dinov2_last2.py", "bounded_feature_reader.py",
            "build_f005_native_fast16_handoff.py",
        )
    ] + [ROOT / "configs/experiments" / name for name in CONFIGS.values()]
    archive_path = output / "f005-native-fast16-code.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    gate_path = "data/native392/resource-gate-fast16-v3.json"
    common = f"--manifest data/manifests/v1-vmohitrao-research --index data/native392/full-index.json --resource-gate {gate_path}"
    assets = "--repo GENERIC_ASSETS/encoder/dinov2-source --weights GENERIC_ASSETS/encoder/dinov2_vits14_pretrain.pth --provenance GENERIC_ASSETS/encoder/encoder-provenance.json"
    cli = "scripts/train_native_dinov2_fast16.py"
    preflight = [
        f".venv/Scripts/python.exe {cli} preflight {common} --config configs/experiments/{name} --fold {fold}"
        for name in CONFIGS.values() for fold in (0, 1)
    ]
    training = [
        f".venv/Scripts/python.exe {cli} train {common} {assets} --config configs/experiments/{name} --fold {fold} --output artifacts/runs/f005-{'native-frozen-control' if blocks == 0 else 'native-last2'}-fast16-fold{fold}-NEW"
        for blocks, name in CONFIGS.items() for fold in (0, 1)
    ]
    gate = {
        "schema_version": RESOURCE_GATE_SCHEMA,
        "native392_pixel16_complete": False,
        "native392_byte_QC_passed": False,
        "compression_measured": False,
        "direct_archive_read_speed_measured": False,
        "artificial_CUDA_gradient_and_roundtrip_passed": False,
        "source_sha256": native_source_hashes(),
        "CLI_sha256": sha256(ROOT / cli),
        "config_semantic_sha256": {str(blocks): json_hash(config) for blocks, config in configs.items()},
        "evidence": {name: {"path": None, "sha256": None} for name in (
            "pixel16", "processed_QA", "artificial_CUDA", "fast16_full144_QA",
        )},
        "note": "Template only. Root binds actual fast16 processed115 QA, full144 artificial QA and retained shared-model19CUDA evidence. No fitting is enabled.",
    }
    dump_json(output / "resource-gate-template.json", gate)
    metadata = {
        "status": "prepared_fast16_real_fitting_not_started",
        "engineering_lineage": ENGINEERING_LINEAGE,
        "one_model_factor": "Both arms window16; only generic DINO blocks0 vs last2 trainability differs",
        "configs": {name: sha256(ROOT / "configs/experiments" / name) for name in CONFIGS.values()},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in files},
        "native_source_sha256": native_source_hashes(),
        "CLI_sha256": sha256(ROOT / cli),
        "code_zip_sha256": sha256(archive_path),
        "preflight_commands_after_full_cache_gate": preflight, "training_commands_not_executed": training,
        "window_chunk": 16, "microbatch_studies": 1, "effective_batch_studies": 16,
        "epochs": 12, "folds": [0, 1], "seed": 20261002, "head_lr": 0.0003, "encoder_lr": 1e-5,
        "selection": "minimum weak validation observed-cell BCE; earliest tie",
        "accumulation": "micro observed BCE sum divided by entire actual effective group's total observed labels",
        "AMP": "GradScaler overflow skipping and optimizer opportunity/actual-step accounting retained",
        "checkpoint": "fast16-specific schema/config/source; strict generic prefix + suffix/head; artificial checkpoints forbidden for real inference",
        "full_cache_requirement": "Exact4207 weak UID/group/fold index and validated fast16 resource gate before real fitting",
        "old_checkpoint_config_or_resource_gate_accepted": False,
        "real_fit_started": False, "automatic_upload_or_execution_or_submission": False,
        "processed115_root_QA_pending": True,
    }
    dump_json(output / "handoff.json", metadata)
    print(json.dumps({"handoff": str(output.relative_to(ROOT)), "code_zip_sha256": metadata["code_zip_sha256"], "real_fit": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
