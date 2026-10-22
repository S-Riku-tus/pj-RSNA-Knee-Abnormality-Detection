"""Create isolated F010 source copies once; never executes a model or data path.

Every replacement is explicit and counted. Original F006/F009 files and their
runtime qualifications stay intact. The receipt contains their original bytes.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research/20261019-f010-last4-preparation-v2"


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"Source anchor count changed: {before!r}")
    return text.replace(before, after)


def replace_function(text, name, replacement):
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


RESOURCE_GATE = '''def validate_resource_gate(path, config=None):
    """Only actual new-source F010144/processed QA can qualify last4 fitting."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    workspace = Path(__file__).resolve().parents[2]
    cli = workspace / "scripts/train_f010_last4_fast16.py"
    if (value.get("schema_version") != RESOURCE_GATE_SCHEMA
        or value.get("source_sha256") != native_source_hashes()
        or value.get("CLI_sha256") != sha256(cli)
        or value.get("engineering_lineage") != ENGINEERING_LINEAGE
        or value.get("approved_for_real_fit") is not True):
        raise ValueError("New F010 source/CLI/lineage resource gate required")
    if config is not None and value.get("config_semantic_sha256", {}).get(
        str(config["online"]["trainable_blocks"])) != json_hash(config):
        raise ValueError("F010 resource gate belongs to another exact config")
    evidence = {}
    for name in ("pixel16", "processed_QA", "artificial144_QA"):
        ref = value.get("evidence", {}).get(name, {})
        if not isinstance(ref.get("path"), str) or not ref["path"]:
            raise ValueError("Missing F010 resource receipt")
        receipt = (workspace / ref["path"]).resolve()
        if (not receipt.is_relative_to(workspace) or receipt.suffix != ".json"
            or sha256(receipt) != ref.get("sha256")):
            raise ValueError("F010 resource receipt path/SHA differs")
        evidence[name] = json.loads(receipt.read_text(encoding="utf-8"))
    pixels = evidence["pixel16"]
    if (pixels.get("all_passed") is not True or pixels.get("studies") != 16
        or not pixels.get("checks") or not all(flag is True for flag in pixels["checks"].values())):
        raise ValueError("Original full pixel16 byteQC must remain qualified")
    for name, real in (("processed_QA", True), ("artificial144_QA", False)):
        qa = evidence[name]
        if (qa.get("status") != "f010_last4_label_free_QA_fast16_complete"
            or qa.get("real_processed_MRI_pixels_used") is not real
            or qa.get("source_sha256") != native_source_hashes()
            or qa.get("CLI_sha256") != value["CLI_sha256"]
            or qa.get("engineering_lineage") != ENGINEERING_LINEAGE
            or qa.get("config_semantic_sha256") != value.get("config_semantic_sha256")
            or qa.get("real_teacher_labels_read") is not False or qa.get("real_fit") is not False
            or qa.get("optimizer_steps") != 0 or qa.get("studies_simultaneously_on_GPU") != 1
            or qa.get("raw_DICOM_decoded_or_downloaded") is not False
            or len(qa.get("arms", [])) != 2
            or {row.get("arm") for row in qa["arms"]} != {"freeze", "last4"}
            or qa.get("checks_passed") != len(qa.get("checks", [])) or qa.get("checks_passed", 0) < 6
            or not all(row.get("passed") is True for row in qa.get("checks", []))):
            raise ValueError("Actual F010 label-free paired resource QA failed")
        for arm in qa["arms"]:
            parameters = arm.get("parameters", {})
            expected_blocks = 0 if arm["arm"] == "freeze" else 4
            suffix_names = parameters.get("trainable_encoder_names", [])
            expected_suffixes = {
                f"blocks.{block}.{key}"
                for block in (8, 9, 10, 11)
                for key in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                            "attn.proj.weight", "attn.proj.bias", "ls1.gamma", "norm2.weight",
                            "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight",
                            "mlp.fc2.bias", "ls2.gamma")
            }
            if (parameters.get("trainable_blocks") != expected_blocks
                or parameters.get("window_chunk") != 16 or parameters.get("checkpoint_last2") is not True
                or parameters.get("encoder_eval") is not True or parameters.get("final_norm_frozen") is not True
                or parameters.get("trainable_head_parameters") != 83_352
                or parameters.get("trainable_encoder_parameters") != (0 if expected_blocks == 0 else 7_100_928)
                or parameters.get("frozen_encoder_parameters") != (22_056_576 if expected_blocks == 0 else 14_955_648)
                or arm.get("optimizer_steps") != 0 or arm.get("effective_batch_studies") != 16
                or type(arm.get("valid_triplets")) is not int or not 1 <= arm["valid_triplets"] <= 144
                or not all(type(arm.get(key)) is int and 0 < arm[key] <= 21_000_000_000
                           for key in ("gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes"))
                or (expected_blocks == 0 and suffix_names)
                or (expected_blocks == 4 and (len(suffix_names) != 56 or set(suffix_names) != expected_suffixes))):
                raise ValueError("F010 actual suffix scope/resource bound differs")
            if not real and arm["valid_triplets"] != 144:
                raise ValueError("F010 artificial144 must cover the full padded capacity")
        if real and (qa.get("pilot", {}).get("archive_sha256") != pixels.get("archive_sha256")
                     or qa.get("pilot", {}).get("read_decompress_hash_QC_seconds", -1) < 0):
            raise ValueError("F010 processed QA must bind the original byte-reviewed archive")
    return value
'''


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = {
        "model": ROOT / "src/rsna_knee/finetune_model.py",
        "trainer": ROOT / "src/rsna_knee/native_training_fast16.py",
        "CLI": ROOT / "scripts/train_native_dinov2_fast16.py",
        "config": ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json",
    }
    originals = {key: path.read_bytes() for key, path in paths.items()}
    old = {key: payload.decode("utf-8").replace("\r\n", "\n") for key, payload in originals.items()}
    model = old["model"].replace("NativeTripletMIL", "F010Last4MIL")
    model = replace(model, "trainable_blocks not in (0, 2)", "trainable_blocks not in (0, 4)")
    model = replace(model, '("blocks.10.", "blocks.11.")', '("blocks.8.", "blocks.9.", "blocks.10.", "blocks.11.")')
    model = model.replace("last2", "last4").replace("checkpoint_last4", "checkpoint_last2")
    model = model.replace("last two", "last four").replace("blocks10/11", "blocks8/9/10/11")
    model = replace_function(model, "validate_study_gamma", '''def validate_study_gamma(value):
    """F010 changes only the unfrozen suffix; augmentation is forbidden."""
    if value is not None:
        raise ValueError("F010 preserves the augmentation-free F006 control")
    return None
''')
    trainer = old["trainer"].replace("from .finetune_model import NativeTripletMIL", "from .f010_last4_model import F010Last4MIL")
    trainer = trainer.replace("NativeTripletMIL", "F010Last4MIL")
    trainer = trainer.replace('"native_training_fast16.py"', '"f010_last4_training.py"')
    trainer = trainer.replace('"finetune_model.py"', '"f010_last4_model.py"')
    trainer = trainer.replace("last2", "last4").replace("checkpoint_last4", "checkpoint_last2")
    trainer = replace(trainer, '"trainable_blocks": (0, 2)', '"trainable_blocks": (0, 4)')
    trainer = replace(trainer, "if model.trainable_blocks == 2", "if model.trainable_blocks == 4")
    trainer = replace(trainer, '("blocks.10.", "blocks.11.")', '("blocks.8.", "blocks.9.", "blocks.10.", "blocks.11.")')
    trainer = replace(trainer, 'CHECKPOINT_SCHEMA = "native392_generic_dino_suffix_head_fast16_v1"', 'CHECKPOINT_SCHEMA = "f010_last4_generic_dino_suffix_head_fast16_v1"')
    trainer = replace(trainer, 'RESOURCE_GATE_SCHEMA = "native392_training_resource_gate_fast16_v1"', 'RESOURCE_GATE_SCHEMA = "f010_last4_training_resource_gate_fast16_v1"')
    trainer = replace(trainer, '"schema_version": "native392_last4_training_fast16_v1"', '"schema_version": "f010_native392_last4_training_fast16_v1"')
    trainer = replace(trainer, '"schema_version": "native392_training_run_fast16_v1"', '"schema_version": "f010_native392_training_run_fast16_v1"')
    trainer = trainer.replace('"native392_epoch_suffix_head_retention_v1"', '"f010_native392_epoch_suffix_head_retention_v1"')
    lineage = next(line for line in trainer.splitlines() if line.startswith("ENGINEERING_LINEAGE ="))
    trainer = replace(trainer, lineage, "ENGINEERING_LINEAGE = " + repr({"version": "f010_last4_fast16_v1", "base_fast16_training_sha256": hashlib.sha256(originals["trainer"]).hexdigest(), "change": "trainable_encoder_blocks_10_11_to_8_9_10_11_only"}))
    start = trainer.index("    if augmentation is not None and (")
    stop = trainer.index("    if settings != {", start)
    trainer = trainer[:start] + '    if augmentation is not None or validate_native_learning_rate_schedule(config["train"].get("lr_schedule"))["name"] != "late_cosine":\n        raise ValueError("F010 requires augmentation-free fixed F006 late cosine")\n' + trainer[stop:]
    trainer = replace_function(trainer, "validate_resource_gate", RESOURCE_GATE)
    trainer = replace(trainer, 'def apply_native_checkpoint(model, checkpoint, provenance):\n    validate_native_config(model.config)\n', 'def apply_native_checkpoint(model, checkpoint, provenance):\n    validate_native_config(model.config)\n    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or checkpoint.get("source_sha256") != native_source_hashes():\n        raise ValueError("F010 direct application requires its exact new checkpoint/source schema")\n')
    trainer = replace(trainer, '    ensure_cuda()\n    # Validate all metadata before creating a new run or reading the model checkpoint.', '    ensure_cuda()\n    if config.get("online", {}).get("trainable_blocks") != 4:\n        raise ValueError("F010 real fitting permits only the last4 candidate; frozen config is QA only")\n    # Validate all metadata before creating a new run or reading the model checkpoint.')
    cli = old["CLI"].replace("from rsna_knee.finetune_model import NativeTripletMIL", "from rsna_knee.f010_last4_model import F010Last4MIL")
    cli = cli.replace("from rsna_knee.native_training_fast16 import", "from rsna_knee.f010_last4_training import")
    cli = cli.replace("NativeTripletMIL", "F010Last4MIL")
    cli = cli.replace("last2", "last4").replace("checkpoint_last4", "checkpoint_last2")
    cli = cli.replace('"f005-dinov2-native-frozen-control-fast16.json"', '"f010-dinov2-native-frozen-QA-only.json"')
    cli = cli.replace('"f005-dinov2-native-last4-fast16.json"', '"f010-dinov2-native-last4-late-cosine.json"')
    cli = replace(cli, "for blocks in (0, 2):", "for blocks in (0, 4):")
    cli = replace(cli, "from profile_dinov2_last4 import synthetic_study", "from profile_dinov2_last2 import synthetic_study")
    cli = replace(cli, '"status": "native392_label_free_QA_fast16_complete"', '"status": "f010_last4_label_free_QA_fast16_complete"')
    config = json.loads(old["config"])
    config["reason"] = "F006 gammaなし固定E12との一要因比較。generic DINO/392px cache/教師/seed/fold/head/opt/LR/12epochを維持し、解凍encoderを末尾2blockから末尾4block(8..11)へ増やす。final normは凍結。Gold/Public/途中epochで選ばず、E12同士の固定10 weak比較後に別fold確認。"
    config["online"]["schema_version"] = "f010_native392_last4_training_fast16_v1"
    config["online"]["trainable_blocks"] = 4
    frozen = json.loads(json.dumps(config))
    frozen["online"]["trainable_blocks"] = 0
    frozen["reason"] = "F010 source-bound人工/processed資源QA専用。実学習には利用しない。"
    new = {
        ROOT / "src/rsna_knee/f010_last4_model.py": model,
        ROOT / "src/rsna_knee/f010_last4_training.py": trainer,
        ROOT / "scripts/train_f010_last4_fast16.py": cli,
        ROOT / "configs/experiments/f010-dinov2-native-last4-late-cosine.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        ROOT / "configs/experiments/f010-dinov2-native-frozen-QA-only.json": json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
    }
    for path, text in new.items():
        if path.suffix == ".py":
            ast.parse(text)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    for key, payload in originals.items():
        (OUT / f"base-{key}.snapshot").write_bytes(payload)
    receipt = {"schema_version": "F010_isolated_source_copy_preparation_v1", "actual_GPU_model_data_fit": False,
        "original": {key: {"path": str(paths[key].relative_to(ROOT)), "sha256": hashlib.sha256(value).hexdigest()} for key, value in originals.items()},
        "new": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in new},
        "original_bytes_unchanged": all(paths[key].read_bytes() == payload for key, payload in originals.items())}
    (OUT / "source-preparation.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
