"""Offline frozen-encoder experiment contracts; independent of the legacy image cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .contracts import sha256


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_feature_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_feature_config(config)
    return config


def validate_feature_config(config):
    if config.get("schema_version") != "frozen_features_v1":
        raise ValueError("Unsupported feature config schema")
    p, e, h, t = (config[key] for key in ("preprocess", "encoder", "head", "train"))
    counts = [
        config["n_folds"],
        p["image_size"],
        p["max_series"],
        p["slices_per_series"],
        e["batch_size"],
        h["hidden_dim"],
        t["epochs"],
        t["batch_size"],
    ]
    if any(type(value) is not int or value < 1 for value in counts) or config["n_folds"] < 2:
        raise ValueError("Invalid positive integer count")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Invalid seed")
    if p["image_size"] % 14 or p["image_size"] < 28 or p["max_series"] > 6:
        raise ValueError("DINOv2 input must be a multiple of 14; max_series <= 6")
    if p["version"] != "physical_letterbox_triplet_v1" or p["percentiles"] != [1, 99]:
        raise ValueError("Unsupported preprocessing")
    if e["model_name"] != "dinov2_vits14" or e["feature_dim"] != 384:
        raise ValueError("Only generic DINOv2 ViT-S/14 CLS features are supported")
    if e["precision"] not in ("fp32", "fp16") or h["pooling"] not in ("mean", "attention"):
        raise ValueError("Invalid encoder precision or head pooling")
    if t["selection"] not in ("auc", "bce") or type(t["num_workers"]) is not int or t["num_workers"] < 0:
        raise ValueError("Invalid training setting")
    for value in (h["dropout"], t["learning_rate"], t["weight_decay"]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid finite hyperparameter")
    if h["dropout"] >= 1 or t["learning_rate"] == 0:
        raise ValueError("Invalid dropout or learning rate")
    if not isinstance(config.get("reason"), str) or not config["reason"].strip():
        raise ValueError("Record the hypothesis/adoption reason before running")


def implementation_hashes():
    root = Path(__file__).parent
    names = ("feature_contracts.py", "feature_imaging.py", "imaging.py", "contracts.py")
    return {
        name: hashlib.sha256((root / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        for name in names
    }


def feature_contract(config, encoder_provenance):
    return {
        "schema_version": "frozen_features_v1",
        "preprocess": config["preprocess"],
        "encoder": config["encoder"],
        "encoder_provenance": encoder_provenance,
        "implementation_sha256": implementation_hashes(),
    }


def validate_encoder_provenance(value):
    for name in ("source_url", "source_revision", "weights_url", "license", "reviewed_at", "exposure_audit"):
        if not isinstance(value.get(name), str) or not value[name].strip() or value[name].startswith("TODO"):
            raise ValueError(f"Complete encoder provenance: {name}")
    for name in ("weights_sha256", "source_tree_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get(name, "")) or value[name] == "0" * 64:
            raise ValueError(f"Pin full encoder digest: {name}")
    if value.get("pretraining") != "generic_dinov2_lvd142m" or value.get("competition_finetuned") is not False:
        raise ValueError("Competition-finetuned encoder is not an independent frozen-feature baseline")
    if value.get("gold_used_for_tuning") is not False:
        raise ValueError("Gold-tuned initialization is forbidden for this experiment")


def source_tree_hash(root):
    root = Path(root)
    if not (root / "hubconf.py").is_file() or not (root / "dinov2").is_dir():
        raise ValueError("Expected a local official DINOv2 source snapshot")
    paths = [root / "hubconf.py", *sorted((root / "dinov2").rglob("*.py"))]
    return json_hash({path.relative_to(root).as_posix(): sha256(path) for path in paths})


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Use a new run/cache directory; existing paths are never overwritten")
    path.mkdir(parents=True)
    return path


def validate_fold_audit(manifest_dir, audit_path, manifest):
    """Bind the independent original-report/group audit to the unchanged fold CSVs."""
    root = Path(manifest_dir)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8-sig"))
    if audit.get("integrity_valid") is not True or audit.get("ready_for_training") is not True:
        raise ValueError("A successful fold audit is required before feature-head fitting")
    live = {name: sha256(root / name) for name in ("manifest.json", "weak.csv", "gold.csv")}
    if audit.get("manifest_sha256") != live or audit.get("inputs_sha256") != manifest["inputs_sha256"]:
        raise ValueError("Stale fold audit; manifest/source hashes changed")
    if audit.get("gold_used_for_tuning") is not False:
        raise ValueError("Fold audit reports gold tuning")
    return audit


def selection_structure_hash(rows):
    return json_hash(sorted((row["StudyInstanceUID"], row["group_id"], str(row["fold"])) for row in rows))


def head_implementation_hash():
    return hashlib.sha256(
        Path(__file__).with_name("feature_model.py").read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    ).hexdigest()
