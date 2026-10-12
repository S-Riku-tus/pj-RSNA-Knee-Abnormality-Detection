"""Prepared native392 training/inference path; no fitting is started by importing it.

Both arms use microbatch1/effective16. Each accumulation group divides every
microbatch BCE sum by the group's total observed-label count. The generic frozen
prefix is reconstructed from audited assets; checkpoints save only suffix/head.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoints import OptimizerStepCounter
from .contracts import (
    ID,
    SERIES_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    probability,
    read_csv,
    sha256,
    uid,
    validate_submission,
    write_csv,
)
from .feature_contracts import (
    head_implementation_hash,
    json_hash,
    new_directory,
    selection_structure_hash,
    validate_feature_config,
    validate_fold_audit,
)
from .feature_imaging import load_frozen_encoder
from .finetune_model import NativeTripletMIL
from .metrics import evaluate_rows
from .runtime import check_split, ensure_cuda
from .triplet_cache import extract_pixel_study, load_archive_study, pixel_contract

CHECKPOINT_SCHEMA = "native392_generic_dino_suffix_head_v1"
INDEX_SCHEMA = "native392_pixel_archive_index_v1"


def validate_native_config(config):
    validate_feature_config(config)
    expected = {
        "schema_version": "native392_last2_training_v1",
        "trainable_blocks": (0, 2),
        "microbatch_studies": 1,
        "effective_batch_studies": 16,
        "window_chunk": 2,
        "checkpoint_last2": True,
        "encoder_learning_rate": 1e-5,
        "accumulation_denominator": "total_observed_labels_in_effective_batch",
        "folds": [0, 1],
        "amp_grad_scaler": True,
    }
    online = config.get("online", {})
    if set(online) != set(expected) or any(
        online[key] != value for key, value in expected.items() if key != "trainable_blocks"
    ):
        raise ValueError("Keep the prespecified native training engineering/config contract")
    if online["trainable_blocks"] not in expected["trainable_blocks"]:
        raise ValueError("Only frozen-control or last2 arms supported")
    if config["seed"] != 20261002 or config["n_folds"] != 5:
        raise ValueError("Use the reviewed fixed seed/group folds")
    pixel_contract(config["preprocess"])
    if config["encoder"] != {"model_name": "dinov2_vits14", "feature_dim": 384, "batch_size": 16, "precision": "fp16"}:
        raise ValueError("Preserve the generic native392 encoder and fp16 CLS contract")
    if config["head"] != {"hidden_dim": 192, "dropout": 0.2, "pooling": "attention"}:
        raise ValueError("Preserve the original f002 head")
    if config["train"] != {
        "epochs": 12,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "num_workers": 0,
        "selection": "bce",
    }:
        raise ValueError("Preserve twelve epochs, old head LR and weak-BCE selection")
    return config


def native_source_hashes():
    here = Path(__file__).parent
    return {
        name: hashlib.sha256((here / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        for name in (
            "native_training.py",
            "finetune_model.py",
            "feature_model.py",
            "triplet_cache.py",
            "feature_imaging.py",
            "feature_contracts.py",
            "imaging.py",
            "contracts.py",
        )
    }


def label_tensors(row, device="cpu"):
    values = [probability(row.get(target, ""), missing=True) for target in TARGETS]
    observed = torch.tensor([[value is not None for value in values]], device=device, dtype=torch.bool)
    targets = torch.tensor(
        [[value if value is not None else 0 for value in values]], device=device, dtype=torch.float32
    )
    return targets, observed


def observed_bce_sum(logits, targets, observed):
    if logits.shape != targets.shape or targets.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("Malformed observed-label mask")
    if not observed.any():
        raise ValueError("No observed targets; no optimizer update is permitted")
    # Index before BCE: unobserved placeholders (including NaNs) do not contribute.
    result = F.binary_cross_entropy_with_logits(logits[observed], targets[observed], reduction="sum")
    if not torch.isfinite(result):
        raise ValueError("Nonfinite observed training loss")
    return result


def backward_observed_micro(logits, targets, observed, effective_denominator, *, scaler=None):
    if (
        type(effective_denominator) is not int
        or effective_denominator < int(observed.sum())
        or effective_denominator < 1
    ):
        raise ValueError("Effective denominator must include all observed cells in this microbatch")
    numerator = observed_bce_sum(logits, targets, observed)
    objective = numerator / effective_denominator
    (scaler.scale(objective) if scaler is not None else objective).backward()
    return float(numerator.detach())


def effective_observation_count(rows):
    count = sum(int(label_tensors(row)[1].sum()) for row in rows)
    if count <= 0:
        raise ValueError("Effective batch has zero observed labels")
    return count


def suffix_keys(model):
    return {
        name
        for name in model.encoder.state_dict()
        if model.trainable_blocks == 2 and name.startswith(("blocks.10.", "blocks.11."))
    }


def prefix_digest(model):
    digest = hashlib.sha256()
    omitted = suffix_keys(model)
    for name, tensor in sorted(model.encoder.state_dict().items()):
        if name not in omitted:
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_native_checkpoint(model, provenance, *, fold, epoch, weak_bce, prefix_sha256, audit, artificial=False):
    validate_native_config(model.config)
    if fold not in (0, 1) or type(epoch) is not int or epoch < 0 or not math.isfinite(weak_bce):
        raise ValueError("Invalid fixed fold/epoch/weak-BCE metadata")
    if not artificial and epoch < 1:
        raise ValueError("A real checkpoint must come from a completed epoch")
    if prefix_digest(model) != prefix_sha256:
        raise ValueError("The frozen generic prefix/norm changed; refuse a suffix-only checkpoint")
    state = model.encoder.state_dict()
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config,
        "targets": list(TARGETS),
        "encoder_provenance": provenance,
        "pixel_contract": pixel_contract(model.config["preprocess"]),
        "source_sha256": native_source_hashes(),
        "head_implementation_sha256": head_implementation_hash(),
        "encoder_suffix": {name: state[name].detach().cpu().clone() for name in sorted(suffix_keys(model))},
        "head": {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()},
        "frozen_prefix_sha256": prefix_sha256,
        "fold": fold,
        "epoch": epoch,
        "weak_bce": weak_bce,
        "selection_data": "artificial_contract" if artificial else "weak_validation",
        "artificial": artificial,
        "gold_used_for_training_or_selection": False,
        "audit": audit,
        "resume_supported": False,
        "optimizer_scaler_RNG_saved": False,
        "note": "Inference weights only. No optimizer/scaler/RNG state; resuming a training run is unsupported.",
    }


def load_native_checkpoint(path, *, allow_artificial=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != CHECKPOINT_SCHEMA or value.get("targets") != list(TARGETS):
        raise ValueError("Unsupported native checkpoint or target order")
    validate_native_config(value["config"])
    if (
        value.get("source_sha256") != native_source_hashes()
        or value.get("head_implementation_sha256") != head_implementation_hash()
    ):
        raise ValueError("Trained model/shared implementation changed")
    if value.get("pixel_contract") != pixel_contract(value["config"]["preprocess"]):
        raise ValueError("Trained native preprocessing changed")
    if value.get("artificial") is not False:
        if not allow_artificial or value.get("selection_data") != "artificial_contract":
            raise ValueError("Artificial checkpoints cannot be used for real predictions")
    elif value.get("selection_data") != "weak_validation":
        raise ValueError("Real checkpoint must be selected on weak validation")
    if value.get("gold_used_for_training_or_selection") is not False or value.get("resume_supported") is not False:
        raise ValueError("Gold-tuned/resumable checkpoint is not permitted here")
    if value.get("fold") not in (0, 1) or not math.isfinite(value.get("weak_bce", float("nan"))):
        raise ValueError("Invalid fixed fold/BCE checkpoint metadata")
    epoch = value.get("epoch")
    if type(epoch) is not int or epoch < (0 if value.get("artificial") else 1):
        raise ValueError("Invalid checkpoint epoch; real selection requires a completed positive epoch")
    return value


def apply_native_checkpoint(model, checkpoint, provenance):
    if checkpoint["config"] != model.config or checkpoint["encoder_provenance"] != provenance:
        raise ValueError("Generic initialization/config differs from checkpoint")
    if prefix_digest(model) != checkpoint["frozen_prefix_sha256"]:
        raise ValueError("Generic prefix/norm does not reconstruct the trained model")
    expected = suffix_keys(model)
    if set(checkpoint["encoder_suffix"]) != expected:
        raise ValueError("Suffix checkpoint keys must exactly match the selected blocks; no prefix/norm keys")
    merged = model.encoder.state_dict()
    for label, supplied, reference in (
        ("suffix", checkpoint["encoder_suffix"], {name: merged[name] for name in expected}),
        ("head", checkpoint["head"], model.head.state_dict()),
    ):
        if set(supplied) != set(reference):
            raise ValueError(f"Strict {label} keys differ")
        for name, tensor in supplied.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != reference[name].shape
                or tensor.dtype != reference[name].dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Strict {label} tensor differs or is nonfinite: {name}")
    merged.update(checkpoint["encoder_suffix"])
    model.encoder.load_state_dict(merged, strict=True)
    model.head.load_state_dict(checkpoint["head"], strict=True)
    return model


def new_native_model(repo, weights, provenance, config, device="cuda"):
    validate_native_config(config)
    encoder = load_frozen_encoder(repo, weights, provenance, device)
    options = config["online"]
    return NativeTripletMIL(
        encoder,
        config,
        trainable_blocks=options["trainable_blocks"],
        window_chunk=options["window_chunk"],
        checkpoint_last2=options["checkpoint_last2"],
    ).to(device)


class NativePixelStore:
    """Metadata preflight and one hashed compressed study at a time; no bulk unpacking."""

    def __init__(self, index_path, config):
        self.index_path = Path(index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if self.index.get("schema_version") != INDEX_SCHEMA or self.index.get("complete") is not True:
            raise ValueError("A complete native pixel archive index is required")
        expected = pixel_contract(config["preprocess"])
        if self.index.get("contract") != expected or self.index.get("fingerprint") != json_hash(expected):
            raise ValueError("Native pixel cache preprocessing/source differs")
        self.studies, self.summaries = {}, []
        for shard in self.index["shards"]:
            summary_path, archive_path = [self._inside_root(shard[key]) for key in ("summary_path", "archive_path")]
            if sha256(summary_path) != shard["summary_sha256"]:
                raise ValueError("Pixel shard summary changed")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("complete") is not True
                or summary.get("for_submission") is not False
                or summary.get("contract") != expected
            ):
                raise ValueError("Incomplete/foreign native pixel shard")
            if (
                summary.get("fingerprint") != self.index["fingerprint"]
                or archive_path.stat().st_size != summary["archive"]["bytes"]
            ):
                raise ValueError("Pixel shard fingerprint/archive size differs")
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(summary["files"]):
                    raise ValueError("Pixel archive member inventory differs")
            for name in summary["files"]:
                key = uid(name.removesuffix(".npz"))
                if name != f"{key}.npz" or key in self.studies:
                    raise ValueError("Unsafe or duplicate native pixel study")
                self.studies[key] = (archive_path, summary)
            self.summaries.append(summary)
        if sorted(self.studies) != self.index["study_ids"]:
            raise ValueError("Pixel index UID inventory differs from shards")

    def _inside_root(self, relative):
        path = (self.index_path.parent / relative).resolve()
        if not path.is_relative_to(self.index_path.parent.resolve()):
            raise ValueError("Pixel shard paths must stay inside the index root")
        return path

    def load(self, key):
        archive, summary = self.studies[uid(key)]
        return load_archive_study(archive, key, summary)


def validate_resource_gate(path, config=None):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = (
        "native392_pixel16_complete",
        "native392_byte_QC_passed",
        "compression_measured",
        "direct_archive_read_speed_measured",
        "artificial_CUDA_gradient_and_roundtrip_passed",
    )
    if value.get("schema_version") != "native392_training_resource_gate_v1" or not all(
        value.get(key) is True for key in required
    ):
        raise ValueError(
            "Pixel16 byteQC/resources/compression/read-speed and artificial CUDA gates must pass before real fitting"
        )
    if value.get("source_sha256") != native_source_hashes():
        raise ValueError("Resource gate belongs to different training/shared code")
    if config is not None and value.get("config_semantic_sha256", {}).get(
        str(config["online"]["trainable_blocks"])
    ) != json_hash(config):
        raise ValueError("Resource gate belongs to different paired config")
    evidence = {}
    workspace = Path(__file__).resolve().parents[2]
    for name in ("pixel16", "processed_QA", "artificial_CUDA"):
        receipt = value.get("evidence", {}).get(name, {})
        supplied_path = receipt.get("path")
        if not isinstance(supplied_path, str) or not supplied_path:
            raise ValueError("Missing hash-bound resource evidence")
        evidence_path = (workspace / supplied_path).resolve()
        if (
            not evidence_path.is_relative_to(workspace)
            or evidence_path.suffix != ".json"
            or sha256(evidence_path) != receipt.get("sha256")
        ):
            raise ValueError("Resource evidence path/hash differs")
        evidence[name] = json.loads(evidence_path.read_text(encoding="utf-8"))
    pixels, qa, artificial = (evidence[key] for key in ("pixel16", "processed_QA", "artificial_CUDA"))
    if (
        pixels.get("all_passed") is not True
        or pixels.get("studies") != 16
        or not all(pixels.get("checks", {}).values())
    ):
        raise ValueError("Actual pixel16 independent evidence failed")
    if (
        qa.get("status") != "native392_label_free_QA_complete"
        or qa.get("real_processed_MRI_pixels_used") is not True
        or qa.get("source_sha256") != native_source_hashes()
    ):
        raise ValueError("Actual processed-pixel CUDA evidence differs or failed")
    if (
        qa.get("optimizer_steps") != 0
        or qa.get("real_teacher_labels_read") is not False
        or qa.get("studies_simultaneously_on_GPU") != 1
    ):
        raise ValueError("Resource QA must be label-free with one GPU study and optimizer0")
    if qa.get("pilot", {}).get("archive_sha256") != pixels["archive_sha256"] or not all(
        row["passed"] for row in qa.get("checks", [])
    ):
        raise ValueError("Processed QA and pixel16 cache or checks differ")
    if {row["arm"] for row in qa.get("arms", [])} != {"freeze", "last2"} or qa.get(
        "config_semantic_sha256"
    ) != value.get("config_semantic_sha256"):
        raise ValueError("Both exact paired configs must pass resource QA")
    if artificial.get("cuda_checks_passed", 0) < 19 or not all(
        row["passed"] for row in artificial.get("cuda_checks", [])
    ):
        raise ValueError("Artificial CUDA gradient/checkpoint evidence failed")
    if artificial.get("source_sha256", {}).get("finetune_model.py") != native_source_hashes()["finetune_model.py"]:
        raise ValueError("Artificial CUDA model implementation differs")
    return value


def read_manifest_structure(path, fields):
    """Project only ID/source/group/fold during the eligibility pass; no teacher conversion."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)) or not set(fields) <= set(header):
            raise ValueError("Malformed manifest structure header")
        indices = [header.index(name) for name in fields]
        rows = []
        for cells in reader:
            if len(cells) != len(header):
                raise ValueError("Malformed manifest structure row")
            rows.append({name: cells[index] for name, index in zip(fields, indices)})
    return rows


def preflight_native_training(manifest_dir, index_path, config, fold, gate_path):
    ensure_cuda()  # No config/labels/cache/model read or output mutation before this guard.
    validate_native_config(config)
    if fold not in (0, 1):
        raise ValueError("This paired experiment permits only fixed folds0/1")
    gate = validate_resource_gate(gate_path, config)
    manifest_dir = Path(manifest_dir)
    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["seed"] != config["seed"]
        or manifest["n_folds"] != config["n_folds"]
        or manifest["label_provenance"].get("gold_used_for_tuning") is not False
    ):
        raise ValueError("Reviewed fixed weak split/provenance differs")
    audit = validate_fold_audit(manifest_dir, manifest_dir / "fold-audit-v2.json", manifest)
    weak_structure = read_manifest_structure(manifest_dir / "weak.csv", (ID, "source", "group_id", "fold"))
    gold = read_manifest_structure(manifest_dir / "gold.csv", (ID, "source", "group_id"))
    # Assert ID/group/source/folds before retaining or converting any weak teacher values.
    train_structure, valid_structure = check_split(weak_structure, gold, fold, config["n_folds"])
    grouped = {}
    for row in weak_structure:
        if row["group_id"] in grouped and grouped[row["group_id"]] != int(row["fold"]):
            raise ValueError("One supplied/report group crosses fixed folds")
        grouped[row["group_id"]] = int(row["fold"])
    _, weak = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "source", "group_id", "fold"))
    keyed = index_studies(weak)
    if set(keyed) != set(index_studies(weak_structure)) or any(
        any(keyed[row[ID]][field] != row[field] for field in ("source", "group_id", "fold")) for row in weak_structure
    ):
        raise ValueError("Manifest structure changed between eligibility and teacher passes")
    training = [keyed[row[ID]] for row in train_structure]
    validation = [keyed[row[ID]] for row in valid_structure]
    for row in training + validation:
        effective_observation_count([row])
    store = NativePixelStore(index_path, config)
    hashes = {
        name: sha256(manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")
    }
    if (
        store.index.get("split") != "train"
        or store.index.get("limited") is not False
        or set(store.studies) != set(index_studies(weak))
    ):
        raise ValueError(
            "Real fitting requires the exact complete weak pixel set; a16-study pilot cannot initialize a full run"
        )
    if store.index.get("manifest_sha256") != hashes or store.index.get(
        "selection_structure_sha256"
    ) != selection_structure_hash(weak):
        raise ValueError("Pixel set and reviewed weak group/fold manifest are not bound")
    return (
        training,
        validation,
        store,
        {
            "manifest": manifest,
            "manifest_sha256": hashes,
            "fold_audit": audit,
            "pixel_index_sha256": sha256(index_path),
            "resource_gate_sha256": sha256(gate_path),
            "resource_gate": gate,
        },
    )


def tensor_study(arrays, device="cuda"):
    return {name: torch.from_numpy(value).unsqueeze(0).to(device) for name, value in arrays.items()}


def read_prediction_ids(path):
    _, rows = read_csv(path, (ID,))
    selected = [{ID: row[ID]} for row in rows]
    index_studies(selected)
    return selected


@torch.inference_mode()
def evaluate_native(model, rows, store):
    model.eval()
    predictions, numerator, denominator = [], 0.0, 0
    for row in rows:
        logits = model(**tensor_study(store.load(row[ID]))).float()
        targets, observed = label_tensors(row, logits.device)
        numerator += float(observed_bce_sum(logits, targets, observed))
        denominator += int(observed.sum())
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite native validation output")
        predictions.append({ID: row[ID], **dict(zip(TARGETS, logits.sigmoid()[0].cpu().tolist()))})
    return predictions, numerator / denominator


def train_native(manifest_dir, index_path, output, config, fold, repo, weights, provenance, gate_path):
    ensure_cuda()
    # Validate all metadata before creating a new run or reading the model checkpoint.
    training, validation, store, audit = preflight_native_training(manifest_dir, index_path, config, fold, gate_path)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    model = new_native_model(repo, weights, provenance, config)
    prefix_sha = prefix_digest(model)
    options, settings = config["online"], config["train"]
    groups = [{"params": list(model.head.parameters()), "lr": settings["learning_rate"], "name": "head"}]
    encoder_params = [value for value in model.encoder.parameters() if value.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": options["encoder_learning_rate"], "name": "last2"})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    counter = OptimizerStepCounter(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=options["amp_grad_scaler"])
    parameters = [value for value in model.parameters() if value.requires_grad]
    out = new_directory(output)
    dump_json(out / "config.json", config)
    record = {
        "schema_version": "native392_training_run_v1",
        "status": "running",
        "config": config,
        "seed": config["seed"],
        "fold": fold,
        "train_studies": len(training),
        "valid_studies": len(validation),
        "audit": audit,
        "source_sha256": native_source_hashes(),
        "encoder_provenance": provenance,
        "parameters": model.parameter_contract(),
        "targets": list(TARGETS),
        "gpu": torch.cuda.get_device_name(0),
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
        "selection_data": "weak_validation",
        "selection_rule": "bce",
        "gold_used_for_training_or_selection": False,
        "gold_evaluated": False,
        "patient_independence_verified": False,
        "history": [],
        "public_lb": None,
        "resume_supported": False,
        "microbatch_studies": 1,
        "effective_batch_studies": 16,
        "denominator": "sum observed teacher cells over each actual effective group, including the short final group",
    }
    dump_json(out / "run.json", record)
    generator = torch.Generator().manual_seed(config["seed"])
    selected, start = None, time.perf_counter()
    try:
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            ordered = [training[index] for index in torch.randperm(len(training), generator=generator).tolist()]
            loss_sum, count = 0.0, 0
            steps_start, opportunities_start = counter.steps, counter.opportunities
            for offset in range(0, len(ordered), options["effective_batch_studies"]):
                group = ordered[offset : offset + options["effective_batch_studies"]]
                denominator = effective_observation_count(group)
                optimizer.zero_grad(set_to_none=True)
                for row in group:
                    inputs = tensor_study(store.load(row[ID]))  # One study on GPU at a time.
                    targets, observed = label_tensors(row, "cuda")
                    logits = model(**inputs)
                    loss_sum += backward_observed_micro(logits, targets, observed, denominator, scaler=scaler)
                    count += int(observed.sum())
                    del inputs, targets, observed, logits
                scaler.unscale_(optimizer)
                finite = all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in parameters)
                if finite:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                counter.opportunity()
                scaler.step(optimizer)  # AMP records and skips overflow; never count skipped opportunities as updates.
                scaler.update()
            if counter.steps == steps_start:
                raise ValueError("No optimizer update completed in the epoch")
            predictions, bce = evaluate_native(model, validation, store)
            metrics = evaluate_rows(index_studies(validation), index_studies(predictions))
            predictions_path = out / f"epoch_{epoch:03d}_weak.csv"
            write_csv(predictions_path, SUBMISSION_COLUMNS, predictions)
            item = {
                "epoch": epoch,
                "train_bce": loss_sum / count,
                "weak_bce": bce,
                "weak_macro_auc_12": metrics["macro_auc_12"],
                "metrics": metrics,
                "optimizer_steps": counter.steps - steps_start,
                "optimizer_step_opportunities": counter.opportunities - opportunities_start,
                "amp_skipped_steps": counter.opportunities - opportunities_start - (counter.steps - steps_start),
                "elapsed_seconds": time.perf_counter() - start,
                "prediction_sha256": sha256(predictions_path),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            record["history"].append(item)
            checkpoint = make_native_checkpoint(
                model, provenance, fold=fold, epoch=epoch, weak_bce=bce, prefix_sha256=prefix_sha, audit=audit
            )
            torch.save(checkpoint, out / "last.pt")
            if selected is None or bce < selected:
                selected = bce
                torch.save(checkpoint, out / "best.pt")
                write_csv(out / "weak_valid_predictions.csv", SUBMISSION_COLUMNS, predictions)
                record["selected"] = {**item, "checkpoint_sha256": sha256(out / "best.pt")}
            dump_json(out / "run.json", record)
            print(f"Native epoch {epoch}/12: weak BCE={bce:.6f}, AUC12={metrics['macro_auc_12']}", flush=True)
        record.update(status="complete", elapsed_seconds=time.perf_counter() - start)
    except Exception as error:
        record.update(
            status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - start
        )
        raise
    finally:
        counter.close()
        dump_json(out / "run.json", record)
    return record


def predict_native(root, checkpoint_path, output_csv, repo, weights, provenance, *, reader):
    ensure_cuda()
    root = Path(root)
    if not Path("/kaggle/input").is_dir():
        raise ValueError("Native MRI inference belongs on Kaggle; local CLI accepts processed caches for training only")
    if Path(output_csv).exists():
        raise ValueError("Use a new prediction path")
    checkpoint = load_native_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    model = new_native_model(repo, weights, provenance, config)
    apply_native_checkpoint(model, checkpoint, provenance).eval()
    ids = read_prediction_ids(root / "test.csv")  # Report/targets are never required or passed to the model.
    _, series = read_csv(root / "test_series.csv", SERIES_COLUMNS)
    by_study = {}
    for row in series:
        by_study.setdefault(row[ID], []).append(row)

    # Reuse the complete shared pixel extraction contract; only the split-root mapping changes.
    def test_reader(train_shaped_path, settings):
        relative = Path(train_shaped_path).relative_to(root / "train_series")
        return reader(root / "test_series" / relative, settings)

    predictions = []
    with torch.inference_mode():
        for row in ids:
            arrays, _ = extract_pixel_study(
                root, row[ID], by_study.get(row[ID], []), config["preprocess"], reader=test_reader
            )
            logits = model(**tensor_study(arrays))
            if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                raise ValueError("Invalid native inference output")
            predictions.append({ID: row[ID], **dict(zip(TARGETS, logits.sigmoid()[0].cpu().tolist()))})
    write_csv(output_csv, SUBMISSION_COLUMNS, predictions)
    receipt = validate_submission(output_csv, root / "test.csv", root / "sample_submission.csv")
    dump_json(
        Path(output_csv).with_suffix(".receipt.json"),
        {
            **receipt,
            "checkpoint_sha256": sha256(checkpoint_path),
            "submission_sha256": sha256(output_csv),
            "Report_required": False,
            "public_score": None,
        },
    )
    return receipt
