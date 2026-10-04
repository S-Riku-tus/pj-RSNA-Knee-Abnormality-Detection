"""Exercise CUDA with generated noise only; never download weights or decode MRI."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rsna_knee.contracts import (
    ID,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    load_config,
    preprocess_fingerprint,
    read_csv,
    sha256,
    source_hashes,
    write_csv,
)
from rsna_knee.imaging import load_cached
from rsna_knee.model import KneeMIL
from rsna_knee.prepare import prepare
from rsna_knee.runtime import StudyDataset, ensure_cuda, masked_bce, predict, train

ROOT = Path(__file__).resolve().parents[1]


def create_noise_cache(directory, rows, csv_path, config, split, *, seed):
    directory.mkdir()
    p = config["preprocess"]
    windows = p["max_series"] * p["windows_per_series"]
    shape = (windows, 3, p["image_size"], p["image_size"])
    fingerprint = preprocess_fingerprint(config)
    dump_json(
        directory / "cache.json",
        {"fingerprint": fingerprint, "split": split, "studies_csv_sha256": sha256(csv_path)},
    )
    generator = np.random.default_rng(seed)
    for row in rows:
        images = generator.integers(0, 256, shape, dtype=np.uint8)
        mask = np.ones(windows, dtype=bool)
        if windows > p["windows_per_series"]:
            mask[-p["windows_per_series"] :] = False  # Artificial missing series.
        np.savez_compressed(directory / f"{row[ID]}.npz", images=images, mask=mask, fingerprint=fingerprint)


def verify_missing_mask(rows, cache_dir, config):
    example = StudyDataset(rows, cache_dir, config)[0]
    _, windows, targets, observed, _ = example
    assert windows.any() and not windows.all()
    assert observed.any() and not observed.all()
    changed = targets.clone()
    changed[~observed] = 1
    logits = torch.linspace(-1, 1, len(TARGETS)).reshape(1, -1)
    torch.testing.assert_close(
        masked_bce(logits, targets.unsqueeze(0), observed.unsqueeze(0)),
        masked_bce(logits, changed.unsqueeze(0), observed.unsqueeze(0)),
        rtol=0,
        atol=0,
    )
    return {"missing_labels_excluded": True, "invalid_windows_present": True}


def reject_normalization_mismatch(checkpoint, test_csv, cache_dir, directory):
    broken = copy.deepcopy(checkpoint)
    original = broken["config"]["model"].get("input_normalization", "legacy")
    broken["config"]["model"]["input_normalization"] = "imagenet" if original == "legacy" else "legacy"
    path = directory / "mismatched-normalization.pt"
    torch.save(broken, path)
    output = directory / "must-not-exist.csv"
    try:
        predict(test_csv, cache_dir, path, output, "cuda")
    except ValueError as error:
        assert "input normalization" in str(error)
    else:
        raise AssertionError("Checkpoint normalization mismatch was accepted")
    assert not output.exists()
    return True


def check_mode(mode, output, config, manifest_dir, train_cache, test_cache, test_csv, gold_ids):
    directory = output / mode
    directory.mkdir()
    config = copy.deepcopy(config)
    config["model"]["input_normalization"] = mode
    dump_json(directory / "config.json", config)
    _, weak_rows = read_csv(manifest_dir / "weak.csv")
    mask_checks = verify_missing_mask(weak_rows, train_cache, config)
    torch.manual_seed(config["seed"])
    initial = KneeMIL(config["model"]["dropout"]).encoder.conv1.weight.detach().clone()
    read_ids = []

    def read_generated_only(cache, study, settings):
        assert study.startswith("synthetic-") and study not in gold_ids
        assert Path(cache).resolve() == train_cache.resolve()
        read_ids.append(study)
        return load_cached(cache, study, settings)

    with (
        patch("rsna_knee.runtime.load_cached", side_effect=read_generated_only),
        patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Downloads are forbidden")),
    ):
        training = train(manifest_dir, train_cache, directory / "run", config, 0)
    assert not gold_ids.intersection(read_ids)
    assert all(not (train_cache / f"{study}.npz").exists() for study in gold_ids)
    run_dir = directory / "run"
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == config["train"]["epochs"]
    assert all(np.isfinite(row["weak_valid_masked_bce"]) for row in history)
    assert not (run_dir / "gold_predictions.csv").exists()
    assert not (run_dir / "gold_metrics.json").exists()
    assert (run_dir / "epochs" / "001" / "train_eval_metrics.json").is_file()
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert not torch.equal(initial, checkpoint["model"]["encoder.conv1.weight"])
    with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Downloads are forbidden")):
        contract = predict(test_csv, test_cache, run_dir / "best.pt", directory / "submission.csv", "cuda")
    assert contract["valid"] and contract["studies"] == 3 and contract["targets"] == 12
    rejected = reject_normalization_mismatch(checkpoint, test_csv, test_cache, directory)
    dataset = StudyDataset([{ID: "synthetic-test-0"}], test_cache, config)
    images, mask, _, _, _ = dataset[0]
    model = KneeMIL(config["model"]["dropout"]).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    images, mask = images.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda()
    changed = images.clone()
    changed[~mask] = 999
    with torch.inference_mode():
        torch.testing.assert_close(model(images, mask), model(changed, mask), rtol=0, atol=0)
    legacy_contract = None
    if mode == "legacy":
        old = copy.deepcopy(checkpoint)
        old.pop("input_normalization")
        for name in ("input_normalization", "initialization", "bn_running_stats"):
            old["config"]["model"].pop(name, None)
        old_path = directory / "legacy-format.pt"
        torch.save(old, old_path)
        legacy_contract = predict(test_csv, test_cache, old_path, directory / "legacy-format-submission.csv", "cuda")
        assert legacy_contract["valid"]
        assert sha256(directory / "legacy-format-submission.csv") == sha256(directory / "submission.csv")
    return {
        "normalization": mode,
        "training": training,
        "synthetic_epochs": len(history),
        "amp_enabled": config["train"]["amp"],
        "optimizer_changed_encoder": True,
        "gold_cache_created": False,
        "gold_uids_loaded": False,
        "synthetic_cache_reads": len(read_ids),
        "epoch_diagnostics_written": True,
        "submission_contract": contract,
        "invalid_windows_do_not_change_prediction": True,
        "normalization_mismatch_rejected": rejected,
        "legacy_format_contract": legacy_contract,
        **mask_checks,
    }


def check_legacy_checkpoint(path, output, test_csv, rows):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint.get("schema_version") == 1 and checkpoint.get("targets") == list(TARGETS)
    assert checkpoint.get("input_normalization", "legacy") == "legacy"
    config = checkpoint["config"]
    assert config["model"].get("input_normalization", "legacy") == "legacy"
    directory = output / "existing-legacy-checkpoint"
    directory.mkdir()
    cache = directory / "artificial-test-cache"
    create_noise_cache(cache, rows, test_csv, config, "test", seed=808)
    with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Downloads are forbidden")):
        contract = predict(test_csv, cache, path, directory / "submission.csv", "cuda")
    assert contract["valid"] and contract["studies"] == 3
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256(path),
        "input_size": config["preprocess"]["image_size"],
        "test_images": "independently generated noise, no MRI",
        "submission_contract": contract,
        "normalization_mismatch_rejected": reject_normalization_mismatch(checkpoint, test_csv, cache, directory),
    }


def check(output_dir, legacy_checkpoint=None):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; previous QA artifacts must be preserved")
    ensure_cuda()
    if legacy_checkpoint is not None and not Path(legacy_checkpoint).is_file():
        raise ValueError("The optional local legacy checkpoint does not exist")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    config = load_config(ROOT / "configs" / "baseline.json")
    config["preprocess"].update(image_size=32, max_series=3, windows_per_series=2)
    config["train"].update(epochs=2, batch_size=1, accumulation_steps=2, num_workers=0, amp=True)
    config["model"].update(initialization="random", bn_running_stats="update")
    config["diagnostics"] = {"enabled": True, "train_eval_studies": 4, "audit_gold": False}
    rows = [
        {
            ID: f"synthetic-train-{index:03d}",
            "Report": f"Artificial report {index}",
            **{target: "" for target in TARGETS},
        }
        for index in range(26)
    ]
    for index in range(2):
        rows[index].update({target: str(index) for target in TARGETS})
    gold_ids = {row[ID] for row in rows[:2]}
    labels = [
        {
            ID: row[ID],
            **{
                target: "" if (index + column) % 7 == 0 else str((index + column) % 2)
                for column, target in enumerate(TARGETS)
            },
        }
        for index, row in enumerate(rows[2:])
    ]
    write_csv(output / "train.csv", (ID, "Report", *TARGETS), rows)
    write_csv(output / "labels.csv", SUBMISSION_COLUMNS, labels)
    dump_json(
        output / "provenance.json",
        {
            "source_url": "synthetic://generated-noise",
            "source_version": "controlled-pipeline-1",
            "license": "generated-test-only",
            "method": "Artificial arrays and labels, no competition MRI or report",
            "created_at": "2026-10-04",
            "gold_used_for_tuning": False,
        },
    )
    manifest_dir = output / "manifests"
    manifest = prepare(
        output / "train.csv",
        output / "labels.csv",
        output / "provenance.json",
        manifest_dir,
        n_folds=config["n_folds"],
        seed=config["seed"],
    )
    assert manifest["gold_studies"] == 2 and manifest["weak_studies"] == 24
    _, weak_rows = read_csv(manifest_dir / "weak.csv")
    assert index_studies(weak_rows).keys().isdisjoint(gold_ids)
    train_cache = output / "artificial-train-cache"
    create_noise_cache(train_cache, rows[2:], output / "train.csv", config, "train", seed=101)
    test_rows = [{ID: f"synthetic-test-{index}"} for index in range(3)]
    test_csv = output / "test.csv"
    write_csv(test_csv, (ID,), test_rows)  # No Report column.
    test_cache = output / "artificial-test-cache"
    create_noise_cache(test_cache, test_rows, test_csv, config, "test", seed=202)
    results = {
        mode: check_mode(mode, output, config, manifest_dir, train_cache, test_cache, test_csv, gold_ids)
        for mode in ("legacy", "imagenet")
    }
    legacy = None
    if legacy_checkpoint is not None:
        legacy = check_legacy_checkpoint(Path(legacy_checkpoint), output, test_csv, test_rows)
    summary = {
        "scope": "Generated noise only; engineering QA, not competition learning or model performance",
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "model_downloaded": False,
        "actual_mri_decoded": False,
        "real_training_started": False,
        "test_csv_has_report": False,
        "synthetic_weak_studies": len(weak_rows),
        "synthetic_reserved_gold_studies": len(gold_ids),
        "gold_cache_created": False,
        "checks": results,
        "existing_legacy_checkpoint": legacy,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.perf_counter() - started,
        "script_sha256": sha256(__file__),
        "source_sha256": source_hashes(),
        "output_dir": str(output),
    }
    dump_json(output / "checks.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--legacy-checkpoint", type=Path, help="Optional existing local checkpoint; uses generated test images"
    )
    arguments = parser.parse_args()
    result = check(arguments.output_dir, arguments.legacy_checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
