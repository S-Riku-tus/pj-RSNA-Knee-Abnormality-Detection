"""Label-free, same-input A5 numerical/speed probe embedded by build_a5_precision_probe.py."""

import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def select_uids(values, count, seed):
    values = set(values)
    if count < 2 or len(values) < count or any(not isinstance(uid, str) or not uid.strip() for uid in values):
        raise ValueError("Need at least the requested number of distinct nonempty train UIDs (count >= 2)")
    return sorted(values, key=lambda uid: (hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(), uid))[:count]


def average_ranks(values):
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Ranks require finite nonempty values")
    indices = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    begin = 0
    while begin < len(indices):
        end = begin + 1
        while end < len(indices) and values[indices[end]] == values[indices[begin]]:
            end += 1
        for index in indices[begin:end]:
            result[index] = (begin + end - 1) / 2
        begin = end
    return result


def compare_vectors(reference, candidate):
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("Need equally sized vectors with at least two studies")
    reference, candidate = list(map(float, reference)), list(map(float, candidate))
    rank_r, rank_c = average_ranks(reference), average_ranks(candidate)
    diffs = [abs(a - b) for a, b in zip(reference, candidate)]
    rank_diffs = [abs(a - b) / (len(reference) - 1) for a, b in zip(rank_r, rank_c)]
    inversions = tie_changes = comparable = 0
    for i, j in itertools.combinations(range(len(reference)), 2):
        dr, dc = reference[i] - reference[j], candidate[i] - candidate[j]
        inversions += int(dr * dc < 0)
        tie_changes += int((dr == 0) != (dc == 0))
        comparable += int(dr != 0 and dc != 0)
    center = (len(reference) - 1) / 2
    denom = math.sqrt(sum((x - center) ** 2 for x in rank_r) * sum((x - center) ** 2 for x in rank_c))
    return {
        "max_abs_probability_difference": max(diffs),
        "mean_abs_probability_difference": statistics.mean(diffs),
        "max_normalized_rank_difference": max(rank_diffs),
        "mean_normalized_rank_difference": statistics.mean(rank_diffs),
        "rank_spearman": sum((a - center) * (b - center) for a, b in zip(rank_r, rank_c)) / denom if denom else None,
        "pair_inversions": inversions,
        "pair_tie_changes": tie_changes,
        "comparable_pairs": comparable,
    }


def check_probability_shape(array, expected):
    import numpy as np

    if array.shape != expected:
        raise ValueError(f"Unexpected output shape {array.shape}, expected {expected}")
    bad = int((~np.isfinite(array)).sum())
    outside = int(((array < 0) | (array > 1)).sum())
    return {"nonfinite": bad, "outside_probability_range": outside, "valid": bad == outside == 0}


def resolve_inputs(contract):
    resolved, receipts = {}, []
    for entry in contract["inputs"]:
        roots = sorted({Path(root).resolve() for root in entry["roots"] if Path(root).is_dir()})
        if len(roots) != 1:
            raise ValueError(f"Exactly one fixed mount required for {entry['key']}: {roots}")
        root = roots[0]
        resolved[entry["key"]] = root
        for expected in entry["files"]:
            path = root / expected["path"]
            if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Fixed input hash/size mismatch: {path}")
            receipts.append({"path": str(path), "bytes": expected["bytes"], "sha256": expected["sha256"]})
    return resolved, receipts


def environment(torch):
    versions = {}
    for package in ("torch", "torchvision", "timm", "numpy", "pandas", "pydicom", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
                "native_bf16_hardware": torch.cuda.get_device_capability(i) >= (8, 0),
            }
            for i in range(torch.cuda.device_count())
        ],
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def save_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_models(ns, checkpoint_paths, torch):
    models, configs, folds = [], [], []
    started = time.perf_counter()
    keys = ("backbone", "img", "cond", "pool", "stem", "n_slice", "n_meta", "norm")
    for checkpoint in checkpoint_paths:
        # The full checkpoint has already passed the pinned SHA-256 gate.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        if cfg.get("n_meta", 0) != 0 or cfg.get("n_slice", 16) != 16 or cfg["img"] != 336:
            raise ValueError("A5 checkpoint changed metadata/slice/image input contract")
        if configs and any(cfg.get(key) != configs[0].get(key) for key in keys):
            raise ValueError("A5 folds disagree on model/preprocessing config")
        stem = cfg.get("stem", "native")
        channels = 3 if stem == "compress" else cfg.get("n_slice", 16)
        kwargs = {"img_size": cfg["img"]} if "vit_" in cfg["backbone"] else {}
        encoder = ns["timm"].create_model(cfg["backbone"], pretrained=False, num_classes=0, in_chans=channels, **kwargs)
        if cfg["cond"] == "token":
            encoder = ns["ViTSlotToken"](encoder, ns["N_SLOT_TYPES"])
        model = ns["Net"](
            encoder, cfg["cond"], cfg.get("n_meta", 0), cfg["pool"], stem=stem, n_slice=cfg.get("n_slice", 16)
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model.eval().to("cuda:0"))
        folds.append(int(payload["fold"]))
        configs.append(cfg)
        del payload
    if folds != list(range(5)):
        raise ValueError(f"Expected ordered folds 0..4, got {folds}")
    torch.cuda.synchronize(0)
    ns.update(models=models, CFG=configs[-1], DEV="cuda:0")
    return {"seconds": time.perf_counter() - started, "folds": folds, "configs": configs}


def prepare_train_cache(ns, competition, run_dir, count, seed, np, pd):
    started = time.perf_counter()
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"]
    metadata = competition / "train_series.csv"
    frame = pd.read_csv(metadata, usecols=cols, dtype={cols[0]: str, cols[1]: str})
    ids = select_uids(frame.StudyInstanceUID.unique().tolist(), count, seed)
    selected = frame[frame.StudyInstanceUID.isin(ids)]
    by = {uid: group.to_dict("records") for uid, group in selected.groupby("StudyInstanceUID")}
    (run_dir / "uids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    selected.to_csv(run_dir / "selected_series_metadata.csv", index=False)
    ns["SERIES_ROOT"] = competition / "train_series"
    if not ns["SERIES_ROOT"].is_dir():
        raise FileNotFoundError(ns["SERIES_ROOT"])
    image_path = run_dir / "preprocessed_images.npy"
    mask_path = run_dir / "preprocessed_masks.npy"
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype="uint8", shape=(count, 6, 16, 336, 336))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype="uint8", shape=(count, 6))
    per_study = []
    decode_failures = [0]
    original_read_crop = ns["read_crop"]

    def observed_read_crop(path):
        result = original_read_crop(path)
        if result is None:
            decode_failures[0] += 1
        return result

    ns["read_crop"] = observed_read_crop
    try:
        for index, uid in enumerate(ids):
            before = decode_failures[0]
            _, image, mask = ns["build_study"]((index, uid, by[uid]))
            if not (mask > 0).any() or not image.any():
                raise ValueError(f"No valid A5 image slots for fixed train study {uid}")
            images[index], masks[index] = image, mask
            per_study.append(
                {
                    "uid": uid,
                    "mask_counts": mask.tolist(),
                    "nonzero_slices": (image.max(axis=(2, 3)) > 0).sum(axis=1).tolist(),
                    "failed_crop_reads": decode_failures[0] - before,
                }
            )
            print(f"A5 fixed train preprocessing {index + 1}/{count}", flush=True)
        images.flush()
        masks.flush()
    finally:
        ns["read_crop"] = original_read_crop
    del images, masks
    images = np.load(image_path, mmap_mode="c", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="c", allow_pickle=False)
    return (
        ids,
        images,
        masks,
        {
            "seconds": time.perf_counter() - started,
            "selection": "sha256(seed:UID) ascending; train_series only",
            "metadata_sha256": digest(metadata),
            "metadata_columns_read": cols,
            "uid_sha256": digest(run_dir / "uids.txt"),
            "images_sha256": digest(image_path),
            "masks_sha256": digest(mask_path),
            "image_shape": list(images.shape),
            "per_study": per_study,
            "failed_crop_reads": decode_failures[0],
            "source_preprocessing_preserved": True,
            "execution": "serial once; forward timing excludes preprocessing",
        },
    )


def measure_once(ns, images, masks, mode, torch, trace_dtypes=False):
    dtype, enabled = {"bf16": (torch.bfloat16, True), "fp16": (torch.float16, True), "fp32": (torch.float32, False)}[
        mode
    ]
    ns["AMP_DT"], ns["AMP_ON"] = dtype, enabled
    # Same input/model/MICRO/device and autocast scope as p002 _micro().
    observed, hooks = {}, []
    logits_by_fold = [[] for _ in ns["models"]]

    def capture_logits(fold):
        def hook(module, inputs, output):
            # Retain without modifying logits; reduction/copy happen after synchronized timing.
            logits_by_fold[fold].append(output.detach())

        return hook

    for fold, model in enumerate(ns["models"]):
        hooks.append(model.register_forward_hook(capture_logits(fold)))
    if trace_dtypes:

        def capture(name):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                observed[name] = {
                    "output_dtype": str(tensor.dtype) if hasattr(tensor, "dtype") else type(tensor).__name__,
                    "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                }

            return hook

        for fold, model in enumerate(ns["models"]):
            # The first Linear/Conv2d and final logits expose actual dtypes, not just requested AMP.
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    hooks.append(module.register_forward_hook(capture(f"fold{fold}.{name}")))
                    break
            hooks.append(model.register_forward_hook(capture(f"fold{fold}.logits")))
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prediction = ns["predict"](images, masks)
            torch.cuda.synchronize(0)
            seconds = time.perf_counter() - start
    finally:
        for hook in hooks:
            hook.remove()
    logits = torch.stack([torch.cat(values, dim=0).float() for values in logits_by_fold]).cpu().numpy()
    if logits.shape != prediction.shape:
        raise ValueError("Captured raw logits do not cover every fold/study/label")
    return (
        prediction,
        logits,
        {
            "seconds": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "warnings": sorted({str(item.message) for item in caught}),
            "dtype": str(dtype),
            "autocast": enabled,
            "observed_dtypes": observed,
            "nonfinite_logits": int((~ns["np"].isfinite(logits)).sum()),
            "timing_observer": "nonmutating forward hooks retain logits; checks/copy/save excluded",
        },
    )


def comparisons(raw, labels, pd, np):
    rows = []
    for left, right in itertools.combinations(sorted(raw), 2):
        # compare against FP32 where present and BF16 otherwise.
        reference, candidate = (right, left) if right == "fp32" else (left, right)
        for fold in range(5):
            for col, label in enumerate(labels):
                rows.append(
                    {
                        "reference": reference,
                        "candidate": candidate,
                        "fold": fold,
                        "label": label,
                        **compare_vectors(raw[reference][fold, :, col], raw[candidate][fold, :, col]),
                    }
                )
        # Reproduce the source's ordinal rank followed by fold mean, including its tie handling.
        rank_means = {}
        for mode in (reference, candidate):
            per_fold = raw[mode].argsort(axis=1).argsort(axis=1).astype(np.float64)
            rank_means[mode] = (per_fold / max(raw[mode].shape[1] - 1, 1)).mean(axis=0)
        for col, label in enumerate(labels):
            rows.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "fold": "p002_ordinal_rank_mean",
                    "label": label,
                    **compare_vectors(rank_means[reference][:, col], rank_means[candidate][:, col]),
                }
            )
    return pd.DataFrame(rows)


def run_probe(contract, definition_source, studies=64, seed=20261005, repeats=3):
    """Called only by the user on Kaggle. Never creates or modifies a submission CSV."""
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("This diagnostic is restricted to Kaggle; no local MRI/model execution")
    if studies != 64 or repeats < 2:
        raise ValueError("First comparison uses exactly 64 fixed train studies and at least two repeats")
    if hashlib.sha256(definition_source.encode()).hexdigest() != contract["source"]["definitions_sha256"]:
        raise ValueError("Definition source hash mismatch")
    roots, input_receipts = resolve_inputs(contract)
    wheel = roots["offline_timm"] / "timm-1.0.22-py3-none-any.whl"
    if "timm" in sys.modules:
        raise RuntimeError("Use a fresh session before installing the fixed offline timm wheel")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)], check=True
    )
    import numpy as np
    import pandas as pd
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Use T4 x2, as in the scored p002 notebook")
    if any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("This comparison must use T4 x2")
    if importlib.metadata.version("timm") != "1.0.22":
        raise RuntimeError("Offline timm version mismatch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    run_dir = Path("/kaggle/working") / (
        "a5-precision-probe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run_dir.mkdir(exist_ok=False)
    report_path = run_dir / "report.json"
    report = {
        "schema": contract["schema"],
        "status": "running",
        "run_directory": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "studies": studies,
        "repeats": repeats,
        "micro": 8,
        "folds": list(range(5)),
        "device": "cuda:0",
        "labels_read": False,
        "gold_used": False,
        "training_performed": False,
        "public_score_measured": False,
        "submission_generated": False,
        "adoption_decision": "undecided; diagnostic only; manual review required; no automatic precision change",
        "environment": environment(torch),
        "input_files": input_receipts,
        "source": contract["source"],
        "modes": {},
        "comparisons": {},
        "runtime_scope": "A5 only; full 9-hour submission budget remains unverified",
    }
    save_report(report_path, report)
    try:
        ns = {"__name__": "a5_p002_definitions"}
        exec(compile(definition_source, "<p002-a5-definitions>", "exec"), ns)
        ns["cv2"].setNumThreads(1)
        paths = sorted((roots["consolidated"] / "knee-mri-fold-weights").glob("*_f*.pt"))
        expected_paths = sorted(
            roots["consolidated"] / item["path"]
            for entry in contract["inputs"]
            if entry["key"] == "consolidated"
            for item in entry["files"]
        )
        if paths != expected_paths or len(paths) != 5:
            raise ValueError("Expected exactly the five hash-verified A5 checkpoints")
        report["model_load"] = make_models(ns, paths, torch)
        ids, images, masks, cache = prepare_train_cache(ns, roots["competition"], run_dir, studies, seed, np, pd)
        report["preprocessing"] = cache
        report["label_order"] = list(ns["LABELS"])
        save_report(report_path, report)
        modes = list(contract["modes"])
        active = []
        for mode in modes:
            record = {"status": "warming_up", "measurements": []}
            report["modes"][mode] = record
            try:
                prediction, _, timing = measure_once(ns, images[:8], masks[:8], mode, torch, trace_dtypes=True)
                valid = check_probability_shape(prediction, (5, 8, 12))
                record["warmup"] = {**timing, **valid}
                if not valid["valid"] or timing["nonfinite_logits"]:
                    raise ValueError("Warmup produced invalid probabilities/logits")
                record["status"] = "ready"
                active.append(mode)
            except Exception as exc:
                record.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                gc.collect()
                torch.cuda.empty_cache()
            save_report(report_path, report)
        raw = {}
        for repeat in range(repeats):
            # Rotate mode order; do not silently change dtype if a mode fails.
            order = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]
            for mode in order:
                if mode not in active:
                    continue
                record = report["modes"][mode]
                try:
                    prediction, logits, timing = measure_once(ns, images, masks, mode, torch)
                    valid = check_probability_shape(prediction, (5, studies, 12))
                    raw_path = run_dir / f"raw_probabilities_{mode}_repeat{repeat + 1}.npy"
                    np.save(raw_path, prediction, allow_pickle=False)
                    logits_path = run_dir / f"raw_logits_{mode}_repeat{repeat + 1}.npy"
                    np.save(logits_path, logits, allow_pickle=False)
                    record["measurements"].append(
                        {
                            "repeat": repeat + 1,
                            **timing,
                            **valid,
                            "raw_path": raw_path.name,
                            "sha256": digest(raw_path),
                            "raw_logits_path": logits_path.name,
                            "raw_logits_sha256": digest(logits_path),
                        }
                    )
                    if not valid["valid"] or timing["nonfinite_logits"]:
                        raise ValueError("Invalid probabilities/logits saved; no neutralization permitted")
                    if mode in raw:
                        record["measurements"][-1]["max_abs_repeat_difference"] = float(
                            np.abs(raw[mode] - prediction).max()
                        )
                    else:
                        raw[mode] = prediction.copy()
                        for fold in range(5):
                            frame = pd.DataFrame(prediction[fold], columns=ns["LABELS"])
                            frame.insert(0, "StudyInstanceUID", ids)
                            frame.to_csv(run_dir / f"a5_raw_{mode}_fold{fold}.csv", index=False)
                    print(f"A5 {mode} repeat {repeat + 1}/{repeats}: {timing['seconds']:.2f} sec", flush=True)
                except Exception as exc:
                    record.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
                    active.remove(mode)
                    raw.pop(mode, None)
                    gc.collect()
                    torch.cuda.empty_cache()
                save_report(report_path, report)
        for mode in active:
            record = report["modes"][mode]
            seconds = [item["seconds"] for item in record["measurements"]]
            record.update(
                status="completed",
                median_seconds=statistics.median(seconds),
                min_seconds=min(seconds),
                max_seconds=max(seconds),
                studies_per_second=studies / statistics.median(seconds),
            )
        table = comparisons(raw, ns["LABELS"], pd, np)
        table.to_csv(run_dir / "comparisons.csv", index=False)
        report["comparisons"] = {
            "path": "comparisons.csv",
            "rows": len(table),
            "sha256": digest(run_dir / "comparisons.csv"),
            "raw_reference_repeat": 1,
            "rank_scope": "64 fixed train studies; not the full hidden test",
            "accuracy_claim": "none; no labels read and no precision adopted",
        }
        # Verify that every dtype saw the identical cached bytes.
        if (
            digest(run_dir / "preprocessed_images.npy") != cache["images_sha256"]
            or digest(run_dir / "preprocessed_masks.npy") != cache["masks_sha256"]
        ):
            raise ValueError("Preprocessed cache changed during precision comparison")
        report["status"] = "diagnostic_complete" if len(active) == len(modes) else "diagnostic_incomplete"
        report["preprocessing"]["cache_hash_unchanged"] = True
    except Exception as exc:
        report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        save_report(report_path, report)
        raise
    save_report(report_path, report)
    return report
