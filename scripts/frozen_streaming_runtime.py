"""Same frozen preprocessing/head, one study at a time, with local failure diagnostics."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_SYNTAXES = (
    "1.2.840.10008.1.2.4.57",
    "1.2.840.10008.1.2.4.70",
    "1.2.840.10008.1.2.4.90",
    "1.2.840.10008.1.2.4.91",
)


def decoder_inventory():
    from pydicom.pixels import get_decoder

    result = {}
    for syntax in REQUIRED_SYNTAXES:
        decoder = get_decoder(syntax)
        result[syntax] = {
            "available": decoder.is_available,
            "plugins": list(decoder.available_plugins),
            "missing_dependencies": decoder.missing_dependencies,
        }
    return result


def install_offline_decoders(wheels, manifest_sha256):
    """Install pinned Linux wheels only, without network or changing NumPy/torch."""
    directory = Path(wheels)
    manifest_path = directory.parent / "decoder-manifest.json"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(manifest_path) != manifest_sha256:
        raise ValueError("Decoder manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sys.platform != "linux" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Decoder wheels target Kaggle Linux Python3.13; rebuild for another runtime")
    for wheel in manifest["wheels"]:
        path = directory / wheel["filename"]
        if digest(path) != wheel["sha256"]:
            raise ValueError("Decoder wheel SHA256 mismatch")
    before = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pydicom")}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(directory),
            *[f"{item['name']}=={item['version']}" for item in manifest["wheels"]],
        ],
        check=True,
    )
    after = {name: importlib.metadata.version(name) for name in before}
    if after != before:
        raise ValueError("Decoder install changed the model/preprocessing environment")
    return {"manifest_sha256": manifest_sha256, "packages": manifest["wheels"], "unchanged": before}


def predict_arrays(model, arrays, device):
    import torch

    with torch.inference_mode():
        logits = model(**{name: torch.from_numpy(value).unsqueeze(0).to(device) for name, value in arrays.items()})
        if logits.shape != (1, 12) or not torch.isfinite(logits).all():
            raise ValueError("Nonfinite or malformed streaming head output")
        return logits.float().cpu().sigmoid()[0].tolist()


def verify_compressed_phantoms(directory, manifest_sha256):
    """Exercise installed decoders on our synthetic pixels, never on patient data."""
    import numpy as np
    import pydicom

    directory = Path(directory)
    manifest_path = directory / "phantom-manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError("Synthetic decoder phantom manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_path = directory / manifest["expected_filename"]
    if hashlib.sha256(expected_path.read_bytes()).hexdigest() != manifest["expected_sha256"]:
        raise ValueError("Synthetic expected pixels SHA256 mismatch")
    expected = np.load(expected_path, allow_pickle=False)
    results = []
    for item in manifest["files"]:
        path = directory / item["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Synthetic compressed fixture SHA256 mismatch")
        dataset = pydicom.dcmread(path)
        plugin_pixels = pydicom.pixels.pixel_array(dataset, decoding_plugin="pylibjpeg")
        default_pixels = dataset.pixel_array
        if not np.array_equal(plugin_pixels, expected) or not np.array_equal(default_pixels, expected):
            raise ValueError("Compressed decoder changed synthetic pixel values")
        results.append({"transfer_syntax": item["syntax"], "default_and_pylibjpeg_exact_pixels": True})
    return results


def resource_snapshot(device):
    import torch

    result = {"free_disk_bytes": shutil.disk_usage(Path.cwd()).free}
    try:
        import resource

        # Kaggle Linux reports KiB; this process high-water mark includes host arrays.
        result["cpu_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        result["cpu_peak_rss_bytes"] = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        result.update(
            gpu_allocated_bytes=torch.cuda.memory_allocated(device),
            gpu_reserved_bytes=torch.cuda.memory_reserved(device),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return result


def run_streaming(
    root,
    checkpoint_path,
    provenance,
    encoder_repo,
    encoder_weights,
    output,
    *,
    mode="test",
    weak_manifest=None,
    profile_limit=256,
    selection_seed=20261007,
    expected_checkpoint_sha256,
    expected_sources,
    expected_weak_sha256=None,
    require_decoders=True,
    device="cuda",
):
    import numpy as np
    import torch

    from rsna_knee import feature_imaging
    from rsna_knee.contracts import (
        ID,
        SERIES_COLUMNS,
        SERIES_ID,
        SUBMISSION_COLUMNS,
        TARGETS,
        dump_json,
        index_studies,
        read_csv,
        sha256,
        source_hashes,
        uid,
        validate_submission,
        write_csv,
    )
    from rsna_knee.feature_contracts import feature_contract
    from rsna_knee.feature_model import FrozenFeatureHead
    from rsna_knee.feature_runtime import load_head_checkpoint

    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a fresh output directory/kernel; preserve the previous diagnostics")
    output.mkdir(parents=True)
    summary_path = output / "F002_STREAMING_SUMMARY.json"
    summary = {
        "status": "started",
        "phase": "preflight",
        "mode": mode,
        "public_lb": None,
        "training_performed": False,
        "processed": 0,
        "selection_seed": selection_seed,
        "fallback_predictions": 0,
        "skipped_series": 0,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pydicom", "Pillow")},
        "note": "Saved-run diagnostics only; private scoring logs are not exposed by Kaggle",
    }
    began = time.perf_counter()
    journal = output / "events.jsonl"
    original_read = feature_imaging.read_feature_windows
    original_encode = feature_imaging.encode_windows
    current = {}
    syntax_counts = Counter()

    def event(phase, **details):
        summary.update(
            phase=phase,
            current_context={**current, **details},
            elapsed_seconds=time.perf_counter() - began,
            resources=resource_snapshot(device),
        )
        dump_json(summary_path, summary)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": phase, **current, **details}, ensure_ascii=False) + "\n")

    def observed_read(series_dir, settings):
        import pydicom

        current["series_uid"] = Path(series_dir).name
        current.pop("header", None)
        event("read_series_header")
        paths = sorted(Path(series_dir).glob("*.dcm"))
        if paths:
            header = pydicom.dcmread(paths[0], stop_before_pixels=True)
            syntax = str(getattr(header.file_meta, "TransferSyntaxUID", ""))
            syntax_counts[syntax] += 1
            current["header"] = {
                "transfer_syntax": syntax,
                "slices": len(paths),
                "rows": int(getattr(header, "Rows", 0)),
                "columns": int(getattr(header, "Columns", 0)),
                "number_of_frames": int(getattr(header, "NumberOfFrames", 1)),
                "pixel_spacing": [float(v) for v in getattr(header, "PixelSpacing", [])],
                "orientation_present": hasattr(header, "ImageOrientationPatient"),
                "position_present": hasattr(header, "ImagePositionPatient"),
            }
        event("decode_and_preprocess_series")
        result = original_read(series_dir, settings)
        event("series_preprocessed", windows=len(result[0]))
        return result

    def observed_encode(model, windows, config, device):
        event("encode_series", input_shape=list(windows.shape))
        result = original_encode(model, windows, config, device)
        event("series_encoded", centers=len(result))
        return result

    try:
        event("preflight")
        if mode not in {"test", "profile_train"}:
            raise ValueError("Unknown streaming mode")
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        if source_hashes() != expected_sources or sha256(checkpoint_path) != expected_checkpoint_sha256:
            raise ValueError("Code/checkpoint differs from the audited f002 baseline")
        checkpoint = load_head_checkpoint(checkpoint_path)
        config = checkpoint["config"]
        if (checkpoint["fold"], checkpoint["epoch"]) != (0, 11):
            raise ValueError("Expected fixed f002 fold0 BCE-selected epoch11")
        if feature_contract(config, provenance) != checkpoint["feature_contract"]:
            raise ValueError("Encoder/preprocessing contract changed")
        summary.update(
            checkpoint_sha256=expected_checkpoint_sha256,
            feature_fingerprint=checkpoint["feature_fingerprint"],
            decoders=decoder_inventory(),
        )
        event("decoder_preflight")
        if require_decoders and any(not entry["available"] for entry in summary["decoders"].values()):
            raise RuntimeError("JPEG Lossless/JPEG2000 decoder unavailable; attach the pinned offline decoder Input")
        split = "test" if mode == "test" else "train"
        _, studies = read_csv(root / f"{split}.csv", (ID,))
        all_ids = index_studies(studies)
        if not all_ids:
            raise ValueError("No studies")
        if mode == "profile_train":
            if weak_manifest is None or type(profile_limit) is not int or profile_limit < 1:
                raise ValueError("Profile requires a fixed weak manifest and positive limit")
            if expected_weak_sha256 is None or sha256(weak_manifest) != expected_weak_sha256:
                raise ValueError("Profile weak manifest differs from the fixed gold-excluded training list")
            _, weak_rows = read_csv(weak_manifest, (ID,))
            eligible = index_studies(weak_rows)
            if not eligible or not eligible.keys() <= all_ids.keys():
                raise ValueError("Profile manifest does not match train.csv")
            keys = sorted(eligible, key=lambda key: hashlib.sha256(f"{selection_seed}:{key}".encode()).hexdigest())[
                :profile_limit
            ]
            studies = [{ID: key} for key in keys]
            truth_path = output / "profile-studies.csv"
            write_csv(truth_path, (ID,), studies)
        else:
            studies = [{ID: row[ID]} for row in studies]
            truth_path = root / "test.csv"
        _, series = read_csv(root / f"{split}_series.csv", SERIES_COLUMNS)
        by_study, pairs = defaultdict(list), set()
        for row in series:
            pair = (uid(row[ID]), uid(row[SERIES_ID]))
            if pair in pairs or pair[0] not in all_ids:
                raise ValueError("Orphan/duplicate series")
            pairs.add(pair)
            by_study[pair[0]].append(row)
        summary.update(
            studies=len(studies),
            study_csv_sha256=sha256(truth_path),
            series_csv_sha256=sha256(root / f"{split}_series.csv"),
            report_required=False,
            intermediate_feature_files=0,
            gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "synthetic CPU test",
        )
        event("load_encoder")
        encoder = feature_imaging.load_frozen_encoder(encoder_repo, encoder_weights, provenance, device)
        head = FrozenFeatureHead(config).to(device)
        head.load_state_dict(checkpoint["model"], strict=True)
        head.eval()
        feature_imaging.read_feature_windows = observed_read
        feature_imaging.encode_windows = observed_encode
        predictions = []
        for number, row in enumerate(studies, 1):
            current.clear()
            current["study_uid"] = row[ID]
            event("extract_study", study_number=number)
            arrays, _ = feature_imaging.extract_study(root, split, row[ID], by_study[row[ID]], config, encoder, device)
            for value in arrays.values():
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite extracted feature arrays")
            if not arrays["mask"].any() or (arrays["positions"] < 0).any() or (arrays["positions"] > 1).any():
                raise ValueError("Empty mask or invalid normalized physical position")
            event("predict_head")
            scores = predict_arrays(head, arrays, device)
            predictions.append(
                {ID: row[ID], **{target: str(value) for target, value in zip(TARGETS, scores, strict=True)}}
            )
            del arrays
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            summary.update(processed=number, transfer_syntax_counts=dict(syntax_counts))
            event("study_complete")
            if number == 1 or number % 16 == 0 or number == len(studies):
                print(f"Streaming {number}/{len(studies)}; elapsed={time.perf_counter() - began:.1f}s", flush=True)
        event("validate_csv")
        # Publish final submission only after complete coverage and finite probabilities.
        temporary = output / "predictions.partial.csv"
        write_csv(temporary, SUBMISSION_COLUMNS, predictions)
        sample_path = root / "sample_submission.csv"
        receipt = validate_submission(
            temporary, truth_path, sample_path if mode == "test" and sample_path.is_file() else None
        )
        destination = output.parent / "submission.csv" if mode == "test" else output / "profile_predictions.csv"
        if destination.exists():
            raise ValueError("Refuse to replace a previous submission CSV")
        temporary.replace(destination)
        summary.update(
            status="passed" if mode == "test" else "profile_complete_not_for_submission",
            phase="complete",
            receipt=receipt,
            submission_sha256=sha256(destination),
            csv_name=destination.name,
            elapsed_seconds=time.perf_counter() - began,
            resources=resource_snapshot(device),
            per_study_seconds=(time.perf_counter() - began) / len(studies),
        )
        dump_json(summary_path, summary)
        return summary
    except Exception as error:
        summary.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter() - began,
        )
        dump_json(summary_path, summary)
        print(summary["traceback"], flush=True)
        raise
    finally:
        feature_imaging.read_feature_windows = original_read
        feature_imaging.encode_windows = original_encode
        gc.collect()
