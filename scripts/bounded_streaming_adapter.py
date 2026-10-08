"""Bind the memory reader temporarily around the immutable streaming runtime."""

from __future__ import annotations

import json
import traceback
from pathlib import Path


def run_memory_profile(
    runtime,
    reader_factory,
    root,
    checkpoint_path,
    provenance,
    encoder_repo,
    encoder_weights,
    output,
    *,
    summary_path,
    reader_sha256,
    baseline_runtime_sha256,
    resource_fn=None,
    **runtime_arguments,
):
    """Profile only; no submitted predictions or modification of core files."""
    from rsna_knee import feature_imaging

    output, summary_path = Path(output), Path(summary_path)
    workspace = output.parent / "f002-native-workspace"
    events_path = output / "memory-reader-events.jsonl"
    original_reader = feature_imaging.read_feature_windows
    previous_output = output.exists()
    previous_workspace = workspace.exists()
    owned_workspace = False
    current = {}
    policy = {
        "change": "native intermediate pixel storage only; strict geometry/series selection and global p1/p99 arithmetic unchanged",
        "reader_sha256": reader_sha256,
        "baseline_runtime_sha256": baseline_runtime_sha256,
        "runtime_binding": "feature_imaging.read_feature_windows temporarily replaced; core source files unchanged",
        "storage": "two temporary float32 memmaps per selected series; percentile scratch partitioned in place; normalization by native slice",
        "copy_chunk_bytes": 8 * 1024**2,
        "disk_reserve_bytes": 256 * 1024**2,
        "observed_series_allocations": 0,
        "maximum_observed_native_file_bytes": 0,
        "maximum_observed_copy_chunk_bytes": 0,
        "memory_events_file": str(events_path.name),
        "reader_binding_restored": False,
        "temporary_files_remaining": None,
        "limitation": "No constant peak RSS guarantee: scratch partition uses file-backed pages. Temporary disk and I/O increase; hidden root cause and completion remain unverified.",
    }

    def event(phase, detail):
        resources = resource_fn(runtime_arguments.get("device", "cuda")) if resource_fn is not None else {}
        row = {"phase": phase, **current, **detail, "resources": resources}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def observer(phase, detail):
        if phase == "allocation_plan":
            policy["observed_series_allocations"] += 1
            policy["maximum_observed_native_file_bytes"] = max(
                policy["maximum_observed_native_file_bytes"], detail["native_file_bytes"]
            )
            event(phase, detail)
        elif phase == "quantile_copy":
            current["quantile_copy_chunks"] += 1
            policy["maximum_observed_copy_chunk_bytes"] = max(
                policy["maximum_observed_copy_chunk_bytes"], detail["copied_bytes"]
            )
            if current["quantile_copy_chunks"] % 16 == 0:
                event(phase, detail)
        elif phase == "normalize_slice":
            current["normalized_slices"] += 1
            if current["normalized_slices"] % 16 == 0:
                event(phase, detail)
        else:
            event(phase, detail)

    result = None
    failure = None
    try:
        if runtime_arguments.get("mode") != "profile_train":
            raise ValueError("Memory profile is not a submission notebook; mode must be profile_train")
        if previous_output or previous_workspace:
            raise ValueError("Use a fresh memory-profile kernel/output workspace; preserve previous diagnostics")
        workspace.mkdir()
        owned_workspace = True
        bounded_reader = reader_factory(workspace, observer=observer)

        def observed_reader(series_dir, settings):
            current.clear()
            current.update(
                study_uid=Path(series_dir).parent.name,
                series_uid=Path(series_dir).name,
                quantile_copy_chunks=0,
                normalized_slices=0,
            )
            try:
                value = bounded_reader(series_dir, settings)
            except Exception as error:
                event("bounded_series_failed", {"error_type": type(error).__name__, "error": str(error)})
                raise
            event("bounded_series_complete", {"windows": len(value[0])})
            return value

        feature_imaging.read_feature_windows = observed_reader
        result = runtime(
            root,
            checkpoint_path,
            provenance,
            encoder_repo,
            encoder_weights,
            output,
            **runtime_arguments,
        )
        return result
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        feature_imaging.read_feature_windows = original_reader
        policy["reader_binding_restored"] = feature_imaging.read_feature_windows is original_reader
        if owned_workspace:
            policy["temporary_files_remaining"] = sum(1 for _ in workspace.rglob("*"))
            if not any(workspace.iterdir()):
                workspace.rmdir()
        else:
            policy["temporary_workspace_created"] = False
        nested = output / "F002_STREAMING_SUMMARY.json"
        if result is None:
            result = json.loads(nested.read_text(encoding="utf-8")) if nested.is_file() else {}
        if failure is not None:
            result.update(failure)
        result["memory_reader"] = policy
        # Do not replace previous nested diagnostics after the fresh-run guard.
        if not previous_output:
            output.mkdir(parents=True, exist_ok=True)
            nested.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
