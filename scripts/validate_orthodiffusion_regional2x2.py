"""Stdlib-only actual regional1024 audit; no image/model reads or fitting."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import struct
import sys
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import validate_orthodiffusion_fullweak as V

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_regional_features as R  # noqa: E402

SLUG = "orthodiff-sag-regional2x2-20261017-v1"
STEM = "orthodiff-regional2x2/ORTHODIFF_REGIONAL2X2_"


def source_evidence(e, pack, launch, kernel_id):
    manifest, payload, saved = e.json(pack / "manifest.json"), e.json(launch / "payload.json"), e.json(launch / "get-saved-raw.json")
    raw = e.raw(pack / "source.ipynb")
    V.require(V.sha(raw) == R.SOURCE_SHA and V.pin(raw) == manifest["source"] and payload["text"].encode() == raw, "Fixed regional source/payload differs")
    book, actual = V.parse(raw), V.parse(saved["blob"]["source"])
    V.require([(c["cell_type"], V.text(c)) for c in book["cells"]] == [(c["cell_type"], V.text(c)) for c in actual["cells"]], "Saved regional cell text differs")
    meta = saved["metadata"]
    V.require(type(kernel_id) is int and kernel_id > 0 and meta["id"] == kernel_id and meta["currentVersionNumber"] == 1
              and meta["ref"] == payload["slug"] == "rsraki/" + SLUG and meta["isPrivate"] is True and meta["enableGpu"] is True
              and meta["enableTpu"] is False and meta["enableInternet"] is False
              and meta["machineShape"] == payload["machine_shape"] == "NvidiaTeslaT4" and meta["dockerImage"] == payload["docker_image"]
              and payload["docker_image"].endswith("sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"), "Regional owned private/offline T4 identity differs")
    kernels = ["rsraki/orthodiff-sag-assets-20261017-v3-enabled/1", "rsraki/orthodiff-sag-fullweak4207-20261017-v2/1"]
    datasets = ["rsraki/rsna-dicom-decoders-py313-v1/1"]
    V.require(payload["kernel_data_sources"] == kernels and payload["dataset_data_sources"] == datasets
              and payload["competition_data_sources"] == meta["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
              and meta["kernelDataSources"] == [s.rsplit("/", 1)[0] for s in kernels]
              and meta["datasetDataSources"] == [s.rsplit("/", 1)[0] for s in datasets], "Four regional Input versions differ")
    tree = ast.parse(next(V.text(c) for c in book["cells"] if c["cell_type"] == "code"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_r2_raw" for t in n.targets))
    stream = zlib.decompressobj()
    carrier = stream.decompress(base64.b85decode(ast.literal_eval(node.value.args[0].args[0])), 2 * 1024**2)
    V.require(stream.eof and not stream.unused_data and not stream.unconsumed_tail and V.pin(carrier) == manifest["bundle_pin"], "Bounded pinned regional carrier differs")
    bundle = V.parse(carrier)
    config = bundle["config"]
    V.require(config == manifest["config"] == e.json(pack / "config.json") and set(bundle["runtimes"]) == {*V.RUNTIME_SHA, "regional"}, "Regional config/namespaces differ")
    for name, source in bundle["runtimes"].items():
        V.require(source.encode() == e.raw(pack / (name + "-runtime.py")) and V.pin(source.encode()) == bundle["runtime_pins"][name] == manifest["runtime_pins"][name]
                  and V.sha(source.encode()) == ({**V.RUNTIME_SHA, "regional": R.RUNTIME_SHA})[name], "Regional runtime pin differs")
    old_pack = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
    old_config, vendor = e.json(old_pack / "config.json"), e.json(old_pack / "vendor.json")
    V.require({k: v for k, v in config.items() if k not in {"regional_parent_input", "regional_source_bindings", "regional_protocol"}} == old_config
              and bundle["vendor"] == vendor == e.json(pack / "vendor.json"), "Regional changed original science/config or vendor")
    for item in vendor.values():
        V.require(V.pin(item["text"].encode()) == {k: item[k] for k in ("bytes", "sha256")}, "Vendored source text differs")
    old, new = bundle["runtimes"]["full"], bundle["runtimes"]["regional"]
    begin = '                        x = ((x - lo)'
    old_block = old[old.index(begin):old.index('                        journal.append(', old.index(begin))]
    new_block = new[new.index(begin):new.index('                        record = {', new.index(begin))]
    V.require(old_block == new_block and V.sha(old_block.encode()) == "3bcdc83b58cd6c60b81867ad536de5934256a356230c1f29b9b1bfa70e284646", "Original FP32/forward/global/RNG science changed")
    return manifest, config, bundle


def terminal(summary, status, config):
    V.require(summary.get("complete") is status.get("complete") is True and "error" not in summary and status.get("error_type") is None
              and summary.get("phase") == status.get("phase") == "regional4207_finished"
              and summary["schema_version"] == "orthodiff_regional2x2_frozen_continuous16_v1"
              and status["schema_version"] == "orthodiff_regional2x2_small_status_v1"
              and summary["requested_UIDs"] == config["uids"] and summary["cohort_UID_hash"] == R.UID_SHA
              and summary["protocol"] == config["regional_protocol"] and summary["frozen_parent_protocol"] == config["preprocess"]
              and summary["source_bindings"] == config["regional_source_bindings"] and summary["owned_temp_removed"] is True,
              "Regional complete terminal/protocol/source qualification differs")
    V.require(summary["coverage"] == status["coverage"] and summary["coverage"]["requested"] == summary["coverage"]["recorded"] == 4207
              and summary["coverage"]["success"] == summary["parent_scientific_parity_successes"] == 3063
              and summary["coverage"]["failure"] == summary["parent_failures_inherited"] == 1144
              and summary["all_requested_MRI_success"] is False
              and summary["current_UID"] == status["current_UID"] == config["uids"][-1], "Regional original3063/1144 terminal partition differs")
    V.require(summary["scope"] == {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                                  "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False}, "Regional scope differs")
    scope = summary["full_scope"]
    V.require(scope["requested_count"] == 4207 and scope["original_supplied_report_groups_included_whole"] is True
              and scope["current_operational_resource_limits"] == config["operational_resource_limits"]
              and scope["frozen_pilot_pins"] == config["frozen_pilot_pins"]
              and scope["runtime_embeds_UIDs_not_groups_folds_Report_or_targets"] is True
              and scope["resumption_or_skip_existing"] is scope["missing_feature_zero_fill"] is False, "Regional fullscope differs")


def stream_records(e, path, parent_path, uids, series, visible_gpus):
    R.pin(path)
    digest, vector_digest, size, count = hashlib.sha256(), hashlib.sha256(), 0, 0
    successes, failures, strata, reasons, prefixes = [], [], Counter(), Counter(), {}
    data_seconds, feature_seconds, pool_seconds = 0., 0., 0.
    with path.open("rb") as new, parent_path.open("rb") as old:
        for count, line in enumerate(new, 1):
            previous_line = old.readline()
            V.require(count <= 4207 and line.endswith(b"\n") and len(line) <= 256 * 1024 and previous_line, "Regional record bounds/count differ")
            record, previous = V.parse(line), V.parse(previous_line)
            uid = uids[count - 1]
            V.require(record.get("StudyInstanceUID") == previous.get("StudyInstanceUID") == uid, "Regional unknown/duplicate/out-of-order UID")
            R.compare_parent(record, previous, previous_line)
            if record["status"] == "success":
                V.validate_success(record, series.get(uid, {}), visible_gpus)
                successes.append(uid)
                strata[record["metadata"]["stratum"]] += 1
                vector_digest.update(struct.pack("<1024f", *R.vector(record)))
                data_seconds += record["data_seconds"]
                feature_seconds += record["feature_seconds"]
                pool_seconds += record["regional_pooling_seconds"]
            else:
                V.require(line == previous_line, "Inherited failure bytes differ")
                V.validate_failure(record, series.get(uid, {}))
                failures.append(uid)
                reasons[record["reason_type"]] += 1
            digest.update(line)
            size += len(line)
            if count % 32 == 0 or count == 4207:
                prefixes[count] = {"sha256": digest.hexdigest(), "success": len(successes), "failure": len(failures)}
        V.require(not old.readline(), "Parent4207 stream has extra record")
    V.require(count == 4207 and len(successes) == 3063 and len(failures) == 1144, "Incomplete regional original3063/1144 stream")
    value = {"bytes": size, "sha256": digest.hexdigest()}
    V.require(value == R.pin(path), "Regional streaming/full pin differs")
    e.pins[str(path.resolve())] = value
    return {"pin": value, "coverage": {"requested": 4207, "recorded": count, "success": len(successes), "failure": len(failures),
                                         "success_strata": dict(strata), "failure_reason_types": dict(reasons)},
            "prefixes": prefixes, "success_UIDs": successes, "failure_UIDs": failures,
            "concatenated_regional_FP32_SHA256": vector_digest.hexdigest(),
            "data_seconds": data_seconds, "feature_seconds": feature_seconds, "regional_pooling_seconds": pool_seconds}


def audit(observation, pack, launch, parent_path, kernel_id, e):
    _, config, bundle = source_evidence(e, pack, launch, kernel_id)
    parent, parent_audit, parent_summary = R.parent_evidence(parent_path)
    e.pins[str(parent_path.resolve())] = R.pin(parent_path)
    e.pins[str(Path(parent["qualification"]["path"]).resolve())] = R.pin(parent["qualification"]["path"])
    for value in parent["files"].values():
        e.pins[str(Path(value["path"]).resolve())] = R.pin(value["path"])
    old_manifest = e.json(ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2/manifest.json")
    cohort, series = V.cohort_evidence(e, ROOT, old_manifest, config)
    V.require(config["regional_parent_input"]["members"] == {Path(v["path"]).name: {k: v[k] for k in ("bytes", "sha256")} for v in parent["files"].values()}
              and config["regional_parent_input"]["qualification_pin"] == R.pin(parent["qualification"]["path"])
              and config["regional_parent_input"]["coverage"] == parent["coverage"], "Regional fixedV1 parent input differs from actual audit")
    V.require(e.json(observation / "status.json").get("status") == "COMPLETE", "Official regional COMPLETE required")
    receipts = e.json(observation / "receipts.json")
    expected = {STEM + suffix for suffix in ("SUMMARY.json", "STATUS.json", "RECORDS.jsonl")}
    downloads = [row for row in receipts if "file" in row]
    V.require(len(downloads) == 3 and {row["file"] for row in downloads} == expected and not any("error_type" in row for row in receipts), "Exactly3 successful regional downloads required")
    files = observation / "files"
    paths = {name: files / (STEM + suffix) for name, suffix in (("summary", "SUMMARY.json"), ("status", "STATUS.json"), ("records", "RECORDS.jsonl"))}
    for row in downloads:
        value = R.pin(files / row["file"])
        V.require(row["http_status"] == 200 and {k: row[k] for k in ("bytes", "sha256")} == value, "Regional HTTP/output bytes differ")
        e.pins[str((files / row["file"]).resolve())] = value
    summary, status = e.json(paths["summary"]), e.json(paths["status"])
    terminal(summary, status, config)
    parent_input = summary["parent_input"]
    V.require(parent_input["members"] == config["regional_parent_input"]["members"]
              and parent_input["independent_actual_qualification_pin"] == config["regional_parent_input"]["qualification_pin"]
              and parent_input["fixed_Input"] == config["regional_parent_input"]["Input"]
              and parent_input["requested"] == 4207 and parent_input["recomputed_successes_required"] == 3063
              and parent_input["featureless_failures_inherited_without_image_access"] == 1144, "Actual regional parent Input receipt differs")
    env = summary["environment"]
    V.require(all(env[k] == parent_summary["environment"][k] for k in ("torch", "numpy", "pydicom", "cv2", "cuda", "selected_GPU"))
              and env["python"].startswith("3.13.") and type(env["visible_GPUs"]) is int and env["visible_GPUs"] >= 1, "Regional numerical environment differs")
    stream = stream_records(e, paths["records"], Path(parent["files"]["records"]["path"]), config["uids"], series, env["visible_GPUs"])
    V.require(summary["coverage"] == parent["coverage"] == stream["coverage"]
              and summary["journal"] == {"filename": paths["records"].name, **stream["pin"], "final_full_file_rehash_matches_incremental": True}
              and status["journal"] == {"filename": paths["records"].name, **stream["pin"]}
              and summary["forward_counts"] == {"denoiser": 3063, "decoder_blocks": 0, "final_out": 0}, "Regional journal/terminal coverage/forwards differ")
    V.require(summary["model_load_calls"] == 1 and summary["checkpoint"] == parent_summary["checkpoint"]
              and summary["checkpoint"]["keys"] == 257 and set(summary["checkpoint"]["schedule_buffers_exact"]) == V.BUFFER_NAMES
              and all(summary["checkpoint"]["schedule_buffers_exact"].values())
              and summary["verified_generic_assets"] == parent_summary["verified_generic_assets"], "Regional strictEMA257/13 or nine assets differ")
    V.require(set(summary["vendor_imports"]) == {"__init__", "unet", "modules", "fp16_util", "features"}, "Regional imported closure differs")
    for name, row in summary["vendor_imports"].items():
        V.require({k: row[k] for k in ("bytes", "sha256")} == {k: bundle["vendor"][name + ".py"][k] for k in ("bytes", "sha256")}
                  and row["path"].endswith("/od_qual_pinned/" + name + ".py"), "Regional vendor origin/source differs")
    decoder = summary["decoder"]
    V.require(all(decoder[k] == parent_summary["decoder"][k] for k in ("verified_members", "fixtures", "core_versions_unchanged", "packages"))
              and decoder["install_returncode"] == 0 and decoder["no_index"] is decoder["no_deps"] is True, "Regional decoder packages/fourphantoms differ")
    resources, limits = summary["resources"], config["operational_resource_limits"]
    V.require(0 < resources["peak_cuda_allocated_bytes"] <= resources["peak_cuda_reserved_bytes"] <= limits["CUDA_reserved_bytes"]
              and 0 < resources["peak_process_RSS_bytes"] <= limits["process_RSS_bytes"]
              and 0 < resources["total_seconds"] <= limits["total_seconds"]
              and stream["data_seconds"] + stream["feature_seconds"] + stream["regional_pooling_seconds"] <= resources["total_seconds"]
              and 0 < summary["strict_load_seconds"] <= resources["total_seconds"], "Regional posthoc8h/14GiB/8GiB resources differ")
    listing = e.json(observation / "output-list.json")
    V.require({row["fileName"] for row in listing["files"]} == expected, "Regional custom output set differs")
    stdout = "".join(row["data"] for row in V.parse(listing["log"]) if row.get("stream_name") == "stdout")
    markers, progress = [], []
    for line in stdout.splitlines():
        if line.startswith("ORTHODIFF_REGIONAL2X2 "):
            markers.append(V.parse(line.split(" ", 1)[1]))
        elif line.startswith("ORTHODIFF_REGIONAL2X2_PROGRESS "):
            progress.append(V.parse(line.split(" ", 1)[1]))
    V.require(len(markers) == 1 and markers[0]["complete"] is True and markers[0]["pin"] == R.pin(paths["summary"])
              and any(row["phase"] == "single_frozen_backbone_loaded" for row in progress)
              and any(row["phase"] == "regional4207_finished" and row["complete"] is True and row["recorded"] == 4207 for row in progress), "Regional typed stdout terminal differs")
    for row in progress:
        n = row["recorded"]
        V.require(row["requested"] == 4207 and n == row["success"] + row["failure"]
                  and (stream["prefixes"].get(n) == {"sha256": row["journal_prefix_SHA256"], "success": row["success"], "failure": row["failure"]}
                       if n else row["journal_prefix_SHA256"] == V.sha(b"") and row["success"] == row["failure"] == 0), "Regional progress prefix differs")
    for row in receipts:
        if "label" in row:
            V.require(row["http_status"] == 200 and {k: row[k] for k in ("bytes", "sha256")} == V.pin(e.raw(observation / (row["label"] + ".json"))), "Regional raw API pin differs")
    fold_by_uid = {row["StudyInstanceUID"]: row["fold"] for row in cohort["requested"]}
    return {"schema_version": "orthodiff_regional2x2_actual_qualification_v1", "passed": True, "head_features_qualified": True,
            "at_UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": R.SOURCE_SHA, "UID_order_sha256": R.UID_SHA,
            "kernel_id": kernel_id, "version": 1, "requested_count": 4207, "feature_dimension": 1024,
            "records_pin": stream["pin"], "summary_pin": R.pin(paths["summary"]), "status_pin": R.pin(paths["status"]),
            "coverage": stream["coverage"], "parent_bundle_pin": R.pin(parent_path), "parent_qualification_pin": R.pin(parent["qualification"]["path"]),
            "parent_comparison": {"passed": True, "successful_scientific_parity": 3063, "failure_records_byte_exact": 1144,
                                  "SID_SOP16_geometry_normalized_noisy_mid2_global256_bits_same": True,
                                  "same_UID_group_fold_source_strata_partition": True},
            "FP32_feature_values_checked": 3063 * 1024, "concatenated_regional_FP32_SHA256": stream["concatenated_regional_FP32_SHA256"],
            "success_fold_counts": dict(Counter(fold_by_uid[uid] for uid in stream["success_UIDs"])),
            "failure_fold_counts": dict(Counter(fold_by_uid[uid] for uid in stream["failure_UIDs"])),
            "FD_availability_mixed": False, "resources": resources, "environment": env, "input_pins": e.pins,
            "limits": ["Raw pixels, raw mid2 and raw IPP absent; actual regional means cannot be independently replayed from full spatial feature maps.",
                       "Source-fixed quadrant definition and all1024 FP32 serialization values/SHA checked; separate artificial GPU oracle is not actual-MRI replay.",
                       "Cooperative/posthoc resource budgets do not provide a hard native decoder deadline."],
            "scope": "Regional1024 engineering/parent parity qualification only, no new head fit, clinical or Public score/adoption claim",
            "local_activity": {"labels_Report_MRI_model_scientific_import_GPU_network_fit_Save_Submit": 0},
            "parent_prior32_comparison_passed": parent_audit["prior32_comparison"]["passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "pack", "launch", "parent-bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel-id", type=int, required=True)
    args = parser.parse_args()
    V.require(not args.output.exists(), "New regional qualification output required")
    e = V.Evidence()
    for path in (Path(__file__), Path(V.__file__), Path(R.__file__)):
        e.raw(path)
    try:
        result = audit(args.observation, args.pack, args.launch, args.parent_bundle, args.kernel_id, e)
    except Exception as error:
        result = {"schema_version": "orthodiff_regional2x2_actual_qualification_v1", "passed": False, "head_features_qualified": False,
                  "error_type": type(error).__name__, "error": str(error), "input_pins": e.pins}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "head_features_qualified", "coverage", "error")}))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
