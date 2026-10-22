"""Stdlib-only actual FD audit; no PD claim, images, weights, GPU or fitting."""
from __future__ import annotations

import argparse
import ast
import base64
import builtins
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
from rsna_knee import orthodiff_fluid_fallback as F  # noqa: E402
from rsna_knee import orthodiff_fluid_features as C  # noqa: E402

SLUG = "orthodiff-sag-fluid1080-20261017-v3"
STEM = "orthodiff-fluid-fallback1080/ORTHODIFF_FLUID_FALLBACK_"
FLUID_RUNTIME_SHA = "b2e6dd861ee0fe4408e413241e950329d9b5885616b8dc8709396a0606d42833"


def source_evidence(e, pack, launch):
    m, payload, saved = e.json(pack / "manifest.json"), e.json(launch / "payload.json"), e.json(launch / "get-saved-raw.json")
    raw = e.raw(pack / "source.ipynb")
    V.require(V.sha(raw) == C.CHILD_SOURCE_SHA and V.pin(raw) == m["source"] and payload["text"].encode() == raw, "Fixed child source/payload mismatch")
    book, actual = V.parse(raw), V.parse(saved["blob"]["source"])
    V.require([(cell["cell_type"], V.text(cell)) for cell in book["cells"]] == [(cell["cell_type"], V.text(cell)) for cell in actual["cells"]], "Saved child cell text differs")
    meta = saved["metadata"]
    V.require(meta["id"] == 139158713 and meta["currentVersionNumber"] == 1 and meta["ref"] == "rsraki/" + SLUG
              and meta["isPrivate"] is True and meta["enableGpu"] is True and meta["enableTpu"] is False and meta["enableInternet"] is False
              and meta["machineShape"] == payload["machine_shape"] == "NvidiaTeslaT4" and meta["dockerImage"] == payload["docker_image"]
              and payload["docker_image"].endswith("sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"), "Owned/offline T4 identity differs")
    V.require(payload["kernel_data_sources"] == ["rsraki/orthodiff-sag-assets-20261017-v3-enabled/1"]
              and payload["dataset_data_sources"] == ["rsraki/rsna-dicom-decoders-py313-v1/1"]
              and payload["competition_data_sources"] == meta["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
              and meta["kernelDataSources"] == [payload["kernel_data_sources"][0].rsplit("/", 1)[0]]
              and meta["datasetDataSources"] == [payload["dataset_data_sources"][0].rsplit("/", 1)[0]], "Three immutable child Inputs differ")
    tree = ast.parse(next(V.text(cell) for cell in book["cells"] if cell["cell_type"] == "code"))
    node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_ff_raw" for t in node.targets))
    stream = zlib.decompressobj()
    raw_bundle = stream.decompress(base64.b85decode(ast.literal_eval(node.value.args[0].args[0])), 2 * 1024**2)
    V.require(stream.eof and not stream.unused_data and not stream.unconsumed_tail and V.pin(raw_bundle) == m["bundle_pin"], "Exact bounded child carrier required")
    bundle = V.parse(raw_bundle)
    config = bundle["config"]
    V.require(config == m["config"] == e.json(pack / "config.json"), "Child config mismatch")
    expected_shas = {**V.RUNTIME_SHA, "fluid": FLUID_RUNTIME_SHA}
    V.require(set(bundle["runtimes"]) == set(expected_shas), "Four immutable namespaces required")
    for name, text in bundle["runtimes"].items():
        V.require(V.pin(text.encode()) == bundle["runtime_pins"][name] == m["runtime_pins"][name]
                  and V.sha(text.encode()) == expected_shas[name] and text.encode() == e.raw(pack / (name + "-runtime.py")), "Scientific/runtime pin mismatch")
    old_pack = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
    old = e.json(old_pack / "config.json")
    V.require(bundle["vendor"] == e.json(old_pack / "vendor.json") and config["preprocess"] == old["preprocess"]
              and config["assets"] == old["assets"] and config["decoder"] == old["decoder"]
              and config["operational_resource_limits"] == old["operational_resource_limits"], "Frozen numerical/decode/resource contract changed")
    # Restore the two removed PD-only statements; every other reader AST node is fixed.
    def function(text, name):
        return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    original = function(bundle["runtimes"]["pilot"], "read_clip")
    reader = function(bundle["runtimes"]["fluid"], "read_fluid_clip")
    reader.name = "read_clip"
    loop = next(n for n in reader.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "header")
    old_loop = next(n for n in original.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "header")
    missing = [n for n in old_loop.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "classification" for t in n.targets))
               or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and any(isinstance(t, ast.Constant) and t.value == "mixed_or_unknown_PD_header" for t in n.value.args))]
    insertion = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                     and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "geometry")
    V.require(len(missing) == 2, "Exactly two original PD statements required")
    loop.body[insertion:insertion] = missing
    V.require(ast.dump(reader, include_attributes=False) == ast.dump(original, include_attributes=False), "Reader changed beyond the new non-PD route")
    return m, config, bundle


def parent_evidence(e, parent_path, config):
    parent, audit = C.parent_bundle(parent_path, C.PARENT_BUNDLE_SHA)
    e.pins[str(parent_path.resolve())] = C.pin(parent_path)
    e.pins[str(Path(parent["qualification"]["path"]).resolve())] = C.pin(parent["qualification"]["path"])
    for item in parent["files"].values():
        e.pins[str(Path(item["path"]).resolve())] = C.pin(item["path"])
    subset_path = ROOT / "artifacts/research/20261017-orthodiff-fluid-subset-actual-v1/subset.json"
    subset = e.json(subset_path)
    V.require(V.pin(e.raw(subset_path)) == config["fluid_source_bindings"]["subset_pin"]
              and V.sha(e.raw(subset_path)) == C.SUBSET_SHA and subset["parent_qualification_sha256"] == C.PARENT_AUDIT_SHA, "Fixed actual parent subset binding differs")
    requests = [{key: entry[key] for key in ("StudyInstanceUID", "parent_record_canonical_sha256", "route", "confirmed_PD", "status", "selected_series")}
                for entry in subset["entries"]]
    uids = [row["StudyInstanceUID"] for row in requests]
    V.require(requests == config["fluid_requests"] and uids == config["uids"] and len(uids) == len(set(uids)) == 1080
              and C.uid_hash(uids) == config["UID_order_sha256"] == C.CHILD_UID_SHA
              and sum(row["selected_series"] is not None for row in requests) == 1077, "Fixed1080/1077/3 request subset mismatch")
    cohort_path = ROOT / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1/cohort.json"
    cohort = e.json(cohort_path)
    V.require(V.sha(e.raw(cohort_path)) == "86d976f55bb67bd3a738679625c9281aead06b0c7f01d7c894ffe870ead65246"
              and cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0, "Original cohort/Gold UID/group metadata binding differs")
    structure = {row["StudyInstanceUID"]: row for row in cohort["requested"]}
    for entry in subset["entries"]:
        V.require(all(entry[key] == structure[entry["StudyInstanceUID"]][key] for key in ("group_id", "fold", "source")), "Child original group/fold/source mapping differs")
    parent_summary = e.json(Path(parent["files"]["summary"]["path"]))
    wanted, successes, count = {row["StudyInstanceUID"]: row for row in requests}, set(), 0
    with Path(parent["files"]["records"]["path"]).open("rb") as handle:
        for count, line in enumerate(handle, 1):
            record = V.parse(line)
            uid = record["StudyInstanceUID"]
            V.require(count <= 4207 and uid == parent_summary["requested_UIDs"][count - 1], "Parent full record order changed")
            eligible, _ = F.unknown_pd_family(record)
            V.require(eligible == (uid in wanted), "Child must equal the complete unknown-PD naming family")
            if record["status"] == "success":
                C.feature_values(record)
                successes.add(uid)
            if eligible:
                canonical = (json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
                V.require(V.sha(canonical) == wanted[uid]["parent_record_canonical_sha256"], "Canonical parent failure hash changed")
    V.require(count == 4207 and len(successes) == 3063 and successes.isdisjoint(wanted), "Parent success overwrite/unknown UID")
    V.require(config["fluid_source_bindings"]["parent_journal_pin"] == audit["records_pin"]
              and config["fluid_source_bindings"]["parent_qualification_pin"] == C.pin(parent["qualification"]["path"]), "Source bindings differ from actual parent")
    return parent, audit, subset, {row["StudyInstanceUID"]: row for row in requests}


def selected_contract(selected, request):
    expected = request["selected_series"]
    V.require(expected is not None and selected["SeriesInstanceUID"] == expected["SeriesInstanceUID"]
              and selected["route"] == "fluid_sensitive_fallback" and selected["confirmed_PD"] is False
              and selected["retry_another_series_after_chosen_failure"] is False
              and all(selected[key] == value for key, value in expected["metadata"].items())
              and selected["stratum"] == ("fluid_sensitive_FS" if expected["Fat_Suppression"] else "fluid_sensitive_nonFS"), "Frozen selected fluid series/route/flags differ")


def validate_record(record, request, visible_gpus):
    V.require(record["StudyInstanceUID"] == request["StudyInstanceUID"], "Child record UID differs")
    if record.get("status") == "data_failure":
        error = getattr(builtins, record.get("reason_type", ""), None)
        V.require(record.get("reason_type") in {"PilotDataError", "InvalidDicomError"} or (isinstance(error, type) and issubclass(error, (OSError, ValueError, RuntimeError))), "Unexpected recoverable error type")
        V.require(record.get("route") == "fluid_sensitive_fallback" and record.get("confirmed_PD") is False and record.get("feature_forward_calls") == 0
                  and record.get("fallback_features") is False and record.get("reason") and record["data_seconds"] >= 0
                  and record["parent_record_canonical_sha256"] == request["parent_record_canonical_sha256"]
                  and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input", "noisy_input", "noise_seed")), "Child failure fabricated features or lost parent identity")
        if request["selected_series"] is None:
            V.require(record["selected_series"] is None and record["reason"] == "fluid_sensitive_metadata_unavailable", "Unavailable metadata must remain featureless")
        else:
            V.require(record["selected_series"] is not None, "A fixed candidate failure lost its chosen series identity")
            selected_contract(record["selected_series"], request)
        return
    V.require(record.get("status") == "success" and request["selected_series"] is not None, "Unexpected fluid status or unavailable-success")
    detail = record["metadata"]
    selected_contract(detail, request)
    V.geometry(detail)  # Geometry only: deliberately no parent PD-name eligibility check.
    for name, shape in (("normalized_input", [1, 1, 16, 256, 256]), ("noisy_input", [1, 1, 16, 256, 256]),
                        ("mid_2", [1, 256, 1, 16, 16]), ("pooled", [1, 256])):
        V.tensor(record[name], shape)
    C.feature_values(record)
    bounds = record["normalization_min_max_float32"]
    V.require(len(bounds) == 2 and V.finite(bounds) and bounds[1] > bounds[0]
              and all(struct.unpack("<f", struct.pack("<f", value))[0] == value for value in bounds), "FP32 clip minmax metadata differs")
    seed = json.dumps(["orthodiff-artificial-noise-v1", record["StudyInstanceUID"], detail["SeriesInstanceUID"], [detail["clip_start"], detail["clip_end"]], 20261017], ensure_ascii=True, separators=(",", ":")).encode()
    V.require(record["noise_seed"] == int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % (2**63 - 1)
              and record["forward_counts"] == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0} and record["all_RNG_restored"] is True
              and len(record["all_CUDA_RNG_before_after_SHA"]) == visible_gpus, "Scoped seed/RNG/one-forward mismatch")
    for before, after in [record["CPU_RNG_before_after_SHA"], *record["all_CUDA_RNG_before_after_SHA"]]:
        V.require(V.digest(before) and before == after, "CPU/all-visible-CUDA RNG changed")
    V.require(record["data_seconds"] >= 0 and record["feature_seconds"] > 0, "Invalid feature phase latencies")


def audit(observation, pack, launch, parent_path, e):
    manifest, config, bundle = source_evidence(e, pack, launch)
    parent, parent_audit, subset, requests = parent_evidence(e, parent_path, config)
    V.require(e.json(observation / "status.json").get("status") == "COMPLETE", "Official child COMPLETE required")
    receipts = e.json(observation / "receipts.json")
    expected = {STEM + suffix for suffix in ("SUMMARY.json", "STATUS.json", "RECORDS.jsonl")}
    downloaded = [item for item in receipts if "file" in item]
    V.require(len(downloaded) == 3 and {item["file"] for item in downloaded} == expected and not any("error_type" in item for item in receipts), "Exactly three successful child downloads required")
    files = observation / "files"
    paths = {name: files / (STEM + suffix) for name, suffix in (("summary", "SUMMARY.json"), ("status", "STATUS.json"), ("records", "RECORDS.jsonl"))}
    for item in downloaded:
        value = C.pin(files / item["file"])
        V.require(item["http_status"] == 200 and {key: item[key] for key in ("bytes", "sha256")} == value, "Official child output bytes differ")
        e.pins[str((files / item["file"]).resolve())] = value
    summary, status = e.json(paths["summary"]), e.json(paths["status"])
    V.require(summary.get("complete") is status.get("complete") is True and "error" not in summary
              and summary.get("phase") == status.get("phase") == "fluid1080_finished" and status.get("error_type") is None
              and summary["schema_version"] == "orthodiff_fluid_fallback1080_frozen_continuous16_v1"
              and summary["route"] == "fluid_sensitive_fallback" and summary["confirmed_PD"] is False
              and summary["protocol"] == config["fluid_protocol"] and summary["frozen_parent_protocol"] == config["preprocess"]
              and summary["source_bindings"] == config["fluid_source_bindings"] and summary["owned_temp_removed"] is True,
              "Incomplete child or wrong fluid protocol cannot qualify")
    V.require(summary["requested_UIDs"] == config["uids"] and summary["cohort_UID_hash"] == C.CHILD_UID_SHA
              and summary["coverage"] == status["coverage"] and summary["coverage"]["requested"] == summary["coverage"]["recorded"] == 1080
              and summary["scope"] == {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                                       "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False}, "Child UID/coverage/scope differs")
    env = summary["environment"]
    V.require(env["python"].startswith("3.13.") and type(env["visible_GPUs"]) is int and env["visible_GPUs"] >= 1,
              "Qualified Python/CUDA device coverage differs")
    digest, size, successes, failures, strata, reasons, prefixes = hashlib.sha256(), 0, 0, 0, Counter(), Counter(), {}
    data_seconds, feature_seconds, count = 0., 0., 0
    with paths["records"].open("rb") as handle:
        for count, line in enumerate(handle, 1):
            V.require(count <= 1080 and line.endswith(b"\n") and 0 < len(line) <= 256 * 1024, "Bounded exact child JSONL order required")
            record = V.parse(line)
            uid = config["uids"][count - 1]
            V.require(record.get("StudyInstanceUID") == uid, "Child unknown/duplicate/out-of-order UID")
            validate_record(record, requests[uid], env["visible_GPUs"])
            data_seconds += record["data_seconds"]
            if record["status"] == "success":
                successes += 1
                strata[record["metadata"]["stratum"]] += 1
                feature_seconds += record["feature_seconds"]
            else:
                failures += 1
                reasons[record["reason_type"]] += 1
            digest.update(line)
            size += len(line)
            if count % 32 == 0 or count == 1080:
                prefixes[count] = {"sha256": digest.hexdigest(), "success": successes, "failure": failures}
    records_pin = {"bytes": size, "sha256": digest.hexdigest()}
    coverage = {"requested": 1080, "recorded": count, "success": successes, "failure": failures, "success_strata": dict(strata), "failure_reason_types": dict(reasons)}
    V.require(count == 1080 and summary["coverage"] == coverage and records_pin == C.pin(paths["records"])
              and summary["journal"] == {"filename": paths["records"].name, **records_pin, "final_full_file_rehash_matches_incremental": True}
              and status["journal"] == {"filename": paths["records"].name, **records_pin}
              and summary["all_requested_MRI_success"] == (successes == 1080)
              and summary["forward_counts"] == {"denoiser": successes, "decoder_blocks": 0, "final_out": 0}, "Child journal/terminal counts/hash mismatch")
    parent_summary = e.json(Path(parent["files"]["summary"]["path"]))
    V.require(summary["model_load_calls"] == 1 and summary["checkpoint"] == parent_summary["checkpoint"]
              and summary["checkpoint"]["keys"] == 257 and len(summary["checkpoint"]["schedule_buffers_exact"]) == 13
              and all(summary["checkpoint"]["schedule_buffers_exact"].values())
              and summary["verified_generic_assets"] == parent_summary["verified_generic_assets"], "Strict257/13/single-load/asset guards differ")
    for name, item in summary["vendor_imports"].items():
        V.require({key: item[key] for key in ("bytes", "sha256")} == {key: bundle["vendor"][name + ".py"][key] for key in ("bytes", "sha256")}
                  and item["path"].endswith("/od_qual_pinned/" + name + ".py"), "Actual scientific source origin differs")
    V.require(set(summary["vendor_imports"]) == set(parent_summary["vendor_imports"])
              and all(env[key] == parent_summary["environment"][key] for key in ("torch", "numpy", "pydicom", "cv2", "cuda", "selected_GPU"))
              and summary["decoder"]["verified_members"] == config["decoder"]["members"]
              and summary["decoder"]["fixtures"] == parent_summary["decoder"]["fixtures"] and summary["decoder"]["install_returncode"] == 0
              and summary["decoder"]["no_index"] is summary["decoder"]["no_deps"] is True
              and summary["decoder"]["core_versions_unchanged"] == parent_summary["decoder"]["core_versions_unchanged"]
              and summary["decoder"]["packages"] == parent_summary["decoder"]["packages"], "Qualified environment/decoder fixtures differ")
    resource, limits = summary["resources"], config["operational_resource_limits"]
    V.require(0 < resource["peak_cuda_allocated_bytes"] <= resource["peak_cuda_reserved_bytes"] <= limits["CUDA_reserved_bytes"]
              and 0 < resource["peak_process_RSS_bytes"] <= limits["process_RSS_bytes"] and 0 < resource["total_seconds"] <= limits["total_seconds"]
              and data_seconds + feature_seconds <= resource["total_seconds"], "Posthoc child resource/8h gates failed")
    listing = e.json(observation / "output-list.json")
    V.require({row["fileName"] for row in listing["files"]} == expected, "Unexpected child output paths")
    stdout = "".join(row["data"] for row in V.parse(listing["log"]) if row.get("stream_name") == "stdout")
    markers, progress = [], []
    for line in stdout.splitlines():
        if line.startswith("ORTHODIFF_FLUID_FALLBACK "):
            markers.append(V.parse(line.split(" ", 1)[1]))
        elif line.startswith("ORTHODIFF_FLUID_FALLBACK_PROGRESS "):
            progress.append(V.parse(line.split(" ", 1)[1]))
    V.require(len(markers) == 1 and markers[0]["complete"] is True and markers[0]["pin"] == C.pin(paths["summary"])
              and any(row["phase"] == "single_frozen_backbone_loaded" for row in progress)
              and any(row["phase"] == "fluid1080_finished" and row["complete"] is True and row["recorded"] == 1080 for row in progress), "Child stdout terminal binding differs")
    for row in progress:
        n = row["recorded"]
        V.require(row["requested"] == 1080 and n == row["success"] + row["failure"]
                  and (prefixes.get(n) == {"sha256": row["journal_prefix_SHA256"], "success": row["success"], "failure": row["failure"]}
                       if n else row["journal_prefix_SHA256"] == V.sha(b"") and row["success"] == row["failure"] == 0), "Child progress prefix hash/counts differ")
    for item in receipts:
        if "label" in item:
            V.require(item["http_status"] == 200 and {key: item[key] for key in ("bytes", "sha256")} == V.pin(e.raw(observation / (item["label"] + ".json"))), "Raw API observation pin mismatch")
    return {"schema_version": "orthodiff_fluid1080_actual_qualification_v1", "passed": True, "child_features_qualified": True,
            "at_UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": C.CHILD_SOURCE_SHA, "UID_order_sha256": C.CHILD_UID_SHA,
            "kernel_id": 139158713, "version": 1, "requested_count": 1080, "coverage": coverage,
            "records_pin": records_pin, "summary_pin": C.pin(paths["summary"]), "status_pin": C.pin(paths["status"]),
            "parent_bundle_pin": C.pin(parent_path), "parent_qualification_pin": C.pin(parent["qualification"]["path"]),
            "subset_pin": config["fluid_source_bindings"]["subset_pin"], "parent_success_unchanged": True, "parent_success_overwrite_count": 0,
            "parent_success_count": 3063, "exact_parent_unknown_family_and_canonical_pins": True,
            "confirmed_PD": False, "route": "fluid_sensitive_fallback", "FP32_feature_values_checked": successes * 256,
            "decoded_SOPs_checked": successes * 16, "resources": resource, "environment": env, "input_pins": e.pins,
            "scope": "Complete new fluid-sensitive route engineering qualification, not confirmed PD, all1080 MRI success, head utility or adoption.",
            "limits": ["Raw pixels/IPP vectors/rawmid2 absent; geometry metadata/FP32 vector serialization/source contracts checked, not independent pixel or pooling replay.",
                       "Parent vectors stay in immutable parent files; child requests are disjoint and only child successes may be merged.",
                       "Resource gates are cooperative/posthoc, not native decoder hard deadlines."],
            "local_activity": {"labels_Report_MRI_model_weights_scientific_import_GPU_network_fit": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "pack", "launch", "parent-bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    V.require(not args.output.exists(), "New child qualification output required")
    e = V.Evidence()
    for path in (Path(__file__), Path(V.__file__), ROOT / "src/rsna_knee/orthodiff_fluid_features.py", ROOT / "src/rsna_knee/orthodiff_fluid_fallback.py"):
        e.raw(path)
    try:
        result = audit(args.observation, args.pack, args.launch, args.parent_bundle, e)
    except Exception as error:
        result = {"schema_version": "orthodiff_fluid1080_actual_qualification_v1", "passed": False, "child_features_qualified": False,
                  "error_type": type(error).__name__, "error": str(error), "input_pins": e.pins}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "child_features_qualified", "coverage", "error")}))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
