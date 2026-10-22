"""Qualify completed full4207 frozen OrthoDiff outputs using only stdlib.

Reads metadata and recorded feature vectors, never image pixels, model weights,
labels or scientific model libraries. No training, network, Save or Submit.
A failed/incomplete run never receives head_features_qualified=True.
"""
from __future__ import annotations

import argparse
import ast
import base64
import builtins
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
SLUG = "orthodiff-sag-fullweak4207-20261017-v2"
KERNEL_ID = 139150600
STEM = "orthodiff-fullweak4207/ORTHODIFF_FULLWEAK_"
RUNTIME_SHA = {"base": "d0b432859d6fcae23192cd31a86330dc92ab3052a1b06253f9cfb7f82b5afe90",
               "pilot": "4e8d671d6b0cf4402afac0c6c4dc3a3a007bb45878a6c40e7340159762244a1a",
               "full": "a7c2a3a6c42a26f0b8f647842e88d47e78b9cc4baf259c67b9ec8278dd268ee6"}
BUFFER_NAMES = {"betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod", "posterior_variance",
                "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2", "posterior_mean_coef3"}

def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw):
    return {"bytes": len(raw), "sha256": sha(raw)}


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def tensor(item, shape):
    require(item["shape"] == shape and item["dtype"] == "torch.float32" and item["bytes"] == math.prod(shape) * 4
            and digest(item["sha256"]), "Tensor shape/precision/bytes/SHA mismatch")


def finite(values):
    return all(type(v) in (int, float) and math.isfinite(v) for v in values)


def pd_tokens(item):
    value = re.sub(r"[_\-.]+", " ", (item["SeriesDescription"] + " " + item["SequenceName"]).casefold())
    contradiction = re.search(r"(?<![a-z0-9])(t1|t1w|t1tse|t1fse|t2|t2w|t2tse|t2fse)(?![a-z0-9])", value) is not None
    positive = re.search(r"(?<![a-z0-9])(pd|pdw|pdtse|pdfse|proton\s*density)(?![a-z0-9])", value) is not None
    require((item["explicit_PD"], item["T1_T2_contradiction"], item["eligible"])
            == (positive, contradiction, positive and not contradiction), "Independent PD token classification differs")
    require(item["reason"] == ("T1_T2_contradiction" if contradiction else "explicit_PD" if positive else "unknown_PD"),
            "PD reason classification differs")


def geometry(meta):
    iop = meta["raw_IOP"]
    require(len(iop) == 6 and finite(iop), "Finite IOP6 required")
    col, row = iop[:3], iop[3:]
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))
    require(abs(dot(col, col) - 1) <= .001 and abs(dot(row, row) - 1) <= .001 and abs(dot(col, row)) <= .001,
            "Unit and orthogonal IOP mismatch")
    normal = [col[1] * row[2] - col[2] * row[1], col[2] * row[0] - col[0] * row[2], col[0] * row[1] - col[1] * row[0]]
    length = math.sqrt(dot(normal, normal))
    normal = [v / length for v in normal]
    if normal[0] > 0:
        normal = [-v for v in normal]
    transpose = abs(col[2]) >= .9 and abs(row[1]) >= .9
    direct = abs(row[2]) >= .9 and abs(col[1]) >= .9
    require(transpose != direct and abs(normal[0]) >= .9, "Unique sagittal axis assignment required")
    outrow, outcol = (col, row) if transpose else (row, col)
    require(meta["transpose"] == transpose and meta["flip_rows"] == (outrow[2] > 0)
            and meta["flip_columns"] == (outcol[1] < 0)
            and all(abs(a - b) <= 1e-12 for a, b in zip(meta["depth_normal_LPS"], normal)), "IOP mapping/sign mismatch")
    require(meta["dominance_cosines"] == [abs(normal[0]), abs(outrow[2]), abs(outcol[1])], "Axis dominance values mismatch")
    depth = meta["depth"]
    order, positions = meta["physical_order"], meta["physical_positions_mm"]
    require(type(depth) is int and depth >= 16 and len(order) == len(positions) == depth
            and sorted(order) == list(range(depth)) and finite(positions), "Full physical order permutation required")
    gaps = [b - a for a, b in zip(positions[:-1], positions[1:])]
    median = statistics.median(gaps)
    require(min(gaps) > .0001 and max(abs(value - median) for value in gaps) <= max(.1, .1 * median)
            and meta["spacing_mm"] == median and meta["minimum_gap_mm"] == min(gaps) and meta["maximum_gap_mm"] == max(gaps),
            "Duplicate/nonuniform spacing or summary mismatch")
    start = depth // 2 - 8
    require((meta["clip_start"], meta["clip_end"]) == (start, start + 16)
            and meta["selected_indices"] == order[start:start + 16], "Central adjacent16 mismatch")
    matrix, spacing, bounds = meta["native_matrix"], meta["pixel_spacing_mm"], meta["retained_pre_resize_range"]
    require(len(matrix) == 2 and all(type(v) is int and v > 0 for v in matrix) and math.prod(matrix) * 16 * 8 <= 512 * 1024**2
            and len(spacing) == 2 and finite(spacing) and min(spacing) > 0 and len(bounds) == 2 and finite(bounds)
            and bounds[1] > bounds[0] and meta["rescale_resize_dtype"] == "float64"
            and meta["photometric"] in ("MONOCHROME1", "MONOCHROME2"), "Matrix/spacing/rescale type mismatch")
    sources = meta["selected_sources"]
    require(len(sources) == len({item["SOPInstanceUID"] for item in sources}) == len({item["filename"] for item in sources}) == 16,
            "Sixteen unique selected SOPs/files required")
    for item in sources:
        require(item["filename"] == item["SOPInstanceUID"] + ".dcm" and item["bytes"] > 0 and digest(item["sha256"])
                and len(item["rescale"]) == 2 and finite(item["rescale"]) and item["rescale"][0] != 0
                and item["syntax"] in {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2",
                                       "1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91"},
                "Selected source SOP/pin/rescale/syntax mismatch")


class Evidence:
    """All consumed file bytes are pinned; large journals are streamed."""

    def __init__(self):
        self.pins = {}

    def raw(self, path, maximum=4 * 1024**2):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, "Bounded regular input required")
        raw = path.read_bytes()
        self.pins[str(path.resolve())] = pin(raw)
        return raw

    def json(self, path):
        return parse(self.raw(path))


def carrier(book):
    tree = ast.parse(next(text(cell) for cell in book["cells"] if cell["cell_type"] == "code"))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id in ("_of_raw", "_od_raw") for target in node.targets)]
    require(len(values) == 1, "One pinned carrier required")
    compressed = base64.b85decode(ast.literal_eval(values[0].args[0].args[0]))
    stream = zlib.decompressobj()
    raw = stream.decompress(compressed, 2 * 1024**2)
    require(stream.eof and not stream.unused_data and not stream.unconsumed_tail, "Bounded exact carrier required")
    return parse(raw), pin(raw)


def source_evidence(evidence, project, pack, launch):
    manifest = evidence.json(pack / "manifest.json")
    payload = evidence.json(launch / "payload.json")
    saved = evidence.json(launch / "get-saved-raw.json")
    source = evidence.raw(pack / "source.ipynb")
    require(sha(source) == SOURCE_SHA and pin(source) == manifest["source"]
            and payload["text"].encode() == source, "Pinned prepared/payload source mismatch")
    book, actual = parse(source), parse(saved["blob"]["source"])
    require([(cell["cell_type"], text(cell)) for cell in book["cells"]]
            == [(cell["cell_type"], text(cell)) for cell in actual["cells"]], "Saved source mismatch")
    meta = saved["metadata"]
    require(meta["id"] == KERNEL_ID and meta["currentVersionNumber"] == 1 and meta["ref"] == "rsraki/" + SLUG
            and meta["isPrivate"] is True and meta["enableGpu"] is True and meta["enableTpu"] is False
            and meta["enableInternet"] is False and meta["machineShape"] == payload["machine_shape"] == "NvidiaTeslaT4"
            and meta["dockerImage"] == payload["docker_image"]
            and payload["docker_image"].endswith("sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"),
            "Owned private/offline GPU identity mismatch")
    require(payload["kernel_data_sources"] == ["rsraki/orthodiff-sag-assets-20261017-v3-enabled/1"]
            and payload["dataset_data_sources"] == ["rsraki/rsna-dicom-decoders-py313-v1/1"]
            and payload["competition_data_sources"] == meta["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
            and meta["kernelDataSources"] == [payload["kernel_data_sources"][0].rsplit("/", 1)[0]]
            and meta["datasetDataSources"] == [payload["dataset_data_sources"][0].rsplit("/", 1)[0]], "Three fixed Inputs mismatch")
    bundle, bundle_pin = carrier(book)
    config = bundle["config"]
    require(bundle_pin == manifest["bundle_pin"] and config == manifest["config"] == evidence.json(pack / "config.json"), "Carrier/config mismatch")
    require(set(bundle["runtimes"]) == set(RUNTIME_SHA), "Three pinned runtime namespaces required")
    for name, value in bundle["runtimes"].items():
        expected = {"bytes": len(value.encode()), "sha256": RUNTIME_SHA[name]}
        require(pin(value.encode()) == bundle["runtime_pins"][name] == manifest["runtime_pins"][name] == expected
                and value.encode() == evidence.raw(pack / (name + "-runtime.py")), "Runtime source mismatch")
    original_pack = project / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1"
    original, _ = carrier(evidence.json(original_pack / "source.ipynb"))
    require(all(bundle["runtimes"][key] == original["runtimes"][key] for key in ("base", "pilot"))
            and bundle["vendor"] == original["vendor"] == evidence.json(pack / "vendor.json"), "Frozen scientific source changed")
    for item in bundle["vendor"].values():
        require(pin(item["text"].encode()) == {key: item[key] for key in ("bytes", "sha256")}, "Vendored source pin mismatch")
    allowed = {"uids", "UID_order_sha256", "predeclared_cohort_pin", "operational_resource_limits", "frozen_pilot_pins"}
    require({key for key in set(config) | set(original["config"]) if config.get(key) != original["config"].get(key)} == allowed,
            "Only full-cohort/control config changes permitted")
    require(config["operational_resource_limits"] == {"CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
            "total_seconds": 28800, "UID_boundary_cooperative_clock_check": True, "hard_C_decoder_timeout": False}, "Full resource contract mismatch")
    return manifest, config, bundle


def cohort_evidence(evidence, project, manifest, config):
    root = project / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1"
    cohort = evidence.json(root / "cohort.json")
    metadata = evidence.json(root / "metadata-audit.json")
    alignment = evidence.json(root / "alignment-audit.json")
    require(pin(evidence.raw(root / "cohort.json")) == config["predeclared_cohort_pin"] == manifest["cohort_pin"] == alignment["cohort_pin"]
            and pin(evidence.raw(root / "alignment-audit.json")) == manifest["alignment_pin"] and metadata["passed"] is alignment["passed"] is True,
            "Prior cohort metadata/alignment binding mismatch")
    rows = cohort["requested"]
    uids = [row["StudyInstanceUID"] for row in rows]
    require(len(uids) == len(set(uids)) == cohort["requested_count"] == 4207 and uids == config["uids"]
            and sha("".join(uid + "\n" for uid in uids).encode()) == config["UID_order_sha256"] == cohort["UID_order_sha256"] == metadata["UID_order_sha256"] == UID_SHA,
            "All4207 frozen UID order mismatch")
    groups = defaultdict(list)
    for row in rows:
        require(row["source"] == "report_weak" and row["fold"] in {"0", "1", "2", "3", "4"}, "Original weak metadata classification mismatch")
        groups[row["group_id"]].append(row)
    mapping = [(row["StudyInstanceUID"], row["group_id"], row["fold"]) for row in rows]
    require(sha(json.dumps(mapping, separators=(",", ":")).encode()) == cohort["UID_group_fold_sha256"] == alignment["UID_group_fold_sha256"]
            and all(len({row["fold"] for row in members}) == 1 for members in groups.values()) and len(groups) == 4102
            and dict(Counter(row["fold"] for row in rows)) == cohort["fold_counts"] == metadata["fold_counts"] == alignment["fold_counts_explicit"],
            "Original whole-group/fivefold mapping mismatch")
    require(cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == metadata["Gold_UID_and_supplied_group_overlap"] == 0
            and cohort["whole_supplied_report_groups_included"] is True and cohort["patient_independence_claimed"] is False
            and cohort["target_values_Report_or_Gold_label_values_retained"] is False
            and alignment["native_index_UID_set_exactly_equal_full4207"] is True
            and alignment["F006_E12_prediction_UID_set_exactly_equal_fold0_821"] is True
            and alignment["fold0_valid_UID_count"] == 821 and alignment["unchanged_fold_seed"] == 20261002,
            "Gold exclusion or original-control metadata qualification mismatch")
    series_path = project / "data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train_series.csv"
    series_raw = evidence.raw(series_path)
    require(pin(series_raw) == config["series_metadata_pin"] == cohort["series_metadata_pin"], "Official series metadata byte binding mismatch")
    series = defaultdict(dict)
    wanted = set(uids)
    for row in csv.DictReader(series_raw.decode("utf-8-sig").splitlines()):
        if row["StudyInstanceUID"] in wanted and row["Anatomical_Plane"].strip().casefold() == "sagittal":
            item = series[row["StudyInstanceUID"]]
            require(row["SeriesInstanceUID"] not in item, "Duplicate official sagittal series")
            item[row["SeriesInstanceUID"]] = {key: row[key] for key in ("Fat_Suppression", "Fluid_Sensitive")}
    return cohort, series


def terminal_gate(summary, status, config):
    require(summary.get("complete") is True and status.get("complete") is True and "error" not in summary
            and summary.get("phase") == status.get("phase") == "full4207_finished" and status.get("error_type") is None,
            "Incomplete/fatal extraction cannot qualify head features")
    require(summary["requested_UIDs"] == config["uids"] and summary["cohort_UID_hash"] == config["UID_order_sha256"]
            and summary["protocol"] == config["preprocess"] and summary["owned_temp_removed"] is True
            and summary["scope"] == {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                                     "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False}, "Full summary identity/scope mismatch")
    require(summary["schema_version"] == "orthodiff_full4207_frozen_continuous16_v1"
            and status["schema_version"] == "orthodiff_full4207_small_status_v1" and summary["coverage"] == status["coverage"]
            and summary["coverage"]["requested"] == summary["coverage"]["recorded"] == 4207
            and summary["current_UID"] == status["current_UID"] == config["uids"][-1], "Terminal coverage/UID binding mismatch")
    scope = summary["full_scope"]
    require(scope["requested_count"] == 4207 and scope["original_supplied_report_groups_included_whole"] is True
            and scope["current_operational_resource_limits"] == config["operational_resource_limits"]
            and scope["frozen_pilot_pins"] == config["frozen_pilot_pins"]
            and scope["runtime_embeds_UIDs_not_groups_folds_Report_or_targets"] is True
            and scope["resumption_or_skip_existing"] is scope["missing_feature_zero_fill"] is False, "New full-cohort scope mismatch")


def validate_selection(detail, official):
    pd_tokens(detail)
    candidates, rejected = detail["eligible_candidate_series"], detail["rejected_candidates"]
    require(detail["eligible"] is True and detail["Fat_Suppression"] in {"0", "1"}
            and candidates and len(candidates) == len(set(candidates))
            and len(rejected) == len({item["SeriesInstanceUID"] for item in rejected})
            and set(candidates).isdisjoint({item["SeriesInstanceUID"] for item in rejected})
            and set(candidates) | {item["SeriesInstanceUID"] for item in rejected} == set(official), "PD candidate partition mismatch")
    for item in [detail, *rejected]:
        if "explicit_PD" in item:
            classified = dict(item)
            classified["reason"] = "T1_T2_contradiction" if item["T1_T2_contradiction"] else "explicit_PD" if item["explicit_PD"] else "unknown_PD"
            pd_tokens(classified)
        original = official[item["SeriesInstanceUID"]]
        require(item["Fat_Suppression"] == original["Fat_Suppression"] and item["Fluid_Sensitive"] == original["Fluid_Sensitive"], "Official series flags mismatch")
    require(all(item.get("reason") and (not item.get("eligible") or item["reason"] not in {"unknown_PD", "T1_T2_contradiction", "explicit_PD"}) for item in rejected)
            and all(official[sid]["Fat_Suppression"] in {"0", "1"} for sid in candidates)
            and detail["SeriesInstanceUID"] == min(candidates, key=lambda sid: (int(official[sid]["Fat_Suppression"]), sid)),
            "Frozen before-geometry PD/FS/UID selection mismatch")
    require(detail["stratum"] == ("PD_nonFS" if detail["Fat_Suppression"] == "0" else "PD_FS_fallback"), "PD stratum mismatch")


def validate_success(record, official, visible_gpus):
    detail = record["metadata"]
    validate_selection(detail, official)
    geometry(detail)
    for name, shape in (("normalized_input", [1, 1, 16, 256, 256]), ("noisy_input", [1, 1, 16, 256, 256]),
                        ("mid_2", [1, 256, 1, 16, 16]), ("pooled", [1, 256])):
        tensor(record[name], shape)
    bounds = record["normalization_min_max_float32"]
    require(len(bounds) == 2 and finite(bounds) and bounds[1] > bounds[0]
            and all(struct.unpack("<f", struct.pack("<f", value))[0] == value for value in bounds), "Postresize FP32 minmax metadata mismatch")
    values = record["pooled_float32_values"]
    require(len(values) == 256 and finite(values) and sha(struct.pack("<256f", *values)) == record["pooled"]["sha256"]
            and all(struct.unpack("<f", struct.pack("<f", value))[0] == value for value in values), "Independent256 FP32 representation/SHA mismatch")
    seed_fields = ["orthodiff-artificial-noise-v1", record["StudyInstanceUID"], detail["SeriesInstanceUID"], [detail["clip_start"], detail["clip_end"]], 20261017]
    seed = int.from_bytes(hashlib.sha256(json.dumps(seed_fields, ensure_ascii=True, separators=(",", ":")).encode()).digest()[:8], "big") % (2**63 - 1)
    require(record["noise_seed"] == seed and record["all_RNG_restored"] is True
            and record["forward_counts"] == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}
            and len(record["all_CUDA_RNG_before_after_SHA"]) == visible_gpus, "UID seed/feature-only/RNG contract mismatch")
    for before, after in [record["CPU_RNG_before_after_SHA"], *record["all_CUDA_RNG_before_after_SHA"]]:
        require(digest(before) and before == after, "CPU or all-visible CUDA RNG not restored")
    require(0 <= record["data_seconds"] and 0 < record["feature_seconds"], "Invalid success phase latency")


def validate_failure(record, official):
    builtin_error = getattr(builtins, record.get("reason_type", ""), None)
    allowed_error = record.get("reason_type") in {"PilotDataError", "InvalidDicomError"} or (
        isinstance(builtin_error, type) and issubclass(builtin_error, (OSError, ValueError, RuntimeError, NotImplementedError)))
    require(allowed_error
            and isinstance(record["reason"], str) and record["reason"] and record["feature_forward_calls"] == 0
            and record["fallback_features"] is False and 0 <= record["data_seconds"]
            and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input", "noisy_input", "noise_seed")),
            "Invalid failure/forward/feature fabrication contract")
    selected = record["selected_series"]
    if selected is not None:
        validate_selection(selected, official)
        return
    require(record["reason"].startswith("no_explicit_noncontradictory_PD_sagittal_series:") and record["reason_type"] == "PilotDataError",
            "Unselected failure must retain source-declared candidate reason")
    rejected = parse(record["reason"].split(":", 1)[1])
    require(isinstance(rejected, list) and len(rejected) == len({item["SeriesInstanceUID"] for item in rejected})
            and {item["SeriesInstanceUID"] for item in rejected} == set(official), "No-selection rejected candidate coverage mismatch")
    for item in rejected:
        require(item.get("reason"), "Rejected candidate reason required")
        if "SeriesDescription" in item and item["reason"] in {"unknown_PD", "T1_T2_contradiction"}:
            pd_tokens(item)
            require(not item["eligible"], "Rejected candidate cannot be eligible")


def compare_prior(record, previous):
    require(record["StudyInstanceUID"] == previous["StudyInstanceUID"] and record["status"] == previous["status"], "Prior32 status/UID changed")
    keys = ("metadata", "noise_seed", "normalization_min_max_float32", "normalized_input", "noisy_input", "mid_2", "pooled", "pooled_float32_values", "forward_counts", "all_RNG_restored") if record["status"] == "success" else (
        "selected_series", "reason_type", "reason", "feature_forward_calls", "fallback_features")
    require(all(record[key] == previous[key] for key in keys), "Prior32 fixed scientific metadata/array/feature evidence changed")
    if record["status"] == "success":
        require(struct.pack("<256f", *record["pooled_float32_values"]) == struct.pack("<256f", *previous["pooled_float32_values"]), "Prior32 pooled FP32 bits changed")


def stream_records(evidence, path, uids, series, visible_gpus, previous):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 128 * 1024**2, "Bounded full JSONL required")
    digest_all, vector_digest, size = hashlib.sha256(), hashlib.sha256(), 0
    count, successes, failures, data_seconds, feature_seconds = 0, [], [], [], []
    strata, reasons, prefixes, prior_seen = Counter(), Counter(), {}, set()
    branches, photometric = Counter(), Counter()
    with path.open("rb") as handle:
        for line in handle:
            require(len(line) <= 256 * 1024 and line.endswith(b"\n") and line.strip(), "Bounded complete nonempty JSONL line required")
            require(count < len(uids), "Extra JSONL records")
            record = parse(line)
            require(record.get("StudyInstanceUID") == uids[count], "Exact4207 record UID order mismatch")
            if record.get("status") == "success":
                validate_success(record, series.get(uids[count], {}), visible_gpus)
                successes.append(uids[count])
                strata[record["metadata"]["stratum"]] += 1
                vector_digest.update(struct.pack("<256f", *record["pooled_float32_values"]))
                feature_seconds.append(record["feature_seconds"])
                for key in ("transpose", "flip_rows", "flip_columns"):
                    branches[key] += bool(record["metadata"][key])
                photometric[record["metadata"]["photometric"]] += 1
            elif record.get("status") == "data_failure":
                validate_failure(record, series.get(uids[count], {}))
                failures.append(uids[count])
                reasons[record["reason_type"]] += 1
            else:
                raise ValueError("Unexpected JSONL status")
            data_seconds.append(record["data_seconds"])
            if uids[count] in previous:
                compare_prior(record, previous[uids[count]])
                prior_seen.add(uids[count])
            digest_all.update(line)
            size += len(line)
            count += 1
            if count % 32 == 0 or count == len(uids):
                prefixes[count] = {"sha256": digest_all.hexdigest(), "success": len(successes), "failure": len(failures)}
    require(count == len(uids) == 4207 and prior_seen == set(previous), "Incomplete stream or missing prior32 comparisons")
    result_pin = {"bytes": size, "sha256": digest_all.hexdigest()}
    evidence.pins[str(path.resolve())] = result_pin
    return {"pin": result_pin, "coverage": {"requested": 4207, "recorded": count, "success": len(successes), "failure": len(failures),
            "success_strata": dict(strata), "failure_reason_types": dict(reasons)}, "prefixes": prefixes,
            "success_UIDs": successes, "failure_UIDs": failures, "data_seconds": data_seconds, "feature_seconds": feature_seconds,
            "concatenated_success_FP32_SHA256": vector_digest.hexdigest(), "branches": dict(branches), "photometric": dict(photometric)}


def audit(observation, pack, launch, project, evidence):
    manifest, config, bundle = source_evidence(evidence, project, pack, launch)
    cohort, series = cohort_evidence(evidence, project, manifest, config)
    require(evidence.json(observation / "status.json").get("status") == "COMPLETE", "Official COMPLETE required before qualification")
    downloads = evidence.json(observation / "receipts.json")
    records = {item["file"]: item for item in downloads if "file" in item}
    expected = {STEM + suffix for suffix in ("SUMMARY.json", "STATUS.json", "RECORDS.jsonl")}
    require(set(records) == expected and len([item for item in downloads if "file" in item]) == 3
            and not any("error_type" in item for item in downloads), "Exactly three successful official output downloads required")
    files = observation / "files"
    summary_path, status_path, records_path = (files / (STEM + suffix) for suffix in ("SUMMARY.json", "STATUS.json", "RECORDS.jsonl"))
    summary, status = evidence.json(summary_path), evidence.json(status_path)
    terminal_gate(summary, status, config)
    for path in (summary_path, status_path):
        require(records[path.relative_to(files).as_posix()].get("http_status") == 200
                and {k: records[path.relative_to(files).as_posix()][k] for k in ("bytes", "sha256")} == pin(evidence.raw(path)), "Official JSON bytes mismatch")
    original_path = project / "artifacts/qa/odm1/20261017T034429492163Z/files/ORTHODIFF_MRI_PILOT.json"
    original = evidence.json(original_path)
    require(pin(evidence.raw(original_path))["sha256"] == "f033452c022f1c292b810f639b3de59a9ef0455744047bc989ec88d92f102f75"
            and original["complete"] is True and original["coverage"]["success"] == 24 and len(original["records"]) == 32, "Exact prior32 receipt required")
    previous = {row["StudyInstanceUID"]: row for row in original["records"]}
    require(len(previous) == 32, "Unique prior32 UIDs required")
    env = summary["environment"]
    require(env["python"].startswith("3.13.") and env["torch"] == "2.11.0+cu128" and env["numpy"] == "2.1.3"
            and env["pydicom"] == "3.0.2" and env["cv2"] == original["environment"]["cv2"]
            and env["cuda"] == "12.8" and env["selected_GPU"] == "Tesla T4" and env["visible_GPUs"] >= 1, "Actual frozen numerical environment mismatch")
    stream = stream_records(evidence, records_path, config["uids"], series, env["visible_GPUs"], previous)
    journal = summary["journal"]
    require(journal == {"filename": records_path.name, **stream["pin"], "final_full_file_rehash_matches_incremental": True}
            and status["journal"] == {"filename": records_path.name, **stream["pin"]}
            and records[STEM + "RECORDS.jsonl"].get("http_status") == 200
            and {k: records[STEM + "RECORDS.jsonl"][k] for k in ("bytes", "sha256")} == stream["pin"]
            and summary["coverage"] == stream["coverage"], "Full journal/prefix/terminal counts mismatch")
    require(summary["all_requested_MRI_success"] == (stream["coverage"]["success"] == 4207)
            and summary["forward_counts"] == {"denoiser": stream["coverage"]["success"], "decoder_blocks": 0, "final_out": 0}, "Aggregate forwards/all-success classification mismatch")
    require(summary["verified_generic_assets"] == [{"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"], "verified": True}
                                                   for row in config["assets"]["assets"]], "Nine guarded assets mismatch")
    require(set(summary["vendor_imports"]) == {"__init__", "unet", "modules", "fp16_util", "features"}, "Actual imported closure mismatch")
    for name, item in summary["vendor_imports"].items():
        require({k: item[k] for k in ("bytes", "sha256")} == {k: bundle["vendor"][name + ".py"][k] for k in ("bytes", "sha256")}
                and item["path"].endswith("/od_qual_pinned/" + name + ".py"), "Imported scientific source pin mismatch")
    checkpoint = summary["checkpoint"]
    shapes = checkpoint["all_key_shapes_dtypes"]
    require(checkpoint == original["checkpoint"] and summary["model_load_calls"] == checkpoint["load_calls"] == 1
            and checkpoint["weights_only"] is checkpoint["strict"] is checkpoint["all_loaded_tensors_equal"] is True
            and checkpoint["arm"] == "ema" and len(shapes) == checkpoint["keys"] == 257
            and sum(math.prod(item["shape"]) for name, item in shapes.items() if name not in BUFFER_NAMES) == checkpoint["parameters"] == 34524545
            and set(checkpoint["schedule_buffers_exact"]) == BUFFER_NAMES and all(checkpoint["schedule_buffers_exact"].values()), "Strict full EMA and exact13 constructor buffers mismatch")
    decoder = summary["decoder"]
    require(decoder["verified_members"] == config["decoder"]["members"] and decoder["fixtures"] == original["decoder"]["fixtures"]
            and decoder["install_returncode"] == 0 and decoder["no_index"] is decoder["no_deps"] is True
            and decoder["core_versions_unchanged"] == {key: env[key] for key in ("numpy", "torch", "pydicom")}
            and decoder["packages"] == original["decoder"]["packages"], "Own4fixture decoder/package/core binding mismatch")
    resource, limits = summary["resources"], config["operational_resource_limits"]
    require(0 < resource["peak_cuda_allocated_bytes"] <= resource["peak_cuda_reserved_bytes"] <= limits["CUDA_reserved_bytes"]
            and 0 < resource["peak_process_RSS_bytes"] <= limits["process_RSS_bytes"] and 0 < resource["total_seconds"] <= limits["total_seconds"]
            and sum(stream["data_seconds"]) + sum(stream["feature_seconds"]) <= resource["total_seconds"]
            and 0 < summary["strict_load_seconds"] <= resource["total_seconds"], "Posthoc8h/14GiB/8GiB resource gates mismatch")
    listing = evidence.json(observation / "output-list.json")
    require({row["fileName"] for row in listing["files"]} == expected, "Unexpected fullweak custom outputs")
    events = parse(listing["log"])
    stdout = "".join(row["data"] for row in events if row.get("stream_name") == "stdout")
    marker, progress = [], []
    for line in stdout.splitlines():
        if line.startswith("ORTHODIFF_FULLWEAK "):
            marker.append(parse(line.split(" ", 1)[1]))
        elif line.startswith("ORTHODIFF_FULLWEAK_PROGRESS "):
            progress.append(parse(line.split(" ", 1)[1]))
    require(len(marker) == 1 and marker[0]["complete"] is True and marker[0]["pin"] == pin(evidence.raw(summary_path)), "Typed stdout summary binding mismatch")
    require(progress and any(row["phase"] == "single_frozen_backbone_loaded" for row in progress)
            and any(row["phase"] == "full4207_finished" and row["recorded"] == 4207 and row["complete"] is True for row in progress), "Loaded/terminal progress markers required")
    for row in progress:
        require(set(row) == {"phase", "elapsed_seconds", "complete", "requested", "recorded", "success", "failure", "journal_prefix_SHA256"}
                and row["requested"] == 4207 and row["recorded"] == row["success"] + row["failure"], "Small progress contract mismatch")
        count = row["recorded"]
        if count:
            require(count in stream["prefixes"] and stream["prefixes"][count] == {"sha256": row["journal_prefix_SHA256"], "success": row["success"], "failure": row["failure"]},
                    "Reported progress prefix differs from downloaded journal")
        else:
            require(row["journal_prefix_SHA256"] == sha(b"") and row["success"] == row["failure"] == 0, "Empty loaded-prefix mismatch")
    for item in downloads:
        if "label" in item:
            require(item.get("http_status") == 200 and {k: item[k] for k in ("bytes", "sha256")} == pin(evidence.raw(observation / (item["label"] + ".json"))), "Official status/list raw pin mismatch")
    fold_by_uid = {row["StudyInstanceUID"]: row["fold"] for row in cohort["requested"]}
    success_folds = Counter(fold_by_uid[uid] for uid in stream["success_UIDs"])
    failure_folds = Counter(fold_by_uid[uid] for uid in stream["failure_UIDs"])
    return {"schema_version": "orthodiff_fullweak4207_actual_qualification_v1", "passed": True, "head_features_qualified": True,
            "at_UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": SOURCE_SHA, "UID_order_sha256": UID_SHA, "requested_count": 4207,
            "kernel_id": KERNEL_ID, "version": 1, "records_pin": stream["pin"], "summary_pin": pin(evidence.raw(summary_path)), "status_pin": pin(evidence.raw(status_path)),
            "coverage": stream["coverage"], "all_requested_MRI_success": summary["all_requested_MRI_success"],
            "prior32_comparison": {"passed": True, "UID_joined": 32, "success": 24, "failure": 8,
                "scientific_metadata_and_selected16_SOP_pins_equal": True, "normalized_noisy_mid2_pooled_SHA_and_6144_FP32_bits_equal": True,
                "ignored": ["timing", "temporary paths", "process-start CPU/CUDA RNG state hashes; each run must restore its own states"]},
            "success_vectors": {"count": stream["coverage"]["success"], "width": 256, "dtype": "float32", "values_and_SHA_checked": stream["coverage"]["success"] * 256,
                "concatenated_success_FP32_SHA256": stream["concatenated_success_FP32_SHA256"], "failure_zero_fill": False},
            "success_UID_order_sha256": sha("".join(uid + "\n" for uid in stream["success_UIDs"]).encode()),
            "failure_UID_order_sha256": sha("".join(uid + "\n" for uid in stream["failure_UIDs"]).encode()),
            "original_group_fold_metadata": {"UID_group_fold_sha256": cohort["UID_group_fold_sha256"], "group_count": cohort["group_count"],
                "fold_counts": cohort["fold_counts"], "Gold_UID_overlap": 0, "Gold_group_overlap": 0, "patient_independence_claimed": False},
            "feature_coverage_by_original_fold": {fold: {"requested": cohort["fold_counts"][fold], "success": success_folds[fold], "failure": failure_folds[fold]}
                                                   for fold in ("0", "1", "2", "3", "4")},
            "resources": resource, "environment": env, "progress_prefixes_verified": len(progress), "actual_mapping_branch_counts": stream["branches"], "photometric_counts": stream["photometric"],
            "input_pins": evidence.pins,
            "scope": "Complete classified4207 metadata/feature transport qualifies downstream head input, not all4207 successful MRI or head accuracy.",
            "limits": ["Raw pixels/full headers/IPP vectors/resized arrays/rawmid2 not exported: pixel math, IPP reprojection and mean pooling cannot be independently replayed.",
                       "Data failure reasons and no-feature/zero-forward semantics are source-bound; native decoder failures are not independently reproduced.",
                       "8h gate is cooperative/posthoc, not a native decoder hard timeout. No training/accuracy/Public gain or patient independence qualification."],
            "local_activity": {"labels_Report_MRI_model_scientific_import_GPU_network_Save_Submit_fit": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "pack", "launch", "project", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "New qualification output required")
    evidence = Evidence()
    evidence.raw(Path(__file__))
    try:
        result = audit(args.observation, args.pack, args.launch, args.project, evidence)
    except Exception as error:
        result = {"schema_version": "orthodiff_fullweak4207_actual_qualification_v1", "passed": False, "head_features_qualified": False,
                  "error_type": type(error).__name__, "error": str(error), "input_pins": evidence.pins}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "head_features_qualified", "coverage", "error")}))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
