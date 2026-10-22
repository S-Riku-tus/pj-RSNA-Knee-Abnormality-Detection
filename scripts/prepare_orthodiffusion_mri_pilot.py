"""Prepare a fixed32 label-free MRI pilot offline; never runs MRI/GPU/API code."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
SERIES = ROOT / "data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train_series.csv"
SPEC = ROOT / "artifacts/research/20261017-orthodiff-mri-pilot-spec-v1"
SPEC_V2 = ROOT / "artifacts/research/20261017-orthodiff-mri-pilot-spec-v2"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1"
SEED = 20261017
ID = "StudyInstanceUID"


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def structure_rows(path, fields):
    """Project CSV structure columns only; never retain targets or Report text."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(name) for name in fields]
        return [{name: row[index] for name, index in zip(fields, indices, strict=True)} for row in reader]


def select_cohort(manifest=MANIFEST, series=SERIES):
    manifest, series = Path(manifest), Path(series)
    meta = json.loads((manifest / "manifest.json").read_bytes())
    audit = json.loads((manifest / "fold-audit-v2.json").read_bytes())
    hashes = {name: pin(manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")}
    if (meta["weak_studies"], meta["gold_studies"], meta["seed"], meta["n_folds"]) != (4207, 58, 20261002, 5):
        raise ValueError("Original reviewed4207 weak/58Gold/fivefold metadata required")
    if (audit.get("manifest_sha256") != {name: hashes[name]["sha256"] for name in ("manifest.json", "weak.csv", "gold.csv")}
            or audit.get("inputs_sha256") != meta["inputs_sha256"] or audit.get("integrity_valid") is not True
            or audit.get("gold_used_for_tuning") is not False):
        raise ValueError("Original unchanged fold audit binding required")
    weak = structure_rows(manifest / "weak.csv", (ID, "group_id", "fold", "source"))
    gold = structure_rows(manifest / "gold.csv", (ID, "group_id"))
    series_rows = structure_rows(series, (ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"))
    weak_uids, gold_uids = {row[ID] for row in weak}, {row[ID] for row in gold}
    weak_groups, gold_groups = {row["group_id"] for row in weak}, {row["group_id"] for row in gold}
    if (len(weak), len(weak_uids), len(gold), len(gold_uids)) != (4207, 4207, 58, 58):
        raise ValueError("Original unique weak/Gold UID counts required")
    if weak_uids & gold_uids or weak_groups & gold_groups:
        raise ValueError("Weak cohort overlaps reserved Gold UID or supplied/report group")
    if not weak_uids <= {row[ID] for row in series_rows}:
        raise ValueError("Weak UID missing from official series metadata")
    groups = defaultdict(list)
    for row in weak:
        if row["source"] != "report_weak" or row["fold"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError("Unchanged reviewed weak source/fold required")
        groups[row["group_id"]].append(row)
    if any(len({row["fold"] for row in members}) != 1 for members in groups.values()):
        raise ValueError("Supplied/report group crosses original folds")
    eligible = [members[0] for members in groups.values() if len(members) == 1]
    def rank(row):
        return hashlib.sha256(f"orthodiff-sag-pilot-v1:{SEED}:{row[ID]}".encode()).hexdigest()
    selected = sorted(eligible, key=lambda row: (rank(row), row[ID]))[:32]
    if len(selected) != 32:
        raise ValueError("At least32 already-singleton groups required; never split groups")
    chosen = {row[ID] for row in selected}
    selected_series = [row for row in series_rows if row[ID] in chosen]
    uids_raw = "".join(row[ID] + "\n" for row in selected).encode()
    structure_raw = json.dumps([(row[ID], row["group_id"], row["fold"]) for row in selected], separators=(",", ":")).encode()
    return {"schema_version": "orthodiff_sagittal_fixed32_UID_only_pilot_v1", "seed": SEED,
            "selection": "existing singleton supplied/report groups only; SHA256('orthodiff-sag-pilot-v1:'+seed+':'+UID) ascending; first32; no eligibility/coverage replacement",
            "weak_count": len(weak_uids), "gold_count": len(gold_uids), "singleton_eligible_count": len(eligible),
            "excluded_non_singleton_studies": len(weak) - len(eligible), "gold_UID_overlap_all_weak": len(weak_uids & gold_uids),
            "gold_group_overlap_all_weak": len(weak_groups & gold_groups), "selected_gold_UID_overlap": len(chosen & gold_uids),
            "selected_gold_group_overlap": len({row["group_id"] for row in selected} & gold_groups),
            "selected_count": len(selected), "selected": [{**row, "selection_hash": rank(row)} for row in selected],
            "UID_order_sha256": hashlib.sha256(uids_raw).hexdigest(), "UID_group_fold_sha256": hashlib.sha256(structure_raw).hexdigest(),
            "fold_counts": dict(Counter(row["fold"] for row in selected)), "manifest_pins": hashes, "series_metadata_pin": pin(series),
            "selected_series_metadata": selected_series, "unchanged_fold_seed": meta["seed"],
            "Gold_labels_or_Report_used": False, "target_values_retained_or_embedded": False,
            "patient_independence_verified": False,
            "selection_counts_not_a_population_prevalence_or_accuracy_sample": True}


def preprocess_spec():
    return {"schema_version": "orthodiff_sagittal_DICOM_adapter_predeclared_v1",
            "source": {"author_commit": "99d15186bb8face1728263023d8414d9405c8f7e",
                       "dataset": "dataset.py:162-179,210-241", "transform": "linear_classifier.py:307-313",
                       "author_NIfTI_orientation_known": False,
                       "adapter_is_a_new_RSNA_DICOM_convention_not_proven_author_orientation": True},
            "series_selection": {"candidate": "train_series.csv Anatomical_Plane casefold=sagittal; no target/Report input",
                "PD_header": "SeriesDescription or SequenceName explicit token PD/PDW/PDTSE/PD_FSE/PROTON DENSITY/PROTONDENSITY; token boundaries, no arbitrary substring",
                "contradiction": "explicit T1/T1W/T2/T2W token rejects series even if PD token present; unknown PD rejects",
                "priority": "non-FS PD before FS PD, then SeriesInstanceUID lexicographic; no image-quality/depth/coverage/feature/score ranking",
                "FS": "official Fat_Suppression=0/1 is required; FS PD fallback is recorded as a separate stratum",
                "selection_before_geometry_or_pixel_decode": True, "failed_chosen_series_replacement": False},
            "geometry": {"IOP_first_triplet": "direction of increasing pixel column", "IOP_second_triplet": "direction of increasing pixel row",
                "require_all_headers_IOP_IPP_PixelSpacing": True, "IOP_unit_and_orthogonal_tolerance": 0.001,
                "IOP_consistency_atol": 0.001, "axis_dominance_cosine_min": 0.90,
                "axes": "normal dominant LPS-x, in-plane bases must uniquely dominate LPS-y and LPS-z; nearest-axis permutation/sign only; retained obliquity is measured and not resampled",
                "tensor_depth": "L-to-R: align physical normal toward LPS-negative-x; ascending dot(IPP, aligned normal)",
                "tensor_rows": "S-to-I: negative LPS-z", "tensor_columns": "A-to-P: positive LPS-y",
                "duplicate_or_nonfinite_positions": "reject; no InstanceNumber fallback",
                "spacing": "strictly positive gaps; max abs(gap-median) <= max(0.1mm,0.10*median); reject otherwise",
                "matrix_and_spacing": "all chosen series slices must agree; positive spacing recorded, no invented physical scale"},
            "clip": {"minimum_depth": 16, "indices": "if d==16 use all; otherwise start=d//2-8, end=d//2+8; exactly16 adjacent original physical-order slices",
                     "sampling_padding_duplication": False, "decode_only_selected16_after_header_checks": True},
            "pixels": {"decode": "pydicom/pylibjpeg after own fixed4synthetic decoder checks; single-frame monochrome only",
                "rescale": "each slice float32 * RescaleSlope(default1) + RescaleIntercept(default0), finite required",
                "MONOCHROME1": "invert entire retained16 volume via retainedvolume min+max-volume; not independent per-slice inversion",
                "inplane": "transpose/flip only by fixed IOP axis mapping; no crop/letterbox/physical-spacing interpolation",
                "resize": "per-slice cv2.INTER_LINEAR to256x256 (skip only if already256x256)",
                "normalize": "after resize, entire retained16 tensor float32 minmax: (v-min)/(max-min+1e-8)*2-1; reject constant/nonfinite",
                "no_percentile_uint8_or_DINO_normalization": True},
            "features": {"single_backbone_load_reused_across_32": True, "precision": "FP32/AMPoff", "timestep": 100,
                "name": "mid_2", "pool": "float32 globalmeanDHW ->256", "per_UID_passes": 1,
                "seed": "existing scoped_seed(StudyUID,SeriesUID,[start,end],20261017); CUDA0 generator only and allRNG restored",
                "no_feature_or_label_based_series_reselection": True},
            "failure": "one fixed record per requested UID, no replacement/zero features/parent fallback; geometry/decode failures retain reason; model/numerical/resource contract failures fatal",
            "outputs": "only256feature values + arraySHA/metadata/per-phase latency/coverage/reasons/receipt; no pixel/DICOM/Report/label/submission dump",
            "runtime_inputs": ["rsraki/orthodiff-sag-assets-20261017-v3-enabled/1", "rsna-knee-abnormality-detection", "rsraki/rsna-dicom-decoders-py313-v1/1"],
            "runtime_embeds": "selected32 UID strings + fixed seed/protocol/pins; no weak/Gold labels, Gold UIDlist, group/fold values or Report",
            "Kaggle_only_real_MRI": True, "Internet": False, "private": True,
            "existing_sampled16_cache_used": False, "accuracy_Public_improvement_or_training_qualification": False}


def preprocess_spec_v2():
    value = preprocess_spec()
    value["schema_version"] = "orthodiff_sagittal_DICOM_adapter_predeclared_v2"
    value["pixels"]["rescale"] = "each slice float64 * RescaleSlope(default1) + RescaleIntercept(default0), finite and nonzero slope required"
    value["pixels"]["resize"] = "float64 per-slice cv2.INTER_LINEAR to256x256 (skip only if already256x256); torch.float32 cast occurs after resize, matching NIfTI get_fdata/cv2 -> torch.float() order"
    value["pixels"]["native_clip_resource_limit"] = "16*Rows*Columns*8<=512MiB float64 before decode; no reduced-resolution substitute"
    value["resource_limits"] = {"CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3, "total_seconds": 1800,
                                "limits_are_measured_postexecution_not_a_hard_C_decoder_timeout": True}
    value["v1_to_v2_only_scientific_difference"] = "DICOM modality-rescale/resize float64 then postresize Torch.float32, matching observed author cast order; UID/series/geometry/clip/model/RNG unchanged"
    return value


def save_spec(output=SPEC, *, version=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new predeclared pilot spec run")
    cohort, preprocess = select_cohort(), preprocess_spec_v2() if version == 2 else preprocess_spec()
    output.mkdir(parents=True)
    for name, value in (("cohort.json", cohort), ("preprocess-spec.json", preprocess)):
        (output / name).write_bytes((json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode())
    summary = {"cohort_pin": pin(output / "cohort.json"), "preprocess_pin": pin(output / "preprocess-spec.json"),
               "selected32": cohort["selected_count"], "UID_order_sha256": cohort["UID_order_sha256"],
               "gold_UID_and_group_overlap": 0, "local_activity": {"MRI_Torch_GPU_network_API_Save_Submit": 0}}
    (output / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode())
    return summary


def load_artificial_builder():
    path = ROOT / "scripts/prepare_orthodiffusion_artificial_gpu.py"
    spec = importlib.util.spec_from_file_location("od_pilot_source_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(output=OUTPUT, spec_dir=SPEC_V2, slug="rsraki/orthodiff-sag-mri32-20261017-v1"):
    output, spec_dir = Path(output), Path(spec_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new immutable MRI pilot package")
    if not spec_dir.exists():
        save_spec(spec_dir, version=2)
    cohort = json.loads((spec_dir / "cohort.json").read_bytes())
    live_cohort = select_cohort()
    preprocess = json.loads((spec_dir / "preprocess-spec.json").read_bytes())
    if cohort != live_cohort or preprocess != preprocess_spec_v2():
        raise ValueError("Predeclared cohort or preprocessing differs")
    old = ROOT / "artifacts/research/20261017-orthodiff-mri-pilot-spec-v1/cohort.json"
    if old.is_file() and json.loads(old.read_bytes())["UID_order_sha256"] != cohort["UID_order_sha256"]:
        raise ValueError("v1 fixed32 UID order changed")
    artificial = load_artificial_builder()
    manifest_raw, upstream, texts = artificial.source_package()
    vendor, import_audit = artificial.vendor_package(texts)
    artificial_config = json.loads((ROOT / "artifacts/kaggle/orthodiff-sag-artificial-gpu-20261017-v1/config.json").read_bytes())
    if artificial_config["stage_manifest_pin"] != {"bytes": len(manifest_raw), "sha256": hashlib.sha256(manifest_raw).hexdigest()}:
        raise ValueError("Artificial-qualified stage pin differs")
    decoder = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007"
    decoder_manifest = json.loads((decoder / "decoder-manifest.json").read_bytes())
    phantom = json.loads((decoder / "phantoms/phantom-manifest.json").read_bytes())
    names = ["decoder-manifest.json", "phantoms/phantom-manifest.json", "phantoms/" + phantom["expected_filename"]]
    names += ["wheels/" + row["filename"] for row in decoder_manifest["wheels"]]
    names += ["phantoms/" + row["filename"] for row in phantom["files"]]
    decoder_members = {name: pin(decoder / name) for name in names}
    for row in decoder_manifest["wheels"]:
        if decoder_members["wheels/" + row["filename"]] != {"bytes": row["bytes"], "sha256": row["sha256"]}:
            raise ValueError("Own decoder wheel positive pin differs")
    if any(decoder_members["phantoms/" + row["filename"]]["sha256"] != row["sha256"] for row in phantom["files"]):
        raise ValueError("Own synthetic fixture positive pin differs")
    config = {"uids": [row[ID] for row in cohort["selected"]], "UID_order_sha256": cohort["UID_order_sha256"],
              "predeclared_cohort_pin": pin(spec_dir / "cohort.json"), "predeclared_preprocess_pin": pin(spec_dir / "preprocess-spec.json"),
              "Gold_UID_group_exclusion_counts": [0, 0], "preprocess": preprocess, "assets": artificial_config,
              "competition_candidates": ["/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection"],
              "series_metadata_pin": cohort["series_metadata_pin"],
              "decoder": {"Input": "rsraki/rsna-dicom-decoders-py313-v1/1", "members": decoder_members,
                          "candidates": ["/kaggle/input/datasets/rsraki/rsna-dicom-decoders-py313-v1", "/kaggle/input/rsna-dicom-decoders-py313-v1"]},
              "artificial_qualification": {"receipt_sha256": "49d2c384", "receipt_SHA_prefix_only_for_historical_link_not_byte_guard": True,
                                         "exact_receipt_pin": pin(ROOT / "artifacts/qa/odg1/20261017T033240998466Z/files/ORTHODIFF_ARTIFICIAL_GPU.json"),
                                         "independent_audit_pin": pin(ROOT / "artifacts/research/20261017-orthodiff-artificial-independent-audit-v1/verification.json")}}
    base_runtime_path = ROOT / "scripts/orthodiffusion_artificial_gpu_runtime.py"
    pilot_runtime_path = ROOT / "scripts/orthodiffusion_mri_pilot_runtime.py"
    runtimes = {"base": base_runtime_path.read_bytes().decode(), "pilot": pilot_runtime_path.read_bytes().decode()}
    for name, text in runtimes.items():
        compile(ast.parse(text), name, "exec")
    bundle = {"runtimes": runtimes, "runtime_pins": {key: {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()} for key, text in runtimes.items()},
              "config": config, "vendor": vendor}
    raw = (json.dumps(bundle, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode()
    cell = ("# Fixed32 UID-only continuous16 MRI pilot; never submit for scoring.\n"
            "import base64 as _od_b64,hashlib as _od_h,json as _od_j,zlib as _od_z\n"
            f"_od_raw=_od_z.decompress(_od_b64.b85decode({encoded!r}))\n"
            f"assert _od_h.sha256(_od_raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n"
            "_od_bundle=_od_j.loads(_od_raw)\n_od_spaces={}\n"
            "for _od_name,_od_text in _od_bundle['runtimes'].items():\n"
            "    _od_bytes=_od_text.encode('utf-8')\n"
            "    assert {'bytes':len(_od_bytes),'sha256':_od_h.sha256(_od_bytes).hexdigest()}==_od_bundle['runtime_pins'][_od_name]\n"
            "    _od_spaces[_od_name]={'__name__':'orthodiff_pilot_'+_od_name}\n"
            "    exec(compile(_od_bytes,'<pinned-'+_od_name+'>','exec'),_od_spaces[_od_name])\n"
            "_od_result=_od_spaces['pilot']['run'](_od_bundle['config'],_od_bundle['vendor'],_od_spaces['base'])\n"
            "if not _od_result['complete']:\n    raise RuntimeError('MRIpilot failed; inspect ORTHODIFF_MRI_PILOT.json')\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": "# Fixed32 OrthoDiffusion sagittal MRI pilot\n\nPredeclared singleton-group UID sample excludes all reserved Gold UID/groups. Runtime embeds only32 UID strings, seed, protocol and hashes. No Report, Gold/weak targets, model training or submission. All actual DICOM access stays on Kaggle. Series selection uses explicit PD header tokens and fixed FS/UID priority before geometry or pixels; failed selections are not replaced. Physical-order central continuous16, explicit independent LPS adapter, float64 resize then Torch.float32 volume minmax, t100/mid_2/global256mean. One strict-loaded frozen backbone serves all32. Only numerical features/metadata/coverage/latency/reasons are output. This does not establish author NIfTI orientation equivalence, clinical accuracy, leaderboard improvement or hidden-test capacity.\n"},
                            {"cell_type": "code", "metadata": {}, "source": cell, "outputs": [], "execution_count": None}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(notebook, indent=2) + "\n").encode()
    if len(source) >= 950000:
        raise ValueError("Notebook capacity exceeded")
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": source.decode(), "language": "python", "kernel_type": "notebook",
               "dataset_data_sources": [config["decoder"]["Input"]], "competition_data_sources": ["rsna-knee-abnormality-detection"],
               "kernel_data_sources": [artificial.STAGE], "is_private": True, "enable_gpu": True, "enable_internet": False,
               "machine_shape": "NvidiaTeslaT4", "docker_image": artificial.DOCKER, "docker_image_pinning_type": "original",
               "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    summary = {"schema_version": "orthodiff_sagittal_fixed32_MRI_pilot_offline_v1", "source": {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()},
               "runtime_pins": bundle["runtime_pins"], "bundle_pin": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
               "config": config, "source_import_audit": import_audit, "runtime_UIDs_only_no_Goldlist_group_fold_targets_or_Report": True,
               "local_activity": {"Torch_source_model_import_MRI_GPU_network_Save_Submit": 0}, "actual_MRI_pilot": "not run"}
    output.mkdir(parents=True)
    values = {"source.ipynb": source, "base-runtime.py": runtimes["base"].encode(), "pilot-runtime.py": runtimes["pilot"].encode()}
    values.update({name: (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode() for name, value in
                   (("save-payload.json", payload), ("manifest.json", summary), ("config.json", config), ("vendor.json", vendor))})
    for name, content in values.items():
        (output / name).write_bytes(content)
    return {"output": str(output), "source": pin(output / "source.ipynb"), "payload": pin(output / "save-payload.json"), "manifest": pin(output / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-output", type=Path, default=SPEC)
    parser.add_argument("--prepare-package", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output) if args.prepare_package else save_spec(args.spec_output), indent=2))


if __name__ == "__main__":
    main()
