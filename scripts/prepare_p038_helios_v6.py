"""Prepare the author's scoredV6 full recipe offline; never Save/Submit/run it."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import importlib.util
import json
import lzma
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/research/20261017-public-frontier-refresh-v4"
PREP = ROOT / "artifacts/research/20261017-p038-preparation-v1"
P037 = ROOT / "artifacts/kaggle/p037-goodpjw-v26-fixed-recipe-20261017-v2"
FULL = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
OUTPUT = ROOT / "artifacts/kaggle/p038-helios-v6-fixed-recipe-20261017-v3"
SLUG = "rsraki/p038-helios-v6-fixed-20261017-v3"
AUTHOR_SHA = "fac1723c32de2d27f59e10ebca7102327bb33346117a3fec3a58937ecf2d3a0e"
P037_RUNTIME_SHA = "47c4c0687bec5dc0f8aad0cf5937f6fdc50edadac42b590eeadd7d0896cd3720"
STAGE = "rsraki/p029-r6v1-stage-20261015-b2/1"
DECODER = "rsraki/rsna-dicom-decoders-py313-v1/1"
GOOD = "goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/27"
DOCKER_SHA = "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
STAGE_LINE = ("    cands += [INPUT / 'notebooks' / 'rsraki' / 'p029-r6v1-stage-20261015-b2' / "
              "'rsna-knee-r6-frozen-fe055406'] if slug == C_SLUG else []\n")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": sha(raw)}


def read(path):
    return json.loads(Path(path).read_bytes())


def require(value, message):
    if not value:
        raise ValueError(message)


def inventory(texts):
    return [{"bytes": len(text.encode()), "sha256": sha(text.encode())} for text in texts]


def patch_r6(text):
    anchor = "    cands = [INPUT / slug, INPUT / 'competitions' / slug]\n"
    require(text.count(anchor) == 1 and STAGE_LINE not in text, "Exactly one original find_mount anchor required")
    result = text.replace(anchor, anchor + STAGE_LINE, 1)
    require(result.replace(STAGE_LINE, "", 1) == text, "Only the declared stage mount line may differ")
    ast.parse(result)
    return result


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def prepare(output=OUTPUT, *, slug=SLUG):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Use a fresh P038 run directory")
    require(re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug), "Own private candidate slug required")
    author_path = EVIDENCE / "helios-new958-V6-source.ipynb"
    require(pin(author_path)["sha256"] == AUTHOR_SHA, "Author scoredV6 source bytes differ")
    book, view, analysis = read(author_path), read(EVIDENCE / "helios-new958-latest-view.json"), read(EVIDENCE / "new-candidate-analysis.json")
    require(view["kernel"]["id"] == 138670458 and view["kernelRun"]["id"] == view["submission"]["sourceScriptVersionId"] == 359117069
            and view["kernelRun"]["kernelVersionNumber"] == 6 and view["submission"]["id"] == 57280318
            and view["submission"]["scoreFormatted"] == "0.958" and view["kernelRun"]["status"] == "COMPLETE",
            "Exact official author scoredV6/submission binding required")
    require(view["kernelRun"]["runInfo"]["dockerImageDigest"] == DOCKER_SHA, "Author Docker digest differs")
    good = read(PREP / "Good-output-source359009299-view.json")
    require(good["kernelRun"]["id"] == 359009299 and good["kernelRun"]["kernelVersionNumber"] == 27
            and good["kernelRun"]["status"] == "COMPLETE", "Fixed author upstream Good outputV27 binding required")
    indices = [index for index, cell in enumerate(book["cells"]) if cell["cell_type"] == "code"]
    original = ["".join(book["cells"][index]["source"]) for index in indices]
    require(indices == list(range(2, 10)) and len(original) == 8, "All eight author science cells required")
    texts = list(original)
    texts[indices.index(6)] = patch_r6(texts[indices.index(6)])
    require([index for index, old, new in zip(indices, original, texts) if old != new] == [6], "Only R6 cell6 operational mount may change")
    for text in texts:
        ast.parse(text.split("\n", 1)[1] if text.startswith("%%writefile ") else text)
    spec = importlib.util.spec_from_file_location("p038_bounded_static_assignment_helper", EVIDENCE / "analyze_new_scored_recipe.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # stdlib static helpers; main()/source are never executed.
    profile = ast.literal_eval(helper.assignment(original[6], "W_RAPTOR_C").value)
    require(profile == {"Medial OA": .5, "Lateral OA": .15, "PF OA": .65, "Synovitis": .85, "Baker's": .30}
            and ast.literal_eval(helper.assignment(original[6], "A224").value) == .35, "Fixed author coefficient profile differs")
    parent = read(P037 / "manifest.json")
    require(pin(P037 / "operational-runtime.py")["sha256"] == parent["operational_runtime_sha256"] == P037_RUNTIME_SHA,
            "Existing P037 operational producer differs")
    old_config = parent["guard_config"]
    config = {key: copy.deepcopy(old_config[key]) for key in ("stage_slug", "R6_folder", "Reader_slug", "R6", "Reader", "fixture_input_slug", "fixture_members", "fixture_syntaxes")}
    decoder = read(FULL / "config.json")["decoder"]
    require(decoder["Input"] == DECODER, "Existing qualified own decoderInput differs")
    require({name.removeprefix("phantoms/"): value for name, value in decoder["members"].items() if name.startswith("phantoms/")} == config["fixture_members"],
            "Existing P037/full four-phantom fixture pins differ")
    config["decoder_manifest_members"] = {"decoder-manifest.json": decoder["members"]["decoder-manifest.json"]}
    config["decoder_wheel_members"] = {name.removeprefix("wheels/"): value for name, value in decoder["members"].items() if name.startswith("wheels/")}
    config["decoder_ABI"] = "cp313"
    require(len(config["R6"]) == 9 and len(config["Reader"]) == len(config["decoder_wheel_members"]) == 3, "R6nine/Readerthree/CP313three members required")
    descriptor = {"candidate": "P038", "author_ref": "heliosli/this-is-best-public-dataset-new-version-0-9576",
        "author_source_id": 359117069, "author_version": 6, "author_submission": 57280318, "author_Public": "0.958", "own_Public": None,
        "author_source_pin": pin(author_path), "author_Docker_digest": DOCKER_SHA, "original_science_cell_indices": indices,
        "author_environment": {"actual_Docker_digest": view["kernelRun"]["runInfo"]["dockerImageDigest"],
            "source_language_python": book["metadata"]["language_info"]["version"], "actual_GPU": view["kernelRun"]["acceleratorType"],
            "source_stdout_R6_global_install_versions": {"pydicom": "3.0.1", "timm": "1.0.25"},
            "author_torch_numpy_pandas_versions_established": False,
            "reason_select_author2757_CP313": "Reproduce Helios V6 actual runtime; P037 GoodV26 Docker37c64/CP312 is a different upstream recipe.",
            "previous_GoodV26_Docker_reference": parent["requested_docker"], "P037_reuse_scope": "operational guard/namespace/transport and same Reader source only",
            "future_actual_transport_qualification": "Read back saved Docker/Python/Input7; record distribution versions before/after preflight and each author cell; actual visible inference and CSV parity required."},
        "R6_stage": STAGE, "Reader": "goodpjw2008/rsna-knee-2-5d-convnext-reader/1", "extra_original_Good_output": GOOD,
        "decoder_source": DECODER, "decoder_target": "/kaggle/working/_own_env", "decoder_ABI": "cp313",
        "synthetic_fixture_count": 4, "synthetic_fixture_member_count": 6, "R6_nine_Reader_three_positive_guards": True,
        "runtime_namespace": "p038_operational_runtime", "R7": False, "Raptor224_mirror_added": False,
        "coefficient_profile": profile, "A224": .35, "reader_k": 16, "Reader_final_FP64_rank": True,
        "scientific_changes_to_author_recipe": False, "single_factor_effect_claimed": False,
        "Gold_ownfit_or_selection": 0, "upstream_Gold_history": analysis["public_asset_and_license_audit"],
        "operational_deltas": ["one immutable R6 stage mount branch", "own strict CP313 three-plugin decoder plus same four-syntax/six-member synthetic preflight",
                               "isolated runtime namespace and exact-cell compressed restore", "copy already-created CSVs/receipts after original cells"]}
    config["producer_descriptor"] = descriptor
    runtime_raw = (ROOT / "scripts/p038_fixed_recipe_runtime.py").read_bytes()
    runtime = runtime_raw.decode()
    ast.parse(runtime)
    raw = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = lzma.compress(raw, preset=6)
    encoded = base64.b85encode(compressed).decode()
    bootstrap = ("import hashlib as _p038_hash\n"
        f"_p038_runtime_text={runtime!r}\n"
        f"assert _p038_hash.sha256(_p038_runtime_text.encode()).hexdigest()=={sha(runtime_raw)!r}\n"
        "_p038_runtime_ns={'__name__':'p038_operational_runtime'}\n"
        "exec(compile(_p038_runtime_text,'<p038-fixed-recipe-runtime>','exec'),_p038_runtime_ns)\n"
        f"_p038_config={config!r}\n"
        "_p038_preflight=_p038_runtime_ns['preflight'](_p038_config)\n")
    transport = (f"_p038_cells=_p038_runtime_ns['restore_cells']({encoded!r},{sha(compressed)!r},{sha(raw)!r},{inventory(texts)!r})\n"
        f"_p038_execution=_p038_runtime_ns['execute_cells'](_p038_cells,{indices!r},get_ipython())\n")
    markdown = ("# P038: exact author Helios V6 full-recipe reproduction candidate\n\n"
        "Official author V6/source359117069/Submit57280318 reports Public.958. The listed latest359224783 is a different source, not this score binding. "
        "Seven of eight author code cells are byte-identical; cell6 only adds an explicit existing frozen R6 stage mount. "
        "R6flipTTA, A224.35, MedialOA.50/LateralOA.15/PFOA.65/Synovitis.85/Baker.30, Readerthreefold/k16/per-targetsmallweights/finalFP64rank remain exact. "
        "NoR7 or224mirror is added. This complete author recipe changes multiple factors relative to P033; Reader-alone gain is not claimed. "
        "Before any science, twelve R6/Reader members and own fixedCP313 three decoder wheels plus four synthetic transfer syntaxes/reference/manifest must pass positive guards. "
        "The original Reader pip checkFalse and fallback remain source-preserved; future actual environment/participation must be audited. "
        "Operational runtime globals are isolated. Science runs in original order in the shared IPython user namespace; observer copies existing CSVs only. "
        "No own.958/.960, hidden completion, Gold-clean selection or deterministic result is established. No builder API Save/Submit code exists.\n")
    candidate = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown},
        *[copy.deepcopy(cell) for cell in book["cells"] if cell["cell_type"] == "markdown"], code_cell(bootstrap), code_cell(transport)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}},
        "nbformat": 4, "nbformat_minor": 5}
    serialized = (json.dumps(candidate, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(serialized) < 950000, "P038 source capacity exceeded")
    datasets = ["heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1", "goodpjw2008/rsna-knee-2-5d-convnext-reader/1",
                "nartaa/rsna-knee-publication-swa-weights-20261007/1", DECODER]
    kernels = [STAGE, GOOD]
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": serialized.decode(), "language": "python", "kernel_type": "notebook",
        "dataset_data_sources": datasets, "kernel_data_sources": kernels, "competition_data_sources": ["rsna-knee-abnormality-detection"],
        "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:" + DOCKER_SHA, "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    manifest = {"schema_version": "p038_helios_scoredV6_fixed_offline_package_v1", "stage": "offline_only_no_execution_or_Save",
        "source": {"bytes": len(serialized), "sha256": sha(serialized)}, "producer_descriptor": descriptor,
        "author_code_inventory": inventory(original), "candidate_code_inventory": inventory(texts), "author_cell_indices": indices,
        "changed_original_cell_indices": [6], "only_author_source_delta": STAGE_LINE, "all_science_bytes_equal_after_mount_normalization": True,
        "compressed_recipe_sha256": sha(compressed), "restored_recipe_sha256": sha(raw), "operational_runtime": pin(ROOT / "scripts/p038_fixed_recipe_runtime.py"),
        "existing_P037_runtime_sha256": P037_RUNTIME_SHA, "guard_config": config, "builder_pin": pin(__file__),
        "inputs": {"datasets": datasets, "kernels": kernels, "competitions": payload["competition_data_sources"], "total": 7},
        "requested_docker": payload["docker_image"], "actual_saved_docker_or_mounted_versions_verified": False,
        "observer": {"copy_only_after_original_cells": [3, 5, 7, 8, 9], "no_scientific_monkeypatch_or_recompute": True},
        "qualification": {"static_only": True, "CPU_QA": False, "actual_MRI_models_GPU_visible_hidden": False},
        "primary_evidence": {str(path.relative_to(ROOT)): pin(path) for path in (author_path, EVIDENCE / "helios-new958-latest-view.json", EVIDENCE / "new-candidate-analysis.json",
            EVIDENCE / "helios-public-R6-dataset-view.json", PREP / "Good-output-source359009299-view.json", P037 / "manifest.json")},
        "limits": ["Author.958 is not own reproduction or0.960 and does not isolate Reader/coefficients.",
            "Source keeps author conditional Reader/failure policies; decoder preflight is an operational qualification, not proof every hidden MRI decoded.",
            "Public R6 currentV2 containsR7 under same names; own immutable b2V1exactR6 is required.",
            "Reader Apache2.0 is primary-dataset metadata; public R6 currentmetadata licenseOther does not itself establish a blanket reuse grant. Existing upstream exposure/license audit is preserved.",
            "Gold ownfit/select0; Raptor Gold58 selection exposure inherited, Reader author training-exclusion statement is not independent selection proof.",
            "No model/MRI download, science execution, GPU, fit, network mutation, Save or Submit occurred."],
        "local_activity": {"model_MRI_source_execution_GPU_fit_Save_Submit": 0, "public_fixed_Input_metadata_read": 1}}
    output.mkdir(parents=True)
    values = {"source.ipynb": serialized, "operational-runtime.py": runtime_raw, "recipe-cells.json": raw,
        "R6-stage-mount-only.diff": "".join(difflib.unified_diff(original[4].splitlines(keepends=True), texts[4].splitlines(keepends=True), fromfile="authorV6-cell06", tofile="P038-cell06")).encode()}
    for folder, items in (("author-cells", original), ("adapted-cells", texts)):
        directory = output / folder
        directory.mkdir()
        for index, text in zip(indices, items):
            (directory / ("cell-%02d.py" % index)).write_bytes(text.encode())
    light = {key: value for key, value in payload.items() if key != "text"}
    light.update(source_file="source.ipynb", source_sha256=sha(serialized))
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload), ("fixed-input-payload.json", light), ("producer-descriptor.json", descriptor)):
        values[name] = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    for name, value in values.items():
        (output / name).write_bytes(value)
    return {"output": str(output), "source": pin(output / "source.ipynb"), "payload": pin(output / "save-payload.json"), "manifest": pin(output / "manifest.json"), "Inputs": 7}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, slug=args.slug), indent=2))


if __name__ == "__main__":
    main()
