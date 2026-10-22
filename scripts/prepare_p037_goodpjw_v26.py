"""Offline P037 preparation of scored Goodpjw V26's complete fixed recipe.

Only parses/hashes/compresses public source and metadata. Does not import or run
the scientific code, inspect model/MRI tensors, call CUDA/network, Save or Submit.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
import lzma
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "artifacts/research/20261017-public-frontier-v1"
SPEC = ROOT / "artifacts/research/20261017-goodpjw-v26-reproduction-audit-v1/builder-spec.json"
OUTPUT = ROOT / "artifacts/kaggle/p037-goodpjw-v26-fixed-recipe-20261017-v2"
SLUG = "rsraki/p037-goodv26-fixed-20261017-v2"
AUTHOR_SHA = "0cf3f48039f0d13b7cf2a47bb7445412b04eab013184ce315f0673d91236cef5"
SPEC_SHA = "972535f68fd91d6f814b5e8707d50a0708d972f9e233c18b09300da544b4ea35"
RUNTIME = ROOT / "scripts/p037_fixed_recipe_runtime.py"
STAGE = "rsraki/p029-r6v1-stage-20261015-b2/1"
STAGE_LINE = ("    cands += [INPUT / 'notebooks' / 'rsraki' / 'p029-r6v1-stage-20261015-b2' / "
              "'rsna-knee-r6-frozen-fe055406'] if slug == C_SLUG else []\n")
DECODER_INPUT = "rsraki/rsna-dicom-decoders-py313-v1/1"
PHANTOMS = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007/phantoms/phantom-manifest.json"
PHANTOM_SHA = "882a942b99d5fb9a55e7521bcc24ec22edc76fba12f094cffb8afe3100a11dba"
READER_WHEEL_SIZES = {
    "pylibjpeg-2.1.0-py3-none-any.whl": 25472,
    "pylibjpeg_libjpeg-2.4.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 3395407,
    "pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 3409132,
    "pylibjpeg_openjpeg-2.6.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": 1986047,
    "pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl": 1997697,
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError("Fixed source/spec SHA differs: " + str(path))
    return raw, json.loads(raw)


def source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def patch_r6(text):
    anchor = "    cands = [INPUT / slug, INPUT / 'competitions' / slug]\n"
    if text.count(anchor) != 1 or STAGE_LINE in text:
        raise ValueError("Expected one untouched R6 find_mount anchor")
    changed = text.replace(anchor, anchor + STAGE_LINE, 1)
    if changed.replace(STAGE_LINE, "", 1) != text:
        raise ValueError("Changes beyond the explicit operational stage branch")
    ast.parse(changed)
    return changed


def inventory(texts):
    return [{"bytes": len(text.encode()), "sha256": digest(text.encode())} for text in texts]


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def prepare(output=OUTPUT, *, slug=SLUG):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh P037 run directory")
    if not re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Expected an own private candidate slug")
    author_raw, book = pinned(AUTHOR / "goodpjw958-V26-source.ipynb", AUTHOR_SHA)
    _, spec = pinned(SPEC, SPEC_SHA)
    view = json.loads((AUTHOR / "goodpjw958-V26-view.json").read_bytes())
    if (view["kernelRun"]["id"] != 358890016 or view["kernelRun"]["kernelVersionNumber"] != 26
            or view["submission"]["id"] != 57256120 or view["submission"]["sourceScriptVersionId"] != 358890016
            or view["submission"]["scoreFormatted"] != "0.958"):
        raise ValueError("Author score must bind to the exact V26 source")
    indices = [i for i, cell in enumerate(book["cells"]) if cell["cell_type"] == "code"]
    original = [source(book["cells"][index]) for index in indices]
    if len(indices) != 15 or len(author_raw) != 974397 or {
            str(i): entry for i, entry in zip(indices, inventory(original))} != spec["preserve_all_author_code_cells_sha256"]:
        raise ValueError("All 15 author code-cell byte pins required")
    texts = list(original)
    texts[indices.index(9)] = patch_r6(original[indices.index(9)])
    if [index for index, old, new in zip(indices, original, texts) if old != new] != [9]:
        raise ValueError("Only the R6 mount-discovery cell may differ")
    for text in texts:
        ast.parse(text.split("\n", 1)[1] if text.startswith("%%writefile ") else text)
    fixed = spec["stage_reuse_variant"]["retain_R6_nine_member_guard"]
    r6 = {name: {"sha256": sha, "bytes": fixed["EXPECTED_SIZES"][name]}
          for name, sha in fixed["EXPECTED_SHA"].items()}
    phantom_raw, phantom = pinned(PHANTOMS, PHANTOM_SHA)
    fixture_members = {"phantom-manifest.json": {"sha256": PHANTOM_SHA, "bytes": len(phantom_raw)},
                       "expected.npy": {"sha256": phantom["expected_sha256"], "bytes": 640}}
    fixture_members.update({row["filename"]: {"sha256": row["sha256"], "bytes": 1026 if i < 2 else 874}
                            for i, row in enumerate(phantom["files"])})
    config = {"stage_slug": STAGE.split("/")[1], "R6_folder": fixed["R6_SLUG"],
              "R7_slug": "rsna-knee-compact-r7cnxfsss", "Reader_slug": "rsna-knee-2-5d-convnext-reader",
              "R6": r6, "R7": spec["R7_members"], "Reader": spec["Reader_members"],
              "fixture_input_slug": DECODER_INPUT.split("/")[1], "fixture_members": fixture_members,
              "fixture_syntaxes": {row["filename"]: row["syntax"] for row in phantom["files"]},
              "reader_wheel_sizes": READER_WHEEL_SIZES}
    raw = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = lzma.compress(raw, preset=6)
    encoded = base64.b85encode(compressed).decode()
    runtime_raw = RUNTIME.read_bytes()
    runtime_text = runtime_raw.decode()
    ast.parse(runtime_text)
    bootstrap = ("import hashlib as _p037_hash\n"
                 f"_p037_runtime_text = {runtime_text!r}\n"
                 f"if _p037_hash.sha256(_p037_runtime_text.encode()).hexdigest() != {digest(runtime_raw)!r}:\n"
                 "    raise ValueError('P037 operational runtime SHA differs')\n"
                 "_p037_runtime_ns = {'__name__': 'p037_operational_runtime'}\n"
                 "exec(compile(_p037_runtime_text, '<p037-fixed-recipe-runtime>', 'exec'), _p037_runtime_ns)\n"
                 f"_p037_config = {config!r}\n"
                 "_p037_preflight = _p037_runtime_ns['preflight'](_p037_config)\n")
    transport = (f"_p037_cells = _p037_runtime_ns['restore_cells']({encoded!r}, {digest(compressed)!r}, {digest(raw)!r}, {inventory(texts)!r})\n"
                 f"_p037_execution = _p037_runtime_ns['execute_cells'](_p037_cells, {indices!r}, get_ipython())\n")
    candidate = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": (
        "# P037: fixed Goodpjw V26 complete recipe\n\n"
        "Author V26/source358890016/Submit57256120 reports Public .958; this is a future own reproduction candidate. "
        "Fourteen of its fifteen code cells are byte-identical. R6 cell9 adds one explicit immutable stage-mount branch. "
        "Raptor384/224 anatomical mirror, R6 TTA ON, R7 TTA OFF, Reader k16/FS counterpart, all coefficients, "
        "FP32 core CSV, final float64 rank and partial-failure policies are unchanged. "
        "The complete package changes several arms relative to P033/P036; no isolated scientific gain is claimed. "
        "A separate preflight requires all 21 R6/R7/Reader member hashes and the existing own four synthetic "
        "decoder phantoms, reference and manifest hashes. Reader V1 contains no author fixture; the seventh Input "
        "provides only this extra operational qualification. Three exact Reader ABI wheels are installed strictly. "
        "All four transfer syntaxes must match reference shape, dtype and values before scientific code. "
        "Operational functions use an isolated namespace; scientific cells keep the original IPython user namespace. "
        "Fixed cells are restored, hashed and run in original order through IPython.run_cell. "
        "The wrapper copies already-created prediction CSVs/receipts for independent replay. "
        "No own .958/.960, hidden completion, Gold-clean status or deterministic predictions are established. "
        "No builder upload/Save/Submit operation exists. The following inherited author prose preserves credits "
        "and upstream declarations; its reported results are not this candidate's own scores.\n")},
        *[copy.deepcopy(cell) for cell in book["cells"] if cell["cell_type"] == "markdown"],
        code_cell(bootstrap), code_cell(transport)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    serialized = (json.dumps(candidate, ensure_ascii=False, indent=1) + "\n").encode()
    if len(serialized) >= 950000:
        raise ValueError("P037 capacity exceeds conservative Save source ceiling")
    for cell in candidate["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    datasets = [ref for ref in spec["fixed_datasets_exact_author"] if not ref.startswith("rostigor/")]
    datasets.append(DECODER_INPUT)
    docker = "gcr.io/kaggle-private-byod/python@sha256:" + spec["runtime"]["authorDockerSHA"]
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": serialized.decode(),
               "language": "python", "kernel_type": "notebook", "dataset_data_sources": datasets,
               "kernel_data_sources": [STAGE], "competition_data_sources": [spec["competition"]],
               "is_private": True, "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
               "docker_image": docker, "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    manifest = {"schema_version": "p037_goodpjw_V26_fixed_package_offline_v2",
                "prepared_at_UTC": datetime.now(timezone.utc).isoformat(), "stage": "offline_only_no_execution_or_Save",
                "source_sha256": digest(serialized), "source_bytes": len(serialized), "author_source_sha256": AUTHOR_SHA,
                "author_script_version": 358890016, "author_notebook_version": 26, "author_submission": 57256120,
                "author_Public": "0.958", "own_Public": None, "goal_0960_guaranteed": False,
                "author_cell_indices": indices, "author_code_inventory": inventory(original),
                "candidate_code_inventory": inventory(texts), "changed_original_cell_indices": [9],
                "only_author_source_delta": STAGE_LINE, "scientific_bodies_byte_preserved_after_mount_normalization": True,
                "compressed_recipe_sha256": digest(compressed), "restored_recipe_sha256": digest(raw),
                "operational_runtime_sha256": digest(runtime_raw), "builder_sha256": digest(Path(__file__).read_bytes()),
                "guard_config": config, "stage_source_sha256": spec["stage_reuse_variant"]["stage_source_sha256"],
                "inputs": {"datasets": datasets, "kernels": [STAGE], "competitions": [spec["competition"]],
                           "total": len(datasets) + 2}, "requested_docker": docker,
                "actual_saved_docker_verified": False, "actual_mounted_Input_versions_verified": False,
                "transport": "isolated_runtime_globals_then_exact_scientific_IPython_user_ns_cells",
                "decoder_preflight": {"strict_install": True, "all_four_existing_own_synthetic_phantoms_required": True,
                                      "same_target_as_author": "/kaggle/working/_own_env", "no_index": True,
                                      "seventh_Input_fixture_only": True, "author_fixture_found": False,
                                      "installed_plugins": ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
                                      "pyjpegls_gdcm_not_present_in_Reader_Input": True,
                                      "actual_wheel_compatibility_verified": False},
                "observer": {"copy_only_after_cells": [4, 7, 10, 13, 15, 21], "raw224_missing_in_author_source": True,
                             "no_recompute_or_monkeypatch": True},
                "qualification": {"static_only": True, "CPU_transport_fixture": False, "real_MRI": False,
                                  "models": False, "GPU": False, "visible": False, "hidden": False},
                "limits": ["Author .958 full-package reproduction is not own improvement or a single-factor contrast.",
                           "Author hidden R7/Reader participation, 9h completion, Gold selection independence and bit determinism are unverified.",
                           "Author Docker may be remapped; actual saved Docker/Python/Input versions require future readback.",
                           "Reader check=False five-package behavior remains; separate strict three-plugin synthetic evidence cannot prove every MRI decoded or the later timm environment.",
                           "No new model/MRI/GPU/network/authentication/training/Save/Submit operation was performed."],
                "source_evidence": {"spec_sha256": SPEC_SHA, "source_original_preserved": True,
                                    "own_synthetic_manifest_sha256": PHANTOM_SHA},
                "v1_failure": "Reader V1 has no author fixture/reference; five optional install names are not five provided wheels. v1 remains unqualified.",
                "v2_operational_deltas": ["own fixed synthetic fixture Input7", "strict provided three ABI wheel installation",
                                          "all four pinned synthetic transfer syntaxes", "partial member progress on failure",
                                          "runtime globals isolated from scientific IPython user namespace"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(serialized)
    (output / "operational-runtime.py").write_bytes(runtime_raw)
    (output / "recipe-cells.json").write_bytes(raw)
    for directory, values in (("author-cells", original), ("adapted-cells", texts)):
        target = output / directory
        target.mkdir()
        for index, text in zip(indices, values):
            (target / ("cell-%02d.py" % index)).write_bytes(text.encode())
    difference = "\n".join(difflib.unified_diff(original[5].splitlines(), texts[5].splitlines(),
                                                fromfile="GoodV26-cell09", tofile="P037-cell09")) + "\n"
    (output / "R6-stage-mount-only.diff").write_text(difference, encoding="utf-8")
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    light = {key: value for key, value in payload.items() if key != "text"}
    light.update(source_file="source.ipynb", source_sha256=digest(serialized))
    (output / "fixed-input-payload.json").write_text(json.dumps(light, indent=2) + "\n", encoding="utf-8")
    if digest((AUTHOR / "goodpjw958-V26-source.ipynb").read_bytes()) != AUTHOR_SHA:
        raise ValueError("Original author source changed during preparation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args()
    manifest = prepare(args.output_dir, slug=args.slug)
    print(json.dumps({"source_sha256": manifest["source_sha256"], "source_bytes": manifest["source_bytes"],
                      "Inputs": manifest["inputs"]["total"], "stage": manifest["stage"]}))


if __name__ == "__main__":
    main()
