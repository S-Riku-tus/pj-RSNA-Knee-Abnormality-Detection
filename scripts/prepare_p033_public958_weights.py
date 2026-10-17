"""Prepare the public .958 coefficient vector on the successful P029M source.

Offline text/metadata preparation only. No source inference, model load, MRI,
network, authentication, upload, Kaggle Save or submission is performed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "artifacts/research/20261016-f006-execution-v1/recovery"
PUBLIC = ROOT / "artifacts/research/20261016-public958-readonly-v1"
OUTPUT = ROOT / "artifacts/kaggle/p033-p029m-public958-weights-20261016-v1"
SLUG = "rsraki/p033-r6-classweights-20261016-v1"
PARENT_SHA = "a00ea9bb525221ee7fac290e92cfd83bcda4402b322e70bcb2d56cdbca1aa992"
AUTHOR_SHA = "f7bfe3a10c2f82d72f582e0561d7864eed771acfc74c0b744e56b63e6f4058bf"
GUARD_SHA = "411e9598b806757de14adf8de6e586eae3209c0460797f8e24eeb65fdf960536"
R6_SHA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
STAGE_SOURCE_SHA = "fe5e4bbf74491fbcc3b3036fca69515ad4fec1ab364eae8897b34f09c7bb2ee1"
STAGE = "rsraki/p029-r6v1-stage-20261015-b2/1"
STAGE_BRANCH = "    cands += [INPUT / 'notebooks' / 'rsraki' / 'p029-r6v1-stage-20261015-b2' / slug]\n"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
DATASETS = ["nartaa/rsna-knee-publication-swa-weights-20261007/1",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1"]
PARENT_WEIGHTS = {"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}
PUBLIC_WEIGHTS = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6,
                  "Synovitis": 0.8, "Baker's": 0.35}
PARENT_CODE_SHA = [GUARD_SHA,
    "3858126b82681ba469934c84a79f360eb199df59c16159b7a168df94e415cb37",
    "d08db7bd194a78dac8c923a50048d098dd297223cc971e0fec40ba996d840e78",
    "0b0545b46407a46d5a4ebd6a1a941b24103e62844feffda96d6fc37a48cff090",
    "10e20e388df7fed2aaecc900cfdebb294735eea8965924748d31d4db28d6d823",
    "93b631c8b857a0f8f02671cf3e3833043f964f536cce6d46e44118d9550a6a0c",
    "68f70dd31cc3544cb5734313d06538355a34b122ff430a32820756175c9c9dd9",
    "d0fbbf1c24806ee136d33df0e463de8763eea846f0be77b3a4ed98366a0d4e79"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError("Pinned source/metadata differs: " + Path(path).name)
    return raw, json.loads(raw)


def assignment(tree, name):
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name)
              and n.id == name and isinstance(n.ctx, ast.Store)]
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    if len(stores) != 1 or len(nodes) != 1:
        raise ValueError("Exactly one literal assignment required: " + name)
    return nodes[0]


def replace_weights(text, weights):
    before = ast.parse(text)
    node = assignment(before, "W_RAPTOR_C")
    if ast.literal_eval(node.value) != PARENT_WEIGHTS or weights != PUBLIC_WEIGHTS:
        raise ValueError("Only the fixed public .958 vector is accepted")
    value = node.value
    if value.lineno != value.end_lineno:
        raise ValueError("Expected single-line coefficient dictionary")
    lines = text.splitlines(keepends=True)
    line = lines[value.lineno - 1].encode("utf-8")
    lines[value.lineno - 1] = (line[:value.col_offset] + repr(weights).encode()
                              + line[value.end_col_offset:]).decode("utf-8")
    result = "".join(lines)
    node.value = ast.parse(repr(weights), mode="eval").body
    if ast.dump(before, include_attributes=False) != ast.dump(ast.parse(result), include_attributes=False):
        raise ValueError("Scientific AST differs beyond the public coefficient vector")
    return result


def code_ast(text, filename):
    if text.startswith("%%writefile "):
        header, text = text.split("\n", 1)
        if header not in {"%%writefile /tmp/raptor_run.py", "%%writefile /tmp/raptor224_run.py"}:
            raise ValueError("Unexpected writefile destination")
    ast.parse(text, filename=filename)


def prepare(output=OUTPUT, *, recovery=RECOVERY, public=PUBLIC, slug=SLUG):
    output, recovery, public = Path(output), Path(recovery), Path(public)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new P033 directory; preserve old sources/results")
    if not re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Expected an own private candidate slug")
    parent_raw, parent = pinned(recovery / "P029M-source.ipynb", PARENT_SHA)
    author_raw, author = pinned(public / "public958-source.ipynb", AUTHOR_SHA)
    _, stage = pinned(recovery / "R6-stage-source.ipynb", STAGE_SOURCE_SHA)
    r6_raw, r6 = pinned(public / "w1-manifest.json", R6_SHA)
    own_score = json.loads((recovery / "P029M-submission57225898-raw.json").read_text(encoding="utf-8"))
    view = json.loads((public / "public958-view.json").read_text(encoding="utf-8"))
    metadata = json.loads((recovery / "P029M-raw.json").read_text(encoding="utf-8"))["metadata"]
    if (own_score.get("ref") != 57225898 or own_score.get("status") != "COMPLETE"
            or own_score.get("publicScore") != "0.957"):
        raise ValueError("Existing successful parent official score binding differs")
    if (view["kernelRun"]["id"] != 358751289 or view["kernelRun"]["kernelVersionNumber"] != 1
            or view["kernelRun"]["status"] != "COMPLETE"
            or view["submission"].get("sourceScriptVersionId") != 358751289
            or view["submission"].get("scoreFormatted") != "0.958"):
        raise ValueError("Public .958 must be linked to the exact scored source")
    if (metadata["ref"] != "rsraki/p029m-r6-957-20261015-a91d20e5dd62"
            or metadata["currentVersionNumber"] != 1 or not metadata["isPrivate"]
            or not metadata["enableGpu"] or metadata["enableInternet"]
            or metadata["dockerImage"] != DOCKER
            or set(metadata["datasetDataSources"]) != {x.rsplit("/", 1)[0] for x in DATASETS}
            or metadata["kernelDataSources"] != [STAGE.rsplit("/", 1)[0]]
            or metadata["competitionDataSources"] != ["rsna-knee-abnormality-detection"]):
        raise ValueError("Successful parent's source/Input/environment contract differs")
    if len(parent["cells"]) != 11 or len(author["cells"]) != 9:
        raise ValueError("Unexpected source cell layout")
    codes = [source(c) for c in parent["cells"] if c["cell_type"] == "code"]
    if [digest(x.encode()) for x in codes] != PARENT_CODE_SHA:
        raise ValueError("Eight successful parent cell hashes differ")
    guard = ast.parse(source(parent["cells"][1]))
    guard_values = {name: ast.literal_eval(assignment(guard, name).value)
                    for name in ("EXPECTED_SHA", "EXPECTED_SIZES", "EXPECTED_MODELS", "STAGE_SLUG", "R6_SLUG")}
    expected_members = {**r6["sha256"], "manifest.json": R6_SHA}
    if (guard_values["EXPECTED_SHA"] != expected_members or len(expected_members) != 9
            or guard_values["EXPECTED_MODELS"] != r6["models"]
            or r6.get("bundle_id") != "fe055406fad6a12f"
            or guard_values["STAGE_SLUG"] != STAGE.split("/")[1]
            or guard_values["R6_SLUG"] != "rsna-knee-r6-frozen-fe055406"):
        raise ValueError("Nine-member fixed R6 guard differs")
    parent_compact = ast.parse(source(parent["cells"][8]))
    author_compact = ast.parse(source(author["cells"][6]))
    for tree in (parent_compact, author_compact):
        if ast.literal_eval(assignment(tree, "FLIP_TTA").value) is not True:
            raise ValueError("Both compared recipes must retain TTA ON")
    modules = ast.literal_eval(assignment(parent_compact, "SOURCES").value)
    author_modules = ast.literal_eval(assignment(author_compact, "SOURCES").value)
    pins = ast.literal_eval(assignment(parent_compact, "SOURCE_SHA").value)
    if modules != author_modules or len(modules) != 7:
        raise ValueError("Seven embedded inference module bodies differ")
    for name, text in modules.items():
        if digest(text.encode()) != pins[name]:
            raise ValueError("Embedded source SHA differs: " + name)
        ast.parse(text, filename=name)
    if source(parent["cells"][8]).count(STAGE_BRANCH) != 1:
        raise ValueError("Original exact b2 stage mount branch is required")
    compact_no_mount = source(parent["cells"][8]).replace(STAGE_BRANCH, "")
    author_normal = copy.deepcopy(author_compact)
    assignment(author_normal, "C_SLUG").value = ast.Constant(value="rsna-knee-r6-frozen-fe055406")
    if ast.dump(ast.parse(compact_no_mount), include_attributes=False) != ast.dump(author_normal, include_attributes=False):
        raise ValueError("Author Compact code differs beyond its mirror dataset slug")
    fixed_vector = ast.literal_eval(assignment(ast.parse(source(author["cells"][8])), "W_RAPTOR_C").value)
    changed = replace_weights(source(parent["cells"][10]), fixed_vector)
    if ast.dump(ast.parse(changed), include_attributes=False) != ast.dump(ast.parse(source(author["cells"][8])), include_attributes=False):
        raise ValueError("Candidate blend AST does not reproduce the public scored blend")
    for own_i, author_i in ((4, 2), (5, 3), (6, 4), (7, 5), (9, 7)):
        if source(parent["cells"][own_i]) != source(author["cells"][author_i]):
            raise ValueError("Unchanged public/P029M code cell differs")
    candidate = copy.deepcopy(parent)
    candidate["cells"][10]["source"] = changed
    notice = ("# P033: public .958 per-finding weights on successful P029M\n\n"
              "Parent-only control: replace only the W_RAPTOR_C dictionary. Raptor weights: "
              "Synovitis 0.80, Lateral OA 0.20, PF OA 0.60, Baker's 0.35; all other findings 0.50. "
              "R6 V1 / original nine-member b2 guard / TTA ON / A224=0.35 / failure and rank policy remain fixed. "
              "Author source358751289 V1 scored .958; this private candidate has no own score yet. "
              "No F006 is included. Historical parent descriptions below refer to the previous .50 vector. "
              "Upstream Gold58 selection exposure is inherited; no independent Gold-clean claim.\n")
    candidate["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": notice})
    for item in candidate["cells"]:
        item["metadata"] = {}
        if item["cell_type"] == "code":
            item.update(execution_count=None, outputs=[])
            code_ast(source(item), "P033-cell")
    candidate["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}
    serialized = (json.dumps(candidate, ensure_ascii=False, indent=1) + "\n").encode()
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": serialized.decode(),
               "language": "python", "kernel_type": "notebook", "dataset_data_sources": DATASETS,
               "kernel_data_sources": [STAGE], "competition_data_sources": ["rsna-knee-abnormality-detection"],
               "is_private": True, "enable_gpu": True, "enable_internet": False,
               "machine_shape": metadata["machineShape"], "docker_image": DOCKER,
               "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    manifest = {"schema_version": "p033_public958_vector_parent_only_offline_v1",
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "offline_prepared_no_Save_or_execution", "source_sha256": digest(serialized),
                "source_bytes": len(serialized), "parent_source_sha256": digest(parent_raw),
                "parent_source_official_submission": own_score, "parent_source_preserved": True,
                "author_source_sha256": digest(author_raw), "author_scored_source": 358751289,
                "author_version": 1, "author_submission": view["submission"],
                "candidate_own_Public": None, "only_scientific_change": "W_RAPTOR_C literal dictionary",
                "parent_weights": PARENT_WEIGHTS, "candidate_weights": PUBLIC_WEIGHTS,
                "TTA": True, "A224": 0.35, "F006_included": False,
                "guard_source_sha256": GUARD_SHA, "fixed_R6_guard": guard_values,
                "fixed_R6_manifest_sha256": digest(r6_raw), "fixed_R6_bundle_id": r6["bundle_id"],
                "fixed_stage_source_sha256": STAGE_SOURCE_SHA,
                "author_w1_dataset_replaced_by_existing_identical_manifest_b2_stage": True,
                "model_member_actual_content_verification_this_preparation": False,
                "same_P029M_code_cells_except_final": list(range(7)),
                "public_author_same_5_of_7_scientific_cells": [2, 3, 4, 5, 7],
                "public_author_compact_AST_equal_after_mount_and_dataset_name_normalization": True,
                "public_author_blend_AST_equal": True,
                "embedded_modules": [{"name": name, "sha256": digest(text.encode()), "same_parent_author": True}
                                     for name, text in sorted(modules.items())],
                "candidate_code_sha256": [digest(source(c).encode()) for c in candidate["cells"] if c["cell_type"] == "code"],
                "fixed_inputs": {"datasets": DATASETS, "kernel": [STAGE], "competition": payload["competition_data_sources"]},
                "runtime": {key: payload[key] for key in ("is_private", "enable_gpu", "enable_internet", "machine_shape", "docker_image")},
                "historical_outputs_and_execution_counts_cleared": True,
                "qualifications": {"static_source": True, "real_MRI_forward": False, "visible_replay": False,
                                   "hidden_completion": False, "own_Public": False, "actual_remote_Input_bindings": False},
                "gold_note": "Inherited upstream Gold58 selection. Vector choice provenance not independently established. Own Gold fit/select0.",
                "actions": {"network": 0, "authentication": 0, "model_load": 0, "MRI": 0, "training": 0, "Save": 0, "Submit": 0},
                "builder_sha256": digest(Path(__file__).read_bytes())}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(serialized)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "save-payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    light = {key: value for key, value in payload.items() if key != "text"}
    light.update(source_sha256=digest(serialized), source_file="source.ipynb")
    (output / "fixed-input-payload.json").write_text(json.dumps(light, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if digest((recovery / "P029M-source.ipynb").read_bytes()) != PARENT_SHA:
        raise ValueError("Successful parent changed during preparation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args()
    result = prepare(args.output_dir, slug=args.slug)
    print(json.dumps({"output": str(args.output_dir), "source_sha256": result["source_sha256"],
                      "stage": result["stage"], "own_Public": None}))


if __name__ == "__main__":
    main()
