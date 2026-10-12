"""Prepare one physical CoAt crop140->160 factor; no MRI/weights or remote execution."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p009_control", ROOT / "scripts/build_p006_convnext_density.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
BASELINE = control.BASELINE
OUTPUT_DIR = ROOT / "artifacts/kaggle/p009-coat-fov160-v1-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/19_submit_p009_coat_fov160.ipynb"
AUDIT_PATH = ROOT / "experiments/p009-coat-fov160-implementation-20261008.json"
FASTREAD_BASELINE_SHA = "5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0"
CROP_ORIGINAL, CROP_CANDIDATE = "CROP_MM = 140.0", "CROP_MM = 160.0"
PREFACE = """p009：自己Public0.950のp005を基準に、CoAtNetの物理crop目標だけ140→160mmへ変更します。
モデル入力320、native canvas384、K94、SWA、左右鏡映、ConvNeXt k16/3fold、12所見の融合係数は固定です。
outerと埋込fastreadの両constantを変更し、実ロードsourceとmoduleの一致を検査します。
名目上の中央320の視野は116.7→133.3mmですが、native画像の範囲・spacing・roundingで変わります。
精度改善は未測定の仮説です。元の帰属・利用条件は以下に保持します。

---

"""


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fastread(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "_FASTREAD_B64"]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one original embedded fastread source")
    return nodes[0], base64.b64decode(ast.literal_eval(nodes[0].value), validate=True).decode("utf-8")


def modified_main(source):
    node, embedded = fastread(source)
    if sha_text(embedded) != FASTREAD_BASELINE_SHA or embedded.count(CROP_ORIGINAL) != 1:
        raise ValueError("Original fastread source/crop differs")
    if source.count(CROP_ORIGINAL) != 1:
        raise ValueError("Original outer crop differs")
    altered = embedded.replace(CROP_ORIGINAL, CROP_CANDIDATE, 1)
    old_literal = ast.get_source_segment(source, node.value)
    encoded = base64.b64encode(altered.encode("utf-8")).decode("ascii")
    source = source.replace(old_literal, repr(encoded), 1).replace(CROP_ORIGINAL, CROP_CANDIDATE, 1)
    tree = ast.parse(source)
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_fastread")
    original_loader = ast.get_source_segment(source, loader)
    guard = """    if os.path.exists(p):
        with open(p, "rb") as stream:
            assert stream.read() == _b.b64decode(_FASTREAD_B64), "p009 stale fastread source"
"""
    if original_loader.count("    if not os.path.exists(p):") != 1 or original_loader.count("    return m") != 1:
        raise ValueError("Original fastread loader structure differs")
    new_loader = original_loader.replace("    if not os.path.exists(p):", guard + "    if not os.path.exists(p):", 1)
    new_loader = new_loader.replace("    return m", '    assert m.CROP_MM == CROP_MM, "p009 loaded module FOV differs"\n    return m', 1)
    source = source.replace(original_loader, new_loader, 1)
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    original_main = ast.get_source_segment(source, main)
    if original_main.count("    import pandas as pd\n") != 1:
        raise ValueError("Original main entry differs")
    # Write/verify source once in the parent before forking. Stale source fails before Pool respawn.
    new_main = original_main.replace("    import pandas as pd\n", "    import pandas as pd\n    _load_fastread()\n", 1)
    return source.replace(original_main, new_main, 1)


def verify_change(baseline, candidate):
    before, after = control.code_sources(baseline), control.code_sources(candidate)
    if list(before) != list(after) or [index for index in before if before[index] != after[index]] != [4]:
        raise ValueError("Only the CoAt main cell may change")
    if after[4] != modified_main(before[4]) or candidate["metadata"] != baseline["metadata"]:
        raise ValueError("Undeclared source/metadata change")
    _, old_embedded = fastread(before[4])
    _, new_embedded = fastread(after[4])
    if new_embedded != old_embedded.replace(CROP_ORIGINAL, CROP_CANDIDATE, 1):
        raise ValueError("Embedded fastread may change only physical CROP_MM")
    control.verify_runtime_binding(candidate, 16)
    return {"changed_code_cells": [4], "six_other_code_cells_byte_identical": True,
            "embedded_change_only_CROP_MM140_to160": True, "ConvNeXt_k": 16,
            "extra_runtime_binding": ["stale fastread file rejection", "actual imported FOV assertion",
                                      "parent loader preflight before fork workers"]}


def prepare_notebook(baseline):
    candidate = copy.deepcopy(baseline)
    candidate["cells"][4]["source"] = modified_main(control.cell_source(baseline, 4)).splitlines(keepends=True)
    candidate["cells"][0]["source"] = (PREFACE + control.cell_source(baseline, 0)).splitlines(keepends=True)
    for cell in candidate["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    verify_change(baseline, candidate)
    return candidate


def prepare_package(output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH, audit_path=AUDIT_PATH):
    output_dir, notebook_path, audit_path = map(Path, (output_dir, notebook_path, audit_path))
    if any(path.exists() for path in (output_dir, notebook_path, audit_path)):
        raise FileExistsError("Use new package/Notebook/audit paths; preserve earlier evidence")
    if control.digest(BASELINE) != control.BASELINE_SHA256:
        raise ValueError("Protected14 SHA changed")
    baseline = control.read_json(BASELINE)
    candidate = prepare_notebook(baseline)
    syntax = control.syntax_check(candidate)
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    control.write_json(notebook_path, candidate)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(control.CONTROL_PACKAGE / name, output_dir / name)
    _, candidate_embedded = fastread(control.cell_source(candidate, 4))
    receipt = {
        "schema_version": "p009_coat_physical_FOV160_candidate_v1", "prepared_at_jst": "2026-10-08",
        "status": "prepared_not_executed_accuracy_unmeasured",
        "baseline_sha256": control.BASELINE_SHA256, "baseline_own_public_lb": "0.950",
        "candidate_sha256": control.digest(notebook_path), "candidate_own_public_lb": None,
        "factor": "CoAt physical crop target140->160mm, including the actual embedded module constant",
        "code_difference": verify_change(baseline, candidate),
        "original_fastread_sha256": FASTREAD_BASELINE_SHA,
        "candidate_fastread_sha256": sha_text(candidate_embedded),
        "model_shapes_and_loading": "Unchanged INPUT_RES320, IMG384, K94, architecture functions and strict checkpoint loading; actual checkpoint not obtained or loaded.",
        "unchanged": ["SWA/checkpoint bytes", "CoAt two-view mean", "ConvNeXt k16/3fold", "LUT/global percentile/series/slice policies", "rank fusion12weights", "same3Inputs/Originalcontainer/T4/InternetOFF"],
        "nominal_final_FOV_mm": {"original": 140 * 320 / 384, "candidate": 160 * 320 / 384},
        "limits": ["Native crop clipping/rounding/scalar spacing limit the nominal FOV formula.",
                   "Mask and thus evaluated centres may change as consequences of physical crop.",
                   "Anatomy becomes12.5% smaller in the fixed320grid; extra context may help or lose detail.",
                   "No training augmentation/corpus distribution independently established; improvement unmeasured."],
        "syntax_checks": syntax, "real_MRI_weights_fit_remote_or_submission_performed": False,
    }
    control.write_json(audit_path, receipt)
    control.write_json(output_dir / "provenance.json", receipt)
    control.write_json(output_dir / "package.json", {
        "schema_version": "p009_coat_fov160_package_v1",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": control.digest(path)}
                  for path in sorted(output_dir.iterdir()) if path.is_file()},
    })
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    result = prepare_package(args.output_dir, args.notebook, args.audit)
    print({"candidate_sha256": result["candidate_sha256"], "status": result["status"]})
