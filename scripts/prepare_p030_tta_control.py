"""Prepare an unexecuted R6 TTA-off control from the immutable P029 notebook."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P029 = ROOT / "artifacts/kaggle/p029-public957-r6-v1-20261015"
OUTPUT = ROOT / "artifacts/kaggle/p030-r6-tta-off-control-v1-20261015"
SOURCE_SHA = "449ce7786021dc17fa671783d86d6f5ae4e10638f013c93085eadaba847cbd02"
P029_MANIFEST_SHA = "3acd5dbdbbb04320d7708a0256746457500ab03a9fb2878c0c5d9ca1ef9a1cf5"
R6_MANIFEST_SHA = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
R6_BUNDLE_ID = "fe055406fad6a12f"
R6_MANIFEST_EVIDENCE = ROOT / "artifacts/research/20261015-r6-v1-manifest-only-v1/manifest.json"
EXPECTED_INPUT_IDS = [154281, 20297367, 20440357, 20611930]
MODULE_NAMES = {"px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py",
                "probe_dual.py", "knee_anchor_bank_v2.py", "side_common.py"}
WRITEFILE_HEADERS = {"%%writefile /tmp/raptor_run.py", "%%writefile /tmp/raptor224_run.py"}
NOTICE = (
    "# P030: R6のflip TTAだけをOFFにする未実行の対照\n\n"
    "親P029のコードからFLIP_TTA=Trueの一箇所だけをFalseへ変更しました。"
    "同じR6 V1、2つのRaptor、固定係数、前処理と失敗処理を維持します。"
    "以下の原作者説明は親のflip TTA ON設定についての履歴です。"
    "作者Public 0.957はこの候補や自己アカウントの実測値ではありません。"
    "旧出力・実行番号・実行時メタデータは除去しました。このPCでは実行しません。\n\n---\n\n"
)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def source_text(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def flip_assignment(tree, expected):
    stores = [node for node in ast.walk(tree)
              if isinstance(node, ast.Name) and node.id == "FLIP_TTA" and isinstance(node.ctx, ast.Store)]
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "FLIP_TTA"]
    if len(stores) != 1 or len(assignments) != 1:
        raise ValueError("Exactly one top-level FLIP_TTA assignment is required")
    node = assignments[0]
    if not isinstance(node.value, ast.Constant) or node.value.value is not expected:
        raise ValueError(f"FLIP_TTA must be the literal {expected}")
    return node


def assert_boolean_only_change(before, after):
    original, candidate = ast.parse(before), ast.parse(after)
    assignment = flip_assignment(original, True)
    flip_assignment(candidate, False)
    assignment.value.value = False
    if ast.dump(original, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        raise ValueError("AST changed beyond the one FLIP_TTA boolean")


def disable_flip(source):
    tree = ast.parse(source)
    value = flip_assignment(tree, True).value
    if value.lineno != value.end_lineno:
        raise ValueError("Unexpected multiline boolean token")
    lines = source.splitlines(keepends=True)
    line = lines[value.lineno - 1].encode("utf-8")
    start, end = value.col_offset, value.end_col_offset
    if line[start:end] != b"True":
        raise ValueError("Expected the exact True token at the AST location")
    lines[value.lineno - 1] = (line[:start] + b"False" + line[end:]).decode("utf-8")
    result = "".join(lines)
    assert_boolean_only_change(source, result)
    return result, {"cell": 6, "line": value.lineno, "before": "True", "after": "False",
                    "AST_difference_is_single_boolean": True,
                    "other_source_bytes_on_all_lines_preserved": True}


def embedded_sources(source):
    tree = ast.parse(source)
    nodes = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError("Exactly one literal embedded SOURCES dictionary is required")
    modules = ast.literal_eval(nodes[0])
    if not isinstance(modules, dict) or set(modules) != MODULE_NAMES:
        raise ValueError("Embedded module membership differs")
    for name, text in modules.items():
        if not isinstance(text, str):
            raise ValueError("Embedded modules must be source text")
        compile(text, f"p030_embedded_{name}", "exec")
    return modules


def compile_cell(source, index):
    body = source
    if body.startswith("%%writefile "):
        header, body = body.split("\n", 1)
        if header not in WRITEFILE_HEADERS:
            raise ValueError("Unexpected writefile destination")
    elif body.startswith("%%"):
        raise ValueError("Unexpected cell magic")
    compile(body, f"p030_cell_{index}", "exec")


def make_candidate(parent):
    if parent.get("nbformat") != 4 or [cell["cell_type"] for cell in parent["cells"]] != ["markdown"] * 2 + ["code"] * 7:
        raise ValueError("Expected the nine-cell P029 notebook")
    candidate = copy.deepcopy(parent)
    original = source_text(parent["cells"][6])
    changed, change = disable_flip(original)
    candidate["cells"][6]["source"] = changed if isinstance(parent["cells"][6]["source"], str) else changed.splitlines(keepends=True)
    candidate["cells"][0]["source"] = (NOTICE + source_text(parent["cells"][0])).splitlines(keepends=True)
    removed = []
    for index, cell in enumerate(candidate["cells"]):
        metadata = cell.get("metadata", {})
        for key in ("execution", "papermill"):
            if key in metadata:
                metadata.pop(key)
                removed.append(f"cells[{index}].metadata.{key}")
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            before, after = source_text(parent["cells"][index]), source_text(cell)
            if index != 6 and before != after:
                raise ValueError("Another code cell changed")
            compile_cell(after, index)
    for key in ("execution", "papermill"):
        if key in candidate.get("metadata", {}):
            candidate["metadata"].pop(key)
            removed.append(f"metadata.{key}")
    before_modules, after_modules = embedded_sources(original), embedded_sources(changed)
    if before_modules != after_modules:
        raise ValueError("Embedded module bytes changed")
    if source_text(parent["cells"][1]) != source_text(candidate["cells"][1]):
        raise ValueError("Author credits changed")
    facts = {
        "only_scientific_change": change,
        "same_other_code_cells": [2, 3, 4, 5, 7, 8],
        "code_cells": [
            {"cell": index, "parent_source_sha256": digest(source_text(parent["cells"][index]).encode("utf-8")),
             "candidate_source_sha256": digest(source_text(candidate["cells"][index]).encode("utf-8")),
             "source_same": source_text(parent["cells"][index]) == source_text(candidate["cells"][index])}
            for index in range(2, 9)
        ],
        "embedded_modules": [{"name": name, "sha256": digest(text.encode("utf-8")), "same_bytes": True}
                             for name, text in sorted(before_modules.items())],
        "author_credits_unchanged": True,
        "historical_outputs_cleared": True,
        "all_code_execution_counts_cleared": True,
        "historical_execution_metadata_removed": removed,
        "candidate_notice_added_to_first_markdown_only": True,
        "all_code_and_embedded_modules_compiled_without_execution": True,
    }
    return candidate, facts


def pinned_json(path, expected):
    body = Path(path).read_bytes()
    if digest(body) != expected:
        raise ValueError(f"Pinned parent SHA mismatch: {Path(path).name}")
    return body, json.loads(body)


def prepare(output_dir=OUTPUT, *, parent_dir=P029, r6_manifest_path=R6_MANIFEST_EVIDENCE):
    output, parent_dir = Path(output_dir), Path(parent_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new P030 directory; preserve previous results")
    parent_bytes, parent = pinned_json(parent_dir / "source.ipynb", SOURCE_SHA)
    manifest_bytes, parent_manifest = pinned_json(parent_dir / "manifest.json", P029_MANIFEST_SHA)
    fixed = parent_manifest["fixed_inputs"]
    if ([item["source_id"] for item in fixed] != EXPECTED_INPUT_IDS
            or fixed[-1].get("version") != 1
            or fixed[-1].get("expected_manifest_sha256") != R6_MANIFEST_SHA
            or parent_manifest["version_hazard"]["required_bundle_id"] != R6_BUNDLE_ID):
        raise ValueError("The P030 control must retain the exact R6 V1 Input")
    r6_bytes, r6_metadata = pinned_json(r6_manifest_path, R6_MANIFEST_SHA)
    if r6_metadata.get("bundle_id") != R6_BUNDLE_ID or bool(r6_metadata.get("flip_tta")):
        raise ValueError("Frozen R6 metadata must not enable flip TTA")
    candidate, checks = make_candidate(parent)
    source = (json.dumps(candidate, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
    manifest = {
        "schema_version": "p030_R6_TTAoff_single_boolean_control_v1",
        "prepared_at_UTC": datetime.now(timezone.utc).isoformat(),
        "stage": "local_unexecuted_source_preparation_only",
        "parent": {"directory": str(parent_dir), "source_bytes": len(parent_bytes), "source_sha256": SOURCE_SHA,
                   "manifest_bytes": len(manifest_bytes), "manifest_sha256": P029_MANIFEST_SHA,
                   "author_Public": "0.957", "own_parent_Public_confirmed_here": False,
                   "parent_evidence_is_not_candidate_score": True},
        "source": {"file": "source.ipynb", "bytes": len(source), "sha256": digest(source),
                   "cells": 9, "code_cells": 7, "unexecuted": True,
                   "original_RUN_INFERENCE_true_retained_but_not_executed": True},
        "fixed_inputs": fixed,
        "runtime_plan": parent_manifest["runtime_plan"],
        "frozen_input_preflight": {"reuse_from_P029": str(parent_dir / "verify_frozen_input.py"),
                                   "expected_manifest_sha256": R6_MANIFEST_SHA, "expected_bundle_id": R6_BUNDLE_ID,
                                   "requires_R6_V1_internal_version": 20611930},
        "local_R6_manifest_metadata_evidence": {
            "path": str(r6_manifest_path), "bytes": len(r6_bytes), "sha256": digest(r6_bytes),
            "bundle_id_verified": True, "flip_tta_key_present": "flip_tta" in r6_metadata,
            "flip_tta_value": r6_metadata.get("flip_tta"),
            "effective_source_flag_with_this_metadata": False,
            "mounted_manifest_or_model_content_verified": False,
        },
        "scientific_change": {"factor": "Compact flip TTA only", "P029_FLIP_TTA": True, "P030_FLIP_TTA": False,
                              "bundle": R6_BUNDLE_ID, "Raptor384_Raptor224_unchanged": True,
                              "outer_weights": {"Raptor": 0.5, "R6": 0.5}, "Raptor224_share": 0.35,
                              "failure_policy_and_rank_basis_unchanged": True, "coefficient_search": False,
                              "effective_TTA_still_requires_mounted_manifest_check": "Source uses manifest.get('flip_tta') or FLIP_TTA; require the frozen R6 V1 manifest to omit or set flip_tta=false before calling this a TTA-off run"},
        "static_checks": checks,
        "qualification": {"source_prepared": True, "actual_mounted_manifest_or_weights_verified": False,
                          "effective_runtime_TTA_off_verified": False, "own_GPU_MRI_inference": False,
                          "visible_Serving_qualified": False, "hidden_completion_or_9h_verified": False,
                          "own_Public_verified": False, "own_Public": None,
                          "Gold_read_fit_checkpoint_or_coefficient_selection": 0,
                          "network_download_upload_Save_Submit_or_training": False},
        "interpretation_ja": [
            "P029の自己再現を先に確認し、P030は同じR6 V1のTTAだけをOFFにする対照として比較する。作者0.957を候補の自己値へ転用しない。",
            "元の全科学コードを保ち、cell6 top-level FLIP_TTA=TrueのTrue tokenだけFalseへ置換した。説明用markdownと旧実行表示の除去以外に変更はない。",
            "生成Notebookの旧作者説明やコードコメントは親のTTA設定についての履歴として残す。先頭のP030説明とこのmanifestが今回の変更を示す。",
            "元コードはmanifest.get('flip_tta') OR FLIP_TTAで決定する。R6 V1 rawmanifest SHAとflip_ttaキーが欠損またはfalseであることをKaggle上で確認してからTTA-off比較と認定する。",
            "Inputの最新V2はR7でありP030では使わない。原2Raptor・係数・失敗処理・rankは固定。Internet OFF、T4x2、実Docker/Input設定を再確認する。",
            "P030の後、既存作者V2 R7/TTA-offと比較するとbundle差を切り分けられる。ただしR7V2のbestV1 Public0.957をV2のスコアと誤認しない。",
            "P029が持つpickle可能な公開重み読込、224のFP32 fallback、公開重みのGold選択歴、上流利用条件、8.3hがhardtimeoutでない限界は引き継ぐ。",
            "このPCで元Notebookは実行しない。手動実行と手動提出は別の段階で、今回ネットワーク・重み・MRI・学習・外部保存・提出は0。"
        ],
        "builder_sha256": digest(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "source.ipynb").open("xb") as stream:
        stream.write(source)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="New directory only")
    args = parser.parse_args(argv)
    result = prepare(args.output_dir)
    print(json.dumps({"stage": result["stage"], "output": str(args.output_dir),
                      "source_sha256": result["source"]["sha256"], "own_Public": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
