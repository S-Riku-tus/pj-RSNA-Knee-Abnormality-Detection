"""Prepare an Apex k16 ConvNeXt anatomical-mirror candidate; never uploads or runs inference."""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p006_control", ROOT / "scripts/build_p006_convnext_density.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
BASELINE = control.BASELINE
BASELINE_SHA256 = control.BASELINE_SHA256
OUTPUT_DIR = ROOT / "artifacts/kaggle/p007-convnext-mirror-v2-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/16_submit_p007_convnext_mirror.ipynb"
AUDIT_PATH = ROOT / "experiments/p007-convnext-mirror-implementation-v2-20261008.json"
RUN_SIGNATURE = control.ORIGINAL_SIGNATURE
RUN_DECORATED_SIGNATURE = "@torch.no_grad()\n" + RUN_SIGNATURE
PLAIN_PREDICTION = (
    "            p = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)"
)
MIRROR_PREDICTION = """            p_plain = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)
            xm, maskm, posm = anatomical_lr_mirror(x, mask, pos)
            p_mirror = torch.stack([m(xm, maskm, posm, res=res).float().sigmoid() for m, res in models]).mean(0)
            p = 0.5 * (p_plain + p_mirror)
            del xm, maskm, posm"""
MIRROR_HELPER = '''def anatomical_lr_mirror(x, mask, pos):
    """Mirror sampled canonical uint8 tokens and sagittal coordinates; preserve absent slots.

    ConvNeXt slots are plane + 3*(1-fat_suppression): sagittal0/3, coronal1/4, axial2/5.
    Sagittal left/right is slice depth: reverse local channels and reflect centre position.
    Coronal/axial left/right is image width. Apply before ImageNet channel normalization.
    Window order is retained; each token carries its reflected explicit position.
    """
    assert x.ndim == 6 and x.shape[1] == 6 and x.shape[3] == 3
    assert x.dtype == torch.uint8 and mask.dtype == torch.bool
    assert mask.shape == x.shape[:2] and pos.shape == x.shape[:3]
    assert x.device == mask.device == pos.device
    mirrored, mirrored_pos = x.clone(), pos.clone()
    for slot in (0, 3):
        valid = mask[:, slot].view(-1, 1, 1, 1, 1)
        mirrored[:, slot] = torch.where(valid, x[:, slot].flip(2), x[:, slot])
        mirrored_pos[:, slot] = torch.where(mask[:, slot, None], 1.0 - pos[:, slot], pos[:, slot])
    for slot in (1, 2, 4, 5):
        valid = mask[:, slot].view(-1, 1, 1, 1, 1)
        mirrored[:, slot] = torch.where(valid, x[:, slot].flip(-1), x[:, slot])
    return mirrored, mask, mirrored_pos


'''
PREFACE = """p007: 自己Public0.950のp005を固定し、ConvNeXtだけplain/解剖学的左右鏡映の2view等率平均に変更。
baselineのk16を維持します。p006のk24とは独立した候補で、同時変更しません。
Sagittal slots0/3は隣接3sliceチャンネル逆順とvalid centre位置p→1-p、Coronal/Axialは幅反転。
slot・mask・所見名・Fat Suppressionは保持。鏡映はuint8に適用し、明暗を反転しません。
元window集合の位置座標を鏡映する仕様で、native volume反転後に窓を再抽出する仕様ではありません。
CoAtNet主系・全checkpoint・前処理・batch4・workers4・12係数・fallbackは同一です。
ConvNeXt公開augmentation経路には鏡映がなく、実checkpointの訓練全コードは未取得。
Public改善・隠し完走・GPU時間/メモリは未測定です。元作者の帰属・利用条件を以下に保持します。

---

"""


def modified_reader(source):
    if not source.startswith(control.INFER_MAGIC + "\n"):
        raise ValueError("Fixed generated reader path differs")
    if source.count(RUN_DECORATED_SIGNATURE) != 1 or source.count(PLAIN_PREDICTION) != 1:
        raise ValueError("Fixed k16 signature or original probability-mean block differs")
    if "def anatomical_lr_mirror(" in source:
        raise ValueError("Baseline already contains another mirror implementation")
    return source.replace(RUN_DECORATED_SIGNATURE, MIRROR_HELPER + RUN_DECORATED_SIGNATURE, 1).replace(
        PLAIN_PREDICTION, MIRROR_PREDICTION, 1
    )


def verify_mirror_change(baseline, candidate):
    before, after = control.code_sources(baseline), control.code_sources(candidate)
    if list(before) != list(after):
        raise ValueError("Apex code-cell layout changed")
    changed = [index for index in before if before[index] != after[index]]
    if changed != [8] or after[8] != modified_reader(before[8]):
        raise ValueError("Only the declared ConvNeXt mirror helper/prediction block may change")
    if candidate["metadata"] != baseline["metadata"]:
        raise ValueError("Notebook metadata changed")
    for index in (1, 2):
        if control.cell_source(candidate, index) != control.cell_source(baseline, index):
            raise ValueError("Original acknowledgement markdown changed")
    control.verify_runtime_binding(candidate, 16)
    tree = ast.parse(control.writefile_body(after[8]))
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "anatomical_lr_mirror"]
    if len(helpers) != 1:
        raise ValueError("Expected exactly one declared mirror helper")
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    if [ast.unparse(node) for node in run.decorator_list] != ["torch.no_grad()"]:
        raise ValueError("The original reader run no_grad decorator must stay attached to run")
    return {"changed_code_cells": [8], "other_code_cells_byte_identical": 6, "effective_k": 16,
            "original_run_no_grad_decorator_preserved": True}


def prepare_notebook(baseline):
    control.verify_runtime_binding(baseline, 16)
    candidate = copy.deepcopy(baseline)
    candidate["cells"][8]["source"] = modified_reader(control.cell_source(baseline, 8)).splitlines(keepends=True)
    candidate["cells"][0]["source"] = (PREFACE + control.cell_source(baseline, 0)).splitlines(keepends=True)
    for cell in candidate["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    verify_mirror_change(baseline, candidate)
    return candidate


def prepare_package(
    baseline_path=BASELINE, output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH, audit_path=AUDIT_PATH,
):
    baseline_path, output_dir, notebook_path, audit_path = map(
        Path, (baseline_path, output_dir, notebook_path, audit_path)
    )
    if any(path.exists() for path in (output_dir, notebook_path, audit_path)):
        raise FileExistsError("Use new package/Notebook/audit paths; existing runs are never overwritten")
    if control.digest(baseline_path) != BASELINE_SHA256:
        raise ValueError("Fixed p005 SHA256 differs")
    baseline = control.read_json(baseline_path)
    candidate = prepare_notebook(baseline)
    syntax = control.syntax_check(candidate)
    metadata = control.read_json(control.CONTROL_PACKAGE / "original-metadata.json")
    inputs = control.read_json(control.CONTROL_PACKAGE / "inputs-manual.json")
    if (
        metadata["enableInternet"] or not metadata["enableGpu"]
        or not metadata["dockerImage"].endswith(
            "@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
        )
        or [row["sourceId"] for row in inputs["inputs"]] != [154281, 20325538, 20440357]
    ):
        raise ValueError("Fixed p005 environment or three Input snapshots differ")
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    control.write_json(notebook_path, candidate)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    shutil.copyfile(baseline_path, output_dir / "baseline-p005.ipynb")
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(control.CONTROL_PACKAGE / name, output_dir / name)
    audit = {
        "schema_version": "p007_convnext_anatomical_mirror_v2", "prepared_at_jst": "2026-10-08",
        "status": "prepared_single_factor_not_executed",
        "baseline_sha256": BASELINE_SHA256, "baseline_own_public_lb": 0.950,
        "candidate_sha256": control.digest(notebook_path), "candidate_own_public_lb": None,
        "code_difference": verify_mirror_change(baseline, candidate),
        "runtime_binding": control.verify_runtime_binding(candidate, 16),
        "mirror_contract": {
            "input": "uint8 B,S=6,K=16,C=3,H,W before make_views normalization",
            "sagittal_slots": [0, 3], "coronal_slots": [1, 4], "axial_slots": [2, 5],
            "sagittal_valid_only": "local channel reverse and pos->1-pos",
            "coronal_axial_valid_only": "width reverse; position unchanged",
            "mask_slots_labels_fat_suppression_unchanged": True,
            "missing_slot_pixels_and_positions_unchanged": True,
            "no_intensity_inversion_or_label_swap": True,
            "window_order_retained": True,
            "view_probability_average": [0.5, 0.5],
            "model_probability_average": "original mean over the same3 fold checkpoints per view",
        },
        "coordinate_semantics": {
            "name": "sampled-token-coordinate left/right mirror",
            "window_permutation": "Reversing sagittal K order together with position merely permutes tokens. The eval Transformer/Attention is permutation equivariant/invariant apart from floating reduction order, so sorting is not required.",
            "native_grid_contract": "Artificial audit confirms c[::-1]==native_depth-1-c for released window_centres and every native depth3..64 at k16/k24. Thus sagittal token reversal is consistent with native mirrored-grid centres in the qualified case. Other grids can differ at banker's half ties (depth4/k3 counterexample). Position reflection retains float32 rounding tolerance.",
            "channel_normalization": "ImageNet mean/std differ between channels. Reverse uint8 slice channels before normalization; reversing normalized channels is not equivalent.",
        },
        "released_training_augmentation_audit": {
            "source": "p005 embedded knee.py StudyDataset/make_views",
            "observed": ["random source-series choice", "stratified random window centres", "series-shared rotation", "scale", "translation", "gamma", "gain"],
            "mirror_in_released_path": False,
            "actual_train_py_and_all_training_invocations_downloaded": False,
            "conclusion": "Mirror is a test-time hypothesis, not demonstrated training-supported augmentation.",
        },
        "unchanged": ["main CoAt branch and all checkpoint bytes", "ConvNeXt k16/series/reader/res/batch4/workers4", "rank fusion12 coefficients", "original decoder/fallback", "fixed3Inputs/Original container/Internet OFF"],
        "syntax_checks": syntax,
        "cost": "ConvNeXt forward approximately2x plus one cloned uint8 batch. Total wall time and peak VRAM unmeasured; mirrored batch deleted before next batch.",
        "accuracy_and_independence": "Improvement unmeasured. Label-free diagnostics cannot establish AUC; public checkpoint exposure prevents true retrospective OOF; own gold is never fit/selection.",
        "no_real_MRI_weights_training_remote_or_submission_in_builder": True,
        "supersedes_unexecuted_v1": {
            "notebook_sha256": "e5a9f4692fb3206b18a53a03ec10d14b8cd2e5bb287d843f965838a5a9c924bd",
            "package": "artifacts/kaggle/p007-convnext-mirror-v1-20261008",
            "reason": "V1 helper insertion moved @torch.no_grad from run to the helper. V1 is invalidated before real execution; V2 keeps the decorator on run.",
            "prior_artificial_tests_limitation": "Earlier tests used inference_mode around model forwards and did not directly call the production run with gradients initially enabled.",
        },
    }
    control.write_json(audit_path, audit)
    control.write_json(output_dir / "provenance.json", audit)
    control.write_json(output_dir / "package.json", {
        "schema_version": "p007_convnext_mirror_package_v2",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": control.digest(p)}
                  for p in sorted(output_dir.iterdir()) if p.is_file()},
    })
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    audit = prepare_package(args.baseline, args.output_dir, args.notebook, args.audit)
    print(json.dumps({"candidate_sha256": audit["candidate_sha256"], "status": audit["status"]}))


if __name__ == "__main__":
    main()
