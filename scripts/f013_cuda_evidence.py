"""Consume the exact actual local CUDA proof; no model, API or runtime work."""
from __future__ import annotations

import hashlib
import json
import math

ROOT_PROOF_SHA = "da0f9e51b2ea8afee8188b5eb333528c653da5c56ce8dedb1e839174b690dd68"
LOCAL_SOURCE_SHA = "9fa7baa0ee224396294a944edef319a6888e128a1daff8f0fe842fba382b9d43"
LOCAL_POLICY_SHA = "5da3be045253c1d52582d42da669fa24b31767924c78b19b8581e33334fd57f4"
CHECKPOINTS = ("16ff8220be0155779587f48705b161f2b789c2a1425367b94544a9daa21f2e63",
               "6805bfa94b1356b71d474d6dd918518749a42dd08ed791d403aa4c256c3e4d78")
GPU = "NVIDIA GeForce RTX 4090"


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_prior_cuda(raw):
    require(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == ROOT_PROOF_SHA, "Exact actual local CUDA root proof required")
    proof = json.loads(raw)
    require(proof["schema_version"] == "F013_local_RTX4090_actual_artificial_CUDA_root_evidence_v1"
            and proof["all_pass"] is True and len(proof["checks"]) == 19
            and all(v is True for v in proof["checks"].values()), "All actual root checks")
    value = proof["local_result"]
    require(proof["source_SHA"] == value["source_SHA"] == LOCAL_SOURCE_SHA
            and value["policy_SHA"] == LOCAL_POLICY_SHA
            and value["original_GPU_helper_SHA"] == "d55b4cf4bd3aec386d10251add5be29693b3be3eb0e7bf8a79f663dfcd6b3d75"
            and value["only_one_hardware_guard_constant_changed_inverse_AST_exact"] is True,
            "Exact predeclared original CUDA checks with single hardware guard adapter")
    require(proof["GPU"] == value["GPU"] == GPU and value["devices"] == 1
            and proof["kernelId"] is None and proof["saved_version"] is None, "Actual local GPU proof, never a fictitious Kaggle version")
    require(value["complete"] is True and value["all_181_staged_members_preserved"] is True
            and value["original_F009_HOLD_unchanged"] is True
            and value["MRI_Report_Gold_training_API_Save_Submit_upload"] == 0, "Actual finite source/weight scope")
    require(len(value["folds"]) == 2, "Both fixed folds")
    for fold, item in enumerate(value["folds"]):
        require(item["complete"] is True and item["fold"] == fold and item["epoch"] == 12
                and item["checkpoint_SHA"] == CHECKPOINTS[fold] and item["GPU"] == GPU
                and item["class"] == "source14.finetune_model.NativeTripletMIL", "Exact fixed E12/class/device")
        require(all(item[key] is True for key in ("full_native392_forward_finite12", "all3planes",
                    "padding_pixel_0_vs255_logits_bitexact", "allmissing_rejected_beforeencoder",
                    "evalgamma_and_CPU_CUDA_RNG_unchanged", "suffix_head_prefix_unchanged"))
                and item["valid_triplets"] == 4, "Full native actual CUDA contracts")
        error = item["CPUmanual_FP32_reference_probability_maxabs"]
        require(item["fixed_probability_tolerance"] == .005 and math.isfinite(error) and 0 <= error <= .005
                and item["frozen_prefix_SHA"] == "67a05dff5f621b78dc8b16cee95a56aea314437806637950ad8336a4ab522347"
                and item["MRI_Report_Gold_labels_training_reads"] == 0, "Same fixed numerical tolerance and preserved prefix")
    require(value["future_actual_current_8_models_twoT4_GPU_QA_still_required_before_nativeMRI"] is True
            and value["Kaggle_T4_realMRI_fusion_or_Public_qualified"] is False, "Retain actual current two-T4 serving qualification")
    return {"GPU": GPU, "kind": "actual_local_RTX4090", "source_SHA": LOCAL_SOURCE_SHA,
            "root_proof_SHA": ROOT_PROOF_SHA, "current_twoT4_GPU_gate_still_required": True}
