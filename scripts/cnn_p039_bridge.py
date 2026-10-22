"""Disabled CNN/P039 interface and source contracts, using the standard library.

This preview cannot construct a model, restore a package or enter inference.
Future runtime code needs a separate review after an actual CNN pair qualifies.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

ACTUAL_ENABLED = False
PREVIEW_POLICY = "artifacts/research/20261021-CNN-P039-disabled-preview-prospective-v1/policy.json"
PREVIEW_POLICY_SHA = "8edaa47d44ca058c7c62de8bc1389ed6e5343f6bc25a3677506d81667470937d"
WHEEL_LOCK = "artifacts/tools/cnn-Linux-wheels-20261021-v1/wheel-lock.json"
WHEEL_LOCK_SHA = "0939aa524c89de0091eb0f40b128cec394ba2409225800aece3c546703e12b21"
LINUX_NOTES = "artifacts/research/20261021-F014-Linux-serving-constraints-v1/notes.json"
LINUX_NOTES_SHA = "d2a30afbfd77c307839ded21cae2f3eaf3013d7472473bcb48a00ec7d949c442"
ROUTER_SHA = "6dfb95abb56055c04da11a697b9491a72e1bca774d2fdc36285f296b34c4f02c"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SOURCE16 = {
    "convnext_model.py": "5bb76633e5f9c7321f6eeaf5a32215c378647d5d0aec8da51e2fa072c4af6555",
    "convnext_training.py": "744d2737943eaa97793c41bb990116a383512600d2e7d93d0a933ac51fb3b4c2",
    "native_training_fast16.py": "9f5c3b490374201f72cc73ebf11a26f8a8163945387d4d493e7a04c08e729d57",
    "finetune_model.py": "c43bcedf1d97b713528b4157a0a9c8867741b785d58c5be084cd7072c2b2c624",
    "feature_model.py": "af85d8eaed6015ed2ddb7f0b2029eca446471876750ebde60bed403cf8ced0e1",
    "triplet_cache.py": "0b4c666407cb72d7ef3249f479af1e13cca1b4422b18f18cebf3eed2b768f26f",
    "feature_imaging.py": "de36a21929b482eee897b12e06bb9964cee59beab6a9bc2b2d44c15ace279546",
    "feature_contracts.py": "be76ecc28b14491f4d1a0a459794c7f94456e845ce0e46c46ced699bfaf54ed6",
    "imaging.py": "07a114888cde19ae0e5799f2de48c6751ec9c4b5637ec1f4f4c220f2911db9ff",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
    "checkpoints.py": "4ffc94da42c34798c263360ecdb4af7340f417d653cb7e37ad1b6169937be541",
    "metrics.py": "c5f3944e55392ae8a98fd6ed79b140a95d7af58fc3d3537d41ceefcc4c3b4f1c",
    "runtime.py": "09c7ee998e1c3be02f0ed3f00db6acc92351ea374fcf8a469c8d4fcc57233f05",
    "diagnostics.py": "aa89c9f29a8c641d005405b306f060a782c864d3e35a3b3371c7d3ffb4c1428a",
    "inputs.py": "2399ebaaf1f1c9fe3e3bcc302c90d2afa8f81c5352e16f5af6ae03efe93ff87f",
    "model.py": "86d21b96701c7ca99231fce3ddbd8f8d1c0967d81059b42787ef831cf203a96a",
}
SMALL_EXTRA = {
    "convnext_small_model.py": "e744d03266b29e6dc3f6c22318af9e1c290c4bdafdaec5754d23a6953a45018f",
    "convnext_small_training.py": "7d3c25b9e70f7e84bd9bd097f31bee8166763003fb3ae4cc582ad2db507c7163",
}
SPECS = {
    "tiny": {
        "experiment": "F014", "namespace": "cnn_f014_source16", "source_count": 16,
        "model_module": "convnext_model", "model_class": "ConvNeXtTripletMIL",
        "model_name": "convnext_tiny.fb_in22k_ft_in1k_384", "stage_depths": [3, 3, 9, 3],
        "checkpoint_schema": "f014_convnext_native392_suffix_head_v1", "config_schema": "f014_convnext_native392_v1",
        "online_schema": "f014_convnext_native392_last2_fast16_v1",
        "config": "configs/experiments/f014-convnext-tiny-native-last2-late-cosine.json",
        "config_SHA": "e474ad325899969f8587cc42ceebb8681ea20096b7234aef87e4a450eefffd0b",
        "provenance": "configs/f014-convnext-provenance.json",
        "provenance_SHA": "1a0ce6417e8591c31bc2da2ce3fd553ca2fe2727e21bfbbefd2d4b4376955c89",
        "training_policy": "artifacts/research/20261021-F014-ConvNeXt-encoder-prospective-v1/policy.json",
        "training_policy_SHA": "c86f51554480c4cb3be08745f569d8c56111bd5e825d2ceecde80ce8a377a68b",
        "generic_weights_SHA": "a5cbe10cfa2e90ec787a082253033ac0a61de0264b68210815a988e53a1a5d7c",
        "generic_weights_bytes": 114374272,
    },
    "small": {
        "experiment": "F015", "namespace": "cnn_f015_source18", "source_count": 18,
        "model_module": "convnext_small_model", "model_class": "ConvNeXtSmallTripletMIL",
        "model_name": "convnext_small.fb_in22k_ft_in1k_384", "stage_depths": [3, 3, 27, 3],
        "checkpoint_schema": "f015_convnext_native392_suffix_head_v1", "config_schema": "f015_convnext_native392_v1",
        "online_schema": "f015_convnext_native392_last2_fast16_v1",
        "config": "configs/experiments/f015-convnext-small-native-last2-late-cosine.json",
        "config_SHA": "b47ea2b01a77336496e2a1cef217b3f5fe1537fc860419f516ecdef91fb1b8ee",
        "provenance": "configs/f015-convnext-small-provenance.json",
        "provenance_SHA": "b06ab28faec92dcd10cf3450e0c7f04099362a4f8076a6a938f2c2393d50ace6",
        "training_policy": "artifacts/research/20261021-F015-ConvNeXtSmall-prospective-v1/policy.json",
        "training_policy_SHA": "45d4cf42149064f2205f061223fa297f602ca64f64927b5bf26f3fa362b95dfb",
        "generic_weights_SHA": "17e4aa3c6ba437a7ba160dc2792c86ec35a22d4a9d51443f0422e97e14f49097",
        "generic_weights_bytes": 200928946,
    },
}
WHEELS = {
    "timm-1.0.22-py3-none-any.whl": (2530238, "888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0"),
    "huggingface_hub-0.34.4-py3-none-any.whl": (
        561452, "9b365d781739c93ff90c359844221beef048403f1bc1f1c123c191257c3c890a"),
    "safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": (
        485835, "8045db2c872db8f4cbe3faa0495932d89c38c899c603f21e9b6486951a5ecb8f"),
}


class PreviewOnlyError(RuntimeError):
    """This code has no actual serving entry, even if a proof is supplied."""


def execute(*args, **kwargs):
    """Reject before inspecting arguments, paths, imports or any alleged proof."""
    raise PreviewOnlyError("CNN preview is disabled; actual model-package proof and runtime are not implemented")


def prepare_package(*args, **kwargs):
    """No unqualified archive/model restoration entry exists in this preview."""
    raise PreviewOnlyError("CNN preview cannot restore a model package; a future reviewed runtime is required")


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def candidate_spec(candidate):
    require(type(candidate) is str and candidate in SPECS, "Specify exactly one candidate: tiny or small")
    return copy.deepcopy(SPECS[candidate])


def expected_sources(candidate):
    candidate_spec(candidate)
    return {**SOURCE16, **(SMALL_EXTRA if candidate == "small" else {})}


def read_pinned(root, relative, expected_sha):
    root = Path(root).resolve(strict=True)
    path = root / relative
    require(path.resolve(strict=True).is_relative_to(root) and not path.is_symlink() and path.is_file(),
            "Pinned local source must remain within workspace")
    raw = path.read_bytes()
    require(pin(raw)["sha256"] == expected_sha, "Frozen source/metadata differs: " + relative)
    return raw


def literal(tree, name):
    values = [n.value for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(values) == 1, "One model source assignment: " + name)
    return ast.literal_eval(values[0])


def source_import_closure(files, model_module):
    """Read ASTs only; do not import even a source package initializer."""
    found, pending = set(), [model_module + ".py"]
    while pending:
        name = pending.pop()
        if name in found:
            continue
        require(name in files, "Complete inference-only relative import closure")
        found.add(name)
        for node in ast.walk(ast.parse(files[name])):
            if isinstance(node, ast.ImportFrom) and node.level:
                require(node.level == 1 and node.module is not None, "Explicit inference relative import")
                pending.append(node.module.replace(".", "/") + ".py")
    require(found == {model_module + ".py", "feature_model.py", "contracts.py"},
            "Only the model/head/contracts may be imported for CNN inference")
    return sorted(found)


def inspect_sources(root, candidate):
    """Read exact text/config bytes only; no checkpoint, MRI or wheel binaries."""
    spec = candidate_spec(candidate)
    sources = {name: read_pinned(root, "src/rsna_knee/" + name, digest)
               for name, digest in expected_sources(candidate).items()}
    require(len(sources) == spec["source_count"], "Exact producer source inventory")
    tree = ast.parse(sources[spec["model_module"] + ".py"])
    require(set(literal(tree, "SOURCE_FILES")) == set(sources)
            and literal(tree, "MODEL") == spec["model_name"]
            and literal(tree, "WEIGHTS_SHA") == spec["generic_weights_SHA"]
            and literal(tree, "CHECKPOINT_SCHEMA") == spec["checkpoint_schema"]
            and literal(tree, "SUFFIX_PREFIXES") == ("stages.3.blocks.1.", "stages.3.blocks.2."),
            "Exact independent CNN producer identity; no DINO/Tiny substitution")
    require(sum(isinstance(n, ast.ClassDef) and n.name == spec["model_class"] for n in tree.body) == 1,
            "One original producer model class")
    metadata = {"config.json": read_pinned(root, spec["config"], spec["config_SHA"]),
                "provenance.json": read_pinned(root, spec["provenance"], spec["provenance_SHA"]),
                "training-policy.json": read_pinned(root, spec["training_policy"], spec["training_policy_SHA"]),
                "preview-policy.json": read_pinned(root, PREVIEW_POLICY, PREVIEW_POLICY_SHA),
                "Linux-wheel-lock.json": read_pinned(root, WHEEL_LOCK, WHEEL_LOCK_SHA),
                "Linux-serving-notes.json": read_pinned(root, LINUX_NOTES, LINUX_NOTES_SHA)}
    config, provenance = (json.loads(metadata[name]) for name in ("config.json", "provenance.json"))
    require(config["schema_version"] == spec["config_schema"]
            and config["online"]["schema_version"] == spec["online_schema"]
            and config["encoder"]["model_name"] == spec["model_name"] and config["encoder"]["feature_dim"] == 768
            and config["seed"] == 20261002 and config["train"]["epochs"] == 12
            and config["preprocess"]["image_size"] == 392 and "augmentation" not in config,
            "Frozen native392 no-Gamma fixedE12 CNN config")
    require(provenance["model_name"] == spec["model_name"]
            and provenance["weights_sha256"] == spec["generic_weights_SHA"]
            and provenance["weights_bytes"] == spec["generic_weights_bytes"]
            and provenance["competition_finetuned"] is False and provenance["gold_used_for_tuning"] is False,
            "Generic CNN initialization metadata only, never trained-model qualification")
    lock = json.loads(metadata["Linux-wheel-lock.json"])
    actual = {p["filename"]: (p["bytes"], p["SHA"]) for p in lock["packages"]}
    require(actual == WHEELS and len(lock["packages"]) == len(WHEELS), "Exact separate Linux wheel lock")
    closure = source_import_closure(sources, spec["model_module"])
    return {"spec": spec, "sources": sources, "metadata": metadata, "inference_import_closure": closure}


def future_preconditions(candidate):
    """Requirements describe future evidence; they are not passed check results."""
    spec = candidate_spec(candidate)
    return {
        "schema_version": "CNN_P039_disabled_future_interface_v1", "candidate": candidate,
        "actual_enabled": False, "checkpoint_bindings_generated": False,
        "model_package_proof_consumed": False, "runtime_constructor_implemented": False,
        "stage_order": ["weak_fixedE12_pair", "root_package_review", "fresh_Linux_bootstrap",
                        "unchanged_REG_and_new_CNN_CPU", "current_eight_models_twoT4", "actual_native_MRI",
                        "independent_finalCSV_and_source_quota", "manual_formal_candidate"],
        "producer": spec,
        "required_evidence": {
            "weak_fixedE12_pair": "Each independently heldout fold0/fold1 fixedE12 weak10 delta >0 vs own F006 E12; exact source/config/seed/mask/group/cache/run/AMP counters; no own Gold numerical labels. No intermediate/BCE-best/label/alpha rescue.",
            "root_package_review": "Bind actual complete fixed epoch_012.pt and epoch_012_weak.csv for folds0/1 with full hashes and own CPU strict roundtrip evidence; source16/18 original bytes and current package policy hashes. No placeholder checkpoint or unbound manifest.",
            "fresh_Linux_bootstrap": "Fresh child and fresh vendor/producer namespace; cuBLAS env :4096:8 and HF_HUB_OFFLINE set before Torch. Verify Linuxx86_64 CPython/ABI3 and exact three wheel hashes; no Windows native wheel or parent R6 timm reuse. Original bootstrap/reader guards retained.",
            "unchanged_REG_and_new_CNN_CPU": "Original REG2 passes original strict reconstruction. Old CLS fingerprint is historical only. Full generic finiteFP32 CNN1000class strictload then remove classifier, own exact class/config/checkpoint/epoch12/source/prefix, complete18suffix/head tensor independent roundtrip. Never DINO encoder deepcopy or trainer import. Six CPU storage pairs independent.",
            "current_eight_models_twoT4": "Actual current CNN2+REG2 on each of cuda0/cuda1, finiteFP32 frozen parameters/exact classes, 12 GPU storage pairs independent. Native392 four-valid/allplanes, finite12, padding poison, allmissing, no CPU/CUDA RNG drift; original consume_four equals separate orderedFP32 fold50 then family50 reference. All originalREG/newCNN suffix/head/prefix fingerprints preserved after QA before first MRI.",
            "actual_native_MRI": "Original P039 exact reader, preprocessing and UID180 isolated recovery; no Report required. Current source/package-specific real MRI receipt with nonzero native contribution, UID coverage, typed failure union, no stale constructor certificate, no parent-only candidate if native fails; prefix/suffix/head unchanged after inference.",
            "independent_finalCSV_and_source_quota": "Actual current 12-column finite/probability/UID contract and separate oracle replay: exact P0338 parent/fold50/family50/native20/excludedMCL-Synovitis/success-subset/failureunion. Fresh saved source/Input versions and daily quota bound to one candidate before any formal action.",
            "manual_formal_candidate": "Separate root review and manual external actions only. Preview never grants Save/Submit or claims official gain; completion requires own official >=0.959 bound to submitted saved version.",
        },
        "invariants": {"P033_eight_cells": "original bytes", "REG_pair": "original strict identity",
                       "CNN_pair_folds": [0, 1], "selected_epoch": 12, "fold_weights": [.5, .5],
                       "family_weights": {"CLS_role_CNN": .5, "REG": .5}, "native_weight": .2,
                       "excluded_from_native": ["MCL", "Synovitis"], "label_order": list(TARGETS),
                       "failure_rule": "unchanged typed union and success-subset", "uid_deadline_seconds": 180.0},
        "not_qualified_by": ["this_preview", "F013_receipts", "old_CLS_CPU_receipt", "local_RTX4090_QA",
                             "parent_only_submission", "author_Public_claim"],
    }
