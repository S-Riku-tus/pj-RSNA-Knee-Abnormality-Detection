"""CPU-only current F013/P039 constructor proof; no MRI, labels, GPU or APIs."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import f013_cpu_qualifier as C
import f013_p039_bridge as F

POLICY_SHA = "fdd5bc2afe2060eaacc30a2414356f94f5aeb2e30d89dd7173e1aaaa0b85504c"


def stage_private_package():
    """CPU qualification has its own entry; it cannot enable the serving entry."""
    roots = [Path("/kaggle/input") / F.PACKAGE, *Path("/kaggle/input").glob("datasets/rsraki/" + F.PACKAGE),
             *Path("/kaggle/input").glob("datasets/rsraki/" + F.PACKAGE + "/*")]
    found = list(dict.fromkeys(p for p in roots if (p / "package-manifest.json").is_file()))
    C.require(len(found) == 1 and C.sha(found[0] / "package-manifest.json") == F.MANIFEST_SHA, "One exact private package")
    manifest = json.loads((found[0] / "package-manifest.json").read_bytes())
    archive = found[0] / "f013-models.bin"
    C.require(C.sha(archive) == manifest["archive"]["sha256"] and archive.stat().st_size == manifest["archive"]["bytes"], "Fixed new archive")
    destination = Path("/tmp/F013-P039-mixed-CPU-package")
    C.extract(archive, manifest["members"], destination, exact=True)
    C.require(C.sha(destination / "qualification-policy.json") == C.POLICY_SHA, "Fixed original F013 qualification policy")
    C.require(not any(n == "source14" or n.startswith("source14.") for n in sys.modules), "Fresh source14 namespace")
    sys.path.insert(0, str(destination))
    trainer = importlib.import_module("source14.native_training_fast16")
    C.require(Path(trainer.__file__) == destination / "source14/native_training_fast16.py", "Exact F013 source")
    return destination, trainer


def artificial_cls(model, torch):
    C.require(not model.training and model._gamma_draws == 0, "CPU eval gamma disabled")
    rng, gamma = torch.get_rng_state().clone(), model._gamma_generator.get_state().clone()
    image = (torch.arange(3 * 392 * 392).reshape(1, 3, 392, 392) % 251).float() / 255
    mean, std = [image.new_tensor(v)[None, :, None, None] for v in ([.485, .456, .406], [.229, .224, .225])]
    with torch.inference_mode():
        feature = model.encode_normalized((image - mean) / std)
        C.require(feature.shape == (1, 384) and bool(torch.isfinite(feature).all()), "Current synthetic native392 encoder")
        features = feature.new_zeros((1, 3, 48, 384))
        features[0, 0, 0] = feature[0]
        mask = torch.zeros((1, 3, 48), dtype=torch.bool)
        mask[0, 0, 0] = True
        positions, planes, flags = torch.zeros((1, 3, 48)), torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3, 2))
        logits = model.head(features, mask, positions, planes, flags)
        C.require(logits.shape == (1, 12) and bool(torch.isfinite(logits).all()), "Current finite ordered12 CPU head")
        poisoned, poison_positions = features.clone(), positions.clone()
        poisoned[~mask], poison_positions[~mask] = float("nan"), float("nan")
        C.require(torch.equal(logits, model.head(poisoned, mask, poison_positions, planes, flags)), "Current masked CPU padding exact")
        refused = False
        try:
            model.head(features, torch.zeros_like(mask), positions, planes, flags)
        except ValueError:
            refused = True
        C.require(refused, "Current allmissing CPU head refused")
    C.require(model._gamma_draws == 0 and torch.equal(rng, torch.get_rng_state())
              and torch.equal(gamma, model._gamma_generator.get_state()), "Current eval CPU random streams unchanged")
    return {"complete": True, "manual_FP32_normalized392_encoder_and_head": True, "finite_ordered12": True,
            "masked_padding_exact": True, "allmissing_refused": True, "eval_gamma_CPU_RNG_unchanged": True,
            "full_native_forward_or_GPU_qualified": False}


def run(runtime, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XFORMERS_DISABLED"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    C.require("torch" not in sys.modules and Path("/kaggle/working").is_dir(), "Fresh Kaggle CPU process")
    runtime, output = Path(runtime), Path(output)
    C.require(not output.exists() and C.sha(runtime / "mixed-CPU-policy.json") == POLICY_SHA, "Fresh CPU output/frozen policy")
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(runtime))
    old = importlib.import_module("p032_native_child")
    # The unchanged bootstrap restores only positively pinned source/metadata.
    # Parent science cells, run(), fixture, reader and model-GPU paths are never called.
    exec(compile(old.BOOTSTRAP, "P039-original-CPU-source-bootstrap", "exec"), old.__dict__)
    c = old._p18_contracts
    plan = c.preflight(old._p18_context)
    for directory in (old._p18_source / "serving", plan["roots"]["mixed"], plan["roots"]["common"] / "reused",
                      plan["roots"]["common"] / "reused/code/scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    diagnostic = importlib.import_module("diagnostic_notebook")
    isolated = importlib.import_module("isolated_contracts")
    original_mixed = importlib.import_module("isolated_mixed_runtime")
    (output / "transports").mkdir(exist_ok=False)
    model_roots, transport = diagnostic.transport_models(plan, output / "transports")
    regional, boundary = plan["p"].dependencies(plan["roots"]["regional"])
    core, _ = boundary.trusted_imports(plan["roots"]["common"], model_roots["REG"][0] / "code")
    _, determinism = boundary.bootstrap()
    import torch

    torch.set_num_threads(1)
    C.require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), "CPU-only actual platform")
    manifests = {family: {int(k): v for k, v in values.items()} for family, values in plan["review"]["family_fold_manifest_sha256"].items()}
    pair, views, documents = isolated.bind_pairs(core, regional, model_roots, manifests)
    C.require(c.canonical(pair) == c.canonical(plan["pair"]), "Original four-model metadata must match")
    plan["pair"] = pair
    package, trainer = stage_private_package()
    mixed = F.Mixed(original_mixed, plan, package, trainer)
    replacement = F.new_pair(pair, mixed.policy)
    pair.clear()
    pair.update(replacement)
    models, component = mixed.reconstruct_cpu_models(model_roots, pair, views, documents, {},
                  core=core, regional=regional, boundary=boundary, execute=True,
                  common_src=plan["roots"]["common"] / "reused/code/src")
    checks = {}
    for fold in (0, 1):
        checks[str(fold)] = artificial_cls(models["CLS"][fold], torch)
    C.require(component["CPU_storage_alias_pairs_checked"] == 6 and component["CPU_storage_alias_count"] == 0
              and component["new_F013_strict_roundtrip"] and not torch.cuda.is_initialized(), "Actual CPU constructor final invariant")
    result = {"schema_version": "F013_P039_current_mixed_CPU_constructor_qualification_v1", "complete": True,
              "policy_SHA": POLICY_SHA, "bridge_source_SHA": hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),
              "package_manifest_SHA": F.MANIFEST_SHA, "component": component, "artificial_CLS": checks,
              "transport_receipts": transport, "pair": pair, "determinism": determinism,
              "Torch": torch.__version__, "CUDA_initialized": False, "current_REG_unchanged": True,
              "MRI_Report_Gold_labels_training_reads": 0, "old_F009_HOLD_unchanged": True,
              "qualified_for_GPU_MRI_serving_or_Submit": False}
    (output / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("F013_P039_CURRENT_MIXED_CPU_CONSTRUCTOR_PASS", flush=True)
    return result
