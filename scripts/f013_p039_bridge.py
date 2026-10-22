"""Separate F013 constructor with the frozen P039 reader and inference arithmetic.

This module has no API, upload, training, or model work at import time. It does
not issue a legacy F009 release or change any of its qualification consumers.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import f013_cpu_qualifier as C
from f013_cuda_evidence import validate_prior_cuda

POLICY_SHA = "f14432a21fd6f19bcb776fadff374f7d2c88c8aee42ae63f50238192f5cb1540"
MANIFEST_SHA = "69bcd8f922697e7fe6a8c761393312f524f3336be27cf041fffd25a0c7b6c0be"
ROUTER_SHA = "6dfb95abb56055c04da11a697b9491a72e1bca774d2fdc36285f296b34c4f02c"
PACKAGE = "f013-gamma-e12-qualification-20261020-v1"
NATIVE50_SHA = "e748dd5b2acf0237412d5f75f6b02caf8c3846611f1bd28f1e85f99536fba2f3"
NATIVE50_POLICY_SHA = "0f24b3625e4d555795cadfe4409ba81087d2c5dcdec5c5f17775993173c1842f"
FAMILIES = ("CLS", "REG")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_package(runtime):
    """Verify positive private-package inventory before importing its producers."""
    runtime = Path(runtime)
    C.require(C.sha(runtime / "P048-bridge-policy.json") == POLICY_SHA, "Frozen new-scope bridge policy")
    release = json.loads((runtime / "P048-release.json").read_bytes())
    C.require(release["actual_entry_enabled"] is True and release["both_fixed_folds_artificial_CUDA_pass"] is True
              and release["old_F009_HOLD_unchanged"] is True, "Separate actual F013 CUDA prerequisite")
    weak_path = runtime / "P048-native50-weak-check.json"
    C.require(C.sha(weak_path) == release["native50_check_SHA"] == NATIVE50_SHA, "Pinned fixed native50 check")
    weak = json.loads(weak_path.read_bytes())
    C.require(weak["complete"] is True and weak["bothfold_fixed10_positive"] is True
              and weak["policy_SHA"] == NATIVE50_POLICY_SHA
              and all(weak["folds"][str(f)]["fixed10_delta"] > 0 for f in (0, 1)), "Both fixed weak-family deltas before serving")
    proof = runtime / "P048-prior-CUDA-evidence.json"
    C.require(C.sha(proof) == release["CUDA_evidence_SHA"], "Pinned actual prior CUDA evidence")
    prior = validate_prior_cuda(proof.read_bytes())
    C.require(release["prior_CUDA_hardware"] == prior["GPU"]
              and release["current_twoT4_GPU_gate_still_required"] is True, "Local prior proof retains actual current two-T4 gate")
    roots = [Path("/kaggle/input") / PACKAGE, *Path("/kaggle/input").glob("datasets/rsraki/" + PACKAGE),
             *Path("/kaggle/input").glob("datasets/rsraki/" + PACKAGE + "/*")]
    found = list(dict.fromkeys(p for p in roots if (p / "package-manifest.json").is_file()))
    C.require(len(found) == 1, "One fixed private F013 package mount")
    mount = found[0]
    C.require(C.sha(mount / "package-manifest.json") == MANIFEST_SHA, "Fixed F013 manifest")
    manifest = json.loads((mount / "package-manifest.json").read_bytes())
    archive = mount / "f013-models.bin"
    C.require(C.sha(archive) == manifest["archive"]["sha256"]
              and archive.stat().st_size == manifest["archive"]["bytes"], "Exact new checkpoint archive")
    destination = Path(tempfile.mkdtemp(prefix="P048-package-parent-")) / "verified"
    C.extract(archive, manifest["members"], destination, exact=True)
    C.require(C.sha(destination / "qualification-policy.json") == C.POLICY_SHA, "Original F013 qualification policy")
    C.require(not any(n == "source14" or n.startswith("source14.") for n in sys.modules), "Fresh producer namespace")
    # The original router's first bootstrap requires Torch to be unimported.
    # Import the new producers only after that bootstrap and old CPU guards.
    return destination, None, release


def derive_execute(original_source):
    """Only three integration boundaries in a separate clone of old execute."""
    C.require(type(original_source) is bytes and hashlib.sha256(original_source).hexdigest() == ROUTER_SHA,
              "Exact frozen P039 router source")
    tree = ast.parse(original_source)
    execute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    expected = ast.dump(ast.parse('plan["values"]["CPU_smoke"]["model_fingerprints"]', mode="eval").body)
    counts = {"mixed": 0, "initial": 0, "expected": 0}

    class Adapt(ast.NodeTransformer):
        def visit_Import(self, node):
            if [(n.name, n.asname) for n in node.names] == [("isolated_mixed_runtime", "mixed")]:
                counts["mixed"] += 1
                return ast.parse('mixed = plan["F013_mixed"]').body[0]
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "result" and isinstance(node.value, ast.Dict):
                counts["initial"] += 1
                return ast.parse('result = plan["F013_initial"](args, plan, determinism)').body[0]
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            if ast.dump(node) == expected:
                counts["expected"] += 1
                return ast.parse('plan["F013_expected_fingerprints"]', mode="eval").body
            return self.generic_visit(node)

    result = ast.fix_missing_locations(ast.Module(body=[Adapt().visit(execute)], type_ignores=[]))
    C.require(counts == {"mixed": 1, "initial": 1, "expected": 1}, "Exactly three named constructor/metadata boundaries")
    return result


def new_pair(old_pair, policy):
    pair = copy.deepcopy(old_pair)
    cls = {"schema_version": "F013_fixed_E12_CLS_pair_v1", "bindings": {}}
    for fold in (0, 1):
        ref = policy["fixed_checkpoints"][str(fold)]
        left, right = (copy.deepcopy(v) for v in (old_pair["family_pairs"]["CLS"]["bindings"][fold]["config"], ref["config"]))
        for value in (left, right):
            value.pop("reason", None)
            value.pop("augmentation", None)
        C.require(left == right and ref["epoch"] == 12 and ref["fold"] == fold,
                  "Only training gamma may change the fixed two CLS configurations")
        cls["bindings"][fold] = {"config": ref["config"], "checkpoint_sha256": ref["sha256"],
                                  "selected_epoch": 12, "fold": fold, "selection": "fixed E12 exploratory; no Gold",
                                  "producer_class": "source14.finetune_model.NativeTripletMIL"}
    cls["pair_fingerprint_sha256"] = digest(cls)
    pair["family_pairs"]["CLS"] = cls
    pair["schema_version"] = "P048_F013_CLS_unchanged_P039_REG_pair_v1"
    pair["experimental_factor"] = "Two fixed E12 CLS roles become F013 gamma-trained; original REG and all inference arithmetic fixed"
    pair["model_classes"] = {"CLS": "source14.finetune_model.NativeTripletMIL", "REG": old_pair["model_classes"]["REG"]}
    pair["original_P039_pair_SHA_reference"] = old_pair["pair_fingerprint_sha256"]
    # These fields describe old bundles, not the new CLS checkpoint package.
    if "family_fold_manifest_sha256" in pair:
        pair["family_fold_manifest_sha256"]["CLS"] = {fold: MANIFEST_SHA for fold in (0, 1)}
    pair.pop("pair_fingerprint_sha256", None)
    pair["pair_fingerprint_sha256"] = digest(pair)
    return pair


def initial(args, plan, determinism):
    return {"schema_version": "P048_F013_new_constructor_raw_component_v1", "complete": False,
            "phase_contract_passed": False, "startup_complete": False, "determinism": determinism,
            "F013_bridge_policy_SHA": POLICY_SHA, "F013_package_manifest_SHA": MANIFEST_SHA,
            "pair_fingerprint_sha256": plan["pair"]["pair_fingerprint_sha256"],
            "input_receipt": plan["input_receipt"], "baseline_csv_sha256": plan["baseline_csv_sha256"],
            "old_F009_HOLD_unchanged": True, "old_F006_CPU_receipt_certifies_new_CLS": False,
            "teacher_Report_read": False, "own_Gold_fit_or_selection": False,
            "accuracy_evaluated": False, "final_CSV_written": False,
            "remaining_child_budget_seconds": args.budget_seconds,
            "numerical_scope": "Original P039 fold/family FP32 arithmetic, decoder and UID180 executor; only two CLS models replaced"}


class Mixed:
    def __init__(self, original, plan, package, trainer):
        self.original, self.plan, self.package, self.trainer = original, plan, package, trainer
        self.old_pair = copy.deepcopy(plan["pair"])
        self.policy = json.loads((package / "qualification-policy.json").read_bytes())
        self.classes = {}

    def reconstruct_cpu_models(self, roots, pair, views, documents, permission, **kwargs):
        import torch

        core = kwargs["core"]
        models, old = self.original.reconstruct_cpu_models(roots, self.old_pair, views, documents, permission, **kwargs)
        C.require(old["model_fingerprints"] == self.plan["values"]["CPU_smoke"]["model_fingerprints"],
                  "Original unchanged REG and encoder construction component must pass its own guard")
        if self.trainer is None:
            C.require(not any(n == "source14" or n.startswith("source14.") for n in sys.modules), "Fresh new producer namespace after original bootstrap")
            sys.path.insert(0, str(self.package))
            self.trainer = importlib.import_module("source14.native_training_fast16")
            module = importlib.import_module("source14.finetune_model")
            C.require(Path(self.trainer.__file__) == self.package / "source14/native_training_fast16.py"
                      and self.trainer.NativeTripletMIL is module.NativeTripletMIL, "Exact new producer after original CPU guards")
        expected = {key: copy.deepcopy(value) for key, value in old["model_fingerprints"].items() if key.startswith("REG:")}
        for fold in (0, 1):
            ref = self.policy["fixed_checkpoints"][str(fold)]
            path = self.package / ("fold%d/epoch_012.pt" % fold)
            C.require(C.sha(path) == ref["sha256"], "Fixed F013 checkpoint hash before load")
            checkpoint = self.trainer.load_native_checkpoint(path, allow_artificial=False)
            C.require(checkpoint["fold"] == fold and checkpoint["epoch"] == 12 and checkpoint["config"] == ref["config"]
                      and checkpoint["encoder_provenance"] == ref["encoder_provenance"]
                      and checkpoint["gold_used_for_training_or_selection"] is False, "Exact F013 identity")
            # No second DINO import/factory. The complete frozen prefix and all
            # suffix/head states are checked independently of the constructor.
            options = ref["config"]["online"]
            with torch.random.fork_rng(devices=[]):
                model = self.trainer.NativeTripletMIL(copy.deepcopy(models["CLS"][fold].encoder), copy.deepcopy(ref["config"]),
                            trainable_blocks=options["trainable_blocks"], window_chunk=options["window_chunk"],
                            checkpoint_last2=options["checkpoint_last2"]).to("cpu")
            C.require(self.trainer.prefix_digest(model) == checkpoint["frozen_prefix_sha256"], "Inherited generic prefix before strict apply")
            C.require(self.trainer.apply_native_checkpoint(model, checkpoint, ref["encoder_provenance"]) is model, "Original F013 strict apply")
            model.eval().requires_grad_(False)
            C.require(type(model) is self.trainer.NativeTripletMIL and self.trainer.prefix_digest(model) == checkpoint["frozen_prefix_sha256"], "New exact class and full frozen prefix")
            for part, actual in (("encoder_suffix", {k: v for k, v in model.encoder.state_dict().items() if k in self.trainer.suffix_keys(model)}),
                                 ("head", model.head.state_dict())):
                C.require(set(actual) == set(checkpoint[part]) and all(torch.equal(v, checkpoint[part][k])
                          and v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for k, v in actual.items()), "Independent exact finite suffix/head roundtrip")
            expected["CLS:" + str(fold)] = {"model_class": "source14.finetune_model.NativeTripletMIL", "selected_epoch": 12,
                    "checkpoint_sha256": ref["sha256"], "frozen_prefix_sha256": checkpoint["frozen_prefix_sha256"],
                    "encoder_suffix_tensor_sha256": core.tensor_digest(checkpoint["encoder_suffix"]),
                    "head_tensor_sha256": core.tensor_digest(checkpoint["head"]), "package_manifest_SHA": MANIFEST_SHA}
            C.require(core.tensor_digest(model.head.state_dict()) == expected["CLS:" + str(fold)]["head_tensor_sha256"], "Actual new head fingerprint")
            models["CLS"][fold] = model
        self.classes = {family: {fold: type(models[family][fold]) for fold in (0, 1)} for family in FAMILIES}
        checked = self.original.p.check_alias_pairs({f + ":" + str(n): models[f][n] for f in FAMILIES for n in (0, 1)},
                                                    lambda m: m.state_dict(), core.storage_aliases)
        C.require(checked == 6, "All six independent CPU storage pairs")
        for fold in (0, 1):
            C.require(core.tensor_digest(models["REG"][fold].head.state_dict()) == expected["REG:" + str(fold)]["head_tensor_sha256"], "Unchanged original REG state")
            keys = self.old_pair["family_pairs"]["REG"]["bindings"][fold]["suffix_state_keys"]
            suffix = {k: v for k, v in models["REG"][fold].encoder.state_dict().items() if k in keys}
            C.require(core.tensor_digest(suffix) == expected["REG:" + str(fold)]["encoder_suffix_tensor_sha256"], "Unchanged original REG suffix")
        self.plan["F013_expected_fingerprints"] = expected
        self.component = {"schema_version": "P048_F013_new_CPU_component_v1", "complete": True,
                          "model_fingerprints": expected, "CPU_storage_alias_pairs_checked": checked,
                          "CPU_storage_alias_count": 0, "original_P039_component_reference": old,
                          "original_reference_scope": "Original REG/prefix construction only; does not certify new CLS",
                          "new_F013_strict_roundtrip": True, "old_F009_HOLD_unchanged": True}
        return models, self.component

    def models_on_two_devices(self, cpu_models, core):
        import torch

        models = {family: core.models_on_two_devices(cpu_models[family]) for family in FAMILIES}
        for family in FAMILIES:
            for device in ("cuda:0", "cuda:1"):
                for fold in (0, 1):
                    model = models[family][device][fold]
                    C.require(type(model) is self.classes[family][fold] and not model.training and not model.encoder.training
                              and all(str(v.device) == device and v.dtype == torch.float32 and not v.requires_grad for v in model.parameters()), "Exact frozen current model on declared device")
        checked = sum(self.original.p.check_alias_pairs({f + ":" + str(n): models[f][device][n] for f in FAMILIES for n in (0, 1)},
                      lambda m: m.state_dict(), core.storage_aliases) for device in ("cuda:0", "cuda:1"))
        C.require(checked == 12, "All twelve GPU storage pairs")
        self.component["current_GPU_synthetic"] = artificial(models, core, self.plan["pair"], self.original)
        for family in FAMILIES:
            for device in ("cuda:0", "cuda:1"):
                for fold in (0, 1):
                    model = models[family][device][fold]
                    ref = self.plan["F013_expected_fingerprints"][family + ":" + str(fold)]
                    suffix = {k: v for k, v in model.encoder.state_dict().items() if k in self.trainer.suffix_keys(model)}
                    C.require(self.trainer.prefix_digest(model) == ref["frozen_prefix_sha256"]
                              and core.tensor_digest(suffix) == ref["encoder_suffix_tensor_sha256"]
                              and core.tensor_digest(model.head.state_dict()) == ref["head_tensor_sha256"], "All current GPU tensors unchanged after artificial forward")
        return models, checked

    def consume_four(self, *args, **kwargs):
        return self.original.consume_four(*args, **kwargs)


def artificial(models, core, pair, original):
    """Exercise the actual four-model consumer before allowing any MRI task."""
    import torch

    from rsna_knee.native_training_fast16 import tensor_study
    from rsna_knee.triplet_cache import array_hashes

    images = torch.zeros((3, 48, 3, 392, 392), dtype=torch.uint8)
    mask = torch.zeros((3, 48), dtype=torch.bool)
    for series, window in ((0, 0), (0, 20), (1, 13), (2, 25)):
        mask[series, window] = True
    images[mask] = (torch.arange(4 * 3 * 392 * 392).reshape(4, 3, 392, 392) % 251).to(torch.uint8)
    arrays = {"images": images.numpy(), "mask": mask.numpy(),
              "positions": torch.linspace(0, 1, 48)[None, :].expand(3, 48).clone().numpy(),
              "planes": torch.tensor([0, 1, 2], dtype=torch.long).numpy(),
              "flags": torch.tensor([[1., 0.], [0., 1.], [1., 1.]]).numpy()}
    receipt = {"native_SHA_verified_before_model": True, "valid_windows": 4, "array_sha256": array_hashes(arrays)}
    results = {}
    CPU_rng, CUDA_rng = torch.get_rng_state().clone(), [torch.cuda.get_rng_state(i).clone() for i in (0, 1)]
    for device in ("cuda:0", "cuda:1"):
        inputs = tensor_study(arrays, device=device)
        reference = {}
        with torch.cuda.device(device), torch.inference_mode():
            for family in FAMILIES:
                values = []
                for fold in (0, 1):
                    model = models[family][device][fold]
                    gamma = model._gamma_generator.get_state().clone() if family == "CLS" else None
                    logits = model(**inputs).float()
                    C.require(logits.shape == (1, 12) and bool(torch.isfinite(logits).all()), "Current synthetic ordered12")
                    poisoned = dict(inputs, images=inputs["images"].clone())
                    poisoned["images"][~inputs["mask"]] = 255
                    C.require(torch.equal(logits, model(**poisoned).float()), "Current padding invariance")
                    if family == "CLS":
                        C.require(model._gamma_draws == 0 and torch.equal(gamma, model._gamma_generator.get_state()), "No inference gamma draws")
                        refused = False
                        try:
                            model(**dict(inputs, mask=torch.zeros_like(inputs["mask"])))
                        except ValueError:
                            refused = True
                        C.require(refused, "New model allmissing refusal")
                    values.append(logits.sigmoid()[0].cpu().tolist())
                reference[family] = core.probability_mean_reference(*values)
        predictions, receipts = {}, {}
        family_predictions, family_receipts = {f: {} for f in FAMILIES}, {f: {} for f in FAMILIES}
        original.consume_four(device, ("P048_artificial", arrays, receipt), models=models, predictions=predictions, receipts=receipts,
                              lock=threading.Lock(), core=core, pair=pair, family_predictions=family_predictions,
                              family_receipts=family_receipts, array_hashes=array_hashes)
        C.require(predictions["P048_artificial"] == core.probability_mean_reference(reference["CLS"], reference["REG"]), "Actual original consumer equals independent FP32 four-model mean")
        results[device] = {"finite_all_four_current_models": True, "padding_exact": True, "new_CLS_gamma_disabled": True,
                           "original_consumer_independent_FP32_mean_exact": True, "probabilities": predictions["P048_artificial"]}
    C.require(torch.equal(CPU_rng, torch.get_rng_state()) and all(torch.equal(CUDA_rng[i], torch.cuda.get_rng_state(i)) for i in (0, 1)), "Inference consumes no global CPU/CUDA random streams")
    return {"complete": True, "devices": results, "MRI_Report_Gold_reads": 0}


def execute(args, active, plan, router, *, task_source_sha256, started, runtime):
    """Keep original reader source pins; apply proxies to the actual clone globals."""
    import isolated_mixed_runtime as original
    from p032blend.operational_uid_recovery import execute_router_with_operational_recovery

    C.require(args.native_executor == "isolated-uid" and task_source_sha256 == ROUTER_SHA, "Fixed original UID180 reader")
    package, trainer, release = prepare_package(runtime)
    mixed = Mixed(original, plan, package, trainer)
    replacement = new_pair(plan["pair"], mixed.policy)
    plan["pair"].clear()
    plan["pair"].update(replacement)  # The outer adapter holds this same pair object.
    plan.update(F013_mixed=mixed, F013_initial=initial)
    clone = ModuleType("_p048_current_router")
    clone.__dict__.update(router.__dict__)
    clone.__name__ = "_p048_current_router"
    # Recompilation gives execute these globals; copying the callable alone
    # would leave the operational multiprocessing/recovery proxies ineffective.
    exec(compile(derive_execute(Path(router.__file__).read_bytes()), "P048-current-execute", "exec"), clone.__dict__)
    C.require(clone.prepare_hidden_task is router.prepare_hidden_task and clone.verify_hidden_payload is router.verify_hidden_payload,
              "Original decoder callables retained")
    with tempfile.TemporaryDirectory(prefix="P048-operational-component-", dir=active.output.parent) as temporary:
        intermediate = copy.copy(active)
        intermediate.output = Path(temporary) / "operational.json"
        result = execute_router_with_operational_recovery(clone, intermediate, plan, started,
                    task_source_sha256=task_source_sha256, uid_deadline_seconds=180.0)
        C.require(mixed.component["current_GPU_synthetic"]["complete"], "Actual current GPU proof before MRI")
        result.update(F013_current_constructor_qualification=True, F013_release=release,
                      F013_bridge_policy_SHA=POLICY_SHA, old_F009_HOLD_unchanged=True)
        result["runtime_changes"].update(reader_and_model_source_bytes_changed=True, reader_source_bytes_changed=False,
                                         two_CLS_model_sources_changed=True, REG_model_sources_changed=False,
                                         success_four_fold_FP32_math_changed=False)
        result["legacy_assertions_scope"] = "Internal original router engineering invariants with new F013 constructor; no original F009 release or unchanged CLS claim"
        if "unchanged_model_startup_component" in result:
            result["F013_current_model_startup_component"] = result.pop("unchanged_model_startup_component")
            result.pop("unchanged_model_startup_component_scope", None)
        C.require(not active.output.exists(), "Fresh final new-scope result")
        active.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return result
