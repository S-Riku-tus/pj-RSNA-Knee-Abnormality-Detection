"""Synthetic metadata and a fake generic encoder; no actual F009 run or weights."""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsna_knee import f009_cpu_proof_v2 as P

ROOT = Path(__file__).resolve().parents[1]


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return {"path": relative, **P.pin(path)}


def fixture(root):
    spec = importlib.util.spec_from_file_location("_f009_old_artificial_stage_fixture", ROOT / "tests/test_f009_pair_stage.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    original = old.fixture(root)
    c = P._contracts()
    for relative in (P.LEGACY_PRODUCER, P.SNAPSHOT, P.AMENDED_WRAPPER, P.CLI_REL):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    p0, p1 = P.read(root / c.POLICY0), P.read(root / c.POLICY1)
    manifest_pins = {name: hashlib.sha256(("FAKE_META_" + name).encode()).hexdigest()
                     for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")}
    p0["input_cache"] = {**save(root, p0["input_cache"]["path"], {"manifest_sha256": manifest_pins}),
                         "manifest_sha256": manifest_pins}
    p0["resource_gate"] = save(root, p0["resource_gate"]["path"], {"source_sha256": P.sources.CORE_SHA})
    save(root, c.POLICY0, p0)
    p1["unchanged_fold0_policy"].update(P.pin(root / c.POLICY0))
    p1["training"]["config"] = p0["config"]
    completed = {}
    for fold in (0, 1):
        item = original["folds"][str(fold)]
        run_path = root / item["run"]["path"]
        run = P.read(run_path)
        for row in run["history"]:
            row.update(optimizer_steps=212 if fold == 0 else 208,
                       optimizer_step_opportunities=212 if fold == 0 else 208, train_bce=.3)
        run["audit"] = {"pixel_index_sha256": p0["input_cache"]["sha256"],
                        "manifest_sha256": manifest_pins, "resource_gate_sha256": p0["resource_gate"]["sha256"]}
        item["run"] = save(root, item["run"]["path"], run)
        index = P.read(root / item["index"]["path"])
        completed[fold] = {"candidate": {"run": P.pin(run_path), "index": P.pin(root / item["index"]["path"]),
            "E12_prediction": {"bytes": 10, "sha256": index["checkpoints"][0]["prediction_sha256"]},
            "E12_checkpoint_body_stream_hash": P.pin(run_path.parent / "epoch_012.pt")}}
    control_folder = Path(p1["fixed_control"]["run"]["path"]).parent
    control_run = copy.deepcopy(P.read(root / original["folds"]["1"]["run"]["path"]))
    control_run["history"][9].update(optimizer_steps=207, amp_skipped_steps=1)
    control_run_ref = save(root, (control_folder / "run.json").as_posix(), control_run)
    control_index = copy.deepcopy(P.read(root / original["folds"]["1"]["index"]["path"]))
    control_cp = b"ARTIFICIAL_NON_MODEL_HISTORICAL_CONTROL"
    cp_path = root / control_folder / "epoch_012.pt"
    cp_path.write_bytes(control_cp)
    control_index["checkpoints"][0].update(checkpoint_bytes=len(control_cp), checkpoint_sha256=hashlib.sha256(control_cp).hexdigest())
    control_index_ref = save(root, (control_folder / "epoch_checkpoint_index.json").as_posix(), control_index)
    pred = {"path": (control_folder / "epoch_012_weak.csv").as_posix(),
            "bytes": 10, "sha256": control_index["checkpoints"][0]["prediction_sha256"]}
    p1["fixed_control"].update(run=control_run_ref, retention_index=control_index_ref, epoch12_predictions=pred)
    save(root, c.POLICY1, p1)
    completed[1]["control"] = {"run": P.pin(root / control_run_ref["path"]),
        "index": P.pin(root / control_index_ref["path"]), "E12_prediction": {k: pred[k] for k in ("bytes", "sha256")},
        "E12_checkpoint_body_stream_hash": P.pin(cp_path)}
    sourcepins = {"src/rsna_knee/" + name: {"sha256": value} for name, value in P.sources.CORE_SHA.items()}
    sourcepins.update({p0[key]["path"]: {"sha256": p0[key]["sha256"]} for key in ("config", "resource_gate", "input_cache")})
    gbase = {"complete": True, "Gold_body_opened_or_hashed": False, "Gold_hash_metadata_equal": True,
             "model_deserialized": False, "input_source_pins": sourcepins}
    r0 = P.read(root / original["result0"]["path"])
    r0.update(policy_sha256=P.pin(root / c.POLICY0)["sha256"])
    r0["inputs"]["candidate"]["run_sha256"] = completed[0]["candidate"]["run"]["sha256"]
    r0ref = save(root, "proof/result0.json", r0)
    g0 = {"schema_version": "F009_fixed_E12_guarded_evaluation_receipt_v1", "complete": True,
        "original_evaluate_calls": 1, "wrapper_pin": {"sha256": c.F0_WRAPPER_SHA},
        "original_result_pin": {k: r0ref[k] for k in ("bytes", "sha256")}, "fold1_gate_conditions_met": True,
        "original_policy_thresholds_CI_BCE_output_modified": False,
        "guards": {**gbase, "target_projection_or_statistics_before_legacy_evaluate": False,
                   "completed_E12_control_candidate": completed[0]}}
    g0ref = save(root, "proof/guard0.json", g0)
    prior = {"result": {k: r0ref[k] for k in ("bytes", "sha256")},
             "guard_receipt": {k: g0ref[k] for k in ("bytes", "sha256")},
             "fixed10_delta": r0["fixed10_delta"], "engineering_gate_passed": True}
    lg = {"schema_version": "F009_fold1_fixed12_input_guard_v1", **gbase,
          "machine_engineering_passed": False, "AMP_skipped_steps": {"control": 1, "candidate": 0},
          "fold0_positive_sha_bound": prior, "completed_E12_control_candidate": completed[1],
          "fold1_target_projection_or_statistics_before_guards": False}
    wrapper = P._amended_wrapper()
    policy = P.read(ROOT / wrapper.POLICY_REL)
    def descriptor(relative):
        return {"path": relative, **P.pin(root / relative)}
    policy.update(original_policy_immutable=descriptor(c.POLICY1), fold0_policy_immutable=descriptor(c.POLICY0),
                  fold0_primary_result=r0ref, fold0_guard=g0ref, immutable_historical_control=p1["fixed_control"],
                  training_unchanged=p1["training"], scientific_evaluation_unchanged=p1["evaluation"])
    old_eval = root / policy["original_evaluator_immutable"]["path"]
    old_eval.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / policy["original_evaluator_immutable"]["path"], old_eval)
    snapshot = copy.deepcopy(P.read(root / original["folds"]["1"]["run"]["path"]))
    snapshot.update(status="running", history=[])
    policy["candidate_epoch0_snapshot"] = save(root, policy["candidate_epoch0_snapshot"]["path"], snapshot)
    policy["procedural_failure_record"] = save(root, "proof/failure-record.json", {"artificial": True})
    pref = save(root, wrapper.POLICY_REL, policy)
    declaration = {"schema_version": "F009_fold1_metric_free_amendment_declaration_receipt_v2",
        "declared_at_UTC": policy["declared_at_UTC"], "policy": pref,
        "candidate_epoch0_snapshot": policy["candidate_epoch0_snapshot"], "observed_candidate_history_length": 0,
        "candidate_fold1_metric_read": False, "old_policy_unchanged": True,
        "known_control_exact_history_verified": True, "original_v1_joint_screen": "HOLD", "fit_already_running": True}
    dref = save(root, wrapper.DECLARATION_REL, declaration)
    bindings = {"policy_sha256": pref["sha256"], "declaration_sha256": dref["sha256"],
                "legacy_sha256": P.AMENDED_BINDINGS["legacy_sha256"]}
    engineering = wrapper.engineering_guard(policy, snapshot, lg,
        {"control": control_run, "candidate": P.read(root / original["folds"]["1"]["run"]["path"])})
    lr = {"status": "F009_FOLD1_FIXED12_PAIRED_AND_INDEPENDENT_TWO_FOLD_SCREEN_V1", "fold": 1,
          "candidate_epoch": 12, "control_epoch": 12, "policy0_sha256": policy["fold0_policy_immutable"]["sha256"],
          "policy1_sha256": policy["original_policy_immutable"]["sha256"], "fold0": prior, "inputs": completed[1],
          "fold1_machine_engineering_passed": False, "joint_candidate_screen_passed": False,
          "fold1_independent_positive_screen": False,
          "AMP_skipped_steps": {"control": 1, "candidate": 0},
          "fixed10_delta": .002, "all12_auxiliary_delta": .002, "fixed10_interval": [-.1, .1],
          "all12_auxiliary_interval": [-.1, .1], "per_target": [], "bootstrap": {}, "weak_BCE": {}, "ordered_UID_sha256": "a"*64}
    lgref, lrref = save(root, "proof/legacy-guard.json", lg), save(root, "proof/legacy-comparison.json", lr)
    r1 = {"schema_version": "F009_fold1_known_control_AMP_amended_joint_result_v2",
          "status": "FIXED12_PREMETRIC_ENGINEERING_AMENDMENT_V2", "policy2_sha256": pref["sha256"],
          "declaration_sha256": dref["sha256"], "actual_comparison": True, "legacy_v1_joint_screen": "HOLD",
          "legacy_v1_machine_passed": False, "legacy_CPU_producer_sealer_compatible": False,
          "root_review_still_required": True, "BCE_CI_REG50_report_only": True, "automatic_fit_stage_Save_Submit": False,
          "legacy_guard_pin": {k: lgref[k] for k in ("bytes", "sha256")},
          "legacy_comparison_pin": {k: lrref[k] for k in ("bytes", "sha256")}, "fold": 1, "candidate_epoch": 12,
          "control_epoch": 12, "fold0": prior, "inputs": completed[1], "v2_engineering": engineering,
          "v2_joint_candidate_screen_passed": True, **{key: lr[key] for key in P.STATISTICS_KEYS}}
    r1ref = save(root, "proof/amended-result.json", r1)
    g1 = {"schema_version": "F009_fold1_amended_evaluation_receipt_v2", "complete": True, "actual_comparison": True,
          "wrapper_pin": P.pin(ROOT / P.AMENDED_WRAPPER), "policy_pin": {k: pref[k] for k in ("bytes", "sha256")},
          "declaration_pin": {k: dref[k] for k in ("bytes", "sha256")},
          "epoch0_snapshot_pin": {k: policy["candidate_epoch0_snapshot"][k] for k in ("bytes", "sha256")},
          "legacy_evaluator_pin": policy["original_evaluator_immutable"], "legacy_comparison_calls": 1,
          "result_pin": {k: r1ref[k] for k in ("bytes", "sha256")}, "legacy_guard_pin": r1["legacy_guard_pin"],
          "legacy_comparison_pin": r1["legacy_comparison_pin"], "original_v1_preserved_as_HOLD": True,
          "legacy_consumer_compatible": False, "automatic_fit_stage_Save_Submit": False,
          "engineering_v2": engineering, "v2_joint_candidate_screen_passed": True}
    qualification = {"result0": r0ref, "guard0": g0ref, "result1": r1ref,
                     "guard1": save(root, "proof/amended-guard.json", g1), "legacy_guard1": lgref, "legacy_comparison1": lrref}
    spec = {"schema_version": "f009_CPU_proof_amended_root_inputs_v2", "qualification": qualification,
        "folds": {str(n): {key: original["folds"][str(n)][key] for key in ("run", "index")} for n in (0, 1)},
        "provenance": save(root, "proof/provenance.json", {"weights_sha256": P.GENERIC_SHA})}
    return spec, bindings


def rewrite(root, qualification, key, value):
    qualification[key] = save(root, qualification[key]["path"], value)


def rebind_chain(root, qualification, legacy_guard, legacy_result, result, receipt):
    """Coherent synthetic rehashes reach semantic guards, not only stale pins."""
    rewrite(root, qualification, "legacy_guard1", legacy_guard)
    rewrite(root, qualification, "legacy_comparison1", legacy_result)
    result["legacy_guard_pin"] = {k: qualification["legacy_guard1"][k] for k in ("bytes", "sha256")}
    result["legacy_comparison_pin"] = {k: qualification["legacy_comparison1"][k] for k in ("bytes", "sha256")}
    rewrite(root, qualification, "result1", result)
    receipt.update(result_pin={k: qualification["result1"][k] for k in ("bytes", "sha256")},
                   legacy_guard_pin=result["legacy_guard_pin"], legacy_comparison_pin=result["legacy_comparison_pin"])
    rewrite(root, qualification, "guard1", receipt)


class AmendedCPUProof(unittest.TestCase):
    def test_positive_exact_exception_preserves_HOLD_and_no_model_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            original = P.pin
            def no_model(path):
                self.assertNotEqual(Path(path).suffix, ".pt")
                self.assertNotEqual(Path(path).name, "gold.csv")
                return original(path)
            with patch.object(P, "AMENDED_BINDINGS", bindings), patch.object(P, "pin", side_effect=no_model):
                gate = P.validate_amended_joint_metadata(root, spec["qualification"])
                context = P.prepare_inputs(root, spec, 1)
            self.assertTrue(gate["passed"])
            self.assertFalse(gate["legacy_guard1"]["machine_engineering_passed"])
            self.assertFalse(gate["legacy_comparison1"]["joint_candidate_screen_passed"])
            self.assertEqual(context["qualification_pins"], spec["qualification"])

    def test_artificial_comparison_statistics_pin_and_HOLD_tamper_rejected(self):
        for kind in ("actual_result", "actual_receipt", "statistics", "legacy_HOLD", "extra_key", "cross_pin"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec, bindings = fixture(root)
                q = spec["qualification"]
                key = "guard1" if kind in ("actual_receipt", "cross_pin") else "result1"
                value = P.read(root / q[key]["path"])
                if kind.startswith("actual"):
                    value["actual_comparison"] = False
                elif kind == "statistics":
                    value["fixed10_delta"] += .1
                elif kind == "legacy_HOLD":
                    value["legacy_v1_machine_passed"] = True
                elif kind == "cross_pin":
                    value["legacy_guard_pin"]["sha256"] = "f"*64
                if kind == "extra_key":
                    q["extra"] = q["guard1"]
                else:
                    rewrite(root, q, key, value)
                    if key == "result1":
                        g = P.read(root / q["guard1"]["path"])
                        g["result_pin"] = {k: q["result1"][k] for k in ("bytes", "sha256")}
                        rewrite(root, q, "guard1", g)
                with patch.object(P, "AMENDED_BINDINGS", bindings), self.assertRaises(ValueError):
                    P.validate_amended_joint_metadata(root, q)

    def test_candidate_skip_or_control_change_or_joint_nonpositive_stops(self):
        for kind in ("candidate_skip", "control_skip", "negative", "zero"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec, bindings = fixture(root)
                q = spec["qualification"]
                lg = P.read(root / q["legacy_guard1"]["path"])
                lr = P.read(root / q["legacy_comparison1"]["path"])
                r1 = P.read(root / q["result1"]["path"])
                g1 = P.read(root / q["guard1"]["path"])
                if kind in ("candidate_skip", "control_skip"):
                    relative = spec["folds"]["1"]["run"]["path"]
                    if kind == "control_skip":
                        relative = P.read(root / P._amended_wrapper().POLICY_REL)["immutable_historical_control"]["run"]["path"]
                    value = P.read(root / relative)
                    value["history"][0].update(optimizer_steps=207, amp_skipped_steps=1)
                    ref = save(root, relative, value)
                    arm = "candidate" if kind == "candidate_skip" else "control"
                    lg["completed_E12_control_candidate"][arm]["run"] = {k: ref[k] for k in ("bytes", "sha256")}
                    lr["inputs"] = copy.deepcopy(lg["completed_E12_control_candidate"])
                    r1["inputs"] = copy.deepcopy(lg["completed_E12_control_candidate"])
                else:
                    r1["fixed10_delta"] = lr["fixed10_delta"] = -.1 if kind == "negative" else 0.
                rebind_chain(root, q, lg, lr, r1, g1)
                with patch.object(P, "AMENDED_BINDINGS", bindings), self.assertRaises(ValueError):
                    P.prepare_inputs(root, spec, 1)

    def test_default_disabled_and_legacy_consumers_reject_new_spec(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / P.CLI_REL)], capture_output=True, text=True, check=True,
                             env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertFalse(json.loads(run.stdout)["actual_CPU_strict_reconstruction"])
        self.assertFalse(json.loads(run.stdout)["run_or_checkpoint_or_encoder_binary_opened"])
        from rsna_knee import f009_cpu_proof as old
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, _ = fixture(root)
            with self.assertRaises(ValueError):
                old.prepare_inputs(root, spec, 0)

    def test_model_suffix_rejected_before_metadata_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            spec["qualification"]["result1"]["path"] = "forbidden.pt"
            with patch.object(P, "AMENDED_BINDINGS", bindings), patch.object(P, "pin", wraps=P.pin) as touched:
                with self.assertRaisesRegex(ValueError, "metadata/source"):
                    P.validate_amended_joint_metadata(root, spec["qualification"])
            self.assertFalse(any(Path(call.args[0]).suffix == ".pt" for call in touched.call_args_list))

    def test_produce_tamper_and_cleanup_failure_never_publish_CPU(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            spec_path = root / "inputs.json"
            save(root, "inputs.json", spec)
            with patch.object(P, "ROOT", root), patch.object(P, "AMENDED_BINDINGS", bindings):
                context = P.prepare_inputs(root, spec, 0)
                cp_path = context["checkpoint_path"]
                original = cp_path.read_bytes()
                cp_path.write_bytes(b"ARTIFICIAL_TAMPER_NON_MODEL")
                with patch.object(P.sources, "load_f009_sources", side_effect=AssertionError("Too early model import")):
                    with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                        P.produce(spec_path, P.sha(spec_path), 0, root/"artifacts/qa/tamper",
                                  root/"absent", root/"absent", P.sha(P.__file__))
                self.assertFalse((root/"artifacts/qa/tamper").exists())
                cp_path.write_bytes(original)
                fake = SimpleNamespace(trainer=SimpleNamespace(native_source_hashes=lambda: P.sources.CORE_SHA),
                    receipt={"artificial_metadata_lifecycle_only": True},
                    close=lambda: (_ for _ in ()).throw(RuntimeError("Artificial cleanup failure")))
                result = {"producer_CLI_sha256": P.sha(root/P.CLI_REL), "artificial_metadata_lifecycle_only": True}
                with patch.object(P.sources,"load_f009_sources",return_value=fake), patch.object(P,"_reconstruct",return_value=result):
                    with self.assertRaisesRegex(RuntimeError,"Artificial cleanup failure"):
                        P.produce(spec_path,P.sha(spec_path),0,root/"artifacts/qa/close",root/"absent",root/"absent",P.sha(P.__file__))
                self.assertFalse((root/"artifacts/qa/close/CPU.json").exists())
                self.assertFalse(P.read(root/"artifacts/qa/close/FAILED.json")["complete"])
                def changed_on_close():
                    spec_path.write_bytes(spec_path.read_bytes() + b" ")
                fake.close = changed_on_close
                with patch.object(P.sources,"load_f009_sources",return_value=fake), patch.object(P,"_reconstruct",return_value=result):
                    with self.assertRaisesRegex(ValueError,"Final spec/source/qualification"):
                        P.produce(spec_path,P.sha(spec_path),0,root/"artifacts/qa/late",root/"absent",root/"absent",P.sha(P.__file__))
                self.assertFalse((root/"artifacts/qa/late/CPU.json").exists())
                self.assertFalse(P.read(root/"artifacts/qa/late/FAILED.json")["complete"])

    def test_real_source14_classes_fake_encoder_strict_CPU_numeric_no_GPU(self):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        class Parts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gamma = nn.Parameter(torch.ones(384))
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1, self.norm2 = nn.LayerNorm(384), nn.LayerNorm(384)
                self.attn = nn.Module()
                self.attn.qkv, self.attn.proj = nn.Linear(384,1152), nn.Linear(384,384)
                self.mlp = nn.Module()
                self.mlp.fc1, self.mlp.fc2 = nn.Linear(384,1536), nn.Linear(1536,384)
                self.ls1, self.ls2 = Parts(), Parts()
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [Block(),Block()])
                self.norm = nn.LayerNorm(384)
                self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384
            def forward(self, *args, **kwargs):
                raise AssertionError("Forward forbidden")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            P.sources.stage_f009_sources(ROOT / "src/rsna_knee", ROOT / "scripts/train_native_dinov2_fast16.py",
                ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json", root / "source14")
            loaded = P.sources.load_f009_sources(root / "source14", "_rsna_f009_amended_cpu_fixture")
            try:
                t = loaded.trainer
                config = P.read(ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")
                provenance = {"fake_generic_encoder": True}
                model = t.NativeTripletMIL(Encoder(), config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True)
                prefix = t.prefix_digest(model)
                cp = t.make_native_checkpoint(model, provenance, fold=1, epoch=12, weak_bce=.3,
                    prefix_sha256=prefix, audit={"artificial_fixture_only": True}, artificial=False)
                path = root / "artificial.pt"
                torch.save(cp, path)
                context = {"checkpoint_path": path, "checkpoint_pin": P.pin(path), "config": config, "fold": 1,
                    "row": {"weak_bce": .3}, "prefix": prefix, "producer_sha256": P.sha(P.__file__),
                    "provenance": provenance, "run_pin": {"bytes": 1,"sha256":"1"*64},
                    "retained_index_pin": {"bytes": 2,"sha256":"2"*64}, "qualification_pins": {}, "qualification_binding_pins": {}}
                calls=[]
                def factory(*args, **kwargs):
                    calls.append(1)
                    return t.NativeTripletMIL(Encoder(), config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True)
                with patch.object(t, "new_native_model", side_effect=factory), patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden")):
                    result=P._reconstruct(context,loaded,root/"missing-repo",root/"missing-weights",fixture_only=True)
                    self.assertFalse(result["actual_CPU_strict_reconstruction"])
                    self.assertEqual(result["schema_version"], P.CPU_SCHEMA)
                    self.assertEqual([len(result["tensor_metadata"][k]) for k in ("encoder_suffix","head")],[28,12])
                    self.assertEqual(result["encoder_suffix_tensor_sha256"],P.tensor_digest(cp["encoder_suffix"]))
                    self.assertEqual(result["head_tensor_sha256"],P.tensor_digest(cp["head"]))
                    self.assertTrue(result["eval_gamma_RNG_unchanged"])
                    self.assertEqual(len(calls),1)
                    for kind in ("NaN", "shape", "key", "source", "prefix"):
                        bad=copy.deepcopy(cp)
                        name=sorted(bad["head"])[0]
                        if kind=="NaN":
                            bad["head"][name].flatten()[0]=float("nan")
                        elif kind=="shape":
                            bad["head"][name]=torch.zeros(1)
                        elif kind=="key":
                            del bad["head"][name]
                        elif kind=="source":
                            bad["source_sha256"]["finetune_model.py"]="0"*64
                        else:
                            bad["frozen_prefix_sha256"]="0"*64
                        torch.save(bad,path)
                        before=len(calls)
                        with self.assertRaises(ValueError):
                            P._reconstruct(context,loaded,root/"missing",root/"missing",fixture_only=True)
                        self.assertEqual(len(calls),before)
                    torch.save(cp,path)
                    spoof = type("NativeTripletMIL", (nn.Module,), {"__module__": loaded.namespace + ".finetune_model"})
                    with patch.object(t,"new_native_model",return_value=spoof()):
                        with self.assertRaisesRegex(ValueError,"class identity"):
                            P._reconstruct(context,loaded,root/"missing",root/"missing",fixture_only=True)
            finally:
                loaded.close()
            self.assertNotIn("_rsna_f009_amended_cpu_fixture",sys.modules)


if __name__ == "__main__":
    unittest.main()
