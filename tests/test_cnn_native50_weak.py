"""Synthetic gate-chain and family50 arithmetic tests; no real diagnostic."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cnn_native50_test_target", ROOT / "scripts/check_cnn_native50_weak.py")
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def evaluations(C, control=.6, candidate=.7):
    return {role: {"per_target": {t: {"auc": value} for t in C.TARGETS}}
            for role, value in (("control", control), ("candidate", candidate))}


def primary_fixture(C, policy, candidate="tiny", fold=0):
    spec = policy["primaries"][candidate]
    pp = json.loads((ROOT / spec["policy"]["path"]).read_bytes())
    items = {path: {"bytes": 1, "sha256": "synthetic"} for path in S.allowed_primary_artifacts(C, pp, fold)}
    control_dir = pp["controls"][str(fold)]["directory"]
    candidate_dir = pp["candidate"]["directories"][str(fold)]
    control_sha = pp["controls"][str(fold)]["file_pins"]["epoch_012.pt"]["sha256"]
    items[control_dir + "/epoch_012.pt"]["sha256"] = control_sha
    items[candidate_dir + "/epoch_012.pt"]["sha256"] = "futureCNNhash"
    ev = evaluations(C)
    value = {"schema_version": spec["experiment"] + "_fixedE12_paired_weak_CSV_audit_v1", "complete": True,
             "fold": fold, "fixed_epoch": 12, "policy_SHA": spec["policy"]["sha256"],
             "master_policy_SHA": spec["master_policy"]["sha256"],
             "strict_metadata_and_all12_artifact_checks_passed_before_metrics": True,
             "gold_numeric_labels_read": False, "checkpoint_tensor_deserialize_GPU_MRI_network_Submit": 0,
             "qualified_for_serving_or_Submit": False, "Public_gain_established": False,
             "primary": C.primary_delta(ev["control"], ev["candidate"]), "evaluation": ev,
             "source_scope": {"candidate": f"current_exact_source{spec['source_count']}"},
             "paired_UIDs": pp["candidate"]["counts"][str(fold)]["valid"],
             "accessed_artifacts": items, "fixedE12_checkpoint_SHA": {"control": control_sha, "candidate": "futureCNNhash"}}
    return value, spec, pp


class Native50Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only prospective policy/source definitions, never a run or teacher.
        cls.C, cls.policy = S.frozen_primary("tiny")
        cls.Small, _ = S.frozen_primary("small")

    def test_candidate_is_single_explicit_identity_and_sources_remain_independent(self):
        for bad in (None, True, ["tiny"], "tiny,small", "F014"):
            with self.assertRaises(ValueError):
                S.frozen_primary(bad)
        self.assertIsNot(self.C, self.Small)
        self.assertNotEqual(self.C.POLICY_SHA, self.Small.POLICY_SHA)
        self.assertEqual(self.policy["primaries"]["tiny"]["source_count"], 16)
        self.assertEqual(self.policy["primaries"]["small"]["source_count"], 18)

    def test_original_six_runs_and_REG_E8_E10_pins_are_preserved(self):
        ref = self.policy["reference_pins"]["old_native50_policy"]
        raw = (ROOT / ref["path"]).read_bytes()
        self.assertEqual(S.pin(raw), {k: ref[k] for k in ("bytes", "sha256")})
        old = json.loads(raw)
        self.assertEqual(self.policy["legacy_six_run_reference_pins"], old["inputs"])
        for f, epoch in (("0", 8), ("1", 10)):
            reg = self.policy["REG"][f]
            for name in ("run", "prediction", "checkpoint"):
                self.assertEqual(reg[name], old["inputs"][f]["REG"][name])
            self.assertEqual(reg["selected_epoch"], epoch)
            control = self.C.load_policy(self.C.Snapshot(ROOT))["controls"][f]
            for name in ("run.json", "epoch_012.pt", "epoch_012_weak.csv"):
                role = {"run.json": "run", "epoch_012.pt": "checkpoint", "epoch_012_weak.csv": "prediction"}[name]
                self.assertEqual(control["file_pins"][name], {k: old["inputs"][f]["control"][role][k] for k in ("bytes", "sha256")})

    def test_FP32_input_quantization_add_then_multiply_is_observable(self):
        a, b = .7661074377979527, .7042198668434126
        self.assertEqual(S.mean_two(a, b), .735163688659668)
        one_round = struct.unpack("<f", struct.pack("<f", (a + b) * .5))[0]
        self.assertNotEqual(S.mean_two(a, b), one_round)
        self.assertEqual(S.mean_two(0, 1), .5)
        self.assertEqual(S.mean_two(1, 1), 1)
        self.assertEqual(S.mean_two(a, b), S.mean_two(b, a))
        for invalid in (float("nan"), float("inf"), -.001, 1.001, True):
            with self.assertRaises(ValueError):
                S.mean_two(invalid, .5)

    def test_family50_average_ties_missingmask_and_soft_exclusion(self):
        C = self.C
        truth, cnn, reg = [], [], []
        for uid, y, a, b in (("p", "1", ".9", ".1"), ("n", "0", ".1", ".9"),
                             ("missing", "", ".9", ".9"), ("soft", ".2", ".9", ".9")):
            truth.append({C.ID: uid, **dict.fromkeys(C.TARGETS, y)})
            cnn.append({C.ID: uid, **dict.fromkeys(C.TARGETS, a)})
            reg.append({C.ID: uid, **dict.fromkeys(C.TARGETS, b)})
        result = S.family50(C, truth, cnn, reg)
        for t in C.TARGETS:
            self.assertEqual(result["per_target"][t]["auc"], .5)
            self.assertEqual(result["per_target"][t]["binary_observed"], 2)
            self.assertEqual(result["per_target"][t]["missing_excluded"], 1)
            self.assertEqual(result["per_target"][t]["soft_excluded"], 1)
        with self.assertRaises(ValueError):
            S.family50(C, truth, cnn[::-1], reg)

    def test_fixed10_rejects_excluded_target_or_denominator_rescue(self):
        ev = evaluations(self.C, .7, .69)
        for t in self.C.EXCLUDED:
            ev["candidate"]["per_target"][t]["auc"] = 1.
        self.assertFalse(self.C.primary_delta(ev["control"], ev["candidate"])["strictly_positive"])
        ev["candidate"]["per_target"]["ACL"]["auc"] = None
        with self.assertRaises(ValueError):
            self.C.primary_delta(ev["control"], ev["candidate"])

    def test_complete_positive_primary_is_required_in_each_namespace(self):
        for candidate, C in (("tiny", self.C), ("small", self.Small)):
            value, spec, pp = primary_fixture(C, self.policy, candidate)
            S.validate_primary_value(C, value, candidate, 0, spec, pp)
            mutations = [lambda v: v.update(complete=False), lambda v: v.update(fold=1),
                         lambda v: v.update(fixed_epoch=8), lambda v: v.update(policy_SHA="other"),
                         lambda v: v.update(gold_numeric_labels_read=True),
                         lambda v: v.update(qualified_for_serving_or_Submit=True),
                         lambda v: v["source_scope"].update(candidate="current_exact_source14"),
                         lambda v: v["accessed_artifacts"].update({"best.pt": {"bytes": 1, "sha256": "x"}}),
                         lambda v: v["fixedE12_checkpoint_SHA"].update(candidate="stale")]
            for mutate in mutations:
                bad = copy.deepcopy(value)
                mutate(bad)
                with self.assertRaises(ValueError):
                    S.validate_primary_value(C, bad, candidate, 0, spec, pp)
            for score in (.6, .59):
                bad = copy.deepcopy(value)
                bad["evaluation"] = evaluations(C, .6, score)
                bad["primary"] = C.primary_delta(bad["evaluation"]["control"], bad["evaluation"]["candidate"])
                with self.assertRaisesRegex(ValueError, "cannot be rescued"):
                    S.validate_primary_value(C, bad, candidate, 0, spec, pp)

    def test_wrong_result_SHA_refuses_before_primary_artifact_or_numeric_access(self):
        class Snapshot:
            def raw(self, name):
                return b"{}"

            def pin(self, *args):
                raise AssertionError("Wrong SHA reached input artifacts")

        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/research") as temporary:
            path = Path(temporary) / "synthetic.json"
            with patch.object(self.C, "load_policy", side_effect=AssertionError("Too early")):
                with self.assertRaisesRegex(ValueError, "root binding"):
                    S.bind_primary(self.C, Snapshot(), self.policy, "tiny", 0, path, "0" * 64)

    def test_no_primary_pass_means_no_teacher_run_or_prediction_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/research") as temporary, \
                patch.object(S, "bind_primary", side_effect=ValueError("Nonpositive primary")), \
                patch.object(self.C, "verify_inputs") as inputs, patch.object(self.C, "verify_run") as runs, \
                patch.object(S, "family50") as metrics, patch.object(S, "frozen_primary", return_value=(self.C, self.policy)):
            output = Path(temporary) / "not-created"
            with self.assertRaisesRegex(ValueError, "Nonpositive"):
                S.audit("tiny", 0, "never-read.json", "a" * 64, output)
            inputs.assert_not_called()
            runs.assert_not_called()
            metrics.assert_not_called()
            self.assertFalse(output.exists())

    def test_primary_result_revalidates_current_checkpoint_bytes_before_metrics(self):
        value, _, pp = primary_fixture(self.C, self.policy)
        raw = json.dumps(value).encode()
        candidate_pt = pp["candidate"]["directories"]["0"] + "/epoch_012.pt"

        class ChangedCheckpointSnapshot:
            def raw(self, name):
                return raw

            def pin(self, name, expected):
                if name == candidate_pt:
                    raise ValueError("Artifact bytes differ: synthetic checkpoint mutation")

        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/research") as temporary, \
                patch.object(self.C, "load_policy", return_value=pp), patch.object(self.C, "evaluate_weak") as metrics:
            with self.assertRaisesRegex(ValueError, "checkpoint mutation"):
                S.bind_primary(self.C, ChangedCheckpointSnapshot(), self.policy, "tiny", 0,
                               Path(temporary) / "synthetic.json", S.pin(raw)["sha256"])
            metrics.assert_not_called()

    def test_historical_REG_selected_writer_requires_original_epoch_and_bytes(self):
        ref = self.policy["REG"]["0"]
        run = {**copy.deepcopy(ref["run_identity"]), "status": "complete",
               "history": [{"epoch": e} for e in range(1, 13)],
               "selected": {"epoch": 8, "checkpoint_sha256": ref["checkpoint"]["sha256"],
                            "prediction_sha256": ref["prediction"]["sha256"]}}

        class REGSnapshot:
            def __init__(self, body):
                self.body = body
                self.bound = []

            def pin(self, name, expected):
                self.bound.append((name, expected))
                return expected

            def json(self, name):
                return self.body

        audit = run["audit"]
        snap = REGSnapshot(run)
        S.verify_reg(self.C, snap, self.policy, 0, audit["fold_audit"], audit["manifest_sha256"])
        self.assertEqual({n for n, _ in snap.bound}, {ref[r]["path"] for r in ("run", "prediction", "checkpoint")})
        for mutate in (lambda r: r["selected"].update(epoch=10),
                       lambda r: r["selected"].update(prediction_sha256="changed"),
                       lambda r: r["config"]["train"].update(epochs=13),
                       lambda r: r.update(gold_evaluated=True),
                       lambda r: r["history"].reverse()):
            bad = copy.deepcopy(run)
            mutate(bad)
            with self.assertRaises(ValueError):
                S.verify_reg(self.C, REGSnapshot(bad), self.policy, 0, audit["fold_audit"], audit["manifest_sha256"])

    def test_negative_prior_screen_blocks_fold1_before_teacher_or_current_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/research") as temporary, \
                patch.object(S, "bind_primary", return_value=({}, {}, {})), \
                patch.object(S, "bind_previous", side_effect=ValueError("Fold0 screen must be strictly positive")), \
                patch.object(S, "frozen_primary", return_value=(self.C, self.policy)), \
                patch.object(self.C, "verify_inputs") as inputs, patch.object(self.C, "verify_run") as runs:
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                S.audit("tiny", 1, "primary1.json", "a" * 64, Path(temporary) / "not-created",
                        previous_fold0="screen0.json", previous_sha="b" * 64)
            inputs.assert_not_called()
            runs.assert_not_called()

    def test_fold1_requires_prior_fold0_before_any_new_float_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/research") as temporary:
            with self.assertRaisesRegex(ValueError, "Fold1 requires"):
                S.audit("tiny", 1, "never-read.json", "a" * 64, Path(temporary) / "never")
        ev = evaluations(self.C)
        value = {"schema_version": S.SCHEMA, "candidate": "tiny", "fold": 0, "complete": True,
                 "policy_SHA": S.POLICY_SHA, "source_SHA": "current", "all_strict_checks_passed_before_new_numeric_reads": True,
                 "qualified_for_serving_or_Submit": False, "Public_gain_established": False, "gold_numeric_labels_read": False,
                 "screen": self.C.primary_delta(ev["control"], ev["candidate"]), "native50_evaluation": ev,
                 "primary_policy_SHA": self.policy["primaries"]["tiny"]["policy"]["sha256"]}
        S.validate_previous_value(self.C, value, "tiny", self.policy, "current")
        for key, bad in (("candidate", "small"), ("source_SHA", "old"), ("fold", 1), ("complete", False)):
            with self.assertRaises(ValueError):
                S.validate_previous_value(self.C, {**value, key: bad}, "tiny", self.policy, "current")
        bad = copy.deepcopy(value)
        bad["native50_evaluation"] = evaluations(self.C, .7, .7)
        bad["screen"] = self.C.primary_delta(bad["native50_evaluation"]["control"], bad["native50_evaluation"]["candidate"])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            S.validate_previous_value(self.C, bad, "tiny", self.policy, "current")

    def test_primary_inventory_does_not_contain_any_BCE_alias_or_other_fold(self):
        value, _, _ = primary_fixture(self.C, self.policy)
        names = set(value["accessed_artifacts"])
        self.assertTrue(any(n.endswith("epoch_012.pt") for n in names))
        self.assertFalse(any("/best.pt" in n or "weak_valid_predictions" in n for n in names))
        self.assertFalse(any("f014-convnext-tiny-fold1" in n for n in names))

    def test_no_model_GPU_network_imports_or_search_coefficient_flags(self):
        tree = ast.parse((ROOT / "scripts/check_cnn_native50_weak.py").read_text(encoding="utf-8"))
        imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse(imports & {"torch", "numpy", "kaggle", "requests", "rsna_knee", "pickle"})
        flags = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"}
        self.assertEqual(flags, {"--candidate", "--fold", "--primary-result", "--primary-sha", "--previous-fold0", "--previous-sha", "--output"})
        constants = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
        self.assertTrue(any(any(isinstance(k, ast.Constant) and k.value == "qualified_for_serving_or_Submit"
                                   and isinstance(v, ast.Constant) and v.value is False for k, v in zip(n.keys, n.values)) for n in constants))


if __name__ == "__main__":
    unittest.main()
