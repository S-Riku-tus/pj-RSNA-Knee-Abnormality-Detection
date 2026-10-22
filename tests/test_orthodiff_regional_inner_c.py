"""Synthetic CPU tests; never read weak target files or regional vectors."""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rsna_knee import orthodiff_regional_inner_c as I  # noqa: E402
from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402


def synthetic():
    rows, features = [], {}
    for fold in I.INNER_FOLDS:
        for n in range(4):
            uid = "f%d-%d" % (fold, n)
            rows.append({ID: uid, "group_id": "g" + uid, "fold": str(fold), "source": "report_weak",
                         "targets": [float(n % 2)] * 12})
            features[uid] = [float(n % 2), float(n), float(fold)]
    outer = [{ID: "outer", "group_id": "outergroup", "fold": "2", "source": "report_weak"}]
    features["outer"] = [1., 999999., 2.]
    return rows, outer, features


def fake_fitter(tx, ty, vx, C):
    if C in (0.01, 0.1):
        p = .1 + .8 * vx[:, 0]
    elif C == .001:
        p = np.full(len(vx), .5)
    else:
        p = .9 - .8 * vx[:, 0]
    model = {"heads": [{"target": t, "status": "qualified", "warnings": []} for t in TARGETS]}
    return model, np.column_stack([p] * 12), {"synthetic_fitter": True}


class PoisonTokens(list):
    def __init__(self, values, forbidden):
        super().__init__(values)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("Heldout/unrelated target token was accessed")
        return super().__getitem__(key)


class InnerC(unittest.TestCase):
    def test_target_branch_poison_and_missing(self):
        header = [ID, *TARGETS, "group_id", "fold", "source", "Report"]
        cohort = [{ID: "train", "group_id": "g0", "fold": "0", "source": "report_weak"},
                  {ID: "heldout", "group_id": "g2", "fold": "2", "source": "report_weak"}]
        train = ["train", "", *(["1"] * 11), "g0", "0", "report_weak", "unused"]
        heldout = ["heldout", *(["UNREAD_POISON"] * 12), "g2", "2", "report_weak", "unused"]
        tokens = [PoisonTokens(train, [len(header) - 1]),
                  PoisonTokens(heldout, [*range(1, 13), len(header) - 1])]
        view = I.project_training_rows(header, tokens, cohort)
        self.assertIsNone(view["training"][0]["targets"][0])
        self.assertEqual(view["outer2_metadata"], [cohort[1]])
        self.assertNotIn("UNREAD_POISON", json.dumps(view))
        self.assertNotIn("Report", view["training"][0])
        bad = list(train)
        bad[1] = "0.4"
        with self.assertRaises(ValueError):
            I.project_training_rows(header, [bad, tokens[1]], cohort)
        bad[1] = "1"
        bad[0] = "swapped"
        with self.assertRaises(ValueError):
            I.project_training_rows(header, [bad, tokens[1]], cohort)

    def test_common_C_objective_tie_and_same_masks(self):
        rows, _, features = synthetic()
        # A missing auxiliary label never changes the selected fixed10 mask.
        rows[0]["targets"][1] = None
        calls = []

        def fitter(tx, ty, vx, C):
            calls.append((C, tx.copy(), vx.copy()))
            self.assertNotIn(999999., tx[:, 1])
            self.assertNotIn(999999., vx[:, 1])
            return fake_fitter(tx, ty, vx, C)

        result = I.select_common_C(rows, features, fitter)
        self.assertEqual(result["selected_C"], .01)
        self.assertEqual(len(calls), 16)
        self.assertEqual([r["score"] for r in result["table"]], [.5, 1., 1., 0.])
        self.assertEqual(sum(len(r["inner"]) * len(r["inner"][0]["AUC"]) for r in result["table"]), 160)
        for first in range(4):
            for Coffset in range(1, 4):
                np.testing.assert_array_equal(calls[first][2], calls[first + Coffset * 4][2])
        tied = [{"C": C, "eligible": True, "score": .8 + (5e-13 if C == 1 else 0)} for C in I.GRID]
        self.assertEqual(I.choose_C(tied), .001)

    def test_unsupported_fixed10_aborts_before_any_fit(self):
        rows, _, features = synthetic()
        fracture = TARGETS.index("Fracture")
        for row in rows:
            if row["fold"] == "3":
                row["targets"][fracture] = 0.
        with patch.object(I, "fit_at_C", side_effect=AssertionError("Must not fit")):
            with self.assertRaises(I.HoldRecipe):
                I.select_common_C(rows, features)

    def test_C_ineligible_auxiliary_unsupported_and_all_fail(self):
        rows, _, features = synthetic()

        def fitter(tx, ty, vx, C):
            m, p, r = fake_fitter(tx, ty, vx, C)
            if C == .001:
                m["heads"][0]["warnings"] = [{"category": "ConvergenceWarning"}]
            if C == .01:
                m["heads"][1]["status"] = "insufficient_training_classes"
                p[:, 1] = np.nan
            return m, p, r

        result = I.select_common_C(rows, features, fitter)
        self.assertFalse(result["table"][0]["eligible"])
        self.assertEqual(result["selected_C"], .01)
        self.assertEqual(result["table"][1]["inner"][0]["auxiliary_status"]["MCL"], "insufficient_training_classes")
        calls = []

        def fails(tx, ty, vx, C):
            calls.append(C)
            raise ValueError("Synthetic nonfinite solver output")

        with self.assertRaises(I.HoldRecipe) as caught:
            I.select_common_C(rows, features, fails)
        self.assertEqual(calls, list(I.GRID))
        self.assertEqual(len(caught.exception.selection["table"]), 4)

    def test_refit_failure_has_no_second_choice_or_label_input(self):
        rows, outer, features = synthetic()
        calls = []

        def fails(tx, ty, vx, C):
            calls.append(C)
            raise ValueError("Synthetic refit failure")

        with self.assertRaises(I.HoldRecipe):
            I.refit_selected(rows, outer, features, .01, fails)
        self.assertEqual(calls, [.01])
        with self.assertRaises(ValueError):
            I.refit_selected(rows, [{**outer[0], "targets": [1.] * 12}], features, .01, fake_fitter)
        with self.assertRaises(ValueError):
            I.refit_selected(rows, [{**outer[0], "group_id": rows[0]["group_id"]}], features, .01, fake_fitter)
        with self.assertRaises(ValueError):
            I.support_and_partitions(rows + [rows[0]], features)

    def test_AST_C1_numeric_parity_train_only_scaler_and_roundtrip(self):
        self.assertTrue(I.audit_fit_AST())
        gen = np.random.default_rng(20261002)
        tx, vx = gen.normal(size=(32, 6)), gen.normal(size=(5, 6))
        y = np.column_stack([np.arange(32) % 2] * 12).astype(np.float64)
        y[0, 0] = np.nan
        one, probabilities, report = I.fit_at_C(tx, y, vx, 1.)
        old, original, _ = H.fit_numeric(tx, y, vx)
        self.assertEqual(one, old)
        np.testing.assert_array_equal(probabilities.astype(np.float32).view(np.uint32), original.astype(np.float32).view(np.uint32))
        np.testing.assert_allclose(one["scaler"]["mean"], tx.mean(axis=0), rtol=0, atol=1e-15)
        changed, _, _ = I.fit_at_C(tx, y, vx + 1000000, .01)
        self.assertEqual(changed["scaler"], one["scaler"])
        self.assertEqual(one["heads"][0]["train_observed"], 31)
        self.assertTrue(report["sklearn_NumPy_FP32_bitwise_equal"])
        for C in (0., .5, True):
            with self.assertRaises(ValueError):
                I.fit_at_C(tx, y, vx, C)
        with tempfile.TemporaryDirectory() as tmp:
            for precision in ("float64", "float32"):
                path = Path(tmp) / (precision + ".csv")
                I.write_predictions(path, ["u%d" % i for i in range(5)], probabilities, precision)
                with path.open(newline="") as h:
                    records = list(csv.reader(h))
                restored = np.array([[float(x) for x in r[1:]] for r in records[1:]])
                dtype = np.float32 if precision == "float32" else np.float64
                np.testing.assert_array_equal(restored.astype(dtype), probabilities.astype(dtype))
                with self.assertRaises(FileExistsError):
                    I.write_predictions(path, ["u%d" % i for i in range(5)], probabilities, precision)

    def test_source_guards_disabled_CLI_and_feature_before_label_boundary(self):
        cli = ROOT / "scripts/fit_orthodiffusion_regional_inner_c.py"
        I.prepare_sources(I.POLICY_PATH, I.POLICY_SHA, H.pin(I.__file__)["sha256"],
                          cli, H.pin(cli)["sha256"], I.REVIEW_SHA)
        with self.assertRaises(ValueError):
            I.prepare_sources(I.POLICY_PATH, I.POLICY_SHA, "0" * 64, cli, H.pin(cli)["sha256"], I.REVIEW_SHA)
        spec = importlib.util.spec_from_file_location("inner_c_CLI", cli)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from types import SimpleNamespace
        with patch.object(I, "load_training_view", side_effect=AssertionError("Must not read labels")):
            with self.assertRaises(ValueError):
                module.prepare_qualified_inputs(SimpleNamespace(expected_bundle_sha256="0" * 64))
        result = subprocess.run([sys.executable, str(cli)], cwd=str(ROOT), capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "DISABLED_MANUAL_FIT_FLAG_REQUIRED")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
