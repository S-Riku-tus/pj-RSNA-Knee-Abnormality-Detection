"""Artificial CPU tabular data only; never access real weak labels/features."""
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402

SPEC = importlib.util.spec_from_file_location("ortho_head_cli_test", ROOT / "scripts/fit_orthodiffusion_weak_head.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class WeakHeadCPU(unittest.TestCase):
    def test_missing_binary_contract_and_duplicate_UID_refusal(self):
        labels = H.labels_from_cells(["0", "1", "", "NaN", "null", "NA"])
        self.assertEqual(labels[:2].tolist(), [0., 1.])
        self.assertTrue(np.isnan(labels[2:]).all())
        for bad in ("-1", ".5", "2", "Inf"):
            with self.assertRaises(ValueError):
                H.labels_from_cells([bad])
        with self.assertRaises(ValueError):
            H.index_rows([{ID: "a"}, {ID: "a"}])
        self.assertEqual(list(H.index_rows([{ID: "b"}, {ID: "a"}])), ["b", "a"])
        controls = {"a": {name: ".1" for name in TARGETS}, "b": {name: ".8" for name in TARGETS}}
        aligned = H.align_controls(["b", "a"], controls)
        np.testing.assert_array_equal(aligned[:, 0], [.8, .1])
        with self.assertRaises(ValueError):
            H.align_controls(["a", "c"], controls)

    def test_train_only_scaler_numeric_roundtrip_and_insufficient_target(self):
        generator = np.random.RandomState(21)
        train = generator.normal(size=(80, 8))
        train[:, -1] = 2.5  # A constant training feature must have scale1.
        labels = np.column_stack([(train[:, i % 7] + generator.normal(size=80) * .1 > 0).astype(float) for i in range(12)])
        labels[::4, 0] = np.nan
        labels[:, 1] = np.nan  # Explicitly unsupported MCL head; never fabricate a constant fit.
        valid = generator.normal(size=(13, 8))
        model, prediction, receipt = H.fit_numeric(train, labels, valid)
        alternate, _, _ = H.fit_numeric(train, labels, valid + 1e5)
        self.assertEqual(model["scaler"], alternate["scaler"])
        self.assertEqual(model["heads"], alternate["heads"])
        np.testing.assert_allclose(model["scaler"]["mean"], train.mean(axis=0), rtol=0, atol=0)
        self.assertEqual(model["scaler"]["scale"][-1], 1.)
        self.assertEqual(model["heads"][0]["train_missing"], 20)
        self.assertEqual(model["heads"][1]["status"], "insufficient_training_classes")
        self.assertIsNone(model["heads"][1]["coef"])
        self.assertTrue(np.isnan(prediction[:, 1]).all())
        restored = json.loads(json.dumps(model, allow_nan=False))
        np.testing.assert_array_equal(H.predict_numeric(restored, valid), prediction)
        self.assertTrue(receipt["sklearn_NumPy_FP32_bitwise_equal"])
        self.assertLess(receipt["sklearn_NumPy_max_abs_difference"], 1e-12)
        self.assertFalse(receipt["validation_labels_used_for_fit_or_selection"])
        self.assertNotIn("torch", sys.modules)

    def test_rank_rank_FP32_formula_failure_and_degenerate_exact_fallback(self):
        raw = np.tile(np.asarray([.9, .2, .2, .7, .1])[:, None], (1, 12))
        head = np.full((5, 12), np.nan)
        success = np.asarray([True, False, True, True, False])
        head[success] = np.asarray([.9, .1, .3])[:, None]
        head[success, 2] = .25  # One supported target has degenerate FP32 ranks.
        baseline, mixed, details = H.rank_blend(raw, head, success, {x: True for x in TARGETS})
        # Independent pairwise average-rank oracle over all five raw inputs.
        oracle = np.asarray([(1 + sum(y < x for y in raw[:, 0]) + .5 * (sum(y == x for y in raw[:, 0]) - 1)) / 5 for x in raw[:, 0]], dtype=np.float32)
        np.testing.assert_array_equal(baseline[:, 0], oracle)
        q = [1., 1 / 3, 2 / 3]

        def f32(value):
            return struct.unpack('<f', struct.pack('<f', value))[0]

        expected = [f32(f32(f32(.8) * float(base)) + f32(f32(.2) * f32(rank))) for base, rank in zip(oracle[success], q)]
        np.testing.assert_array_equal(mixed[success, 0], np.asarray(expected, dtype=np.float32))
        np.testing.assert_array_equal(mixed[~success].view(np.uint32), baseline[~success].view(np.uint32))
        for target in ("MCL", "Synovitis", "Medial Meniscus"):
            i = TARGETS.index(target)
            np.testing.assert_array_equal(mixed[:, i].view(np.uint32), baseline[:, i].view(np.uint32))
        self.assertFalse(details["Medial Meniscus"]["active"])

    def test_masked_AUC_BCE_and_paired_supplied_group_CI(self):
        count = 8
        y = np.tile(np.asarray([0., 1.] * 4)[:, None], (1, 12))
        y[0, 0] = np.nan
        raw = np.tile(np.arange(count, dtype=float)[:, None], (1, 12)) / count
        head = np.tile(np.asarray([.05, .95] * 4)[:, None], (1, 12))
        success = np.ones(count, dtype=bool)
        success[-1] = False  # Explicitly distinguish the paired subset from all821-style evaluation.
        baseline, mixed, details = H.rank_blend(raw, head, success, {x: True for x in TARGETS})
        model = {"heads": [{"target": x, "status": "qualified"} for x in TARGETS]}
        groups = [str(i // 2) for i in range(count)]
        result = H.evaluate(y, raw, baseline, mixed, head, success, groups, model, details)
        self.assertTrue(result["raw_rank_masked_AUC_exactly_equal_all_targets"])
        self.assertTrue(result["paired_success_raw_rank_masked_AUC_exactly_equal_all_targets"])
        for name in TARGETS:
            part = result["per_target"][name]
            self.assertEqual(part["paired_success_raw_control"]["auc"], part["paired_success_control"]["auc"])
        self.assertEqual(result["per_target"]["MCL"]["paired_success_raw_control"]["observed"], 7)
        self.assertEqual(result["per_target"]["ACL"]["raw_control"]["missing"], 1)
        self.assertEqual(result["paired_group_bootstrap"]["fixed10"]["requested_draws"], 1000)
        self.assertEqual(result["paired_group_bootstrap"]["groups"], 4)
        self.assertTrue(result["BCE_and_CI_report_only"])
        self.assertFalse(result["continuation_automatic_start"])
        # A small independent pairwise AUC calculation verifies ties/missing weighting.
        labels, scores = [0, 1, 0, 1], [.2, .2, .7, .9]
        pairs = [float(b > a) + .5 * (b == a) for a, ya in zip(scores, labels) if ya == 0 for b, yb in zip(scores, labels) if yb == 1]
        self.assertEqual(H.metric(np.asarray(labels), np.asarray(scores))["auc"], sum(pairs) / len(pairs))
        self.assertIsNone(H.metric(np.asarray([np.nan, 1.]), np.asarray([.3, .5]))["auc"])
        second = H.paired_group_bootstrap(y, baseline, mixed, groups)
        self.assertEqual(second, result["paired_group_bootstrap"])

    def test_unqualified_bundle_rejected_before_real_labels_or_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unqualified.json"
            H.save_json(path, {"qualified": False})
            args = SimpleNamespace(output=Path(temp) / "new-run", policy=CLI.POLICY, expected_policy_sha256=H.POLICY_SHA,
                bundle=path, expected_bundle_sha256=H.pin(path)["sha256"], manifest=Path(temp) / "no-labels",
                control=Path(temp) / "no-control")
            with patch.object(H, "load_weak_inputs", side_effect=AssertionError("must not read labels")) as reader:
                with patch.object(H, "fit_numeric", side_effect=AssertionError("must not fit")) as fitter:
                    with self.assertRaises(ValueError):
                        CLI.fit(args)
                    reader.assert_not_called()
                    fitter.assert_not_called()
            self.assertFalse(args.output.exists())
            with self.assertRaises(ValueError):
                H.load_qualified_features(path, "0" * 64)

    def test_seal_requires_actual_qualification_and_positive_SHA_before_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            audit = folder / "audit.json"
            H.save_json(audit, {"schema_version": "orthodiff_fullweak4207_actual_qualification_v1", "passed": False,
                                "head_features_qualified": False})
            for digest in (H.pin(audit)["sha256"], "0" * 64):
                with self.assertRaises(ValueError):
                    H.seal_bundle(audit, digest, folder / "absent-records", folder / "absent-summary",
                                  folder / "absent-status", folder / "bundle.json")
            self.assertFalse((folder / "bundle.json").exists())
        self.assertTrue(math.isfinite(float(np.float32(.8))))


if __name__ == "__main__":
    unittest.main()
