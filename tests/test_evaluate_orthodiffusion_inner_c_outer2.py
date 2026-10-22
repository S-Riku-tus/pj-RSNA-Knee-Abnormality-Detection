"""Artificial CPU cases only: no real target projection or control readiness."""
import copy
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_orthodiffusion_inner_c_outer2.py"
spec = importlib.util.spec_from_file_location("new_outer2_evaluator", SCRIPT)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ranks(values):
    n = len(values)
    return [f32((sum(v < x for v in values) + (sum(v == x for v in values) + 1) / 2) / n) for x in values]


def native_fixture():
    sources = {"f006_fold2_training.py": "synthetic_source"}
    config = {"seed": 20261002, "online": {"folds": [2]}, "train": {"epochs": 12}}
    semantic = E.hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    gate = {"schema_version": "native392_training_resource_gate_fast16_fold2_v1", "source_sha256": sources,
            "config_semantic_sha256": {"2": semantic}, **{k: True for k in E.GATE_FLAGS}}
    history, entries = [], []
    for epoch in range(1, 13):
        history.append({"epoch": epoch, "optimizer_steps": 207, "amp_skipped_steps": 1,
                        "optimizer_step_opportunities": 208, "train_bce": .4, "weak_bce": .5,
                        "prediction_sha256": "pred%d" % epoch})
        entries.append({**history[-1], "predictions_csv": "epoch_%03d_weak.csv" % epoch,
                        "checkpoint": "epoch_%03d.pt" % epoch, "checkpoint_sha256": "checkpoint%d" % epoch,
                        "checkpoint_bytes": 123})
    run = {"schema_version": "native392_training_run_fast16_fold2_v1", "status": "complete", "fold": 2,
           "seed": 20261002, "train_studies": 3319, "valid_studies": 888, "targets": list(E.TARGETS),
           "config": config, "source_sha256": sources, "selection_data": "weak_validation", "selection_rule": "bce",
           "gold_used_for_training_or_selection": False, "gold_evaluated": False, "resume_supported": False,
           "history": history, "audit": {"resource_gate_sha256": "gate", "resource_gate": gate}}
    index = {"schema_version": "native392_epoch_suffix_head_retention_v1",
             "checkpoint_schema": "native392_generic_dino_suffix_head_fast16_fold2_v1",
             "fold": 2, "source_sha256": sources, "targets": list(E.TARGETS), "selection_data": "weak_validation",
             "selection_rule": "minimum observed weak BCE; earliest exact tie", "resume_supported": False,
             "checkpoints": entries, "aliases": {"best.pt": {"epoch": 3}}}
    preflight = {"ready": True, "train_studies": 3319, "valid_studies": 888, "cache_studies": 4207,
                 "fitting_performed": False}
    return [run, index, gate, preflight, config, sources, "gate", {"sha256": "pred12"},
            {"sha256": "checkpoint12", "bytes": 123}]


class PoisonTokens(list):
    def __init__(self, values, forbidden):
        super().__init__(values)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("Otherfold target or Report token accessed")
        return super().__getitem__(key)


class NewOuter2Evaluator(unittest.TestCase):
    def test_fold2_only_target_projection_otherfold_poison_missing_and_metadata(self):
        header = [E.ID, *E.TARGETS, "group_id", "fold", "source", "Report"]
        cohort, tokens = [], []
        for fold in range(5):
            row = {E.ID: "uid%d" % fold, "group_id": "group%d" % fold,
                   "fold": str(fold), "source": "report_weak"}
            cohort.append(row)
            cells = [row[E.ID], *(["" if i == 0 else "1" for i in range(12)]), row["group_id"], str(fold), "report_weak", "ignore"]
            forbidden = [len(header) - 1] if fold == 2 else [*range(1, 13), len(header) - 1]
            tokens.append(PoisonTokens(cells, forbidden))
        view = E.project_fold2(header, tokens, cohort)
        self.assertEqual(len(view), 1)
        self.assertEqual(view[0][E.ID], "uid2")
        self.assertIsNone(view[0]["targets"][0])
        self.assertNotIn("Report", view[0])
        bad = list(tokens)
        bad[2] = list(tokens[2])
        bad[2][1] = ".5"
        with self.assertRaises(ValueError):
            E.project_fold2(header, bad, cohort)
        bad[2][1] = "1"
        bad[2][0] = "swapped_uid"
        with self.assertRaises(ValueError):
            E.project_fold2(header, bad, cohort)

    def test_native_E12_guard_wrong_fold_source_epoch_pin_and_AMP_accounting(self):
        fixture = native_fixture()
        result = E.validate_control_records(*fixture)
        self.assertEqual(result["fixed_epoch"], 12)
        self.assertEqual(result["AMP_skips_total"], 12)  # Accounting is enforced, not a new zero-skip condition.
        mutations = [lambda x: x[0].update(fold=0),
                     lambda x: x[0].update(source_sha256={"wrong": "source"}),
                     lambda x: x[1]["checkpoints"][11].update(checkpoint="best.pt"),
                     lambda x: x[1]["checkpoints"][11].update(prediction_sha256="different"),
                     lambda x: x[0]["history"][11].update(optimizer_steps=208),
                     lambda x: x[2].update(schema_version="old_resource_schema"),
                     lambda x: x[3].update(valid_studies=891)]
        for mutate in mutations:
            altered = copy.deepcopy(fixture)
            mutate(altered)
            with self.assertRaises(ValueError):
                E.validate_control_records(*altered)

    def test_failed_control_or_release_pin_never_reaches_target_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            release_path = Path(tmp) / "release.json"
            release_path.write_text("{}", encoding="utf-8")
            with patch.object(E, "project_fold2", side_effect=AssertionError("Must not project")) as projector:
                with self.assertRaises(ValueError):
                    E.execute_release({}, release_path, "0" * 64, "unused", Path(tmp) / "out")
                altered = native_fixture()
                altered[0]["fold"] = 1

                def reject_control(*_):
                    E.validate_control_records(*altered)

                with patch.object(E, "prepare_before_targets", side_effect=reject_control):
                    with self.assertRaises(ValueError):
                        E.execute_release({}, release_path, E.H.pin(release_path)["sha256"], "unused", Path(tmp) / "out")
                projector.assert_not_called()
                self.assertFalse((Path(tmp) / "out").exists())
        # Source pin mismatch is rejected before any cohort/control/target JSON is loaded.
        template = E.H.read_json(E.TEMPLATE)
        template.update(authorized=True, status="AUTHORIZED_ONE_FIXED_OUTER2_COMPARISON",
                        decision="RELEASE_FIXED_OUTER2_TARGETS_FOR_ONE_PREDECLARED_COMPARISON",
                        root_release_created_at_UTC="synthetic-time")
        template["design"]["sha256"] = "0" * 64
        with patch.object(E, "project_fold2", side_effect=AssertionError("Must not project")) as projector:
            with self.assertRaises(ValueError):
                E.prepare_before_targets(template, "unused")
            projector.assert_not_called()

    def test_exact_rank20_oracle_failed_parent_MCL_Synovitis_and_reportonly_CI(self):
        n = 12
        raw = E.np.asarray([[(5 * r + c) % n / n for c in range(12)] for r in range(n)], dtype=E.np.float64)
        success = E.np.asarray([True] * 10 + [False] * 2)
        head = E.np.full((n, 12), E.np.nan)
        for col in range(12):
            head[:10, col] = [(7 * r + col) % 10 / 10 for r in range(10)]
        head[:10, 0] = .5  # Degenerate FP32 head must keep the baseline.
        labels = E.np.asarray([[float(r % 2)] * 12 for r in range(n)])
        labels[0, 3] = E.np.nan
        model = {"heads": [{"target": t, "status": "qualified"} for t in E.TARGETS]}
        groups = ["pair%d" % (i // 2) for i in range(n)]
        baseline, blend, evaluation, result = E.compare_once(labels, raw, head, success, groups, model)
        for col, target in enumerate(E.TARGETS):
            base = ranks(raw[:, col].tolist())
            candidates = ranks([f32(x) for x in head[:10, col]])
            for row in range(n):
                expected = base[row]
                if row < 10 and target in E.H.FIXED10 and col != 0:
                    expected = f32(f32(f32(.8) * base[row]) + f32(f32(.2) * candidates[row]))
                self.assertEqual(struct.pack("<f", float(blend[row, col])), struct.pack("<f", expected))
        self.assertTrue(E.np.array_equal(blend[~success].view(E.np.uint32), baseline[~success].view(E.np.uint32)))
        for name in ("MCL", "Synovitis"):
            col = E.TARGETS.index(name)
            self.assertTrue(E.np.array_equal(blend[:, col].view(E.np.uint32), baseline[:, col].view(E.np.uint32)))
        self.assertEqual(evaluation["paired_group_bootstrap"]["repeats"], 1000)
        self.assertEqual(evaluation["paired_group_bootstrap"]["seed"], 20261012)
        self.assertTrue(evaluation["BCE_and_CI_report_only"])
        self.assertTrue(evaluation["raw_rank_masked_AUC_exactly_equal_all_targets"])
        self.assertEqual(result["weak_screen_passed"], evaluation["fixed10_delta"] > 0)
        positive = {**evaluation, "fixed10_delta": .001, "BCE": 999,
                    "paired_group_bootstrap": {"fixed10": {"high_97.5_percent": -.1}}}
        self.assertTrue(E.decision(positive)["weak_screen_passed"])
        self.assertFalse(E.decision({**positive, "fixed10_delta": None})["weak_screen_passed"])
        self.assertFalse(E.decision({**positive, "fixed10_delta": -.001})["weak_screen_passed"])

    def test_unapproved_template_defaults_disabled_and_output_is_exclusive(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "DISABLED_NO_ACTUAL_ROOT_RELEASE")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "scores.csv"
            E.write_scores(target, [{E.ID: "fake"}], E.np.ones((1, 12), dtype=E.np.float32))
            with self.assertRaises(FileExistsError):
                E.write_scores(target, [{E.ID: "fake"}], E.np.ones((1, 12), dtype=E.np.float32))
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
