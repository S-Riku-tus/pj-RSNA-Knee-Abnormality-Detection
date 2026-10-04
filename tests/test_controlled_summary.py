"""CSV/JSON-only comparison audits; no MRI, torch, weights or GPU execution."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, sha256, source_hashes, write_csv
from rsna_knee.diagnostics import prediction_diagnostics

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_controlled_runs.py"
SPEC = importlib.util.spec_from_file_location("controlled_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class ControlledSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest"
        self.manifest.mkdir()
        self.output = self.root / "summary.json"
        self.runs = [self.root / name for name in ("A", "B", "C")]
        self.weak = [
            {
                ID: f"private-fixture-{i}",
                **{t: str(i % 2) for t in TARGETS},
                "fold": str(i // 2),
                "group_id": f"group-{i}",
                "source": "report_weak",
            }
            for i in range(4)
        ]
        self.weak[0][TARGETS[-1]] = ""  # Undefined final-class AUC and a real observation-mask difference.
        write_csv(self.manifest / "weak.csv", (*SUBMISSION_COLUMNS, "fold", "group_id", "source"), self.weak)
        write_csv(
            self.manifest / "gold.csv",
            (*SUBMISSION_COLUMNS, "group_id", "source"),
            [
                {
                    ID: "private-gold-fixture",
                    **{t: "1" for t in TARGETS},
                    "group_id": "gold-group",
                    "source": "official_gold",
                }
            ],
        )
        manifest = {"seed": 123, "n_folds": 2, "inputs_sha256": {"train": "synthetic-train", "weak": "synthetic-weak"}}
        dump_json(self.manifest / "manifest.json", manifest)
        hashes = {name: sha256(self.manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv")}
        common = {
            "seed": 123,
            "n_folds": 2,
            "preprocess": {"image_size": 32},
            "model": {"initialization": "random", "input_normalization": "legacy", "dropout": 0.2},
            "train": {"epochs": 2, "learning_rate": 0.001},
            "diagnostics": {"enabled": True, "audit_gold": False, "train_eval_studies": 2},
        }
        for condition, run in zip("ABC", self.runs):
            run.mkdir()
            config = copy.deepcopy(common)
            if condition in "BC":
                config["model"]["input_normalization"] = "imagenet"
            if condition == "C":
                config["model"].update(
                    initialization="imagenet1k_v1",
                    pretrained_path="synthetic-local-only.pth",
                    pretrained_sha256="f37072fd" + "0" * 56,
                )
            original = self.root / f"{condition}-config.json"
            original.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
            dump_json(run / "config.json", config)
            write_csv(run / "diagnostic_subset.csv", (ID,), [{ID: r[ID]} for r in self.weak[2:]])
            dump_json(
                run / "diagnostic_selection.json",
                {"subset_csv_sha256": sha256(run / "diagnostic_subset.csv"), "studies": 2, "seed": 123, "fold": 0},
            )
            initialization = {"name": config["model"]["initialization"]}
            if condition == "C":
                initialization["sha256"] = config["model"]["pretrained_sha256"]
            metadata = {
                "diagnostic_mode": False,
                "gold_audit_enabled": False,
                "fold": 0,
                "manifest": manifest,
                "source_sha256": source_hashes(),
                "manifest_sha256": hashes,
                "train_studies": 2,
                "valid_studies": 2,
                "cache": {
                    "studies_csv_sha256": "synthetic-train",
                    "fingerprint": "same-cache",
                    "preprocess": config["preprocess"],
                },
                "checkpoint_selection": "minimum weak validation masked BCE; gold never used for selection",
                "initialization": initialization,
                "initial_head_sha256": "same-head",
                "runtime": {"python": "synthetic"},
                "gpu": "synthetic-unused",
            }
            dump_json(run / "run.json", metadata)
            truth = {r[ID]: r for r in self.weak[:2]}
            history = []
            for epoch, correct_probability in enumerate((0.8, 0.7), 1):
                predicted = {
                    r[ID]: {ID: r[ID], **{t: correct_probability if i else 1 - correct_probability for t in TARGETS}}
                    for i, r in enumerate(self.weak[:2])
                }
                metrics = prediction_diagnostics(truth, predicted)
                loss = -math.log(correct_probability)
                metrics.update(cell_mean_bce=loss, observed_target_mean_bce=loss, observed_cells=23)
                for t, cell in metrics["per_target"].items():
                    cell["masked_bce"] = loss
                directory = run / "epochs" / f"{epoch:03d}"
                dump_json(directory / "metrics.json", metrics)
                write_csv(directory / "predictions.csv", SUBMISSION_COLUMNS, predicted.values())
                history.append(
                    {
                        "epoch": epoch,
                        "weak_valid_masked_bce": loss,
                        "weak_valid_macro_auc_12": metrics["macro_auc_12"],
                        "defined_auc_classes": metrics["defined_classes"],
                        "elapsed_seconds": epoch * 2.0,
                    }
                )
            dump_json(run / "history.json", history)
            (run / "weak_valid_predictions.csv").write_bytes((run / "epochs/001/predictions.csv").read_bytes())
            (run / "best.pt").write_bytes(b"artificial opaque checkpoint; never deserialized")
            execution = {
                "status": "completed",
                "config": config,
                "config_path": str(original),
                "config_file_sha256": sha256(original),
                "fold": 0,
                "diagnostic_studies": None,
                "source_sha256": metadata["source_sha256"],
                "result": {"best_weak_valid_bce": history[0]["weak_valid_masked_bce"]},
                "elapsed_seconds": 5.0,
                "peak_allocated_bytes": 0,
                "git_head_at_start": "synthetic",
                "elapsed_scope": "synthetic fixture only",
            }
            dump_json(run / "execution.json", execution)

    def summarize(self):
        return SUMMARY.summarize(self.runs, self.output, self.manifest)

    def mutate(self, path, transform):
        value = json.loads(path.read_text(encoding="utf-8"))
        transform(value)
        dump_json(path, value)

    def test_valid_comparison_preserves_missing_labels_and_private_identifiers(self):
        report = self.summarize()
        self.assertTrue(report["validation"]["valid"])
        self.assertIsNone(report["selected_winner"])
        self.assertIsNone(report["gold_macro_auc_12"])
        self.assertIsNone(report["public_lb"])
        for run in report["runs"]:
            self.assertEqual(run["best_epoch_by_weak_bce"], 1)
            self.assertEqual(run["best_metrics"]["defined_classes"], 11)
            self.assertEqual(run["best_metrics"]["observed_cells"], 23)
            self.assertIsNone(run["best_metrics"]["macro_auc_12"])
            self.assertNotEqual(run["original_config_file_sha256"], run["saved_config_file_sha256"])
            self.assertFalse(run["finish_source_hash_recorded"])
            self.assertIsNone(run["finish_source_matches_start"])
        serialized = self.output.read_text(encoding="utf-8")
        self.assertNotIn("private-fixture", serialized)
        self.assertNotIn("private-gold-fixture", serialized)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.summarize()
        with self.subTest(finish_source="matching optional finish hashes"):
            for run in self.runs:
                self.mutate(run / "execution.json", lambda d: d.update(source_sha256_at_finish=d["source_sha256"]))
            finished = SUMMARY.summarize(self.runs, self.root / "summary-with-finish.json", self.manifest)
            for run in finished["runs"]:
                self.assertTrue(run["finish_source_hash_recorded"])
                self.assertTrue(run["finish_source_matches_start"])

    def test_incomplete_and_source_changed_runs_rejected_before_output(self):
        execution = self.runs[0] / "execution.json"
        original = execution.read_bytes()
        for corruption, message in (
            (lambda d: d.update(status="running"), "not completed"),
            (lambda d: d.update(source_sha256={"changed": "source"}), "Source changed"),
            (lambda d: d.update(source_sha256_at_finish={"changed": "source"}), "Source changed before completion"),
        ):
            with self.subTest(message=message):
                execution.write_bytes(original)
                self.mutate(execution, corruption)
                with self.assertRaisesRegex(ValueError, message):
                    self.summarize()
                self.assertFalse(self.output.exists())

    def test_second_factor_in_config_rejected(self):
        run = self.runs[1]
        saved = json.loads((run / "config.json").read_text())
        saved["train"]["learning_rate"] = 0.002
        dump_json(run / "config.json", saved)
        source = self.root / "B-config.json"
        dump_json(source, saved)
        self.mutate(run / "execution.json", lambda d: d.update(config=saved, config_file_sha256=sha256(source)))
        with self.assertRaisesRegex(ValueError, "A/B must differ only"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_manifest_and_observed_counts_tampering_rejected(self):
        path = self.runs[0] / "epochs/001/metrics.json"
        self.mutate(path, lambda d: d["per_target"][TARGETS[0]].update(observed=1))
        with self.assertRaisesRegex(ValueError, "Inconsistent epoch metrics"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_best_csv_and_validation_study_set_tampering_rejected(self):
        best = self.runs[0] / "weak_valid_predictions.csv"
        best.write_bytes((self.runs[0] / "epochs/002/predictions.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "first minimum-BCE"):
            self.summarize()
        self.assertFalse(self.output.exists())
        best.write_bytes((self.runs[0] / "epochs/001/predictions.csv").read_bytes())
        epoch = self.runs[0] / "epochs/001/predictions.csv"
        epoch.write_text(epoch.read_text().replace("private-fixture-0", "nonvalidation-fixture"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "study set differs"):
            self.summarize()

    def test_nonfinite_predictions_and_nontraining_diagnostic_subset_rejected(self):
        epoch = self.runs[0] / "epochs/001/predictions.csv"
        original = epoch.read_bytes()
        epoch.write_text(epoch.read_text().replace("0.8", "nan"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite probability"):
            self.summarize()
        epoch.write_bytes(original)
        subset = self.runs[0] / "diagnostic_subset.csv"
        write_csv(subset, (ID,), [{ID: self.weak[0][ID]}])
        with self.assertRaisesRegex(ValueError, "non-training"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_different_initial_head_and_original_config_change_rejected(self):
        metadata = self.runs[2] / "run.json"
        original = metadata.read_bytes()
        self.mutate(metadata, lambda d: d.update(initial_head_sha256="different-head"))
        with self.assertRaisesRegex(ValueError, "initial_head_sha256 differs"):
            self.summarize()
        metadata.write_bytes(original)
        source = self.root / "C-config.json"
        source.write_text(source.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Original config file changed"):
            self.summarize()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
