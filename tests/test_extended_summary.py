"""Artificial CSV/JSON audits for predeclared checkpoint comparisons; standard library only."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, sha256, source_hashes, write_csv
from rsna_knee.diagnostics import prediction_diagnostics

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_extended_runs.py"
SPEC = importlib.util.spec_from_file_location("extended_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class ExtendedSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_dir = self.root / "manifest"
        self.manifest_dir.mkdir()
        self.output = self.root / "summary.json"
        self.runs = [self.root / name for name in SUMMARY.CONDITIONS]
        self.weak = [
            {
                ID: f"private-weak-fixture-{fold}-{index}",
                **{target: str(index % 2) for target in TARGETS},
                "fold": str(fold),
                "group_id": f"private-group-{fold}-{index}",
                "source": "report_weak",
            }
            for fold in (0, 1)
            for index in range(3)
        ]
        for row in (self.weak[2], self.weak[5]):
            row[TARGETS[-1]] = ""
            row[TARGETS[1]] = "0.2"
        write_csv(self.manifest_dir / "weak.csv", (*SUBMISSION_COLUMNS, "fold", "group_id", "source"), self.weak)
        write_csv(
            self.manifest_dir / "gold.csv",
            (*SUBMISSION_COLUMNS, "group_id", "source"),
            [
                {
                    ID: "private-gold-fixture",
                    **{target: "1" for target in TARGETS},
                    "group_id": "gold-group",
                    "source": "official_gold",
                }
            ],
        )
        self.manifest = {
            "seed": 123,
            "n_folds": 2,
            "inputs_sha256": {"train": "synthetic-train", "weak": "synthetic-weak"},
        }
        dump_json(self.manifest_dir / "manifest.json", self.manifest)
        self.hashes = {name: sha256(self.manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv")}
        self.config = {
            "seed": 123,
            "n_folds": 2,
            "preprocess": {"image_size": 32},
            "model": {
                "initialization": "random",
                "input_normalization": "imagenet",
                "bn_running_stats": "update",
                "dropout": 0.2,
            },
            "train": {"epochs": 5, "batch_size": 1, "accumulation_steps": 2, "learning_rate": 0.001},
            "diagnostics": {"enabled": True, "audit_gold": False, "train_eval_studies": 2},
        }
        for index, run in enumerate(self.runs):
            config = copy.deepcopy(self.config)
            if index:
                config["model"].update(
                    initialization="imagenet1k_v1",
                    pretrained_path="artificial-local-only.pth",
                    pretrained_sha256="f37072fd" + "0" * 56,
                )
            if index >= 2:
                config["train"].update(
                    epochs=20, checkpoint_selection="auc", checkpoint_milestones=list(SUMMARY.MILESTONES)
                )
            if index == 3:
                config["model"]["bn_running_stats"] = "freeze"
            self.make_run(run, config, 1 if index < 2 else 0)

    def make_run(self, run, config, fold):
        run.mkdir()
        original_config = self.root / f"{run.name}-config.json"
        original_config.write_text(json.dumps(config, separators=(",", ":")), encoding="utf-8")
        dump_json(run / "config.json", config)
        criterion = config["train"].get("checkpoint_selection", "bce")
        training = [row for row in self.weak if int(row["fold"]) != fold]
        truth = {row[ID]: row for row in self.weak if int(row["fold"]) == fold}
        write_csv(run / "diagnostic_subset.csv", (ID,), [{ID: row[ID]} for row in training[:2]])
        dump_json(
            run / "diagnostic_selection.json",
            {
                "subset_csv_sha256": sha256(run / "diagnostic_subset.csv"),
                "studies": 2,
                "seed": 123,
                "fold": fold,
            },
        )
        initialization = {"name": config["model"]["initialization"]}
        if initialization["name"] == "imagenet1k_v1":
            initialization["sha256"] = config["model"]["pretrained_sha256"]
        metadata = {
            "diagnostic_mode": False,
            "gold_audit_enabled": False,
            "fold": fold,
            "manifest": self.manifest,
            "manifest_sha256": self.hashes,
            "source_sha256": source_hashes(),
            "train_studies": 3,
            "valid_studies": 3,
            "cache": {
                "studies_csv_sha256": "synthetic-train",
                "fingerprint": "same-cache",
                "preprocess": config["preprocess"],
            },
            "checkpoint_selection": SUMMARY.SELECTION_RULES[criterion],
            "checkpoint_selection_criterion": criterion,
            "checkpoint_milestones": list(SUMMARY.MILESTONES),
            "checkpoint_resume_supported": False,
            "initialization": initialization,
            "initial_head_sha256": "same-head",
            "runtime": {"python": "artificial"},
            "gpu": "artificial-unused",
        }
        dump_json(run / "run.json", metadata)
        history, epochs = [], []
        total_steps = 0
        for epoch in range(1, config["train"]["epochs"] + 1):
            q = 0.9 if epoch in (2, 5) else 0.8 if epoch >= 4 else 0.7
            predicted = {}
            for row in truth.values():
                predicted[row[ID]] = {ID: row[ID]}
                for target in TARGETS:
                    label = row[target]
                    p = q if label == "1" else 1 - q
                    if target == TARGETS[0] and epoch in (2, 5):
                        p = 0.49 if label == "1" else 0.51
                    if label == "0.2":
                        p = 0.3
                    predicted[row[ID]][target] = p
            metrics = prediction_diagnostics(truth, predicted)
            total, count, mean_losses = 0.0, 0, []
            for target in TARGETS:
                losses = [
                    -float(row[target]) * math.log(predicted[row[ID]][target])
                    - (1 - float(row[target])) * math.log1p(-predicted[row[ID]][target])
                    for row in truth.values()
                    if row[target] != ""
                ]
                loss = sum(losses) / len(losses)
                metrics["per_target"][target]["masked_bce"] = loss
                total += sum(losses)
                count += len(losses)
                mean_losses.append(loss)
            metrics.update(
                cell_mean_bce=total / count, observed_cells=count, observed_target_mean_bce=sum(mean_losses) / 12
            )
            directory = run / "epochs" / f"{epoch:03d}"
            dump_json(directory / "metrics.json", metrics)
            write_csv(directory / "predictions.csv", SUBMISSION_COLUMNS, predicted.values())
            train_truth = {row[ID]: row for row in training[:2]}
            train_predictions = {
                row[ID]: {**scores, ID: row[ID]} for row, scores in zip(training[:2], list(predicted.values())[:2])
            }
            train_metrics = prediction_diagnostics(train_truth, train_predictions)
            train_losses = []
            for target in TARGETS:
                losses = [
                    -float(row[target]) * math.log(train_predictions[row[ID]][target])
                    - (1 - float(row[target])) * math.log1p(-train_predictions[row[ID]][target])
                    for row in train_truth.values()
                ]
                train_metrics["per_target"][target]["masked_bce"] = sum(losses) / 2
                train_losses.extend(losses)
            train_metrics.update(
                cell_mean_bce=sum(train_losses) / len(train_losses),
                observed_target_mean_bce=sum(train_losses) / len(train_losses),
                observed_cells=len(train_losses),
            )
            dump_json(directory / "train_eval_metrics.json", train_metrics)
            write_csv(directory / "train_eval_predictions.csv", SUBMISSION_COLUMNS, train_predictions.values())
            steps, skipped = (1, 1) if epoch == 2 else (2, 0)
            total_steps += steps
            item = {
                "epoch": epoch,
                "weak_valid_masked_bce": total / count,
                "weak_valid_macro_auc_12": metrics["macro_auc_12"],
                "defined_auc_classes": metrics["defined_classes"],
                "elapsed_seconds": epoch * 2.0,
                "optimizer_steps": steps,
                "optimizer_step_opportunities": 2,
                "amp_skipped_steps": skipped,
                "cumulative_optimizer_steps": total_steps,
                "cumulative_optimizer_step_opportunities": epoch * 2,
                "cumulative_amp_skipped_steps": int(epoch >= 2),
                "train_eval_cell_mean_bce": train_metrics["cell_mean_bce"],
                "train_eval_observed_target_mean_bce": train_metrics["observed_target_mean_bce"],
            }
            history.append(item)
            epochs.append(
                {
                    "epoch": epoch,
                    "weak_valid_bce": total / count,
                    "weak_macro_auc_12": metrics["macro_auc_12"],
                    "item": item,
                }
            )
        dump_json(run / "history.json", history)
        bce, auc = SUMMARY.best_bce(epochs), SUMMARY.best_auc(epochs)
        expected = {"best_bce": bce, "best_auc": auc, "last": epochs[-1]}
        for milestone in SUMMARY.MILESTONES:
            if milestone <= len(epochs):
                expected[f"epoch_{milestone:03d}"] = epochs[milestone - 1]
        entries = {}
        for name, row in expected.items():
            (run / f"{name}.pt").write_bytes(f"opaque artificial model epoch {row['epoch']}".encode())
            entries[name] = self.checkpoint_entry(run, row, f"{name}.pt")
        selected_row = bce if criterion == "bce" else auc
        (run / "best.pt").write_bytes((run / f"best_{criterion}.pt").read_bytes())
        (run / "weak_valid_predictions.csv").write_bytes(
            (run / f"epochs/{selected_row['epoch']:03d}/predictions.csv").read_bytes()
        )
        selected = self.checkpoint_entry(run, selected_row, "best.pt")
        selected.update(
            predictions_csv="weak_valid_predictions.csv",
            source_checkpoint=f"best_{criterion}.pt",
            source_predictions_csv=f"epochs/{selected_row['epoch']:03d}/predictions.csv",
            criterion=criterion,
        )
        dump_json(
            run / "checkpoint_index.json",
            {
                "schema_version": 1,
                "checkpoint_selection": criterion,
                "selection_data": "weak_validation",
                "resume_supported": False,
                "checkpoints": entries,
                "selected": selected,
            },
        )
        dump_json(
            run / "execution.json",
            {
                "status": "completed",
                "config": config,
                "config_path": str(original_config),
                "config_file_sha256": sha256(original_config),
                "fold": fold,
                "diagnostic_studies": None,
                "source_sha256": metadata["source_sha256"],
                "source_sha256_at_finish": metadata["source_sha256"],
                "result": {
                    "best_weak_valid_bce": bce["weak_valid_bce"],
                    "selected_epoch": selected_row["epoch"],
                    "selected_masked_bce": selected_row["weak_valid_bce"],
                    "selected_macro_auc_12": selected_row["weak_macro_auc_12"],
                    "checkpoint_selection": criterion,
                    "optimizer_steps": total_steps,
                    "optimizer_step_opportunities": len(epochs) * 2,
                    "amp_skipped_steps": 1,
                },
                "elapsed_seconds": 50.0,
                "peak_allocated_bytes": 0,
                "git_head_at_start": "artificial",
                "elapsed_scope": "artificial fixture",
            },
        )

    @staticmethod
    def checkpoint_entry(run, row, checkpoint):
        directory = f"epochs/{row['epoch']:03d}"
        return {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(run / checkpoint),
            "epoch": row["epoch"],
            "masked_bce": row["weak_valid_bce"],
            "macro_auc_12": row["weak_macro_auc_12"],
            "predictions_csv": f"{directory}/predictions.csv",
            "predictions_sha256": sha256(run / directory / "predictions.csv"),
            "metrics_json": f"{directory}/metrics.json",
            "metrics_sha256": sha256(run / directory / "metrics.json"),
            "optimizer_steps": row["item"]["cumulative_optimizer_steps"],
            "optimizer_step_opportunities": row["item"]["cumulative_optimizer_step_opportunities"],
        }

    def mutate(self, path, transform):
        value = json.loads(path.read_text(encoding="utf-8"))
        transform(value)
        dump_json(path, value)

    def summarize(self, legacy=None):
        return SUMMARY.summarize(self.runs, self.output, self.manifest_dir, legacy)

    def test_complete_comparison_audits_both_selection_rules_and_actual_steps(self):
        self.mutate(self.runs[0] / "epochs/001/metrics.json", lambda value: value.update(Report="private-report"))
        self.mutate(
            self.runs[0] / "checkpoint_index.json",
            lambda value: value["checkpoints"]["best_bce"].update(Report="private-report"),
        )
        report = self.summarize()
        self.assertTrue(report["validation"]["valid"])
        self.assertIsNone(report["selected_winner"])
        for index, run in enumerate(report["runs"]):
            self.assertEqual(run["best_epoch_by_weak_bce"], 2)
            self.assertEqual(run["best_epoch_by_weak_auc"], 4)
            self.assertEqual(run["selected_epoch"], 2 if index < 2 else 4)
            self.assertEqual(run["actual_optimizer_steps"], len(run["epochs"]) * 2 - 1)
            self.assertEqual(run["amp_skipped_steps"], 1)
            self.assertEqual(run["selected_metrics"]["per_target"][TARGETS[-1]]["missing"], 1)
            self.assertEqual(run["selected_metrics"]["per_target"][TARGETS[1]]["soft_excluded"], 1)
        self.assertEqual([row["epoch"] for row in report["runs"][2]["milestones"]], [5, 10, 15, 20])
        self.assertEqual(report["comparisons"]["fold0_bn_statistics"]["selected_epoch_difference"]["macro_auc_12"], 0)
        text = self.output.read_text(encoding="utf-8")
        for private in ("private-weak-fixture", "private-gold-fixture", "private-group", "private-report"):
            self.assertNotIn(private, text)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.summarize()

    def test_selection_rejects_wrong_best_auc_tie_epoch_and_weight_corruption(self):
        path = self.runs[2] / "checkpoint_index.json"
        original = path.read_bytes()
        self.mutate(path, lambda value: value["checkpoints"]["best_auc"].update(epoch=1))
        with self.assertRaisesRegex(ValueError, "checkpoint epoch"):
            self.summarize()
        self.assertFalse(self.output.exists())
        path.write_bytes(original)
        (self.runs[2] / "best_auc.pt").write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "Checkpoint file hash"):
            self.summarize()

    def test_steps_finish_source_and_manifest_tampering_rejected(self):
        history = self.runs[0] / "history.json"
        original = history.read_bytes()
        self.mutate(history, lambda values: values[1].update(cumulative_optimizer_steps=4))
        with self.assertRaisesRegex(ValueError, "cumulative optimizer_steps"):
            self.summarize()
        history.write_bytes(original)
        execution = self.runs[0] / "execution.json"
        original_execution = execution.read_bytes()
        self.mutate(execution, lambda value: value.update(source_sha256_at_finish={"changed": "source"}))
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.summarize()
        execution.write_bytes(original_execution)
        self.mutate(self.manifest_dir / "manifest.json", lambda value: value.update(seed=456))
        with self.assertRaisesRegex(ValueError, "Run manifest metadata"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_prediction_and_loss_count_tampering_rejected(self):
        metrics = self.runs[0] / "epochs/001/metrics.json"
        original = metrics.read_bytes()
        self.mutate(metrics, lambda value: value["per_target"][TARGETS[-1]].update(observed=3))
        with self.assertRaisesRegex(ValueError, "Inconsistent epoch metrics"):
            self.summarize()
        metrics.write_bytes(original)
        self.mutate(metrics, lambda value: value["per_target"][TARGETS[0]].update(masked_bce=1.0))
        with self.assertRaisesRegex(ValueError, "target-mean BCE"):
            self.summarize()
        metrics.write_bytes(original)
        predicted = self.runs[0] / "epochs/001/predictions.csv"
        predicted.write_text(
            predicted.read_text().replace("private-weak-fixture-1-0", "outside-validation"), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "study set differs"):
            self.summarize()

    def test_bn_second_factor_rejected_even_with_consistent_config_records(self):
        run = self.runs[3]
        config = json.loads((run / "config.json").read_text())
        config["train"]["learning_rate"] *= 2
        original_config = self.root / f"{run.name}-config.json"
        dump_json(original_config, config)
        dump_json(run / "config.json", config)
        self.mutate(
            run / "execution.json",
            lambda value: value.update(config=config, config_file_sha256=sha256(original_config)),
        )
        with self.assertRaisesRegex(ValueError, "must differ only in BN"):
            self.summarize()

    def test_milestone_and_selected_index_tampering_rejected(self):
        path = self.runs[2] / "checkpoint_index.json"
        original = path.read_bytes()
        self.mutate(path, lambda value: value["checkpoints"].pop("epoch_010"))
        with self.assertRaisesRegex(ValueError, "Checkpoint index entries"):
            self.summarize()
        path.write_bytes(original)
        self.mutate(path, lambda value: value["selected"].update(source_checkpoint="best_bce.pt"))
        with self.assertRaisesRegex(ValueError, "Selected checkpoint source"):
            self.summarize()
        self.assertFalse(self.output.exists())

    def test_undefined_auxiliary_auc_keeps_fixed_denominators(self):
        metrics = json.loads((self.runs[0] / "epochs/001/metrics.json").read_text())
        metrics["per_target"][TARGETS[0]]["auc"] = None
        summary = SUMMARY.metric_summary(metrics)
        self.assertIsNone(summary["macro_auc_11_without_synovitis"])
        self.assertEqual(summary["auxiliary_defined_classes"], 10)
        rows = [{"epoch": 2, "weak_valid_bce": 0.1, "weak_macro_auc_12": None}]
        self.assertIsNone(SUMMARY.best_auc(rows))

    def test_legacy_csv_trajectory_exact_match_allows_recorded_source_change(self):
        legacy = self.root / "historical_C"
        config = json.loads((self.runs[2] / "config.json").read_text())
        config["train"]["epochs"] = 5
        config["train"].pop("checkpoint_selection")
        config["train"].pop("checkpoint_milestones")
        self.make_run(legacy, config, 0)
        self.mutate(
            legacy / "run.json", lambda value: value.update(source_sha256={"historical.py": "old-recorded-source"})
        )
        self.mutate(
            legacy / "execution.json",
            lambda value: value.update(
                source_sha256={"historical.py": "old-recorded-source"},
                source_sha256_at_finish={"historical.py": "old-recorded-source"},
            ),
        )
        report = self.summarize(legacy)
        reference = report["legacy_C_first_five"]
        self.assertTrue(reference["first_five_epochs_exact_match"])
        self.assertFalse(reference["historical_source_matches_current"])
        self.assertFalse(reference["legacy_current_source_match_required"])
        self.output.unlink()
        predicted = legacy / "epochs/005/predictions.csv"
        predicted.write_text(predicted.read_text().replace("0.9", "0.91"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "legacy epoch metrics"):
            self.summarize(legacy)


if __name__ == "__main__":
    unittest.main()
