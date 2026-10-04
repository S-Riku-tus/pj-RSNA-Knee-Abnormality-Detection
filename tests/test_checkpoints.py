"""Artificial checkpoint selection, exact step accounting and training neutrality."""

import copy
import importlib.util
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, load_config, sha256, write_csv

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision unavailable")
class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.config = load_config(CONFIG)
        self.config["train"]["checkpoint_milestones"] = [5]
        self.model = torch.nn.Linear(3, 12)

    def tearDown(self):
        self.temp.cleanup()

    def update(self, tracker, epoch, bce, auc):
        import torch

        with torch.no_grad():
            self.model.weight.fill_(epoch)
        epoch_dir = self.out / "epochs" / f"{epoch:03d}"
        rows = [{ID: str(i), **{target: (i + epoch) / (epoch + 2) for target in TARGETS}} for i in range(2)]
        write_csv(epoch_dir / "predictions.csv", SUBMISSION_COLUMNS, rows)
        dump_json(
            epoch_dir / "metrics.json",
            {
                "macro_auc_12": auc if auc is None or math.isfinite(auc) else None,
                "cell_mean_bce": bce if math.isfinite(bce) else None,
            },
        )
        tracker.update(
            self.model,
            epoch,
            bce,
            auc,
            epoch_dir / "predictions.csv",
            epoch_dir / "metrics.json",
            steps=epoch * 2,
            opportunities=epoch * 3,
        )

    def test_auc_ties_undefined_and_checkpoint_prediction_hashes(self):
        import torch

        from rsna_knee.checkpoints import CheckpointTracker

        self.config["train"]["checkpoint_selection"] = "auc"
        tracker = CheckpointTracker(self.out, self.config, 1)
        self.update(tracker, 1, 0.4, None)
        self.assertNotIn("best_auc", tracker.index["checkpoints"])
        self.assertFalse((self.out / "best.pt").exists())
        with self.assertRaisesRegex(ValueError, "defined macro AUC"):
            _ = tracker.selected
        for epoch, bce, auc in [(2, 0.3, 0.7), (3, 0.32, 0.71), (4, 0.31, 0.71), (5, 0.31, 0.71), (6, 0.29, None)]:
            self.update(tracker, epoch, bce, auc)
        self.assertEqual(tracker.selected["epoch"], 4)
        self.assertEqual(tracker.index["checkpoints"]["best_bce"]["epoch"], 6)
        self.assertEqual(tracker.index["checkpoints"]["last"]["epoch"], 6)
        self.assertEqual(tracker.index["checkpoints"]["epoch_005"]["epoch"], 5)
        self.assertEqual(sha256(self.out / "best.pt"), sha256(self.out / "best_auc.pt"))
        for record in [*tracker.index["checkpoints"].values(), tracker.selected]:
            self.assertEqual(sha256(self.out / record["checkpoint"]), record["checkpoint_sha256"])
            self.assertEqual(sha256(self.out / record["predictions_csv"]), record["predictions_sha256"])
            self.assertEqual(sha256(self.out / record["metrics_json"]), record["metrics_sha256"])
            checkpoint = torch.load(self.out / record["checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], record["epoch"])
            self.assertFalse(checkpoint["resume_supported"])
            self.assertEqual(checkpoint["targets"], list(TARGETS))
            self.assertEqual(checkpoint["predictions_sha256"], record["predictions_sha256"])
            self.assertEqual(checkpoint["optimizer_steps"], record["optimizer_steps"])
        checkpoint = torch.load(self.out / "best.pt", map_location="cpu", weights_only=True)
        self.assertTrue(torch.all(checkpoint["model"]["weight"] == 4))
        self.assertEqual(json.loads((self.out / "checkpoint_index.json").read_text()), tracker.index)

    def test_default_bce_keeps_earliest_tie_and_diagnostic_forces_bce(self):
        from rsna_knee.checkpoints import CheckpointTracker, selection_criterion

        tracker = CheckpointTracker(self.out, self.config, 0)
        self.update(tracker, 1, 0.3, 0.6)
        self.update(tracker, 2, 0.3, 0.7)
        self.assertEqual(tracker.selected["criterion"], "bce")
        self.assertEqual(tracker.selected["epoch"], 1)
        self.assertEqual(tracker.index["checkpoints"]["best_auc"]["epoch"], 2)
        self.config["train"]["checkpoint_selection"] = "auc"
        diagnostic = CheckpointTracker(self.out, self.config, 0, diagnostic_mode=True)
        self.assertEqual(diagnostic.criterion, "bce")
        self.assertEqual(diagnostic.index["selection_data"], "training_diagnostic")
        self.config["train"]["checkpoint_selection"] = "gold_auc"
        with self.assertRaises(ValueError):
            selection_criterion(self.config)

    def test_nonfinite_auc_cannot_be_selected_and_invalid_bce_fails(self):
        from rsna_knee.checkpoints import CheckpointTracker

        tracker = CheckpointTracker(self.out, self.config, 0)
        self.update(tracker, 1, 0.3, float("nan"))
        self.assertNotIn("best_auc", tracker.index["checkpoints"])
        self.assertIsNone(tracker.selected["macro_auc_12"])
        with self.assertRaisesRegex(ValueError, "BCE must be finite"):
            self.update(tracker, 2, float("inf"), 0.7)


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision unavailable")
class CheckpointTrainingTests(unittest.TestCase):
    def test_cuda_training_records_selection_progress_and_actual_accumulated_updates(self):
        import numpy as np
        import torch

        from rsna_knee.contracts import preprocess_fingerprint
        from rsna_knee.runtime import train

        if not torch.cuda.is_available() or importlib.util.find_spec("pydicom") is None:
            self.skipTest("CUDA and optional image dependencies required for artificial training")
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, cache, out = root / "manifest", root / "cache", root / "run"
            config = load_config(CONFIG)
            config["preprocess"].update(image_size=32, max_series=1, windows_per_series=2)
            config["train"].update(
                epochs=2,
                batch_size=1,
                accumulation_steps=2,
                num_workers=0,
                amp=False,
                checkpoint_selection="auc",
                checkpoint_milestones=[1, 2],
            )
            config["diagnostics"] = {"enabled": False, "audit_gold": False}
            rows = [
                {
                    ID: f"synthetic{i}",
                    **dict.fromkeys(TARGETS, str(i % 2)),
                    "fold": "1" if i < 2 else "0",
                    "group_id": f"synthetic-group{i}",
                    "source": "report_weak",
                }
                for i in range(4)
            ]
            write_csv(manifest / "weak.csv", [*SUBMISSION_COLUMNS, "fold", "group_id", "source"], rows)
            write_csv(manifest / "gold.csv", [*SUBMISSION_COLUMNS, "group_id", "source"], [])
            input_hash = sha256(manifest / "weak.csv")
            dump_json(
                manifest / "manifest.json",
                {"seed": config["seed"], "n_folds": config["n_folds"], "inputs_sha256": {"train": input_hash}},
            )
            fingerprint = preprocess_fingerprint(config)
            dump_json(
                cache / "cache.json", {"fingerprint": fingerprint, "split": "train", "studies_csv_sha256": input_hash}
            )
            rng = np.random.default_rng(31415)
            for row in rows:
                np.savez_compressed(
                    cache / f"{row[ID]}.npz",
                    images=rng.integers(0, 256, (2, 3, 32, 32), dtype=np.uint8),
                    mask=np.array([True, True]),
                    fingerprint=np.array(fingerprint),
                )
            with redirect_stdout(io.StringIO()):
                result = train(manifest, cache, out, config, 0)
            self.assertEqual((result["optimizer_steps"], result["optimizer_step_opportunities"]), (2, 2))
            self.assertEqual(result["checkpoint_selection"], "auc")
            self.assertFalse(result["diagnostic_mode"])
            history = json.loads((out / "history.json").read_text())
            for epoch, record in enumerate(history, 1):
                self.assertEqual((record["optimizer_steps"], record["optimizer_step_opportunities"]), (1, 1))
                self.assertEqual(record["amp_skipped_steps"], 0)
                self.assertEqual(record["cumulative_optimizer_steps"], epoch)
                self.assertEqual(record["defined_auc_classes"], 12)
            index = json.loads((out / "checkpoint_index.json").read_text())
            self.assertEqual(index["selection_data"], "weak_validation")
            self.assertEqual(index["selected"]["source_checkpoint"], "best_auc.pt")
            self.assertEqual(sha256(out / "best.pt"), sha256(out / "best_auc.pt"))
            self.assertTrue((out / "epoch_001.pt").is_file())
            self.assertTrue((out / "epoch_002.pt").is_file())
            progress = json.loads((out / "progress.json").read_text())
            self.assertEqual(
                (progress["status"], progress["epoch"], progress["completed_microbatches"]), ("completed", 2, 2)
            )
            self.assertEqual(progress["cumulative_optimizer_steps"], 2)

    def test_optimizer_counter_excludes_amp_overflow_skips(self):
        import torch

        from rsna_knee.checkpoints import OptimizerStepCounter

        for device in ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]:
            with self.subTest(device=device):
                parameter = torch.nn.Parameter(torch.ones(2, device=device))
                optimizer = torch.optim.AdamW([parameter], lr=0.01)
                scaler = torch.amp.GradScaler(device, enabled=True)
                counter = OptimizerStepCounter(optimizer)
                for overflow in [False, True, False]:
                    optimizer.zero_grad(set_to_none=True)
                    loss = parameter.sum() * (float("inf") if overflow else 1)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    counter.opportunity()
                    scaler.step(optimizer)
                    scaler.update()
                self.assertEqual((counter.steps, counter.opportunities, counter.skipped), (2, 3, 1))
                self.assertEqual(int(optimizer.state[parameter]["step"]), 2)
                counter.close()

    def test_extra_saves_and_step_hook_preserve_full_training_trajectory(self):
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from rsna_knee.checkpoints import CheckpointTracker, OptimizerStepCounter, write_progress
        from rsna_knee.diagnostics import prediction_diagnostics
        from rsna_knee.runtime import infer_loader, masked_bce

        torch.set_num_threads(1)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)
                self.frozen_bn = nn.BatchNorm1d(3)
                self.dropout = nn.Dropout(0.3)
                self.linear = nn.Linear(3, 12)

            def train(self, mode=True):
                super().train(mode)
                self.frozen_bn.eval()
                return self

            def forward(self, images, mask):
                value = self.frozen_bn(self.bn(images.flatten(0, 1)))
                return self.linear(self.dropout(value.reshape(len(images), -1, 3).mean(1)))

        def run(extra, device, out):
            random.seed(53)
            np.random.seed(53)
            torch.manual_seed(53)
            if device == "cuda":
                torch.cuda.manual_seed_all(53)
            dataset = TensorDataset(
                torch.arange(72, dtype=torch.float32).reshape(6, 4, 3) / 10,
                torch.ones(6, 4, dtype=torch.bool),
                torch.arange(6).remainder(2).float()[:, None].expand(6, 12),
                torch.ones(6, 12, dtype=torch.bool),
                torch.arange(6),
            )
            generator = torch.Generator().manual_seed(53)
            training = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
            validation = DataLoader(dataset, batch_size=2, generator=generator)
            model = TinyModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
            counter = OptimizerStepCounter(optimizer) if extra else None
            config = load_config(CONFIG)
            config["train"].update(checkpoint_selection="auc", checkpoint_milestones=[1, 2, 3])
            tracker = CheckpointTracker(out, config, 0) if extra else None
            order = []
            for epoch in range(3):
                if counter:
                    write_progress(out, epoch + 1, 3, 0, len(training), counter, 0, "training")
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for step, (images, mask, labels, observed, keys) in enumerate(training):
                    order.extend(keys.tolist())
                    group_size = min(2, len(training) - step // 2 * 2)
                    with torch.autocast(device_type=device, enabled=device == "cuda"):
                        loss = masked_bce(
                            model(images.to(device), mask.to(device)), labels.to(device), observed.to(device)
                        )
                    scaler.scale(loss / group_size).backward()
                    if (step + 1) % 2 == 0 or step + 1 == len(training):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        if counter:
                            counter.opportunity()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    if counter and (step + 1 == len(training)):
                        write_progress(out, epoch + 1, 3, step + 1, len(training), counter, 0, "validating")
                predictions, bce, details = infer_loader(model, validation, device, return_details=True)
                if tracker:
                    predictions = [{**row, ID: str(int(row[ID]))} for row in predictions]
                    truth = {str(i): {ID: str(i), **{target: str(i % 2) for target in TARGETS}} for i in range(6)}
                    metrics = prediction_diagnostics(truth, {row[ID]: row for row in predictions}, details)
                    epoch_dir = out / "epochs" / f"{epoch + 1:03d}"
                    write_csv(epoch_dir / "predictions.csv", SUBMISSION_COLUMNS, predictions)
                    dump_json(epoch_dir / "metrics.json", metrics)
                    tracker.update(
                        model,
                        epoch + 1,
                        bce,
                        metrics["macro_auc_12"],
                        epoch_dir / "predictions.csv",
                        epoch_dir / "metrics.json",
                        steps=counter.steps,
                        opportunities=counter.opportunities,
                    )
            if counter:
                write_progress(out, 3, 3, len(training), len(training), counter, 0, "completed")
                self.assertEqual((counter.steps, counter.opportunities), (6, 6))
                self.assertEqual(json.loads((out / "progress.json").read_text())["status"], "completed")
                counter.close()
            return {
                "order": order,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "modes": [module.training for module in model.modules()],
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "scaler": copy.deepcopy(scaler.state_dict()),
                "generator": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            }

        with tempfile.TemporaryDirectory() as temporary:
            for device in ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]:
                with self.subTest(device=device):
                    out = Path(temporary) / device
                    baseline, measured = run(False, device, out), run(True, device, out)
                    for field in ("order", "modes", "python_rng", "scaler"):
                        self.assertEqual(baseline[field], measured[field], field)
                    for field in ("generator", "cpu_rng"):
                        self.assertTrue(torch.equal(baseline[field], measured[field]), field)
                    for left, right in zip(baseline["cuda_rng"], measured["cuda_rng"]):
                        self.assertTrue(torch.equal(left, right))
                    np.testing.assert_array_equal(baseline["numpy_rng"][1], measured["numpy_rng"][1])
                    self.assertEqual(baseline["numpy_rng"][2:], measured["numpy_rng"][2:])
                    for key in baseline["state"]:
                        self.assertTrue(torch.equal(baseline["state"][key], measured["state"][key]), key)
                    for key, left in baseline["optimizer"]["state"].items():
                        right = measured["optimizer"]["state"][key]
                        for field in left:
                            torch.testing.assert_close(left[field], right[field], rtol=0, atol=0)

    def test_saved_milestone_predicts_without_report_or_initialization_file(self):
        import numpy as np
        import torch

        from rsna_knee.checkpoints import CheckpointTracker
        from rsna_knee.contracts import preprocess_fingerprint
        from rsna_knee.model import KneeMIL
        from rsna_knee.runtime import predict

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            config = load_config(CONFIG)
            config["preprocess"].update(image_size=32, max_series=1, windows_per_series=2)
            config["train"]["checkpoint_milestones"] = [5]
            config["model"].update(
                initialization="imagenet1k_v1", pretrained_path="absent.pt", pretrained_sha256="0" * 64
            )
            test_csv, cache = out / "test.csv", out / "cache"
            write_csv(test_csv, [ID], [{ID: "synthetic"}])
            cache.mkdir()
            fingerprint = preprocess_fingerprint(config)
            dump_json(
                cache / "cache.json",
                {"split": "test", "fingerprint": fingerprint, "studies_csv_sha256": sha256(test_csv)},
            )
            np.savez_compressed(
                cache / "synthetic.npz",
                images=np.arange(2 * 3 * 32 * 32).reshape(2, 3, 32, 32).astype(np.uint8),
                mask=np.array([True, True]),
                fingerprint=np.array(fingerprint),
            )
            predictions_csv, metrics_json = out / "epoch_predictions.csv", out / "epoch_metrics.json"
            write_csv(predictions_csv, SUBMISSION_COLUMNS, [{ID: "synthetic", **dict.fromkeys(TARGETS, 0.5)}])
            dump_json(metrics_json, {"cell_mean_bce": 0.3, "macro_auc_12": 0.7})
            tracker = CheckpointTracker(out, config, 0)
            tracker.update(KneeMIL(), 5, 0.3, 0.7, predictions_csv, metrics_json, steps=5, opportunities=5)
            for name in ("best.pt", "best_bce.pt", "best_auc.pt", "last.pt", "epoch_005.pt"):
                result = predict(test_csv, cache, out / name, out / f"{name}.csv", "cpu")
                self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
