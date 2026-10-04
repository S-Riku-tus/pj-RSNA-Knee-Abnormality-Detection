"""Extra evaluation must preserve training order, updates, buffers and RNG."""

import copy
import importlib.util
import random
import unittest

from rsna_knee.contracts import ID, TARGETS
from rsna_knee.diagnostics import prediction_diagnostics

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


class ProbabilityDiagnosticsTests(unittest.TestCase):
    def test_missing_soft_and_undefined_classes_are_explicit(self):
        truth = {
            str(i): {ID: str(i), **{target: value for target in TARGETS}}
            for i, value in enumerate(("0", "1", "0.3", ""))
        }
        predictions = {
            str(i): {ID: str(i), **{target: score for target in TARGETS}}
            for i, score in enumerate((0.2, 0.8, 0.7, 0.9))
        }
        result = prediction_diagnostics(truth, predictions)
        self.assertEqual(result["macro_auc_12"], 1)
        for value in result["per_target"].values():
            self.assertEqual((value["observed"], value["missing"], value["soft_excluded"]), (3, 1, 1))
            self.assertEqual((value["positive"], value["negative"]), (1, 1))
            self.assertEqual(value["predictions"]["count"], 4)
        for row in truth.values():
            row[TARGETS[0]] = ""
        result = prediction_diagnostics(truth, predictions)
        self.assertIsNone(result["macro_auc_12"])
        self.assertEqual(result["defined_classes"], 11)


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision unavailable")
class TrainingNeutralityTests(unittest.TestCase):
    def test_extra_diagnostics_preserve_full_training_trajectory(self):
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from rsna_knee.runtime import diagnostic_inference, infer_loader

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

        def run(extra, device):
            random.seed(53)
            np.random.seed(53)
            torch.manual_seed(53)
            if device == "cuda":
                torch.cuda.manual_seed_all(53)
            images = torch.arange(72, dtype=torch.float32).reshape(6, 4, 3) / 10
            dataset = TensorDataset(
                images,
                torch.ones(6, 4, dtype=torch.bool),
                torch.zeros(6, 12),
                torch.ones(6, 12, dtype=torch.bool),
                torch.arange(6),
            )
            generator = torch.Generator().manual_seed(53)
            training = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
            validation = DataLoader(dataset, batch_size=2, generator=generator)
            diagnostic = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(54))
            model = TinyModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            order = []
            for _ in range(2):
                model.train()
                for batch, mask, labels, observed, keys in training:
                    order.extend(keys.tolist())
                    optimizer.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(batch.to(device), mask.to(device)), labels.to(device)
                    )
                    loss.backward()
                    optimizer.step()
                infer_loader(model, validation, device)
                model.train()  # Includes mixed train/eval BN modes that must survive extra evaluation.
                if extra:
                    before = [module.training for module in model.modules()]
                    diagnostic_inference(model, diagnostic, device)
                    generator_state = generator.get_state().clone()
                    diagnostic_inference(model, validation, device)
                    self.assertTrue(torch.equal(generator_state, generator.get_state()))
                    self.assertEqual(before, [module.training for module in model.modules()])
            return {
                "order": order,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "generator": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
            }

        for device in ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]:
            with self.subTest(device=device):
                baseline, measured = run(False, device), run(True, device)
                self.assertEqual(baseline["order"], measured["order"])
                self.assertEqual(baseline["python_rng"], measured["python_rng"])
                for name in ("generator", "cpu_rng"):
                    self.assertTrue(torch.equal(baseline[name], measured[name]))
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

    def test_detailed_loss_preserves_canonical_masked_bce(self):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from rsna_knee.runtime import infer_loader

        class Output(nn.Module):
            def forward(self, images, mask):
                return images

        labels = torch.zeros(3, 12)
        labels[0] = 1
        observed = torch.ones(3, 12, dtype=torch.bool)
        observed[1, :6] = False
        observed[:, -1] = False
        data = TensorDataset(
            torch.arange(36).reshape(3, 12).float() / 10, torch.ones(3, 1), labels, observed, torch.arange(3)
        )
        loader = DataLoader(data, batch_size=2)
        rows, old_loss = infer_loader(Output(), loader, "cpu")
        new_rows, new_loss, details = infer_loader(Output(), loader, "cpu", return_details=True)
        self.assertEqual(rows, new_rows)
        self.assertEqual(old_loss, new_loss)
        self.assertEqual(details["observed_cells"], int(observed.sum()))
        self.assertIsNone(details["per_target"][TARGETS[-1]]["masked_bce"])


if __name__ == "__main__":
    unittest.main()
