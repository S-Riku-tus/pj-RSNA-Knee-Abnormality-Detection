"""Meaningful synthetic regional contracts; no MRI/model acquisition or optimizer."""

import copy
import json
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from rsna_knee.contracts import TARGETS
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.regional_triplet_model import REGIONS_PER_WINDOW, RegionalTripletMIL, pool_region_tokens

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2-fast16.json").read_text(encoding="utf-8"))


class ArtificialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(384))
        self.bias = nn.Parameter(torch.linspace(-0.01, 0.01, 384))
        self.grad_modes = []

    def forward(self, tokens):
        self.grad_modes.append(torch.is_grad_enabled())
        return tokens * self.scale + self.bias


class ArtificialPatchEncoder(nn.Module):
    """Token-shaped differentiable stand-in, not a genuine DINO validation."""

    def __init__(self):
        super().__init__()
        self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384
        self.blocks = nn.ModuleList(ArtificialBlock() for _ in range(12))
        self.patch_embed = nn.Linear(3, 384)
        self.cls_token = nn.Parameter(torch.linspace(-1, 1, 384).reshape(1, 1, -1))
        self.norm = nn.LayerNorm(384)
        self.window_counts = []

    def prepare_tokens_with_masks(self, images):
        self.window_counts.append(len(images))
        patches = F.avg_pool2d(images, kernel_size=14, stride=14).flatten(2).transpose(1, 2)
        # CLS intentionally independent of image: a changed output tests patch information.
        return torch.cat((self.cls_token.expand(len(images), -1, -1), self.patch_embed(patches)), dim=1)


def config():
    value = copy.deepcopy(CONFIG)
    value["encoder"]["precision"] = "fp32"  # Explicit artificial CPU scope only.
    return value


def synthetic_input():
    images = torch.zeros(1, 3, 48, 3, 392, 392, dtype=torch.uint8)
    mask = torch.zeros(1, 3, 48, dtype=torch.bool)
    mask[0, 0, 0] = mask[0, 2, 47] = True
    images[0, 0, 0] = 43
    images[0, 2, 47] = 179
    positions = torch.zeros(1, 3, 48)
    positions[0, 0, 0], positions[0, 2, 47] = 0.1, 0.9
    return {
        "images": images,
        "mask": mask,
        "positions": positions,
        "planes": torch.tensor([[0, 3, 2]]),
        "flags": torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]]),
    }


class RegionalContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)

    def test_exact_disjoint_patch_means_CLS_first_rowmajor_and_storage(self):
        row = torch.arange(28)[:, None]
        col = torch.arange(28)[None, :]
        lattice = (row * 100 + col).float()
        tokens = torch.zeros(2, 785, 384)
        tokens[:, 0] = 0.123456
        tokens[:, 1:] = lattice.flatten()[None, :, None]
        result = pool_region_tokens(tokens)
        self.assertEqual(result.shape, (2, 17, 384))
        self.assertTrue(torch.equal(result[:, 0], tokens[:, 0].half().float()))
        expected = torch.tensor([(r * 7 + 3) * 100 + c * 7 + 3 for r in range(4) for c in range(4)]).half().float()
        self.assertTrue(torch.equal(result[0, 1:, 0], expected))
        self.assertTrue(torch.equal(result[0], result[1]))

    def test_CLS_exact_and_state_shapes_head_parameter_count_unchanged(self):
        encoder = ArtificialPatchEncoder()
        baseline = NativeTripletMIL(copy.deepcopy(encoder), config(), trainable_blocks=0).eval()
        regional = RegionalTripletMIL(copy.deepcopy(encoder), config(), trainable_blocks=0).eval()
        regional.load_state_dict(baseline.state_dict(), strict=True)
        baseline.load_state_dict(regional.state_dict(), strict=True)
        self.assertEqual(
            {k: v.shape for k, v in baseline.state_dict().items()},
            {k: v.shape for k, v in regional.state_dict().items()},
        )
        self.assertEqual(
            baseline.parameter_contract()["trainable_head_parameters"],
            regional.parameter_contract()["trainable_head_parameters"],
        )
        images = torch.rand(2, 3, 392, 392)
        self.assertTrue(torch.equal(baseline.encode_normalized(images), regional.encode_normalized(images)[:, 0]))
        self.assertEqual(regional.parameter_contract()["instances_per_series"], 816)
        self.assertFalse(regional.parameter_contract()["trainer_checkpoint_or_serving_qualified"])

    def test_sparse_window_roworder_mask_position_repeat_and_corrupt_padding(self):
        model = RegionalTripletMIL(ArtificialPatchEncoder(), config(), window_chunk=1).eval()
        inputs = synthetic_input()
        seen = []
        handle = model.head.register_forward_pre_hook(lambda _, args: seen.append(args))
        output = model(**inputs)
        handle.remove()
        features, mask, positions, planes, flags = seen[0]
        self.assertEqual(output.shape, (1, len(TARGETS)))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.encoder.window_counts, [1, 1])
        self.assertEqual(features.shape, (1, 3, 48 * REGIONS_PER_WINDOW, 384))
        self.assertTrue(torch.equal(mask, inputs["mask"].repeat_interleave(17, dim=2)))
        self.assertTrue(torch.equal(positions, inputs["positions"].repeat_interleave(17, dim=2)))
        self.assertTrue(torch.equal(planes, inputs["planes"]))
        self.assertTrue(torch.equal(flags, inputs["flags"]))
        # No compacting across the two selected series or swapping the terminal window.
        for slot, window in ((0, 0), (2, 47)):
            image = inputs["images"][:, slot, window].float() / 255
            mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
            expected = model.encode_normalized((image - mean) / std)[0]
            self.assertTrue(torch.equal(features[0, slot, window * 17 : (window + 1) * 17], expected))
        changed = dict(inputs)
        changed["images"] = inputs["images"].clone()
        changed["images"][~inputs["mask"]] = 254
        changed["positions"] = inputs["positions"].masked_fill(~inputs["mask"], float("nan"))
        self.assertTrue(torch.equal(output, model(**changed)))
        corrupted = features.masked_fill(~mask[..., None], float("nan"))
        self.assertTrue(torch.equal(output, model.head(corrupted, mask, positions, planes, flags)))

    def test_patch_information_changes_logits_while_CLS_stays_equal(self):
        model = RegionalTripletMIL(ArtificialPatchEncoder(), config()).eval()
        inputs = synthetic_input()
        before = model(**inputs)
        changed = dict(inputs)
        changed["images"] = inputs["images"].clone()
        changed["images"][0, 0, 0, :, :98, :98] = 241
        self.assertFalse(torch.equal(before, model(**changed)))
        a = inputs["images"][:, 0, 0].float()
        b = changed["images"][:, 0, 0].float()
        self.assertTrue(torch.equal(model.encode_normalized(a)[:, 0], model.encode_normalized(b)[:, 0]))

    def test_two_study_rows_and_unchanged_twelve_label_columns(self):
        model = RegionalTripletMIL(ArtificialPatchEncoder(), config()).eval()
        one = synthetic_input()
        batch = {key: torch.cat((value, value), dim=0) for key, value in one.items()}
        batch["images"][1, 0, 0, :, :196, :196] = 223
        together = model(**batch)
        self.assertEqual(together.shape, (2, 12))
        for index in (0, 1):
            separate = model(**{key: value[index : index + 1] for key, value in batch.items()})
            torch.testing.assert_close(together[index : index + 1], separate, atol=1e-6, rtol=0)
        self.assertFalse(torch.equal(together[0], together[1]))
        # The untouched head's column-specific output bias reaches the same column.
        with torch.no_grad():
            model.head.output_weight.zero_()
            model.head.output_bias.copy_(torch.arange(len(TARGETS), dtype=torch.float32) / 10)
        expected = model.head.output_bias.detach()[None].expand(2, -1)
        self.assertTrue(torch.equal(model(**batch), expected))

    def test_frozen_and_last2_gradients_only_allowed_suffix_head_and_eval(self):
        inputs = synthetic_input()
        for blocks in (0, 2):
            with self.subTest(blocks=blocks):
                model = RegionalTripletMIL(
                    ArtificialPatchEncoder(), config(), trainable_blocks=blocks, checkpoint_last2=True
                )
                model.train()
                self.assertTrue(model.head.training)
                self.assertFalse(model.encoder.training)
                output = model(**inputs)
                output.sum().backward()
                for name, value in model.encoder.named_parameters():
                    if value.requires_grad:
                        self.assertTrue(name.startswith(("blocks.10.", "blocks.11.")))
                        self.assertIsNotNone(value.grad)
                        self.assertTrue(torch.isfinite(value.grad).all())
                    else:
                        self.assertIsNone(value.grad, name)
                self.assertTrue(
                    all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.head.parameters())
                )
                self.assertTrue(all(not any(block.grad_modes) for block in model.encoder.blocks[: 12 - blocks]))
                self.assertTrue(model.parameter_contract()["final_norm_frozen"])

    def test_checkpoint_suffix_output_and_gradient_match_noncheckpointed(self):
        model = RegionalTripletMIL(ArtificialPatchEncoder(), config(), trainable_blocks=2).eval()
        images = torch.rand(1, 3, 392, 392)
        first = model.encode_normalized(images)
        weight = torch.linspace(-1, 1, 384)[None, None, :]
        (first * weight).sum().backward()
        grads = {name: value.grad.clone() for name, value in model.encoder.named_parameters() if value.requires_grad}
        model.zero_grad(set_to_none=True)
        model.checkpoint_last2 = True
        second = model.encode_normalized(images)
        (second * weight).sum().backward()
        self.assertTrue(torch.equal(first, second))
        for name, value in model.encoder.named_parameters():
            if value.requires_grad:
                self.assertTrue(torch.equal(value.grad, grads[name]), name)

    def test_malformed_and_nonfinite_tokens_storage_and_empty_study_rejected(self):
        for tokens in (
            torch.zeros(1, 784, 384),
            torch.zeros(1, 785, 383),
            torch.full((1, 785, 384), float("nan")),
            torch.full((1, 785, 384), 1e10),
        ):
            with self.subTest(shape=tokens.shape), self.assertRaises(ValueError):
                pool_region_tokens(tokens)
        model = RegionalTripletMIL(ArtificialPatchEncoder(), config()).eval()
        inputs = synthetic_input()
        inputs["mask"][:] = False
        with self.assertRaisesRegex(ValueError, "valid slice"):
            model(**inputs)
        inputs = synthetic_input()
        inputs["images"] = inputs["images"].float()
        with self.assertRaisesRegex(ValueError, "uint8"):
            model(**inputs)


if __name__ == "__main__":
    unittest.main()
