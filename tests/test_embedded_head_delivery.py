"""Check integrity, fresh outputs and unchanged inference when bypassing a head Input."""

import base64
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materializer = load_script("embedded_head_payload")
builder = load_script("build_f002_embedded_head_notebook")


class EmbeddedHeadDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.destination = Path(self.temporary.name) / "head"
        self.raw = {
            "best.pt": b"artificial-checkpoint",
            "config.json": b'{"artificial":true}',
            "checkpoint-provenance.json": b'{"source":"artificial"}',
        }
        self.payload = {
            name: {"base64": base64.b64encode(raw).decode(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in self.raw.items()
        }

    def test_verified_bundle_restores_exact_bytes_and_receipt(self):
        receipt = materializer.materialize_embedded_head(self.payload, self.destination)
        self.assertEqual({path.name: path.read_bytes() for path in self.destination.iterdir()}, self.raw)
        self.assertEqual(Path(receipt["checkpoint_path"]), self.destination / "best.pt")
        self.assertFalse(receipt["head_dataset_required"])
        self.assertFalse(receipt["downloaded"])

    def test_invalid_base64_hash_or_size_fail_before_any_output(self):
        for change in ({"base64": "%%bad"}, {"sha256": "a" * 64}, {"bytes": 1}, {"bytes": True}):
            with self.subTest(change=change):
                payload = copy.deepcopy(self.payload)
                payload["checkpoint-provenance.json"].update(change)
                with self.assertRaises(ValueError):
                    materializer.materialize_embedded_head(payload, self.destination)
                self.assertFalse(self.destination.exists())

    def test_extra_path_and_existing_destination_are_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["../best.pt"] = payload.pop("best.pt")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            materializer.materialize_embedded_head(payload, self.destination)
        self.destination.mkdir()
        marker = self.destination / "user.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "fresh"):
            materializer.materialize_embedded_head(self.payload, self.destination)
        self.assertEqual(marker.read_text(), "preserve")

    def test_notebook_embedded_bytes_and_inference_call_match_original(self):
        if not builder.BASE_NOTEBOOK.exists():
            self.skipTest("Immutable local serving package absent")
        notebook, original_bytes, _ = builder.make_notebook()
        saved_path = ROOT / "notebooks/public" / builder.NOTEBOOK_NAME
        if saved_path.exists():
            self.assertEqual(json.loads(saved_path.read_text(encoding="utf-8")), notebook)
        base = json.loads(builder.BASE_NOTEBOOK.read_text(encoding="utf-8"))
        # Only head delivery and the diagnostic receipt change. Setup/runtime/decoder cells remain exact.
        self.assertEqual(notebook["cells"][1], base["cells"][1])
        self.assertEqual(notebook["cells"][3:5], base["cells"][2:4])
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertNotIn("HEAD_INPUT", code)
        self.assertNotIn("rsna-f002-head-fold0-v1", code)
        self.assertEqual(notebook["metadata"]["rsna_manual_inputs"], builder.REQUIRED_INPUTS)
        for item in notebook["cells"]:
            if item["cell_type"] == "code":
                compile("".join(item["source"]), "embedded-head-notebook", "exec")
                self.assertIsNone(item["execution_count"])
                self.assertEqual(item["outputs"], [])
        embedded_source = "".join(notebook["cells"][2]["source"]).split("\ntry:\n    startup['phase']")[0]
        namespace = {}
        exec(embedded_source, namespace)
        receipt = namespace["materialize_embedded_head"](namespace["EMBEDDED_HEAD_PAYLOAD"], self.destination)
        self.assertEqual({path.name: path.read_bytes() for path in self.destination.iterdir()}, original_bytes)
        self.assertEqual(receipt["files"]["best.pt"]["sha256"], builder.HEAD_SHA256)
        before = "".join(base["cells"][-1]["source"])
        after = "".join(notebook["cells"][-1]["source"])
        self.assertEqual(
            before[before.index("    result = run_streaming("):before.index("    result['decoder_install']")]
            .replace("HEAD_INPUT / 'best.pt'", "EMBEDDED_CHECKPOINT"),
            after[after.index("    result = run_streaming("):after.index("    result['decoder_install']")],
        )

    def test_changed_core_sources_are_rejected(self):
        if not builder.BASE_NOTEBOOK.exists():
            self.skipTest("Immutable local serving package absent")
        with patch.object(builder, "source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "core implementation"):
                builder.make_notebook()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch absent")
    def test_actual_restored_head_keeps_synthetic_forward_and_target_contract(self):
        import torch

        from rsna_knee.contracts import TARGETS
        from rsna_knee.feature_model import FrozenFeatureHead
        from rsna_knee.feature_runtime import load_head_checkpoint

        if not builder.BASE_NOTEBOOK.exists():
            self.skipTest("Immutable local serving package absent")
        notebook, _, _ = builder.make_notebook()
        source = "".join(notebook["cells"][2]["source"]).split("\ntry:\n    startup['phase']")[0]
        namespace = {}
        exec(source, namespace)
        receipt = namespace["materialize_embedded_head"](namespace["EMBEDDED_HEAD_PAYLOAD"], self.destination)
        original = load_head_checkpoint(ROOT / "artifacts/runs/f002-dinov2-attention-fold0/best.pt")
        restored = load_head_checkpoint(receipt["checkpoint_path"])
        self.assertEqual(restored["targets"], list(TARGETS))
        self.assertEqual((restored["fold"], restored["epoch"]), (0, 11))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        before = FrozenFeatureHead(original["config"]).to(device).eval()
        after = FrozenFeatureHead(restored["config"]).to(device).eval()
        before.load_state_dict(original["model"])
        after.load_state_dict(restored["model"])
        torch.manual_seed(20261007)
        arrays = {
            "features": torch.randn(17, 3, 48, 384, device=device, dtype=torch.float16),
            "mask": torch.ones(17, 3, 48, device=device, dtype=torch.bool),
            "positions": torch.linspace(0, 1, 48, device=device).view(1, 1, 48).expand(17, 3, 48),
            "planes": torch.tensor([0, 1, 2], device=device).view(1, 3).expand(17, 3),
            "flags": torch.ones(17, 3, 2, device=device),
        }
        arrays["mask"][:, 2, :] = False
        with torch.inference_mode():
            expected = before(**arrays).sigmoid()
            actual = after(**arrays).sigmoid()
        self.assertEqual(actual.shape, (17, 12))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
