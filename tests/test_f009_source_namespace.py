"""Source-only fixtures and actual CPU imports; no model/weights/MRI/fit/CUDA."""
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_source_namespace as S

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/train_native_dinov2_fast16.py"
CONFIG = ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json"


def fixture(directory, trainer=None):
    directory.mkdir()
    bodies = {name: 'MARKER = "new-package-only"\n' for name in {**S.CORE_SHA, **S.SUPPORT_SHA}}
    bodies["feature_model.py"] = 'class FrozenFeatureHead:\n    def __init__(self):\n        raise AssertionError("No construction")\n'
    bodies["finetune_model.py"] = ('from .feature_model import FrozenFeatureHead\n'
                                   'class NativeTripletMIL:\n    def __init__(self):\n        raise AssertionError("No construction")\n')
    bodies["native_training_fast16.py"] = trainer or 'from .finetune_model import NativeTripletMIL\n'
    for name, body in bodies.items():
        (directory / name).write_text(body, encoding="utf-8")
    return {name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()}


class F009SourceNamespace(unittest.TestCase):
    def test_artificial_identity_with_old_canonical_poison_and_two_namespaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = fixture(root / "src")
            S._stage(root / "src", CLI, CONFIG, root / "pack", expected)
            old = types.ModuleType("rsna_knee.finetune_model")
            old.NativeTripletMIL = type("OldWrongModel", (), {})
            with patch.dict(sys.modules, {"rsna_knee.finetune_model": old}):
                before = {n: m for n, m in sys.modules.items() if n == "rsna_knee" or n.startswith("rsna_knee.")}
                a = S._load(root / "pack", "_rsna_f009_fake_a", expected)
                b = S._load(root / "pack", "_rsna_f009_fake_b", expected)
                try:
                    self.assertIs(a.trainer.NativeTripletMIL, a.model.NativeTripletMIL)
                    self.assertIsNot(a.model.NativeTripletMIL, b.model.NativeTripletMIL)
                    self.assertIsNot(a.model.NativeTripletMIL, old.NativeTripletMIL)
                    self.assertFalse(a.receipt["production_F009_pins"])
                    self.assertFalse(a.receipt["model_constructed_or_weights_loaded"])
                    self.assertEqual(before, {n: m for n, m in sys.modules.items() if n == "rsna_knee" or n.startswith("rsna_knee.")})
                    with self.assertRaises(ValueError):
                        S._load(root / "pack", a.namespace, expected)
                finally:
                    a.close()
                    b.close()
                self.assertFalse(any(n.startswith("_rsna_f009_fake_") for n in sys.modules))

    def test_tamper_unknown_module_and_old_absolute_import_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = fixture(root / "src")
            S._stage(root / "src", CLI, CONFIG, root / "pack", expected)
            path = root / "pack/package/finetune_model.py"
            path.write_text('raise AssertionError("Must not execute modified source")\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                S._load(root / "pack", "_rsna_f009_tamper", expected)
            self.assertFalse(any(n.startswith("_rsna_f009_tamper") for n in sys.modules))
            (root / "src/native_training_fast16.py").write_text('from rsna_knee.finetune_model import NativeTripletMIL\n', encoding="utf-8")
            body = (root / "src/native_training_fast16.py").read_text(encoding="utf-8").encode()
            expected["native_training_fast16.py"] = hashlib.sha256(body).hexdigest()
            with self.assertRaisesRegex(ValueError, "Canonical old package"):
                S._stage(root / "src", CLI, CONFIG, root / "no-stage", expected)
            self.assertFalse((root / "no-stage").exists())
            (root / "src/native_training_fast16.py").write_text('from .missing import Something\n', encoding="utf-8")
            expected["native_training_fast16.py"] = hashlib.sha256((root / "src/native_training_fast16.py").read_text(encoding="utf-8").encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "escapes the pinned closure"):
                S._stage(root / "src", CLI, CONFIG, root / "no-stage", expected)

    def test_cache_descendant_wrong_class_and_import_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = fixture(root / "src", trainer='class NativeTripletMIL: pass\n')
            S._stage(root / "src", CLI, CONFIG, root / "pack", expected)
            with patch.dict(sys.modules, {"_rsna_f009_busy.finetune_model": types.ModuleType("wrong")}):
                with self.assertRaisesRegex(ValueError, "already imported"):
                    S._load(root / "pack", "_rsna_f009_busy", expected)
            for namespace in ("rsna_knee", "arbitrary", "_rsna_f009_x.model"):
                with self.assertRaisesRegex(ValueError, "Dedicated"):
                    S._load(root / "pack", namespace, expected)
            before = list(sys.meta_path)
            with self.assertRaisesRegex(ValueError, "class resolved outside"):
                S._load(root / "pack", "_rsna_f009_wrongclass", expected)
            self.assertEqual(before, sys.meta_path)
            self.assertFalse(any(n.startswith("_rsna_f009_wrongclass") for n in sys.modules))
            expected = fixture(root / "badsrc", trainer='raise RuntimeError("planned-import-failure")\n')
            S._stage(root / "badsrc", CLI, CONFIG, root / "badpack", expected)
            with self.assertRaisesRegex(RuntimeError, "planned-import-failure"):
                S._load(root / "badpack", "_rsna_f009_failed", expected)
            self.assertEqual(before, sys.meta_path)
            self.assertFalse(any(n.startswith("_rsna_f009_failed") for n in sys.modules))

    def test_actual_fixed14_CPU_import_source8_class_shared_identity_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staged = S.stage_f009_sources(ROOT / "src/rsna_knee", CLI, CONFIG, root / "pack")
            self.assertTrue(staged["production_F009_pins"])
            loaded = S.load_f009_sources(root / "pack", "_rsna_f009_actual_cpu")
            try:
                self.assertEqual(loaded.trainer.native_source_hashes(), S.CORE_SHA)
                self.assertEqual(loaded.receipt["module_count"], 14)
                self.assertIs(loaded.trainer.NativeTripletMIL, loaded.model.NativeTripletMIL)
                self.assertIs(loaded.trainer.load_frozen_encoder, loaded.modules["feature_imaging"].load_frozen_encoder)
                self.assertIs(loaded.trainer.extract_pixel_study, loaded.modules["triplet_cache"].extract_pixel_study)
                self.assertIs(loaded.trainer.TARGETS, loaded.modules["contracts"].TARGETS)
                self.assertIs(loaded.modules["feature_contracts"].sha256, loaded.modules["contracts"].sha256)
                self.assertIs(loaded.modules["feature_imaging"].geometric_order, loaded.modules["imaging"].geometric_order)
                self.assertEqual(loaded.model.STUDY_GAMMA_CONFIG, {"schema_version": "native392_train_study_gamma_v1", "minimum": .85, "maximum": 1.15})
            finally:
                loaded.close()
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                S.stage_f009_sources(ROOT / "src/rsna_knee", CONFIG, CLI, root / "bad-input")
            self.assertFalse((root / "bad-input").exists())
            # A fixture policy cannot be published through the actual F009 load entry.
            expected = fixture(root / "src")
            S._stage(root / "src", CLI, CONFIG, root / "fixture", expected)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                S.load_f009_sources(root / "fixture", "_rsna_f009_no_fake")
            manifest = json.loads((root / "pack/source-manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["runtime_or_adoption_qualified"])


if __name__ == "__main__":
    unittest.main()
