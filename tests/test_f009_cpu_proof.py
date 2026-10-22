"""Artificial CPU suffix/head tensors only; no real run, weights, MRI or forward."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsna_knee import f009_cpu_proof as P

ROOT = Path(__file__).resolve().parents[1]
fixture_spec = importlib.util.spec_from_file_location(
    "f009_artificial_stage_fixture", ROOT / "tests/test_f009_pair_stage.py"
)
fixture_module = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)
S = fixture_module.S


def metadata_fixture(root):
    spec = fixture_module.fixture(root)
    spec["schema_version"] = "f009_CPU_proof_root_inputs_v1"
    spec["folds"] = {key: {name: value[name] for name in ("run", "index")} for key, value in spec["folds"].items()}
    spec["provenance"] = fixture_module.save(root, "toy-provenance.json", {"weights_sha256": P.GENERIC_SHA})
    snapshot = root / P.SNAPSHOT
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / P.SNAPSHOT, snapshot)
    return spec


class ArtificialCPUProof(unittest.TestCase):
    def test_metadata_pair_guards_before_checkpoint_access_and_default_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = metadata_fixture(root)
            c = P._contracts()
            original = c.pin

            def no_checkpoint(path):
                self.assertNotEqual(Path(path).suffix, ".pt", "Checkpoint body touched during metadata gates")
                return original(path)

            with patch.object(P, "_contracts", return_value=c), patch.object(c, "pin", side_effect=no_checkpoint):
                context = P.prepare_inputs(root, spec, 0)
                self.assertEqual(context["fold"], 0)
                bad = copy.deepcopy(spec)
                value = S.read(root / bad["result0"]["path"])
                value["fixed10_delta"] = -0.001
                bad["result0"] = fixture_module.save(root, "proof/negative.json", value)
                with self.assertRaisesRegex(ValueError, "positive fold0"):
                    P.prepare_inputs(root, bad, 0)
        run = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/check_f009_E12_cpu.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertFalse(json.loads(run.stdout)["actual_CPU_strict_reconstruction"])
        self.assertFalse(json.loads(run.stdout)["run_or_checkpoint_or_encoder_binary_opened"])

    def test_checkpoint_bytes_rejected_before_source_import_or_factory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = metadata_fixture(root)
            spec_path = root / "inputs.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            context = P.prepare_inputs(root, spec, 0)
            context["checkpoint_path"].write_bytes(b"ARTIFICIAL_TAMPER_NOT_A_MODEL")
            with (
                patch.object(P, "ROOT", root),
                patch.object(P.sources, "load_f009_sources", side_effect=AssertionError("Model namespace opened")),
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
                    P.produce(
                        spec_path,
                        P.sha(spec_path),
                        0,
                        root / "artifacts/qa/proof",
                        root / "absent-repo",
                        root / "absent-encoder.pt",
                        P.sha(P.__file__),
                    )
            self.assertFalse((root / "artifacts/qa/proof").exists())

            # Artificial metadata-only lifecycle fault: never publish CPU.json if close fails.
            context["checkpoint_path"].write_bytes(b"ARTIFICIAL_NON_MODEL_FOLD0")
            context["checkpoint_pin"] = S.pin(context["checkpoint_path"])
            fake_loaded = SimpleNamespace(
                trainer=SimpleNamespace(native_source_hashes=lambda: P.sources.CORE_SHA),
                receipt={"artificial_metadata_lifecycle_probe": True},
                close=lambda: (_ for _ in ()).throw(RuntimeError("Artificial close failure")),
            )
            with (
                patch.object(P, "ROOT", root),
                patch.object(P, "prepare_inputs", return_value=context),
                patch.object(P.sources, "load_f009_sources", return_value=fake_loaded),
                patch.object(P, "_reconstruct", return_value={"artificial_metadata_lifecycle_probe": True}),
            ):
                with self.assertRaisesRegex(RuntimeError, "Artificial close failure"):
                    P.produce(
                        spec_path,
                        P.sha(spec_path),
                        0,
                        root / "artifacts/qa/close-failure",
                        root / "absent-repo",
                        root / "absent-encoder.pt",
                        P.sha(P.__file__),
                    )
            self.assertFalse((root / "artifacts/qa/close-failure/CPU.json").exists())
            self.assertFalse(S.read(root / "artifacts/qa/close-failure/FAILED.json")["complete"])

    def test_real_pinned_classes_strict_apply_exact_toy_tensors_no_forward(self):
        import torch
        from torch import nn

        class Parts(nn.Module):
            def __init__(self, shapes):
                super().__init__()
                for name, shape in shapes.items():
                    self.register_parameter(name, nn.Parameter(torch.ones(shape, dtype=torch.float32)))

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1, self.norm2 = nn.LayerNorm(384), nn.LayerNorm(384)
                self.attn = nn.Module()
                self.attn.qkv, self.attn.proj = nn.Linear(384, 1152), nn.Linear(384, 384)
                self.mlp = nn.Module()
                self.mlp.fc1, self.mlp.fc2 = nn.Linear(384, 1536), nn.Linear(1536, 384)
                self.ls1, self.ls2 = Parts({"gamma": [384]}), Parts({"gamma": [384]})

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [Block(), Block()])
                self.norm = nn.LayerNorm(384)
                self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384

            def forward(self, *args, **kwargs):
                raise AssertionError("Artificial CPU proof must never forward")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            P.sources.stage_f009_sources(
                ROOT / "src/rsna_knee",
                ROOT / "scripts/train_native_dinov2_fast16.py",
                ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json",
                root / "source14",
            )
            loaded = P.sources.load_f009_sources(root / "source14", "_rsna_f009_cpu_proof_fixture")
            try:
                trainer = loaded.trainer
                config = S.read(ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")
                config_original = copy.deepcopy(config)
                provenance = {"artificial_test_encoder_only": True}
                model = trainer.NativeTripletMIL(
                    Encoder(), config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True
                )
                prefix = trainer.prefix_digest(model)
                checkpoint = trainer.make_native_checkpoint(
                    model,
                    provenance,
                    fold=0,
                    epoch=12,
                    weak_bce=0.3,
                    prefix_sha256=prefix,
                    audit={"artificial_fixture_only": True},
                    artificial=False,
                )
                checkpoint_path = root / "artificial-epoch_012.pt"
                torch.save(checkpoint, checkpoint_path)
                context = {
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_pin": S.pin(checkpoint_path),
                    "config": config,
                    "fold": 0,
                    "row": {"weak_bce": 0.3},
                    "prefix": prefix,
                    "producer_sha256": P.sha(P.__file__),
                    "provenance": provenance,
                    "run_pin": {"bytes": 3, "sha256": "1" * 64},
                    "retained_index_pin": {"bytes": 4, "sha256": "2" * 64},
                }
                calls = []

                def factory(repo, weights, supplied_provenance, supplied_config, device):
                    self.assertEqual(device, "cpu")
                    self.assertEqual(supplied_provenance, provenance)
                    self.assertEqual(supplied_config, config_original)
                    calls.append(1)
                    value = trainer.NativeTripletMIL(
                        Encoder(), config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True
                    )
                    value.forward = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Forward forbidden"))
                    return value

                with (
                    patch.object(trainer, "new_native_model", side_effect=factory),
                    patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")),
                ):
                    result = P._reconstruct(
                        context, loaded, root / "absent-repo", root / "absent-weights.pt", fixture_only=True
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(result["actual_CPU_strict_reconstruction"])
                    self.assertEqual(result["evidence_scope"], "artificial_fixture_only")
                    self.assertEqual(
                        result["encoder_suffix_tensor_sha256"], P.tensor_digest(checkpoint["encoder_suffix"])
                    )
                    self.assertEqual(result["head_tensor_sha256"], P.tensor_digest(checkpoint["head"]))
                    self.assertEqual(
                        [len(result["tensor_metadata"][key]) for key in ("encoder_suffix", "head")], [28, 12]
                    )
                    self.assertEqual((result["gamma_draws"], result["model_forward_calls"]), (0, 0))
                    self.assertTrue(result["eval_gamma_RNG_unchanged"])
                    self.assertEqual(result["model_class"], loaded.namespace + ".finetune_model.NativeTripletMIL")
                    c = P._contracts()
                    c.validate_cpu(
                        result,
                        fold=0,
                        run_pin=context["run_pin"],
                        index_pin=context["retained_index_pin"],
                        checkpoint=context["checkpoint_pin"],
                        prefix=prefix,
                        fixture_only=True,
                    )
                    with self.assertRaisesRegex(ValueError, "not implemented/reviewed/pinned"):
                        c.validate_cpu(
                            result,
                            fold=0,
                            run_pin=context["run_pin"],
                            index_pin=context["retained_index_pin"],
                            checkpoint=context["checkpoint_pin"],
                            prefix=prefix,
                            fixture_only=False,
                        )

                    # Corrupt the persisted artificial checkpoint; every case is rejected before a generic factory.
                    for kind in ("key", "shape", "NaN", "prefix", "source"):
                        bad = copy.deepcopy(checkpoint)
                        name = sorted(bad["encoder_suffix"])[0]
                        if kind == "key":
                            del bad["encoder_suffix"][name]
                        elif kind == "shape":
                            bad["encoder_suffix"][name] = torch.zeros(1)
                        elif kind == "NaN":
                            bad["encoder_suffix"][name].flatten()[0] = float("nan")
                        elif kind == "prefix":
                            bad["frozen_prefix_sha256"] = "f" * 64
                        else:
                            bad["source_sha256"] = {**bad["source_sha256"], "finetune_model.py": "0" * 64}
                        torch.save(bad, checkpoint_path)
                        before = len(calls)
                        with self.subTest(kind=kind), self.assertRaises(ValueError):
                            P._reconstruct(
                                context, loaded, root / "absent-repo", root / "absent-weights.pt", fixture_only=True
                            )
                        self.assertEqual(len(calls), before)
                    torch.save(checkpoint, checkpoint_path)
                    # A class with the same printed module/name still fails exact identity.
                    Spoof = type("NativeTripletMIL", (nn.Module,), {"__module__": loaded.namespace + ".finetune_model"})
                    with patch.object(trainer, "new_native_model", return_value=Spoof()):
                        with self.assertRaisesRegex(ValueError, "class identity"):
                            P._reconstruct(
                                context, loaded, root / "absent-repo", root / "absent-weights.pt", fixture_only=True
                            )
            finally:
                loaded.close()
            self.assertNotIn("_rsna_f009_cpu_proof_fixture", sys.modules)


if __name__ == "__main__":
    unittest.main()
