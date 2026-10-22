"""Artificial generic encoder and four real pinned CPU model classes only."""
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rsna_knee import f009_mixed_runtime as M
from rsna_knee import f009_source_namespace as S

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "artifacts/research/20261016-f006-execution-v1/static-native-source"
REG_SRC = (ROOT / "artifacts/research/20261010-f005-regional-sealer-source-preparation-v1/"
           "source-only-packet-v3/bundle/code/src/rsna_knee")
CONFIG = ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json"
torch.set_num_threads(1)


class Parts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(384), nn.LayerNorm(384)
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(384, 1152), nn.Linear(384, 384)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(384, 1536), nn.Linear(1536, 384)
        self.ls1, self.ls2 = Parts(), Parts()

    def forward(self, values):
        return values  # Artificial encoder only, never official encoder numerics.


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [Block(), Block()])
        self.norm = nn.LayerNorm(384)
        self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384

    def prepare_tokens_with_masks(self, images):
        values = torch.arange(785*384, device=images.device, dtype=torch.float32).reshape(1, 785, 384)
        return values.expand(len(images), -1, -1).div(785*384)


def extracted(path, name, globals_):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@contextmanager
def fixture():
    """Every checkpoint below is newly synthetic, not a real trained artifact."""
    old_package = "_f009_mixed_old_reg_fixture"
    package = types.ModuleType(old_package)
    package.__path__ = [str(REG_SRC)]
    sys.modules[old_package] = package
    before_canonical = {key: value for key, value in sys.modules.items() if key == "rsna_knee" or key.startswith("rsna_knee.")}
    handles = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            reg = importlib.import_module(old_package + ".regional_training_fast16")
            assert M.sha(reg.__file__) == M.STATIC_DEPENDENCIES["regional_original"]
            cls_config = json.loads(CONFIG.read_bytes())
            reg_config = json.loads((STATIC / "regional/driver-pins.json").read_bytes())["config"]
            M._config_compatibility(cls_config, reg_config)
            provenance = {name: "artificial_only" for name in
                          ("source_url", "source_revision", "weights_url", "license", "reviewed_at", "exposure_audit")}
            provenance.update(weights_sha256="1"*64, source_tree_sha256="2"*64,
                              pretraining="generic_dinov2_lvd142m", competition_finetuned=False, gold_used_for_tuning=False)
            roots = {family: {fold: root / (family+str(fold)) for fold in M.FOLDS} for family in M.FAMILIES}
            docs, views, bindings = {"CLS": {}, "REG": {}}, {"CLS": {}, "REG": {}}, {"CLS": {}, "REG": {}}
            for fold in M.FOLDS:
                source14 = roots["CLS"][fold] / "source14"
                S.stage_f009_sources(ROOT / "src/rsna_knee", ROOT / "scripts/train_native_dinov2_fast16.py", CONFIG, source14)
                handles.append(S.load_f009_sources(source14, "_rsna_f009_mixed_fixture_f"+str(fold)))
            for family in M.FAMILIES:
                for fold in M.FOLDS:
                    trainer = handles[fold].trainer if family == "CLS" else reg
                    config = cls_config if family == "CLS" else reg_config
                    constructor = trainer.NativeTripletMIL if family == "CLS" else trainer.RegionalTripletMIL
                    torch.manual_seed(123+fold+(2 if family == "REG" else 0))
                    model = constructor(Encoder(), config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True)
                    prefix = trainer.prefix_digest(model)
                    epoch = 12 if family == "CLS" else (8 if fold == 0 else 10)
                    make = trainer.make_native_checkpoint if family == "CLS" else trainer.make_regional_checkpoint
                    checkpoint = make(model, provenance, fold=fold, epoch=epoch, weak_bce=.3+fold*.01,
                                      prefix_sha256=prefix, audit={"artificial_fixture_only": True}, artificial=False)
                    path = roots[family][fold] / ("trained/epoch_012.pt" if family == "CLS" else "trained/best.pt")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, path)
                    b = {"config": config, "encoder_provenance": provenance, "checkpoint_sha256": M.sha(path),
                         "frozen_prefix_sha256": prefix, "encoder_suffix_tensor_sha256": M._tensor_digest(checkpoint["encoder_suffix"]),
                         "head_tensor_sha256": M._tensor_digest(checkpoint["head"]), "selected_epoch": epoch,
                         "weak_bce": checkpoint["weak_bce"], "CPU_receipt_sha256": "3"*64, "run_json_sha256": "4"*64,
                         "producer_model_class": "_artificial_cpu_nonce.finetune_model.NativeTripletMIL",
                         "producer_head_class": "_artificial_cpu_nonce.feature_model.FrozenFeatureHead",
                         "source14_manifest_sha256": M.sha(roots["CLS"][fold]/"source14/source-manifest.json"),
                         "original_audit_sha256": M.json_sha(checkpoint["audit"])}
                    bindings[family][fold] = b
                    views[family][fold] = {key: checkpoint[key] for key in ("config", "source_sha256", "targets", "encoder_provenance", "fold")}
                    if family == "REG":
                        views[family][fold].update({key: checkpoint[key] for key in ("representation", "representation_sha256", "execution", "CLI_sha256")})
                    else:
                        docs[family][fold] = {"checkpoint_path": path, "source14": roots["CLS"][fold]/"source14"}
                    del model
            p = SimpleNamespace(require=M.require, json_sha=M.json_sha,
                                pins=lambda: {"producer": {"schemas": {"CHECKPOINT_SCHEMA": reg.CHECKPOINT_SCHEMA}}})
            boundary = SimpleNamespace(guard_checkpoint=extracted(STATIC / "regional/regional_runtime.py", "guard_checkpoint", {"p": p}))
            calls = []

            def factory(repo, weights, provenance_, config_, device):
                assert device == "cpu" and provenance_ == provenance and config_ == reg_config
                calls.append((repo, weights))
                return reg.RegionalTripletMIL(Encoder(), config_, trainable_blocks=2, window_chunk=16, checkpoint_last2=True)

            pair = {"family_pairs": {f: {"bindings": bindings[f], "pair_fingerprint_sha256": f*64} for f in M.FAMILIES},
                    "pair_fingerprint_sha256": "5"*64, "family_fold_manifest_sha256": {f: {} for f in M.FAMILIES},
                    "FP32_aggregation": "unchanged twofold then family FP32"}
            yield SimpleNamespace(roots=roots, pair=pair, views=views, docs=docs, reg=reg,
                                  boundary=boundary, handles=dict(enumerate(handles)), factory=factory, calls=calls)
        finally:
            for handle in reversed(handles):
                if handle.namespace in sys.modules:
                    handle.close()
            for name in list(sys.modules):
                if name == old_package or name.startswith(old_package+"."):
                    del sys.modules[name]
            assert all(sys.modules.get(key) is value for key, value in before_canonical.items())


def construct(f):
    return M._construct_cpu_models(f.roots, f.pair, f.views, f.docs, core=None, reg=f.reg,
                                  boundary=f.boundary, loaded_by_fold=f.handles,
                                  fixture_only=True, artificial_factory=f.factory)


class MixedRuntime(unittest.TestCase):
    def test_default_disabled_and_module_source_guard(self):
        with patch.object(M, "sha", side_effect=AssertionError("Asset body must stay closed")):
            with self.assertRaisesRegex(ValueError, "Explicit"):
                M.reconstruct_cpu_models(None, None, None, None, None, core=None, regional=None, boundary=None,
                    regional_original=None, cls_pair=None, conditional_plan=None, cls_transport=None,
                    namespace_helper=None, dependency_pins={})
            with self.assertRaisesRegex(ValueError, "Explicit"):
                M.bind_pairs(None, None, None, None, cls_pair=None, conditional_plan=None,
                    cls_transport=None, namespace_helper=None, dependency_pins={})
        # The fourth router argument is preflight documents, not input manifest pins.
        pair = {"family_pairs": {"CLS": {"bundle_pins": {0: {"bytes": 1, "sha256": "a"*64},
                                                           1: {"bytes": 2, "sha256": "b"*64}}}},
                "family_fold_manifest_sha256": {"REG": {0: "c"*64, 1: "d"*64}}}
        views, documents = {"synthetic_views": True}, {"REG": {"synthetic_documents": True}}
        regional = SimpleNamespace(verify_assets=lambda *args: (_ for _ in ()).throw(RuntimeError("stop before assets")))
        with patch.object(M, "_permission"), patch.object(M, "_module"), patch.object(M, "_reg_identity"), \
                patch.object(M, "bind_pairs", return_value=(pair, views, documents)) as binder:
            with self.assertRaisesRegex(RuntimeError, "stop before assets"):
                M.reconstruct_cpu_models({"REG": {}}, pair, views, documents, {}, core=None, regional=regional,
                    boundary=None, regional_original=None, cls_pair=None, conditional_plan=None, cls_transport=None,
                    namespace_helper=None, dependency_pins={}, execute=True)
            self.assertEqual(binder.call_args.args[3], {"CLS": pair["family_pairs"]["CLS"]["bundle_pins"],
                                                       "REG": pair["family_fold_manifest_sha256"]["REG"]})
            binder.return_value = (pair, views, {"wrong_documents": True})
            with self.assertRaisesRegex(ValueError, "views/documents"):
                M.reconstruct_cpu_models({"REG": {}}, pair, views, documents, {}, core=None, regional=regional,
                    boundary=None, regional_original=None, cls_pair=None, conditional_plan=None, cls_transport=None,
                    namespace_helper=None, dependency_pins={}, execute=True)
        module = S
        pins = {"namespace_helper": {"path": S.__file__, "sha256": M.STATIC_DEPENDENCIES["namespace_helper"]}}
        self.assertIs(M._module(module, "namespace_helper", pins), module)
        with self.assertRaisesRegex(ValueError, "imported module"):
            M._module(SimpleNamespace(__file__=S.__file__), "namespace_helper", pins)
        pins["namespace_helper"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "dependency"):
            M._module(module, "namespace_helper", pins)
        runs = {fold: {"audit": {"manifest_sha256": {"weak.csv": "a"*64}, "pixel_index_sha256": "b"*64}}
                for fold in M.FOLDS}
        bindings = [{fold: {"checkpoint_sha256": hashlib.sha256((family+str(fold)).encode()).hexdigest()}
                     for fold in M.FOLDS} for family in M.FAMILIES]
        M._shared_inputs(runs, copy.deepcopy(runs), *bindings)
        bad_runs = copy.deepcopy(runs)
        bad_runs[1]["audit"]["pixel_index_sha256"] = "c"*64
        with self.assertRaisesRegex(ValueError, "Shared weak/group/pixel"):
            M._shared_inputs(bad_runs, runs, *bindings)
        bindings[0][1]["checkpoint_sha256"] = bindings[1][0]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "Four distinct"):
            M._shared_inputs(runs, runs, *bindings)
        with self.assertRaisesRegex(ValueError, "initialized two-CUDA"):
            M.models_on_two_devices(M.MixedModels({}, []), None)

    def test_four_pinned_models_strict_cpu_identity_and_namespace_lifetime(self):
        with fixture() as f, patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            models, receipt = construct(f)
            try:
                self.assertEqual(len(f.calls), 1)
                self.assertEqual(receipt["CPU_storage_alias_pairs_checked"], 6)
                self.assertEqual(receipt["logical_CPU_fold_models"], 4)
                self.assertFalse(receipt["actual_CPU_four_model_reconstruction"])
                self.assertFalse(receipt["serving_candidate_gate_passed"])
                self.assertIsNot(type(models["CLS"][0]), type(models["CLS"][1]))
                for fold in M.FOLDS:
                    self.assertIs(type(models["CLS"][fold]), f.handles[fold].model.NativeTripletMIL)
                    self.assertIn(f.handles[fold].namespace, sys.modules)
                    self.assertEqual(receipt["model_fingerprints"]["CLS:"+str(fold)]["producer_model_class"],
                                     "_artificial_cpu_nonce.finetune_model.NativeTripletMIL")
                rng = torch.get_rng_state().clone()
                replicas = M.MixedModels(copy.deepcopy(dict(models)), owner=models.owner, classes=models.classes)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for fold in M.FOLDS:
                    self.assertIs(type(replicas["CLS"][fold]), type(models["CLS"][fold]))
                    self.assertIsNot(replicas["CLS"][fold]._gamma_generator, models["CLS"][fold]._gamma_generator)
                    self.assertTrue(torch.equal(replicas["CLS"][fold]._gamma_generator.get_state(),
                                                models["CLS"][fold]._gamma_generator.get_state()))
                f.pair["schema_version"] = M.PAIR_SCHEMA
                f.pair["model_classes"] = receipt["model_classes"]
                for fold in M.FOLDS:
                    f.pair["family_pairs"]["CLS"]["bindings"][fold]["runtime_head_class"] = (
                        receipt["model_fingerprints"]["CLS:"+str(fold)]["runtime_head_class"])
                self.assertEqual(M.model_fingerprints_from_pair(f.pair), receipt["model_fingerprints"])
                models.close()
                self.assertTrue(all(handle.namespace in sys.modules for handle in f.handles.values()))
                replicas.close()
                self.assertTrue(all(handle.namespace not in sys.modules for handle in f.handles.values()))
            finally:
                if not models.closed:
                    models.close()

    def test_invalid_checkpoint_states_source_and_cross_namespace_reject(self):
        with fixture() as f, patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            path = f.docs["CLS"][0]["checkpoint_path"]
            original = torch.load(path, map_location="cpu", weights_only=True)
            binding = f.pair["family_pairs"]["CLS"]["bindings"][0]
            for kind in ("key", "shape", "NaN", "prefix", "source"):
                bad = copy.deepcopy(original)
                key = sorted(bad["encoder_suffix"])[0]
                if kind == "key":
                    del bad["encoder_suffix"][key]
                elif kind == "shape":
                    bad["encoder_suffix"][key] = torch.zeros(1)
                elif kind == "NaN":
                    bad["encoder_suffix"][key].flatten()[0] = float("nan")
                elif kind == "prefix":
                    bad["frozen_prefix_sha256"] = "f"*64
                else:
                    bad["source_sha256"]["finetune_model.py"] = "0"*64
                torch.save(bad, path)
                binding["checkpoint_sha256"] = M.sha(path)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    construct(f)
                self.assertEqual(f.calls, [])
            torch.save(original, path)
            binding["checkpoint_sha256"] = M.sha(path)
            with patch.dict(f.handles, {1: f.handles[0]}), self.assertRaisesRegex(ValueError, "Distinct per-fold"):
                construct(f)
            wrong = f.docs["CLS"][0]["source14"]
            f.docs["CLS"][0]["source14"] = f.docs["CLS"][1]["source14"]
            with self.assertRaisesRegex(ValueError, "identity"):
                construct(f)
            f.docs["CLS"][0]["source14"] = wrong
            self.assertEqual(f.calls, [])
            Spoof = type("RegionalTripletMIL", (nn.Module,), {"__module__": f.reg.RegionalTripletMIL.__module__})
            f.factory = lambda *args, **kwargs: Spoof()
            with self.assertRaisesRegex(ValueError, "factory class"):
                construct(f)

    def test_original_numeric_kernel_ast_and_four_real_cpu_forwards(self):
        old = ast.parse((STATIC / "serving/isolated_mixed_runtime.py").read_text(encoding="utf-8"))
        new = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))
        a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "consume_four")
        b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "_consume_four_numeric")
        b.name = a.name
        self.assertEqual(ast.dump(a), ast.dump(b))
        with fixture() as f, patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            cpu, _ = construct(f)
            # CPU device adapter is artificial; the four full model forwards are real.
            models = M.MixedModels({fam: {"cpu": cpu[fam]} for fam in M.FAMILIES}, owner=cpu.owner, classes=cpu.classes)
            try:
                arrays = (np.zeros((1, 3, 48, 3, 392, 392), dtype=np.uint8), np.zeros((1, 3, 48), dtype=np.bool_),
                          np.zeros((1, 3, 48), dtype=np.float32), np.zeros((1, 3), dtype=np.int64),
                          np.zeros((1, 3, 2), dtype=np.float32))
                arrays[1][0, 0, 0] = True
                def hashes(values):
                    return [hashlib.sha256(v.tobytes()).hexdigest() for v in values]
                payload = ("ARTIFICIAL_UID", arrays, {"array_sha256": hashes(arrays)})
                forward_calls = []
                mean = extracted(STATIC / "common/reused/twofold_runtime.py", "probability_mean_reference", {
                    "struct": struct, "math": math, "p": SimpleNamespace(c=SimpleNamespace(require=M.require), TARGETS=list(range(12)))})

                def consume(device, payload_, *, models, predictions, receipts, lock, pair_sha):
                    key, values, _ = payload_
                    probabilities = []
                    for fold in M.FOLDS:
                        with torch.inference_mode():
                            logits = models[device][fold](*[torch.from_numpy(v) for v in values])
                            probabilities.append(torch.sigmoid(logits.float())[0].tolist())
                        forward_calls.append(type(models[device][fold]))
                    predictions[key] = mean(*probabilities)
                    receipts[key] = {"fold_order": [0, 1], "logical_full_fold_model_forward_calls": 2,
                                     "fallback": False, "device": device, "fold_probabilities": probabilities}

                core = SimpleNamespace(consume_twofold=consume, probability_mean_reference=mean)
                predictions, receipts = {}, {}
                family_predictions, family_receipts = {f: {} for f in M.FAMILIES}, {f: {} for f in M.FAMILIES}
                M.consume_four("cpu", payload, models=models, predictions=predictions, receipts=receipts,
                               lock=threading.Lock(), core=core, pair=f.pair, family_predictions=family_predictions,
                               family_receipts=family_receipts, array_hashes=hashes)
                self.assertEqual(len(forward_calls), 4)
                self.assertEqual(len(set(forward_calls)), 3)  # distinct CLS0, CLS1 and shared REG class.
                self.assertEqual(receipts["ARTIFICIAL_UID"]["logical_full_fold_model_forward_calls"], 4)
                self.assertEqual(predictions["ARTIFICIAL_UID"], mean(*[family_predictions[f]["ARTIFICIAL_UID"] for f in M.FAMILIES]))
                self.assertEqual(hashes(arrays), payload[2]["array_sha256"])
                self.assertEqual([cpu["CLS"][n]._gamma_draws for n in M.FOLDS], [0, 0])
                cpu["CLS"][0].train()
                with self.assertRaisesRegex(ValueError, "Eval model"):
                    M.consume_four("cpu", payload, models=models, predictions={}, receipts={}, lock=threading.Lock(),
                                   core=core, pair=f.pair, family_predictions={f: {} for f in M.FAMILIES},
                                   family_receipts={f: {} for f in M.FAMILIES}, array_hashes=hashes)
            finally:
                models.close()
                cpu.close()

    def test_close_failure_not_claimed_successful(self):
        calls = []
        fail = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("artificial close failure")))
        owner = M.MixedModels({}, [SimpleNamespace(close=lambda: calls.append(1)), fail])
        with self.assertRaisesRegex(RuntimeError, "close failure"):
            owner.close()
        self.assertFalse(owner.closed)
        self.assertEqual(owner.owner.references, 1)
        fail.close = lambda: calls.append(2)
        owner.close()
        self.assertEqual(calls, [2, 1])


if __name__ == "__main__":
    unittest.main()
