"""Source-bound author224 branch; no downloads, training or remote operations.

The extracted numerical functions remain literal author source.  This wrapper
qualifies their use, isolates their globals, and rejects stock FP32 fallback.
Artificial CPU/CUDA execution can never qualify for production.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUTHOR_NOTEBOOK_SHA256 = "b80d3b72a1c34f2b39b3ea0d2952f039b8d27c3feb5c4db0a7d0ab62ae75354e"
AUTHOR_NORMALIZED_SHA256 = "993c756c94a8b7d04e91cc0cc568c845f60593c84d6a23b0a5e70419100f1d06"
CHECKPOINT_SHA256 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"
CHECKPOINT_BYTES = 292842755
CHECKPOINT_FILENAME = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"
ARCH = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"
RESOLUTION = 224
LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
SYMBOL_SHA256 = {
    "_crop": "1abff48888d855b653320b23f01ed0618bad29adc565a63e2061c1d34a0615ca",
    "build_backbone": "f5a1634d2e2a54a81265cf1e5b809afaf28b145d88d2cd24bcc48f5bb08ef1a3",
    "RaptorClassifier": "da765dd0bd9b371e563a541d5ffbc445d1138d2320d99cc0d8c2b7275e588b75",
    "load_model": "17a8beda51380fe0f0983b8bc7061558187ae112a10f0af7523bd49aecc1a3d5",
    "_eval_centers": "4ddfae29b35a0580a8b1df43bb4dcce0cac15b6a4d6473313d23390780d1acb3",
    "eval_windows": "6b5b2b026d9dd0e3396fc388e77237dcab76c7bc1ca49561337b30bfb704c9c4",
    "gpu_windows": "aa59baadaf18b1b276d7b6c99cd1680291e3e8fe0d6c12cbe8e5e2efe501d0a7",
    "infer_probs": "1185858b02024a7653421dda7b560bdc67eff06bf7fbe006582c9605cc1eb470",
    "rankpct": "e4ec008a6db3f7ada25c9968d72adbd47d1902a9933585610e20fc4326b12e34",
}
_CONSTANT_NAMES = {"IMG", "CROP_MM", "SLOTS", "MAXS", "K_EVAL", "NORM", "LAB", "CROP_FRAC",
                   "_MEAN", "_STD", "ARMS", "_WLUT"}
_IMPORT_TEXT = {
    "import os, sys, glob, time, json, gc", "import numpy as np",
    "import torch, torch.nn as nn, torch.nn.functional as F",
}


class Expert224ContractError(ValueError):
    """A source, checkpoint or invocation differs from the qualified contract."""


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _text(cell: Mapping[str, Any]) -> str:
    source = cell.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(line, str) for line in source):
        return "".join(source)
    raise Expert224ContractError("Notebook source must be text or a list of text lines")


def _read_author(author_notebook: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(author_notebook, Mapping):
        book = dict(author_notebook)
    else:
        path = Path(author_notebook)
        if sha256_file(path) != AUTHOR_NOTEBOOK_SHA256:
            raise Expert224ContractError("Author savedV2 Notebook SHA differs")
        book = json.loads(path.read_text(encoding="utf-8"))
    normalized = json.dumps(book, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != AUTHOR_NORMALIZED_SHA256:
        raise Expert224ContractError("Author savedV2 Notebook content differs")
    return book


def _literal_with_decorators(source: str, node: ast.AST) -> str:
    start = node.lineno
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.decorator_list:
        start = min(start, *(decorator.lineno for decorator in node.decorator_list))
    lines = source.splitlines(keepends=True)
    return "".join(lines[start - 1:node.end_lineno]).rstrip("\r\n")


def build_stock_source(author_notebook: str | Path | Mapping[str, Any]) -> str:
    """Return literal constants/imports and nine definitions from exact savedV2.

    The stock ``import timm`` is provided by ``make_stock_namespace`` so CPU
    tests can inject a tiny encoder without altering global import state.
    """
    book = _read_author(author_notebook)
    source = _text(book["cells"][4])
    nodes = ast.parse(source).body
    pieces = ["# Literal nartaa savedV2 author224 numerical source; timm is namespace-injected."]
    found_imports: set[str] = set()
    found_constants: set[str] = set()
    found_symbols: set[str] = set()
    for node in nodes:
        segment = ast.get_source_segment(source, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and segment in _IMPORT_TEXT:
            found_imports.add(segment)
            pieces.append(_literal_with_decorators(source, node))
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & _CONSTANT_NAMES:
                if len(names) != 1 or not names <= _CONSTANT_NAMES or names & found_constants:
                    raise Expert224ContractError("Stock constant definition is ambiguous")
                found_constants.update(names)
                pieces.append(_literal_with_decorators(source, node))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in SYMBOL_SHA256:
            if node.name in found_symbols or hashlib.sha256(segment.encode()).hexdigest() != SYMBOL_SHA256[node.name]:
                raise Expert224ContractError(f"Stock literal definition differs: {node.name}")
            if node.name == "infer_probs":
                if [ast.unparse(d) for d in node.decorator_list] != ["torch.no_grad()"]:
                    raise Expert224ContractError("Stock infer_probs no_grad decorator differs")
            found_symbols.add(node.name)
            pieces.append(_literal_with_decorators(source, node))
    if found_imports != _IMPORT_TEXT or found_constants != _CONSTANT_NAMES or found_symbols != set(SYMBOL_SHA256):
        raise Expert224ContractError("Required stock numerical imports/constants/symbols are incomplete")
    return "\n\n".join(pieces) + "\n"


def _source_contract(stock_source: str) -> str:
    """Validate an extracted source even when the original Notebook is absent."""
    tree = ast.parse(stock_source)
    found: dict[str, str] = {}
    constants: dict[str, ast.AST] = {}
    imports: set[str] = set()
    for node in tree.body:
        segment = ast.get_source_segment(stock_source, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.add(segment)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if len(names) != 1 or names[0] in constants:
                raise Expert224ContractError("Unexpected stock assignments")
            constants[names[0]] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in found:
                raise Expert224ContractError("Duplicate stock numerical symbol")
            found[node.name] = hashlib.sha256(segment.encode()).hexdigest()
            expected_decorators = ["torch.no_grad()"] if node.name == "infer_probs" else []
            if [ast.unparse(d) for d in node.decorator_list] != expected_decorators:
                raise Expert224ContractError(f"Stock decorators differ: {node.name}")
        else:
            raise Expert224ContractError("Unexpected executable stock source node")
    if found != SYMBOL_SHA256 or set(constants) != _CONSTANT_NAMES or imports != _IMPORT_TEXT:
        raise Expert224ContractError("Extracted stock source does not match its bound definitions")
    # Constants are included in the full extracted-source digest, not accepted
    # merely because numerical functions happen to match.
    if hashlib.sha256(stock_source.encode()).hexdigest() != STOCK_SOURCE_SHA256:
        raise Expert224ContractError("Extracted stock constants/imports/source SHA differs")
    return STOCK_SOURCE_SHA256


# Filled from the exact savedV2 extraction; changing this requires a new audit.
STOCK_SOURCE_SHA256 = "b61046e344425c446370d3df1ca019cf278f13fba385117035f849484a5a85f0"


def make_stock_namespace(stock_source: str, *, timm_module: Any = None,
                         artificial_cpu: bool = False, artificial_device: str | None = None) -> dict[str, Any]:
    digest = _source_contract(stock_source)
    if artificial_cpu:
        if artificial_device not in (None, "cpu"):
            raise Expert224ContractError("Conflicting artificial device declarations")
        artificial_device = "cpu"
    if artificial_device not in (None, "cpu", "cuda"):
        raise Expert224ContractError("Artificial device must be explicitly cpu or cuda")
    artificial = artificial_device is not None
    if timm_module is not None and not artificial:
        raise Expert224ContractError("Injected encoder is restricted to artificial verification")
    if artificial and timm_module is None:
        raise Expert224ContractError("Artificial verification requires an injected tiny encoder")
    if timm_module is None:
        timm_module = importlib.import_module("timm")
    namespace: dict[str, Any] = {"__name__": "_p010_author224_stock", "timm": timm_module}
    exec(compile(stock_source, "<p010-author224-stock>", "exec"), namespace)
    if namespace["LAB"] != list(LABELS) or namespace["K_EVAL"] != 94 or namespace["IMG"] != 384:
        raise Expert224ContractError("Author target order or window dimensions differ")
    if namespace["NORM"] != "imagenet" or namespace["CROP_FRAC"] != 0.8 or namespace["MAXS"] != 96:
        raise Expert224ContractError("Author224 preprocessing constants differ")
    namespace["_p010_binding"] = {
        "stock_source_sha256": digest, "author_notebook_sha256": AUTHOR_NOTEBOOK_SHA256,
        "artificial_cpu": artificial_device == "cpu", "artificial_device": artificial_device,
        "artificial_mode": artificial, "production_qualified": not artificial,
        "namespace_identity": id(namespace),
    }
    namespace["_p010_bound_symbols"] = {name: namespace[name] for name in SYMBOL_SHA256}
    namespace["_p010_bound_modules"] = {name: namespace[name] for name in ("torch", "nn", "F", "np", "timm")}
    namespace["_p010_bound_lut"] = namespace["_WLUT"]
    namespace["_p010_bound_normalization"] = {name: namespace[name].clone() for name in ("_MEAN", "_STD")}
    return namespace


def _namespace_binding(namespace: dict[str, Any]) -> dict[str, Any]:
    binding = namespace.get("_p010_binding", {})
    if binding.get("stock_source_sha256") != STOCK_SOURCE_SHA256 or binding.get("namespace_identity") != id(namespace):
        raise Expert224ContractError("Stock namespace has no valid source binding")
    if any(namespace.get(name) is not value for name, value in namespace.get("_p010_bound_symbols", {}).items()):
        raise Expert224ContractError("A stock numerical callable was replaced after extraction")
    if set(namespace.get("_p010_bound_symbols", {})) != set(SYMBOL_SHA256):
        raise Expert224ContractError("Stock callable identity binding is incomplete")
    if any(namespace.get(name) is not value for name, value in namespace.get("_p010_bound_modules", {}).items()):
        raise Expert224ContractError("A stock dependency was replaced after extraction")
    if namespace.get("_WLUT") is not namespace.get("_p010_bound_lut"):
        raise Expert224ContractError("Stock expert LUT dictionary was replaced/shared with another branch")
    if set(namespace.get("_p010_bound_normalization", {})) != {"_MEAN", "_STD"} or any(
        not namespace["torch"].equal(namespace[name], expected)
        for name, expected in namespace["_p010_bound_normalization"].items()
    ):
        raise Expert224ContractError("Stock normalization tensors were changed after extraction")
    if (namespace["LAB"] != list(LABELS) or namespace["K_EVAL"] != 94 or namespace["IMG"] != 384
            or namespace["NORM"] != "imagenet" or namespace["CROP_FRAC"] != 0.8 or namespace["MAXS"] != 96
            or namespace["CROP_MM"] != 140.0
            or namespace["SLOTS"] != [("Sagittal", 1, 26), ("Sagittal", 0, 22),
                                      ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]):
        raise Expert224ContractError("Stock numerical constants were changed after extraction")
    return binding


def _device(namespace: dict[str, Any], device: Any):
    binding = _namespace_binding(namespace)
    torch = namespace["torch"]
    dev = torch.device(device)
    if binding["artificial_mode"]:
        if dev.type != binding["artificial_device"]:
            raise Expert224ContractError("Execution device differs from the explicit artificial device")
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise Expert224ContractError("Artificial CUDA verification requires available CUDA")
    elif dev.type != "cuda" or not torch.cuda.is_available():
        raise Expert224ContractError("Production author224 requires CUDA before checkpoint loading")
    return dev


def load_expert224(path: str | Path, device: Any, *, stock_namespace: dict[str, Any],
                   artificial_expected_sha256: str | None = None):
    """Verify SHA/architecture/state, then invoke the unchanged strict loader."""
    namespace = stock_namespace
    binding = _namespace_binding(namespace)
    dev = _device(namespace, device)
    torch = namespace["torch"]
    path = Path(path)
    if artificial_expected_sha256 is not None and not binding["artificial_mode"]:
        raise Expert224ContractError("Production checkpoint SHA cannot be overridden")
    if binding["artificial_mode"] and artificial_expected_sha256 is None:
        raise Expert224ContractError("Artificial checkpoint requires its explicit test SHA")
    expected_sha = artificial_expected_sha256 if binding["artificial_mode"] else CHECKPOINT_SHA256
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise Expert224ContractError("Author224 checkpoint SHA differs")
    if not binding["artificial_mode"] and path.stat().st_size != CHECKPOINT_BYTES:
        raise Expert224ContractError("Author224 checkpoint file size differs")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, Mapping) or ck.get("arch") != ARCH or ck.get("res") != RESOLUTION:
        raise Expert224ContractError("Explicit checkpoint arch/res do not match author224")
    state = ck.get("model")
    if not isinstance(state, Mapping) or not state or not all(isinstance(key, str) for key in state):
        raise Expert224ContractError("Author224 model state is missing or malformed")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise Expert224ContractError(f"Checkpoint state is not a tensor: {key}")
        if value.is_complex():
            raise Expert224ContractError(f"Checkpoint complex state is unsupported: {key}")
        if value.is_floating_point():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise Expert224ContractError(f"Checkpoint floating state must be finite FP32: {key}")
    cls = state.get("clsW")
    if cls is None or cls.ndim != 2 or cls.shape[0] != 12 or cls.shape[1] < 1:
        raise Expert224ContractError("Checkpoint classifier must contain12 target rows")
    feature_dim = int(cls.shape[1])
    head_shapes = {"clsb": (12,), "norm.weight": (feature_dim,), "norm.bias": (feature_dim,),
                   "att.0.weight": (256, feature_dim), "att.0.bias": (256,),
                   "att.3.weight": (12, 256), "att.3.bias": (12,)}
    if any(key not in state or tuple(state[key].shape) != shape for key, shape in head_shapes.items()):
        raise Expert224ContractError("Checkpoint attention/head dimensions differ")
    label_fields = {key: ck[key] for key in ("labels", "label_order", "target_columns", "LAB") if key in ck}
    if any(not isinstance(value, (list, tuple)) or tuple(value) != LABELS for value in label_fields.values()):
        raise Expert224ContractError("Checkpoint explicit target metadata differs")
    state_count = len(state)
    del state, cls, ck
    model, resolution = namespace["load_model"](str(path), ARCH, RESOLUTION, str(dev), ngpu=1)
    if resolution != RESOLUTION or model.n != 12 or tuple(model.clsW.shape) != (12, feature_dim):
        raise Expert224ContractError("Stock strict loader returned unexpected architecture/head")
    if any(module.training for module in model.modules()):
        raise Expert224ContractError("Stock model is not completely in eval mode")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise Expert224ContractError("Stock model parameters must remain FP32 for autocast")
    if any(parameter.device != dev for parameter in model.parameters()):
        # torch.device('cuda') has no index: compare against its resolved device.
        if dev.type != "cuda" or dev.index is not None or any(
            parameter.device.type != "cuda" or parameter.device.index != torch.cuda.current_device()
            for parameter in model.parameters()
        ):
            raise Expert224ContractError("Stock model is on an unexpected device")
    receipt = {
        "status": f"artificial_{dev.type}_passed" if binding["artificial_mode"] else "passed",
        "production_qualified": not binding["artificial_mode"], "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": path.stat().st_size, "checkpoint_expected_sha256": expected_sha,
        "author_notebook_sha256": AUTHOR_NOTEBOOK_SHA256, "stock_source_sha256": STOCK_SOURCE_SHA256,
        "arch": ARCH, "resolution": resolution, "labels": list(LABELS), "head_outputs": 12,
        "state_tensors": state_count, "state_floating_dtype": "float32", "state_finite": True,
        "strict_load": True, "model_eval": True, "device": str(dev),
        "checkpoint_label_metadata_present": bool(label_fields),
        "checkpoint_label_metadata_fields": sorted(label_fields),
        "label_order_evidence": "literal author LAB and12-head; explicit checkpoint labels checked when present",
        "author_gold_selection_reuse": True,
        "author_gold_training_exclusion_independently_verified": False,
        "own_gold_fit_or_selection": False,
        "own_OOF_checkpoint": False,
        "author_provenance": "seed42 SWA15/13/12; publication card reports Gold58 selection/development reuse",
        "namespace_identity": id(namespace),
    }
    model._p010_expert224_binding = dict(receipt)
    return model, receipt


def _array_sha256(array) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def infer_expert224(stock_namespace: dict[str, Any], model: Any, vol: Any, mask: Any, device: Any):
    """Invoke exact windows/single stock forward with independent invocation gates."""
    namespace = stock_namespace
    binding = _namespace_binding(namespace)
    dev = _device(namespace, device)
    np, torch = namespace["np"], namespace["torch"]
    model_binding = getattr(model, "_p010_expert224_binding", {})
    if model_binding.get("namespace_identity") != id(namespace) or not model_binding.get("strict_load"):
        raise Expert224ContractError("Expert model is not bound to this stock strict loader")
    if not binding["artificial_mode"] and model_binding.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise Expert224ContractError("Production expert model checkpoint binding differs")
    if not isinstance(vol, np.ndarray) or vol.shape != (96, 384, 384) or vol.dtype != np.uint8:
        raise Expert224ContractError("Shared volume must be96x384x384 uint8")
    if not isinstance(mask, np.ndarray) or mask.shape != (96,) or mask.dtype != np.uint8:
        raise Expert224ContractError("Shared mask must be96 uint8")
    if not bool(np.isin(mask, [0, 1]).all()):
        raise Expert224ContractError("Shared valid mask must contain only0/1")
    if any(module.training for module in model.modules()):
        raise Expert224ContractError("Expert eval mode was changed after loading")
    before = {"volume_sha256": _array_sha256(vol), "mask_sha256": _array_sha256(mask)}
    events: list[dict[str, Any]] = []
    counts = {"model": 0, "backbone": 0}

    def observe(kind: str, shape: tuple[int, ...]):
        def hook(module, args):
            counts[kind] += 1
            if counts[kind] != 1:
                # A ValueError bypasses the stock RuntimeError-to-FP32 handler.
                raise Expert224ContractError("Stock author224 retried/fell back instead of one forward")
            if len(args) != 1 or tuple(args[0].shape) != shape or args[0].dtype != torch.float32:
                raise Expert224ContractError(f"Stock {kind} batch shape/dtype differs")
            if module.training or torch.is_grad_enabled():
                raise Expert224ContractError("Expert forward must use eval and no_grad")
            autocast = bool(torch.is_autocast_enabled("cuda"))
            amp_dtype = str(torch.get_autocast_dtype("cuda")) if autocast else None
            if dev.type == "cuda" and (not autocast or amp_dtype != "torch.float16"):
                raise Expert224ContractError("CUDA stock forward did not use FP16 autocast")
            events.append({"kind": kind, "shape": list(shape), "input_dtype": str(args[0].dtype),
                           "training": bool(module.training), "grad_enabled": bool(torch.is_grad_enabled()),
                           "cuda_autocast": autocast, "autocast_dtype": amp_dtype})
        return hook

    handles = []
    try:
        handles.append(model.register_forward_pre_hook(observe("model", (1, 94, 3, 224, 224))))
        handles.append(model.backbone.register_forward_pre_hook(observe("backbone", (94, 3, 224, 224))))
        xwins = namespace["gpu_windows"](vol, mask, 94, RESOLUTION, str(dev))
        if tuple(xwins.shape) != (94, 3, 224, 224) or xwins.dtype != torch.float32:
            raise Expert224ContractError("Stock224 windows shape/dtype differs")
        if not bool(torch.isfinite(xwins).all()):
            raise Expert224ContractError("Stock224 windows contain nonfinite values")
        raw = namespace["infer_probs"](model, xwins, str(dev))
        if not isinstance(raw, np.ndarray) or raw.shape != (12,) or raw.dtype != np.float32:
            raise Expert224ContractError("Stock expert must return12 float32 raw probabilities")
        if not bool(np.isfinite(raw).all()) or not bool(((raw >= 0) & (raw <= 1)).all()):
            raise Expert224ContractError("Stock expert probabilities are nonfinite/out of range")
        if counts != {"model": 1, "backbone": 1}:
            raise Expert224ContractError("Stock expert did not execute exactly one full94 forward")
    finally:
        for handle in handles:
            handle.remove()
        # On success or error, shared input mutation is a fatal contract error.
        after = {"volume_sha256": _array_sha256(vol), "mask_sha256": _array_sha256(mask)}
        if before != after:
            raise Expert224ContractError("Expert mutated the shared volume/mask")
    receipt = {
        "status": f"artificial_{dev.type}_passed" if binding["artificial_mode"] else "passed",
        "production_qualified": not binding["artificial_mode"], "checkpoint_sha256": model_binding["checkpoint_sha256"],
        "author_notebook_sha256": AUTHOR_NOTEBOOK_SHA256, "stock_source_sha256": STOCK_SOURCE_SHA256,
        "arch": ARCH, "resolution": RESOLUTION, "labels": list(LABELS), "raw_outputs": 12,
        "raw_dtype": "float32", "finite": True, "in_range_0_1": True, "model_eval": True,
        "no_grad": True, "stock_forward_calls": 1, "backbone_forward_calls": 1,
        "backbone_batch_windows": 94, "views": 1, "fallback_calls": 0,
        "cuda_fp16_observed": dev.type == "cuda", "device": str(dev), "forward_events": events,
        "input_shape": [96, 384, 384], "input_dtype": "uint8", "mask_shape": [96],
        "mask_dtype": "uint8", "mask_valid_count": int(mask.sum()), "allmissing": not bool(mask.any()),
        "volume_mask_unchanged": True, "input_before": before, "input_after": after,
        "raw_probability_sha256": _array_sha256(raw),
        "author_gold_selection_reuse": True, "own_gold_fit_or_selection": False,
        "own_OOF_checkpoint": False,
    }
    return raw, receipt
