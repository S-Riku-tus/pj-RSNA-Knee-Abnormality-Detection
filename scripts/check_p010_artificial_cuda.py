"""Untrained tiny-encoder CUDA contracts; never reads MRI, teachers or real weights."""

import ast
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/qa/20261008-p010-shared224-artificial-cuda-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


author = load("_p010_cuda_author_runtime", ROOT / "scripts/p010_author224_runtime.py")
shared = load("_p010_cuda_shared_runtime", ROOT / "scripts/p010_shared_runtime.py")
builder = load("_p010_cuda_builder", ROOT / "scripts/build_p010_shared224.py")


class TinyEncoder(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, value):
        return self.projection(value.mean(dim=(-1, -2)))


class TinyTimm:
    @staticmethod
    def create_model(_arch, **kwargs):
        assert kwargs["pretrained"] is False and kwargs["num_classes"] == 0
        return TinyEncoder()


def base_namespace(source):
    names = {
        "INPUT_RES",
        "INPUT_MODE",
        "IMG",
        "MAXS",
        "SLOTS",
        "K_EVAL",
        "NORM",
        "_MEAN",
        "_STD",
        "SLOT_SIZES",
        "SAGITTAL_SLOTS",
        "_WLUT",
    }
    functions = {
        "RaptorClassifier",
        "_eval_centers",
        "_slot_index",
        "_mirror_tri",
        "infer_pair_probs",
        "gpu_windows",
        "infer_probs",
    }
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions:
            nodes.append(node)
    namespace = {"__name__": "_p010_untrained_base", "torch": torch, "np": np, "nn": nn, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<p010-original-base-numerics>", "exec"), namespace)
    return namespace


def run(output=OUTPUT):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required before artificial checkpoint preparation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    original = builder.text(json.loads(builder.CONTROL.read_text(encoding="utf-8"))["cells"][4])
    base = base_namespace(original)
    base_model = base["RaptorClassifier"](TinyEncoder(), F_dim=4).eval().to("cuda:0")
    stock = author.build_stock_source(builder.AUTHOR)
    expert = author.make_stock_namespace(stock, timm_module=TinyTimm(), artificial_device="cuda")
    # Save a new local untrained artificial state solely to exercise the strict loader.
    temporary_model = expert["RaptorClassifier"](TinyEncoder(), F_dim=4).eval()
    path = output / "untrained-artificial-only.pt"
    torch.save({"arch": author.ARCH, "res": 224, "model": temporary_model.state_dict()}, path)
    checkpoint_sha = author.sha256_file(path)
    model, load_receipt = author.load_expert224(
        path, "cuda:0", stock_namespace=expert, artificial_expected_sha256=checkpoint_sha
    )
    del temporary_model
    axis = np.arange(384, dtype=np.uint16)
    plane = (axis[:, None] * 5 + axis[None, :] * 7) % 256
    volume = ((np.arange(96, dtype=np.uint16)[:, None, None] * 11 + plane) % 256).astype(np.uint8)
    cases = []
    torch.cuda.reset_peak_memory_stats()
    for positions in (list(range(96)), [10, 48, 90], []):
        mask = np.zeros(96, dtype=np.uint8)
        mask[positions] = 1
        before = shared.shared_input_receipt(volume, mask)
        base_forward = []

        def base_hook(module, inputs, outputs):
            assert all(not part.training for part in module.modules())
            assert not torch.is_grad_enabled()
            assert tuple(inputs[0].shape) == (1, 94, 3, 320, 320)
            base_forward.append(tuple(outputs.shape))

        handle = base_model.register_forward_hook(base_hook)
        xw = base["gpu_windows"](volume, mask, 94, 320, "cuda:0")
        xm = base["gpu_windows"](volume, mask, 94, 320, "cuda:0", mirror=True)
        reference_base = base["infer_pair_probs"](base_model, xw, xm, "cuda:0")
        del xw, xm
        baseline_before_expert = reference_base.copy()
        raw, receipt = author.infer_expert224(expert, model, volume, mask, "cuda:0")
        reference_windows = expert["gpu_windows"](volume, mask, 94, 224, "cuda:0")
        reference_expert = expert["infer_probs"](model, reference_windows, "cuda:0")
        del reference_windows
        # Recompute the original base after sequential expert consumption.
        xw = base["gpu_windows"](volume, mask, 94, 320, "cuda:0")
        xm = base["gpu_windows"](volume, mask, 94, 320, "cuda:0", mirror=True)
        after_base = base["infer_pair_probs"](base_model, xw, xm, "cuda:0")
        del xw, xm
        handle.remove()
        assert len(base_forward) == 4 and all(shape == (1, 12) for shape in base_forward)
        assert np.array_equal(reference_base, after_base) and np.array_equal(reference_base, baseline_before_expert)
        assert np.array_equal(raw, reference_expert)
        assert shared.shared_input_receipt(volume, mask) == before
        assert receipt["cuda_fp16_observed"] and receipt["no_grad"] and receipt["fallback_calls"] == 0
        assert receipt["backbone_batch_windows"] == 94 and receipt["stock_forward_calls"] == 1
        assert not receipt["production_qualified"]
        assert all(parameter.grad is None for instance in (base_model, model) for parameter in instance.parameters())
        cases.append(
            {
                "valid_slices": len(positions),
                "base_reference_max_difference": 0,
                "stock224_reference_max_difference": 0,
                "base_forwards": len(base_forward),
                "expert_receipt": receipt,
                "shared_input_receipt": before,
                "untrained_model_parameters_received_gradient": False,
            }
        )
    torch.cuda.synchronize()
    record = {
        "schema_version": "p010_artificial_untrained_CUDA_contract_v1",
        "script_sha256": author.sha256_file(__file__),
        "author_helper_sha256": author.sha256_file(ROOT / "scripts/p010_author224_runtime.py"),
        "shared_helper_sha256": author.sha256_file(ROOT / "scripts/p010_shared_runtime.py"),
        "production_builder_sha256": author.sha256_file(ROOT / "scripts/build_p010_shared224.py"),
        "all_passed": True,
        "seed": 20261008,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "baseline_sha256": builder.digest(builder.CONTROL),
        "author_source_sha256": author.STOCK_SOURCE_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "actual_expert_checkpoint_sha256_verified": False,
        "encoder": "injected_random_TinyEncoder_Linear_3_to_4_not_CoAtNet",
        "checkpoint_created_untrained_only": True,
        "cases": cases,
        "case_count": len(cases),
        "elapsed_seconds": time.monotonic() - started,
        "tiny_encoder_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "real_MRI_decodes": 0,
        "teacher_values_consumed": 0,
        "optimiser_updates": 0,
        "weights_downloaded": 0,
        "production_qualified": False,
        "candidate_gate_passed": False,
        "limitation": "RTX4090 tiny untrained encoder does not prove actual 224 CoAtNet/T4 memory, speed or checkpoint compatibility.",
        "artificial_load_receipt": load_receipt,
    }
    (output / "checks.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("cases", "artificial_load_receipt")},
            ensure_ascii=False,
            indent=2,
        )
    )
