"""Stock-only CPU/CUDA contracts with random tiny encoder; no actual MRI/weights."""

import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/qa/20261009-p011-artificial-stock224-v1"


class TinyEncoder(torch.nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.projection(x.mean(dim=(-1, -2)))


class TinyTimm:
    def create_model(self, arch, **kwargs):
        return TinyEncoder()


def load_helper():
    spec = importlib.util.spec_from_file_location("_p011_artificial_author", ROOT / "scripts/p010_author224_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(output=OUTPUT):
    if not torch.cuda.is_available():
        raise RuntimeError("Available CUDA required for explicitly artificial forward")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    helper = load_helper()
    author = (
        ROOT / "artifacts/research/20261008-public-candidates/nartaa-rsna-knee-0945-efficient-224crop-v2/source.ipynb"
    )
    stock = helper.build_stock_source(author)
    torch.manual_seed(20261009)
    torch.cuda.manual_seed_all(20261009)
    torch.set_num_threads(2)  # Artificial CPU check only; no serving backend setter.
    cpu_namespace = helper.make_stock_namespace(stock, timm_module=TinyTimm(), artificial_device="cpu")
    cuda_namespace = helper.make_stock_namespace(stock, timm_module=TinyTimm(), artificial_device="cuda")
    if cpu_namespace["_WLUT"] is cuda_namespace["_WLUT"]:
        raise AssertionError("Independent namespace LUT isolation failed")
    original = cpu_namespace["RaptorClassifier"](TinyEncoder(), F_dim=4).eval()
    checkpoint = output / "artificial-untrained-only.pt"
    torch.save(
        {"arch": helper.ARCH, "res": 224, "labels": list(helper.LABELS), "model": original.state_dict()}, checkpoint
    )
    digest = helper.sha256_file(checkpoint)
    started = time.monotonic()
    cpu, cpu_load = helper.load_expert224(
        checkpoint, "cpu", stock_namespace=cpu_namespace, artificial_expected_sha256=digest
    )
    cuda, cuda_load = helper.load_expert224(
        checkpoint, "cuda:0", stock_namespace=cuda_namespace, artificial_expected_sha256=digest
    )
    for namespace, device in ((cpu_namespace, "cpu"), (cuda_namespace, "cuda:0")):
        try:
            helper.load_expert224(checkpoint, device, stock_namespace=namespace, artificial_expected_sha256="0" * 64)
        except helper.Expert224ContractError:
            pass
        else:
            raise AssertionError("Incorrect artificial checkpoint SHA accepted")
    axis = np.arange(384, dtype=np.uint16)
    plane = (axis[:, None] * 5 + axis[None, :] * 7) % 256
    volume = ((np.arange(96, dtype=np.uint16)[:, None, None] * 11 + plane) % 256).astype(np.uint8)
    cases = []
    torch.cuda.reset_peak_memory_stats()
    for name, valid in (("full", list(range(96))), ("sparse", [10, 48, 90]), ("allmissing", [])):
        mask = np.zeros(96, dtype=np.uint8)
        mask[valid] = 1
        before = (hashlib.sha256(volume.tobytes()).hexdigest(), hashlib.sha256(mask.tobytes()).hexdigest())
        records = {}
        for device, ns, model in (("cpu", cpu_namespace, cpu), ("cuda:0", cuda_namespace, cuda)):
            raw, receipt = helper.infer_expert224(ns, model, volume, mask, device)
            if raw.shape != (12,) or not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all():
                raise AssertionError("Stock finite12 probability contract failed")
            if receipt["production_qualified"] or receipt["fallback_calls"] != 0 or receipt["stock_forward_calls"] != 1:
                raise AssertionError("Artificial single stock forward contract failed")
            if receipt["labels"] != list(helper.LABELS) or not receipt["no_grad"]:
                raise AssertionError("Target order or no_grad contract failed")
            if device.startswith("cuda") and not receipt["cuda_fp16_observed"]:
                raise AssertionError("CUDA stock FP16 not observed")
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError("Artificial model accumulated gradients")
            records[device] = {"raw": raw.tolist(), "receipt": receipt}
        after = (hashlib.sha256(volume.tobytes()).hexdigest(), hashlib.sha256(mask.tobytes()).hexdigest())
        assert before == after
        cases.append(
            {
                "mask_case": name,
                "valid_slices": len(valid),
                "native_uint8_input_unchanged": True,
                "observations": records,
            }
        )
    with tempfile.TemporaryDirectory(dir=output) as temp:
        changed = Path(temp) / "wrong-labels.pt"
        torch.save(
            {"arch": helper.ARCH, "res": 224, "labels": list(reversed(helper.LABELS)), "model": original.state_dict()},
            changed,
        )
        try:
            helper.load_expert224(
                changed, "cpu", stock_namespace=cpu_namespace, artificial_expected_sha256=helper.sha256_file(changed)
            )
        except helper.Expert224ContractError:
            pass
        else:
            raise AssertionError("Wrong target metadata accepted")
    torch.cuda.synchronize()
    record = {
        "schema_version": "p011_stock224_artificial_CPU_CUDA_v1",
        "all_passed": True,
        "seed": 20261009,
        "torch": str(torch.__version__),
        "GPU": torch.cuda.get_device_name(0),
        "artificial_checkpoint_SHA": digest,
        "stock_source_SHA": helper.STOCK_SOURCE_SHA256,
        "cpu_load": cpu_load,
        "cuda_load": cuda_load,
        "cases": cases,
        "negative_SHA_and_label_guards_pass": True,
        "allmissing_policy_verified": "Exact stock outputs finite12 under both artificial CPU and CUDA; retained, no base-only substitution.",
        "tiny_CUDA_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20,
        "elapsed_seconds": time.monotonic() - started,
        "real_MRI_or_public_checkpoint": False,
        "training_updates": 0,
        "candidate_gate_passed": False,
        "production_qualified": False,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "scripts/p010_author224_runtime.py",
                ROOT / "scripts/p011_independent_runtime.py",
                ROOT / "scripts/p011_contracts.py",
            )
        },
        "limitations": "One RTX4090/random TinyEncoder, not actual twoT4/CoAtNet payload memory, precision repeatability or accuracy.",
    }
    with (output / "checks.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return record


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_passed",
                    "GPU",
                    "elapsed_seconds",
                    "tiny_CUDA_peak_reserved_mib",
                    "candidate_gate_passed",
                    "production_qualified",
                )
            }
        )
    )
