"""Check new observation/persistence with original classes and tiny CUDA encoders."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/qa/20261009-p010-AA-artificial-cuda-v4-r2"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load("_p010_AA_artificial_original_contracts", ROOT / "scripts/check_p010_artificial_cuda.py")
transport = load("_p010_AA_artificial_transport", ROOT / "scripts/p010_aa_transport.py")
with mock.patch.dict(sys.modules, {"p010_aa_transport": transport}):
    runtime = load("_p010_AA_artificial_runtime", ROOT / "scripts/p010_aa_runtime_v4_r2.py")


def run(output=OUTPUT):
    if not torch.cuda.is_available():
        raise RuntimeError("Artificial forward requires available CUDA before any model creation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(20261009)
    torch.cuda.manual_seed_all(20261009)
    torch.cuda.reset_peak_memory_stats()
    original = old.builder.text(json.loads(old.builder.CONTROL.read_text(encoding="utf-8"))["cells"][4])
    base = old.base_namespace(original)
    base_model = base["RaptorClassifier"](old.TinyEncoder(), F_dim=4).eval().to("cuda:0")
    stock = old.author.make_stock_namespace(
        old.author.build_stock_source(old.builder.AUTHOR), timm_module=old.TinyTimm(), artificial_device="cuda"
    )
    untrained = stock["RaptorClassifier"](old.TinyEncoder(), F_dim=4).eval()
    checkpoint = output / "artificial-untrained-only.pt"
    torch.save({"arch": old.author.ARCH, "res": 224, "model": untrained.state_dict()}, checkpoint)
    expert, load_receipt = old.author.load_expert224(
        checkpoint, "cuda:0", stock_namespace=stock, artificial_expected_sha256=old.author.sha256_file(checkpoint)
    )
    del untrained
    uids = ["artificial-full", "artificial-sparse", "artificial-missing"]
    labels = list(stock["LAB"])

    def flags():
        return {
            "artificial_GPU_only": True,
            "torch": torch.__version__,
            "CUDA": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }

    base_recorder = runtime.PhaseRecorder(output, "base_original_ordinary_A1", uids, labels, flags)
    stock_recorder = runtime.PhaseRecorder(output, "stock224_ordinary_A1", uids, labels, flags)
    base_checkpoint = output / "artificial-untrained-base.pt"
    torch.save({"artificial_only": True, "model": base_model.state_dict()}, base_checkpoint)
    with mock.patch.dict(runtime.BASE_CHECKPOINT_SHA256_BY_NAME, {base_checkpoint.name: runtime.sha(base_checkpoint)}):
        base_recorder.load(base_model, 320, checkpoint_path=base_checkpoint)
    stock_recorder.load(expert, 224, checkpoint_receipt=load_receipt)
    handle = base_model.register_forward_hook(lambda _module, inputs, result: base_recorder.forward(inputs, result))
    axis = np.arange(384, dtype=np.uint16)
    plane = (axis[:, None] * 5 + axis[None, :] * 7) % 256
    volume = ((np.arange(96, dtype=np.uint16)[:, None, None] * 11 + plane) % 256).astype(np.uint8)
    cases = []
    try:
        for uid, valid in zip(uids, (list(range(96)), [10, 48, 90], []), strict=True):
            mask = np.zeros(96, dtype=np.uint8)
            mask[valid] = 1
            before = old.shared.shared_input_receipt(volume, mask)
            base_recorder.begin(uid, volume, mask, "cuda:0", before)
            plain = base["gpu_windows"](volume, mask, 94, 320, "cuda:0")
            mirrored = base["gpu_windows"](volume, mask, 94, 320, "cuda:0", mirror=True)
            raw_base = base["infer_pair_probs"](base_model, plain, mirrored, "cuda:0")
            del plain, mirrored
            base_recorder.complete(raw_base.tolist(), "cuda:0")
            stock_recorder.begin(uid, volume, mask, "cuda:0", before)
            raw_expert, inference = old.author.infer_expert224(stock, expert, volume, mask, "cuda:0")
            stock_recorder.complete(raw_expert.tolist(), "cuda:0", inference=inference)
            assert before == old.shared.shared_input_receipt(volume, mask)
            assert base_recorder.raw[uid]["raw_probabilities"] == raw_base.tolist()
            assert stock_recorder.raw[uid]["raw_probabilities"] == raw_expert.tolist()
            assert inference["cuda_fp16_observed"] and inference["no_grad"] and inference["fallback_calls"] == 0
            assert not inference["production_qualified"]
            assert all(parameter.grad is None for model in (base_model, expert) for parameter in model.parameters())
            cases.append(
                {
                    "valid_slices": len(valid),
                    "base_views": len(base_recorder.raw[uid]["view_events"]),
                    "raw_base_saved_bit_exact": True,
                    "raw_expert_saved_bit_exact": True,
                    "native_uint8_volume_mask_unchanged": True,
                    "no_model_gradient": True,
                }
            )
    finally:
        handle.remove()
    with mock.patch.dict(sys.modules, {"p010_shared_metrics": old.shared}):
        base_report = base_recorder.finish(state={"artificial_only": True}, resources={}, peaks=[])
        stock_report = stock_recorder.finish(state={"artificial_only": True}, resources={}, peaks=[])
    assert len(base_recorder.forwards) == 6
    assert all(
        item["input_shape"] == [1, 94, 3, 320, 320]
        and item["output_shape"] == [1, 12]
        and not item["grad_enabled"]
        and item["autocast_cuda_enabled"]
        for item in base_recorder.forwards
    )
    assert all(item["base_views"] == 2 for item in cases)
    assert base_report["input_receipts"] == stock_report["input_receipts"]
    torch.cuda.synchronize()
    record = {
        "schema_version": "p010_AA_observation_artificial_CUDA_v4_r2",
        "all_passed": True,
        "seed": 20261009,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cases": cases,
        "phase_raw_CSVs": [base_report["raw_csv"], stock_report["raw_csv"]],
        "source_sha256": {
            str(path.relative_to(ROOT)): runtime.sha(path)
            for path in (
                Path(__file__),
                ROOT / "scripts/p010_aa_runtime_v4_r2.py",
                ROOT / "scripts/p010_aa_transport.py",
                ROOT / "scripts/p010_author224_runtime.py",
                ROOT / "scripts/p010_shared_runtime.py",
                old.builder.CONTROL,
            )
        },
        "elapsed_seconds": time.monotonic() - started,
        "tiny_untrained_GPU_peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20,
        "artificial_checkpoint_sha256": load_receipt["checkpoint_sha256"],
        "actual_model_checkpoint_or_MRI_access": False,
        "optimizer_updates": 0,
        "candidate_gate_passed": False,
        "production_qualified": False,
        "scope": "Original Raptor numerical classes with injected random TinyEncoder on one RTX4090. Does not prove T4x2 scheduling, actual CoAtNet memory, precision reproducibility or AUC.",
    }
    runtime.write_json_new(output / "checks.json", record)
    return record


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key not in ("cases", "source_sha256")}, indent=2))
