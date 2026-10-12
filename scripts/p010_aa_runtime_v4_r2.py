"""Persist source-bound A/A diagnostics before any cross-phase comparison."""

import contextlib
import gc
import hashlib
import json
import math
import os
import struct
import sys
import threading
from pathlib import Path

from p010_aa_transport import fixed_pipeline

RAW_ABSOLUTE_TOLERANCE = 1e-7
RAW_RELATIVE_TOLERANCE = 0.0
GPU_LIMIT_MIB = 13000
BASE_CHECKPOINT_SHA256_BY_NAME = {
    "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
}
AUTHOR224_CHECKPOINT_SHA256 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"
PHASES = (
    ("base_original_ordinary_A1", "base", "original"),
    ("base_original_ordinary_A2", "base", "original"),
    ("base_bounded_ordinary_A1", "base", "ordinary"),
    ("base_bounded_ordinary_A2", "base", "ordinary"),
    ("base_bounded_fixed_A1", "base", "fixed"),
    ("base_bounded_fixed_A2", "base", "fixed"),
    ("stock224_ordinary_A1", "stock224", "ordinary"),
    ("stock224_ordinary_A2", "stock224", "ordinary"),
    ("stock224_fixed_A1", "stock224", "fixed"),
    ("stock224_fixed_A2", "stock224", "fixed"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def float32_values(values):
    if len(values) != 12 or any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
        raise ValueError("Exactly twelve finite probabilities in [0,1] are required")
    packed = struct.pack("<12f", *values)
    result = list(struct.unpack("<12f", packed))
    return result, {"little_endian_float32_bits_hex": packed.hex(), "sha256": hashlib.sha256(packed).hexdigest()}


def backend_flags():
    import timm
    import torch

    matmul = torch.backends.cuda.matmul
    cuda = torch.backends.cuda
    sdp = {}
    for key in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        function = getattr(cuda, key, None)
        sdp[key] = None if function is None else bool(function())
    reduction = {}
    for key in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"):
        reduction[key] = getattr(matmul, key, None)
    return {
        "torch": torch.__version__,
        "timm": timm.__version__,
        "python": sys.version,
        "CUDA": torch.version.cuda,
        "cuDNN": torch.backends.cudnn.version(),
        "cudnn": {
            key: getattr(torch.backends.cudnn, key) for key in ("enabled", "benchmark", "deterministic", "allow_tf32")
        },
        "cuda_matmul": {"allow_tf32": matmul.allow_tf32, **reduction},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "SDP": sdp,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "devices": [
            {
                "device": f"cuda:{index}",
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                "multi_processor_count": torch.cuda.get_device_properties(index).multi_processor_count,
            }
            for index in range(torch.cuda.device_count())
        ],
        "driver_version": None,
        "driver_version_scope": "Not exposed by a relied-upon public PyTorch API; no extra command executed.",
    }


class PhaseRecorder:
    def __init__(self, root, phase, uids, labels, flags_provider=backend_flags):
        if phase not in {row[0] for row in PHASES} or len(uids) != len(set(uids)) or len(labels) != 12:
            raise ValueError("Preregistered phase, unique studies and twelve labels are required")
        self.root, self.phase = Path(root), phase
        self.directory = self.root / ("p010_aa_" + phase)
        self.directory.mkdir(exist_ok=False)
        self.uids, self.labels = list(uids), list(labels)
        self.positions = {uid: index for index, uid in enumerate(uids)}
        self.lock = threading.Lock()
        self.current = threading.local()
        self.inputs, self.raw, self.loads, self.forwards = {}, {}, [], []
        self.checkpoint_digests = {}
        self.start_flags = flags_provider()
        self.flags_provider = flags_provider
        self.finished = False
        self.result = None

    def begin(self, uid, volume, mask, device, input_receipt):
        del volume, mask
        if uid not in self.positions:
            raise ValueError("Unexpected study in diagnostic input")
        self.current.uid, self.current.device = uid, str(device)
        self.current.views = []
        with self.lock:
            if uid in self.inputs:
                raise ValueError("Repeated diagnostic study")
            self.inputs[uid] = dict(input_receipt)

    def load(self, model, res, *, checkpoint_path=None, checkpoint_receipt=None):
        checkpoint = None
        if checkpoint_path is not None:
            path = Path(checkpoint_path).resolve()
            expected = BASE_CHECKPOINT_SHA256_BY_NAME.get(path.name)
            if expected is None:
                raise ValueError("Base load filename is outside the audited checkpoint inventory")
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            cached = key in self.checkpoint_digests
            digest = self.checkpoint_digests.get(key)
            if digest is None:
                digest = sha(path)
                self.checkpoint_digests[key] = digest
            if digest != expected:
                raise ValueError("Actual base checkpoint SHA differs from the audited InputV1 inventory")
            checkpoint = {
                "actual_path": str(path),
                "filename": path.name,
                "actual_bytes": stat.st_size,
                "actual_sha256": digest,
                "expected_sha256": expected,
                "CPU_digest_cached_same_path_size_mtime": cached,
                "SHA_verified": True,
            }
        elif checkpoint_receipt is not None:
            if not checkpoint_receipt.get("strict_load") or not checkpoint_receipt.get("checkpoint_sha256"):
                raise ValueError("Stock checkpoint strict binding receipt missing")
            if (
                checkpoint_receipt.get("production_qualified")
                and checkpoint_receipt["checkpoint_sha256"] != AUTHOR224_CHECKPOINT_SHA256
            ):
                raise ValueError("Actual stock224 checkpoint SHA differs from audited InputV1")
            checkpoint = {
                "actual_sha256": checkpoint_receipt["checkpoint_sha256"],
                "expected_sha256": checkpoint_receipt["checkpoint_expected_sha256"],
                "actual_bytes": checkpoint_receipt["checkpoint_bytes"],
                "SHA_verified": True,
                "strict_load": True,
                "production_qualified": checkpoint_receipt["production_qualified"],
            }
        manifest = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in model.state_dict().items()]
        parameter = next(model.parameters())
        receipt = {
            "device": str(parameter.device),
            "resolution": res,
            "tensor_count": len(manifest),
            "tensor_shape_dtype_sha256": hashlib.sha256(
                json.dumps(manifest, separators=(",", ":")).encode()
            ).hexdigest(),
            "head_shapes": {
                name: list(tensor.shape)
                for name, tensor in model.state_dict().items()
                if name in ("clsW", "clsb", "norm.weight", "norm.bias", "att.0.weight", "att.3.weight")
            },
            "model_class": type(model).__qualname__,
            "parameter_dtype": str(parameter.dtype),
            "actual_checkpoint_binding": checkpoint,
            "all_eval": all(not module.training for module in model.modules()),
            "strict_checkpoint_source_loader_not_replaced": True,
            "backend_after_source_before_inference": self.flags_provider(),
        }
        with self.lock:
            self.loads.append(receipt)

    def forward(self, inputs, output):
        import torch

        uid = getattr(self.current, "uid", None)
        if uid not in self.positions or self.current.device != str(inputs[0].device):
            raise ValueError("Model forward lost its thread-local study/device identity")
        receipt = {
            "UID_sha256": hashlib.sha256(uid.encode()).hexdigest(),
            "device": str(inputs[0].device),
            "input_shape": list(inputs[0].shape),
            "input_dtype": str(inputs[0].dtype),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "grad_enabled": bool(torch.is_grad_enabled()),
            "autocast_cuda_enabled": bool(torch.is_autocast_enabled("cuda")),
            "view_index": len(self.current.views),
            "finite_guard": "Unchanged source hook checks finiteness after this observation.",
        }
        self.current.views.append(receipt)
        with self.lock:
            self.forwards.append(receipt)

    def complete(self, raw, device, *, inference=None):
        uid = getattr(self.current, "uid", None)
        if uid not in self.positions or self.current.device != str(device):
            raise ValueError("Raw probability identity differs from current consumer")
        values, bits = float32_values(raw)
        receipt = {
            "uid": uid,
            "UID_sha256": hashlib.sha256(uid.encode()).hexdigest(),
            "device": str(device),
            "raw_probabilities": values,
            "raw_float32": bits,
            "input": self.inputs[uid],
            "view_events": list(self.current.views),
            "inference": inference,
        }
        with self.lock:
            if uid in self.raw:
                raise ValueError("Repeated raw probability output")
            write_json_new(self.directory / f"study-{self.positions[uid]:06d}.json", receipt)
            self.raw[uid] = receipt

    def finish(self, *, state=None, exception=None, resources=None, peaks=None):
        if self.finished:
            raise ValueError("Phase receipt must be written exactly once")
        from p010_shared_metrics import write_submission_exclusive

        done = [uid for uid in self.uids if uid in self.raw]
        raw_path = self.root / f"p010_aa_{self.phase}_raw.csv"
        if done:
            write_submission_exclusive(
                raw_path, done, [self.raw[uid]["raw_probabilities"] for uid in done], self.labels
            )
        value = {
            "schema_version": "p010_AA_phase_receipt_v4",
            "phase": self.phase,
            "status": "phase_complete_before_comparison"
            if exception is None and len(done) == len(self.uids)
            else "phase_failed",
            "uids": self.uids,
            "labels": self.labels,
            "completed_UIDs": done,
            "missing_UID_sha256": [
                hashlib.sha256(uid.encode()).hexdigest() for uid in self.uids if uid not in self.raw
            ],
            "raw": {uid: self.raw[uid] for uid in done},
            "input_receipts": self.inputs,
            "model_load_receipts": self.loads,
            "forward_events": self.forwards,
            "backend_before_source": self.start_flags,
            "backend_after_phase": self.flags_provider(),
            "raw_csv": None
            if not done
            else {"path": raw_path.name, "sha256": sha(raw_path), "actual_bytes": raw_path.stat().st_size},
            "state": state,
            "resources": resources,
            "torch_peak_memory_mib": peaks,
            "exception": None if exception is None else {"type": type(exception).__name__, "message": str(exception)},
            "candidate_gate_passed": False,
            "accuracy_evaluated": False,
            "protected_prediction_literals_unchanged": True,
            "qualification_guard_not_yet_evaluated": True,
        }
        destination = self.root / f"P010_AA_{self.phase}.json"
        write_json_new(destination, value)
        self.finished, self.result = True, value
        return value


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Source-bound A/A observer insertion point changed")
    return source.replace(old, new, 1)


def observed_main_source(source, phase, *, original_transport):
    from p010_previous_metrics import observed_source

    result = observed_source(source, phase, original_transport=original_transport)
    if original_transport:
        old = "            sid, vol, mask = item; i = pos[sid]"
        result = replace_once(
            result, old, old + "\n            _AA_recorder.begin(sid,vol,mask,d,shared_input_receipt(vol,mask))"
        )
    else:
        old = "        sid, vol, mask = item\n        i = pos[sid]"
        result = replace_once(
            result, old, old + "\n        _AA_recorder.begin(sid,vol,mask,d,shared_input_receipt(vol,mask))"
        )
    result = replace_once(
        result,
        "_p010_base_loaded.append(str(next(model.parameters()).device))",
        "_p010_base_loaded.append(str(next(model.parameters()).device))\n    _AA_recorder.load(model,res,checkpoint_path=args[0])",
    )
    result = replace_once(
        result,
        "    def observe(module, inputs, output):",
        "    def observe(module, inputs, output):\n        _AA_recorder.forward(inputs,output)",
    )
    result = replace_once(
        result,
        '    _p010_base_pairs.append({"device":str(args[3]),"finite":True})',
        '    _p010_base_pairs.append({"device":str(args[3]),"finite":True})\n    _AA_recorder.complete(value.tolist(),args[3])',
    )
    compile(result, "<p010-AA-observed-main>", "exec")
    return result


@contextlib.contextmanager
def fixed_dispatch_context(assignment):
    import p010_shared_metrics as shared

    original = shared.run_bounded_pipeline
    replacement = fixed_pipeline(original, assignment)
    shared.run_bounded_pipeline = replacement
    try:
        yield
    finally:
        shared.run_bounded_pipeline = original


def torch_peaks():
    import torch

    return [
        {
            "device": f"cuda:{index}",
            "allocated": torch.cuda.max_memory_allocated(index) / 2**20,
            "reserved": torch.cuda.max_memory_reserved(index) / 2**20,
        }
        for index in range(torch.cuda.device_count())
    ]


def resource_result_or_unentered(resources):
    if resources.started is None:
        return {"status": "resource_context_not_entered", "elapsed_seconds": None}
    return resources.result()


def enforce_phase_resource_contract(report):
    from p010_previous_metrics import resource_gate

    if report["status"] != "phase_complete_before_comparison":
        raise ValueError("Incomplete diagnostic phase")
    resource_gate(report["resources"], [value["reserved"] for value in report["torch_peak_memory_mib"]])


def base_phase(source, phase, uids, labels, *, dispatch):
    import torch
    from p006_metrics import Resources, check_csv
    from p009_metrics import capture_stdout, embedded_source, prepare_fastread, registered_namespace

    root = Path("/kaggle/working")
    binding = prepare_fastread(source, {hashlib.sha256(embedded_source(source)).hexdigest()})
    workers = root / f"p010_workers_{phase}"
    workers.mkdir(exist_ok=False)
    recorder = PhaseRecorder(root, phase, uids, labels)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    for name in ("submission.csv", "_sota_0949.csv"):
        (root / name).unlink(missing_ok=True)
    if (root / "p010_expert_raw.csv").exists():
        raise ValueError("A/A base-only phase must not create an expert candidate file")
    original_transport = dispatch == "original"
    runtime = observed_main_source(source, phase, original_transport=original_transport)
    values = {
        "__file__": str(root / (phase + ".py")),
        "_AA_recorder": recorder,
        "_p010_worker_directory": str(workers),
        "_p010_expected_fastread_sha": binding["source_sha256"],
    }
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    assignment = {uid: devices[index % len(devices)] for index, uid in enumerate(uids)}
    context = fixed_dispatch_context(assignment) if dispatch == "fixed" else contextlib.nullcontext()
    state, error, resources = None, None, Resources()
    name = "_p010_AA_" + phase
    try:
        with registered_namespace(name, values) as namespace, context, resources, capture_stdout() as captured:
            exec(compile(runtime, f"<{name}>", "exec"), namespace)
            state = namespace["_p010_state"]
            state.update(
                {
                    "base_forward_observations": namespace["_p010_base_forwards"],
                    "base_model_load_devices": namespace["_p010_base_loaded"],
                    "task_pickle_identity": namespace["_p010_task_identity"],
                    "worker_initializers": [json.loads(path.read_text()) for path in sorted(workers.glob("*.json"))],
                    "runtime_source_sha256": hashlib.sha256(runtime.encode()).hexdigest(),
                    "prediction_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "dispatch_policy": dispatch,
                    "fixed_assignment": assignment if dispatch == "fixed" else None,
                }
            )
            main = root / f"p010_aa_{phase}_main.csv"
            with main.open("xb") as stream:
                stream.write((root / "_sota_0949.csv").read_bytes())
            state["main_csv"] = check_csv(main, uids, labels)
    except BaseException as exception:
        error = exception
        if hasattr(exception, "receipt"):
            state = {"transport_failure_receipt": exception.receipt, "dispatch_policy": dispatch}
        raise
    finally:
        report = recorder.finish(
            state=state, exception=error, resources=resource_result_or_unentered(resources), peaks=torch_peaks()
        )
        (root / "submission.csv").unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
    # Persisted evidence is authoritative even when these internal gates fail.
    assert "FALLBACK" not in captured.getvalue() and "FAILED (" not in captured.getvalue()
    assert captured.getvalue().count("wrote /kaggle/working/submission.csv |") == 1
    assert len(recorder.raw) == len(uids) and len(recorder.forwards) == 2 * len(uids)
    assert all(
        event["input_shape"] == [1, 94, 3, 320, 320] and not event["grad_enabled"] for event in recorder.forwards
    )
    assert len(state["base_raw_probabilities"]) == 1
    assert len(recorder.loads) == 2 and all(
        receipt["actual_checkpoint_binding"]
        and receipt["actual_checkpoint_binding"]["SHA_verified"]
        and receipt["actual_checkpoint_binding"]["actual_sha256"]
        == BASE_CHECKPOINT_SHA256_BY_NAME[receipt["actual_checkpoint_binding"]["filename"]]
        for receipt in recorder.loads
    )
    assert state["base_raw_probabilities"][0] == [recorder.raw[uid]["raw_probabilities"] for uid in uids]
    assert state["shared_inputs"] == recorder.inputs
    assert state["worker_initializers"] and all(
        item["source_sha256"] == binding["source_sha256"] and item["CROP_MM"] == 140.0
        for item in state["worker_initializers"]
    )
    if dispatch == "fixed":
        assert all(recorder.raw[uid]["device"] == assignment[uid] for uid in uids)
    enforce_phase_resource_contract(report)
    write_json_new(
        root / f"P010_AA_{phase}_GUARD.json",
        {
            "phase": phase,
            "internal_execution_contract_passed": True,
            "phase_receipt_sha256": sha(root / f"P010_AA_{phase}.json"),
            "candidate_gate_passed": False,
        },
    )
    return report


def stock_phase(source, phase, uids, labels, *, dispatch):
    import multiprocessing as mp

    import pandas as pd
    import torch
    from p006_metrics import Resources
    from p009_metrics import capture_stdout, embedded_source, prepare_fastread, registered_namespace
    from p010_author224_metrics import CHECKPOINT_FILENAME, infer_expert224, load_expert224, make_stock_namespace
    from p010_previous_metrics import observed_source
    from p010_shared_metrics import run_bounded_pipeline, shared_input_receipt

    root = Path("/kaggle/working")
    binding = prepare_fastread(source, {hashlib.sha256(embedded_source(source)).hexdigest()})
    recorder = PhaseRecorder(root, phase, uids, labels)
    stock = make_stock_namespace((root / "p010_src/p010_author224_stock.py").read_text())
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    assignment = {uid: devices[index % len(devices)] for index, uid in enumerate(uids)}
    pipeline = fixed_pipeline(run_bounded_pipeline, assignment) if dispatch == "fixed" else run_bounded_pipeline
    workers = root / f"p010_workers_{phase}"
    workers.mkdir(exist_ok=False)
    definitions = observed_source(source, phase)
    if not definitions.endswith("main()\n"):
        raise ValueError("Original source terminator changed")
    definitions = definitions[: -len("main()\n")]
    values = {
        "__file__": str(root / (phase + ".py")),
        "_p010_worker_directory": str(workers),
        "_p010_expected_fastread_sha": binding["source_sha256"],
    }
    models, loads, inferences = {}, {}, {}
    state, error, resources = None, None, Resources()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    try:
        with registered_namespace("_p010_AA_" + phase, values) as namespace:
            exec(compile(definitions, "<p010-AA-stock-definitions>", "exec"), namespace)
            series = pd.read_csv(
                root / "p010_profile_data/test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str}
            )
            namespace["_SER"] = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}
            namespace["_TSDIR"] = str(root / "p010_profile_data/test_series")
            with resources, capture_stdout() as captured:
                for device in devices:
                    models[device], loads[device] = load_expert224(
                        namespace["find_weight_file"](CHECKPOINT_FILENAME), device, stock_namespace=stock
                    )
                    recorder.load(models[device], 224, checkpoint_receipt=loads[device])

                def consume(device, item):
                    uid, volume, mask = item
                    before = shared_input_receipt(volume, mask)
                    recorder.begin(uid, volume, mask, device, before)
                    raw, inference = infer_expert224(stock, models[device], volume, mask, device)
                    inferences[uid] = inference
                    recorder.complete(raw.tolist(), device, inference=inference)
                    assert before == shared_input_receipt(volume, mask)

                nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
                transport = pipeline(
                    lambda: mp.get_context("fork").Pool(nproc, initializer=namespace["_pre_init"]),
                    namespace["_pre_one"],
                    uids,
                    devices,
                    consume,
                    queue_max=48,
                    outstanding_limit=min(2 * nproc, 48),
                )
            state = {
                "expert_loads": loads,
                "expert_inferences": inferences,
                "transport": transport,
                "worker_initializers": [json.loads(path.read_text()) for path in sorted(workers.glob("*.json"))],
                "task_pickle_identity": namespace["_p010_task_identity"],
                "dispatch_policy": dispatch,
                "fixed_assignment": assignment if dispatch == "fixed" else None,
                "same_fastread_source_sha256": binding["source_sha256"],
                "stock_numerical_source_literal": True,
            }
    except BaseException as exception:
        error = exception
        state = {
            "expert_loads": loads,
            "expert_inferences": inferences,
            "transport_failure_receipt": getattr(exception, "receipt", None),
            "dispatch_policy": dispatch,
        }
        raise
    finally:
        report = recorder.finish(
            state=state, exception=error, resources=resource_result_or_unentered(resources), peaks=torch_peaks()
        )
        models.clear()
        (root / "submission.csv").unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
    assert "FAILED (" not in captured.getvalue()
    assert len(recorder.raw) == len(uids)
    assert all(
        receipt["production_qualified"]
        and receipt["stock_forward_calls"] == 1
        and receipt["backbone_batch_windows"] == 94
        and receipt["views"] == 1
        and receipt["cuda_fp16_observed"]
        and receipt["no_grad"]
        and receipt["fallback_calls"] == 0
        for receipt in inferences.values()
    )
    assert state["worker_initializers"] and all(
        item["source_sha256"] == binding["source_sha256"] and item["CROP_MM"] == 140.0
        for item in state["worker_initializers"]
    )
    if dispatch == "fixed":
        assert all(recorder.raw[uid]["device"] == assignment[uid] for uid in uids)
    enforce_phase_resource_contract(report)
    write_json_new(
        root / f"P010_AA_{phase}_GUARD.json",
        {
            "phase": phase,
            "internal_execution_contract_passed": True,
            "phase_receipt_sha256": sha(root / f"P010_AA_{phase}.json"),
            "candidate_gate_passed": False,
        },
    )
    return report


def raw_measurement(left, right, uids):
    from p010_shared_metrics import average_percentile

    a = [left["raw"][uid]["raw_probabilities"] for uid in uids]
    b = [right["raw"][uid]["raw_probabilities"] for uid in uids]
    differences, ulps = [], []
    changed_cells = 0
    for row1, row2 in zip(a, b, strict=True):
        for x, y in zip(row1, row2, strict=True):
            differences.append(abs(x - y))
            bits_x = 0 if x == 0 else struct.unpack("<I", struct.pack("<f", x))[0]
            bits_y = 0 if y == 0 else struct.unpack("<I", struct.pack("<f", y))[0]
            ulps.append(abs(bits_x - bits_y))
            changed_cells += x != y
    columns = {}
    for column in range(12):
        first, second = [row[column] for row in a], [row[column] for row in b]
        first_rank, second_rank = average_percentile(first), average_percentile(second)
        flips = 0
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                flips += ((first[i] > first[j]) - (first[i] < first[j])) != (
                    (second[i] > second[j]) - (second[i] < second[j])
                )
        columns[left["labels"][column]] = {
            "changed_rank_cells": sum(x != y for x, y in zip(first_rank, second_rank, strict=True)),
            "pair_order_or_tie_relation_changes": flips,
            "maximum_absolute_difference": max(abs(x - y) for x, y in zip(first, second, strict=True)),
        }
    ordered = sorted(ulps)
    maximum = max(differences)
    return {
        "left_phase": left["phase"],
        "right_phase": right["phase"],
        "maximum_absolute_raw_difference": maximum,
        "changed_raw_cells": changed_cells,
        "float32_ULP": {
            "maximum": max(ulps),
            "median": (
                ordered[len(ordered) // 2]
                if len(ordered) % 2
                else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
            ),
            "p99_nearest_rank": ordered[math.ceil(0.99 * len(ordered)) - 1],
        },
        "by_label": columns,
        "same_observed_device_studies": sum(left["raw"][uid]["device"] == right["raw"][uid]["device"] for uid in uids),
        "existing_raw_absolute_tolerance": RAW_ABSOLUTE_TOLERANCE,
        "existing_raw_relative_tolerance": RAW_RELATIVE_TOLERANCE,
        "existing_raw_gate_would_pass": maximum <= RAW_ABSOLUTE_TOLERANCE,
        "diagnostic_measurement_not_gate_relaxation": True,
    }


def paired_shared224(namespace, original, bounded, shared, fusion, reader, uids, labels):
    """Compatibility call shape; V4 measures repetitions, never makes a candidate."""
    import torch
    from p009_metrics import fork_transport_probe
    from p010_previous_metrics import reader_helper_preflight, reader_replay_preflight

    del namespace, shared, fusion, reader
    root = Path("/kaggle/working")
    if len(uids) != 32 or len(set(uids)) != 32:
        raise ValueError("Only the preregistered fixed32 diagnostic is allowed")
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(devices) != 2 or not all("t4" in name.lower() for name in devices):
        raise ValueError("The actual diagnostic requires T4x2")
    directory = Path(__file__).resolve().parent
    helper_sha = {name: sha(directory / name) for name in ("p006_metrics.py", "p007_metrics.py")}
    reports = {}
    phase = None
    try:
        preflight = {
            "actual_child_import": reader_helper_preflight(directory, helper_sha),
            "artificial_exact_reader_replay": reader_replay_preflight(labels),
            "artificial_fork_transport": fork_transport_probe(),
        }
        write_json_new(root / "P010_AA_PREFLIGHT.json", preflight)
        for phase, family, dispatch in PHASES:
            if family == "base":
                reports[phase] = base_phase(
                    original if dispatch == "original" else bounded, phase, uids, labels, dispatch=dispatch
                )
            else:
                reports[phase] = stock_phase(original, phase, uids, labels, dispatch=dispatch)
            print(f"p010 A/A phase persisted: {phase}", flush=True)
        expected_inputs = reports[PHASES[0][0]]["input_receipts"]
        input_matches = {phase: report["input_receipts"] == expected_inputs for phase, report in reports.items()}
        comparisons = {}
        for index in range(0, len(PHASES), 2):
            first, second = PHASES[index][0], PHASES[index + 1][0]
            comparisons[first.removesuffix("_A1")] = raw_measurement(reports[first], reports[second], uids)
        comparisons["original_vs_bounded_A1"] = raw_measurement(
            reports["base_original_ordinary_A1"], reports["base_bounded_ordinary_A1"], uids
        )
        result = {
            "phase_reports": {
                phase: {"file": f"P010_AA_{phase}.json", "sha256": sha(root / f"P010_AA_{phase}.json")}
                for phase in reports
            },
            "comparisons": comparisons,
            "same_input_hashes_all_phases": input_matches,
            "phase_count": len(reports),
            "model_reader_actual_runs": 0,
            "reader_fusion_actual_runs": 0,
            "candidate_or_fixed10_output_created": False,
            "candidate_gate_passed": False,
            "existing_raw_absolute_tolerance": RAW_ABSOLUTE_TOLERANCE,
            "existing_raw_relative_tolerance": RAW_RELATIVE_TOLERANCE,
            "existing_crosscontrol_CSV_byte_identity_required": True,
            "actual_environment_devices": devices,
            "diagnostic_only": True,
        }
        write_json_new(root / "P010_AA_COMPARISONS.json", result)
        assert all(input_matches.values()), "Different native pixels or masks invalidate numerical A/A attribution"
        return result
    except BaseException as exception:
        write_json_new(
            root / "P010_AA_ABORT.json",
            {
                "exception": {"type": type(exception).__name__, "message": str(exception)},
                "aborted_phase_name": phase,
                "completed_phase_names": list(reports),
                "completed_phases": len(reports),
                "previous_phase_receipts_preserved": True,
                "candidate_gate_passed": False,
            },
        )
        raise
    finally:
        (root / "submission.csv").unlink(missing_ok=True)
        write_json_new(
            root / "P010_AA_SUBMISSION_GUARD.json",
            {
                "final_submission_csv_absent": not (root / "submission.csv").exists(),
                "diagnostic_outputs_must_never_be_submitted": True,
                "candidate_gate_passed": False,
            },
        )
