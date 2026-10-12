"""p011: seal the unchanged baseline, then execute a new standalone224 branch.

No model namespace is created until the original final CSV has been sealed.
The extra decoder uses original14 functions; its bounded transport never
replaces the baseline's transport. No remote action or data acquisition.
"""

import atexit
import base64
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import types
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

from p011_contracts import (
    FusionObserver,
    PipelineFailure,
    average_percentile,
    fixed_rank10,
    read_submission,
    run_bounded_pipeline,
    shared_input_receipt,
    write_submission_exclusive,
)

FASTREAD_SHA = "5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0"
BASE_CHECKPOINT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
ORIGINAL_READER_SHA = "667568b4e06f0c54937ad1e4d7059ffeff44e68592d59d10580e13ec337a5fa5"
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
_PROFILE = None
_EXTRA = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def cpu_csv_seal(path, *, working):
    """Read immutable CPU predictions before the extra namespace/model exists."""
    root, path = Path(working).resolve(), Path(path).resolve()
    if path.parent != root or path.name != "submission.csv":
        raise ValueError("Seal the exact original working submission.csv")
    uids, matrix = read_submission(path, LABELS)
    backup = root / "p011_baseline_sealed.csv"
    payload = path.read_bytes()
    with backup.open("xb") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if sha(path) != digest or sha(backup) != digest:
        raise ValueError("Baseline bytes changed while sealing")
    receipt = {
        "schema_version": "p011_baseline_CPU_seal_v1",
        "studies": len(uids),
        "labels": LABELS,
        "baseline_sha256": digest,
        "backup": backup.name,
        "extra_model_or_namespace_created": False,
        "baseline_original_final_fusion": True,
    }
    write_new(root / "P011_BASELINE_SEAL.json", receipt)
    return {
        "uids": uids,
        "matrix": matrix,
        "sha256": digest,
        "backup": backup,
        "path": path,
        "working": root,
        "receipt": receipt,
    }


def assert_sealed(seal):
    if sha(seal["path"]) != seal["sha256"] or sha(seal["backup"]) != seal["sha256"]:
        raise ValueError("Finalized baseline CSV changed after sealing")


def fixed10_from_seal(seal, expert_path):
    assert_sealed(seal)
    _, expert = read_submission(expert_path, LABELS, seal["uids"])
    result = fixed_rank10(seal["matrix"], expert)
    stage = seal["working"] / "p011_candidate.partial.csv"
    write_submission_exclusive(stage, seal["uids"], result, LABELS)
    assert_sealed(seal)
    candidate_sha = sha(stage)
    receipt = {
        "schema_version": "p011_new_model_fixed_rank10_v1",
        "studies": len(seal["uids"]),
        "labels": LABELS,
        "coefficients_predeclared": [0.9, 0.1],
        "ties": "average",
        "normalisation": "rank1_to_N_divided_by_N",
        "baseline_sha256_before_and_after_extra_branch": seal["sha256"],
        "expert_raw_csv_sha256": sha(expert_path),
        "candidate_sha256": candidate_sha,
        "old_p010_gate_relaxed_or_requalified": False,
        "accuracy_evaluated": False,
    }
    write_new(seal["working"] / "P011_CANDIDATE_BLEND_RECEIPT.json", receipt)
    os.replace(stage, seal["path"])
    if sha(seal["path"]) != candidate_sha or sha(seal["backup"]) != seal["sha256"]:
        raise ValueError("Final candidate or immutable baseline backup changed")
    return receipt


def gpu_resources(torch):
    return {
        "torch": str(torch.__version__),
        "CUDA": torch.version.cuda,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "GPU": [
            {
                "index": d,
                "name": torch.cuda.get_device_name(d),
                "allocated_mib": torch.cuda.memory_allocated(d) / 2**20,
                "reserved_mib": torch.cuda.memory_reserved(d) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(d) / 2**20,
            }
            for d in range(torch.cuda.device_count())
        ],
    }


def bind_decoder(namespace):
    decoded = base64.b64decode(namespace["_FASTREAD_B64"])
    if hashlib.sha256(decoded).hexdigest() != FASTREAD_SHA:
        raise ValueError("Original14 embedded native384/FOV140 decoder changed")
    path = Path("/kaggle/working/_fastread.py")
    if not path.is_file() or sha(path) != FASTREAD_SHA:
        raise ValueError("Actual initialized baseline fastread file differs")
    if namespace["IMG"] != 384 or namespace["MAXS"] != 96 or namespace["CROP_MM"] != 140:
        raise ValueError("Original14 preprocessing constants differ")


class ObservedBuilder:
    """Pass through native bytes and record receipts; no image serialization."""

    def __init__(self, builder, directory):
        self.builder, self.directory = builder, Path(directory)

    def __call__(self, uid, series, tree):
        destination = self.directory / (hashlib.sha256(uid.encode()).hexdigest() + ".json")
        try:
            volume, mask = self.builder(uid, series, tree)
        except Exception as error:
            write_new(
                destination,
                {
                    "uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                    "decoder_error_type": type(error).__name__,
                    "native_input_available": False,
                },
            )
            raise  # Original14 _pre_one retains its explicit masked-zero decoder handling.
        write_new(
            destination,
            {
                "uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                "native_input_available": True,
                **shared_input_receipt(volume, mask),
            },
        )
        return volume, mask


def init_observed_builder(namespace, original_init, workers, inputs):
    original_init()
    builder = namespace["_READER"]
    scope = dict(zip(builder.__code__.co_freevars, [cell.cell_contents for cell in builder.__closure__], strict=True))
    if builder.__globals__["CROP_MM"] != 140 or scope["IMG"] != 384 or scope["SLOTS"] != namespace["SLOTS"]:
        raise ValueError("Actual worker decoder FOV/native/slots differ")
    actual_source = sha(builder.__globals__["__file__"])
    if actual_source != FASTREAD_SHA:
        raise ValueError("Actual worker decoder source differs")
    write_new(
        Path(workers) / (str(os.getpid()) + ".json"),
        {"pid": os.getpid(), "FOV_mm": 140, "IMG": 384, "SLOTS": scope["SLOTS"], "source_sha256": actual_source},
    )
    namespace["_READER"] = ObservedBuilder(builder, inputs)


def validate_native_receipt(directory, uid, inputs):
    """Require the worker's exclusive receipt before any extra model call."""
    key = hashlib.sha256(uid.encode()).hexdigest()
    path = Path(directory) / (key + ".json")
    try:
        if path.stat().st_size > 4096:
            raise ValueError("Oversized native worker receipt")
        payload = path.read_bytes()
        record = json.loads(payload)
    except (OSError, ValueError) as error:
        raise ValueError("Missing, corrupt or oversized native worker receipt") from error
    if not isinstance(record, dict) or record.get("uid_sha256") != key:
        raise ValueError("Native worker receipt study identity differs")
    available = record.get("native_input_available")
    if available is True:
        if set(record) != {"uid_sha256", "native_input_available", *inputs} or record != {
            "uid_sha256": key,
            "native_input_available": True,
            **inputs,
        }:
            raise ValueError("Native worker successful receipt bytes/shape/mask contract differs")
    elif available is False:
        if (
            set(record) != {"uid_sha256", "native_input_available", "decoder_error_type"}
            or not isinstance(record["decoder_error_type"], str)
            or not record["decoder_error_type"]
            or inputs["valid_slices"] != 0
            or inputs["mask_sha256"] != hashlib.sha256(bytes(96)).hexdigest()
            or inputs["volume_sha256"] != hashlib.sha256(bytes(96 * 384 * 384)).hexdigest()
        ):
            raise ValueError("Recorded decoder failure must match original zero volume and mask")
    else:
        raise ValueError("Native worker receipt requires an explicit boolean status")
    return {
        "worker_receipt_verified_before_model": True,
        "native_worker_receipt_sha256": hashlib.sha256(payload).hexdigest(),
    }


def profile_pre_init():
    namespace = _PROFILE.namespace
    try:
        namespace["_TSDIR"] = str(_PROFILE.data_root / "test_series")
        init_observed_builder(namespace, _PROFILE.original_init, _PROFILE.base_workers, _PROFILE.base_inputs)
    except Exception as error:
        # Return normally so Pool cannot endlessly respawn failed initializers.
        # Child stdout is not the parent's FusionObserver buffer: the shared
        # flag, checked before original GPU inference, is the fatal contract.
        _PROFILE.initializer_failed.value = 1
        namespace["_READER"] = FailedInitializerBuilder()
        try:
            write_new(
                _PROFILE.base_workers / (str(os.getpid()) + "-FAILED.json"),
                {"initializer_failed": True, "error_type": type(error).__name__},
            )
        except OSError:
            pass  # Shared fatal flag still rejects the profile.


def extra_pre_init():
    namespace = _EXTRA["namespace"]
    try:
        if _EXTRA["data_root"] is not None:
            namespace["_TSDIR"] = str(_EXTRA["data_root"] / "test_series")
        init_observed_builder(namespace, _EXTRA["original_init"], _EXTRA["workers"], _EXTRA["inputs"])
    except Exception as error:
        # Pool otherwise respawns a failing initializer forever without a task
        # callback. Shared flag is set BEFORE a task can reach the parent GPU.
        _EXTRA["initializer_failed"].value = 1
        namespace["_READER"] = FailedInitializerBuilder()
        try:
            write_new(
                Path(_EXTRA["workers"]) / (str(os.getpid()) + "-FAILED.json"),
                {"initializer_failed": True, "error_type": type(error).__name__},
            )
        except OSError:
            pass  # Shared fatal flag still prevents model inference.


class FailedInitializerBuilder:
    def __call__(self, *args):
        raise PipelineFailure("Native decoder initializer failed")


class BaselineProfile:
    """Profile-only execution alias; ALL original code cells remain literal.

    Temporarily alternate __name__ prevents automatic main invocation in its
    unchanged final if-block. A registered module provides ForkingPickler
    identity, then the original main executes once with a working data alias.
    """

    def __init__(self, namespace, data_root, working):
        global _PROFILE
        if _PROFILE is not None:
            raise ValueError("Exactly one profile context allowed")
        self.namespace, self.data_root, self.working = namespace, Path(data_root), Path(working)
        self.original_name, self.name = namespace["__name__"], "_p011_original14_profile"
        if self.name in sys.modules:
            raise ValueError("Profile module collision")
        self.module = types.ModuleType(self.name)
        sys.modules[self.name] = self.module
        namespace["__name__"] = self.name
        self.base_workers, self.base_inputs = self.working / "p011_workers_base", self.working / "p011_inputs_base"
        for path in (self.base_workers, self.base_inputs):
            path.mkdir(exist_ok=False)
        self.state = {"main_runs": 0, "pair_calls": 0, "forwards": [], "loads": [], "reader_runs": 0}
        self.initializer_failed = mp.Value("i", 0)
        self.original_subprocess_run = subprocess.run
        _PROFILE = self
        atexit.register(self.delete_diagnostic_submission)

    def execute_main(self):
        import torch
        from p006_metrics import Resources

        ns = self.namespace
        self.module.__dict__.update(ns)
        self.original_init, root = ns["_pre_init"], ns["find_test_root"]
        load, pair = ns["load_model"], ns["infer_pair_probs"]
        handles = []
        count_lock = threading.Lock()

        def observed_load(*args, **kwargs):
            if sha(args[0]) != BASE_CHECKPOINT_SHA:
                raise ValueError("Actual base checkpoint differs")
            model, resolution = load(*args, **kwargs)
            self.state["loads"].append(
                {"device": str(args[3]), "checkpoint_sha256": sha(args[0]), "resolution": resolution}
            )

            def forward(_model, values, result):
                self.state["forwards"].append(
                    {
                        "input_shape": list(values[0].shape),
                        "output_shape": list(result.shape),
                        "grad_enabled": bool(torch.is_grad_enabled()),
                        "autocast": bool(torch.is_autocast_enabled("cuda")),
                    }
                )

            handles.append(model.register_forward_hook(forward))
            return model, resolution

        def observed_pair(*args, **kwargs):
            if self.initializer_failed.value:
                raise PipelineFailure("Profile preprocessing initializer failed; original GPU pair rejected")
            raw = pair(*args, **kwargs)
            with count_lock:
                self.state["pair_calls"] += 1
            return raw

        ns["_pre_init"], ns["find_test_root"] = profile_pre_init, lambda: str(self.data_root)
        ns["load_model"], ns["infer_pair_probs"] = observed_load, observed_pair
        self.module.__dict__.update(ns)
        assert __import__("pickle").loads(ForkingPickler.dumps(ns["_pre_one"])) is ns["_pre_one"]
        assert __import__("pickle").loads(ForkingPickler.dumps(profile_pre_init)) is profile_pre_init
        observer = FusionObserver()
        try:
            self.state["main_runs"] += 1
            with Resources() as resources:
                ns["main"]()
            self.state["resources"] = resources.result()
        finally:
            log = observer.restore()
            self.state["preprocessing_initializer_failed"] = bool(self.initializer_failed.value)
            self.state["FAILED_or_FALLBACK_in_main"] = "FAILED (" in log or "FALLBACK (" in log
            ns["_pre_init"], ns["find_test_root"], ns["load_model"], ns["infer_pair_probs"] = (
                self.original_init,
                root,
                load,
                pair,
            )
            for handle in handles:
                handle.remove()
            ns["__name__"] = self.original_name
            self.module.__dict__.update(ns)
            write_new(self.working / "P011_BASE_PROFILE.json", self.state)
        if self.state["FAILED_or_FALLBACK_in_main"] or self.state["preprocessing_initializer_failed"]:
            raise PipelineFailure("Original baseline preprocessing/model fallback occurred in profile")

    def prepare_reader(self):
        from p007_metrics import observed_reader_source

        path = self.working / "own_src/infer.py"
        source = path.read_text(encoding="utf-8")
        self.state["original_reader_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if self.state["original_reader_sha256"] != ORIGINAL_READER_SHA:
            raise ValueError("Actual original14 reader source differs before observations")
        prefix = "import sys as _p011_reader_sys\n_p011_reader_sys.path.insert(0, '/kaggle/working/p011_src')\n"
        observed = prefix + observed_reader_source(source, "plain")
        path.write_text(observed, encoding="utf-8")
        self.state["observed_reader_sha256"] = sha(path)

        def routed_run(args, *positional, **keywords):
            if isinstance(args, (list, tuple)) and len(args) > 1 and args[1] == "/kaggle/working/own_src/infer.py":
                if (
                    positional
                    or len(args) != 8
                    or args[0] != sys.executable
                    or args[2]
                    not in (
                        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection",
                    )
                    or args[3] != "test"
                    or args[4] != "/kaggle/working/_own.csv"
                    or keywords.get("check") is not True
                    or [Path(x).name for x in args[5:]] != [f"cnxt_v0_fold{i}.pt" for i in range(3)]
                ):
                    raise ValueError("Unexpected original reader command")
                if self.state["reader_runs"] != 0:
                    raise ValueError("Repeated reader execution")
                self.state["reader_runs"] += 1
                args = [*args[:2], str(self.data_root), *args[3:]]
                from p006_metrics import Resources

                with Resources() as resources:
                    result = self.original_subprocess_run(args, *positional, **keywords)
                self.state["reader_resources"] = resources.result()
                return result
            return self.original_subprocess_run(args, *positional, **keywords)

        subprocess.run = routed_run

    def restore_reader(self):
        subprocess.run = self.original_subprocess_run

    def delete_diagnostic_submission(self):
        path = self.working / "submission.csv"
        if path.parent.resolve() == self.working.resolve():
            path.unlink(missing_ok=True)

    def close(self):
        global _PROFILE
        self.restore_reader()
        self.namespace["__name__"] = self.original_name
        if sys.modules.get(self.name) is not self.module:
            raise ValueError("Profile namespace identity changed")
        del sys.modules[self.name]
        _PROFILE = None


def run_standalone224(namespace, seal, stock_source, *, data_root=None):
    """Second decode, independent author224 numerics; all unexpected errors fatal."""
    global _EXTRA
    import multiprocessing as mp

    import torch
    from p006_metrics import Resources
    from p010_author224_metrics import CHECKPOINT_FILENAME, infer_expert224, load_expert224, make_stock_namespace

    assert_sealed(seal)
    bind_decoder(namespace)
    if __import__("pickle").loads(ForkingPickler.dumps(namespace["_pre_one"])) is not namespace["_pre_one"]:
        raise ValueError("Original extra preprocessing function module identity differs")
    if namespace["LAB"] != LABELS or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError("p011 requires original12 labels and two CUDA devices")
    if "fork" not in mp.get_all_start_methods():
        raise ValueError("Original native decoder requires Linux fork")
    root = seal["working"]
    input_directory, workers, receipts = (
        root / "p011_inputs_extra",
        root / "p011_workers_extra",
        root / "p011_inference_extra",
    )
    for path in (input_directory, workers, receipts):
        path.mkdir(exist_ok=False)
    gc.collect()
    torch.cuda.empty_cache()
    before = gpu_resources(torch)
    write_new(
        root / "P011_BEFORE_EXTRA.json",
        {
            "baseline_sealed_sha256": seal["sha256"],
            "resources": before,
            "main_returned_and_reader_subprocess_completed": True,
        },
    )
    # Independent namespace creation occurs here, after the exact CPU seal.
    stock = make_stock_namespace(stock_source)
    models, loads, raw = {}, {}, {}
    lock = threading.Lock()
    _EXTRA = {
        "namespace": namespace,
        "original_init": namespace["_pre_init"],
        "data_root": Path(data_root) if data_root is not None else None,
        "workers": workers,
        "inputs": input_directory,
        "initializer_failed": mp.get_context("fork").Value("i", 0),
    }
    started = time.monotonic()
    try:
        for device in range(2):
            torch.cuda.reset_peak_memory_stats(device)
        with Resources() as resources:
            checkpoint = namespace["find_weight_file"](CHECKPOINT_FILENAME)
            for device in ("cuda:0", "cuda:1"):
                models[device], loads[device] = load_expert224(checkpoint, device, stock_namespace=stock)
                write_new(root / ("P011_EXPERT_LOAD_" + device.replace(":", "_") + ".json"), loads[device])

            def consume(device, item):
                if _EXTRA["initializer_failed"].value:
                    raise PipelineFailure(
                        "Extra preprocessing initializer failed; no masked-zero model substitute accepted"
                    )
                uid, volume, mask = item
                if uid not in seal["uids"]:
                    raise ValueError("Foreign UID from extra decoder")
                inputs = shared_input_receipt(volume, mask)
                worker_guard = validate_native_receipt(input_directory, uid, inputs)
                values, inference = infer_expert224(stock, models[device], volume, mask, device)
                if shared_input_receipt(volume, mask) != inputs:
                    raise ValueError("Standalone expert mutated native input")
                record = {
                    "uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                    "native_input": inputs,
                    "inference": inference,
                    "raw_probabilities": values.tolist(),
                    "device": device,
                    **worker_guard,
                }
                write_new(receipts / (record["uid_sha256"] + ".json"), record)
                with lock:
                    if uid in raw:
                        raise ValueError("Duplicate extra prediction")
                    raw[uid] = values.tolist()

            nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
            transport = run_bounded_pipeline(
                lambda: mp.get_context("fork").Pool(nproc, initializer=extra_pre_init),
                namespace["_pre_one"],
                seal["uids"],
                list(models),
                consume,
                queue_max=48,
                outstanding_limit=min(2 * nproc, 48),
            )
        if set(raw) != set(seal["uids"]):
            raise ValueError("Incomplete extra predictions")
        destination = root / "p011_expert224_raw.csv"
        write_submission_exclusive(destination, seal["uids"], [raw[uid] for uid in seal["uids"]], LABELS)
        assert_sealed(seal)
        per_study = {p.stem: json.loads(p.read_text()) for p in receipts.glob("*.json")}
        keys = {hashlib.sha256(uid.encode()).hexdigest() for uid in seal["uids"]}
        if set(per_study) != keys:
            raise ValueError("Standalone inference receipt coverage differs")
        report = {
            "schema_version": "p011_standalone224_new_model_branch_v1",
            "studies": len(raw),
            "labels": LABELS,
            "expert_loads": loads,
            "transport": transport,
            "resources": resources.result(),
            "GPU_resources": gpu_resources(torch),
            "elapsed_seconds": time.monotonic() - started,
            "per_study_receipts": per_study,
            "extra_native_decode_passes": 1,
            "baseline_transport_modified": False,
            "baseline_CSV_byteexact_before_after": True,
            "allmissing_policy": "Keep original stock finite masked-input prediction; no base-only perUID fallback.",
            "decoder_error_policy": "Original _pre_one records decoding exception then zeros volume/mask; extra model must still return finite12 stock outputs.",
            "CUDA_model_or_contract_error_policy": "Fatal with prior per-study/load receipts preserved; no generic catch or silent baseline restore.",
            "allmissing_UID_count": sum(
                json.loads(p.read_text())["inference"]["allmissing"] for p in receipts.glob("*.json")
            ),
            "decoder_error_UID_count": sum(
                "decoder_error_type" in json.loads(p.read_text()) for p in input_directory.glob("*.json")
            ),
            "preprocessing_initializer_failed": bool(_EXTRA["initializer_failed"].value),
            "candidate_gate_passed": False,
            "accuracy_evaluated": False,
            "old_p010_gates_relaxed_or_requalified": False,
            "transport_liveness_limit": "OS-killed task workers with missing callbacks are not covered by a new per-task watchdog; no hidden completion guarantee.",
        }
        write_new(root / "P011_STANDALONE224_SUMMARY.json", report)
        return destination, report
    except BaseException as error:
        write_new(
            root / "P011_EXTRA_ABORT.json",
            {
                "error_type": type(error).__name__,
                "completed_UID_count": len(raw),
                "candidate_gate_passed": False,
                "unexpected_errors_fatal": True,
            },
        )
        raise
    finally:
        models.clear()
        stock.clear()
        _EXTRA = None
        gc.collect()
        torch.cuda.empty_cache()


def strict_profile_resources(phases, GPU_resources):
    """Reject missing/invalid observations; no memory threshold is inferred."""

    def positive(value, cap=None):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            and (cap is None or value < cap)
        )

    if len(phases) != 3:
        raise ValueError("Three main/reader/extra resource observations required")
    for phase in phases:
        values = phase["sampled_gpu_peak_used_mib"]
        if len(values) != 2 or not all(positive(value, 13000) for value in values):
            raise ValueError("Actual finite positive twoGPU sampled peaks below13000MiB required")
        if (
            not positive(phase["elapsed_seconds"])
            or not positive(phase["sampled_peak_process_tree_rss_bytes"])
            or not isinstance(phase["samples"], int)
            or isinstance(phase["samples"], bool)
            or phase["samples"] < 1
        ):
            raise ValueError("Finite positive elapsed/RSS and actual sampling count required")
    devices = GPU_resources["GPU"]
    if len(devices) != 2 or [entry["index"] for entry in devices] != [0, 1]:
        raise ValueError("Exactly two indexed CUDA device resource observations required")
    for entry in devices:
        if entry["name"] not in ("Tesla T4", "NVIDIA Tesla T4") or not positive(entry["peak_reserved_mib"], 13000):
            raise ValueError("Actual twoT4 finite positive torch reserved peaks below13000MiB required")
        for name in ("allocated_mib", "reserved_mib"):
            value = entry[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or value >= 13000
            ):
                raise ValueError("Actual finite nonnegative torch current memory observations required")
        if not entry["allocated_mib"] <= entry["reserved_mib"] <= entry["peak_reserved_mib"]:
            raise ValueError("Torch allocated/reserved/peak ordering differs")
    return {"actual_T4_devices": 2, "phases_sampled": 3, "positive_finite_peaks": True, "cap_mib_exclusive": 13000}


def finish_profile(profile, seal, report, blend, fusion_guard):
    from p007_metrics import validate_observations

    root, uids = profile.working, seal["uids"]
    if profile.state["main_runs"] != 1 or profile.state["reader_runs"] != 1 or profile.state["pair_calls"] != len(uids):
        raise ValueError("Original main/reader execution counts differ")
    events = profile.state["forwards"]
    if len(events) != 2 * len(uids) or any(
        e["input_shape"] != [1, 94, 3, 320, 320]
        or e["output_shape"] != [1, 12]
        or e["grad_enabled"]
        or not e["autocast"]
        for e in events
    ):
        raise ValueError("Original main forward contracts differ")
    native_base = {p.stem: json.loads(p.read_text()) for p in profile.base_inputs.glob("*.json")}
    native_extra = {p.stem: json.loads(p.read_text()) for p in (root / "p011_inputs_extra").glob("*.json")}
    keys = {hashlib.sha256(uid.encode()).hexdigest() for uid in uids}
    if (
        set(native_base) != keys
        or native_base != native_extra
        or any(not x["native_input_available"] or x["valid_slices"] == 0 for x in native_base.values())
    ):
        raise ValueError("Two independent native decode passes differ or profile decode unusable")
    inference = report["per_study_receipts"]
    if set(inference) != keys or any(
        record["uid_sha256"] != key
        or record["worker_receipt_verified_before_model"] is not True
        or record["native_input"]
        != {k: v for k, v in native_extra[key].items() if k not in ("uid_sha256", "native_input_available")}
        or record["inference"]["status"] != "passed"
        or not record["inference"]["production_qualified"]
        or record["inference"]["stock_forward_calls"] != 1
        or record["inference"]["backbone_forward_calls"] != 1
        or record["inference"]["backbone_batch_windows"] != 94
        or record["inference"]["fallback_calls"] != 0
        or not record["inference"]["no_grad"]
        or not record["inference"]["cuda_fp16_observed"]
        or record["inference"]["labels"] != LABELS
        for key, record in inference.items()
    ):
        raise ValueError("Actual standalone stock inference receipt contracts differ")
    reader = json.loads((root / "p007_observation_plain/inference.json").read_text())
    reader_gate = validate_observations(reader, "plain", len(uids))
    _, main_values = read_submission(root / "_sota_0949.csv", LABELS, uids)
    _, reader_values = read_submission(root / "_own.csv", LABELS, uids)
    _, original_fused = read_submission(seal["backup"], LABELS, uids)
    fusion_delta = 0.0
    for column, label in enumerate(LABELS):
        left = average_percentile([row[column] for row in main_values])
        right = average_percentile([row[column] for row in reader_values])
        weight = profile.namespace["TARGET_WEIGHTS"][label]
        expected = average_percentile([(1.0 - weight) * a + weight * b for a, b in zip(left, right, strict=True)])
        fusion_delta = max(
            fusion_delta, max(abs(row[column] - value) for row, value in zip(original_fused, expected, strict=True))
        )
    if fusion_delta != 0:
        raise ValueError("Original fusion differs from independent fixed formula")
    resource_gate = strict_profile_resources(
        [profile.state["resources"], profile.state["reader_resources"], report["resources"]], report["GPU_resources"]
    )
    profile.delete_diagnostic_submission()
    result = {
        "schema_version": "p011_independent224_profile_v1",
        "status": "passed",
        "studies": len(uids),
        "ordered_uid_sha256": hashlib.sha256("\n".join(uids).encode()).hexdigest(),
        "main_runs": 1,
        "reader_runs": 1,
        "expert_runs": 1,
        "main_forward_calls": len(events),
        "two_decode_pixel_and_mask_SHA_exact": True,
        "original_fusion_stdlib_maxdiff": fusion_delta,
        "original_fusion_guard": fusion_guard,
        "native_decode_receipts": native_base,
        "expert_stock_forward_calls": len(inference),
        "expert_backbone_forward_calls": len(inference),
        "expert_total_backbone_windows": 94 * len(inference),
        "reader_gate": reader_gate,
        "resource_gate": resource_gate,
        "main": profile.state,
        "expert": report,
        "blend": blend,
        "submission_deleted": not (root / "submission.csv").exists(),
        "Report_or_targets_consumed": False,
        "gold_UID_and_group_excluded": True,
        "candidate_gate_passed": False,
        "accuracy_evaluated": False,
        "qualification_remaining": "Root review of this new p011 actual64 branch result, then256/fullhidden runtime qualification and scored adoption. This does not qualify old p010 shared transport or relax its gates.",
    }
    write_new(root / "P011_PROFILE_SUMMARY.json", result)
    return result
