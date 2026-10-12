"""Paired label-free controls for transport safety and the isolated author224 expert."""

import contextlib
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from p006_metrics import Resources, check_csv, sha256
from p007_metrics import observed_reader_source, validate_observations
from p009_metrics import (
    capture_stdout,
    embedded_source,
    fork_transport_probe,
    fusion_maxdiff,
    prepare_fastread,
    registered_namespace,
)
from p010_author224_metrics import CHECKPOINT_FILENAME, infer_expert224, load_expert224, make_stock_namespace
from p010_shared_metrics import (
    finalise_fixed_rank10,
    read_submission,
    run_bounded_pipeline,
    shared_input_receipt,
    write_submission_exclusive,
)

RAW_ABSOLUTE_TOLERANCE = 1e-7
RAW_RELATIVE_TOLERANCE = 0.0
GPU_RESERVED_LIMIT_MIB = 13000


@contextlib.contextmanager
def replay_reader_once(cache, expected_sha, *, checkpoints, uids, labels, root="/kaggle/working"):
    """Replay only this exact p010 reader command and immutable original cache."""
    cache, root = Path(cache), Path(root)
    expected_checkpoints = tuple(map(str, checkpoints))
    expected_names = [f"cnxt_v0_fold{index}.pt" for index in range(3)]
    if len(expected_checkpoints) != 3 or [Path(path).name for path in expected_checkpoints] != expected_names:
        raise ValueError("Replay requires the original three ordered reader checkpoints")
    if len({str(Path(path).parent) for path in expected_checkpoints}) != 1:
        raise ValueError("Replay checkpoints must share the fixed original Input directory")
    if sha256(cache) != expected_sha:
        raise ValueError("Original reader cache SHA changed")
    read_submission(cache, labels, uids)
    reader, output = str(root / "own_src/infer.py"), str(root / "_own.csv")
    alias = str(root / "p010_profile_data")
    original = subprocess.run
    state = {
        "replay_calls": 0,
        "expected_data_alias": alias,
        "cache_sha256": expected_sha,
        "checkpoint_paths": list(expected_checkpoints),
        "UIDs_and12_labels_checked": True,
    }

    def replay(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 1 and command[1] == reader:
            expected_command = [sys.executable, reader, alias, "test", output, *expected_checkpoints]
            if (
                list(command) != expected_command
                or kwargs.get("check") is not True
                or not isinstance(kwargs.get("env"), dict)
            ):
                raise ValueError("Only the exact original p010 reader command and environment may be replayed")
            if args or set(kwargs) != {"check", "env"}:
                raise ValueError("Replay rejects extra reader subprocess arguments")
            if state["replay_calls"] != 0 or sha256(cache) != expected_sha or Path(output).exists():
                raise ValueError("Replay cache drift, repeated inference, or stale output is forbidden")
            read_submission(cache, labels, uids)
            shutil.copyfile(cache, output)
            state["replay_calls"] += 1
            return subprocess.CompletedProcess(command, 0)
        return original(command, *args, **kwargs)

    subprocess.run = replay
    try:
        yield state
    finally:
        subprocess.run = original


def reader_replay_preflight(labels, *, working="/kaggle/working"):
    """Exercise the same reader command shape, without weights, labels, or MRI."""
    original = subprocess.run
    with tempfile.TemporaryDirectory(prefix="p010-artificial-replay-", dir=working) as directory:
        root = Path(directory)
        (root / "own_src").mkdir()
        cache = root / "artificial_reader.csv"
        uids = ["artificial-reader-0", "artificial-reader-1"]
        write_submission_exclusive(cache, uids, [[0.25] * 12, [0.75] * 12], labels)
        expected_sha = sha256(cache)
        checkpoints = [str(root / "artificial_input" / f"cnxt_v0_fold{index}.pt") for index in range(3)]
        command = [
            sys.executable,
            str(root / "own_src/infer.py"),
            str(root / "p010_profile_data"),
            "test",
            str(root / "_own.csv"),
            *checkpoints,
        ]
        with replay_reader_once(
            cache, expected_sha, checkpoints=checkpoints, uids=uids, labels=labels, root=root
        ) as state:
            completed = subprocess.run(command, check=True, env=dict(os.environ))
        if state["replay_calls"] != 1 or completed.returncode != 0 or sha256(root / "_own.csv") != expected_sha:
            raise RuntimeError("Artificial exact p010 replay contract failed")
    if subprocess.run is not original:
        raise RuntimeError("Artificial replay did not restore subprocess.run")
    print("p010 artificial exact reader replay verified; no model or MRI executed")
    return {
        "exact_command_contract_passed": True,
        "replay_calls": 1,
        "p010_data_alias": True,
        "original_subprocess_restored": True,
        "artificial_cache_and_output_removed": True,
        "real_subprocess_or_model_or_MRI_or_labels_consumed": False,
    }


def reader_helper_guard_source(helper_directory, expected_sha256):
    """Self-contained child-only observation imports; no numerical reader edits."""
    directory = Path(helper_directory).resolve()
    if set(expected_sha256) != {"p006_metrics.py", "p007_metrics.py"}:
        raise ValueError("Only the two fixed reader observation dependencies are allowed")
    if any(
        not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in expected_sha256.values()
    ):
        raise ValueError("Fixed SHA-256 hashes are required")
    return f"""import hashlib as _p010_child_hashlib
import importlib as _p010_child_importlib
from pathlib import Path as _p010_child_Path
import sys as _p010_child_sys
_p010_child_helper_directory = _p010_child_Path({str(directory)!r}).resolve()
_p010_child_expected = {dict(sorted(expected_sha256.items()))!r}
for _p010_child_name, _p010_child_sha in _p010_child_expected.items():
    _p010_child_path = _p010_child_helper_directory / _p010_child_name
    if _p010_child_hashlib.sha256(_p010_child_path.read_bytes()).hexdigest() != _p010_child_sha:
        raise RuntimeError("Reader observation dependency SHA-256 mismatch: " + _p010_child_name)
_p010_child_sys.path.insert(0, str(_p010_child_helper_directory))
for _p010_child_name in _p010_child_expected:
    _p010_child_module = _p010_child_importlib.import_module(_p010_child_name[:-3])
    if _p010_child_Path(_p010_child_module.__file__).resolve() != (_p010_child_helper_directory / _p010_child_name):
        raise RuntimeError("Reader observation module resolved outside fixed helper directory")
"""


def bound_reader_observation(source, helper_directory, expected_sha256):
    observed = observed_reader_source(source, "plain")
    marker = "    from p007_metrics import inference_observer\n"
    if observed.count(marker) != 1:
        raise ValueError("Exactly one fixed reader observation import is required")
    guard = reader_helper_guard_source(helper_directory, expected_sha256)
    return observed.replace(marker, "".join("    " + line + "\n" for line in guard.splitlines()) + marker, 1)


def reader_helper_preflight(helper_directory, expected_sha256, *, probe_directory=None):
    """A real child interpreter, before models, checks the exact injected imports."""
    probe_directory = Path(probe_directory or "/kaggle/working/own_src").resolve()
    probe_path = probe_directory / "p010_reader_helper_probe.py"
    guard = reader_helper_guard_source(helper_directory, expected_sha256)
    with probe_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(guard + '\nprint("p010 child reader observation imports verified")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = str(probe_directory.parent / "_own_env") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(probe_path)], env=env, capture_output=True, text=True, check=True, timeout=30
    )
    if completed.stdout != "p010 child reader observation imports verified\n":
        raise RuntimeError("Unexpected reader observation import probe output")
    print(completed.stdout, end="")
    return {
        "actual_subprocess_import_passed": True,
        "helper_directory": str(Path(helper_directory).resolve()),
        "helper_sha256": dict(sorted(expected_sha256.items())),
        "probe_source_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "probe_python_executable": sys.executable,
        "same_reader_script_directory_and_original_fusion_PYTHONPATH_prefix": True,
        "model_or_MRI_or_GPU_forward_invoked": False,
        "parent_PYTHONPATH_modified": False,
    }


def resource_gate(resource_report, reserved_mib):
    sampled = resource_report["sampled_gpu_peak_used_mib"]
    assert len(sampled) == len(reserved_mib) == 2
    assert all(math.isfinite(value) and 0 < value < GPU_RESERVED_LIMIT_MIB for value in [*sampled, *reserved_mib])
    assert math.isfinite(resource_report["elapsed_seconds"]) and resource_report["elapsed_seconds"] > 0
    assert resource_report["sampled_peak_process_tree_rss_bytes"] > 0 and resource_report["samples"] > 0
    return {
        "all_passed": True,
        "sampled_GPU_and_Torch_reserved_below_mib": GPU_RESERVED_LIMIT_MIB,
        "sampled_memory_does_not_prove_unsampled_peak_or_hidden_runtime": True,
    }


def observed_source(source, phase, *, original_transport=False):
    source = source.replace("    ROOT = find_test_root()", "    ROOT = '/kaggle/working/p010_profile_data'", 1)
    assert "    ROOT = find_test_root()" not in source
    if original_transport:
        marker = "            sid, vol, mask = item; i = pos[sid]"
        assert source.count(marker) == 1
        source = source.replace(
            marker, marker + "\n            _p010_profile_inputs[sid] = shared_input_receipt(vol, mask)", 1
        )
        marker = "    models.clear(); gc.collect()"
        assert source.count(marker) == 1
        source = source.replace(
            marker,
            marker
            + """
    global _p010_state
    _p010_state = {"uids":test_ids, "labels":LAB, "shared_inputs":_p010_profile_inputs,
                  "base_raw_probabilities":[p.tolist() for p in arm_probs],
                  "expert_enabled":False,"expert_loads":{},"expert_inferences":{},
                  "transport":{"delivery":"original_unbounded_imap_unordered"}}
""",
            1,
        )
    marker = 'if __name__ == "__main__":\n    main()'
    assert source.count(marker) == 1
    return (
        source.replace(
            marker,
            """_p010_profile_inputs = {}
_p010_base_pairs = []
_p010_base_forwards = []
_p010_base_loaded = []
_p010_hooks = []
_p010_original_load = load_model
_p010_original_init = _pre_init
_p010_original_pair = infer_pair_probs

def _p010_observe_load(*args, **kwargs):
    model, res = _p010_original_load(*args, **kwargs)
    assert res == 320 and all(not module.training for module in model.modules())
    _p010_base_loaded.append(str(next(model.parameters()).device))
    def observe(module, inputs, output):
        assert not torch.is_grad_enabled() and list(inputs[0].shape) == [1,94,3,320,320]
        assert list(output.shape) == [1,12] and bool(torch.isfinite(output).all())
        _p010_base_forwards.append({"device":str(inputs[0].device),"shape":list(inputs[0].shape),
                                  "grad_enabled":False,"model_training":False,"finite":True})
    _p010_hooks.append(model.register_forward_hook(observe))
    return model, res

def _p010_observe_init():
    _p010_original_init()
    assert float(_READER.__globals__["CROP_MM"]) == 140.0
    _sourcefile = _READER.__globals__["__file__"]
    with open(_sourcefile, "rb") as _file:
        _sha = _p010_hashlib.file_digest(_file,"sha256").hexdigest()
    assert _sha == _p010_expected_fastread_sha
    with open(os.path.join(_p010_worker_directory,str(os.getpid())+".json"),"x") as _file:
        json.dump({"pid":os.getpid(),"CROP_MM":140.0,"source_sha256":_sha},_file)

def _p010_observe_pair(*args, **kwargs):
    value = _p010_original_pair(*args, **kwargs)
    assert value.shape == (12,) and bool(np.isfinite(value).all())
    _p010_base_pairs.append({"device":str(args[3]),"finite":True})
    return value

from p010_shared_metrics import shared_input_receipt
import hashlib as _p010_hashlib
from multiprocessing.reduction import ForkingPickler as _p010_Pickler
import pickle as _p010_pickle
_p010_task_identity = {
    "_pre_one":_p010_pickle.loads(_p010_Pickler.dumps(_pre_one)) is _pre_one,
    "_pre_init":_p010_pickle.loads(_p010_Pickler.dumps(_p010_observe_init)) is _p010_observe_init}
assert all(_p010_task_identity.values())
load_model = _p010_observe_load
_pre_init = _p010_observe_init
infer_pair_probs = _p010_observe_pair
main()
""",
            1,
        ).rstrip()
        + "\n"
    )


def run_main(source, phase, uids, labels, *, original_transport=False):
    import torch

    working = Path("/kaggle/working")
    binding = prepare_fastread(source, {hashlib.sha256(embedded_source(source)).hexdigest()})
    workers = working / f"p010_workers_{phase}"
    workers.mkdir(exist_ok=False)
    for filename in ("submission.csv", "_sota_0949.csv"):
        (working / filename).unlink(missing_ok=True)
    assert not (working / "p010_expert_raw.csv").exists()
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    name = f"_p010_profile_{phase}"
    runtime_source = observed_source(source, phase, original_transport=original_transport)
    values = {
        "__file__": str(working / f"{phase}.py"),
        "_p010_worker_directory": str(workers),
        "_p010_expected_fastread_sha": binding["source_sha256"],
    }
    with registered_namespace(name, values) as namespace:
        with Resources() as resources, capture_stdout() as captured:
            exec(compile(runtime_source, f"<p010-{phase}>", "exec"), namespace)
    stdout = captured.getvalue()
    assert "FALLBACK" not in stdout and "FAILED (" not in stdout
    assert stdout.count("wrote /kaggle/working/submission.csv |") == 1
    state = namespace["_p010_state"]
    assert state["uids"] == uids and state["labels"] == labels
    assert set(state["shared_inputs"]) == set(uids)
    assert all(item["valid_slices"] > 0 for item in state["shared_inputs"].values())
    assert len(namespace["ARMS"]) == 1 and len(namespace["_p010_base_pairs"]) == len(uids)
    assert len(namespace["_p010_base_forwards"]) == 2 * len(uids)
    assert len(namespace["_p010_base_loaded"]) == torch.cuda.device_count()
    assert all(namespace["_p010_task_identity"].values()) and name not in sys.modules
    worker_records = [json.loads(path.read_text()) for path in sorted(workers.glob("*.json"))]
    assert worker_records and all(
        item["source_sha256"] == binding["source_sha256"] and item["CROP_MM"] == 140.0 for item in worker_records
    )
    main = working / f"p010_main_{phase}.csv"
    shutil.copyfile(working / "_sota_0949.csv", main)
    state["main_csv"] = check_csv(main, uids, labels)
    state["base_forward_observations"] = namespace["_p010_base_forwards"]
    state["base_model_load_devices"] = namespace["_p010_base_loaded"]
    state["worker_initializers"] = worker_records
    state["runtime_source_sha256"] = hashlib.sha256(runtime_source.encode()).hexdigest()
    state["prediction_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    state["task_pickle_identity"] = namespace["_p010_task_identity"]
    state["registered_module_removed"] = True
    state["torch_peak_reserved_mib"] = [
        torch.cuda.max_memory_reserved(device) / 2**20 for device in range(torch.cuda.device_count())
    ]
    state["resources"] = resources.result()
    state["resource_gate"] = resource_gate(state["resources"], state["torch_peak_reserved_mib"])
    if state["expert_enabled"]:
        assert set(state["expert_inferences"]) == set(uids)
        assert all(
            row["production_qualified"]
            and row["stock_forward_calls"] == 1
            and row["backbone_batch_windows"] == 94
            and row["views"] == 1
            and row["cuda_fp16_observed"]
            and row["no_grad"]
            and row["fallback_calls"] == 0
            for row in state["expert_inferences"].values()
        )
        destination = working / "p010_shared224_raw.csv"
        (working / "p010_expert_raw.csv").rename(destination)
        state["expert_csv"] = check_csv(destination, uids, labels)
    namespace.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return state


def stock224_reference(main_source, uids, labels):
    """Run literal standalone author224 numerics with the same original fastread.

    It is separate from base-only transport controls, so their measured time/RAM
    is not changed by executing an extra specialist during the reference run.
    """
    import multiprocessing as mp

    import numpy as np
    import pandas as pd
    import torch

    root = Path("/kaggle/working/p010_profile_data")
    working = Path("/kaggle/working")
    binding = prepare_fastread(main_source, {hashlib.sha256(embedded_source(main_source)).hexdigest()})
    stock = make_stock_namespace((working / "p010_src/p010_author224_stock.py").read_text())
    devs = [f"cuda:{device}" for device in range(torch.cuda.device_count())]
    assert devs
    inputs, inferences, probabilities = {}, {}, {}
    name = "_p010_standalone224"
    workers = working / "p010_workers_stock224"
    workers.mkdir(exist_ok=False)
    definitions = observed_source(main_source, "stock224")
    assert definitions.endswith("main()\n")
    definitions = definitions[: -len("main()\n")]
    values = {
        "__file__": str(working / "p010_standalone224.py"),
        "_p010_worker_directory": str(workers),
        "_p010_expected_fastread_sha": binding["source_sha256"],
    }
    with registered_namespace(name, values) as namespace:
        exec(compile(definitions, "<p010-standalone-definitions>", "exec"), namespace)
        assert all(namespace["_p010_task_identity"].values())
        series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
        namespace["_SER"] = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}
        namespace["_TSDIR"] = str(root / "test_series")
        models, loads = {}, {}
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
        with Resources() as resources, capture_stdout() as captured:
            for dev in devs:
                models[dev], loads[dev] = load_expert224(
                    namespace["find_weight_file"](CHECKPOINT_FILENAME), dev, stock_namespace=stock
                )

            def consume(dev, item):
                uid, volume, mask = item
                before = shared_input_receipt(volume, mask)
                raw, inferences[uid] = infer_expert224(stock, models[dev], volume, mask, dev)
                assert before == shared_input_receipt(volume, mask)
                inputs[uid], probabilities[uid] = before, raw.tolist()

            nproc = max(4, min(96, (__import__("os").cpu_count() or 8) - 8))
            transport = run_bounded_pipeline(
                lambda: mp.get_context("fork").Pool(nproc, initializer=namespace["_pre_init"]),
                namespace["_pre_one"],
                uids,
                devs,
                consume,
                queue_max=48,
                outstanding_limit=min(2 * nproc, 48),
            )
        assert "FAILED (" not in captured.getvalue()
        assert set(probabilities) == set(inputs) == set(inferences) == set(uids)
        assert all(record["valid_slices"] > 0 for record in inputs.values())
        assert all(
            row["production_qualified"]
            and row["stock_forward_calls"] == 1
            and row["backbone_batch_windows"] == 94
            and row["cuda_fp16_observed"]
            and row["no_grad"]
            and row["fallback_calls"] == 0
            for row in inferences.values()
        )
        matrix = [probabilities[uid] for uid in uids]
        assert np.isfinite(np.asarray(matrix)).all()
        destination = working / "p010_stock224_raw.csv"
        write_submission_exclusive(destination, uids, matrix, labels)
        peak = [torch.cuda.max_memory_reserved(device) / 2**20 for device in range(torch.cuda.device_count())]
        resource_receipt = resource_gate(resources.result(), peak)
        worker_records = [json.loads(path.read_text()) for path in sorted(workers.glob("*.json"))]
        assert worker_records and all(
            item["source_sha256"] == binding["source_sha256"] and item["CROP_MM"] == 140.0 for item in worker_records
        )
        task_identity = namespace["_p010_task_identity"]
        models.clear()
    namespace.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "shared_inputs": inputs,
        "raw_probabilities": matrix,
        "expert_loads": loads,
        "expert_inferences": inferences,
        "transport": transport,
        "expert_csv": check_csv(destination, uids, labels),
        "torch_peak_reserved_mib": peak,
        "resources": resources.result(),
        "resource_gate": resource_receipt,
        "standalone_literal_source": True,
        "extra_decodes_in_shared_candidate": 0,
        "same_fastread_source_sha256": binding["source_sha256"],
        "worker_initializers": worker_records,
        "task_pickle_identity": task_identity,
        "registered_module_removed": True,
    }


def run_fusion(namespace, source, phase, uids, labels, cache, *, replay):
    working = Path("/kaggle/working")
    main = working / f"p010_main_{phase}.csv"
    shutil.copyfile(main, working / "_sota_0949.csv")
    shutil.copyfile(main, working / "submission.csv")
    (working / "_own.csv").unlink(missing_ok=True)
    context = (
        replay_reader_once(
            cache,
            namespace["_p010_original_reader_cache_sha256"],
            checkpoints=namespace["_p010_original_reader_checkpoints"],
            uids=uids,
            labels=labels,
        )
        if replay
        else contextlib.nullcontext()
    )
    with context as replay_state, Resources() as resources, capture_stdout() as captured:
        exec(compile(source, f"<p010-fusion-{phase}>", "exec"), namespace)
    stdout = captured.getvalue()
    assert stdout.count("[Apex Fusion SUCCESS] Successfully fused") == 1
    assert "[Apex Fusion EXCEPTION]" not in stdout and "[Apex Fusion FALLBACK]" not in stdout
    own, fused = working / "_own.csv", working / "submission.csv"
    if replay:
        assert replay_state["replay_calls"] == 1 and sha256(own) == sha256(cache)
    else:
        assert not cache.exists()
        shutil.copyfile(own, cache)
        namespace["_p010_original_reader_cache_sha256"] = sha256(cache)
        namespace["_p010_original_reader_checkpoints"] = tuple(namespace["_o_ck"])
    difference = fusion_maxdiff(main, cache, fused, uids, labels, namespace["TARGET_WEIGHTS"])
    destination = working / f"p010_fused_{phase}.csv"
    shutil.copyfile(fused, destination)
    return {
        "resources": resources.result(),
        "reader_csv": check_csv(cache, uids, labels),
        "fused_csv": check_csv(destination, uids, labels),
        "fixed_original_fusion_max_difference": difference,
        "success_marker_unique": True,
        "fallbacks": 0,
        "reader_replayed": replay,
        "reader_replay_calls": 1 if replay else 0,
        "reader_replay_binding": replay_state if replay else None,
    }


def raw_difference(left, right):
    if len(left) != len(right) or not left or any(len(row) != 12 for matrix in (left, right) for row in matrix):
        raise ValueError("Raw probability comparison must cover equal studies and all12 targets")
    difference = max(
        abs(a - b) for row1, row2 in zip(left, right, strict=True) for a, b in zip(row1, row2, strict=True)
    )
    assert math.isfinite(difference) and difference <= RAW_ABSOLUTE_TOLERANCE
    return {
        "maximum_absolute_difference": difference,
        "exact_zero": difference == 0,
        "absolute_tolerance_predeclared": RAW_ABSOLUTE_TOLERANCE,
        "relative_tolerance_predeclared": RAW_RELATIVE_TOLERANCE,
        "rationale": "At most approximately2 float32 ULP near1; CSV byte identity remains mandatory. No FP16-scale tolerance.",
    }


def paired_shared224(namespace, original, bounded_base, shared, fusion, reader, uids, labels):
    import timm
    import torch

    assert len(uids) == 32
    devices = [torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())]
    assert len(devices) == 2 and all("t4" in name.lower() for name in devices), "Qualification requires real T4x2"
    working = Path("/kaggle/working")
    helper_directory = Path(__file__).resolve().parent
    reader_helper_sha = {name: sha256(helper_directory / name) for name in ("p006_metrics.py", "p007_metrics.py")}
    helper_probe = reader_helper_preflight(helper_directory, reader_helper_sha)
    replay_probe = reader_replay_preflight(labels)
    transport_probe = fork_transport_probe()
    reader_path = working / "own_src/infer.py"
    reader_path.write_text(bound_reader_observation(reader, helper_directory, reader_helper_sha), encoding="utf-8")
    cache = working / "p010_reader_k16.csv"
    assert not cache.exists()
    main_reports, fusion_reports = {}, {}
    for phase, source in (("original", original), ("bounded", bounded_base), ("shared", shared)):
        main_reports[phase] = run_main(source, phase, uids, labels, original_transport=phase == "original")
        fusion_reports[phase] = run_fusion(namespace, fusion, phase, uids, labels, cache, replay=phase != "original")
    reference = stock224_reference(original, uids, labels)
    expected_inputs = main_reports["original"]["shared_inputs"]
    assert all(report["shared_inputs"] == expected_inputs for report in main_reports.values())
    assert reference["shared_inputs"] == expected_inputs
    assert main_reports["original"]["main_csv"]["sha256"] == main_reports["bounded"]["main_csv"]["sha256"]
    assert main_reports["original"]["main_csv"]["sha256"] == main_reports["shared"]["main_csv"]["sha256"]
    assert len({report["fused_csv"]["sha256"] for report in fusion_reports.values()}) == 1
    comparison = {
        "original_vs_bounded_base": raw_difference(
            main_reports["original"]["base_raw_probabilities"][0], main_reports["bounded"]["base_raw_probabilities"][0]
        ),
        "bounded_base_vs_shared_base": raw_difference(
            main_reports["bounded"]["base_raw_probabilities"][0], main_reports["shared"]["base_raw_probabilities"][0]
        ),
        "stock224_vs_shared224": raw_difference(
            reference["raw_probabilities"], main_reports["shared"]["expert_raw_probabilities"]
        ),
        "same_volume_and_mask_bytes_all_controls": True,
        "original_baseline_fused_csv_byte_exact": True,
        "main_coat_base_executions": 3,
        "author224_executions": 2,
        "reader_convnext_executions": 1,
        "reader_K": 16,
        "additional_DICOM_decodes_per_shared_candidate_study": 0,
        "profile_decoder_passes_over_cohort": 4,
        "labels_or_AUC_evaluated": False,
    }
    directory = working / "p007_observation_plain"
    state = json.loads((directory / "inference.json").read_text())
    reader_gate = validate_observations(state, "plain", len(uids))
    assert all(row["shape"] == [4, 6, 16, 3, 384, 384] for row in state["forwards"])
    coverage = [json.loads(path.read_text()) for path in sorted((directory / "p006_cov_k16").glob("*.json"))]
    assert [row["uid"] for row in coverage] == uids
    assert all(row["shape"] == [6, 16, 3, 384, 384] and all(row["mask"]) for row in coverage)
    shutil.copyfile(working / "p010_fused_shared.csv", working / "submission.csv")
    blend = finalise_fixed_rank10(working / "submission.csv", working / "p010_shared224_raw.csv", labels, uids)
    destination = working / "p010_fixed10.csv"
    shutil.copyfile(working / "submission.csv", destination)
    final_csv = check_csv(destination, uids, labels)
    (working / "submission.csv").unlink()
    assert not (working / "submission.csv").exists()
    return {
        "main": main_reports,
        "standalone224": reference,
        "fusion": fusion_reports,
        "comparison": comparison,
        "fixed_rank10": blend,
        "fixed_rank10_csv": final_csv,
        "reader": {
            "inference_observations": state,
            "forward_gate": reader_gate,
            "coverage": coverage,
            "csv": check_csv(cache, uids, labels),
            "runtime_reader_sha256": sha256(reader_path),
        },
        "artificial_fork_transport": transport_probe,
        "actual_child_reader_helper_preflight": helper_probe,
        "artificial_exact_reader_replay_preflight": replay_probe,
        "final_submission_csv_absent": True,
        "execution_gate_passed": True,
        "all6slots_batch4_exercised": True,
        "actual_environment": {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "python": sys.version,
            "CUDA": torch.version.cuda,
            "GPU_device_names": devices,
        },
        "candidate_gate_passed": False,
        "candidate_gate_note": "Requires root independent completion verification and whole-test runtime decision; no AUC claim.",
    }
