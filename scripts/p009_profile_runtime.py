"""Label-free paired CoAt FOV observations; one unchanged ConvNeXt reader execution."""

import ast
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

from p006_metrics import Resources, check_csv, sha256
from p007_metrics import observed_reader_source, validate_observations


def embedded_source(main_source):
    nodes = [node for node in ast.parse(main_source).body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "_FASTREAD_B64"]
    assert len(nodes) == 1
    return base64.b64decode(ast.literal_eval(nodes[0].value), validate=True)


def prepare_fastread(main_source, allowed_source_hashes, working="/kaggle/working"):
    """Overwrite only this diagnostic source; remove only its exact import-cache file."""
    root = Path(working).resolve()
    path = root / "_fastread.py"
    assert path.resolve().parent == root, "Fastread source must remain within the exact working root"
    payload = embedded_source(main_source)
    expected_hash = hashlib.sha256(payload).hexdigest()
    assert expected_hash in allowed_source_hashes
    old_hash = sha256(path) if path.exists() else None
    assert old_hash is None or old_hash in allowed_source_hashes, "Unexpected pre-existing fastread source"
    cache = Path(importlib.util.cache_from_source(str(path)))
    assert cache.resolve().parent.parent == root and cache.parent.name == "__pycache__", (
        "Exact fastread cache path must remain within this diagnostic working root"
    )
    cache_deleted = cache.is_file()
    if cache_deleted:
        cache.unlink()
    path.write_bytes(payload)
    assert sha256(path) == expected_hash
    return {"source_sha256": expected_hash, "previous_source_sha256": old_hash,
            "specific_cache_file": str(cache), "specific_cache_deleted": cache_deleted,
            "no_other_cache_or_source_removed": True}


@contextlib.contextmanager
def registered_namespace(name, values=None):
    """Expose exec-defined functions to the module lookup used by Pool task pickling."""
    assert name.isidentifier() and name != "__main__" and name not in sys.modules
    module = types.ModuleType(name)
    if values:
        assert "__name__" not in values
        module.__dict__.update(values)
    sys.modules[name] = module
    try:
        yield module.__dict__
    finally:
        assert sys.modules.get(name) is module, "Unexpected replacement of the diagnostic module"
        del sys.modules[name]


def fork_transport_probe():
    """Pure artificial task on the real Linux fork context, before any model/MRI processing."""
    import multiprocessing as mp
    import pickle
    from multiprocessing.reduction import ForkingPickler

    assert "fork" in mp.get_all_start_methods(), "The unchanged original main requires Linux fork"
    name = "_p009_artificial_fork_probe"
    with registered_namespace(name) as namespace:
        exec("def task(value):\n    return value + 3\n", namespace)
        task = namespace["task"]
        assert pickle.loads(ForkingPickler.dumps(task)) is task
        with mp.get_context("fork").Pool(1) as pool:
            actual = pool.map(task, [1, 2, 3])
        assert actual == [4, 5, 6]
    assert name not in sys.modules
    return {"context": "fork", "workers": 1, "artificial_tasks": 3,
            "task_roundtrip_passed": True, "registered_module_removed": True,
            "MRI_model_labels_or_GPU_consumed": False}


def observed_main_source(main_source, root="/kaggle/working/p009_profile_data"):
    """Original predictions plus returned masks, worker binding and forward observers."""
    marker = "    ROOT = find_test_root()"
    assert main_source.count(marker) == 1
    source = main_source.replace(marker, f"    ROOT = {root!r}", 1)
    marker = "            q.put((sid, vol, mask))"
    assert source.count(marker) == 1
    source = source.replace(marker, "            _p009_masks[sid] = int(mask.sum())\n" + marker, 1)
    tail = 'if __name__ == "__main__":\n    main()'
    assert source.count(tail) == 1
    return source.replace(tail, '''_p009_masks = {}
_p009_pairs = []
_p009_forwards = []
_p009_loaded_devices = []
_p009_hooks = []
_p009_original_pair = infer_pair_probs
_p009_original_init = _pre_init
_p009_original_load = load_model

def _p009_observe_init():
    _p009_original_init()
    import hashlib as _h, json as _j
    assert float(_READER.__globals__["CROP_MM"]) == _p009_expected_fov
    _p009_file = _READER.__globals__["__file__"]
    with open(_p009_file, "rb") as _f:
        _p009_hash = _h.file_digest(_f, "sha256").hexdigest()
    assert _p009_hash == _p009_expected_source_hash
    _p009_closure = dict(zip(_READER.__code__.co_freevars,
                              [cell.cell_contents for cell in _READER.__closure__], strict=True))
    assert _p009_closure["IMG"] == IMG == 384 and _p009_closure["SLOTS"] == SLOTS
    _p009_record = {"pid":os.getpid(), "actual_builder_CROP_MM":float(_READER.__globals__["CROP_MM"]),
                    "actual_builder_IMG":int(_p009_closure["IMG"]),
                    "actual_builder_SLOTS":_p009_closure["SLOTS"],
                    "actual_builder_source_sha256":_p009_hash}
    with open(os.path.join(_p009_worker_directory, str(os.getpid())+".json"), "w") as _f:
        _j.dump(_p009_record, _f)

def _p009_observe_load(*args, **kwargs):
    model, res = _p009_original_load(*args, **kwargs)
    assert res == 320 and all(not module.training for module in model.modules())
    _p009_loaded_devices.append(str(next(model.parameters()).device))
    def _p009_forward(module, inputs, output):
        assert not torch.is_grad_enabled()
        assert list(inputs[0].shape) == [1,94,3,320,320]
        assert list(output.shape) == [1,12] and bool(torch.isfinite(output).all())
        _p009_forwards.append({"shape":list(inputs[0].shape),"device":str(inputs[0].device),
                               "grad_disabled":True,"finite_logits":True})
    _p009_hooks.append(model.register_forward_hook(_p009_forward))
    return model, res

def _p009_observe_pair(model, plain, mirrored, device):
    assert plain.shape == mirrored.shape == (94,3,320,320)
    result = _p009_original_pair(model, plain, mirrored, device)
    assert bool(np.isfinite(result).all()) and result.shape == (12,)
    _p009_pairs.append({"device":str(device),"shape":[94,3,320,320],"finite":True})
    return result

_pre_init = _p009_observe_init
load_model = _p009_observe_load
infer_pair_probs = _p009_observe_pair
import pickle as _p009_pickle
from multiprocessing.reduction import ForkingPickler as _p009_ForkingPickler
_p009_task_pickle_identity = {}
for _p009_task_name in ("_pre_one", "_pre_init"):
    _p009_task = globals()[_p009_task_name]
    assert _p009_pickle.loads(_p009_ForkingPickler.dumps(_p009_task)) is _p009_task
    _p009_task_pickle_identity[_p009_task_name] = True
main()
for _p009_hook in _p009_hooks:
    _p009_hook.remove()
''', 1)


@contextlib.contextmanager
def capture_stdout():
    captured, destination = io.StringIO(), sys.stdout

    class Tee:
        def write(self, value):
            destination.write(value)
            return captured.write(value)

        def flush(self):
            destination.flush()

    with contextlib.redirect_stdout(Tee()):
        yield captured


def run_main(main_source, fov, uids, labels, allowed_hashes):
    binding = prepare_fastread(main_source, allowed_hashes)
    worker_directory = Path(f"/kaggle/working/p009_workers_fov{fov}")
    worker_directory.mkdir(exist_ok=False)
    values = {"__file__": f"/kaggle/working/p009_main_fov{fov}.py",
                 "_p009_expected_fov": float(fov),
                 "_p009_expected_source_hash": binding["source_sha256"],
                 "_p009_worker_directory": str(worker_directory)}
    for name in ("submission.csv", "_sota_0949.csv"):
        Path("/kaggle/working", name).unlink(missing_ok=True)
    observed_source = observed_main_source(main_source)
    module_name = f"_p009_profile_main_fov{fov}"
    with registered_namespace(module_name, values) as namespace:
        with Resources() as resources, capture_stdout() as captured:
            exec(compile(observed_source, f"p009-main-fov{fov}", "exec"), namespace)
    assert module_name not in sys.modules
    assert namespace["_p009_task_pickle_identity"] == {"_pre_one": True, "_pre_init": True}
    assert captured.getvalue().count("wrote /kaggle/working/submission.csv |") == 1
    assert "FALLBACK" not in captured.getvalue() and "FAILED (" not in captured.getvalue()
    assert set(namespace["_p009_masks"]) == set(uids)
    assert all(value > 0 for value in namespace["_p009_masks"].values())
    assert len(namespace["ARMS"]) == 1
    assert len(namespace["_p009_pairs"]) == len(uids)
    assert len(namespace["_p009_forwards"]) == len(uids) * 2
    assert len(namespace["_p009_loaded_devices"]) == namespace["torch"].cuda.device_count()
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(worker_directory.glob("*.json"))]
    assert workers and len({row["pid"] for row in workers}) == len(workers)
    assert all(row["actual_builder_CROP_MM"] == fov and row["actual_builder_IMG"] == 384
               and row["actual_builder_source_sha256"] == binding["source_sha256"] for row in workers)
    destination = Path(f"/kaggle/working/p009_main_fov{fov}.csv")
    shutil.copyfile("/kaggle/working/_sota_0949.csv", destination)
    record = check_csv(destination, uids, labels)
    namespace["torch"].cuda.empty_cache()
    return {
        **resources.result(), "fov_mm": fov, "main_csv": record, "fastread_binding": binding,
        "worker_initializers": workers, "valid_slices_by_uid": namespace["_p009_masks"],
        "successful_pair_calls": len(namespace["_p009_pairs"]),
        "forward_observations": namespace["_p009_forwards"],
        "loaded_model_devices": namespace["_p009_loaded_devices"],
        "main_success_marker": True, "main_fallback_or_preprocess_failure": False,
        "prediction_source_sha256": hashlib.sha256(main_source.encode()).hexdigest(),
        "observed_runtime_source_sha256": hashlib.sha256(observed_source.encode()).hexdigest(),
        "task_pickle_identity": namespace["_p009_task_pickle_identity"],
        "registered_module_name": module_name, "registered_module_removed": True,
    }


@contextlib.contextmanager
def replay_reader_once(cache, expected_sha, root="/kaggle/working"):
    """Replay only the exact reader subprocess on the second fusion; restore on any failure."""
    cache = Path(cache)
    assert sha256(cache) == expected_sha
    reader = str(Path(root) / "own_src/infer.py")
    output = str(Path(root) / "_own.csv")
    alias = str(Path(root) / "p009_profile_data")
    original = subprocess.run
    state = {"replay_calls": 0}

    def replay(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) > 1 and command[1] == reader:
            assert len(command) == 8 and command[2:5] == [alias, "test", output]
            assert kwargs.get("check") is True and "env" in kwargs
            assert state["replay_calls"] == 0 and sha256(cache) == expected_sha
            assert not Path(output).exists(), "Old reader CSV must be removed before replay"
            shutil.copyfile(cache, output)
            state["replay_calls"] += 1
            return subprocess.CompletedProcess(command, 0)
        return original(command, *args, **kwargs)

    subprocess.run = replay
    try:
        yield state
    finally:
        subprocess.run = original


def percentile_rank(values):
    indices = sorted(range(len(values)), key=values.__getitem__)
    output = [0.] * len(values)
    start = 0
    while start < len(indices):
        stop = start + 1
        while stop < len(indices) and values[indices[stop]] == values[indices[start]]:
            stop += 1
        value = ((start + 1 + stop) / 2) / len(values)
        for index in indices[start:stop]:
            output[index] = value
        start = stop
    return output


def fusion_maxdiff(main_path, reader_path, fused_path, uids, labels, weights):
    """Independent standard-library average-tie percentile rank formula."""
    records = []
    for path in (main_path, reader_path, fused_path):
        with Path(path).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["StudyInstanceUID"] for row in rows] == uids
        records.append(rows)
    maximum = 0.
    for label in labels:
        main = percentile_rank([float(row[label]) for row in records[0]])
        reader = percentile_rank([float(row[label]) for row in records[1]])
        weight = weights[label]
        expected = percentile_rank([(1 - weight) * a + weight * b for a, b in zip(main, reader, strict=True)])
        actual = [float(row[label]) for row in records[2]]
        maximum = max(maximum, max(abs(a - b) for a, b in zip(expected, actual, strict=True)))
    assert maximum <= 1e-12
    return maximum


def run_fusion(namespace, fusion_source, fov, uids, labels, reader_cache, replay=False):
    root = Path("/kaggle/working")
    main = root / f"p009_main_fov{fov}.csv"
    shutil.copyfile(main, root / "_sota_0949.csv")
    shutil.copyfile(main, root / "submission.csv")
    (root / "_own.csv").unlink(missing_ok=True)
    context = replay_reader_once(reader_cache, sha256(reader_cache)) if replay else contextlib.nullcontext()
    with context as replay_state, Resources() as resources, capture_stdout() as captured:
        exec(compile(fusion_source, f"p009-fusion-fov{fov}", "exec"), namespace)
    log = captured.getvalue()
    assert log.count("[Apex Fusion SUCCESS] Successfully fused") == 1
    assert "[Apex Fusion EXCEPTION]" not in log and "[Apex Fusion FALLBACK]" not in log
    own, fused = root / "_own.csv", root / "submission.csv"
    own_record, fused_record = check_csv(own, uids, labels), check_csv(fused, uids, labels)
    if replay:
        assert replay_state["replay_calls"] == 1 and own_record["sha256"] == sha256(reader_cache)
    else:
        assert not reader_cache.exists()
        shutil.copyfile(own, reader_cache)
    difference = fusion_maxdiff(main, reader_cache, fused, uids, labels, namespace["TARGET_WEIGHTS"])
    shutil.copyfile(fused, root / f"p009_fused_fov{fov}.csv")
    return {**resources.result(), "reader_csv": own_record, "fused_csv": fused_record,
            "fusion_max_difference_fixed_formula": difference, "fusion_success_marker": True,
            "fusion_exception_or_fallback": False, "reader_replayed": replay,
            "reader_replay_calls": replay_state["replay_calls"] if replay else 0}


def paired_fov(namespace, original_main, candidate_main, fusion_source, infer_source, uids, labels):
    import numpy as np
    import pandas as pd
    import torch

    hashes = {hashlib.sha256(embedded_source(value)).hexdigest() for value in (original_main, candidate_main)}
    assert len(hashes) == 2 and len(uids) == 32
    transport = fork_transport_probe()
    runtime_reader = observed_reader_source(infer_source, "plain")
    reader_path = Path("/kaggle/working/own_src/infer.py")
    reader_path.write_text(runtime_reader, encoding="utf-8")
    reader_cache = Path("/kaggle/working/p009_reader_k16.csv")
    assert not reader_cache.exists()
    main_reports, fused_reports = {}, {}
    for fov, main_source in ((140, original_main), (160, candidate_main)):
        main_reports[str(fov)] = run_main(main_source, fov, uids, labels, hashes)
        fused_reports[str(fov)] = run_fusion(namespace, fusion_source, fov, uids, labels, reader_cache, replay=fov == 160)
    directory = Path("/kaggle/working/p007_observation_plain")
    coverage = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((directory / "p006_cov_k16").glob("*.json"))]
    assert [row["uid"] for row in coverage] == uids
    assert all(row["shape"] == [6, 16, 3, 384, 384] and any(row["mask"]) for row in coverage)
    state = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
    gate = validate_observations(state, "plain", len(uids))
    assert fused_reports["140"]["reader_csv"]["sha256"] == fused_reports["160"]["reader_csv"]["sha256"]
    left = pd.read_csv("/kaggle/working/p009_main_fov140.csv")[labels]
    right = pd.read_csv("/kaggle/working/p009_main_fov160.csv")[labels]
    correlations = {label: float(left[label].rank().corr(right[label].rank())) for label in labels}
    assert all(math.isfinite(value) for value in correlations.values())
    comparison = {
        "max_main_rank_delta": float(np.abs(right.to_numpy() - left.to_numpy()).max()),
        "mean_absolute_main_rank_delta": float(np.abs(right.to_numpy() - left.to_numpy()).mean()),
        "per_target_main_rank_correlation": correlations,
        "valid_slice_count_differences": {uid: main_reports["160"]["valid_slices_by_uid"][uid]
                                         - main_reports["140"]["valid_slices_by_uid"][uid] for uid in uids},
        "main_coat_executions": 2, "reader_convnext_executions": 1, "reader_k_fixed": 16,
        "fusion_executions": 2, "second_fusion_reader_csv_replayed_once": True,
        "accuracy_evaluated": False,
        "note": "Label-free prediction differences do not establish AUC improvement. Public weights are fullfit.",
    }
    Path("/kaggle/working/submission.csv").unlink()
    assert not Path("/kaggle/working/submission.csv").exists()
    torch.cuda.empty_cache()
    return {
        "main": main_reports, "fusion": fused_reports,
        "reader": {"coverage": coverage, "inference_observations": state, "forward_gate": gate,
                   "runtime_reader_sha256": sha256(reader_path), "reader_csv": check_csv(reader_cache, uids, labels)},
        "comparison": comparison, "final_submission_csv_absent": True,
        "artificial_fork_transport": transport,
    }
