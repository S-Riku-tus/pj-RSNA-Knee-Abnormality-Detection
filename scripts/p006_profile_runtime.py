"""Observation helpers embedded only in the p006 diagnostic notebook.

No MRI, labels, weights or Kaggle APIs are accessed by this file locally.
"""
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Resources:
    """Sample process-tree RSS and device memory, without changing predictions."""

    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.started = None
        self.stop = threading.Event()
        self.peak_rss = 0
        self.gpu_peak_mib = []
        self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        import psutil
        rss = 0
        for process in [self.proc, *self.proc.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_rss = max(self.peak_rss, rss)
        # RSS is a sum of per-process resident pages; shared pages may be counted twice.
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            values = [int(x.strip()) for x in completed.stdout.splitlines() if x.strip()]
            if not self.gpu_peak_mib:
                self.gpu_peak_mib = values
            else:
                self.gpu_peak_mib = [max(a, b) for a, b in zip(self.gpu_peak_mib, values)]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        self.samples += 1

    def _loop(self):
        while not self.stop.wait(0.5):
            self._sample()

    def __enter__(self):
        self.started = time.monotonic()
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=10)
        self._sample()
        self.elapsed = time.monotonic() - self.started

    def result(self):
        return {
            "elapsed_seconds": self.elapsed,
            "sampled_peak_process_tree_rss_bytes": self.peak_rss,
            "rss_note": "Sum of RSS includes shared pages more than once; sampled, not exact peak.",
            "sampled_gpu_peak_used_mib": self.gpu_peak_mib,
            "gpu_note": "nvidia-smi device-wide sampled memory, includes unrelated allocations on that device.",
            "samples": self.samples,
            "disk_free_bytes_after": shutil.disk_usage("/kaggle/working").free,
        }


@contextlib.contextmanager
def reader_observer(dataset_class, k, output_root="/kaggle/working"):
    """Record exact returned masks; return the original tensors unchanged."""
    directory = Path(output_root) / f"p006_cov_k{k}"
    directory.mkdir(exist_ok=False)
    original = dataset_class.__getitem__

    def observed(self, index):
        result = original(self, index)
        x, mask, pos = result
        record = {
            "uid": str(self.studies[index]),
            "k": k,
            "shape": list(x.shape),
            "mask": mask.tolist(),
            "unique_positions_per_slot": [len(set(v)) for v in pos.tolist()],
        }
        write_json(directory / f"{index:06d}.json", record)
        return result

    dataset_class.__getitem__ = observed
    try:
        yield
    finally:
        dataset_class.__getitem__ = original


def check_csv(path, uids, labels):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    assert list(frame.columns) == ["StudyInstanceUID", *labels], (path, frame.columns)
    assert frame.StudyInstanceUID.tolist() == uids, (path, "UID drift")
    values = frame[labels].to_numpy()
    assert np.isfinite(values).all(), (path, "nonfinite")
    assert ((values >= 0) & (values <= 1)).all(), (path, "range")
    assert (frame[labels].nunique() > 1).all(), (path, "constant")
    return {"sha256": sha256(path), "rows": len(frame), "columns": list(frame.columns)}


def paired_readers(namespace, fusion_source, infer_source, uids, labels):
    """Run the unchanged fusion cell twice, observing k16 and k24 only."""
    import numpy as np
    import pandas as pd
    import torch
    reports = {}
    origin = "    run(data_dir, split, ckpts).to_csv(out_csv)"
    assert infer_source.count(origin) == 1
    signature = "def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):"
    assert infer_source.count(signature) == 1
    path = Path("/kaggle/working/own_src/infer.py")
    for k in (16, 24):
        shutil.copyfile("/kaggle/working/_sota_0949.csv", "/kaggle/working/submission.csv")
        Path("/kaggle/working/_own.csv").unlink(missing_ok=True)
        changed = infer_source.replace(signature, signature.replace("k=16", f"k={k}"))
        # Observer only: model loading, batch size, preprocessing and model calls are unchanged.
        invocation = (
            "    from p006_metrics import reader_observer\n"
            f"    with reader_observer(DicomStudyDataset, {k}):\n"
            "        run(data_dir, split, ckpts).to_csv(out_csv)"
        )
        changed = changed.replace(origin, invocation)
        path.write_text(changed, encoding="utf-8")
        captured = io.StringIO()
        destination = sys.stdout

        class Tee:
            def write(self, text):
                destination.write(text)
                return captured.write(text)

            def flush(self):
                destination.flush()

        with Resources() as resources, contextlib.redirect_stdout(Tee()):
            exec(compile(fusion_source, f"p006-fusion-k{k}", "exec"), namespace)
        log = captured.getvalue()
        assert log.count("[Apex Fusion SUCCESS] Successfully fused") == 1
        assert "[Apex Fusion EXCEPTION]" not in log and "[Apex Fusion FALLBACK]" not in log
        own = Path("/kaggle/working/_own.csv")
        fused = Path("/kaggle/working/submission.csv")
        own_record = check_csv(own, uids, labels)
        fused_record = check_csv(fused, uids, labels)
        # A fallback must not be mistaken for a successfully tested reader.
        expected = pd.read_csv("/kaggle/working/_sota_0949.csv")[labels].rank(pct=True)
        reader = pd.read_csv(own)[labels].rank(pct=True)
        for label in labels:
            w = namespace["TARGET_WEIGHTS"][label]
            expected[label] = (1 - w) * expected[label] + w * reader[label]
        expected = expected.rank(pct=True)
        got = pd.read_csv(fused)[labels]
        assert np.allclose(got, expected, atol=1e-12, rtol=0), "Fusion failed or differs"
        coverage = [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(Path(f"/kaggle/working/p006_cov_k{k}").glob("*.json"))]
        assert [row["uid"] for row in coverage] == uids
        assert all(row["shape"] == [6, k, 3, 384, 384] for row in coverage)
        assert all(any(row["mask"]) for row in coverage), "Study has no valid ConvNeXt slots"
        reports[str(k)] = {
            **resources.result(), "reader_csv": own_record, "fused_csv": fused_record,
            "coverage": coverage, "fusion_matches_fixed_formula": True,
            "fusion_success_marker": True, "fusion_exception_or_fallback": False,
        }
        shutil.copyfile(own, f"/kaggle/working/p006_reader_k{k}.csv")
        shutil.copyfile(fused, f"/kaggle/working/p006_fused_k{k}.csv")
        torch.cuda.empty_cache()
    assert [r["mask"] for r in reports["16"]["coverage"]] == [r["mask"] for r in reports["24"]["coverage"]]
    left = pd.read_csv("/kaggle/working/p006_reader_k16.csv")[labels]
    right = pd.read_csv("/kaggle/working/p006_reader_k24.csv")[labels]
    comparison = {
        "max_probability_delta": float((right - left).abs().to_numpy().max()),
        "mean_absolute_probability_delta": float((right - left).abs().to_numpy().mean()),
        "per_target_rank_correlation": {label: float(left[label].rank().corr(right[label].rank())) for label in labels},
        "accuracy_evaluated": False,
        "note": "Label-free differences are not evidence of AUC improvement; public checkpoints are fullfit.",
    }
    # A diagnostic notebook deliberately does not provide the competition output filename.
    Path("/kaggle/working/submission.csv").unlink()
    return reports, comparison
