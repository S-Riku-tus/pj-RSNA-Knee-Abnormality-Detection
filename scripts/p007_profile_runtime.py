"""Paired ConvNeXt plain/mirror observations, embedded only in a diagnostic Notebook."""

import ast
import contextlib
import io
import json
import math
import shutil
import sys
from pathlib import Path

from p006_metrics import Resources, check_csv, reader_observer, sha256, write_json


@contextlib.contextmanager
def inference_observer(namespace, mode, output_root="/kaggle/working"):
    """Record actual top-level forwards and TTA calls without replacing their outputs."""
    if mode not in ("plain", "mirror"):
        raise ValueError("Expected plain or mirror observation mode")
    directory = Path(output_root) / f"p007_observation_{mode}"
    directory.mkdir(exist_ok=False)
    state = {"mode": mode, "model_count": None, "load_grad_disabled": None,
             "forwards": [], "mirror_calls": [], "complete": False}
    original_load = namespace["load_models"]
    original_mirror = namespace.get("anatomical_lr_mirror")
    if (mode == "mirror") != (original_mirror is not None):
        raise ValueError("Mirror helper presence does not match the expected source")
    handles = []

    def observed_load(*args, **kwargs):
        import torch

        assert not torch.is_grad_enabled(), "Production run must disable gradients before loading models"
        state["load_grad_disabled"] = True
        models = original_load(*args, **kwargs)
        assert state["model_count"] is None, "Model loading repeated unexpectedly"
        state["model_count"] = len(models)
        assert len(models) == 3, "The pinned reader requires the same three checkpoints"
        for model_index, (model, _) in enumerate(models):
            assert all(not part.training for part in model.modules())

            def observe_forward(module, inputs, output, model_index=model_index):
                import torch

                assert not torch.is_grad_enabled(), "Production reader forward enabled gradients"
                x = inputs[0]
                total = len(state["forwards"])
                phase = "plain" if mode == "plain" or total % 6 < 3 else "mirror"
                assert model_index == total % 3, "Original model mean order changed"
                assert list(x.shape[1:]) == [6, 16, 3, 384, 384]
                assert output.shape == (len(x), 12) and bool(torch.isfinite(output).all())
                state["forwards"].append({
                    "model_index": model_index, "phase": phase, "shape": list(x.shape),
                    "finite_logits": True, "grad_disabled": True,
                })
                # None preserves the exact original output object.

            handles.append(model.register_forward_hook(observe_forward))
        return models

    def observed_mirror(x, mask, pos):
        import torch

        assert len(state["forwards"]) % 6 == 3, "Mirror must follow the three plain forwards"
        result = original_mirror(x, mask, pos)
        xm, maskm, posm = result
        assert maskm is mask and xm.shape == x.shape and xm.dtype == x.dtype == torch.uint8
        assert posm.shape == pos.shape and posm.dtype == pos.dtype
        for slot in (0, 3):
            expected = torch.where(mask[:, slot, None], 1.0 - pos[:, slot], pos[:, slot])
            assert torch.equal(posm[:, slot], expected)
        for slot in (1, 2, 4, 5):
            assert torch.equal(posm[:, slot], pos[:, slot])
        state["mirror_calls"].append({
            "shape": list(x.shape), "valid_slots": mask.tolist(), "mask_object_preserved": True,
            "position_contract_passed": True, "uint8_before_normalization": True,
        })
        return result

    namespace["load_models"] = observed_load
    if mode == "mirror":
        namespace["anatomical_lr_mirror"] = observed_mirror
    try:
        with reader_observer(namespace["DicomStudyDataset"], 16, directory):
            yield state
        state["complete"] = True
    finally:
        namespace["load_models"] = original_load
        if mode == "mirror":
            namespace["anatomical_lr_mirror"] = original_mirror
        for handle in handles:
            handle.remove()
        write_json(directory / "inference.json", state)


def observed_reader_source(source, mode):
    origin = "    run(data_dir, split, ckpts).to_csv(out_csv)"
    assert source.count(origin) == 1
    assert source.count("def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):") == 1
    tree = ast.parse(source)
    runs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    assert len(runs) == 1 and [ast.unparse(node) for node in runs[0].decorator_list] == ["torch.no_grad()"], (
        "Original production run no_grad decorator must stay attached to run"
    )
    invocation = (
        "    from p007_metrics import inference_observer\n"
        f"    with inference_observer(globals(), {mode!r}):\n"
        "        run(data_dir, split, ckpts).to_csv(out_csv)"
    )
    return source.replace(origin, invocation, 1)


def validate_observations(state, mode, studies):
    batches = math.ceil(studies / 4)
    views = 1 if mode == "plain" else 2
    assert state["mode"] == mode and state["complete"]
    assert state["model_count"] == 3
    assert state["load_grad_disabled"] is True
    assert len(state["forwards"]) == batches * 3 * views
    assert len(state["mirror_calls"]) == (0 if mode == "plain" else batches)
    assert all(row["finite_logits"] and row["grad_disabled"] for row in state["forwards"])
    assert sum(row["shape"][0] for row in state["forwards"] if row["model_index"] == 0
               and row["phase"] == "plain") == studies
    if mode == "mirror":
        assert sum(row["shape"][0] for row in state["forwards"] if row["model_index"] == 0
                   and row["phase"] == "mirror") == studies
        assert all(row["position_contract_passed"] and row["mask_object_preserved"]
                   for row in state["mirror_calls"])
    return {"batches": batches, "views": views, "model_forward_calls": len(state["forwards"]),
            "mirror_transform_calls": len(state["mirror_calls"]), "grad_disabled_at_load_and_all_forwards": True,
            "all_passed": True}


def paired_mirror_readers(namespace, fusion_source, plain_source, mirror_source, uids, labels):
    """Reuse one CoAt CSV; execute plain once and the exact p007 two-view reader once."""
    import numpy as np
    import pandas as pd
    import torch

    reports = {}
    for mode, reader_source in (("plain", plain_source), ("mirror", mirror_source)):
        shutil.copyfile("/kaggle/working/_sota_0949.csv", "/kaggle/working/submission.csv")
        Path("/kaggle/working/_own.csv").unlink(missing_ok=True)
        runtime_source = observed_reader_source(reader_source, mode)
        path = Path("/kaggle/working/own_src/infer.py")
        path.write_text(runtime_source, encoding="utf-8")
        captured, destination = io.StringIO(), sys.stdout

        class Tee:
            def write(self, text):
                destination.write(text)
                return captured.write(text)

            def flush(self):
                destination.flush()

        with Resources() as resources, contextlib.redirect_stdout(Tee()):
            exec(compile(fusion_source, f"p007-fusion-{mode}", "exec"), namespace)
        log = captured.getvalue()
        assert log.count("[Apex Fusion SUCCESS] Successfully fused") == 1
        assert "[Apex Fusion EXCEPTION]" not in log and "[Apex Fusion FALLBACK]" not in log
        own, fused = Path("/kaggle/working/_own.csv"), Path("/kaggle/working/submission.csv")
        own_record, fused_record = check_csv(own, uids, labels), check_csv(fused, uids, labels)
        expected = pd.read_csv("/kaggle/working/_sota_0949.csv")[labels].rank(pct=True)
        reader = pd.read_csv(own)[labels].rank(pct=True)
        for label in labels:
            weight = namespace["TARGET_WEIGHTS"][label]
            expected[label] = (1.0 - weight) * expected[label] + weight * reader[label]
        expected = expected.rank(pct=True)
        assert np.allclose(pd.read_csv(fused)[labels], expected, atol=1e-12, rtol=0)
        directory = Path(f"/kaggle/working/p007_observation_{mode}")
        coverage = [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted((directory / "p006_cov_k16").glob("*.json"))]
        assert [row["uid"] for row in coverage] == uids
        assert all(row["shape"] == [6, 16, 3, 384, 384] for row in coverage)
        assert all(any(row["mask"]) for row in coverage)
        state = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
        observed = validate_observations(state, mode, len(uids))
        reports[mode] = {
            **resources.result(), "reader_csv": own_record, "fused_csv": fused_record,
            "coverage": coverage, "inference_observations": state, "forward_gate": observed,
            "observed_runtime_reader_sha256": sha256(path),
            "fusion_matches_fixed_formula": True, "fusion_success_marker": True,
            "fusion_exception_or_fallback": False,
            "observation_note": "Top-level finite-logit hooks synchronize CUDA; elapsed includes observers, pip and cache effects. This is a feasibility profile, not a pure model speed comparison.",
        }
        shutil.copyfile(own, f"/kaggle/working/p007_reader_{mode}.csv")
        shutil.copyfile(fused, f"/kaggle/working/p007_fused_{mode}.csv")
        torch.cuda.empty_cache()
    assert [r["mask"] for r in reports["plain"]["coverage"]] == [
        r["mask"] for r in reports["mirror"]["coverage"]
    ]
    left = pd.read_csv("/kaggle/working/p007_reader_plain.csv")[labels]
    right = pd.read_csv("/kaggle/working/p007_reader_mirror.csv")[labels]
    comparison = {
        "max_probability_delta": float((right - left).abs().to_numpy().max()),
        "mean_absolute_probability_delta": float((right - left).abs().to_numpy().mean()),
        "per_target_rank_correlation": {
            label: float(left[label].rank().corr(right[label].rank())) for label in labels
        },
        "k_fixed": 16, "main_coat_executions": 1, "accuracy_evaluated": False,
        "note": "Differences and correlations do not show AUC improvement; public models are fullfit.",
    }
    Path("/kaggle/working/submission.csv").unlink()
    return reports, comparison
