"""Run the unchanged CoAt head/windows with random artificial CUDA data; no timm/weights/MRI."""

import ast
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("p009_builder", ROOT / "scripts/build_p009_coat_fov160.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def validate():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F

    from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, validate_submission

    assert torch.cuda.is_available(), "Artificial CUDA verification requires the GPU workstation"
    baseline = builder.control.read_json(builder.BASELINE)
    candidate = builder.prepare_notebook(baseline)
    source = builder.control.cell_source(candidate, 4)
    names = {"RaptorClassifier", "_eval_centers", "_slot_index", "_mirror_tri",
             "gpu_windows", "infer_probs", "infer_pair_probs"}
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    assert len(tree.body) == len(names)
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F, "IMG": 384, "INPUT_RES": 320,
                 "NORM": "imagenet", "_WLUT": {}, "SLOT_SIZES": [26, 22, 18, 12, 18],
                 "SAGITTAL_SLOTS": (0, 1), "_MEAN": torch.tensor([.485, .456, .406]).view(3, 1, 1),
                 "_STD": torch.tensor([.229, .224, .225]).view(3, 1, 1)}
    exec(compile(tree, "p009-original-window-head-functions", "exec"), namespace)
    torch.manual_seed(20261008)
    torch.cuda.reset_peak_memory_stats()

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Conv2d(3, 8, 1)

        def forward(self, x):
            return self.projection(F.adaptive_avg_pool2d(x, (4, 4))).mean((2, 3))

    model = namespace["RaptorClassifier"](TinyEncoder(), F_dim=8, n=12, drop=0).cuda().eval()
    observed = []
    handle = model.register_forward_hook(lambda module, args, out: observed.append({
        "input_shape": list(args[0].shape), "output_shape": list(out.shape),
        "grad_disabled": not torch.is_grad_enabled(), "finite": bool(torch.isfinite(out).all()),
    }))
    uid_values = ["artificial-p009-01", "artificial-p009-02"]
    rows = []
    for index, uid_value in enumerate(uid_values):
        volume = ((np.arange(96 * 384 * 384, dtype=np.uint32).reshape(96, 384, 384)
                   * (index + 3) + 19) % 256).astype(np.uint8)
        mask = np.ones(96, dtype=np.uint8)
        plain = namespace["gpu_windows"](volume, mask, 94, 320, "cuda")
        mirrored = namespace["gpu_windows"](volume, mask, 94, 320, "cuda", mirror=True)
        assert plain.shape == mirrored.shape == (94, 3, 320, 320)
        probability = namespace["infer_pair_probs"](model, plain, mirrored, "cuda")
        assert probability.shape == (12,) and np.isfinite(probability).all()
        rows.append({ID: uid_value, **dict(zip(TARGETS, map(float, probability), strict=True))})
        del plain, mirrored
    handle.remove()
    assert len(observed) == 4 and all(row["grad_disabled"] and row["finite"] for row in observed)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        for name, fields, data in (("submission.csv", SUBMISSION_COLUMNS, rows),
                                   ("test.csv", [ID], [{ID: value} for value in uid_values])):
            with (path / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        contract = validate_submission(path / "submission.csv", path / "test.csv")
    return {
        "schema_version": "p009_artificial_validation_v1", "seed": 20261008,
        "protected_baseline_sha256": builder.control.digest(builder.BASELINE),
        "builder_sha256": builder.control.digest(ROOT / "scripts/build_p009_coat_fov160.py"),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "forwards": observed, "submission_contract": contract,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "six_stdlib_unittests": "run separately before this receipt",
        "limits": [
            "Random 8-dimensional encoder tests original windows/attention and inference API; not real CoAt checkpoint accuracy or memory.",
            "Physical crop paths separately use artificial native pixels and deterministic cv2 stub; no real OpenCV interpolation/MRI equivalence claim.",
            "Real GPU/CPU worker processes and real checkpoint strict loading are not exercised.",
        ],
        "real_MRI_weights_training_remote_or_submission": False,
    }


if __name__ == "__main__":
    target = ROOT / "experiments/p009-coat-fov160-validation-20261008.json"
    assert not target.exists(), "Preserve earlier evidence"
    result = validate()
    builder.control.write_json(target, result)
    print(json.dumps({"validation": str(target), "forwards": len(result["forwards"]),
                      "contract": result["submission_contract"], "gpu": result["gpu"]}))
