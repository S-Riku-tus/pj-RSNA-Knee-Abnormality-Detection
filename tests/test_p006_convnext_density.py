"""One-factor source binding and artificial variable-window inference contracts."""

import ast
import copy
import csv
import importlib.util
import math
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p006_density", ROOT / "scripts/build_p006_convnext_density.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def artificial_namespace():
    """Load the unchanged real Dataset/head bodies with a tiny random synthetic encoder."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    book = builder.read_json(builder.BASELINE)
    knee_tree = ast.parse(builder.writefile_body(
        builder.cell_source(book, 7), "%%writefile /kaggle/working/own_src/knee.py"
    ))
    names = {"LABELS", "PLANES", "N_SLOTS", "MEAN", "STD"}
    retained = []
    for node in knee_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            retained.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
            "window_centres", "make_views", "KneeNet"
        }:
            retained.append(node)

    class TinyRandomEncoder(nn.Module):
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.conv(x).mean((-2, -1))

    namespace = {
        "math": math, "np": np, "torch": torch, "nn": nn, "F": functional,
        "timm": types.SimpleNamespace(create_model=lambda *args, **kwargs: TinyRandomEncoder()),
    }
    exec(compile(ast.Module(body=retained, type_ignores=[]), "unchanged_knee_artificial", "exec"), namespace)
    infer_tree = ast.parse(builder.writefile_body(builder.cell_source(book, 8)))
    dataset = next(node for node in infer_tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "DicomStudyDataset")
    namespace["SIZE"] = 8
    exec(compile(ast.Module(body=[dataset], type_ignores=[]), "unchanged_dataset_artificial", "exec"), namespace)
    return namespace


def run_artificial_gpu_smoke():
    """No real checkpoint, DICOM, labels or timm download is used."""
    import numpy as np
    import torch

    from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, validate_submission

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is unavailable for the artificial p006 contract")
    ns = artificial_namespace()
    torch.manual_seed(20261008)
    model = ns["KneeNet"](pretrained=False, dim=32, depth=2, drop=0, drop_path=0).cuda().eval()
    studies = ["1.2.826.0.1.24", "1.2.826.0.1.16", "1.2.826.0.1.3"]
    table = {
        studies[0]: [["short3"], [], ["short4"], [], [], []],
        studies[1]: [[], ["short4"], [], [], [], []],
        studies[2]: [[], [], [], [], [], []],
    }
    volumes = {
        "short3": np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8),
        "short4": np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8),
    }
    ns["load_series"] = lambda path: (volumes[path.rsplit("/", 1)[-1]], {})
    outputs, receipt = {}, []
    for k in (16, 24):
        ds = ns["DicomStudyDataset"](studies, table, "synthetic-only", k)
        samples = [ds[i] for i in range(3)]
        x = torch.stack([item[0] for item in samples]).cuda()
        mask = torch.stack([item[1] for item in samples]).cuda()
        pos = torch.stack([item[2] for item in samples]).cuda()
        assert tuple(x.shape) == (3, 6, k, 3, 8, 8)
        assert tuple(mask.shape) == (3, 6) and int(mask.sum()) == 3
        assert tuple(pos.shape) == (3, 6, k) and bool(((pos >= 0) & (pos <= 1)).all())
        assert int(x[~mask].sum()) == 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            p = model(x, mask, pos, res=8, train=False).float().sigmoid()
            changed = x.clone()
            changed[~mask] = 255
            q = model(changed, mask, pos, res=8, train=False).float().sigmoid()
        assert tuple(p.shape) == (3, 12) and bool(torch.isfinite(p).all())
        assert bool(((p >= 0) & (p <= 1)).all())
        assert torch.equal(p, q), "Masked slot pixels must not affect the prediction"
        outputs[k] = p.cpu().numpy()
        receipt.append({
            "k": k, "input_shape": list(x.shape), "finite_probabilities": int(p.numel()),
            "short_native_series_lengths": [3, 4], "missing_slots": True,
            "all_missing_study": True, "masked_pixel_prediction_difference": 0.0,
        })
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        test = temporary / "test.csv"
        with test.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[ID])
            writer.writeheader()
            writer.writerows([{ID: uid} for uid in studies])
        csv_path = temporary / "submission.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            writer.writerows([
                {ID: uid, **dict(zip(ns["LABELS"], map(float, p)))}
                for uid, p in zip(studies, outputs[24])
            ])
        contract = validate_submission(csv_path, test)
    return {
        "status": "passed_artificial_only", "device": torch.cuda.get_device_name(0),
        "encoder": "tiny randomly initialized synthetic Conv2d; real timm/weights not exercised",
        "original_dataset_and_head_bodies": True, "densities": receipt, "submission_contract": contract,
        "does_not_prove_real_mri_or_hidden_coverage_or_auc": True,
    }


class DensitySourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline = builder.read_json(builder.BASELINE)

    def test_one_code_line_changed_and_effective_entrypoint_is24(self):
        candidate = builder.prepare_notebook(self.baseline)
        self.assertEqual(builder.verify_single_change(self.baseline, candidate), [8])
        self.assertEqual(builder.verify_runtime_binding(candidate, 24)["effective_convnext_k_per_slot"], 24)
        self.assertEqual(len(builder.syntax_check(candidate)), 7)
        before, after = builder.code_sources(self.baseline), builder.code_sources(candidate)
        self.assertEqual(sum(before[i] == after[i] for i in before), 6)
        self.assertEqual(candidate["metadata"], self.baseline["metadata"])

    def test_explicit_runtime_override_rejected(self):
        book = copy.deepcopy(self.baseline)
        source = builder.cell_source(book, 8).replace(
            "run(data_dir, split, ckpts).to_csv(out_csv)",
            "run(data_dir, split, ckpts, k=16).to_csv(out_csv)",
        )
        book["cells"][8]["source"] = source.splitlines(keepends=True)
        with self.assertRaisesRegex(ValueError, "overrides"):
            builder.prepare_notebook(book)

    def test_other_subprocess_source_or_density_flow_rejected(self):
        for cell_index, before, after in (
            (9, "/kaggle/working/own_src/infer.py", "/kaggle/input/other/infer.py"),
            (8, "DicomStudyDataset(studies, table, root, k)", "DicomStudyDataset(studies, table, root, 16)"),
        ):
            with self.subTest(cell=cell_index):
                book = copy.deepcopy(self.baseline)
                source = builder.cell_source(book, cell_index).replace(before, after)
                book["cells"][cell_index]["source"] = source.splitlines(keepends=True)
                with self.assertRaises(ValueError):
                    builder.prepare_notebook(book)

    def test_extra_factor_or_metadata_change_rejected(self):
        for change in ("reader", "fusion", "metadata"):
            with self.subTest(change=change):
                candidate = builder.prepare_notebook(self.baseline)
                if change == "metadata":
                    candidate["metadata"]["density"] = 24
                else:
                    cell_index = 8 if change == "reader" else 9
                    source = builder.cell_source(candidate, cell_index) + "\nEXTRA_FACTOR = True\n"
                    candidate["cells"][cell_index]["source"] = source.splitlines(keepends=True)
                with self.assertRaises(ValueError):
                    builder.verify_single_change(self.baseline, candidate)

    def test_fixed_hash_drift_rejected_before_outputs(self):
        tampered = self.root / "tampered.ipynb"
        tampered.write_bytes(builder.BASELINE.read_bytes() + b"\n")
        paths = (self.root / "package", self.root / "candidate.ipynb", self.root / "audit.json")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            builder.prepare_package(tampered, *paths)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_package_hashes_and_immutable_outputs(self):
        paths = (self.root / "package", self.root / "candidate.ipynb", self.root / "audit.json")
        audit = builder.prepare_package(builder.BASELINE, *paths)
        self.assertEqual(audit["prediction_code_delta"]["unchanged_other_code_cells"], 6)
        self.assertIsNone(audit["candidate"]["own_public_lb"])
        package = builder.read_json(paths[0] / "package.json")
        for name, entry in package["files"].items():
            self.assertEqual(builder.digest(paths[0] / name), entry["sha256"])
        before = builder.digest(paths[1])
        with self.assertRaises(FileExistsError):
            builder.prepare_package(builder.BASELINE, *paths)
        self.assertEqual(builder.digest(paths[1]), before)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "Torch/NumPy absent")
class ArtificialInferenceTests(unittest.TestCase):
    def test_short_native_series_repeats_valid_centres_at16and24(self):
        namespace = artificial_namespace()
        for n in (3, 4, 5):
            for k in (16, 24):
                with self.subTest(native_slices=n, k=k):
                    centres = namespace["window_centres"](n, k, False)
                    self.assertEqual(len(centres), k)
                    self.assertTrue((centres >= 1).all())
                    self.assertTrue((centres <= n - 2).all())

    def test_cuda_variablek_missing_slots_and_submission_contract(self):
        result = run_artificial_gpu_smoke()
        self.assertEqual(result["status"], "passed_artificial_only")
        self.assertTrue(result["submission_contract"]["valid"])


if __name__ == "__main__":
    unittest.main()
