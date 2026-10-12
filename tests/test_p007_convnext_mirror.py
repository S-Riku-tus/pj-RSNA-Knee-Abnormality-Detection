"""Anatomical-mirror source binding and synthetic tensor/model contracts."""

import ast
import csv
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p007_mirror", ROOT / "scripts/build_p007_convnext_mirror.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def mirror_namespace():
    from tests.test_p006_convnext_density import artificial_namespace

    namespace = artificial_namespace()
    exec(compile(builder.MIRROR_HELPER, "p007_declared_mirror", "exec"), namespace)
    return namespace


def artificial_cuda_check():
    import torch

    namespace = mirror_namespace()
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable for the p007 synthetic forward")
    torch.manual_seed(20261008)
    mirror = namespace["anatomical_lr_mirror"]
    model = namespace["KneeNet"](pretrained=False, dim=32, depth=2, drop=0, drop_path=0).cuda().eval()
    x = torch.randint(0, 256, (2, 6, 16, 3, 8, 8), dtype=torch.uint8, device="cuda")
    mask = torch.tensor([[True, True, True, False, True, False], [False] * 6], device="cuda")
    pos = torch.linspace(0, 1, 16, device="cuda").view(1, 1, 16).expand(2, 6, 16).clone()
    xm, maskm, posm = mirror(x, mask, pos)
    x2, mask2, pos2 = mirror(xm, maskm, posm)
    assert torch.equal(x2, x) and mask2 is mask and maskm is mask
    assert torch.equal(pos2[~mask], pos[~mask])
    assert torch.allclose(pos2, pos, atol=1e-7, rtol=0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        plain = model(x, mask, pos, res=8, train=False).float().sigmoid()
        reflected = model(xm, maskm, posm, res=8, train=False).float().sigmoid()
        paired = 0.5 * (plain + reflected)
        from_reflected = 0.5 * (
            model(xm, maskm, posm, res=8, train=False).float().sigmoid()
            + model(x2, mask2, pos2, res=8, train=False).float().sigmoid()
        )
    assert tuple(paired.shape) == (2, 12) and bool(torch.isfinite(paired).all())
    assert bool(((paired >= 0) & (paired <= 1)).all())
    assert torch.equal(plain[1], reflected[1]), "The all-missing study must stay unchanged"
    assert torch.allclose(paired, from_reflected, atol=1e-4, rtol=0)
    return {
        "status": "passed_artificial_only", "device": torch.cuda.get_device_name(0),
        "k": 16, "finite_probability_count": int(paired.numel()),
        "mask_identity_preserved": True, "pixel_involution_exact": True,
        "position_involution_max_error": float((pos2 - pos).abs().max()),
        "all_missing_prediction_difference": 0.0,
        "paired_mirror_symmetry_max_difference": float((paired - from_reflected).abs().max()),
        "original_KneeNet_body": True,
        "encoder": "tiny random synthetic8dim Conv2d; real checkpoints not tested",
        "no_real_MRI_weights_training_or_accuracy": True,
    }


def artificial_direct_run_cuda_check(reader_source=None):
    """Call the production run itself with grad enabled; no outer inference-mode shield."""
    import numpy as np
    import torch

    from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, validate_submission

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA unavailable for the direct p007 run contract")
    namespace = mirror_namespace()
    torch.manual_seed(20261008)
    if reader_source is None:
        candidate = builder.prepare_notebook(builder.control.read_json(builder.BASELINE))
        reader_source = builder.control.writefile_body(builder.control.cell_source(candidate, 8))
    tree = ast.parse(reader_source)
    retained = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("run", "anatomical_lr_mirror")]
    studies = ["synthetic-grad-a", "synthetic-grad-b"]

    class Series(list):
        def isin(self, values):
            return [value in values for value in self]

    class Table:
        StudyInstanceUID = Series(studies)
        SeriesInstanceUID = Series(["short3", "short4"])

        def __getitem__(self, selector):
            assert all(selector)
            return self

    namespace["pd"] = types.SimpleNamespace(
        read_csv=lambda path: Table(),
        Index=lambda values, name: types.SimpleNamespace(values=values, name=name),
        DataFrame=lambda data, index, columns: types.SimpleNamespace(data=data, index=index, columns=columns),
    )
    namespace["os"] = types.SimpleNamespace(path=types.SimpleNamespace(isdir=lambda path: False))
    namespace["MAX_SLICES"] = 64
    table = {
        studies[0]: [["short3"], [], ["short4"], [], [], []],
        studies[1]: [[], ["short4"], [], [], [], []],
    }
    namespace["build_study_table"] = lambda metadata, sizes: table
    volumes = {"short3": np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8),
               "short4": np.arange(4 * 8 * 8, dtype=np.uint8).reshape(4, 8, 8)}
    namespace["load_series"] = lambda path: (volumes[path.rsplit("/", 1)[-1]], {})
    models = [(namespace["KneeNet"](pretrained=False, dim=32, depth=2, drop=0,
                                  drop_path=0).cuda().eval(), 8) for _ in range(3)]
    calls, handles = [], []

    def pre_forward(module, arguments):
        assert not torch.is_grad_enabled(), "Production run forwards must disable gradients"
        assert any(parameter.requires_grad for parameter in module.parameters())
        calls.append({"grad_enabled": torch.is_grad_enabled(), "input_shape": list(arguments[0].shape)})

    for model, _ in models:
        handles.append(model.register_forward_pre_hook(pre_forward))
    namespace["load_models"] = lambda paths: models
    exec(compile(ast.Module(body=retained, type_ignores=[]), "production_p007_run", "exec"), namespace)
    try:
        # This context explicitly starts enabled and relies solely on the function's own decorator.
        with torch.enable_grad():
            frame = namespace["run"]("synthetic-only", "test", ["a", "b", "c"],
                                     studies=studies, workers=0, bs=1)
            assert torch.is_grad_enabled(), "The decorator must restore caller gradient state"
        assert frame.data.shape == (2, 12) and bool(np.isfinite(frame.data).all())
        assert bool(((frame.data >= 0) & (frame.data <= 1)).all())
        assert len(calls) == 12 and all(not row["grad_enabled"] for row in calls)
        assert frame.index.name == "StudyInstanceUID" and frame.index.values == studies
        assert frame.columns == namespace["LABELS"]
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            test_path, csv_path = temporary / "test.csv", temporary / "submission.csv"
            with test_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[ID])
                writer.writeheader()
                writer.writerows([{ID: uid} for uid in studies])
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SUBMISSION_COLUMNS)
                writer.writeheader()
                writer.writerows([{ID: uid, **dict(zip(namespace["LABELS"], map(float, prediction)))}
                                  for uid, prediction in zip(studies, frame.data)])
            submission_contract = validate_submission(csv_path, test_path)
            assert submission_contract["valid"]
        return {"status": "passed_artificial_direct_production_run", "caller_grad_initially_enabled": True,
                "caller_grad_restored": True, "models_have_trainable_parameters": True,
                "outer_inference_mode_used": False, "model_forward_calls": len(calls),
                "all_forwards_grad_disabled": True, "finite_probability_count": int(frame.data.size),
                "original_run_dataset_KneeNet_bodies": True,
                "submission_contract": submission_contract,
                "synthetic_encoder_and_DICOM_pandas_stubs": True,
                "actual_cuda_cat_numpy_succeeded": True, "no_real_MRI_weights_training": True}
    finally:
        for handle in handles:
            handle.remove()


class MirrorSourceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = builder.control.read_json(builder.BASELINE)

    def test_only_mirror_factor_changes_and_k24_is_not_combined(self):
        candidate = builder.prepare_notebook(self.baseline)
        result = builder.verify_mirror_change(self.baseline, candidate)
        self.assertEqual(result["effective_k"], 16)
        self.assertEqual(result["other_code_cells_byte_identical"], 6)
        source = builder.control.cell_source(candidate, 8)
        self.assertNotIn(builder.control.CANDIDATE_SIGNATURE, source)
        self.assertIn(builder.RUN_SIGNATURE, source)
        self.assertIn("p = 0.5 * (p_plain + p_mirror)", source)
        self.assertEqual(len(builder.control.syntax_check(candidate)), 7)
        self.assertEqual(candidate["metadata"], self.baseline["metadata"])
        self.assertIn('"Medial Meniscus", "Lateral Meniscus"', builder.control.cell_source(candidate, 7))
        tree = ast.parse(builder.control.writefile_body(source))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anatomical_lr_mirror")
        self.assertEqual([ast.unparse(n) for n in run.decorator_list], ["torch.no_grad()"])
        self.assertEqual(helper.decorator_list, [])
        self.assertTrue(result["original_run_no_grad_decorator_preserved"])

    def test_missing_or_reassigned_no_grad_is_rejected(self):
        source = builder.control.cell_source(self.baseline, 8)
        with self.assertRaisesRegex(ValueError, "signature"):
            builder.modified_reader(source.replace("@torch.no_grad()\n", "", 1))
        candidate = builder.prepare_notebook(self.baseline)
        changed = builder.control.cell_source(candidate, 8).replace(builder.RUN_DECORATED_SIGNATURE,
                                                                    builder.RUN_SIGNATURE, 1)
        candidate["cells"][8]["source"] = changed.splitlines(keepends=True)
        with self.assertRaises(ValueError):
            builder.verify_mirror_change(self.baseline, candidate)

    def test_extra_transform_or_default_changes_rejected(self):
        for before, after in (("for slot in (0, 3):", "for slot in (0, 1):"), ("k=16", "k=24")):
            with self.subTest(after=after):
                candidate = builder.prepare_notebook(self.baseline)
                source = builder.control.cell_source(candidate, 8).replace(before, after, 1)
                candidate["cells"][8]["source"] = source.splitlines(keepends=True)
                with self.assertRaises(ValueError):
                    builder.verify_mirror_change(self.baseline, candidate)

    def test_hash_drift_rejected_and_package_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            tampered = temporary / "bad.ipynb"
            tampered.write_bytes(builder.BASELINE.read_bytes() + b"\n")
            paths = (temporary / "package", temporary / "mirror.ipynb", temporary / "audit.json")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                builder.prepare_package(tampered, *paths)
            self.assertTrue(all(not path.exists() for path in paths))
            audit = builder.prepare_package(builder.BASELINE, *paths)
            self.assertEqual(audit["code_difference"]["effective_k"], 16)
            package = builder.control.read_json(paths[0] / "package.json")
            for name, record in package["files"].items():
                self.assertEqual(builder.control.digest(paths[0] / name), record["sha256"])
            before = builder.control.digest(paths[1])
            with self.assertRaises(FileExistsError):
                builder.prepare_package(builder.BASELINE, *paths)
            self.assertEqual(builder.control.digest(paths[1]), before)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "Torch/NumPy absent")
class MirrorTensorTests(unittest.TestCase):
    def test_plane_axes_coordinates_and_missing_slots(self):
        import torch

        ns = mirror_namespace()
        x = torch.arange(2 * 6 * 16 * 3 * 4 * 7).remainder(256).to(torch.uint8).view(2, 6, 16, 3, 4, 7)
        mask = torch.tensor([[True] * 6, [True, False, False, False, True, False]])
        pos = torch.linspace(0, 1, 16).view(1, 1, 16).expand(2, 6, 16).clone()
        before_x, before_pos = x.clone(), pos.clone()
        xm, maskm, pm = ns["anatomical_lr_mirror"](x, mask, pos)
        self.assertIs(maskm, mask)
        self.assertTrue(torch.equal(x, before_x))
        self.assertTrue(torch.equal(pos, before_pos))
        for slot in (0, 3):
            self.assertTrue(torch.equal(xm[0, slot], x[0, slot].flip(1)))
            self.assertTrue(torch.equal(pm[0, slot], 1 - pos[0, slot]))
        for slot in (1, 2, 4, 5):
            self.assertTrue(torch.equal(xm[0, slot], x[0, slot].flip(-1)))
            self.assertTrue(torch.equal(pm[0, slot], pos[0, slot]))
        self.assertTrue(torch.equal(xm[~mask], x[~mask]))
        self.assertTrue(torch.equal(pm[~mask], pos[~mask]))
        x2, mask2, p2 = ns["anatomical_lr_mirror"](xm, maskm, pm)
        self.assertIs(mask2, mask)
        self.assertTrue(torch.equal(x2, x))
        self.assertTrue(torch.allclose(p2, pos, atol=1e-7, rtol=0))

    def test_qualified_native_centres_match_and_other_grid_can_differ(self):
        import numpy as np

        f = mirror_namespace()["window_centres"]
        for n in range(3, 65):
            for k in (16, 24):
                centres = f(n, k, False)
                self.assertTrue(np.array_equal(centres[::-1], n - 1 - centres))
        centres = f(4, 3, False)
        self.assertFalse(np.array_equal(centres[::-1], 3 - centres))

    def test_token_permutation_keeps_head_prediction_and_normalization_order_matters(self):
        import torch

        ns = mirror_namespace()
        torch.manual_seed(20261008)
        model = ns["KneeNet"](pretrained=False, dim=32, depth=2, drop=0, drop_path=0).eval()
        feat = torch.randn(2, 6, 16, 8)
        mask = torch.tensor([[True] * 6, [True, False, True, False, True, False]])
        pos = torch.rand(2, 6, 16)
        f2, p2 = feat.clone(), pos.clone()
        for slot in (0, 3):
            f2[:, slot] = feat[:, slot].flip(1)
            p2[:, slot] = pos[:, slot].flip(1)
        with torch.inference_mode():
            before, after = model.head(feat, mask, pos), model.head(f2, mask, p2)
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=0))
        x = torch.arange(3 * 8 * 8).to(torch.uint8).view(1, 3, 8, 8)
        mean = torch.tensor(ns["MEAN"]).view(1, 3, 1, 1)
        std = torch.tensor(ns["STD"]).view(1, 3, 1, 1)
        correct = (x.flip(1).float() / 255 - mean) / std
        wrong = ((x.float() / 255 - mean) / std).flip(1)
        self.assertFalse(torch.allclose(correct, wrong))

    def test_cuda_two_view_symmetry_finite_output_and_all_missing_neutrality(self):
        result = artificial_cuda_check()
        self.assertEqual(result["finite_probability_count"], 24)
        self.assertTrue(result["pixel_involution_exact"])
        self.assertEqual(result["all_missing_prediction_difference"], 0.0)

    def test_direct_run_disables_grad_without_outer_context_and_invalid_v1_fails(self):
        result = artificial_direct_run_cuda_check()
        self.assertTrue(result["all_forwards_grad_disabled"])
        self.assertFalse(result["outer_inference_mode_used"])
        self.assertTrue(result["submission_contract"]["valid"])
        baseline_source = builder.control.cell_source(builder.control.read_json(builder.BASELINE), 8)
        invalid = baseline_source.replace(builder.RUN_SIGNATURE, builder.MIRROR_HELPER + builder.RUN_SIGNATURE, 1)
        invalid = invalid.replace(builder.PLAIN_PREDICTION, builder.MIRROR_PREDICTION, 1)
        invalid_body = builder.control.writefile_body(invalid)
        with self.assertRaisesRegex(AssertionError, "disable gradients"):
            artificial_direct_run_cuda_check(invalid_body)


if __name__ == "__main__":
    unittest.main()
