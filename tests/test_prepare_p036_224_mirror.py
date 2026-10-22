"""CPU synthetic contract checks for the isolated public anatomical mirror.

Only selected pure tensor functions are compiled from pinned text. Notebook
top-level code, MRI loaders, model loaders and GPU paths are never executed.
"""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_p036", ROOT / "scripts/prepare_p036_224_mirror.py")
P036 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P036)

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    np = torch = F = None


def load_sources():
    parent = json.loads((P036.PARENT / "source.ipynb").read_bytes())
    author = json.loads((P036.AUTHOR / "goodpjw958-V26-source.ipynb").read_bytes())
    old = P036.source(parent["cells"][7])
    new = P036.patch_mirror(old, P036.source(author["cells"][6]))
    return old, new


def tensor_namespace(text, crop=.8):
    tree = ast.parse(P036.body(text))
    names = {"_crop", "_MEAN", "_STD", "_eval_centers", "_WLUT", "gpu_windows", "infer_probs", *P036.HELPERS}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {"np": np, "torch": torch, "F": F, "NORM": "imagenet", "CROP_FRAC": crop}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "synthetic-p036-selected-functions", "exec"), ns)
    return ns


class PinnedPreparationTests(unittest.TestCase):
    def test_only224_changes_and_exact_inputs_remain(self):
        original = (P036.PARENT / "source.ipynb").read_bytes()
        parent = json.loads(original)
        old_payload = json.loads((P036.PARENT / "save-payload.json").read_bytes())
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "new"
            audit = P036.prepare(out)
            candidate = json.loads((out / "source.ipynb").read_bytes())
            payload = json.loads((out / "save-payload.json").read_bytes())
            a = [P036.source(c) for c in parent["cells"] if c["cell_type"] == "code"]
            b = [P036.source(c) for c in candidate["cells"] if c["cell_type"] == "code"]
            self.assertEqual([i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]], [3])
            self.assertEqual(payload["text"].encode(), (out / "source.ipynb").read_bytes())
            for key in old_payload.keys() - {"slug", "new_title", "text"}:
                self.assertEqual(payload[key], old_payload[key], key)
            self.assertTrue(audit["mirror"]["raw224_csv_retained"])
            self.assertFalse(audit["author_958_is_mirror_only_evidence"])
            self.assertFalse(audit["Reader_arm_included"])
            self.assertFalse(audit["F006_included"])
            self.assertIsNone(audit["P033_own_score"])
            self.assertIsNone(audit["candidate_own_Public"])
        self.assertEqual((P036.PARENT / "source.ipynb").read_bytes(), original)

    def test_rejects_modified_parent_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder) / "parent"
            parent.mkdir()
            (parent / "source.ipynb").write_bytes((P036.PARENT / "source.ipynb").read_bytes() + b" ")
            out = Path(folder) / "new"
            with self.assertRaises(ValueError):
                P036.prepare(out, parent_dir=parent)
            self.assertFalse(out.exists())

    def test_rejects_existing_run_and_other_account_slug(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaises(FileExistsError):
                P036.prepare(out)
            with self.assertRaises(ValueError):
                P036.prepare(out / "new", slug="goodpjw2008/p036-not-own")

    def test_rejects_author_cell_mutation_instead_of_executing_it(self):
        old, _ = load_sources()
        author = json.loads((P036.AUTHOR / "goodpjw958-V26-source.ipynb").read_bytes())
        changed = P036.source(author["cells"][6]).replace("tri.flip(1)", "tri.flip(2)")
        with self.assertRaises(ValueError):
            P036.patch_mirror(old, changed)


@unittest.skipUnless(torch is not None, "CPU torch and NumPy required for tensor contract fixtures")
class AnatomicalMirrorFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.new = load_sources()

    def setUp(self):
        self.ns = tensor_namespace(self.new)

    def test_all94_centers_use_correct_axis_including_slot_boundaries(self):
        centers = list(range(1, 95))
        # Each value encodes window, channel, row and column independently.
        tri = torch.arange(94 * 3 * 2 * 4, dtype=torch.int64).reshape(94, 3, 2, 4).remainder(251).to(torch.uint8)
        out = self.ns["_mirror_tri"](tri, centers, 96)
        expected = np.empty(tri.shape, dtype=np.uint8)
        raw = tri.numpy()
        for j, center in enumerate(centers):
            for channel in range(3):
                for row in range(2):
                    for column in range(4):
                        expected[j, channel, row, column] = raw[j, 2-channel, row, column] if center < 48 else raw[j, channel, row, 3-column]
        np.testing.assert_array_equal(out.numpy(), expected)
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(out.device.type, "cpu")
        np.testing.assert_array_equal(self.ns["_mirror_tri"](out, centers, 96).numpy(), raw)

    def test_slot_boundary_uses_center_slice_even_for_cross_slot_triplet(self):
        slots, _ = self.ns["_slot_index"](96)
        self.assertEqual(slots[[25, 26, 47, 48, 65, 66, 77, 78]].tolist(), [0, 1, 1, 2, 2, 3, 3, 4])
        tri = torch.arange(8*3*2*4).reshape(8, 3, 2, 4).to(torch.uint8)
        centers = [25, 26, 47, 48, 65, 66, 77, 78]
        out = self.ns["_mirror_tri"](tri, centers, 96)
        self.assertTrue(torch.equal(out[2], tri[2].flip(0)))
        self.assertTrue(torch.equal(out[3], tri[3].flip(-1)))

    def test_wrong_depth_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.ns["_slot_index"](95)

    def test_gpu_windows_plain_path_is_bit_identical_to_parent_on_cpu(self):
        volume = np.arange(96*6*8, dtype=np.int64).reshape(96, 6, 8).astype(np.uint8)
        mask = np.ones(96, dtype=np.uint8)
        for norm in ("imagenet", "none"):
            # Production uses fixed ImageNet norm. Keep separate caches for the
            # independent optional-norm fixture; the inherited LUT stores a
            # 1D crop/resize helper at ('cpu','none') under ImageNet mode.
            old = tensor_namespace(self.old)
            current = tensor_namespace(self.new)
            a = old["gpu_windows"](volume, mask, 94, 4, "cpu", norm=norm)
            b = current["gpu_windows"](volume, mask, 94, 4, "cpu", norm=norm, mirror=False)
            self.assertTrue(torch.equal(a, b), norm)

    def test_raw_uint8_mirror_precedes_crop_resize_and_channel_normalization(self):
        centers = [1, 25, 26, 47, 48, 65, 66, 77, 78, 94]
        self.ns["_eval_centers"] = lambda mask, depth, k: centers
        volume = np.arange(96*6*8, dtype=np.int64).reshape(96, 6, 8).astype(np.uint8)
        actual = self.ns["gpu_windows"](volume, np.ones(96), len(centers), 4, "cpu", mirror=True)
        # Independent Python indexing first, before any FP32 conversion.
        rows = []
        for center in centers:
            channels = [center-1, center, center+1]
            if center < 48:
                channels.reverse()
            raw = np.stack([volume[index] for index in channels])
            if center >= 48:
                raw = raw[:, :, ::-1].copy()
            rows.append(raw)
        expected = torch.from_numpy(np.stack(rows)).float() / 255
        expected = expected[..., 0:5, 1:7]  # round(6*.8)=5, round(8*.8)=6
        expected = F.interpolate(expected, size=(4, 4), mode="bilinear", align_corners=False)
        expected = (expected - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)

    def test_actual_consumer_branch_averages_fp32_probabilities_without_label_swap(self):
        class FakeModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, x):
                self.calls += 1
                logits = torch.linspace(-4, 7, 12) if x.flatten()[0] == 0 else torch.linspace(5, -6, 12)
                return logits.half()[None]  # verifies float() precedes sigmoid

        model = FakeModel()
        ns = self.ns
        ns.update(model=model, xw=torch.zeros(2, 3, 2, 2), vol=None, mask=None,
                  K_EVAL=94, res=224, d="cpu", arm_probs=[np.zeros((1, 12), dtype=np.float32)], a=0, i=0)
        ns["gpu_windows"] = lambda *args, **kwargs: torch.ones(2, 3, 2, 2)
        branch = ast.parse("if False:\n    pass\n" + "\n".join(line[20:] for line in P036.NEW_ELSE.splitlines())).body[0].orelse
        exec(compile(ast.Module(body=branch, type_ignores=[]), "synthetic-consumer-average", "exec"), ns)
        p = torch.sigmoid(torch.linspace(-4, 7, 12)).numpy()
        q = torch.sigmoid(torch.linspace(5, -6, 12)).numpy()
        np.testing.assert_array_equal(ns["arm_probs"][0][0], .5*(p+q))
        self.assertEqual(ns["arm_probs"][0].dtype, np.float32)
        self.assertEqual(model.calls, 2)
        self.assertNotIn("xm", ns)
        self.assertGreater(float(np.max(np.abs(.5*(p+q) - torch.sigmoid(torch.full((12,), .5)).numpy()))), .1)


if __name__ == "__main__":
    unittest.main()
