"""Artificial-only contracts for the isolated native-Raptor scheduling change."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_p003_speed import build, scheduling_source  # noqa: E402
from p003_guard import P003Guard  # noqa: E402
from p003_speed_guard import P003SpeedProfileGuard  # noqa: E402
from p003_speed_runtime import RaptorSchedule, compare_arrays, paired_prefetch  # noqa: E402


class SpeedContracts(unittest.TestCase):
    def test_generator_both_modes_preserve_original_and_reject_overwrite(self):
        source = json.loads((ROOT / "notebooks/public/vendor/haideptry-speedy-v2.ipynb").read_text(encoding="utf-8"))
        frozen = {str(i): "".join(c["source"]) for i, c in enumerate(source["cells"]) if c["cell_type"] == "code"}
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("profile", "submission"):
                out = Path(temporary) / mode
                manifest = build(out, mode)
                notebook = json.loads((out / manifest["notebook"]).read_text(encoding="utf-8"))
                wrapped = {}
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    parsed = ast.parse("".join(cell["source"]))
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    if cell["id"].startswith("p003-source-"):
                        call = parsed.body[-1].value
                        wrapped[str(ast.literal_eval(call.args[0]))] = ast.literal_eval(call.args[1])
                for number, code in wrapped.items():
                    if number == "27":
                        self.assertEqual(code, scheduling_source(frozen[number]))
                    elif mode == "profile" and number == "6":
                        self.assertIn("os.environ['RSNA_COMP_ROOT']", code)
                    else:
                        self.assertEqual(code, frozen[number])
                self.assertEqual("29" in wrapped, mode == "submission")
                self.assertEqual(manifest["speed"]["resident_native_models_after"], 2)
                second = Path(temporary) / (mode + "-rebuilt")
                build(second, mode)
                self.assertEqual(
                    (out / manifest["notebook"]).read_bytes(), (second / manifest["notebook"]).read_bytes()
                )
                with self.assertRaises(FileExistsError):
                    build(out, mode)

    def test_failure_diagnostics_do_not_override_parent_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = "raise ValueError('synthetic failure')\n"
            import hashlib

            guard = P003SpeedProfileGuard(
                {"cell_order": [1], "source_cells": {"1": hashlib.sha256(source.encode()).hexdigest()}},
                work=temporary,
            )
            guard.ready = guard.owns_outputs = True
            guard.speed_phases = []
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                guard.run_cell(1, source, {})
            failure = json.loads((Path(temporary) / "P003_SPEED_FAILURE.json").read_text())
            self.assertIn("ValueError: synthetic failure", failure["traceback"])
            self.assertFalse(guard.ready)
            self.assertTrue((Path(temporary) / "P003_FAILED.json").exists())

    def test_profile_rejects_absent_parity_before_completing(self):
        with tempfile.TemporaryDirectory() as temporary:
            guard = P003SpeedProfileGuard({"speed": {"mode": "profile"}}, work=temporary)
            guard.owns_outputs = True
            runtime = RaptorSchedule(temporary, profile=True)
            with mock.patch.object(P003Guard, "finish") as parent_finish:
                with self.assertRaisesRegex(ValueError, "Missing exact"):
                    guard.finish({"_p003_speed_schedule": runtime})
                parent_finish.assert_not_called()
            self.assertFalse((Path(temporary) / "P003_PROFILE.json").exists())

    def test_patch_refuses_unknown_source(self):
        with self.assertRaises(ValueError):
            scheduling_source("print('changed')\n")

    def test_prefetch_order_and_failure(self):
        visits = []

        def prepare(uid):
            visits.append(uid)
            return uid + "!"

        result = [(i, uid, future.result()) for i, uid, future in paired_prefetch(["c", "a", "b"], prepare)]
        self.assertEqual(result, [(0, "c", "c!"), (1, "a", "a!"), (2, "b", "b!")])
        self.assertEqual(visits, ["c", "a", "b"])
        self.assertEqual(list(paired_prefetch([], prepare)), [])

        def fail(uid):
            raise RuntimeError("decode failed")

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            for _, _, future in paired_prefetch(["x"], fail):
                future.result()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
    def test_comparison_includes_ties_and_rejects_nonfinite(self):
        import numpy as np

        reference = np.array([[0.1, 0.8], [0.1, 0.2], [0.9, 0.4]], dtype=np.float32)
        self.assertTrue(compare_arrays(reference, reference.copy())["exact_equal"])
        candidate = reference.copy()
        candidate[0, 0] = 0.2
        self.assertGreater(compare_arrays(reference, candidate)["rank_changed_values"], 0)
        candidate[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compare_arrays(reference, candidate)

    def test_artificial_cuda_candidate_reference_forward(self):
        try:
            import numpy as np
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch/numpy unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        class TinyRaptor(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.projection = nn.Linear(4, 12)
                with torch.no_grad():
                    for parameter in self.parameters():
                        parameter.fill_(offset)

            def head(self, feats):
                return self.projection(feats.mean(dim=1))

        # Execute only the frozen inference FUNCTION definition, using an artificial model.
        notebook = json.loads((ROOT / "notebooks/public/vendor/haideptry-speedy-v2.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][27]["source"])
        tree = ast.parse(source)
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_ke_infer_input"
        )
        ns = {"_KE_NS": {"torch": torch}, "_ke_np": np, "rsna_event": lambda *a, **k: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<frozen-infer-function>", "exec"), ns)
        visits = []
        preparation = (
            "def prepare(arm, uid, series, root, reader):\n"
            "    visits.append((arm['name'], uid))\n"
            "    return torch.full((9, 3, 8, 8), float(uid) / 10)\n"
        )
        prepare_ns = {"torch": torch, "visits": visits}
        exec(preparation, prepare_ns)

        def reader():
            return None, None, None

        arms = [{"name": f"unused{i}"} for i in range(4)]
        arms[1] = {"name": "native384dense-v10", "file": "1", "arch": "tiny", "res": 8}
        arms[3] = {"name": "native384-v8", "file": "3", "arch": "tiny", "res": 8}
        ke = ns["_KE_NS"]
        ke.update(
            _make_reader=reader,
            infer_probs=ns["_ke_infer_input"],
            find_weight_file=lambda name: name,
            load_model=lambda name, arch, res, device: (TinyRaptor(float(name) / 100).eval().to(device), res),
            LAB=[f"label{i}" for i in range(12)],
        )
        ns.update(
            _ke_prepare_windows=prepare_ns["prepare"],
            _ke_base_make_reader=reader,
            rsna_deadline=lambda *args: None,
        )
        outputs = [np.full((3, 12), np.nan, dtype=np.float32) for _ in arms]
        with tempfile.TemporaryDirectory() as temporary:
            schedule = RaptorSchedule(temporary, profile=True)
            schedule.install(ns)
            schedule.run_native_pair(ns, arms, ["1", "2", "3"], {}, "unused", outputs, torch.device("cuda:0"))
            report = json.loads((Path(temporary) / "P003_SPEED_RAPTOR.json").read_text())
            self.assertTrue(report["parity_passed"])
            self.assertTrue(np.isfinite(outputs[1]).all())
            self.assertTrue(np.isfinite(outputs[3]).all())
            self.assertEqual(visits[:6], [(arms[i]["name"], uid) for uid in ("1", "2", "3") for i in (1, 3)])
            self.assertEqual(visits[6:], [(arms[i]["name"], uid) for i in (1, 3) for uid in ("1", "2", "3")])


if __name__ == "__main__":
    unittest.main()
