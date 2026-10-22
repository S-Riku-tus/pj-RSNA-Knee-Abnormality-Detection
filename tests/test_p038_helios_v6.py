"""Artificial CSV/files/transport only; no inference source, Torch, model or MRI."""
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "artifacts/kaggle/p038-helios-v6-fixed-recipe-20261017-v3"


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runtime = module("p038_cpu_runtime", "scripts/p038_fixed_recipe_runtime.py")
builder = module("p038_cpu_builder", "scripts/prepare_p038_helios_v6.py")


def rank(values):
    ordered = sorted(values)
    return [(ordered.index(value) + 1 + len(ordered) - ordered[::-1].index(value)) / (2 * len(values)) for value in values]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def write_csv(path, ids, labels, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *labels])
        writer.writerows([[uid, *row] for uid, row in zip(ids, values)])


class P038Offline(unittest.TestCase):
    def test_CPU_only_actual_wrapper_callback_success_partial_failure_and_cleanup(self):
        cpu = ROOT / "artifacts/kaggle/p038-helios-preflight7-cpu-20261017-v1"
        book = json.loads((cpu / "source.ipynb").read_bytes())
        source = book["cells"][1]["source"]
        gpu_manifest = json.loads((PACK / "manifest.json").read_bytes())
        cpu_manifest = json.loads((cpu / "manifest.json").read_bytes())
        self.assertEqual(cpu_manifest["exact_guard_config"], gpu_manifest["guard_config"])
        payload = json.loads((cpu / "save-payload.json").read_bytes())
        self.assertFalse(payload["enable_gpu"] or payload["enable_internet"])
        self.assertEqual(payload["docker_image"], gpu_manifest["requested_docker"])
        self.assertNotIn("machine_shape", payload)
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                target = Path(temp) / "result.json"
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and node.value == "/kaggle/working/P038_PREFLIGHT.json":
                        node.value = str(target)
                exec_index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exec")
                tree.body.insert(exec_index + 1, ast.parse("_p038c_runtime_ns['preflight'] = fake_preflight").body[0])
                owned = []
                partial = {"complete": False, "fixed_members": {"R6": {"members": {"manifest.json": {"sha256": "artificial"}}}}}
                def fake_preflight(config, *, work):
                    owned.append(Path(work))
                    self.assertEqual(config, gpu_manifest["guard_config"])
                    if fail:
                        folder = Path(work) / "P038"
                        folder.mkdir()
                        (folder / "preflight.json").write_bytes((json.dumps(partial) + "\n").encode())
                        raise ValueError("artificial preflight failure")
                    return {"complete": True, "artificial_callback_only": True}
                exec(compile(ast.fix_missing_locations(tree), "<CPU-only-real-wrapper-fake-preflight>", "exec"), {"fake_preflight": fake_preflight})
                result = json.loads(target.read_bytes())
                self.assertEqual(result["complete"], not fail)
                self.assertEqual(result["scientific_code_cells_run"], 0)
                self.assertEqual(result["model_deserialization_or_forward"], 0)
                self.assertTrue(owned and not owned[0].exists())
                if fail:
                    self.assertEqual(result["partial_preflight"], partial)
                    self.assertEqual(result["partial_preflight_sha256"], hashlib.sha256((json.dumps(partial) + "\n").encode()).hexdigest())

    def test_exact_eight_cells_single_mount_delta_source_inputs_and_descriptor(self):
        manifest = json.loads((PACK / "manifest.json").read_bytes())
        notebook = json.loads((PACK / "source.ipynb").read_bytes())
        payload = json.loads((PACK / "save-payload.json").read_bytes())
        recipe = json.loads((PACK / "recipe-cells.json").read_bytes())
        code = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
        tree = ast.parse(code[1])
        restore = tree.body[0].value
        arguments = [ast.literal_eval(value) for value in restore.args]
        self.assertEqual(runtime.restore_cells(*arguments), recipe)
        with self.assertRaises(ValueError):
            runtime.restore_cells(arguments[0], "0" * 64, *arguments[2:])
        changed = []
        for index, text in zip(manifest["author_cell_indices"], recipe):
            original = (PACK / "author-cells" / ("cell-%02d.py" % index)).read_bytes().decode()
            if text != original:
                changed.append(index)
                self.assertEqual(text.replace(builder.STAGE_LINE, "", 1), original)
        self.assertEqual(changed, [6])
        self.assertEqual(payload["text"].encode(), (PACK / "source.ipynb").read_bytes())
        self.assertEqual(len(payload["dataset_data_sources"] + payload["kernel_data_sources"] + payload["competition_data_sources"]), 7)
        self.assertTrue(payload["is_private"] and payload["enable_gpu"] and not payload["enable_internet"])
        self.assertTrue(payload["docker_image"].endswith(builder.DOCKER_SHA))
        config = manifest["guard_config"]
        self.assertEqual(len(config["R6"]) + len(config["Reader"]), 12)
        self.assertNotIn("R7", config)
        self.assertEqual(config["decoder_ABI"], "cp313")
        self.assertEqual(len(config["decoder_wheel_members"]), 3)
        self.assertEqual(len(config["fixture_members"]), 6)
        self.assertEqual(config["producer_descriptor"]["author_source_id"], 359117069)
        self.assertEqual(config["producer_descriptor"]["author_submission"], 57280318)
        self.assertNotIn("torch", sys.modules)

    def test_transport_fakeshell_only_and_partial_error_receipt(self):
        texts = json.loads((PACK / "recipe-cells.json").read_bytes())
        for fail_at in (None, 3):
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                work = Path(temp)
                (work / "P038").mkdir()
                calls = []
                def run_cell(text, store_history):
                    self.assertFalse(store_history)
                    index = len(calls)
                    calls.append(text)
                    return SimpleNamespace(error_before_exec=None, error_in_exec=ValueError("fake cell failure") if index == fail_at else None)
                shell = SimpleNamespace(run_cell=run_cell)
                if fail_at is None:
                    result = runtime.execute_cells(texts, list(range(2, 10)), shell, work=work)
                    self.assertTrue(result["complete"])
                    self.assertEqual(calls, texts)
                else:
                    with self.assertRaisesRegex(ValueError, "fake cell"):
                        runtime.execute_cells(texts, list(range(2, 10)), shell, work=work)
                    result = json.loads((work / "P038/execution.json").read_bytes())
                    self.assertFalse(result["complete"])
                    self.assertEqual(result["failed_cell"], 5)
                    self.assertEqual(len(result["completed"]), 3)

    def test_strict_CP313_wheel_pins_four_fixture_result_and_no_real_subprocess(self):
        config = copy.deepcopy(json.loads((PACK / "manifest.json").read_bytes())["guard_config"])
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / "P038").mkdir()
            wheel_root, fixtures = work / "wheels", work / "phantoms"
            wheel_root.mkdir()
            fixtures.mkdir()
            own = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007/phantoms"
            for name in config["fixture_members"]:
                shutil.copyfile(own / name, fixtures / name)
            for name in config["decoder_wheel_members"]:
                content = b"synthetic-not-a-real-wheel"
                (wheel_root / name).write_bytes(content)
                config["decoder_wheel_members"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            value = {"complete": True, "fixtures": [{"file": name, "syntax": syntax, "sha256": config["fixture_members"][name]["sha256"],
                "shape": [16, 16], "dtype": "uint16", "equal": True, "shape_dtype_values_equal": True} for name, syntax in config["fixture_syntaxes"].items()],
                "reference_sha256": config["fixture_members"]["expected.npy"]["sha256"], "reference_shape": [16, 16], "reference_dtype": "uint16"}
            def process(command, **kwargs):
                self.assertTrue(kwargs["check"])
                return SimpleNamespace(returncode=0, stdout=json.dumps(value) if "-c" in command else "", stderr="")
            # Existing pandas CPU environment is Python3.9; production requires
            # CP313. Only this local file-hash helper substitutes file_digest.
            with patch.object(runtime, "sha", lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()), patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.sys, "version_info", (3, 13, 0)), patch.object(runtime.subprocess, "run", side_effect=process) as mocked:
                result = runtime.strict_decoder_preflight(wheel_root, work, fixtures, config)
                self.assertTrue(result["complete"])
                self.assertEqual(mocked.call_count, 2)
                (wheel_root / next(iter(config["decoder_wheel_members"]))).write_bytes(b"wrong")
                mocked.reset_mock()
                with self.assertRaises(ValueError):
                    runtime.strict_decoder_preflight(wheel_root, work, fixtures, config)
                mocked.assert_not_called()
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("pydicom", sys.modules)

    def test_original_pure_CSV_sections_against_independent_stdlib_72_cells(self):
        texts = json.loads((PACK / "recipe-cells.json").read_bytes())
        labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
        ids = ["s2", "s0", "s1"]
        raw = [[(.1, .4, .9)[i] for _ in labels] for i in range(3)]
        ours = [[(.9, .4, .1)[i] for _ in labels] for i in range(3)]
        reader = [[(.2, .8, .5)[i] for _ in labels] for i in range(3)]
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            work = Path(temp)
            data, pred = work / "data", work / "pred"
            data.mkdir()
            (pred / "worker0").mkdir(parents=True)
            write_csv(data / "test.csv", ids, [], [[], [], []])
            write_csv(data / "sample_submission.csv", ids, labels, raw)
            for name, values in (("ours_sub.csv", ours), ("raptor_raw.csv", raw), ("raptor224_raw.csv", raw)):
                write_csv(work / name, ids, labels, values)
            ranks = [[rank([row[j] for row in raw])[i] for j in range(12)] for i in range(3)]
            for name in ("raptor_sub.csv", "raptor224_sub.csv"):
                write_csv(work / name, ids, labels, ranks)
            write_csv(pred / "worker0/predictions.csv", ids, labels, ours)
            scope = {"DATA": data, "WORK": work, "pred": pred, "NB_T0": time.time(), "time": time, "json": json,
                     "RAPTOR_SECONDS": 0., "RAPTOR224_SECONDS": 0., "display": lambda _: None}
            exec(compile(texts[6], "<original-pure-stage3-CSV-on-artificial-data>", "exec"), scope)
            base = pd.read_csv(work / "submission.csv")
            profile = scope["W_RAPTOR_C"]
            expected_base = []
            for j, label in enumerate(labels):
                r, o = rank([row[j] for row in raw]), rank([row[j] for row in ours])
                w = profile.get(label, .5)
                expected_base.append([f32(w * (.65 * r[i] + .35 * r[i]) + (1 - w) * o[i]) for i in range(3)])
                self.assertEqual([struct.pack("<f", x) for x in base[label]], [struct.pack("<f", x) for x in expected_base[-1]])
            shutil.copyfile(work / "submission.csv", work / "_public_stack.csv")
            write_csv(work / "_own.csv", ids, labels, reader)
            tree = ast.parse(texts[7])
            branch = next(node for node in tree.body if isinstance(node, ast.If))
            body = next(node for node in branch.orelse if isinstance(node, ast.Try)).body
            start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_o_stack")
            end = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr) and ast.unparse(node.value).startswith("_o_final.to_csv("))
            math_only = ast.Module(body=copy.deepcopy(body[start:end + 1]), type_ignores=[])
            reader_weights = ast.literal_eval(next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "OWN_W_PER_TARGET" for target in node.targets)))
            scope2 = {"_o_pd": pd, "_o_bak": str(work / "_public_stack.csv"), "_o_pub": str(work / "submission.csv"),
                      "OWN_P33": 0., "OWN_MODE": "end", "OWN_W": .15, "OWN_W_PER_TARGET": reader_weights}
            # Rewrite only the one artificial test path; original arithmetic AST stays exact.
            for node in ast.walk(math_only):
                if isinstance(node, ast.Constant) and node.value == "/kaggle/working/_own.csv":
                    node.value = str(work / "_own.csv")
            exec(compile(ast.fix_missing_locations(math_only), "<original-pure-Reader-CSV-math-on-artificial-data>", "exec"), scope2)
            final = pd.read_csv(work / "submission.csv")
            for j, label in enumerate(labels):
                rb, rr = rank(expected_base[j]), rank([row[j] for row in reader])
                w = reader_weights[label]
                expected = rank([(1 - w) * rb[i] + w * rr[i] for i in range(3)])
                self.assertTrue(all(math.isfinite(x) for x in final[label]))
                self.assertEqual(final[label].tolist(), expected)
            self.assertEqual(final.StudyInstanceUID.tolist(), ids)
            self.assertEqual(final.columns.tolist(), ["StudyInstanceUID", *labels])
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
