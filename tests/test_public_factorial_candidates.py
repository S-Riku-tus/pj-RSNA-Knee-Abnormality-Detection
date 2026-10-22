"""Fixed source identities and artificial lifecycle/math; no public cell execution."""
import ast
import base64
import contextlib
import csv
import importlib.util
import io
import json
import lzma
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("scripts/prepare_p040_p041_factorial.py", "factorial_builder_for_tests")
R = load("scripts/p040_p041_factorial_runtime.py", "factorial_runtime_for_tests")


def write_predictions(path, values):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([R.ID, *R.TARGETS])
        for i, value in enumerate(values):
            writer.writerow(["u" + str(i), *([value] * len(R.TARGETS))])


class FakeShell:
    def __init__(self, work, vector, mode="success"):
        self.work, self.vector, self.mode = Path(work), vector, mode
        self.calls, self.user_ns = 0, {}

    def run_cell(self, _text, store_history):
        assert store_history is False
        index = self.calls
        self.calls += 1
        if index == 7:
            for name in ("submission.csv", "raptor_raw.csv", "raptor224_raw.csv", "ours_sub.csv"):
                write_predictions(self.work / name, [.2, .4, .8] if "raw" in name else [.2, .5, .8])
            (self.work / "blend_receipt.json").write_text(json.dumps({"weights": {"raptor_per_class": self.vector, "a224": .35}}))
            pred = self.work / "r6-workers"
            (pred / "worker0").mkdir(parents=True)
            write_predictions(pred / "worker0/predictions.csv", [.2, .5, .8])
            self.user_ns.update(test_ids=["u0", "u1", "u2"], pred=pred, scored={"u0", "u1", "u2"},
                                r384_fail=[False] * 3, r224_fail=[False] * 3, our_fail=[False] * 3)
        elif index == 8:
            (self.work / "_public_stack.csv").write_bytes((self.work / "submission.csv").read_bytes())
            if self.mode == "reader_failure":
                print("own leg FAILED, public stack submission kept: artificial failure")
            else:
                write_predictions(self.work / "_own.csv", [.9, .5, .1])
                write_predictions(self.work / "submission.csv", [1 / 3, 2 / 3, 1])
                if self.mode == "wrong_numeric":
                    write_predictions(self.work / "submission.csv", [.4, 2 / 3, 1])
                self.user_ns.update(OWN_MODE="end", OWN_P33=0.0, OWN_W_PER_TARGET=R.READER_WEIGHTS,
                                    _o_ck=["/fake/cnxt_v0_fold%d.pt" % i for i in range(3)], _o_final=object())
                print("own leg: blended 3 checkpoints at weight 0.15 in 1s")
        return SimpleNamespace(error_before_exec=None, error_in_exec=None)


def config(candidate):
    cells = ["artificial cell %d" % i for i in range(9 if candidate == "P041" else 8)]
    return cells, {"candidate": candidate, "science_inventory": B.inventory(cells),
                   "class_vector": B.OLD_VECTOR if candidate == "P041" else B.NEW_VECTOR}


class FactorialCandidates(unittest.TestCase):
    def test_both_builders_bind_frozen_scientific_factors_and_inert_payload(self):
        parent = B.read(B.P033 / "source.ipynb")
        original = [B.text(c) for c in parent["cells"] if c["cell_type"] == "code"]
        reader = B.read(B.P038 / "recipe-cells.json")[-1]
        for candidate in ("P040", "P041"):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / candidate
                manifest = B.prepare(candidate, out, "rsraki/" + candidate.lower() + "-artificial-source-only")
                science = B.read(out / "science-cells.json")
                if candidate == "P040":
                    self.assertEqual(science[:-1], original[:-1])
                    self.assertEqual(science[-1], B.replace_vector(original[-1]))
                    self.assertEqual(ast.literal_eval(B.assignment(science[-1], "W_RAPTOR_C")[1].value), B.NEW_VECTOR)
                else:
                    self.assertEqual(science[:8], original)
                    self.assertEqual(science[8], reader)
                payload = B.read(out / "save-payload.json")
                self.assertTrue(payload["is_private"] and payload["enable_gpu"])
                self.assertFalse(payload["enable_internet"] or payload["enable_tpu"])
                self.assertEqual(payload["kernel_execution_type"], "SAVE_AND_RUN_ALL")
                self.assertEqual(payload["docker_image_pinning_type"], "original")
                self.assertEqual(manifest["qualifications"]["actual_new_visible_inference"], False)
                with self.assertRaisesRegex(ValueError, "Fresh output"):
                    B.prepare(candidate, out, "rsraki/no-existing-overwrite")

    def test_wrong_old_vector_refused(self):
        with self.assertRaisesRegex(ValueError, "Exact single-line"):
            B.replace_vector("W_RAPTOR_C={'ACL': .1}\n")

    def test_restore_fails_on_changed_transport_hash(self):
        cells, c = config("P040")
        raw = json.dumps(cells).encode()
        compressed = lzma.compress(raw)
        encoded = base64.b85encode(compressed).decode()
        digest = R.hashlib.sha256
        self.assertEqual(R.restore_cells(encoded, digest(compressed).hexdigest(), digest(raw).hexdigest(), c["science_inventory"]), cells)
        with self.assertRaisesRegex(ValueError, "compressed"):
            R.restore_cells(encoded, "0" * 64, digest(raw).hexdigest(), c["science_inventory"])

    def test_reader_tie_break_math_against_manual_expected_ranks(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            write_predictions(p / "parent.csv", [.2, .2, .8])
            write_predictions(p / "own.csv", [.9, .1, .2])
            write_predictions(p / "final.csv", [2 / 3, 1 / 3, 1])
            self.assertEqual(R.verify_reader_final(p / "parent.csv", p / "own.csv", p / "final.csv")["checked_values"], 36)

    def test_p040_artificial_parent_copies_and_same_scored_UID_masks(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            work = Path(temp)
            cells, c = config("P040")
            shell = FakeShell(work, B.NEW_VECTOR)
            receipt = R.execute_cells(cells, shell, c, work=work)
            self.assertTrue(receipt["complete"])
            self.assertEqual(shell.calls, 8)
            self.assertEqual((work / "parent-submission.csv").read_bytes(), (work / "submission.csv").read_bytes())
            self.assertEqual(B.read(work / "mask.json")["R6_scored_UIDs"], ["u0", "u1", "u2"])
            self.assertTrue((work / "P040-execution.json").is_file())

    def test_p041_artificial_positive_Reader_and_FP64_evidence(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            work = Path(temp)
            cells, c = config("P041")
            shell = FakeShell(work, B.OLD_VECTOR)
            receipt = R.execute_cells(cells, shell, c, work=work)
            self.assertTrue(receipt["complete"])
            self.assertEqual(shell.calls, 9)
            self.assertEqual(receipt["completed_cells"][-1]["reader"]["checked_values"], 36)
            self.assertIn("own leg: blended 3 checkpoints", (work / "P041-reader.stdout.txt").read_text())

    def test_Reader_caught_failure_or_false_success_is_fatal_with_receipt(self):
        for mode in ("reader_failure", "wrong_numeric"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                work = Path(temp)
                cells, c = config("P041")
                shell = FakeShell(work, B.OLD_VECTOR, mode)
                reason = "Positive Reader completion" if mode == "reader_failure" else "Reader FP64 rank replay"
                with self.assertRaisesRegex(ValueError, reason):
                    R.execute_cells(cells, shell, c, work=work)
                self.assertEqual(shell.calls, 9)
                receipt = B.read(work / "P041-execution.json")
                self.assertFalse(receipt["complete"])
                self.assertFalse(receipt["qualification_passed"])


if __name__ == "__main__":
    unittest.main()
