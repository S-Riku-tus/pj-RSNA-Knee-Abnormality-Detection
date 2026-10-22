"""Artificial CSV/stdio contracts; stdlib only, no models, MRI or real inference."""
import ast
import csv
import hashlib
import itertools
import json
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p042_rank_renormalization_runtime as R
import prepare_p042_rank_renormalization as B


def pairwise_rank(values, valid):
    population = [value for value, ok in zip(values, valid) if ok]
    return [float(Fraction(2 * sum(v < value for v in population) + sum(v == value for v in population) + 1,
                           2 * len(population))) if ok else float("nan")
            for value, ok in zip(values, valid)]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fixture(a, b, own, masks=None):
    n = len(a)
    masks = masks or [[False] * n for _ in range(3)]
    ids = ["1.2.3." + str(i + 1) for i in range(n)]
    raw = [[{R.ID: uid, **{c: repr(.5 if mask[i] else values[i]) for c in R.TARGETS}}
            for i, uid in enumerate(ids)] for values, mask in zip((a, b, own), masks)]
    parent = [{R.ID: uid} for uid in ids]
    expected = [{R.ID: uid} for uid in ids]
    ra = pairwise_rank(a, [not x for x in masks[0]])
    rb = pairwise_rank(b, [not x for x in masks[1]])
    ro = pairwise_rank(own, [not x for x in masks[2]])
    rr = [y if fa else x if fb else (1 - .35) * x + .35 * y
          for x, y, fa, fb in zip(ra, rb, *masks[:2])]
    rapfail = [x and y for x, y in zip(*masks[:2])]
    renormalized = pairwise_rank(rr, [not x for x in rapfail])
    for c in R.TARGETS:
        w = R.WR.get(c, .5)
        for i in range(n):
            value = w * rr[i] + (1 - w) * ro[i]
            if rapfail[i]:
                value = ro[i]
            if masks[2][i]:
                value = rr[i]
            if rapfail[i] and masks[2][i]:
                value = .5
            parent[i][c] = format(f32(value), ".9g")
            new = f32(w * renormalized[i] + (1 - w) * ro[i]) if not any(mask[i] for mask in masks) else None
            expected[i][c] = parent[i][c] if new is None else repr(new)
    return parent, raw, masks, expected


class IntermediateRank(unittest.TestCase):
    def test_rank_normalization_can_change_final_order_not_just_final_scale(self):
        found = None
        base = [.1, .2, .3, .4]
        for other in itertools.permutations(base):
            for ours in itertools.permutations(base):
                values = fixture(base, other, ours)
                result, receipt = R.transform(values[0], *values[1], *values[2])
                if receipt["changed_final_rank_cells"]:
                    found = values, result, receipt
                    break
            if found:
                break
        self.assertIsNotNone(found)
        values, result, receipt = found
        self.assertEqual(result, values[3])
        self.assertGreater(receipt["changed_final_rank_cells"], 0)
        self.assertFalse(receipt["new_final_rank_added"])

    def test_pairwise_fraction_oracle_and_average_ties(self):
        values = fixture([.1, .1, .4, .2], [.2, .1, .1, .4], [.4, .1, .1, .2])
        result, receipt = R.transform(values[0], *values[1], *values[2])
        self.assertEqual(result, values[3])
        self.assertEqual(receipt["parent_FP32_values_verified"], 48)
        self.assertEqual(R.pct_on([.1, .1, .3], [True] * 3), [.5, .5, 1.])

    def test_failed_row_strings_and_single_arm_fallback_rank_population(self):
        n = 40
        a = [(i + 1) / 41 for i in range(n)]
        b = list(reversed(a))
        own = a[5:] + a[:5]
        masks = [[False] * n for _ in range(3)]
        masks[0][3] = True
        masks[1][5] = True
        masks[2][7] = True
        masks[0][9] = masks[1][9] = True
        values = fixture(a, b, own, masks)
        result, receipt = R.transform(values[0], *values[1], *values[2])
        self.assertEqual(result, values[3])
        self.assertEqual(receipt["raptor_rank_population"], 39)
        self.assertEqual(receipt["preserved_failure_rows"], 4)
        for i in (3, 5, 7, 9):
            self.assertEqual(result[i], values[0][i])

    def test_empty_and_singleton_no_divide_by_zero_or_fake_change(self):
        for a, b, ours in (([], [], []), ([.1], [.2], [.3])):
            values = fixture(a, b, ours)
            result, receipt = R.transform(values[0], *values[1], *values[2])
            self.assertEqual(result, values[3])
            self.assertEqual(receipt["changed_FP32_values"], 0)

    def test_UID_duplicate_missing_foreign_and_parent_FP32_tamper_refused(self):
        for kind in ("duplicate", "missing", "foreign", "parent"):
            with self.subTest(kind=kind):
                parent, raw, masks, _ = fixture([.1, .2], [.2, .1], [.1, .2])
                if kind == "duplicate":
                    raw[0][1][R.ID] = raw[0][0][R.ID]
                elif kind == "missing":
                    raw[1].pop()
                elif kind == "foreign":
                    raw[2][0][R.ID] = "1.2.999"
                else:
                    parent[0]["ACL"] = ".123456"
                with self.assertRaises(ValueError):
                    R.transform(parent, *raw, *masks)

    def test_wrong_masks_nonfinite_columns_and_header_order_rejected(self):
        parent, raw, masks, _ = fixture([.1, .2], [.2, .1], [.1, .2])
        with self.assertRaisesRegex(ValueError, "failure masks"):
            R.transform(parent, *raw, [0, 0], *masks[1:])
        raw[0][0]["ACL"] = "NaN"
        with self.assertRaisesRegex(ValueError, "Finite"):
            R.transform(parent, *raw, *masks)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "wrong.csv"
            p.write_text(",".join(reversed([R.ID, *R.TARGETS])) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "column order"):
                R.read_predictions(p)

    def test_offline_builder_preserves_all_eight_parent_AST_bytes_and_exact_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "candidate"
            result = B.prepare(out, "rsraki/p042-rank-rr-20261019-v1")
            cells = json.loads((out / "science-cells.json").read_bytes())
            parent = json.loads((B.PARENT / "source.ipynb").read_bytes())
            self.assertEqual(cells, [B.text(c) for c in parent["cells"] if c["cell_type"] == "code"])
            self.assertEqual(result["science_inventory"], result["parent_science_inventory"])
            old = json.loads((B.PARENT / "save-payload.json").read_bytes())
            new = json.loads((out / "save-payload.json").read_bytes())
            for key in ("dataset_data_sources", "kernel_data_sources", "competition_data_sources", "docker_image"):
                self.assertEqual(new[key], old[key])
            for cell in json.loads((out / "source.ipynb").read_bytes())["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"])
            with self.assertRaisesRegex(ValueError, "Fresh"):
                B.prepare(out, "rsraki/p042-rank-rr-20261019-v1")

    def test_restore_refuses_transport_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "candidate"
            B.prepare(out, "rsraki/p042-rank-rr-20261019-v1")
            # Only harmless notebook assignments are inspected; science is never executed.
            notebook = json.loads((out / "source.ipynb").read_bytes())
            call = ast.parse(notebook["cells"][-1]["source"]).body[0].value
            args = [ast.literal_eval(node) for node in call.args]
            cells = R.restore_cells(*args)
            self.assertEqual(len(cells), 8)
            args[1] = "0" * 64
            with self.assertRaisesRegex(ValueError, "compressed"):
                R.restore_cells(*args)

    def test_failed_overlay_is_fatal_and_cannot_qualify_parent_only_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            values = fixture([.1], [.2], [.3])
            R.write_predictions(work / "submission.csv", values[0])
            cells = ["# artificial source " + str(i) for i in range(8)]
            inv = [{"bytes": len(c.encode()), "sha256": hashlib.sha256(c.encode()).hexdigest()} for c in cells]
            ns = {"test_ids": [values[0][0][R.ID]], "labels": R.TARGETS,
                  "W_RAPTOR_C": {}}  # Deliberate invalid scientific weight contract.
            shell = SimpleNamespace(user_ns=ns, run_cell=lambda *_a, **_k: SimpleNamespace(error_before_exec=None, error_in_exec=None))
            with self.assertRaisesRegex(ValueError, "fixed P033 weights"):
                R.execute_cells(cells, shell, inv, work=work)
            receipt = json.loads((work / "P042/execution.json").read_bytes())
            self.assertFalse(receipt["complete"])
            self.assertFalse(receipt["qualification_passed"])
            self.assertFalse((work / "P042/overlay-receipt.json").exists())

    def test_namespace_overlay_writes_independently_replayable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            parent, raw, masks, expected = fixture([.1, .2, .3], [.3, .1, .2], [.2, .3, .1])
            R.write_predictions(work / "submission.csv", parent)
            for name, rows in zip(("raptor_raw.csv", "raptor224_raw.csv", "ours_sub.csv"), raw):
                R.write_predictions(work / name, rows)
            pred = work / "worker-outputs"
            (pred / "worker0").mkdir(parents=True)
            R.write_predictions(pred / "worker0/predictions.csv", raw[2])
            (work / "blend_receipt.json").write_text('{}\n', encoding="utf-8")
            output = work / "P042"
            output.mkdir()
            ns = {"test_ids": [row[R.ID] for row in parent], "labels": R.TARGETS,
                  "W_RAPTOR_C": R.WR, "W_RAPTOR": .5, "W_OURS": .5, "A224": .35,
                  "r384_fail": masks[0], "r224_fail": masks[1], "our_fail": masks[2],
                  "rap_fail": [False] * 3, "pred": pred}
            receipt = R.apply_namespace(ns, work, output)
            self.assertTrue(receipt["complete"])
            self.assertEqual(R.read_predictions(work / "submission.csv"), expected)
            self.assertEqual(R.read_predictions(output / "parent-submission.csv"), parent)
            with (output / "failure-masks.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row[R.ID] for row in rows], ns["test_ids"])
            self.assertTrue(all(row["rap_fail"] == "0" for row in rows))


if __name__ == "__main__":
    unittest.main()
