"""Small local structural and frozen-source tests; no API/model/MRI/GPU."""

import copy
import importlib.util
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_p011_profile256_tests", ROOT / "scripts/build_p011_profile256.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Diagnostic256Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = MODULE.load_json(MODULE.PLAN)
        cls.prefix = MODULE.load_json(MODULE.CURRENT / "cohort.json")
        cls.cohort = MODULE.cohort256()

    def test_exact_extension_preserves_prefix_groups_folds_and_Gold_exclusion(self):
        cohort = self.cohort
        self.assertEqual(cohort, MODULE.cohort256())
        self.assertEqual(len(cohort["uids"]), 256)
        self.assertEqual(len(set(cohort["uids"])), 256)
        self.assertEqual(cohort["uids"][:64], self.prefix["uids"])
        rows = {r["StudyInstanceUID"]: r for r in self.plan["studies"]}
        all_groups = defaultdict(set)
        for row in self.plan["studies"]:
            all_groups[row["group_id"]].add(row["StudyInstanceUID"])
        selected = set(cohort["uids"])
        for uid in selected:
            row = rows[uid]
            self.assertEqual(cohort["groups"][uid], row["group_id"])
            self.assertEqual(cohort["folds"][uid], row["fold"])
            self.assertEqual(cohort["sources"][uid], "report_weak")
            self.assertTrue(all_groups[row["group_id"]] <= selected)
        self.assertFalse(selected & set(self.plan["gold_ids"]))
        self.assertFalse(set(cohort["groups"].values()) & set(self.plan["gold_groups"]))
        self.assertFalse(cohort["Report_or_targets_included"])
        self.assertFalse(cohort["extension_selected_from_teacher_or_scores"])

    def test_reject_teacher_duplicate_Gold_fold_and_partial_group(self):
        mutations = []
        teacher = copy.deepcopy(self.plan)
        teacher["studies"][0]["Report"] = "sentinel must never be used"
        mutations.append(teacher)
        duplicate = copy.deepcopy(self.plan)
        duplicate["studies"].append(copy.deepcopy(duplicate["studies"][0]))
        mutations.append(duplicate)
        gold_uid = copy.deepcopy(self.plan)
        gold_uid["gold_ids"].append(gold_uid["studies"][0]["StudyInstanceUID"])
        mutations.append(gold_uid)
        gold_group = copy.deepcopy(self.plan)
        gold_group["gold_groups"].append(gold_group["studies"][0]["group_id"])
        mutations.append(gold_group)
        wrong_fold = copy.deepcopy(self.plan)
        wrong_fold["studies"][0]["fold"] = True
        mutations.append(wrong_fold)
        partial = copy.deepcopy(self.plan)
        excluded = next(row for row in partial["studies"] if row["StudyInstanceUID"] not in self.prefix["uids"])
        excluded["group_id"] = self.prefix["groups"][self.prefix["uids"][0]]
        mutations.append(partial)
        labels_selected = copy.deepcopy(self.plan)
        labels_selected["targets_used_for_selection"] = True
        mutations.append(labels_selected)
        for plan in mutations:
            with self.subTest(mutation=mutations.index(plan)), self.assertRaises(ValueError):
                MODULE.extend_cohort(plan, self.prefix)

    def test_book_only_intro_and_cohort_change_original7_and6_literals_exact(self):
        book, stock = MODULE.make_book(self.cohort)
        parent = MODULE.load_json(MODULE.CURRENT_BOOK)
        self.assertEqual(len(book["cells"]), 25)
        self.assertEqual(sum(c["cell_type"] == "code" for c in book["cells"]), 22)
        self.assertEqual({k: v for k, v in parent.items() if k != "cells"}, {k: v for k, v in book.items() if k != "cells"})
        self.assertEqual([i for i, (a, b) in enumerate(zip(parent["cells"], book["cells"], strict=True)) if a != b], [0, 12])
        self.assertEqual(parent["cells"][0]["source"][0].replace("64", "256"), book["cells"][0]["source"][0])
        builder = MODULE.load_builder()
        originals = [c for c in MODULE.load_json(builder.BASE)["cells"] if c["cell_type"] == "code"]
        positions = [book["cells"].index(c) for c in originals]
        self.assertEqual(positions, sorted(positions))
        for cell in originals:
            self.assertEqual(book["cells"].count(cell), 1)
        literals = [c for c in book["cells"] if builder.text(c).startswith("%%writefile ")]
        self.assertEqual(len(literals), 9)
        self.assertEqual(sum(builder.text(c).startswith("%%writefile /kaggle/working/p011_src/") for c in literals), 6)
        for cell in literals:
            self.assertIn(cell, parent["cells"])
        self.assertEqual(stock.encode("utf-8"), (MODULE.CURRENT / "p011_author224_stock.py").read_bytes())
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                source = builder.text(cell)
                compile(source.split("\n", 1)[1] if source.startswith("%%writefile ") else source, "<diagnostic256>", "exec")

    def test_existing_output_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "already-existing"
            current.mkdir()
            sentinel = current / "sentinel.txt"
            sentinel.write_bytes(b"immutable")
            with self.assertRaises(FileExistsError):
                MODULE.build(current, Path(directory) / "new.ipynb", Path(directory) / "new.json")
            self.assertEqual(sentinel.read_bytes(), b"immutable")
            self.assertFalse((Path(directory) / "new.ipynb").exists())


if __name__ == "__main__":
    unittest.main()
