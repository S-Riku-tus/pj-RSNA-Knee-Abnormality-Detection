"""Artificial UID/CSV and generated-source checks; never load MRI or checkpoints."""

import ast
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = load_script("p003_guard")
profile = load_script("p003_profile")
load_script("build_p003_candidate")
builder = load_script("build_p003_profile")


class P003ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def csv(self, path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def fixture(self):
        competition = self.root / "competition"
        ids = [f"artificial-{number:03}" for number in range(80)]
        self.csv(competition / "train.csv", [profile.ID, "Report", "ACL"], ([uid, "DO_NOT_COPY", ""] for uid in ids))
        self.csv(
            competition / "train_series.csv",
            [profile.ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"],
            ([uid, f"{uid}-series", "Sagittal", "1", "1"] for uid in reversed(ids)),
        )
        for uid in ids:
            (competition / "train_series" / uid).mkdir(parents=True)
        return competition, ids

    def test_selection_is_order_independent_and_seeded(self):
        ids = [f"s{number}" for number in range(100)]
        selected = profile.select_uids(ids)
        self.assertEqual(len(selected), 64)
        self.assertEqual(selected, profile.select_uids(reversed(ids)))
        self.assertNotEqual(selected, profile.select_uids(ids, seed=1))
        for invalid in (ids + [ids[0]], ["../escape", *ids], ["", *ids], ids[:63]):
            with self.subTest(invalid=invalid[:2]), self.assertRaises(ValueError):
                profile.select_uids(invalid)

    def test_shadow_retains_metadata_and_train_count_but_never_reports_or_labels(self):
        competition, all_ids = self.fixture()
        shadow = self.root / "shadow"
        # Windows often lacks symlink privilege; verify exact intended links without MRI or OS privilege.
        with mock.patch.object(Path, "symlink_to", autospec=True) as link:
            ids, report = profile.create_shadow(competition, shadow)
        self.assertEqual(profile.uid_column(shadow / "train.csv"), all_ids)
        self.assertEqual(profile.uid_column(shadow / "test.csv"), ids)
        self.assertEqual(set(profile.uid_column(shadow / "test_series.csv")), set(ids))
        self.assertEqual(link.call_count, 64)
        for call, uid in zip(link.call_args_list, ids):
            self.assertEqual(call.args, (shadow / "test_series" / uid, (competition / "train_series" / uid).resolve()))
            self.assertEqual(call.kwargs, {"target_is_directory": True})
        for name in ("train.csv", "test.csv", "test_series.csv"):
            text = (shadow / name).read_text(encoding="utf-8")
            self.assertNotIn("DO_NOT_COPY", text)
            self.assertNotIn("Report", text)
            self.assertNotIn("ACL", text)
        self.assertEqual(report["train_studies"], 80)
        self.assertFalse(report["labels_used"])
        self.assertFalse(report["mri_pixels_read_during_selection"])
        self.assertFalse((shadow / "submission.csv").exists())
        with self.assertRaises(FileExistsError):
            profile.create_shadow(competition, shadow)

    def test_metadata_reports_and_missing_selected_studies_are_rejected(self):
        competition, ids = self.fixture()
        path = competition / "train_series.csv"
        self.csv(path, [profile.ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Report"], [])
        with self.assertRaisesRegex(ValueError, "Reports"):
            profile.create_shadow(competition, self.root / "shadow")
        self.csv(path, [profile.ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"], [])
        with self.assertRaisesRegex(ValueError, "metadata"):
            profile.create_shadow(competition, self.root / "shadow")
        self.assertFalse((self.root / "shadow").exists())

    def test_generated_profile_keeps_all_numeric_source_and_omits_publication(self):
        output = self.root / "generated"
        manifest = builder.build(output)
        path = output / "04_profile_p003_64.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        vendor = json.loads((ROOT / "notebooks/public/vendor/haideptry-speedy-v2.ipynb").read_text(encoding="utf-8"))
        count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            parsed = ast.parse(source)
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            if not cell["id"].startswith("p003-source-"):
                continue
            call = parsed.body[-1].value
            number, inner = map(ast.literal_eval, call.args[:2])
            self.assertNotEqual(number, 29)
            original = "".join(vendor["cells"][number]["source"])
            if number == 6:
                self.assertEqual(inner, builder.profile_root_source(original))

                def kept(value):
                    return [
                        ast.dump(node)
                        for node in ast.parse(value).body
                        if not isinstance(node, ast.FunctionDef) or node.name != "find_root"
                    ]

                self.assertEqual(kept(inner), kept(original))
                root_function = next(
                    node
                    for node in ast.parse(inner).body
                    if isinstance(node, ast.FunctionDef) and node.name == "find_root"
                )
                self.assertNotIn("/kaggle/input", ast.unparse(root_function))
            else:
                self.assertEqual(inner, original)
            self.assertEqual(hashlib.sha256(inner.encode()).hexdigest(), manifest["executed_source_cells"][str(number)])
            count += 1
        self.assertEqual(count, 22)
        self.assertEqual(manifest["changed_source_cells"], [6])
        self.assertEqual(manifest["omitted_source_cells"], [29])
        self.assertFalse(manifest["submission_publication_included"])
        other = self.root / "rebuild"
        builder.build(other)
        self.assertEqual(path.read_bytes(), (other / path.name).read_bytes())
        self.assertNotIn(b"\r\n", path.read_bytes())
        with self.assertRaises(FileExistsError):
            builder.build(output)

    def test_profile_final_outputs_and_reject_submission(self):
        contract = {
            "candidate_id": "artificial",
            "cell_order": [27],
            "source_notebook_sha256": "source",
            "source_cells": {"27": "digest"},
            "profile": {"original_source_cells": {"27": "digest", "29": "omitted"}},
        }
        runtime = profile.P003ProfileGuard(contract, work=self.root)
        runtime.ready = runtime.owns_outputs = True
        runtime.ids = [f"synthetic-{number}" for number in range(64)]
        runtime.profile_selection = {"labels_used": False}
        runtime.completed = [27]
        self.csv(
            self.root / "_pipeline_stage.csv",
            [profile.ID, *guard.LABELS],
            ([uid, *([0.5] * 12)] for uid in runtime.ids),
        )
        (self.root / "P003_PREFLIGHT.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(runtime, "check_coats"):
            result = runtime.finish({})
        self.assertEqual(result["status"], "profile_complete_not_for_submission")
        self.assertFalse(result["independent_oof"])
        self.assertTrue((self.root / "profile_predictions.csv").exists())
        self.assertTrue((self.root / "P003_PROFILE.json").exists())
        self.assertFalse((self.root / "submission.csv").exists())
        (self.root / "submission.csv").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never publish"):
            runtime.finish({})
        self.assertFalse((self.root / "submission.csv").exists())
        self.assertTrue((self.root / "P003_REJECTED_submission.csv.disabled").exists())

    def test_selection_receipt_survives_a_later_preparation_failure(self):
        competition, _ = self.fixture()
        contract = {"profile": {"count": 64, "seed": 20261005}, "cell_order": [27]}
        work = self.root / "working"
        work.mkdir()
        runtime = profile.P003ProfileGuard(contract, work=work)

        def original_preflight(instance):
            instance.mounts["competition"] = competition
            instance.owns_outputs = True

        with (
            mock.patch.object(guard.P003Guard, "prepare_files", original_preflight),
            mock.patch.object(Path, "symlink_to", autospec=True),
            mock.patch.dict(profile.os.environ),
        ):
            runtime.prepare_files()
        receipt = work / "P003_PROFILE_SELECTION.json"
        original = receipt.read_bytes()
        with mock.patch.object(guard.P003Guard, "prepare", side_effect=RuntimeError("later CUDA failure")):
            with self.assertRaisesRegex(RuntimeError, "later CUDA failure"):
                runtime.prepare()
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(json.loads(original)["studies"], 64)
        self.assertTrue((work / "P003_FAILED.json").exists())

    def test_cell_that_publishes_submission_is_quarantined_immediately(self):
        source = "from pathlib import Path\nPath(destination).write_text('unwanted')\n"
        contract = {"cell_order": [1], "source_cells": {"1": hashlib.sha256(source.encode()).hexdigest()}}
        runtime = profile.P003ProfileGuard(contract, work=self.root)
        runtime.ready = runtime.owns_outputs = True
        with self.assertRaisesRegex(ValueError, "never publish"):
            runtime.run_cell(1, source, {"destination": str(self.root / "submission.csv")})
        self.assertFalse((self.root / "submission.csv").exists())
        self.assertTrue((self.root / "P003_REJECTED_submission.csv.disabled").exists())

    def test_shadow_cleanup_only_unlinks_links_and_preserves_source(self):
        series = self.root / "_p003_profile_input" / "test_series"
        series.mkdir(parents=True)
        original = self.root / "original_mri.dcm"
        original.write_bytes(b"artificial pixels must stay")
        link = series / "link-to-study"
        link.write_text(str(original), encoding="utf-8")
        untouched = series / "not-a-link"
        untouched.write_text("retain", encoding="utf-8")
        runtime = profile.P003ProfileGuard({}, work=self.root)
        runtime.owns_outputs = True
        with mock.patch.object(Path, "is_symlink", lambda path: path == link):
            self.assertEqual(runtime.remove_image_links(), 1)
        self.assertFalse(link.exists())
        self.assertEqual(original.read_bytes(), b"artificial pixels must stay")
        self.assertEqual(untouched.read_text(encoding="utf-8"), "retain")

    def test_real_symlink_cleanup_preserves_target_when_os_permits(self):
        series = self.root / "_p003_profile_input" / "test_series"
        series.mkdir(parents=True)
        target = self.root / "original-study"
        target.mkdir()
        (target / "synthetic.dcm").write_bytes(b"synthetic")
        link = series / "study"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"OS symlink privilege unavailable: {exc}")
        runtime = profile.P003ProfileGuard({}, work=self.root)
        runtime.owns_outputs = True
        self.assertEqual(runtime.remove_image_links(), 1)
        self.assertFalse(link.exists())
        self.assertEqual((target / "synthetic.dcm").read_bytes(), b"synthetic")


if __name__ == "__main__":
    unittest.main()
