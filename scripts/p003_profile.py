"""Label-free 64-study timing envelope for P003; never publishes a submission."""

import csv
import hashlib
import os
import re
import time
from pathlib import Path

from p003_guard import LABELS, P003Guard, read_scores, save_json, sha256

PROFILE_COUNT = 64
PROFILE_SEED = 20261005
ID = "StudyInstanceUID"


def select_uids(values, count=PROFILE_COUNT, seed=PROFILE_SEED):
    """Select by seeded SHA-256; independent of CSV row order and Python RNG."""
    values = list(values)
    if len(values) != len(set(values)) or len(values) < count or count < 1:
        raise ValueError("Need distinct train UIDs and at least the requested study count")
    if any(not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", uid) for uid in values):
        raise ValueError("Invalid or unsafe train UID")
    return sorted(values, key=lambda uid: (hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(), uid))[:count]


def uid_column(path):
    """Extract only the UID field; no report/label field is retained or inspected."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        if header.count(ID) != 1:
            raise ValueError("Expected one StudyInstanceUID column")
        index = header.index(ID)
        ids = []
        for row in rows:
            if index >= len(row):
                raise ValueError("Missing train UID field")
            ids.append(row[index])
    return ids


def write_csv(path, columns, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def create_shadow(competition, shadow, count=PROFILE_COUNT, seed=PROFILE_SEED):
    """Create metadata + directory symlinks, without reading any MRI pixels."""
    competition, shadow = Path(competition), Path(shadow)
    if shadow.exists():
        raise FileExistsError("Profile shadow must be a new directory")
    train_ids = uid_column(competition / "train.csv")
    ids = select_uids(train_ids, count, seed)
    selected = set(ids)
    metadata_path = competition / "train_series.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        required = {ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression"}
        if not required.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError("Invalid train series metadata schema")
        forbidden = {name.casefold() for name in ["Report", *LABELS]}
        if forbidden.intersection(name.casefold() for name in columns):
            raise ValueError("Reports and finding labels must not be in series metadata")
        series = [row for row in reader if row[ID] in selected]
    if {row[ID] for row in series} != selected:
        raise ValueError("Selected train UIDs do not all have series metadata")
    if any(None in row or any(value is None for value in row.values()) for row in series):
        raise ValueError("Malformed series metadata row")
    train_root = (competition / "train_series").resolve()
    targets = {}
    for uid in ids:
        target = (train_root / uid).resolve()
        target.relative_to(train_root)
        if not target.is_dir():
            raise FileNotFoundError(f"Missing selected train study directory: {uid}")
        targets[uid] = target
    shadow.mkdir(parents=True)
    (shadow / "test_series").mkdir()
    write_csv(shadow / "test.csv", [ID], ([uid] for uid in ids))
    # Preserve full train cohort size for the source's original cache planning.
    write_csv(shadow / "train.csv", [ID], ([uid] for uid in train_ids))
    write_csv(shadow / "test_series.csv", columns, ([row[name] for name in columns] for row in series))
    # These are schema placeholders, never observations, targets, or metric inputs.
    write_csv(shadow / "sample_submission.csv", [ID, *LABELS], ([uid, *([0.5] * 12)] for uid in ids))
    for uid, target in targets.items():
        (shadow / "test_series" / uid).symlink_to(target, target_is_directory=True)
    report = {
        "selection": "ascending SHA256(str(seed) + ':' + UID), then UID",
        "seed": seed,
        "studies": len(ids),
        "train_studies": len(train_ids),
        "series_rows": len(series),
        "study_uids": ids,
        "study_uids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "train_series_csv_sha256": sha256(metadata_path),
        "shadow_csv_sha256": {
            name: sha256(shadow / name)
            for name in ("train.csv", "test.csv", "test_series.csv", "sample_submission.csv")
        },
        "labels_used": False,
        "reports_used": False,
        "mri_pixels_read_during_selection": False,
        "sample_submission_values": "constant 0.5 schema placeholders, not labels",
    }
    return ids, report


class P003ProfileGuard(P003Guard):
    """Reuse every model/composition gate, replacing cohort and final publication."""

    def prepare_files(self):
        super().prepare_files()
        if self.contract["profile"]["count"] != PROFILE_COUNT or self.contract["profile"]["seed"] != PROFILE_SEED:
            raise ValueError("Profile selection contract changed")
        if self.contract["cell_order"][-1] != 27 or 29 in self.contract["cell_order"]:
            raise ValueError("Submission publication cell must not run in the profile")
        shadow = self.work / "_p003_profile_input"
        self.ids, self.profile_selection = create_shadow(self.mounts["competition"], shadow)
        os.environ["RSNA_COMP_ROOT"] = str(shadow)
        save_json(self.work / "P003_PROFILE_SELECTION.json", self.profile_selection)

    def prepare(self):
        try:
            super().prepare()
        except BaseException as exc:
            self.reject(f"Profile preparation: {type(exc).__name__}: {exc}")
            raise

    def remove_image_links(self):
        """Unlink only owned shadow links; never traverse/delete original MRI data."""
        if not self.owns_outputs:
            return 0
        series = self.work / "_p003_profile_input" / "test_series"
        removed = 0
        if series.is_symlink():
            raise ValueError("Shadow series root must not itself be a symlink")
        if series.is_dir():
            for link in series.iterdir():
                if link.is_symlink():
                    link.unlink()
                    removed += 1
        return removed

    def reject(self, reason):
        try:
            self.remove_image_links()
        finally:
            super().reject(reason)

    def no_submission(self):
        if (self.work / "submission.csv").exists():
            raise ValueError("A timing profile must never publish submission.csv")

    def run_cell(self, number, source, namespace):
        super().run_cell(number, source, namespace)
        try:
            self.no_submission()
        except BaseException as exc:
            self.reject(f"Profile publication gate: {type(exc).__name__}: {exc}")
            raise

    def finish(self, namespace):
        try:
            self.no_submission()
            if not self.ready or self.completed != self.contract["cell_order"] or len(self.ids) != PROFILE_COUNT:
                raise ValueError("Incomplete 64-study profile")
            self.check_events(namespace)
            self.check_coats(namespace)
            stage = self.work / "_pipeline_stage.csv"
            read_scores(stage, self.ids)
            destination = self.work / "profile_predictions.csv"
            report_path = self.work / "P003_PROFILE.json"
            if destination.exists() or report_path.exists():
                raise FileExistsError("Profile outputs already exist; use a fresh session")
            elapsed = time.monotonic() - self.started
            result = {
                "status": "profile_complete_not_for_submission",
                "candidate_id": self.contract["candidate_id"],
                "studies": len(self.ids),
                "selection": self.profile_selection,
                "elapsed_seconds_including_preflight": elapsed,
                "timings": self.timings,
                "source_phase_audit": namespace.get("_RSNA_AUDIT", {}),
                "coat_readers_parallel_pairs": False,
                "original_source_notebook_sha256": self.contract["source_notebook_sha256"],
                "original_source_cells": self.contract["profile"]["original_source_cells"],
                "executed_source_cells": self.contract["source_cells"],
                "omitted_source_cells": [29],
                "profile_predictions_sha256": sha256(stage),
                "input_preflight_sha256": sha256(self.work / "P003_PREFLIGHT.json"),
                "public_score": None,
                "independent_oof": False,
                "hidden_test_runtime_verified": False,
                "note": "Train images without labels; possible public-model training exposure. "
                "Measures 64 studies above the 48-study scheduling threshold. "
                "This timing does not guarantee hidden-test completion or improved accuracy.",
            }
            with destination.open("xb") as stream:
                stream.write(stage.read_bytes())
            result["shadow_image_links_removed"] = self.remove_image_links()
            save_json(report_path, result)
            self.ready = False
            print(f"P003 PROFILE COMPLETE | {len(self.ids)} train studies | {elapsed:.1f}s | not for submission")
            return result
        except BaseException as exc:
            self.reject(f"Profile final gate: {type(exc).__name__}: {exc}")
            raise
