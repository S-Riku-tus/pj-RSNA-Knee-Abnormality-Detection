"""Stdlib observation for frozen P033 vector/Reader comparisons; import is inert."""
from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import io
import json
import lzma
import math
import shutil
import sys
import time
from pathlib import Path

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
READER_WEIGHTS = {"Baker's": .035, "Contusion": .03, "Medial OA": .025, "ACL": .025,
                  "Medial Meniscus": .02, "MCL": .02, "Lateral Meniscus": .008,
                  "Fracture": .008, "Lateral OA": .008, "PF OA": .008, "Effusion": .004,
                  "Synovitis": .003}
FILES = ("submission.csv", "blend_receipt.json", "submission_receipt.json", "raptor_raw.csv",
         "raptor224_raw.csv", "ours_sub.csv", "_public_stack.csv", "_own.csv", "parent-submission.csv", "mask.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def body_pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def restore_cells(encoded, compressed_sha, raw_sha, inventory):
    compressed = base64.b85decode(encoded)
    require(len(compressed) < 900000 and hashlib.sha256(compressed).hexdigest() == compressed_sha,
            "Factorial compressed source differs")
    raw = lzma.decompress(compressed, memlimit=128 * 1024**2)
    require(len(raw) < 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == raw_sha,
            "Factorial restored source differs")
    cells = json.loads(raw)
    require(type(cells) is list and len(cells) == len(inventory), "Fixed science cell count differs")
    for text, expected in zip(cells, inventory):
        require(type(text) is str and len(text.encode()) == expected["bytes"]
                and hashlib.sha256(text.encode()).hexdigest() == expected["sha256"], "Science cell SHA differs")
    return cells


def snapshot(work, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = {}
    for name in FILES:
        source = Path(work) / name
        if source.is_file():
            require(not source.is_symlink(), "Regular prediction/receipt required")
            shutil.copyfile(source, destination / name)
            result[name] = body_pin(destination / name)
    return result


def read_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == [ID, *TARGETS], "Exact submission label order required")
        rows = list(reader)
    require(rows and all(set(row) == {ID, *TARGETS} for row in rows), "Nonempty rectangular predictions required")
    uids = [row[ID] for row in rows]
    require(all(uids) and len(set(uids)) == len(uids), "Unique nonempty UID rows required")
    for row in rows:
        for target in TARGETS:
            value = float(row[target])
            require(math.isfinite(value) and 0 <= value <= 1, "Finite probability/rank values required")
    return rows


def parent_evidence(work, namespace):
    """Copy final parent bytes and reconstruct only the original label-free masks."""
    work = Path(work)
    parent = read_predictions(work / "submission.csv")
    uids = [row[ID] for row in parent]
    require(list(map(str, namespace["test_ids"])) == uids, "Actual parent UID order differs")
    destination = work / "parent-submission.csv"
    require(not destination.exists(), "Fresh parent observation required")
    with (work / "submission.csv").open("rb") as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output)
    scored, workers = set(), []
    for path in sorted(Path(namespace["pred"]).glob("worker*/predictions.csv")):
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    values = [float(row[target]) for target in TARGETS]
                except (KeyError, TypeError, ValueError):
                    continue
                if all(0 <= value <= 1 for value in values):
                    scored.add(row[ID])
        workers.append({"worker_file": str(path), **body_pin(path)})
    require(scored == set(namespace["scored"]), "Observed R6 scored UID set differs from exact parent science")
    masks = {}
    for name, file in (("r384_fail", "raptor_raw.csv"), ("r224_fail", "raptor224_raw.csv")):
        raw = read_predictions(work / file)
        by_uid = {row[ID]: row for row in raw}
        require(set(by_uid) == set(uids), "Raptor raw UID set differs")
        masks[name] = [all(float(by_uid[uid][target]) == .5 for target in TARGETS) for uid in uids]
    masks["our_fail"] = [uid not in scored for uid in uids]
    for name, values in masks.items():
        require([bool(value) for value in namespace[name]] == values, "Actual parent mask differs: " + name)
    record = {"schema_version": "p040_p041_label_free_parent_masks_v1", "StudyInstanceUID": uids,
              **masks, "R6_scored_UIDs": sorted(scored), "R6_worker_predictions": workers,
              "parent_CSV": body_pin(destination), "labels_or_gold_read": False}
    save(work / "mask.json", record)
    return {"parent_csv": body_pin(destination), "mask": body_pin(work / "mask.json"), "rows": len(uids)}


def ranks(values):
    n = len(values)
    result = [0.0] * n
    ordered = sorted(enumerate(values), key=lambda entry: entry[1])
    start = 0
    while start < n:
        end = start + 1
        while end < n and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1 + end) / 2) / n
        for index, _ in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def verify_reader_final(parent_path, own_path, final_path):
    parent, own, final = map(read_predictions, (parent_path, own_path, final_path))
    uids = [row[ID] for row in parent]
    require(set(uids) == {row[ID] for row in own} and [row[ID] for row in final] == uids,
            "Reader/parent/final UID identity differs")
    own_by = {row[ID]: row for row in own}
    checked = 0
    for target in TARGETS:
        left = ranks([float(row[target]) for row in parent])
        right = ranks([float(own_by[uid][target]) for uid in uids])
        w = READER_WEIGHTS[target]
        expected = ranks([(1 - w) * a + w * b for a, b in zip(left, right)])
        for row, value in zip(final, expected):
            require(float(row[target]) == value, "Reader FP64 rank replay differs")
            checked += 1
    return {"rows": len(uids), "checked_values": checked, "reader_FP64_rank_replay_exact": True,
            "parent_or_Reader_accuracy_verified": False}


class Tee:
    def __init__(self, visible, recorded):
        self.visible, self.recorded = visible, recorded

    def write(self, text):
        self.recorded.write(text)
        return self.visible.write(text)

    def flush(self):
        self.visible.flush()

    def __getattr__(self, name):
        return getattr(self.visible, name)


def execute_cells(cells, shell, config, *, work="/kaggle/working"):
    candidate, work = config["candidate"], Path(work)
    reader = candidate == "P041"
    require(candidate in ("P040", "P041") and len(cells) == (9 if reader else 8), "Fixed factorial cell layout required")
    require(len(config["science_inventory"]) == len(cells), "Declared complete science inventory required")
    for text, expected in zip(cells, config["science_inventory"]):
        require(body_pin_bytes(text.encode()) == expected, "Executed science text differs from prepared inventory")
    output = work / candidate
    output.mkdir(exist_ok=False)
    receipt = {"schema_version": "p040_p041_exact_factorial_execution_v1", "candidate": candidate,
               "complete": False, "completed_cells": [], "parent_cells": 8, "reader_required": reader,
               "science_inventory": config["science_inventory"], "own_Public": None,
               "hidden_coverage_verified": False, "accuracy_effect_claimed": False,
               "Save_Submit_calls": 0, "inherited_preflight": config.get("inherited_preflight")}
    try:
        for index, text in enumerate(cells):
            if reader and index == 8:
                require("_o_final" not in shell.user_ns, "Reader namespace must be fresh")
                receipt["parent_snapshot"] = snapshot(work, output / "parent-before-reader")
                require("submission.csv" in receipt["parent_snapshot"], "Successful P033 parent CSV required")
                read_predictions(work / "submission.csv")
            recorded = io.StringIO()
            started = time.monotonic()
            with contextlib.redirect_stdout(Tee(sys.stdout, recorded)):
                result = shell.run_cell(text, store_history=False)
            captured = recorded.getvalue()
            (output / ("cell-%02d.stdout.txt" % index)).write_text(captured, encoding="utf-8")
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise error
            row = {"index": index, "source_sha256": config["science_inventory"][index]["sha256"],
                   "elapsed_seconds": time.monotonic() - started,
                   "stdout_sha256": hashlib.sha256(captured.encode()).hexdigest()}
            if reader and index == 8:
                require("own leg: blended 3 checkpoints" in captured and "own leg FAILED" not in captured
                        and "Part B skipped" not in captured, "Positive Reader completion required; parent rollback is not qualification")
                ns = shell.user_ns
                require(ns.get("OWN_MODE") == "end" and ns.get("OWN_P33") == 0.0
                        and ns.get("OWN_W_PER_TARGET") == READER_WEIGHTS, "Exact inherited Reader overlay required")
                require(sorted(Path(p).name for p in ns.get("_o_ck", [])) == ["cnxt_v0_fold0.pt", "cnxt_v0_fold1.pt", "cnxt_v0_fold2.pt"],
                        "Three fixed Reader checkpoints must participate")
                require("_o_final" in ns, "Positive Reader final dataframe required")
                row["reader"] = verify_reader_final(output / "parent-before-reader/submission.csv",
                                                      work / "_own.csv", work / "submission.csv")
                row["reader"].update(positive_completion_marker=True,
                                     checkpoint_names=sorted(Path(p).name for p in ns["_o_ck"]),
                                     scalar_weight_metadata=.15, effective_per_target_weights=READER_WEIGHTS)
                (work / "P041-reader.stdout.txt").write_text(captured, encoding="utf-8")
                require(body_pin(work / "_public_stack.csv") == body_pin(output / "parent-before-reader/submission.csv"),
                        "Reader parent backup differs")
            if index == 7:
                row["parent_observation"] = parent_evidence(work, shell.user_ns)
            receipt["completed_cells"].append(row)
        blend = json.loads((work / "blend_receipt.json").read_bytes())
        require(blend["weights"]["raptor_per_class"] == config["class_vector"]
                and blend["weights"]["a224"] == .35, "Actual P033 factor weights differ")
        read_predictions(work / "submission.csv")
        receipt["final_snapshot"] = snapshot(work, output / "final")
        receipt["complete"] = True
    except Exception as error:
        receipt.update(error_type=type(error).__name__, qualification_passed=False)
        raise
    finally:
        save(output / "execution.json", receipt)
        save(work / (candidate + "-execution.json"), receipt)
    print(candidate + "_EXACT_FACTORIAL_COMPLETE", json.dumps({"cells": len(cells), "reader": reader,
                                                               "accuracy_verified": False}), flush=True)
    return receipt


def body_pin_bytes(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
