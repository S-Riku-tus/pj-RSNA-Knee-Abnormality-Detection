"""Fixed normal-score rank pooling; stdlib only, no import side effects."""
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
import struct
import sys
import time
from pathlib import Path
from statistics import NormalDist

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
WR = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def probability(value):
    result = float(value)
    require(math.isfinite(result) and 0 <= result <= 1, "Finite probability required")
    return result


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", probability(value)))[0]


def read_predictions(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == [ID, *TARGETS], "Exact submission column order required")
        rows = list(reader)
    validate_rows(rows)
    return rows


def validate_rows(rows):
    require(all(set(row) == {ID, *TARGETS} for row in rows), "Rectangular prediction rows required")
    ids = [row[ID] for row in rows]
    require(all(isinstance(uid, str) and uid and uid == uid.strip() for uid in ids)
            and len(set(ids)) == len(ids), "Unique nonempty exact UID strings required")
    for row in rows:
        for target in TARGETS:
            probability(row[target])
    return ids


def pct_on(values, ok):
    """Same average-tie percentiles as pandas, ranked only inside the mask."""
    require(len(values) == len(ok) and all(type(flag) is bool for flag in ok), "Exact boolean rank mask required")
    ordered = sorted((probability(value), index) for index, (value, valid) in enumerate(zip(values, ok)) if valid)
    result, count, start = [float("nan")] * len(values), len(ordered), 0
    while start < count:
        end = start + 1
        while end < count and ordered[end][0] == ordered[start][0]:
            end += 1
        value = ((start + 1 + end) / 2) / count
        for _, index in ordered[start:end]:
            result[index] = value
        start = end
    return result


def gaussian_on(values, ok):
    """Hazen normal scores of exact average ranks in one successful cohort."""
    require(len(values) == len(ok) and all(type(x) is bool for x in ok),
            "Exact boolean rank mask required")
    ordered = sorted((probability(v), i) for i, (v, valid) in enumerate(zip(values, ok)) if valid)
    count, start, result = len(ordered), 0, [float("nan")] * len(values)
    normal = NormalDist()
    while start < count:
        end = start + 1
        while end < count and ordered[end][0] == ordered[start][0]:
            end += 1
        # Mean one-based rank is (start + 1 + end)/2, minus 0.5.
        quantile = (start + end) / (2 * count)
        require(0 < quantile < 1, "Hazen rank must be strictly inside (0,1)")
        z = normal.inv_cdf(quantile)
        require(math.isfinite(z), "Finite normal score required")
        for _, index in ordered[start:end]:
            result[index] = z
        start = end
    return result


def transform(parent, raw384, raw224, ours, fail384, fail224, failours):
    ids = validate_rows(parent)
    n = len(ids)
    aligned = []
    for rows in (raw384, raw224, ours):
        other = validate_rows(rows)
        require(set(other) == set(ids), "Arm UID set differs from parent")
        keyed = {row[ID]: row for row in rows}
        aligned.append([keyed[uid] for uid in ids])
    raw384, raw224, ours = aligned
    for mask in (fail384, fail224, failours):
        require(len(mask) == n and all(type(flag) is bool for flag in mask),
                "Exact boolean failure masks required")
    require(fail384 == [all(float(row[c]) == .5 for c in TARGETS) for row in raw384]
            and fail224 == [all(float(row[c]) == .5 for c in TARGETS) for row in raw224],
            "Failure masks differ from original raw all0.5 convention")
    rapfail = [a and b for a, b in zip(fail384, fail224)]
    success = [not (a or b or c) for a, b, c in zip(fail384, fail224, failours)]
    result = [dict(row) for row in parent]
    changed, order_changes = 0, 0
    normal = NormalDist()
    for target in TARGETS:
        columns = [[row[target] for row in arm] for arm in (raw384, raw224, ours)]
        valid = [[not x for x in mask] for mask in (fail384, fail224, failours)]
        r384, r224, ro = [pct_on(values, ok) for values, ok in zip(columns, valid)]
        z384, z224, zo = [gaussian_on(values, ok) for values, ok in zip(columns, valid)]
        rr = [b if fa else a if fb else .65 * a + .35 * b
              for a, b, fa, fb in zip(r384, r224, fail384, fail224)]
        weight = WR.get(target, .5)
        for index in range(n):
            original = weight * rr[index] + (1 - weight) * ro[index]
            if rapfail[index]:
                original = ro[index]
            if failours[index]:
                original = rr[index]
            if rapfail[index] and failours[index]:
                original = .5
            require(struct.pack("<f", float(parent[index][target])) ==
                    struct.pack("<f", fp32(original)),
                    "Original parent FP32 reconstruction differs")
            if success[index]:
                pooled = weight * (.65 * z384[index] + .35 * z224[index]) + (1 - weight) * zo[index]
                new = fp32(normal.cdf(pooled))
                result[index][target] = repr(new)
                changed += struct.pack("<f", new) != struct.pack("<f", float(parent[index][target]))
        before = pct_on([row[target] for row in parent], [True] * n)
        after = pct_on([row[target] for row in result], [True] * n)
        order_changes += sum(a != b for a, b in zip(before, after))
    require(all(all(result[i][c] == parent[i][c] for c in TARGETS) for i in range(n) if not success[i]),
            "Every original failed-row cell string must be preserved")
    return result, {
        "rows": n, "all_three_success_rows": sum(success), "preserved_failure_rows": n - sum(success),
        "rank_populations": {"r384": n - sum(fail384), "r224": n - sum(fail224), "R6": n - sum(failours)},
        "parent_FP32_values_verified": n * len(TARGETS), "changed_FP32_values": changed,
        "changed_final_rank_cells": order_changes, "normal_score_pooling_three_branches": True,
        "quantile": "(average_rank-0.5)/N", "rr_rerank_added": False, "new_final_percentile_rank_added": False,
        "failure_row_cell_strings_preserved": True, "accuracy_verified": False}


def write_predictions(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[ID, *TARGETS])
        writer.writeheader()
        writer.writerows(rows)


def apply_namespace(ns, work, output):
    work, output = Path(work), Path(output)
    parent = read_predictions(work / "submission.csv")
    require(parent and validate_rows(parent) == ns["test_ids"] and list(ns["labels"]) == list(TARGETS),
            "Nonempty exact parent/test UID and label order required")
    require(ns["W_RAPTOR_C"] == WR and ns["W_RAPTOR"] == ns["W_OURS"] == .5 and ns["A224"] == .35,
            "Original fixed P033 weights required")
    masks = [[bool(value) for value in ns[name]] for name in ("r384_fail", "r224_fail", "our_fail")]
    require([bool(value) for value in ns["rap_fail"]] == [a and b for a, b in zip(*masks[:2])],
            "Original Raptor union mask differs")
    scored = set()
    for file in sorted(Path(ns["pred"]).glob("worker*/predictions.csv")):
        with file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    values = [float(row[c]) for c in TARGETS]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0 <= value <= 1 for value in values):
                    scored.add(row[ID])
    require(masks[2] == [uid not in scored for uid in ns["test_ids"]], "Original worker-based R6 failure mask differs")
    paths = ("raptor_raw.csv", "raptor224_raw.csv", "ours_sub.csv")
    arms = [read_predictions(work / name) for name in paths]
    shutil.copyfile(work / "submission.csv", output / "parent-submission.csv")
    shutil.copyfile(work / "blend_receipt.json", output / "parent-blend-receipt.json")
    for name, dest in zip(paths, ("raptor384-raw.csv", "raptor224-raw.csv", "ours-submission.csv")):
        shutil.copyfile(work / name, output / dest)
    result, receipt = transform(parent, *arms, *masks)
    with (output / "failure-masks.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([ID, "r384_fail", "r224_fail", "our_fail", "rap_fail"])
        writer.writerows([uid, int(a), int(b), int(c), int(a and b)] for uid, a, b, c in zip(ns["test_ids"], *masks))
    write_predictions(output / "final-submission.csv", result)
    require(read_predictions(output / "final-submission.csv") == result, "Written final CSV roundtrip differs")
    shutil.copyfile(output / "final-submission.csv", work / "submission.csv")
    receipt.update(schema_version="p043_normal_score_pooling_overlay_v1", complete=True,
                   parent=pin(output / "parent-submission.csv"), final=pin(work / "submission.csv"),
                   failure_masks=pin(output / "failure-masks.csv"), weights=WR, default_weight=.5, a224=.35)
    save(output / "overlay-receipt.json", receipt)
    return receipt


def restore_cells(encoded, compressed_sha, raw_sha, inventory):
    compressed = base64.b85decode(encoded)
    require(len(compressed) < 900000 and hashlib.sha256(compressed).hexdigest() == compressed_sha,
            "P043 compressed source differs")
    raw = lzma.decompress(compressed, memlimit=128 * 1024**2)
    require(len(raw) < 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == raw_sha, "P043 restored source differs")
    cells = json.loads(raw)
    require(len(cells) == len(inventory) == 8 and all(isinstance(x, str) for x in cells), "Exact eight parent cells required")
    for cell, expected in zip(cells, inventory):
        require({"bytes": len(cell.encode()), "sha256": hashlib.sha256(cell.encode()).hexdigest()} == expected,
                "P043 original science cell SHA differs")
    return cells


class Tee:
    def __init__(self, visible, recorded):
        self.visible, self.recorded = visible, recorded

    def write(self, value):
        self.recorded.write(value)
        return self.visible.write(value)

    def flush(self):
        self.visible.flush()


def execute_cells(cells, shell, inventory, *, work="/kaggle/working"):
    require(len(cells) == len(inventory) == 8, "Exact eight frozen P033 cells required")
    for cell, expected in zip(cells, inventory):
        require({"bytes": len(cell.encode()), "sha256": hashlib.sha256(cell.encode()).hexdigest()} == expected,
                "Executed parent science SHA differs")
    work = Path(work)
    output = work / "P043"
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": "p043_normal_score_pooling_execution_v1", "complete": False,
               "science_inventory": inventory, "completed_cells": [], "parent_cells_byte_exact": True,
               "silent_parent_fallback_permitted": False, "Save_Submit_upload": 0,
               "hidden_coverage_or_Public_gain_verified": False}
    try:
        for index, cell in enumerate(cells):
            captured = io.StringIO()
            started = time.monotonic()
            with contextlib.redirect_stdout(Tee(sys.stdout, captured)):
                result = shell.run_cell(cell, store_history=False)
            (output / ("parent-cell-%02d.stdout.txt" % index)).write_text(captured.getvalue(), encoding="utf-8")
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                raise error
            receipt["completed_cells"].append({"index": index, "seconds": time.monotonic() - started,
                                                "source_sha256": inventory[index]["sha256"]})
        receipt["overlay"] = apply_namespace(shell.user_ns, work, output)
        receipt["complete"] = True
    except Exception as error:
        receipt.update(error_type=type(error).__name__, qualification_passed=False)
        raise
    finally:
        save(output / "execution.json", receipt)
    print("P043_GAUSSIAN_RANK_COMPLETE", json.dumps(receipt["overlay"]), flush=True)
    return receipt

