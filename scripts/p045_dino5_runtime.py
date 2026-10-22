"""Fixed DINO5 addition to exact P033; stdlib, import inert, no remote writes."""
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
import subprocess
import sys
import time
from pathlib import Path

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
WR = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}
DINO_WEIGHT = .10


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


def write_predictions(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[ID, *TARGETS])
        writer.writeheader()
        writer.writerows(rows)


def restore_cells(encoded, compressed_sha, raw_sha, inventory):
    compressed = base64.b85decode(encoded)
    require(len(compressed) < 900000 and hashlib.sha256(compressed).hexdigest() == compressed_sha,
            "P045 compressed source differs")
    raw = lzma.decompress(compressed, memlimit=128 * 1024**2)
    require(len(raw) < 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == raw_sha, "P045 restored source differs")
    cells = json.loads(raw)
    require(len(cells) == len(inventory) == 8 and all(isinstance(x, str) for x in cells), "Exact eight parent cells required")
    for cell, expected in zip(cells, inventory):
        require({"bytes": len(cell.encode()), "sha256": hashlib.sha256(cell.encode()).hexdigest()} == expected,
                "P045 original science cell SHA differs")
    return cells


class Tee:
    def __init__(self, visible, recorded):
        self.visible, self.recorded = visible, recorded

    def write(self, value):
        self.recorded.write(value)
        return self.visible.write(value)

    def flush(self):
        self.visible.flush()


def fuse(parent, folds, parent_failed, dino_success):
    """One fixed .10 ordering component; exact parent strings outside cohort."""
    ids = validate_rows(parent)
    n = len(ids)
    require(len(folds) == 5, "All five DINO folds required")
    require(len(parent_failed) == len(dino_success) == n
            and all(type(x) is bool for x in parent_failed + dino_success), "Exact bool masks required")
    aligned = []
    for rows in folds:
        require(set(validate_rows(rows)) == set(ids), "DINO fold UID set differs")
        by = {row[ID]: row for row in rows}
        aligned.append([by[uid] for uid in ids])
    eligible = [not failed and success for failed, success in zip(parent_failed, dino_success)]
    output = [dict(row) for row in parent]
    for target in TARGETS:
        left = pct_on([row[target] for row in parent], eligible)
        parts = [twice_rank_on([row[target] for row in rows], eligible) for rows in aligned]
        count = sum(eligible)
        average = [sum(part[i] for part in parts) / (10 * count) if eligible[i] else .5 for i in range(n)]
        right = pct_on(average, eligible)
        for i in range(n):
            if eligible[i]:
                output[i][target] = repr(.90 * left[i] + DINO_WEIGHT * right[i])
    require(all(all(parent[i][c] == output[i][c] for c in TARGETS) for i in range(n) if not eligible[i]),
            "Failure rollback must preserve every parent cell string")
    return output, {"rows": n, "eligible_rows": sum(eligible), "parent_failed_rows": sum(parent_failed),
                    "dino_failed_rows": n - sum(dino_success), "preserved_rows": n - sum(eligible),
                    "weight": DINO_WEIGHT, "folds": 5, "average_ties": True,
                    "rank_cohort": "parent3success AND DINO5success", "exact_integer_fold_rank_sum": True,
                    "final_rerank": False,
                    "failure_cell_strings_preserved": True, "accuracy_verified": False}


def twice_rank_on(values, valid):
    """Integer twice-average-ranks avoid false ties from summing rounded ranks."""
    require(len(values) == len(valid) and all(type(x) is bool for x in valid), 'Exact bool rank mask')
    ordered = sorted((probability(value), i) for i, (value, ok) in enumerate(zip(values, valid)) if ok)
    result, start = [None] * len(values), 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        for _, index in ordered[start:end]:
            result[index] = start + 1 + end
        start = end
    return result


def parent_observation(ns, work, output):
    parent = read_predictions(work / 'submission.csv')
    ids = validate_rows(parent)
    require(parent and ids == list(map(str, ns['test_ids'])) and list(ns['labels']) == list(TARGETS),
            "Parent UID and label order must match")
    require(ns['W_RAPTOR_C'] == WR and ns['W_RAPTOR'] == ns['W_OURS'] == .5 and ns['A224'] == .35,
            "Exact old parent weights required")
    masks = [[bool(x) for x in ns[name]] for name in ['r384_fail', 'r224_fail', 'our_fail']]
    require(all(len(mask) == len(ids) for mask in masks), 'Parent masks must align')
    scored = set()
    for path in sorted(Path(ns['pred']).glob('worker*/predictions.csv')):
        with path.open(encoding='utf8', newline='') as stream:
            for row in csv.DictReader(stream):
                try:
                    values = [float(row[c]) for c in TARGETS]
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    scored.add(row[ID])
    require(masks[2] == [uid not in scored for uid in ids], 'R6 actual scored UID mask differs')
    for file, dest, mask in [('raptor_raw.csv', 'raptor384-raw.csv', masks[0]),
                              ('raptor224_raw.csv', 'raptor224-raw.csv', masks[1])]:
        raw = read_predictions(work / file)
        require(set(validate_rows(raw)) == set(ids), 'Raptor raw UID set differs')
        by = {row[ID]: row for row in raw}
        require(mask == [all(float(by[uid][c]) == .5 for c in TARGETS) for uid in ids],
                'Raptor original all.5 failure convention differs')
        shutil.copyfile(work / file, output / dest)
    shutil.copyfile(work / 'ours_sub.csv', output / 'ours-submission.csv')
    shutil.copyfile(work / 'submission.csv', output / 'parent-submission.csv')
    shutil.copyfile(work / 'blend_receipt.json', output / 'parent-blend-receipt.json')
    save(output / 'parent-mask.json', {ID: ids, **dict(zip(['r384_fail', 'r224_fail', 'our_fail'], masks)),
                                      'R6_scored_UIDs': sorted(scored), 'labels_or_Gold_read': False})
    return parent, [any(values) for values in zip(*masks)]


def read_success(path, expected_ids):
    with Path(path).open(encoding='utf8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == [ID, 'success'], 'Exact DINO success columns required')
        rows = list(reader)
    require([row[ID] for row in rows] == expected_ids
            and all(row['success'] in ('0', '1') for row in rows), 'DINO success mask must align')
    return [row['success'] == '1' for row in rows]


def execute_pipeline(cells, shell, inventory, dino_source, dino_sha, config, *,
                     work='/kaggle/working', runner=None):
    require(config.get('candidate') == 'P045' and config.get('weight') == .10, 'Fixed P045 weight required')
    require(len(cells) == len(inventory) == 8, 'Exact eight parent science cells required')
    for cell, expected in zip(cells, inventory):
        require({'bytes': len(cell.encode()), 'sha256': hashlib.sha256(cell.encode()).hexdigest()} == expected,
                'Parent science source changed')
    require(hashlib.sha256(dino_source.encode()).hexdigest() == dino_sha, 'DINO source changed')
    work = Path(work)
    output = work / 'P045'
    output.mkdir(parents=True, exist_ok=False)
    program = output / 'dino-producer.py'
    program.write_text(dino_source, encoding='utf8')
    receipt = {'schema': 'P045_exact_parent_plus_DINO5_v1', 'complete': False,
               'science_inventory': inventory, 'completed_cells': [], 'dino_source_SHA': dino_sha,
               'parent8_byte_exact': True, 'own_Gold_fit_select': 0, 'upstream_history': 'UNKNOWN',
               'accuracy_verified': False, 'Save_Submit_upload': 0}
    try:
        run = subprocess.run if runner is None else runner
        with (output / 'dino.stdout.txt').open('w', encoding='utf8') as stdout, \
                (output / 'dino.stderr.txt').open('w', encoding='utf8') as stderr:
            result = run([sys.executable, str(program)], stdout=stdout, stderr=stderr,
                         timeout=8 * 3600, check=False)
        require(result.returncode == 0, 'DINO process failed; no parent-only qualification')
        dino = json.loads((output / 'dino' / 'receipt.json').read_text(encoding='utf8'))
        require(dino['complete'] and dino['folds'] == 5 and dino['models_strict_keys']
                and dino['label_order_verified'] and dino['artificial_CPU_CUDA_pass'],
                'Positive fivefold model and artificial QA evidence required')
        receipt['dino'] = dino
        print('P045_DINO_STAGE_COMPLETE', json.dumps(dino), flush=True)
        for index, cell in enumerate(cells):
            captured, started = io.StringIO(), time.monotonic()
            with contextlib.redirect_stdout(Tee(sys.stdout, captured)):
                executed = shell.run_cell(cell, store_history=False)
            text = captured.getvalue()
            (output / ('parent-cell-%02d.stdout.txt' % index)).write_text(text, encoding='utf8')
            error = executed.error_before_exec or executed.error_in_exec
            if error is not None:
                raise error
            receipt['completed_cells'].append({'index': index, 'seconds': time.monotonic() - started,
                                                'source_sha256': inventory[index]['sha256'],
                                                'stdout_sha256': hashlib.sha256(text.encode()).hexdigest()})
        parent, failures = parent_observation(shell.user_ns, work, output)
        success = read_success(output / 'dino' / 'mask.csv', validate_rows(parent))
        folds = [read_predictions(output / 'dino' / ('fold-%d.csv' % i)) for i in range(5)]
        require(sum(success) == dino['success_rows'] and len(success) == dino['rows'] and any(success),
                'At least one actual successful DINO study required')
        final, overlay = fuse(parent, folds, failures, success)
        require(overlay['eligible_rows'] > 0, 'At least one actual eligible row required')
        write_predictions(output / 'final-submission.csv', final)
        require(read_predictions(output / 'final-submission.csv') == final, 'Final CSV roundtrip differs')
        shutil.copyfile(output / 'final-submission.csv', work / 'submission.csv')
        overlay.update(complete=True, parent=pin(output / 'parent-submission.csv'), final=pin(work / 'submission.csv'),
                       success_mask=pin(output / 'dino' / 'mask.csv'))
        save(output / 'overlay-receipt.json', overlay)
        receipt['overlay'], receipt['complete'] = overlay, True
    except Exception as error:
        receipt.update(error_type=type(error).__name__, qualified=False)
        raise
    finally:
        save(output / 'execution.json', receipt)
    print('P045_DINO5_OVERLAY_COMPLETE', json.dumps(receipt['overlay']), flush=True)
    return receipt
