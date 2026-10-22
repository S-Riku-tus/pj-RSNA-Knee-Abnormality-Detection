"""Pure frozen-source transformations for the four preregistered R6 choices."""
from __future__ import annotations

import ast
import hashlib

import prepare_p040_p041_factorial as F

CHOICES = ('tta_off', 'tta_quarter', 'no_calibration', 'crop320')
STAGE_SHA = '93b631c8b857a0f8f02671cf3e3833043f964f536cce6d46e44118d9550a6a0c'
WORKER_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
QUARTER_EDITS = (
    ('zt = 0.5 * (z[0] + zm[0][P])', 'zt = 0.75 * z[0] + 0.25 * zm[0][P]'),
    ('stt = 0.5 * (st[0] + stm[0])', 'stt = 0.75 * st[0] + 0.25 * stm[0]'),
    ('"combine": "per fold model 0.5 (z + P z_mirror), logit space"',
     '"combine": "per fold model 0.75 z + 0.25 P z_mirror, logit space"'),
)


def literal(source, name, value):
    node = F.assignment(source, name)[1].value
    F.require(node.lineno == node.end_lineno, 'Frozen single-line dictionary literal required')
    lines = source.splitlines(keepends=True)
    line = lines[node.lineno-1].encode()
    lines[node.lineno-1] = (line[:node.col_offset]+repr(value).encode()+line[node.end_col_offset:]).decode()
    changed = ''.join(lines)
    ast.parse(changed)
    return changed


def transform_stage(source, choice):
    F.require(choice in CHOICES, 'One of the four prespecified choices required')
    F.require(hashlib.sha256(source.encode()).hexdigest() == STAGE_SHA, 'Exact original P033 R6 stage required')
    sources = ast.literal_eval(F.assignment(source, 'SOURCES')[1].value)
    shas = ast.literal_eval(F.assignment(source, 'SOURCE_SHA')[1].value)
    F.require(hashlib.sha256(sources['compact_infer.py'].encode()).hexdigest() == shas['compact_infer.py'] == WORKER_SHA,
              'Frozen original R6 worker required')
    changed = source
    if choice == 'tta_off':
        a, b = 'FLIP_TTA = True', 'FLIP_TTA = False'
        F.require(changed.count(a) == 1, 'Single explicit mirror switch required')
        changed = changed.replace(a, b)
    elif choice == 'crop320':
        a, b = "'--crop', str(manifest['crop'])", "'--crop', '320'"
        F.require(changed.count(a) == 1, 'Single native crop argument required')
        changed = changed.replace(a, b)
    elif choice == 'no_calibration':
        for flag, key in [('--fluid-stacker', 'fluid_stacker'), ('--state-stacker', 'state_stacker')]:
            a = "WORKER_ARGS += ['%s', str(BUNDLE / manifest['%s'])]" % (flag, key)
            F.require(changed.count(a) == 1, 'Single postprocessor argument block required')
            changed = changed.replace(a, 'WORKER_ARGS += []')
    else:
        worker = sources['compact_infer.py']
        for a, b in QUARTER_EDITS:
            F.require(worker.count(a) == 1, 'Exact mirror arithmetic/description required')
            worker = worker.replace(a, b)
        reverse = worker
        for a, b in QUARTER_EDITS:
            F.require(reverse.count(b) == 1, 'Only registered mirror edits permitted')
            reverse = reverse.replace(b, a)
        F.require(reverse == sources['compact_infer.py'], 'All other worker bytes must be identical')
        ast.parse(worker)
        sources = dict(sources, **{'compact_infer.py': worker})
        shas = dict(shas, **{'compact_infer.py': hashlib.sha256(worker.encode()).hexdigest()})
        changed = literal(changed, 'SOURCES', sources)
        changed = literal(changed, 'SOURCE_SHA', shas)
    ast.parse(changed)
    actual = ast.literal_eval(F.assignment(changed, 'SOURCES')[1].value)
    actual_shas = ast.literal_eval(F.assignment(changed, 'SOURCE_SHA')[1].value)
    F.require(all(hashlib.sha256(text.encode()).hexdigest() == actual_shas[name] for name, text in actual.items()),
              'Every actual worker/helper hash must match its declared source')
    return changed, {'choice': choice, 'worker_SHA': actual_shas['compact_infer.py'],
                     'all_source_SHA': actual_shas, 'crop': 320 if choice == 'crop320' else 352,
                     'mirror_weight': {'tta_off': 0., 'tta_quarter': .25}.get(choice, .5),
                     'calibration': choice != 'no_calibration', 'no_new_training_or_weights': True}
