"""Offline P039: own P033 .958 parent plus unchanged P035 native success20.

Only local source/metadata bytes are read. No model/MRI/labels, scientific
source execution, GPU, network, Save, upload or Submit is performed.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
import lzma
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/kaggle/p035-p029m-f006-isolated20-20261017-v2"
PARENT = ROOT / "artifacts/kaggle/p033-p029m-public958-weights-20261016-v1"
PAYLOAD = ROOT / "artifacts/research/20261017-p035-execution-v1/launch-v2/serving/payload.json"
OLD_SHA = "0b67f4d7b8131b5855421e456b108c89b6a407e8a680441fd6f1b11d555ab9ea"
PARENT_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
WEIGHTS = {"raptor": .5, "ours": .5, "raptor_per_class": {
    "Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}, "a224": .35}
OLD_WEIGHTS_LITERAL = '{\n                "raptor": 0.5, "ours": 0.5,\n                "raptor_per_class": {"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}, "a224": 0.35,\n            }'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def pin(path):
    return raw_pin(Path(path).read_bytes())


def source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def codes(book):
    return [source(cell) for cell in book["cells"] if cell["cell_type"] == "code"]


def read(path):
    return json.loads(Path(path).read_bytes())


def unpack_old():
    manifest, book = read(OLD / "manifest.json"), read(OLD / "source.ipynb")
    require(pin(OLD / "source.ipynb")["sha256"] == OLD_SHA == manifest["source_sha256"], "Exact P035 source required")
    stores = []
    for text in codes(book):
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "restore_runtime":
                stores.append([ast.literal_eval(arg) for arg in node.value.args])
    require(len(stores) == 1, "Single original compressed runtime required")
    encoded, digest = stores[0]
    raw = lzma.decompress(base64.b85decode(encoded))
    require(raw_pin(raw)["sha256"] == digest == manifest["runtime_bundle_sha256"], "P035 runtime SHA differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    require(set(files) == set(manifest["runtime_files"]), "P035 runtime inventory differs")
    require(all(raw_pin(raw) == manifest["runtime_files"][name] for name, raw in files.items()), "P035 runtime member differs")
    return book, manifest, files


def replace_once(text, before, after):
    require(text.count(before) == 1, "Single exact source boundary required: " + before[:70])
    return text.replace(before, after, 1)


def new_extension(old, parent_code_shas):
    helpers = '''P039_PARENT_SOURCE_SHA = __SOURCE_SHA__
P039_PARENT_CODE_SHA = __CODE_SHA__
P039_PARENT_WEIGHTS = __WEIGHTS__


def p039_require_native_contribution(numerical):
    if (numerical.get("all_native_disabled") is not False
            or type(numerical.get("native_blended_cell_count")) is not int
            or numerical["native_blended_cell_count"] <= 0
            or numerical.get("blended_target_count", 0) <= 0
            or numerical.get("alpha_requested") != 0.2):
        raise RuntimeError("P039 invalid extension: no active fixed20 native cells; parent-only success forbidden")


def p039_bind_parent_metadata(numerical):
    # Metadata only; the frozen blender's rows/arithmetic/CSV are untouched.
    for target, entry in numerical["per_target"].items():
        wr = P039_PARENT_WEIGHTS["raptor_per_class"].get(target, P039_PARENT_WEIGHTS["raptor"])
        components = {"Raptor384": round((1 - .35) * wr, 12),
                      "Raptor224": round(.35 * wr, 12), "CompactR6": round(1 - wr, 12)}
        active = entry["status"] in {"blended_full_N", "blended_success_subset"}
        parent_factor, native_factor = (.8, .2) if active else (1., 0.)
        entry.pop("conditional_P029M_component_weights_nominal", None)
        entry["conditional_P033_component_weights_nominal"] = {
            "successful_cells": {**{name: round(value * parent_factor, 12) for name, value in components.items()},
                                 "F006_native_rank": native_factor},
            "failed_native_cells": {**components, "F006_native_rank": 0.},
            "condition": "all parent arms available; nominal algebra only, not FP32 flattened equality"}
    numerical["arithmetic"]["parent_component_weights"] = "P033 per-target nominal algebra, conditional on all parent arms available; not flattened FP32 equality"
    numerical["P033_parent_binding"] = {"source_sha256": P039_PARENT_SOURCE_SHA,
        "code_cell_sha256": P039_PARENT_CODE_SHA, "weights_receipt": P039_PARENT_WEIGHTS}
    return numerical


'''.replace("__SOURCE_SHA__", repr(PARENT_SHA)).replace("__CODE_SHA__", repr(parent_code_shas)).replace("__WEIGHTS__", repr(WEIGHTS))
    value = replace_once(old, "def sha(path):\n", helpers + "def sha(path):\n")
    value = replace_once(value, OLD_WEIGHTS_LITERAL, "P039_PARENT_WEIGHTS")
    value = replace_once(value, '"schema_version": "p032_p029m_f006_fixed20_execution_v1"',
                         '"schema_version": "p039_p033_f006_fixed20_execution_v1", "parent_source_sha256": P039_PARENT_SOURCE_SHA')
    value = replace_once(value, '"P029M must complete a regular submission.csv first"',
                         '"P033 must complete a regular submission.csv first"')
    value = replace_once(value, '"Original parent blend weight receipt differs"',
                         '"Exact P033 parent blend weight receipt differs"')
    injection = '''        numerical = p039_bind_parent_metadata(numerical)
        # Replace only new-run descriptive metadata; never alter numeric CSV.
        (blended / "receipt.json").write_text(json.dumps(numerical, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
        (blended / "receipt.sha256").write_text(sha(blended / "receipt.json") + "  receipt.json\\n", encoding="utf-8")
        p039_require_native_contribution(numerical)
'''
    value = replace_once(value, '        receipt["numeric_blend"] = numerical\n', injection + '        receipt["numeric_blend"] = numerical\n')
    value = replace_once(value, 'accuracy_comparison_status="NOT_AN_ACCURACY_COMPARISON_NATIVE_DISABLED"',
                         'accuracy_comparison_status="INVALID_EXTENSION_CANDIDATE_RUN_FAILED"')
    value = replace_once(value, 'receipt.update(parent_fallback_preserved=True, submission_csv_sha256=sha(final))',
                         'receipt.update(parent_artifact_restored=True, parent_only_submission_allowed=False, submission_csv_sha256=sha(final))')
    value = replace_once(value, '        return receipt\n    finally:\n',
                         '        raise  # A restored parent is an artifact, never a successful P039 candidate.\n    finally:\n')
    compile(value, "p039-bound-parent-extension.py", "exec")
    return value


def zip_sources(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 17, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    return out.getvalue()


def cell(text):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def prepare(output, slug="rsraki/p039-p033-native20-20261017-v1"):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh P039 directory required")
    require(slug.startswith("rsraki/p039-") and len(slug.split("/")) == 2
            and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in slug.split("/")[1]), "Fixed own P039 slug required")
    old_book, old_manifest, files = unpack_old()
    old_files = dict(files)
    parent_raw = (PARENT / "source.ipynb").read_bytes()
    require(raw_pin(parent_raw)["sha256"] == PARENT_SHA, "Successful P033 source differs")
    parent = json.loads(parent_raw)
    old_parent = json.loads(files["P029M-source.ipynb"])
    parent_code, old_code = codes(parent), codes(old_parent)
    require(len(parent_code) == len(old_code) == 8 and parent_code[:7] == old_code[:7], "Only final parent cell may differ")
    before, after = ast.parse(old_code[-1]), ast.parse(parent_code[-1])
    def weights(tree):
        return next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "W_RAPTOR_C")
    require(ast.literal_eval(weights(after).value) == WEIGHTS["raptor_per_class"], "Exact own958 class vector required")
    weights(before).value = copy.deepcopy(weights(after).value)
    require(ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), "Parent science differs beyond class vector")
    parent_shas = [raw_pin(text.encode())["sha256"] for text in parent_code]
    extension = new_extension(files["p032_parent_extension.py"].decode(), parent_shas)
    files["p032_parent_extension.py"] = extension.encode()
    files.pop("P029M-source.ipynb")
    files["P033-source.ipynb"] = parent_raw
    unchanged = {name: raw_pin(raw) for name, raw in old_files.items()
                 if name not in {"P029M-source.ipynb", "p032_parent_extension.py"}}
    require(all(files[name] == old_files[name] for name in unchanged) and len(unchanged) == 7,
            "Original native/executor/reader/blender bytes must remain exact")
    bundle = zip_sources(files)
    encoded = base64.b85encode(lzma.compress(bundle, preset=9)).decode()
    setup = ("exec(compile(" + repr(extension) + ", 'p032_parent_extension.py', 'exec'), globals())\n"
             "_p032_runtime = restore_runtime(" + repr(encoded) + ", " + repr(raw_pin(bundle)["sha256"]) + ")\n")
    old_parent_shas = [raw_pin(text.encode())["sha256"] for text in old_code]
    old_runner = codes(old_book)[2]
    runner = replace_once(old_runner, "'P029M-source.ipynb'", "'P033-source.ipynb'")
    runner = replace_once(runner, repr(raw_pin(old_files["P029M-source.ipynb"])["sha256"]), repr(PARENT_SHA))
    runner = replace_once(runner, repr(old_parent_shas), repr(parent_shas))
    runner = runner.replace("unchanged P029M", "unchanged P033")
    extension_runner = codes(old_book)[3] + '''if (_p032_result.get("complete") is not True or _p032_result.get("native_complete") is not True
        or _p032_result.get("native_contribution_used") is not True or _p032_result.get("all_native_disabled") is not False
        or _p032_result.get("operational_recovery_contract_passed") is not True):
    raise RuntimeError("P039 extension incomplete/disabled: candidate run must fail, never parent-only success")
'''
    notice = {"cell_type": "markdown", "metadata": {}, "source": "# P039: confirmed own P033 .958 plus unchanged isolated native fixed20\n\nOnly the parent fixed class vector changes scientific predictions relative to P035. Four native models, source89, original reader, isolated UID180s, union failure mask and FP32 success-subset20 are unchanged. Parent binding/metadata is explicit P033. Whole extension failure/timeout/receipt mismatch/all-disabled raises; a restored parent is a rollback artifact, not a successful submission. Individual failed cells keep exact P033 text. Own P035 formal score remains unknown; .959/.960 is not promised. No training, Gold selection or automatic API submission.\n"}
    book = copy.deepcopy(old_book)
    book["cells"] = [notice, *[copy.deepcopy(c) for c in parent["cells"] if c["cell_type"] == "markdown"],
                     cell(codes(old_book)[0]), cell(setup), cell(runner), cell(extension_runner)]
    for text in codes(book):
        compile(text, "P039-code", "exec")
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(raw) <= 950000, "P039 source exceeds capacity")
    payload = read(PAYLOAD)
    require(json.loads(payload["text"]) == old_book, "Original manual P035 payload does not match source")
    payload.update(slug=slug, new_title=slug.split("/")[1], text=raw.decode())
    require(len(payload["dataset_data_sources"] + payload["kernel_data_sources"] + payload["competition_data_sources"]) == 9
            and payload["is_private"] is True and payload["enable_gpu"] is True and payload["enable_internet"] is False,
            "Identical private/offline9 Input GPU transport required")
    policy = {"schema_version": "p039_P033_parent_existing_native20_prospective_v1",
              "declared_after": "own official P03357290868 COMPLETE .958; P03557292346 raw score/status absent",
              "scientific_factor_only": "P029M W_RAPTOR_C replaced by exact successful P033 literal",
              "W_RAPTOR_C": WEIGHTS["raptor_per_class"], "A224": .35, "R6_TTA": True,
              "native_models_math_reader_executor_masks_fixed": True, "native_weight": .2,
              "whole_extension_failure_or_disabled": "raise; parent snapshot is rollback artifact only",
              "all_UIDs_native_ineligible_or_no_active_cells": "run failure; not a successful parent-only submission",
              "individual_UID_failure": "unchanged success-subset rank and exact P033 cell text",
              "no_native_gain_or_Public_target_guarantee": True, "parameter_or_target_search": False,
              "P035_Public_or_hidden_native_participation_known": False,
              "new_candidate_visible_GPU_or_hidden_or_accuracy_qualified": False}
    manifest = {"schema_version": "p039_p033_existing_native20_source_only_preparation_v1",
                "source": raw_pin(raw), "parent_source": raw_pin(parent_raw), "old_P035_source_sha256": OLD_SHA,
                "parent_code_sha256": parent_shas, "parent_scientific_change_only_W_RAPTOR_C": True,
                "parent_weights_receipt": WEIGHTS, "runtime_bundle": raw_pin(bundle),
                "runtime_files": {name: raw_pin(value) for name, value in files.items()},
                "unchanged_runtime_members": unchanged, "native": old_manifest["native"],
                "inputs": old_manifest["inputs"], "platform": old_manifest["platform"],
                "integration": {**old_manifest["integration"],
                    "native_child_timeout_or_exception_retains_parent_CSV_and_records_no_comparison": False,
                    "whole_extension_failure_restores_parent_artifact_but_raises": True,
                    "all_native_disabled_is_fatal": True},
                "unchanged_scientific_math": {key: old_manifest["integration"][key] for key in (
                    "native_weight", "parent_reranked", "final_reranked", "FP32_separate_multiply_add",
                    "native_rank_population", "failed_native_cells_retain_parent_text", "partial_constant_native_contributes")},
                "engineering_delta": {"P033_parent_bindings_and_nominal_metadata": True,
                    "whole_extension_parent_only_success_disabled": True, "all_native_disabled_is_fatal": True,
                    "legacy_individual_UID_recovery_unchanged": True},
                "builder": pin(__file__), "scope": {"source_and_artificial_only": True,
                    "new_model_MRI_GPU_network_Save_Submit": 0, "new_visible_hidden_Public_qualified": False}}
    output.mkdir(parents=True)
    values = {"source.ipynb": raw, "runtime-bundle.zip": bundle, "p039-parent-extension.py": extension.encode(),
              "save-payload.json": (json.dumps(payload, indent=2) + "\n").encode(),
              "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
              "policy.json": (json.dumps(policy, indent=2) + "\n").encode()}
    for name, value in values.items():
        (output / name).write_bytes(value)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", default="rsraki/p039-p033-native20-20261017-v1")
    args = parser.parse_args(argv)
    manifest = prepare(args.output, args.slug)
    print(json.dumps({"source": manifest["source"], "unchanged_runtime_members": len(manifest["unchanged_runtime_members"]),
                      "visible_qualified": False, "Save_Submit": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
