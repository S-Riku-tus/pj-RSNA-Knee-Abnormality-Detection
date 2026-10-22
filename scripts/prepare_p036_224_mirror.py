"""Offline P033 -> P036: Raptor224 anatomical mirror probability TTA only.

Pinned source text is parsed, never imported or executed. No network, models,
MRI, GPU, authentication, remote Save or Submit operation is implemented.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/p033-p029m-public958-weights-20261016-v1"
AUTHOR = ROOT / "artifacts/research/20261017-public-frontier-v1"
OUTPUT = ROOT / "artifacts/kaggle/p036-p033-224-mirror-20261017-v1"
SLUG = "rsraki/p036-224-mirror-20261017-v1"
PARENT_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
PAYLOAD_SHA = "214e1eaf5d5adef8c0abae9dacb042ef95d5c071323e6364d6d21cde2b0e9dad"
MANIFEST_SHA = "81eaf0cfdce397fa7d582d687a58b90eae8ec078d36b19de564b03d958957e87"
AUTHOR_SHA = "0cf3f48039f0d13b7cf2a47bb7445412b04eab013184ce315f0673d91236cef5"
AUTHOR_224_SHA = "b957ee99427ab2ce125c2f272f8fc85ad3f0524fa1e1fa55184aae7d104268a8"
PARENT_224_SHA = "0b0545b46407a46d5a4ebd6a1a941b24103e62844feffda96d6fc37a48cff090"
HELPERS = ("SLOT_SIZES", "SAGITTAL_SLOTS", "_slot_index", "_mirror_tri")
OLD_ELSE = ("                    else:\n"
            "                        arm_probs[a][i] = infer_probs(model, xw, d)\n")
NEW_ELSE = ("                    else:   # [s9] average the plain and the anatomical-mirror view, as the 0.949 model does\n"
            "                        xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)\n"
            "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))\n"
            "                        del xm\n")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def pinned(path, sha):
    raw = Path(path).read_bytes()
    if digest(raw) != sha:
        raise ValueError("Pinned source/payload/manifest differs: " + Path(path).name)
    return raw, json.loads(raw)


def body(text):
    head, rest = text.split("\n", 1)
    if head != "%%writefile /tmp/raptor224_run.py":
        raise ValueError("Expected exact Raptor224 writefile destination")
    return rest


def top_node(tree, name):
    nodes = [node for node in tree.body if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )]
    if len(nodes) != 1:
        raise ValueError("Expected exactly one top-level declaration: " + name)
    return nodes[0]


def fragment(text, node):
    return "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def dump(tree):
    return ast.dump(tree, include_attributes=False)


def patch_mirror(parent_text, author_text):
    if digest(parent_text.encode()) != PARENT_224_SHA or digest(author_text.encode()) != AUTHOR_224_SHA:
        raise ValueError("Pinned Raptor224 cell differs")
    old, author = body(parent_text), body(author_text)
    old_tree, author_tree = ast.parse(old), ast.parse(author)
    if any(isinstance(n, ast.Name) and n.id in HELPERS for n in ast.walk(old_tree)):
        raise ValueError("Parent already contains mirror helpers")
    helper_nodes = [top_node(author_tree, name) for name in HELPERS]
    if ast.literal_eval(helper_nodes[0].value) != [26, 22, 18, 12, 18]:
        raise ValueError("Mirror slot sizes differ from the original 96-slice corpus")
    if ast.literal_eval(helper_nodes[1].value) != (0, 1):
        raise ValueError("Expected two sagittal slots")
    mirror_block = "\n".join(fragment(author, n).rstrip("\n") for n in helper_nodes) + "\n\n"
    old_gpu = top_node(old_tree, "gpu_windows")
    author_gpu = top_node(author_tree, "gpu_windows")
    old_gpu_text, new_gpu_text = fragment(old, old_gpu), fragment(author, author_gpu)
    changed = old.replace(old_gpu_text, new_gpu_text, 1)
    if changed.count("_WLUT = {}\n") != 1 or changed.count(OLD_ELSE) != 1:
        raise ValueError("Mirror insertion/consumer location differs")
    changed = changed.replace("_WLUT = {}\n", mirror_block + "_WLUT = {}\n", 1)
    changed = changed.replace(OLD_ELSE, NEW_ELSE, 1)
    # Prove the author implements precisely the copied branch, without taking
    # its unrelated deletion of the parent's raw probability CSV export.
    if author.count(NEW_ELSE) != 1:
        raise ValueError("Pinned author probability average branch differs")
    tree = ast.parse(changed)
    for name in (*HELPERS, "gpu_windows"):
        if dump(top_node(tree, name)) != dump(top_node(author_tree, name)):
            raise ValueError("Mirror helper differs from the scored author's source: " + name)
    if dump(top_node(tree, "infer_probs")) != dump(top_node(old_tree, "infer_probs")):
        raise ValueError("FP32 sigmoid / CUDA autocast / FP32 retry policy changed")
    # Undo all three allowed patches and require complete AST identity.
    back = changed.replace(mirror_block, "", 1).replace(new_gpu_text, old_gpu_text, 1).replace(NEW_ELSE, OLD_ELSE, 1)
    if back != old or dump(ast.parse(back)) != dump(old_tree):
        raise ValueError("Changes beyond the three permitted mirror edits")
    raw_dump = '_raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)'
    if changed.count(raw_dump) != 1:
        raise ValueError("Parent's raw probability audit export must remain")
    return parent_text.split("\n", 1)[0] + "\n" + changed


def prepare(output=OUTPUT, *, parent_dir=PARENT, author_dir=AUTHOR, slug=SLUG):
    output, parent_dir, author_dir = Path(output), Path(parent_dir), Path(author_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh P036 directory; preserve previous runs")
    if not re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Expected an own private candidate slug")
    parent_raw, parent = pinned(parent_dir / "source.ipynb", PARENT_SHA)
    _, old_payload = pinned(parent_dir / "save-payload.json", PAYLOAD_SHA)
    _, old_manifest = pinned(parent_dir / "manifest.json", MANIFEST_SHA)
    _, author = pinned(author_dir / "goodpjw958-V26-source.ipynb", AUTHOR_SHA)
    view = json.loads((author_dir / "goodpjw958-V26-view.json").read_text(encoding="utf-8"))
    if (view["kernelRun"]["id"] != 358890016 or view["kernelRun"]["kernelVersionNumber"] != 26
            or view["submission"].get("sourceScriptVersionId") != 358890016
            or view["submission"].get("id") != 57256120 or view["submission"].get("scoreFormatted") != "0.958"):
        raise ValueError("Author .958 score must bind to V26's exact source")
    if old_payload["text"].encode() != parent_raw or len(parent["cells"]) != 12:
        raise ValueError("P033 source/payload differs")
    codes = [source(c) for c in parent["cells"] if c["cell_type"] == "code"]
    if len(codes) != 8 or [digest(s.encode()) for s in codes] != old_manifest["candidate_code_sha256"]:
        raise ValueError("P033 eight code hashes differ")
    if not old_payload["is_private"] or old_payload["enable_internet"] or not old_payload["enable_gpu"]:
        raise ValueError("Private offline GPU contract required")
    changed = patch_mirror(source(parent["cells"][7]), source(author["cells"][6]))
    candidate = copy.deepcopy(parent)
    candidate["cells"][7]["source"] = changed
    candidate["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": (
        "# P036: P033 + Raptor224 anatomical mirror only\n\n"
        "Scored public author Goodpjw V26/source358890016 reports .958 for a multi-change ensemble; "
        "this candidate copies only its Raptor224 mirror implementation. Sagittal slots reverse the "
        "three uint8 slice channels; coronal/axial slots reverse image width before channel-specific "
        "normalization. Average plain/mirror FP32 probabilities after sigmoid; no label swaps. "
        "All eight P033 code cells except Raptor224 remain byte-identical. Raptor384, frozen R6 "
        "fourfold/TTA ON, nine-member guard, reader, A224=.35, class weights, raw CSV export and "
        "failure policy are inherited. No R7, Reader arm, or F006 is added. P033 formal57290868 "
        "was accepted, own score is not established in this preparation. Own proven parent .957 "
        "is separate from author .958. Upstream Gold selection uncertainty is inherited. "
        "Offline source preparation only; no model/MRI/GPU execution or remote operation.\n")})
    for item in candidate["cells"]:
        if item["cell_type"] == "code":
            item.update(execution_count=None, outputs=[])
            text = source(item)
            ast.parse(text.split("\n", 1)[1] if text.startswith("%%writefile ") else text)
    new_codes = [source(c) for c in candidate["cells"] if c["cell_type"] == "code"]
    if [i for i, (a, b) in enumerate(zip(codes, new_codes)) if a != b] != [3]:
        raise ValueError("Exactly the Raptor224 cell may change")
    serialized = (json.dumps(candidate, ensure_ascii=False, indent=1) + "\n").encode()
    payload = copy.deepcopy(old_payload)
    payload.update(slug=slug, new_title=slug.split("/")[1], text=serialized.decode())
    if {k: v for k, v in payload.items() if k not in {"slug", "new_title", "text"}} != {
            k: v for k, v in old_payload.items() if k not in {"slug", "new_title", "text"}}:
        raise ValueError("P033 Input/runtime payload changed")
    manifest = {
        "schema_version": "p036_p033_raptor224_anatomical_mirror_only_offline_v1",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "offline_prepared_no_Save_or_execution", "source_sha256": digest(serialized),
        "source_bytes": len(serialized), "parent_source_sha256": PARENT_SHA,
        "parent_payload_sha256": PAYLOAD_SHA, "parent_source_preserved": True,
        "author_source_sha256": AUTHOR_SHA, "author_scored_version": 26,
        "author_script_version_id": 358890016, "author_submission_id": 57256120,
        "author_public_score": "0.958", "author_958_is_mirror_only_evidence": False,
        "P033_own_score": None, "candidate_own_Public": None,
        "own_proven_ancestor": {"submission": 57225898, "Public": "0.957"},
        "only_scientific_change": "Raptor224 anatomical mirror TTA probability average",
        "changed_code_ordinal_zero_based": [3], "unchanged_code_ordinals": [0, 1, 2, 4, 5, 6, 7],
        "candidate_code_sha256": [digest(t.encode()) for t in new_codes],
        "mirror": {"slot_sizes": [26, 22, 18, 12, 18], "sagittal_slots": [0, 1],
                   "sagittal_axis": "triplet channel axis1", "coronal_axial_axis": "image width axis-1",
                   "applied_at": "uint8 before per-channel LUT / crop / resize / normalize",
                   "label_permutation": None, "average": "0.5*(sigmoid(logits_plain.float())+sigmoid(logits_mirror.float())) in FP32",
                   "infer_probs_unchanged": True, "raw224_csv_retained": True,
                   "XLA_path_unchanged_no_mirror": True, "extra_224_model_passes_per_successful_study": 1},
        "TTA": old_manifest["TTA"], "A224": old_manifest["A224"],
        "candidate_weights": old_manifest["candidate_weights"], "F006_included": False,
        "R7_included": False, "Reader_arm_included": False,
        "fixed_R6_guard": old_manifest["fixed_R6_guard"],
        "fixed_R6_manifest_sha256": old_manifest["fixed_R6_manifest_sha256"],
        "fixed_R6_bundle_id": old_manifest["fixed_R6_bundle_id"],
        "guard_source_sha256": old_manifest["guard_source_sha256"],
        "fixed_stage_source_sha256": old_manifest["fixed_stage_source_sha256"],
        "embedded_modules": old_manifest["embedded_modules"],
        "fixed_inputs": old_manifest["fixed_inputs"], "runtime": old_manifest["runtime"],
        "qualifications": {"static_source": True, "artificial_CPU_fixture": False,
                           "real_MRI_forward": False, "visible_replay": False,
                           "hidden_completion": False, "own_Public": False},
        "gold_note": "Inherited upstream public-model/selection exposure; no own Gold fit or selection.",
        "actions": {"network": 0, "authentication": 0, "model_load": 0, "MRI": 0,
                    "GPU": 0, "training": 0, "Save": 0, "Submit": 0},
        "builder_sha256": digest(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(serialized)
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload)):
        (output / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
    light = {key: value for key, value in payload.items() if key != "text"}
    light.update(source_sha256=digest(serialized), source_file="source.ipynb")
    (output / "fixed-input-payload.json").write_bytes((json.dumps(light, ensure_ascii=False, indent=2) + "\n").encode())
    diff = "\n".join(difflib.unified_diff(codes[3].splitlines(), changed.splitlines(), fromfile="P033-224", tofile="P036-224")) + "\n"
    (output / "224-mirror-only.diff").write_bytes(diff.encode())
    if digest((parent_dir / "source.ipynb").read_bytes()) != PARENT_SHA:
        raise ValueError("Parent changed during preparation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args()
    manifest = prepare(args.output_dir, slug=args.slug)
    print(json.dumps({"output": str(args.output_dir), "source_sha256": manifest["source_sha256"],
                      "stage": manifest["stage"], "own_Public": None}))


if __name__ == "__main__":
    main()
