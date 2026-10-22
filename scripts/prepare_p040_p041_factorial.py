"""Prepare two fixed2x2 public factors offline; never execute science or Save/Submit."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import lzma
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P033 = ROOT / "artifacts/kaggle/p033-p029m-public958-weights-20261016-v1"
P038 = ROOT / "artifacts/kaggle/p038-helios-v6-fixed-recipe-20261017-v3"
P033_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
P038_MANIFEST_SHA = "ae74c8a7a670ae8840cf269b315299657760bf8492275c2e52bd5e6fefa5886b"
OLD_VECTOR = {"Medial OA": .5, "Lateral OA": .2, "PF OA": .6, "Synovitis": .8, "Baker's": .35}
NEW_VECTOR = {"Medial OA": .5, "Lateral OA": .15, "PF OA": .65, "Synovitis": .85, "Baker's": .30}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def inventory(cells):
    return [{"bytes": len(cell.encode()), "sha256": hashlib.sha256(cell.encode()).hexdigest()} for cell in cells]


def assignment(source, name):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
    stores = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)]
    require(len(nodes) == len(stores) == 1, "Exactly one literal assignment required: " + name)
    return tree, nodes[0]


def replace_vector(source):
    tree, node = assignment(source, "W_RAPTOR_C")
    require(ast.literal_eval(node.value) == OLD_VECTOR and node.value.lineno == node.value.end_lineno,
            "Exact single-line P033 vector required")
    lines = source.splitlines(keepends=True)
    value = node.value
    line = lines[value.lineno - 1].encode()
    lines[value.lineno - 1] = (line[:value.col_offset] + repr(NEW_VECTOR).encode() + line[value.end_col_offset:]).decode()
    result = "".join(lines)
    node.value = ast.parse(repr(NEW_VECTOR), mode="eval").body
    require(ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(result), include_attributes=False),
            "Only the fixed public vector may differ")
    return result


def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source, "execution_count": None, "outputs": []}


def prepare(candidate, output, slug):
    require(candidate in ("P040", "P041"), "Fixed factorial candidate required")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh output required; previous runs are immutable")
    require(re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug), "Own private slug required")
    require(pin(P033 / "source.ipynb")["sha256"] == P033_SHA, "Pinned successful P033 source differs")
    parent, parent_manifest = read(P033 / "source.ipynb"), read(P033 / "manifest.json")
    originals = [text(c) for c in parent["cells"] if c["cell_type"] == "code"]
    require(len(originals) == 8 and [r["sha256"] for r in inventory(originals)] == parent_manifest["candidate_code_sha256"],
            "All eight frozen P033 code cells required")
    require(ast.literal_eval(assignment(originals[-1], "W_RAPTOR_C")[1].value) == OLD_VECTOR,
            "Successful original P033 class vector required")
    science = list(originals)
    reader = candidate == "P041"
    descriptor = {"candidate": candidate, "P033_source_sha256": P033_SHA, "baseline_own_Public": ".958",
                  "baseline_submission": 57290868, "own_Public": None,
                  "scientific_factor": "class_vector_only" if not reader else "public_Reader_overlay_only",
                  "class_vector": OLD_VECTOR if reader else NEW_VECTOR,
                  "P033_parent_code_cells": 8, "new_MRI_forward_qualified": False,
                  "upstream_Gold_selection_exposure_inherited": True, "own_Gold_fit_select": 0}
    require(pin(P038 / "manifest.json")["sha256"] == P038_MANIFEST_SHA, "Pinned P038 manifest differs")
    p038_manifest = read(P038 / "manifest.json")
    author_cells = read(P038 / "recipe-cells.json")
    require(len(author_cells) == 8 and inventory(author_cells) == p038_manifest["candidate_code_inventory"],
            "Frozen P038 transported science bytes differ")
    _, vector_node = assignment(author_cells[6], "W_RAPTOR_C")
    require(ast.literal_eval(vector_node.value) == NEW_VECTOR, "P038 public vector must match P040 factor")
    if reader:
        science.append(author_cells[-1])
        descriptor.update(Reader_source_SHA=inventory([author_cells[-1]])[0]["sha256"],
                          Reader_original_author_source=359117069, Reader_original_author_version=6,
                          inherited_P038_operational_preflight=True, Reader_positive_completion_required=True,
                          Reader_failure_not_parent_success=True, parent8_byte_exact=True)
    else:
        science[-1] = replace_vector(science[-1])
        descriptor.update(parent7_byte_exact=True, no_Reader_R7_native_or_224mirror_added=True)
    for cell in science:
        ast.parse(cell.split("\n", 1)[1] if cell.startswith("%%writefile ") else cell)
    runtime_path = ROOT / "scripts/p040_p041_factorial_runtime.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    ast.parse(runtime)
    runtime_sha = hashlib.sha256(runtime.encode()).hexdigest()
    runtime_code = (f"_factor_runtime_source={runtime!r}\n"
                    "import hashlib as _factor_hash\n"
                    f"assert _factor_hash.sha256(_factor_runtime_source.encode()).hexdigest()=={runtime_sha!r}\n"
                    "_factor_ns={'__name__':'p040_p041_factorial_runtime'}\n"
                    "exec(compile(_factor_runtime_source,'<fixed-factorial-runtime>','exec'),_factor_ns)\n")
    guard_config = None
    if reader:
        inherited = (P038 / "operational-runtime.py").read_text(encoding="utf-8")
        require(pin(P038 / "operational-runtime.py") == p038_manifest["operational_runtime"],
                "Inherited P038 operational runtime differs")
        guard_config = copy.deepcopy(p038_manifest["guard_config"])
        guard_config["producer_descriptor"] = descriptor
        runtime_code += (f"_factor_preflight_source={inherited!r}\n"
                         f"assert _factor_hash.sha256(_factor_preflight_source.encode()).hexdigest()=={pin(P038 / 'operational-runtime.py')['sha256']!r}\n"
                         "_factor_preflight_ns={'__name__':'p041_inherited_P038_preflight'}\n"
                         "exec(compile(_factor_preflight_source,'<p041-inherited-preflight>','exec'),_factor_preflight_ns)\n"
                         f"_factor_preflight_config={guard_config!r}\n"
                         "_factor_preflight=_factor_preflight_ns['preflight'](_factor_preflight_config)\n")
    science_inventory = inventory(science)
    config = {"candidate": candidate, "science_inventory": science_inventory, "class_vector": descriptor["class_vector"],
              "inherited_preflight": "exact_P038_runtime_CP313_Input_fixture_guards; new actual pass required" if reader else None}
    raw = json.dumps(science, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = lzma.compress(raw, preset=6)
    transport = (f"_factor_science=_factor_ns['restore_cells']({base64.b85encode(compressed).decode()!r},"
                 f"{hashlib.sha256(compressed).hexdigest()!r},{hashlib.sha256(raw).hexdigest()!r},{science_inventory!r})\n"
                 f"_factor_execution=_factor_ns['execute_cells'](_factor_science,get_ipython(),{config!r})\n")
    notice = (f"# {candidate}: fixed P033 factorial comparison\n\n"
              + ("Only four P033 W_RAPTOR_C values take the public P038 profile. Seven parent code cells remain byte-identical."
                 if not reader else "All eight P033 parent code cells remain byte-identical; only the exact P038 Reader overlay is appended. Its positive participation is mandatory.")
              + "\n\nFrozen source preparation only. Own Public is unmeasured. New visible inference, independent parent FP32"
                " and Reader FP64 replay, saved source/Input/runtime readback are required before root manual Submit."
                " Upstream Gold selection exposure remains; no own Gold fitting or selection. No automatic Save/Submit.\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": notice}, code(runtime_code), code(transport)],
                "metadata": copy.deepcopy(parent["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(serialized) < 950000, "Source capacity exceeded")
    if reader:
        inputs = copy.deepcopy(p038_manifest["inputs"])
    else:
        fixed = parent_manifest["fixed_inputs"]
        inputs = {"datasets": fixed["datasets"], "kernels": fixed["kernel"], "competitions": fixed["competition"], "total": 4}
    requested = read(P033 / "save-payload.json")
    payload = {key: copy.deepcopy(value) for key, value in requested.items() if key != "text"}
    payload.update(slug=slug, new_title=slug.split("/")[1], text=serialized.decode(),
                   dataset_data_sources=inputs["datasets"], kernel_data_sources=inputs["kernels"],
                   competition_data_sources=inputs["competitions"], enable_tpu=False)
    require(payload["is_private"] and payload["enable_gpu"] and not payload["enable_internet"], "Private GPU offline contract required")
    manifest = {"schema_version": "p040_p041_factorial_offline_source_v1", "stage": "source_preparation_only",
                "candidate": candidate, "source": {"bytes": len(serialized), "sha256": hashlib.sha256(serialized).hexdigest()},
                "descriptor": descriptor, "parent_science_inventory": inventory(originals),
                "science_inventory": science_inventory, "runtime": pin(runtime_path), "inputs": inputs,
                "requested_docker": payload["docker_image"], "inherited_preflight_source": p038_manifest["operational_runtime"] if reader else None,
                "guard_config": guard_config, "source_paths": {"P033": str((P033 / 'source.ipynb').relative_to(ROOT)),
                  "P038_recipe": str((P038 / 'recipe-cells.json').relative_to(ROOT))},
                "factorial_cells": {"baseline": "P033 oldvector/noReader", "vector": "P040 newvector/noReader",
                  "Reader": "P041 oldvector/Reader", "both": "P038 newvector/Reader"},
                "qualifications": {"static_source": True, "actual_new_visible_inference": False, "hidden": False,
                  "own_Public": False, "mounted_Input_versions": False},
                "actions": {"network": 0, "models_MRI_labels_GPU_fit": 0, "Save_Submit_upload": 0},
                "builder": pin(__file__), "frozen_P038_manifest": pin(P038 / "manifest.json"),
                "limits": ["Fixed source does not prove improvement or hidden family participation.",
                  "P038 Reader final rank is part of the inherited overlay, not an isolated AUC improvement.",
                  "P033/P038 history lacks public-parent heldout OOF; no local effect estimate or tuned weight claim.",
                  "P041 reuses operational source only; CP313/fixtures/member checks must pass again on its actual run."]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("source.ipynb", serialized), ("science-cells.json", raw), ("factorial-runtime.py", runtime.encode())):
        (output / name).write_bytes(value)
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload),
                        ("fixed-input-payload.json", {**{k: v for k, v in payload.items() if k != "text"},
                                                     "source_sha256": manifest["source"]["sha256"], "source_file": "source.ipynb"})):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", choices=("P040", "P041"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    result = prepare(args.candidate, args.output_dir, args.slug)
    print(json.dumps({"candidate": args.candidate, "source": result["source"], "Inputs": result["inputs"]["total"],
                      "actual_execution": False}))
