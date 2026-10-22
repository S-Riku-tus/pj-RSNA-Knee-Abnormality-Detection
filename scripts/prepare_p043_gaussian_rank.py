"""Offline P043 preparation. No science execution, models, data, network or Save."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import lzma
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/p033-p029m-public958-weights-20261016-v1"
PARENT_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
POLICY = ROOT / "artifacts/research/20261020-goal959-root-resume-and-P043-policy-v1/P043-prospective-policy.json"
POLICY_SHA = "d368e69086c7c0a9d779cd4f95b714bae3882c8019ce7f56b7a5f1ad0d4c9a57"


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def inventory(cells):
    return [{"bytes": len(cell.encode()), "sha256": hashlib.sha256(cell.encode()).hexdigest()} for cell in cells]


def code(value):
    return {"cell_type": "code", "metadata": {}, "source": value, "execution_count": None, "outputs": []}


def prepare(output, slug):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh exclusive P043 output required")
    require(re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug), "Own private slug required")
    require(pin(PARENT / "source.ipynb")["sha256"] == PARENT_SHA, "Successful P033 source SHA differs")
    require(pin(POLICY)["sha256"] == POLICY_SHA, "Prospective fixed P043 policy SHA differs")
    parent = json.loads((PARENT / "source.ipynb").read_bytes())
    parent_manifest = json.loads((PARENT / "manifest.json").read_bytes())
    science = [text(cell) for cell in parent["cells"] if cell["cell_type"] == "code"]
    science_inventory = inventory(science)
    require(len(science) == 8 and [item["sha256"] for item in science_inventory] == parent_manifest["candidate_code_sha256"],
            "All eight original science cells must remain byte exact")
    ast_inventory = []
    for index, cell in enumerate(science):
        body = cell.split("\n", 1)[1] if cell.startswith("%%writefile ") else cell
        tree = ast.dump(ast.parse(body), include_attributes=False)
        ast_inventory.append({"index": index, "AST_sha256": hashlib.sha256(tree.encode()).hexdigest(),
                              "bytes_SHA_unchanged": science_inventory[index]})
    runtime_path = ROOT / "scripts/p043_gaussian_rank_runtime.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    ast.parse(runtime)
    runtime_pin = pin(runtime_path)
    loader = (f"_p043_runtime_source={runtime!r}\n"
              "import hashlib as _p043_hash\n"
              f"assert _p043_hash.sha256(_p043_runtime_source.encode()).hexdigest()=={runtime_pin['sha256']!r}\n"
              "_p043_namespace={'__name__':'p043_rank_runtime'}\n"
              "exec(compile(_p043_runtime_source,'<p043-fixed-runtime>','exec'),_p043_namespace)\n")
    raw = json.dumps(science, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = lzma.compress(raw, preset=6)
    execution = (f"_p043_science=_p043_namespace['restore_cells']({base64.b85encode(compressed).decode()!r},"
                 f"{hashlib.sha256(compressed).hexdigest()!r},{hashlib.sha256(raw).hexdigest()!r},{science_inventory!r})\n"
                 f"_p043_execution=_p043_namespace['execute_cells'](_p043_science,get_ipython(),{science_inventory!r})\n")
    notice = ("# P043: fixed Gaussian rank pooling\n\n"
              "All eight P033 code cells and exact Inputs/settings/models stay byte-exact. "
              "Only the pooling operator changes: for each of the three original image branches, "
              "normal quantiles of Hazen average ranks replace percentiles before the same "
              "A224=.35 and per-target weights; normal CDF then FP32 writes the final output. "
              "No coefficient fitting, rr re-ranking or final percentile rank. "
              "Any original branch failure preserves the parent twelve CSV cell strings. "
              "Parent FP32 replay and positive overlay completion are mandatory. "
              "No own Gold/Report use; inherited public Gold-selection exposure remains. "
              "This bounded hypothesis has no verified accuracy or hidden coverage gain.\n")
    inherited_notes = [copy.deepcopy(cell) for cell in parent["cells"] if cell["cell_type"] == "markdown"]
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": notice}, code(loader), code(execution),
                           {"cell_type": "markdown", "metadata": {},
                            "source": "## Inherited P033 attribution and historical source notes\n\n"
                                      "The following unchanged notes belong to the frozen parent source.\n"},
                           *inherited_notes],
                "metadata": copy.deepcopy(parent["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(serialized) < 950000, "P043 source capacity exceeded")
    payload = copy.deepcopy(json.loads((PARENT / "save-payload.json").read_bytes()))
    payload.update(slug=slug, new_title=slug.split("/")[1], text=serialized.decode(), enable_tpu=False)
    require(payload["is_private"] and payload["enable_gpu"] and not payload["enable_internet"],
            "Private GPU offline parent runtime required")
    fixed = parent_manifest["fixed_inputs"]
    require(payload["dataset_data_sources"] == fixed["datasets"]
            and payload["kernel_data_sources"] == fixed["kernel"]
            and payload["competition_data_sources"] == fixed["competition"], "All exact parent Input refs required")
    manifest = {"schema_version": "p043_normal_score_pooling_offline_preparation_v1",
                "prepared_at_UTC": datetime.now(timezone.utc).isoformat(), "stage": "source_preparation_only",
                "candidate": "P043", "slug": slug, "source": {"bytes": len(serialized), "sha256": hashlib.sha256(serialized).hexdigest()},
                "parent_source": {"path": str((PARENT / 'source.ipynb').relative_to(ROOT)), **pin(PARENT / "source.ipynb")},
                "policy": {"path": str(POLICY.relative_to(ROOT)), **pin(POLICY)}, "parent_science_inventory": science_inventory,
                "science_inventory": science_inventory, "science_AST_inventory": ast_inventory,
                "source_AST_difference": "All8 original parent science AST/bytes unchanged; add separate fixed normal-score pooling overlay only.",
                "inherited_parent_markdown_sha256": [hashlib.sha256(text(cell).encode()).hexdigest() for cell in inherited_notes],
                "runtime": {"path": str(runtime_path.relative_to(ROOT)), **runtime_pin},
                "inputs": {"datasets": fixed["datasets"], "kernels": fixed["kernel"], "competitions": fixed["competition"], "total": 4},
                "runtime_settings": {key: payload[key] for key in ("is_private", "enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image")},
                "outputs_for_readonly_audit": ["P043/parent-submission.csv", "P043/final-submission.csv", "P043/raptor384-raw.csv",
                   "P043/raptor224-raw.csv", "P043/ours-submission.csv", "P043/failure-masks.csv", "P043/parent-blend-receipt.json",
                   "P043/overlay-receipt.json", "P043/execution.json"],
                "qualification": {"static_source": True, "real_visible_inference": False, "parent_FP32_actual_replay": False,
                                  "new_FP32_actual_replay": False, "hidden": False, "own_Public": False},
                "no_silent_parent_fallback": True, "actual_Save_Submit_GPU_models_MRI_labels_fit": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("source.ipynb", serialized), ("science-cells.json", raw), ("p043-runtime.py", runtime.encode())):
        (output / name).write_bytes(value)
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload),
                        ("fixed-input-payload.json", {**{key: value for key, value in payload.items() if key != "text"},
                                                      "source_sha256": manifest["source"]["sha256"], "source_file": "source.ipynb"})):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    result = prepare(args.output_dir, args.slug)
    print(json.dumps({"candidate": "P043", "source": result["source"], "Inputs": result["inputs"]["total"],
                      "actual_execution": False}))

