"""Offline P042 preparation. No science execution, models, data, network or Save."""
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
POLICY = ROOT / "artifacts/research/20261019-p042-rank-renormalization-preparation-v1/prospective-policy.json"
POLICY_SHA = "91df6a64decaa4203a3864f7939ee4f01c7c08fe215df12f449024071b98f5ba"


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
    require(not output.exists() and not output.is_symlink(), "Fresh exclusive P042 output required")
    require(re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug), "Own private slug required")
    require(pin(PARENT / "source.ipynb")["sha256"] == PARENT_SHA, "Successful P033 source SHA differs")
    require(pin(POLICY)["sha256"] == POLICY_SHA, "Prospective fixed P042 policy SHA differs")
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
    runtime_path = ROOT / "scripts/p042_rank_renormalization_runtime.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    ast.parse(runtime)
    runtime_pin = pin(runtime_path)
    loader = (f"_p042_runtime_source={runtime!r}\n"
              "import hashlib as _p042_hash\n"
              f"assert _p042_hash.sha256(_p042_runtime_source.encode()).hexdigest()=={runtime_pin['sha256']!r}\n"
              "_p042_namespace={'__name__':'p042_rank_runtime'}\n"
              "exec(compile(_p042_runtime_source,'<p042-fixed-runtime>','exec'),_p042_namespace)\n")
    raw = json.dumps(science, ensure_ascii=False, separators=(",", ":")).encode()
    compressed = lzma.compress(raw, preset=6)
    execution = (f"_p042_science=_p042_namespace['restore_cells']({base64.b85encode(compressed).decode()!r},"
                 f"{hashlib.sha256(compressed).hexdigest()!r},{hashlib.sha256(raw).hexdigest()!r},{science_inventory!r})\n"
                 f"_p042_execution=_p042_namespace['execute_cells'](_p042_science,get_ipython(),{science_inventory!r})\n")
    notice = ("# P042: intermediate Raptor-half rank normalization\n\n"
              "All eight P033 code cells and inputs stay byte-exact. The only new scientific operation is an average-tie"
              " percentile rank of fallback-aware rr over Raptor-half success rows, followed by the same fixed class"
              " weights and R6 rank. New values apply only to all-three-family success rows; each original failure"
              " row retains its parent twelve CSV cell strings. Parent FP32 replay and positive overlay completion"
              " are mandatory. No parameter search, final rank, new model, training, Gold or Report use."
              " Public gain and hidden coverage are unmeasured. Inherited upstream Gold58 selection exposure remains.\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": notice}, code(loader), code(execution)],
                "metadata": copy.deepcopy(parent["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    serialized = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(serialized) < 950000, "P042 source capacity exceeded")
    payload = copy.deepcopy(json.loads((PARENT / "save-payload.json").read_bytes()))
    payload.update(slug=slug, new_title=slug.split("/")[1], text=serialized.decode(), enable_tpu=False)
    require(payload["is_private"] and payload["enable_gpu"] and not payload["enable_internet"],
            "Private GPU offline parent runtime required")
    fixed = parent_manifest["fixed_inputs"]
    require(payload["dataset_data_sources"] == fixed["datasets"]
            and payload["kernel_data_sources"] == fixed["kernel"]
            and payload["competition_data_sources"] == fixed["competition"], "All exact parent Input refs required")
    manifest = {"schema_version": "p042_intermediate_rank_offline_preparation_v1",
                "prepared_at_UTC": datetime.now(timezone.utc).isoformat(), "stage": "source_preparation_only",
                "candidate": "P042", "slug": slug, "source": {"bytes": len(serialized), "sha256": hashlib.sha256(serialized).hexdigest()},
                "parent_source": {"path": str((PARENT / 'source.ipynb').relative_to(ROOT)), **pin(PARENT / "source.ipynb")},
                "policy": {"path": str(POLICY.relative_to(ROOT)), **pin(POLICY)}, "parent_science_inventory": science_inventory,
                "science_inventory": science_inventory, "science_AST_inventory": ast_inventory,
                "source_AST_difference": "All8 original parent science AST/bytes unchanged; add separate post-parent intermediate rr rank overlay only.",
                "runtime": {"path": str(runtime_path.relative_to(ROOT)), **runtime_pin},
                "inputs": {"datasets": fixed["datasets"], "kernels": fixed["kernel"], "competitions": fixed["competition"], "total": 4},
                "runtime_settings": {key: payload[key] for key in ("is_private", "enable_gpu", "enable_tpu", "enable_internet", "machine_shape", "docker_image")},
                "outputs_for_readonly_audit": ["P042/parent-submission.csv", "P042/final-submission.csv", "P042/raptor384-raw.csv",
                   "P042/raptor224-raw.csv", "P042/ours-submission.csv", "P042/failure-masks.csv", "P042/parent-blend-receipt.json",
                   "P042/overlay-receipt.json", "P042/execution.json"],
                "qualification": {"static_source": True, "real_visible_inference": False, "parent_FP32_actual_replay": False,
                                  "new_FP32_actual_replay": False, "hidden": False, "own_Public": False},
                "no_silent_parent_fallback": True, "actual_Save_Submit_GPU_models_MRI_labels_fit": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("source.ipynb", serialized), ("science-cells.json", raw), ("p042-runtime.py", runtime.encode())):
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
    print(json.dumps({"candidate": "P042", "source": result["source"], "Inputs": result["inputs"]["total"],
                      "actual_execution": False}))
