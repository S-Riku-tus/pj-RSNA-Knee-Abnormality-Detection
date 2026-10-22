"""Build source-only CNN/P039 inspection artifacts; never a Notebook or Save payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cnn_p039_bridge as C
import prepare_f009_p033_native20 as B

ROOT = Path(__file__).resolve().parents[1]
DISABLED_STUB = (
    "raise RuntimeError('CNN source preview is unconditionally disabled; no actual model-package runtime exists')\n"
)
REQUIREMENTS = """# SOURCE PREVIEW ONLY. No installation/extraction/import has occurred here.
# Future fresh Linux x86_64 child only; verify the exact wheel filenames/platform from Linux-wheel-lock.json.
# Keep the pinned Kaggle Torch/torchvision environment; resolve compatibility before runtime qualification.
timm==1.0.22 --hash=sha256:888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0
safetensors==0.6.2 --hash=sha256:8045db2c872db8f4cbe3faa0495932d89c38c899c603f21e9b6486951a5ecb8f
huggingface_hub==0.34.4 --hash=sha256:9b365d781739c93ff90c359844221beef048403f1bc1f1c123c191257c3c890a
"""


def prepare(output, candidate):
    """Preserve raw source bytes and references without reading trained weights."""
    spec = C.candidate_spec(candidate)
    output = Path(output).absolute()
    scope = (ROOT / "artifacts/research").resolve(strict=True)
    C.require(output.resolve().is_relative_to(scope) and output.parent.is_dir()
              and not output.exists() and not output.is_symlink(), "Fresh workspace research preview output")
    checked = C.inspect_sources(ROOT, candidate)
    control = B.inspect_control()
    router = control["components"]["serving/serving_router.py"]
    C.require(C.pin(router)["sha256"] == C.ROUTER_SHA, "Frozen original router reference")
    contract = C.future_preconditions(candidate)
    files = {"disabled_preview.py": DISABLED_STUB.encode(), "requirements.preview.txt": REQUIREMENTS.encode(),
             "cnn_p039_bridge.py": (ROOT / "scripts/cnn_p039_bridge.py").read_bytes(),
             "interface-preconditions.json": B.json_bytes(contract),
             spec["namespace"] + "/__init__.py": b""}
    files.update({spec["namespace"] + "/" + n: raw for n, raw in checked["sources"].items()})
    files.update({"metadata/" + n: raw for n, raw in checked["metadata"].items()})
    files.update({"reference/P039/" + n: raw for n, raw in control["outer_files"].items()})
    files["reference/serving/serving_router.py"] = router
    inventory = {name: B.raw_pin(raw) for name, raw in files.items()}
    archive = B.zip_sources(files)
    C.require(B.checked_zip(archive, inventory) == files, "Exact source-only archive roundtrip")
    # Recheck every immutable producer after packing; no model or trainer import.
    C.require(C.inspect_sources(ROOT, candidate)["sources"] == checked["sources"], "Producer sources unchanged during preview")
    manifest = {"schema_version": "CNN_P039_disabled_preview_pinmanifest_v1", "candidate": candidate,
                "actual_enabled": False, "runtime_constructor_implemented": False,
                "trained_weights_included": False, "trained_checkpoint_bindings_generated": False,
                "model_package_proof_consumed": False, "weak_accuracy_evaluated": False,
                "new_model_or_CUDA_or_MRI_or_training_executed": False,
                "uploadable_Notebook_or_Savepayload_generated": False, "external_API_calls": 0,
                "preview_policy_SHA": C.PREVIEW_POLICY_SHA, "producer": spec,
                "source_sha256": C.expected_sources(candidate),
                "inference_import_closure": checked["inference_import_closure"],
                "training_sources_hashed_but_never_imported": True,
                "original_P039_source_SHA": B.CONTROL_SHA, "P033_parent_SHA": B.PARENT_SHA,
                "P033_original_eight_cell_SHA": control["parent_code_sha256"],
                "original_outer_files": {n: B.raw_pin(raw) for n, raw in control["outer_files"].items()},
                "router_SHA": C.ROUTER_SHA, "Linux_wheel_lock_SHA": C.WHEEL_LOCK_SHA,
                "runtime_sources_archive": B.raw_pin(archive), "members": inventory,
                "preview_builder_SHA": B.raw_pin(Path(__file__).read_bytes())["sha256"],
                "runtime_note": "All P039 reference members are untouched archival evidence. No adapter invokes them. Disabled stub and bridge execute always refuse, even with alleged proof. Future runtime requires separate reviewed implementation."}
    output.mkdir(exist_ok=False)
    outputs = {"runtime-sources.bin": archive, "pinmanifest.json": B.json_bytes(manifest),
               "interface-preconditions.json": B.json_bytes(contract), "requirements.preview.txt": REQUIREMENTS.encode(),
               "disabled_preview.py": DISABLED_STUB.encode()}
    for name, body in outputs.items():
        with (output / name).open("xb") as stream:
            stream.write(body)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("tiny", "small"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.output, args.candidate)
    print(json.dumps({"candidate": value["candidate"], "actual_enabled": value["actual_enabled"],
                      "source_count": len(value["source_sha256"]), "archive": value["runtime_sources_archive"]}))
