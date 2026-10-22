"""Build local pinned source transport only; no scientific imports or API calls."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1"
MODEL_SHA = "c9b5ae26d0ad5895903002ff6bfd2c4947bed152c070c749588be8c76710a714"
PINNED = {
    "scripts/orthodiffusion_mri_pilot_runtime.py": "4e8d671d6b0cf4402afac0c6c4dc3a3a007bb45878a6c40e7340159762244a1a",
    "scripts/orthodiffusion_artificial_gpu_runtime.py": "d0b432859d6fcae23192cd31a86330dc92ab3052a1b06253f9cfb7f82b5afe90",
    "scripts/orthodiffusion_regional2x2_runtime.py": "4b24207e9160a87bce2ee6fce3016ea04f19639eee1ef48f11c5d2278d5272f9",
    "src/rsna_knee/orthodiff_inner_c_numpy.py": "33af760be7c33672f780529cf42af67782dd9eac2215fbedb705b94a8a1983dd",
    "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1/config.json": "5717f2326527fd8a160eb3328fec422f0e93e8117a56718f243733b55dc58e2c",
    "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1/vendor.json": "a78dfc6a788a321068ba2c99903271b5de3abfeb2237b80407273cfda8c7ce58",
    "artifacts/research/20261017-orthodiff-innerC-test-feature-controller-design-v1/design.json": "ad3def59c37a0a9d08f5ff52ffaee0b93b4ee07725909b5df0bf889f65eac85b",
    "artifacts/research/20261017-orthodiff-innerC-test-feature-controller-design-v1/root-review.json": "ae633785e43fcf37bd41cdb682248f53f0f01565e9c074099ffabdf9350840d5",
    "artifacts/research/20261017-orthodiff-innerC-public-blend-preparation-v1/policy.json": "ba28db19a7e4d6b22e3e3a5e4624552bda8a19db575d810064eec87488c42f07",
}
SOURCES = {"base": "scripts/orthodiffusion_artificial_gpu_runtime.py",
           "pilot": "scripts/orthodiffusion_mri_pilot_runtime.py",
           "regional": "scripts/orthodiffusion_regional2x2_runtime.py"}
KEEP = {
    "base": {"BUFFER_NAMES", "PACKAGE", "sha", "require", "normalize_ema", "validate_metadata", "scoped_seed",
             "file_pin", "resolve_stage", "guard_stage", "install_vendor", "tensor_info", "write_receipt"},
    "pilot": {"PilotDataError", "check", "classify_pd", "orientation_plan", "physical_order", "decoder_preflight",
              "HEADER_TAGS", "choose_series", "read_clip", "strict_load"},
    "regional": {"fixed_regional_pool"},
}
IMPORTS = ("from __future__ import annotations\nimport hashlib,json,math,os,re,statistics,subprocess,sys\n"
           "from pathlib import Path\nfrom collections.abc import Mapping\n")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def raw_pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def pin(path):
    return raw_pin(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_bytes())


def node_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def extract_closure(source, names):
    nodes = {node_name(node): node for node in ast.parse(source).body if node_name(node) in names}
    require(set(nodes) == names, "Helper closure definition missing")
    segments = {name: ast.get_source_segment(source, node) for name, node in nodes.items()}
    text = IMPORTS + "\n\n".join(segments.values()) + "\n"
    compile(text, "<isolated-frozen-helper>", "exec")
    result_nodes = {node_name(node): node for node in ast.parse(text).body if node_name(node) in names}
    require(all(ast.dump(node, include_attributes=False) == ast.dump(result_nodes[name], include_attributes=False)
                for name, node in nodes.items()), "Extracted helper AST differs")
    return {"source": text, "pin": raw_pin(text.encode()),
            "segment_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in segments.items()}}


def numerical_parity(original, new):
    """The normalization/noise/forward/RNG assignments are unchanged, not re-fit."""
    names = ("x", "t", "seed", "cpu_rng", "cuda_rng", "before", "feature", "restored", "delta")
    old_tree, new_tree = ast.parse(original), ast.parse(new)
    forward = next(node for node in ast.walk(new_tree) if isinstance(node, ast.FunctionDef) and node.name == "forward")
    proof = {}
    for name in names:
        old = [node for node in ast.walk(old_tree) if isinstance(node, ast.Assign) and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
        new_nodes = [node for node in ast.walk(forward) if isinstance(node, ast.Assign) and len(node.targets) == 1
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
        require(len(new_nodes) == 1, "New scientific assignment missing/ambiguous: " + name)
        normalized = ast.dump(new_nodes[0], include_attributes=False)
        require(any(ast.dump(node, include_attributes=False) == normalized for node in old),
                "Normalization/forward/RNG assignment differs: " + name)
        proof[name] = hashlib.sha256(normalized.encode()).hexdigest()
    # Match the full call expression, including the selected CUDA generator.
    old_calls = {ast.dump(node, include_attributes=False) for node in ast.walk(old_tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"}
    new_calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"]
    require(len(new_calls) == 1 and ast.dump(new_calls[0], include_attributes=False) in old_calls,
            "Selected CUDA seed call differs")
    new_with = next(node for node in ast.walk(forward) if isinstance(node, ast.With)
                    and "fork_rng" in ast.unparse(node.items[0].context_expr))
    new_dump = ast.dump(new_with, include_attributes=False)
    matching_contexts = []
    for node in ast.walk(old_tree):
        if not isinstance(node, ast.With) or "fork_rng" not in ast.unparse(node.items[0].context_expr):
            continue
        candidate = copy.deepcopy(node)
        candidate.body = [statement for statement in candidate.body if not (
            isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "pooled")]
        if ast.dump(candidate, include_attributes=False) == new_dump:
            matching_contexts.append(candidate)
    require(len(matching_contexts) == 1, "FP32 feature/RNG context differs beyond removing unused GAP")
    return proof


def build(output, enable_actual_entry=False):
    require(type(enable_actual_entry) is bool, "Explicit Boolean execution intent required")
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Fresh source transport directory required")
    before = {name: pin(ROOT / name) for name in PINNED}
    require(all(value["sha256"] == PINNED[name] for name, value in before.items()), "Frozen source/design/policy differs")
    original = {name: (ROOT / path).read_text(encoding="utf-8") for name, path in SOURCES.items()}
    closures = {name: extract_closure(original[name], KEEP[name]) for name in SOURCES}
    runtime_raw = (ROOT / "scripts/orthodiffusion_inner_c_test_runtime.py").read_bytes()
    adapter_raw = (ROOT / "src/rsna_knee/orthodiff_inner_c_numpy.py").read_bytes()
    parity = numerical_parity(original["regional"], runtime_raw.decode())
    parent = read(PARENT / "config.json")
    assets = {key: copy.deepcopy(parent["assets"][key]) for key in (
        "input_kernel", "stage_manifest_pin", "assets", "stage_candidates", "code_commit", "HF_revision")}
    vendor = read(PARENT / "vendor.json")
    bundle = {"schema_version": "orthodiff_innerC_dynamic_test_transport_v1",
              "actual_entry_enabled": enable_actual_entry, "model_sha256": MODEL_SHA,
              "feature_dimension": 1024, "base_seed": 20261017,
              "transport_sources": {"runtime": raw_pin(runtime_raw), "builder": pin(__file__)},
              "helper_closures": closures, "adapter": {"source": adapter_raw.decode(), "pin": raw_pin(adapter_raw)},
              "assets": assets, "decoder": copy.deepcopy(parent["decoder"]), "vendor": vendor,
              "scientific_sources": before, "normalization_forward_AST_parity": parity,
              "public_arithmetic_policy_sha256": "ba28db19a7e4d6b22e3e3a5e4624552bda8a19db575d810064eec87488c42f07",
              "parent_selected": None, "Public_submission_contract_qualified": False,
              "requires_separate_actual_model_root_outer2_runtime_and_bundle_qualification": True}
    raw = (json.dumps(bundle, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) < 4 * 1024**2, "Bounded source-only bundle required")
    output.mkdir(parents=True)
    (output / "bundle.json").write_bytes(raw)
    (output / "runtime.py").write_bytes(runtime_raw)
    (output / "adapter.py").write_bytes(adapter_raw)
    manifest = {"schema_version": "orthodiff_innerC_test_source_preparation_v1", "bundle": raw_pin(raw),
                "runtime": raw_pin(runtime_raw), "adapter": raw_pin(adapter_raw), "builder": pin(__file__),
                "pinned_old_sources_unchanged": before == {name: pin(ROOT / name) for name in PINNED},
                "actual_entry_enabled": enable_actual_entry, "actual_model_MRI_T4_qualified": False,
                "normalization_forward_assignments_AST_identical": len(parity),
                "helper_members_AST_identical": sum(len(value["segment_sha256"]) for value in closures.values()),
                "Public_parent_selected": False, "Public_candidate_adopted": False,
                "Kaggle_payload_or_API_Save_Submit": 0,
                "remaining_future_work": ["outer2/root adoption gate", "actual source-bound qualification",
                    "actual T4/visible preprocessing and arbitrary batch numerical verification",
                    "chosen parent assets and outer child timeout/reap/9h orchestration; no launcher in this pack"]}
    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-actual-entry", action="store_true",
                        help="Future source intent only; actual root/model/outer2/runtime qualification is still mandatory")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output, args.enable_actual_entry)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
