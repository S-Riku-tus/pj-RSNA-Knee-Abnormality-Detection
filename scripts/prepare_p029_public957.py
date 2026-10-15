"""Prepare the exact scored public P029 source offline; never run its code.

The copied notebook retains the author's outputs and RUN_INFERENCE=True cells.
Only Python compilation and literal source inspection occur in this builder.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts/research/20261015-helios957-source-metadata-v1"
SOURCE = SOURCE_DIR / "helios957-source-exact.ipynb"
VIEW = ROOT / (
    "artifacts/research/20261015-top957-candidates-metadata-v1/"
    "heliosli--0-9576-latest-public-dataset-score-in-one-click.json"
)
DATASET_VIEW = SOURCE_DIR / "r6-frozen-dataset-view.json"
INPUT_METADATA_DIR = ROOT / "artifacts/research/20261015-p029-public-input-metadata-v1"
INPUT_METADATA_SHA256 = {
    "Ortho-view.json": "8981aaab59ab38e4c2513619be9f9b9d3d1d28b4eba1ce303d39732abf0216ae",
    "SWA-view.json": "1045b99e09a9893b8f9a92fbc9b6ced2a34de4c5cc2f8618893ccaa6831caedd",
    "R6-v1-filelist.json": "93c0bfca1f40a8537108ebaea50fb1620fded055c7331a0a63686cf76654b335",
}
OUTPUT = ROOT / "artifacts/kaggle/p029-public957-r6-v1-20261015"
REF = "heliosli/0-9576-latest-public-dataset-score-in-one-click"
SCRIPT_VERSION = 358332966
KERNEL_ID = 138670458
SUBMISSION_ID = 57183099
SOURCE_SHA256 = "449ce7786021dc17fa671783d86d6f5ae4e10638f013c93085eadaba847cbd02"
VIEW_SHA256 = "586f61642c315e75b6df4a19899b846c652ba196a569cb5e64cd1889d03d57b0"
DATASET_VIEW_SHA256 = "16328c062580facd145c0121ff457febc8c3009ca1991d53be00b385d6d87edb"
MANIFEST_SHA256 = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"
R7_MANIFEST_SHA256 = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"
BUNDLE_ID = "fe055406fad6a12f"
DOCKER_DIGEST = "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
FIXED_INPUTS = (
    {"kind": "competition", "source_id": 154281, "databundle_version_id": 20902466,
     "mount_slug": "competitions/rsna-knee-abnormality-detection", "ref": "rsna-knee-abnormality-detection"},
    {"kind": "dataset_version", "source_id": 20297367, "dataset_id": 12357722, "version": 1,
     "mount_slug": "datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
     "ref": "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1"},
    {"kind": "dataset_version", "source_id": 20440357, "dataset_id": 12431568, "version": 1,
     "mount_slug": "datasets/nartaa/rsna-knee-publication-swa-weights-20261007",
     "ref": "nartaa/rsna-knee-publication-swa-weights-20261007/1"},
    {"kind": "dataset_version", "source_id": 20611930, "dataset_id": 12524886, "version": 1,
     "mount_slug": "datasets/heliosli/rsna-knee-r6-frozen-fe055406",
     "ref": "heliosli/rsna-knee-r6-frozen-fe055406/1", "expected_manifest_sha256": MANIFEST_SHA256},
)
WRITEFILE_HEADERS = {"%%writefile /tmp/raptor_run.py", "%%writefile /tmp/raptor224_run.py"}
R6_V1_FILES = {
    **{f"E_lrprior_s0_f{fold}.pt": 200379391 for fold in range(4)},
    "fluid_stacker.json": 5010, "manifest.json": 4120, "side_cls.pt": 85255211,
    "state_spec.json": 6857, "state_stacker.json": 24698,
}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def verified_json(path, expected_sha):
    path = Path(path)
    body = path.read_bytes()
    if digest(body) != expected_sha:
        raise ValueError(f"Pinned evidence SHA256 mismatch: {path.name}")
    return body, json.loads(body)


def _source(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def inspect_notebook(book):
    """Compile all seven cells and seven literal embedded modules without exec."""
    cells = book["cells"]
    if book["nbformat"] != 4 or len(cells) != 9:
        raise ValueError("Expected the exact nine-cell notebook")
    if [cell["cell_type"] for cell in cells] != ["markdown"] * 2 + ["code"] * 7:
        raise ValueError("Expected two markdown and seven code cells")
    compiled = []
    embedded = None
    for index, cell in enumerate(cells):
        source = _source(cell)
        entry = {"cell": index, "type": cell["cell_type"], "source_sha256": digest(source.encode("utf-8"))}
        if cell["cell_type"] == "code":
            body = source
            removed_header = None
            if source.startswith("%%writefile "):
                removed_header, body = source.split("\n", 1)
                if removed_header not in WRITEFILE_HEADERS:
                    raise ValueError("Unexpected IPython writefile destination")
            elif source.startswith("%%"):
                raise ValueError("Unreviewed IPython cell magic")
            compile(body, f"p029_original_cell_{index}", "exec")
            entry.update(compiled_without_execution=True, stripped_for_compile_only=removed_header)
            if index == 6:
                tree = ast.parse(body)
                candidates = [node.value for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets)]
                if len(candidates) != 1:
                    raise ValueError("Expected one literal SOURCES dictionary")
                embedded = ast.literal_eval(candidates[0])
        compiled.append(entry)
    expected_embedded = {"px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py",
                         "probe_dual.py", "knee_anchor_bank_v2.py", "side_common.py"}
    if not isinstance(embedded, dict) or set(embedded) != expected_embedded:
        raise ValueError("Embedded source membership differs")
    modules = []
    for name, body in embedded.items():
        if not isinstance(body, str):
            raise ValueError("Embedded source is not literal text")
        compile(body, f"p029_embedded_{name}", "exec")
        modules.append({"name": name, "source_sha256": digest(body.encode("utf-8")),
                        "compiled_without_execution": True})
    return compiled, modules


def verify_metadata(view, dataset, book):
    run = view["kernelRun"]
    if (view["kernel"]["id"], view["kernel"]["url"], view["currentVersionNumber"]) != (
        KERNEL_ID, "/code/" + REF, 1
    ):
        raise ValueError("Wrong public notebook identity/version")
    if (run["id"], run["kernelId"], run["kernelVersionNumber"], run["status"]) != (
        SCRIPT_VERSION, KERNEL_ID, 1, "COMPLETE"
    ):
        raise ValueError("Scored author run does not match the pinned V1")
    submission = view["submission"]
    if (submission["id"], submission["sourceScriptVersionId"], submission["scoreFormatted"]) != (
        SUBMISSION_ID, SCRIPT_VERSION, "0.957"
    ):
        raise ValueError("Author score is attached to a different submission/source")
    if (view["bestSubmissionScore"]["scoreFormatted"], view["bestSubmissionScore"]["kernelVersionNumber"]) != (
        "0.957", 1
    ):
        raise ValueError("Best score/version is not the pinned author result")
    if run["dockerImageVersionId"] != 31481 or run["runInfo"]["dockerImageDigest"] != DOCKER_DIGEST:
        raise ValueError("Author actual Docker digest/version differs")
    if run["isGpuEnabled"] is not True or run["acceleratorType"] != "NVIDIA_TESLA_T4":
        raise ValueError("Actual scored GPU configuration differs")
    for observed in (run["dataSources"], view["dataSources"]):
        if len(observed) != 4:
            raise ValueError("Exactly one competition and three pinned Dataset Inputs are required")
        by_id = {item["sourceId"]: item for item in observed}
        if set(by_id) != {item["source_id"] for item in FIXED_INPUTS}:
            raise ValueError("Wrong Dataset version internal ID; latest R6 V2 must not be substituted")
        for expected in FIXED_INPUTS:
            actual = by_id[expected["source_id"]]
            if actual["mountSlug"] != expected["mount_slug"]:
                raise ValueError("Input mount slug differs")
            if expected["kind"] == "dataset_version":
                if (actual.get("sourceType"), actual["datasetId"]) != (
                    "DATA_SOURCE_TYPE_DATASET_VERSION", expected["dataset_id"]
                ):
                    raise ValueError("Input Dataset identity differs")
            elif actual["databundleVersionId"] != expected["databundle_version_id"]:
                raise ValueError("Competition databundle snapshot differs")
    metadata_inputs = book["metadata"]["kaggle"]
    if metadata_inputs["isInternetEnabled"] is not False:
        raise ValueError("Original notebook Internet setting differs")
    if {item["sourceId"] for item in metadata_inputs["dataSources"]} != {
        item["source_id"] for item in FIXED_INPUTS
    }:
        raise ValueError("Notebook embedded Input IDs differ")
    if (dataset["id"], dataset["ref"], dataset["currentVersionNumber"]) != (
        12524886, "heliosli/rsna-knee-r6-frozen-fe055406", 2
    ):
        raise ValueError("Recorded latest-version hazard metadata differs")
    if MANIFEST_SHA256 not in dataset["description"] or R7_MANIFEST_SHA256 not in dataset["description"]:
        raise ValueError("Dataset description lacks the distinct R6 V1/R7 V2 hashes")
    if {item["versionNumber"] for item in dataset["versions"]} != {1, 2}:
        raise ValueError("Recorded Dataset version listing differs")


def frozen_input_verifier_source():
    """A separate metadata-only preflight; it never changes the original cells."""
    return f'''"""Check the mounted R6 V1 manifest bytes only; no model/MRI access or inference."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = {MANIFEST_SHA256!r}
EXPECTED_BUNDLE_ID = {BUNDLE_ID!r}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    body = args.manifest.read_bytes()
    if hashlib.sha256(body).hexdigest() != EXPECTED_SHA256:
        parser.exit(2, 'R6 V1 manifest SHA mismatch: do not run the notebook with latest V2/R7.\\n')
    if json.loads(body).get('bundle_id') != EXPECTED_BUNDLE_ID:
        parser.exit(2, 'R6 V1 bundle ID mismatch.\\n')
    print(json.dumps({{'R6_V1_manifest_verified': True, 'weights_or_inference_verified': False}}))

if __name__ == '__main__':
    main()
'''


def verify_public_inputs(metadata):
    for filename, expected in (("Ortho-view.json", FIXED_INPUTS[1]), ("SWA-view.json", FIXED_INPUTS[2])):
        actual = metadata[filename]
        if (actual["id"], actual["ref"], actual["currentVersionNumber"], actual["isPrivate"]) != (
            expected["dataset_id"], expected["ref"].rsplit("/", 1)[0], 1, False
        ):
            raise ValueError("Public V1 Input metadata differs")
    files = metadata["R6-v1-filelist.json"]
    if files.get("nextPageToken") or files.get("errorMessage"):
        raise ValueError("R6 V1 file listing is incomplete or failed")
    rows = files["datasetFiles"]
    if len(rows) != len(R6_V1_FILES) or {item["name"]: item["totalBytes"] for item in rows} != R6_V1_FILES:
        raise ValueError("R6 V1 file names/sizes differ")


def prepare(output_dir=OUTPUT, *, source=SOURCE, view_path=VIEW, dataset_view_path=DATASET_VIEW,
            input_metadata_dir=INPUT_METADATA_DIR):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new P029 run directory; existing outputs are preserved")
    source_bytes, book = verified_json(source, SOURCE_SHA256)
    view_bytes, view = verified_json(view_path, VIEW_SHA256)
    dataset_bytes, dataset = verified_json(dataset_view_path, DATASET_VIEW_SHA256)
    input_metadata_bytes, input_metadata = {}, {}
    for name, expected_hash in INPUT_METADATA_SHA256.items():
        raw, record = verified_json(Path(input_metadata_dir) / name, expected_hash)
        input_metadata_bytes[name], input_metadata[name] = raw, record
    cells, modules = inspect_notebook(book)
    verify_metadata(view, dataset, book)
    verify_public_inputs(input_metadata)
    verifier = frozen_input_verifier_source()
    compile(verifier, "p029_frozen_input_verifier", "exec")
    run = view["kernelRun"]
    manifest = {
        "schema_version": "p029_exact_public957_R6V1_offline_preparation_v1",
        "prepared_at_UTC": datetime.now(timezone.utc).isoformat(),
        "preparation_environment": {"python": platform.python_version(), "standard_library_only": True},
        "stage": "local_source_preparation_only",
        "source": {"ref": REF, "url": f"https://www.kaggle.com/code/{REF}?scriptVersionId={SCRIPT_VERSION}",
                   "scriptVersionId": SCRIPT_VERSION, "kernelId": KERNEL_ID, "version": 1,
                   "file": "source.ipynb", "bytes": len(source_bytes), "sha256": SOURCE_SHA256,
                   "all_notebook_bytes_cells_metadata_outputs_preserved": True,
                   "code_or_numeric_changes": False,
                   "original_RUN_INFERENCE_true_retained_but_not_executed": True,
                   "transport": "Exact notebook bytes reconstructed by Root from saved HTTP byte-array transport; local SHA bound"},
        "author_result": {"submissionId": SUBMISSION_ID, "scoreFormatted": "0.957", "status": "COMPLETE",
                          "title_claim_0_9576_is_not_verified_full_precision_score": True,
                          "own_reproduction_score": None, "author_visible_runtime_seconds": run["runInfo"]["runTimeSeconds"],
                          "visible_runtime_is_not_hidden_runtime_evidence": True},
        "fixed_inputs": list(FIXED_INPUTS),
        "version_hazard": {"Dataset_latest_version_at_observation": 2, "required_reproduction_version": 1,
                           "required_internal_source_id": 20611930, "required_bundle_id": BUNDLE_ID,
                           "required_manifest_sha256": MANIFEST_SHA256,
                           "latest_V2_is_R7_under_same_filenames": True, "rejected_R7_manifest_sha256": R7_MANIFEST_SHA256,
                           "original_source_checks_bundle_id_and_member_hashes": True,
                           "original_source_asserts_manifest_raw_SHA": False,
                           "separate_preflight_file": "verify_frozen_input.py"},
        "public_Input_metadata": {
            "Ortho_and_SWA_isPrivate_explicit_false_and_V1": True,
            "R6_V1_file_listing": R6_V1_FILES,
            "listing_is_not_weight_content_hash_verification": True,
            "Raptor384_original_checkpoint_SHA_guard": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
            "Raptor224_original_checkpoint_SHA_guard": "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6",
        },
        "runtime_plan": {"Internet": False, "GPU": "Tesla T4 x2", "CPU_execution_authorized_by_builder": False,
                         "expected_scored_docker_digest": DOCKER_DIGEST, "dockerImageVersionId": 31481,
                         "requested_docker_reference": "gcr.io/kaggle-private-byod/python@sha256:" + DOCKER_DIGEST,
                         "author_runInfo_imageName": run["runInfo"]["dockerImageName"],
                         "notebook_embedded_isGpuEnabled": book["metadata"]["kaggle"]["isGpuEnabled"],
                         "author_actual_run_isGpuEnabled": run["isGpuEnabled"],
                         "GPU_metadata_discrepancy_preserved": True,
                         "actual_future_saved_Docker_Input_GPU_settings_must_be_rechecked": True,
                         "author_notebook_python": book["metadata"]["language_info"]["version"],
                         "competition_max_runtime_hours": 9, "source_internal_DEADLINE_HOURS": 8.3,
                         "8_3h_is_not_proven_whole_notebook_hard_timeout": True,
                         "fresh_batch_container_required": True,
                         "existing_tmp_fastread_py_reuse_is_original_unhashed_behavior": True},
        "unchanged_recipe": {"Raptor384_share_of_Raptor_half": 0.65, "Raptor224_share_of_Raptor_half": 0.35,
                             "Raptor_half_weight": 0.5, "R6_weight": 0.5,
                             "all_success_formula": "0.325*pct(raw384)+0.175*pct(raw224)+0.5*pct(ours)",
                             "percentile_basis": "Each arm's own successful UIDs; ascending average ties",
                             "final_blend_is_reranked": False,
                             "fixed20_preview_is_not_P029_recipe_equivalence": True,
                             "R6_four_folds_flip_TTA": True, "MCL_uses_plain_pass": True,
                             "failure_ranking_and_thresholds": "Original source retained, not reimplemented by this builder",
                             "not_a_single_factor_comparison_against_V22R": True},
        "static_checks": {"cells": cells, "embedded_modules": modules,
                          "code_cell_count": 7, "embedded_module_count": 7,
                          "compile_only_no_exec_or_import_of_notebook_sources": True,
                          "original_writefile_header_removed_for_compile_only": True,
                          "GPU_two_T4_author_output_evidence": "Original cell6 records GPU0 and GPU1 Tesla T4 15360 MiB; cell7 historical summary records two Tesla T4s",
                          "notebook_outputs_are_historical_author_outputs_not_new_results": True},
        "credits": {"original_markdown_cells_preserved": [0, 1],
                    "required_acknowledgement_verbatim": _source(book["cells"][1]),
                    "credits_cell_source_sha256": cells[1]["source_sha256"],
                    "upstream_authors": ["heliosli", "nartaa", "dreaddevelopment", "varshithakolupuri",
                                         "stevenleehans", "pilkwang", "riadmohamed42"],
                    "OAI_NDA_Study_DOI": "10.15154/0hcg-f676"},
        "license_and_Gold": {"Dataset_license_as_reported": dataset["licenseName"],
                             "notebook_license_independently_confirmed": False,
                             "all_upstream_weight_license_compatibility_confirmed": False,
                             "R6_Gold_training_or_selection_history": "unknown",
                             "Raptor_Gold58_selection_exposure_declared_in_original_source": True,
                             "upstream_actual_training_history_independently_verified": False,
                             "Gold_clean_claimed": False, "patient_independence_claimed": False,
                             "own_Gold_read_fit_select_in_this_preparation": 0,
                             "original_credits_and_terms_retained": True},
        "steps_ja": [
            "このbundleはローカル準備だけ。source.ipynbは元の全9セル・出力・RUN_INFERENCE=Trueを保持しているので、このPCでは実行しない。",
            "Kaggleで自己rsrakiの新しい非公開候補として準備する際は、数値コードと全セルを変更せず、固定Input3件のV1と内部版IDを照合する。最新Inputへの自動置換を許可しない。",
            "R6 Datasetは必ず heliosli/rsna-knee-r6-frozen-fe055406/1・内部ID20611930を指定する。最新V2は同名ファイルのR7であり再現対象ではない。",
            "Internet OFF、Tesla T4 x2、Docker digest2757e0c7…を指定し、実際に保存された設定・Input内部版ID・ソースSHAを読み取りで再確認する。元Notebook内GPU=falseと作者実run GPU=trueの不整合は保存したまま記録する。",
            "新しいbatchコンテナを使用する。元コードは既存/tmp/_fastread.pyを再利用して再SHA検査しないため、過去の対話セッションへ継ぎ足して実行しない。8.3h定数だけで全Notebookの9時間以内完走は証明されない。",
            "推論を開始する前に別のメタデータ検査として python verify_frozen_input.py --manifest /kaggle/input/datasets/heliosli/rsna-knee-r6-frozen-fe055406/manifest.json を手動実行し、raw SHA6508ce69…とbundle_id fe055406fad6a12fの一致を確認する。旧形式mountならパスだけ実在位置へ合わせる。",
            "上流ライセンスとGold選択履歴の不確かさを保持し、元のCredits/OAI/NDA acknowledgementを維持する。自分のGoldでcheckpointや係数を選ばない。",
            "必要な実行を別途行った後は12列確率・全UID・failure/fallback・checkpointとInput SHA・出力・資源終了処理を検証する。この静的準備をnative/Serving資格へ転用しない。",
            "可視成功とhidden完走/9時間/Publicは別に確認する。正式提出は独立した手動操作とし、結果不明のSave/Submitを再送しない。作者0.957と自己再現値を区別する。"
        ],
        "qualification": {"source_prepared": True, "actual_weight_or_mounted_manifest_verified": False,
                          "own_GPU_MRI_inference_performed": False, "own_visible_Serving_qualified": False,
                          "hidden_completion_or_under9h_verified": False, "own_Public_verified": False,
                          "training_network_upload_Save_or_Submit_performed": False},
        "evidence": {"author_view": {"original_path": str(view_path), "sha256": digest(view_bytes), "bytes": len(view_bytes)},
                     "Dataset_view": {"original_path": str(dataset_view_path), "sha256": digest(dataset_bytes), "bytes": len(dataset_bytes)},
                     "public_Input_metadata": {name: {"sha256": digest(body), "bytes": len(body)}
                                               for name, body in input_metadata_bytes.items()},
                     "builder_source_sha256": digest(Path(__file__).read_bytes())},
        "generated_files": {"source.ipynb": {"sha256": SOURCE_SHA256, "bytes": len(source_bytes)},
                            "verify_frozen_input.py": {"sha256": digest(verifier.encode("utf-8"))}},
    }
    # All source, provenance and syntax checks complete before creating a run.
    output.mkdir(parents=True, exist_ok=False)
    for name, body in (("source.ipynb", source_bytes), ("author-view.json", view_bytes),
                       ("Dataset-view.json", dataset_bytes), ("verify_frozen_input.py", verifier.encode("utf-8")),
                       *input_metadata_bytes.items()):
        with (output / name).open("xb") as stream:
            stream.write(body)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    manifest_hash = digest((output / "manifest.json").read_bytes())
    with (output / "manifest.sha256").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{manifest_hash}  manifest.json\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="New directory only; never overwrite")
    args = parser.parse_args(argv)
    result = prepare(args.output_dir)
    print(json.dumps({"stage": result["stage"], "output_dir": str(args.output_dir),
                      "source_sha256": SOURCE_SHA256, "own_Public": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
