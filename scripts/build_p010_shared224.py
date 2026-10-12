"""Prepare source-bound p010 candidate; no MRI, checkpoint or remote operations."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
CONTROL_SHA = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"
AUTHOR = ROOT / "artifacts/research/20261008-public-candidates/nartaa-rsna-knee-0945-efficient-224crop-v2/source.ipynb"
AUTHOR_HELPER = ROOT / "scripts/p010_author224_runtime.py"
SHARED_HELPER = ROOT / "scripts/p010_shared_runtime.py"
OUTPUT = ROOT / "artifacts/kaggle/p010-shared384-swa224-rank10-v1-20261008"
NOTEBOOK = ROOT / "notebooks/public/21_submit_p010_swa224_rank10.ipynb"
AUDIT = ROOT / "experiments/p010-shared224-preparation-20261008.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def code_cell(value):
    return {
        "cell_type": "code",
        "source": value.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
        "metadata": {},
    }


def load_author_helper():
    spec = importlib.util.spec_from_file_location("_p010_offline_author", AUTHOR_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("The source-bound insertion point changed")
    return source.replace(old, new, 1)


def shared_main_source(original, *, expert_enabled=True):
    """Preserve every original numerical definition; change only main transport."""
    enabled = bool(expert_enabled)
    source = original
    start = source.index("    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()")
    stop = source.index("    # ---- 3. prep every study ONCE", start)
    new_consumers = f"""    # p010 transport is controlled separately from the added 224 branch.
    from pathlib import Path as _p010_Path
    from p010_shared_metrics import run_bounded_pipeline, shared_input_receipt, write_submission_exclusive
    from p010_author224_metrics import make_stock_namespace, load_expert224, infer_expert224
    _p010_enabled = {enabled!r} and N >= 2
    _p010_stock = make_stock_namespace(_p010_Path("/kaggle/working/p010_src/p010_author224_stock.py").read_text(encoding="utf-8"))
    assert _p010_stock["LAB"] == LAB
    assert ngpu >= 1 and xla is None, "p010 requires the qualified CUDA path"
    _p010_experts = {{}}
    _p010_loads = {{}}
    if _p010_enabled:
        for d in devs:
            _p010_experts[d], _p010_loads[d] = load_expert224(
                find_weight_file("raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"), d,
                stock_namespace=_p010_stock)
    _p010_raw = np.full((N, len(LAB)), np.nan, np.float32) if _p010_enabled else None
    _p010_inputs = {{}}
    _p010_inferences = {{}}
    done_n = [0]; lock = threading.Lock()

    def _p010_consumer(d, item):
        sid, vol, mask = item
        i = pos[sid]
        before = shared_input_receipt(vol, mask)
        for a, (model, res) in enumerate(models[d]):
            xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM) if xla is not None else gpu_windows(vol, mask, K_EVAL, res, d)
            assert xla is None, "Anatomical L/R TTA requires CUDA/fp16"
            xm = gpu_windows(vol, mask, K_EVAL, res, d, norm=NORM, mirror=True)
            arm_probs[a][i] = infer_pair_probs(model, xw, xm, d)
            del xm, xw
        if _p010_enabled:
            _p010_raw[i], _p010_inferences[sid] = infer_expert224(_p010_stock, _p010_experts[d], vol, mask, d)
        assert shared_input_receipt(vol, mask) == before, "Shared uint8 pixels or mask changed"
        with lock:
            assert sid not in _p010_inputs, "Repeated study in shared delivery"
            _p010_inputs[sid] = before
            done_n[0] += 1
            if done_n[0] % 200 == 0 or done_n[0] == N:
                print(f"  [gpu] {{done_n[0]}}/{{N}} | {{time.time()-t0:.0f}}s", flush=True)

"""
    source = source[:start] + new_consumers + source[stop:]
    start = source.index('    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:')
    stop = source.index('    print(f"[gpu] done:', start)
    new_transport = """    _p010_transport = run_bounded_pipeline(
        lambda: mp.get_context("fork").Pool(nproc, initializer=_pre_init),
        _pre_one, test_ids, devs, _p010_consumer,
        queue_max=QUEUE_MAX, outstanding_limit=min(2 * nproc, QUEUE_MAX))
    assert done_n[0] == N and set(_p010_inputs) == set(test_ids)
    print(f"[pre] done: {N} studies in {time.time()-t_pre:.0f}s (procs {nproc} x fastread {FASTREAD_VARIANT} {FR_HDR}/{FR_PX} threads)", flush=True)
"""
    source = source[:start] + new_transport + source[stop:]
    marker = "    models.clear(); gc.collect()"
    source = replace_once(
        source,
        marker,
        marker
        + """
    _p010_experts.clear()
    gc.collect()
    torch.cuda.empty_cache()
    if _p010_enabled:
        assert np.isfinite(_p010_raw).all() and set(_p010_inferences) == set(test_ids)
        write_submission_exclusive("/kaggle/working/p010_expert_raw.csv", test_ids, _p010_raw.tolist(), LAB)
    global _p010_state
    _p010_state = {
        "uids":test_ids, "labels":LAB, "expert_enabled":_p010_enabled,
        "extra_DICOM_decodes":0, "shared_inputs":_p010_inputs, "transport":_p010_transport,
        "expert_loads":_p010_loads, "expert_inferences":_p010_inferences,
        "base_raw_probabilities":[p.tolist() for p in arm_probs],
        "expert_raw_probabilities":None if _p010_raw is None else _p010_raw.tolist(),
        "nproc_original_expression_value":nproc,
        "base_arch_resolution_K_FOV_weights_mirror_unchanged":True,
        "candidate_gate_passed":False, "accuracy_improvement_measured":False,
    }
""",
    )
    compile(source, "<p010-shared-main>", "exec")
    assert_numerical_identity(original, source)
    return source


def assert_numerical_identity(original, candidate):
    def functions(source):
        result = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main":
                start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                result[node.name] = "".join(source.splitlines(keepends=True)[start - 1 : node.end_lineno])
        return result

    assert functions(original) == functions(candidate), "An original numerical function or decorator changed"
    original_nodes = ast.parse(original).body
    candidate_nodes = ast.parse(candidate).body

    def assignments(nodes):
        return [
            ast.dump(node, include_attributes=False) for node in nodes if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]

    assert assignments(original_nodes) == assignments(candidate_nodes), "An original main constant changed"
    for marker in (
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))",
        "infer_pair_probs(model, xw, xm, d)",
        "ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs])",
    ):
        assert original.count(marker) == candidate.count(marker) == 1


def make_notebook():
    assert digest(CONTROL) == CONTROL_SHA
    author = load_author_helper()
    stock = author.build_stock_source(AUTHOR)
    baseline = json.loads(CONTROL.read_text(encoding="utf-8"))
    book = copy.deepcopy(baseline)
    helpers = [
        code_cell(
            'from pathlib import Path as _p010_Path\n_p010_Path("/kaggle/working/p010_src").mkdir(exist_ok=False)\n'
        ),
        code_cell(
            "%%writefile /kaggle/working/p010_src/p010_shared_metrics.py\n" + SHARED_HELPER.read_text(encoding="utf-8")
        ),
        code_cell(
            "%%writefile /kaggle/working/p010_src/p010_author224_metrics.py\n"
            + AUTHOR_HELPER.read_text(encoding="utf-8")
        ),
        code_cell("%%writefile /kaggle/working/p010_src/p010_author224_stock.py\n" + stock),
        code_cell("import sys as _p010_sys\n_p010_sys.path.insert(0, '/kaggle/working/p010_src')\n"),
    ]
    book["cells"] = [
        *book["cells"][:4],
        *helpers,
        code_cell(shared_main_source(text(baseline["cells"][4]))),
        *book["cells"][5:9],
        code_cell("from p010_shared_metrics import FusionObserver\n_p010_fusion_observer = FusionObserver()\n"),
        book["cells"][9],
        code_cell("""# The incumbent original fusion has finished before fixed 90:10 is applied.
import json as _p010_json
from p010_shared_metrics import finalise_fixed_rank10
_p010_fusion_guard = _p010_fusion_observer.require_success()
_p010_receipt = finalise_fixed_rank10(
    "/kaggle/working/submission.csv", "/kaggle/working/p010_expert_raw.csv",
    _p010_state["labels"], _p010_state["uids"])
_p010_receipt["transport"] = _p010_state["transport"]
_p010_receipt["expert_enabled"] = _p010_state["expert_enabled"]
_p010_receipt["extra_DICOM_decodes"] = 0
_p010_receipt["candidate_gate_passed"] = False
_p010_receipt["accuracy_improvement_measured"] = False
_p010_receipt["original_fusion_guard"] = _p010_fusion_guard
with _p010_Path("/kaggle/working/P010_SHARED_SUMMARY.json").open("x", encoding="utf-8") as _p010_f:
    _p010_json.dump(_p010_receipt, _p010_f, ensure_ascii=False, indent=2, allow_nan=False)
print(_p010_json.dumps(_p010_receipt, ensure_ascii=False, indent=2))
"""),
    ]
    book["cells"][0]["source"] = [
        "p010準備版：元の0.950予測と独立224モデルの固定90:10 rank融合。実T4互換性・搬送対照・実行時間は未確認でcandidate gateは閉じています。教師・Gold・Reportは使いません。精度改善は未測定です。\n",
        *baseline["cells"][0]["source"],
    ]
    for index in (3, 5, 6, 7, 8, 9):
        assert text(baseline["cells"][index]) in [text(cell) for cell in book["cells"]]
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = text(cell)
            compile(
                source.split("\n", 1)[1] if source.startswith("%%writefile ") else source,
                "<p010-prepared-code>",
                "exec",
            )
            cell["outputs"], cell["execution_count"] = [], None
    return book, stock


def build(output=OUTPUT, notebook_path=NOTEBOOK, audit_path=AUDIT):
    output, notebook_path, audit_path = map(Path, (output, notebook_path, audit_path))
    if any(path.exists() for path in (output, notebook_path, audit_path)):
        raise FileExistsError("Use a new package, Notebook and audit; preserve earlier evidence")
    book, stock = make_notebook()
    output.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(notebook_path, output / notebook_path.name)
    (output / "p010_author224_stock.py").write_text(stock, encoding="utf-8")
    for path in (AUTHOR_HELPER, SHARED_HELPER):
        shutil.copyfile(path, output / path.name)
    config = {
        "schema_version": "p010_shared384_swa224_fixed_rank10_prepared_v1",
        "candidate_gate_passed": False,
        "accuracy_improvement_measured": False,
        "baseline": {
            "notebook_sha256": CONTROL_SHA,
            "main_resolution": 320,
            "FOV_mm": 140,
            "K": 94,
            "main_views": 2,
            "reader_K": 16,
            "original_fusion_retained": True,
        },
        "additional_expert": {
            "checkpoint": load_author_helper().CHECKPOINT_FILENAME,
            "checkpoint_sha256": load_author_helper().CHECKPOINT_SHA256,
            "stock_source_sha256": hashlib.sha256(stock.encode()).hexdigest(),
            "resolution": 224,
            "K": 94,
            "views": 1,
            "encoder_chunk": "all_94_in_one_forward",
            "crop_native": [307, 307],
            "offset_native": [38, 38],
            "resize": "bilinear_align_corners_false",
            "extra_DICOM_decodes": 0,
            "fallback_accepted": False,
        },
        "coefficients": [0.9, 0.1],
        "findings": 12,
        "rank_ties": "average",
        "transport_safety_change_separate_factor": {
            "original_transport": "Pool.imap_unordered_unbounded_result_buffer",
            "candidate_transport": "bounded_apply_async_single_study_tasks",
            "nproc_expression_unchanged": True,
            "queue_max": 48,
            "outstanding": "min(2*nproc,48)",
            "parent_max_payloads_T4x2": 99,
            "parent_native_payload_byte_bound_T4x2": 99 * (96 * 384 * 384 + 96),
            "pool_result_handler_reserve": 1,
            "worker_IPC_native_metadata_memory_excluded_from_parent_bound": True,
            "inference_and_task_errors": "fatal_cancel_drain_finally_task_done",
            "both_delivery_and_inference_unfinished_tasks_at_shutdown": 0,
            "Python_worker_errors_covered": True,
            "OS_killed_worker_missing_callback_timeout": "not implemented or proven; inherited Pool limitation",
        },
        "author_Gold_reuse": "Author224/SWA developed or selected with Gold58; not independent OOF. OwnGold is never used to fit or select.",
        "inherited_missing_or_failed_preprocessing": "Original _pre_one semantics remain unchanged: mask-zero fixed volume on its caught decoding exception. Profile rejects FAILED log markers; no new decoder or hidden all-skip coverage claim.",
        "actual_checkpoint_payload_strict_load_T4_forward": "pending; listed file SHA is not a local payload measurement",
        "execution_provenance_guard_separate_from_prediction": "literal original fusion stdout observed; SUCCESS1 EXCEPTION0 FALLBACK0 required before rank10",
        "profile_required_before_serving": [
            "stock14_original_transport_vs_bounded_base_control",
            "bounded_base_vs_bounded_shared_base_prediction_identity",
            "stock_author224_vs_shared224_pixel_mask_raw_probability_identity",
            "same_fixed32_UID_hash_Gold_group_exclusion",
            "base_final_fusion_CSV_byte_exact_with_one_reader_execution",
            "actual_checkpoint_SHA_arch_head_strict_load",
            "224_FP16_eval_no_grad_one_forward_no_fallback",
            "both_T4_peak_and_time_hidden_scale_feasibility",
        ],
        "real_MRI_decodes_local": 0,
        "real_checkpoint_downloads": 0,
        "training_or_remote_mutations": 0,
    }
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {
        "original14_protected": digest(CONTROL) == CONTROL_SHA,
        "six_other_code_cells_literal_unchanged": True,
        "original_numerical_definitions_decorators_constants_unchanged": True,
        "all_code_cells_compile": True,
        "candidate_gate_closed": True,
    }
    record = {
        "schema_version": "p010_offline_preparation_v1",
        "status": "prepared_no_actual_model_MRI_or_remote",
        "all_passed": all(checks.values()),
        "checks": checks,
        "candidate_gate_passed": False,
        "accuracy_improvement_measured": False,
        "baseline_notebook_sha256": CONTROL_SHA,
        "candidate_notebook_sha256": digest(notebook_path),
        "author_notebook_sha256": digest(AUTHOR),
        "author_runtime_sha256": digest(AUTHOR_HELPER),
        "shared_runtime_sha256": digest(SHARED_HELPER),
        "stock_source_sha256": hashlib.sha256(stock.encode()).hexdigest(),
        "config_sha256": digest(output / "config.json"),
        "production_notebook": str(notebook_path.relative_to(ROOT)),
        "package": str(output.relative_to(ROOT)),
        "unmeasured": config["profile_required_before_serving"],
    }
    audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "package.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.notebook, args.audit), ensure_ascii=False, indent=2))
