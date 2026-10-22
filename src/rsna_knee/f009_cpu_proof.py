"""Strict CPU E12 reconstruction producer; actual weights are manual future work.

Only pinned completed two-fold evaluation metadata is read before checkpoint or
generic model access. No forward, optimizer, MRI, teacher, GPU or network route.
Artificial fixture reconstruction is private and always emits actual=False.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

from . import f009_source_namespace as sources

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "artifacts/research/20261017-f009-strict-CPU-producer-preparation-v1/f009_pair_stage-v1.py"
SNAPSHOT_SHA = "d3a951a2913798d77c7e93977997bff68ad334173cc7b8c0382e96d9f5772781"
GENERIC_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _contracts():
    """Frozen v1 guards/shapes, independent of a future consumer registry v2."""
    path = ROOT / SNAPSHOT
    require(sha(path) == SNAPSHOT_SHA, "Frozen consumer v1 source changed")
    spec = importlib.util.spec_from_file_location("rsna_knee._f009_cpu_frozen_stage_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_gates_only(contract, root, spec):
    """Reuse exact v1 metadata gates up to, but excluding, its CPU consumer loop.

    No placeholder CPU proof is created. The producer has no CPU proof yet.
    The copied prefix is pinned by the entire original v1 source SHA.
    """
    tree = ast.parse((ROOT / SNAPSHOT).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preflight")
    end = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "plan" for t in n.targets)
    )
    fn = copy.deepcopy(fn)
    fn.name = "_completed_pair_metadata_only"
    fn.body = fn.body[:end] + [
        ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id="base", ctx=ast.Load()), ast.Name(id="config", ctx=ast.Load())], ctx=ast.Load()
            )
        )
    ]
    scope = dict(vars(contract))
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(ROOT / SNAPSHOT), "exec"), scope
    )
    converted = {key: value for key, value in spec.items() if key != "provenance"}
    converted["schema_version"] = "f009_pair_stage_root_pins_v1"
    return scope[fn.name](root, converted)


def prepare_inputs(root, spec, fold):
    """Metadata only: no checkpoint stat/hash, source import, or model load."""
    root = Path(root).resolve()
    require(type(fold) is int and fold in (0, 1), "Exact fold0/1 required")
    require(
        spec["schema_version"] == "f009_CPU_proof_root_inputs_v1"
        and set(spec) == {"schema_version", "result0", "guard0", "result1", "guard1", "folds", "provenance"}
        and all(set(value) == {"run", "index"} for value in spec["folds"].values()),
        "Explicit CPU producer inputs; no existing CPU receipt permitted",
    )
    c = _contracts()
    base, config = _pair_gates_only(c, root, spec)
    policy0, policy1 = c.read(root / c.POLICY0), c.read(root / c.POLICY1)
    guards = {n: c.read(c.bound(root, spec["guard" + str(n)]))["guards"] for n in (0, 1)}
    contexts = {}
    for n in (0, 1):
        item = spec["folds"][str(n)]
        run_path, index_path = (c.bound(root, item[key]) for key in ("run", "index"))
        expected = root / (policy0["planned_run_directory"] if n == 0 else policy1["new_run"])
        require(
            run_path.resolve() == expected.resolve() / "run.json"
            and index_path.resolve() == expected.resolve() / "epoch_checkpoint_index.json",
            "Fixed F009 run paths required",
        )
        run, index = c.read(run_path), c.read(index_path)
        prior = guards[n]["completed_E12_control_candidate"]["candidate"]
        require(
            c.pin(run_path) == prior["run"]
            and c.pin(index_path) == prior["index"]
            and run["status"] == "complete"
            and type(run["fold"]) is int
            and run["fold"] == n
            and run["seed"] == 20261002
            and run["config"] == config
            and run["source_sha256"] == sources.CORE_SHA
            and run["targets"] == c.TARGETS
            and run["gold_used_for_training_or_selection"] is False
            and run["gold_evaluated"] is False
            and [r["epoch"] for r in run["history"]] == list(range(1, 13))
            and all(r["amp_skipped_steps"] == 0 for r in run["history"]),
            "Complete exact F009 fold/E12 lineage differs",
        )
        require(index["fold"] == n and index["source_sha256"] == sources.CORE_SHA, "Retained index source/fold differs")
        entries = [row for row in index["checkpoints"] if row["epoch"] == 12]
        require(len(entries) == 1, "Exactly one E12 row required")
        row = entries[0]
        require(
            row["checkpoint"] == "epoch_012.pt"
            and row["predictions_csv"] == "epoch_012_weak.csv"
            and row["weak_bce"] == run["history"][-1]["weak_bce"]
            and prior["E12_checkpoint_body_stream_hash"]
            == {"bytes": row["checkpoint_bytes"], "sha256": row["checkpoint_sha256"]}
            and prior["E12_prediction"]["sha256"]
            == row["prediction_sha256"]
            == run["history"][-1]["prediction_sha256"],
            "Unconditional E12 checkpoint/index/history/CSV bindings differ",
        )
        contexts[n] = {
            "run": run,
            "row": row,
            "run_path": run_path,
            "index_path": index_path,
            "run_pin": prior["run"],
            "retained_index_pin": prior["index"],
            "checkpoint_path": expected / "epoch_012.pt",
            "checkpoint_pin": prior["E12_checkpoint_body_stream_hash"],
        }
    require(
        contexts[0]["checkpoint_pin"]["sha256"] != contexts[1]["checkpoint_pin"]["sha256"],
        "Two separate completed fold checkpoints required",
    )
    provenance_path = c.bound(root, spec["provenance"])
    provenance = c.read(provenance_path)
    require(
        provenance["weights_sha256"] == GENERIC_SHA
        and all(contexts[n]["run"]["encoder_provenance"] == provenance for n in (0, 1)),
        "Same audited generic encoder provenance required",
    )
    context = contexts[fold]
    context.update(
        fold=fold,
        config=config,
        provenance=provenance,
        prefix=base["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"],
        gate_input_spec=spec,
        producer_sha256=sha(__file__),
    )
    return context


def tensor_digest(state):
    checksum = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        checksum.update(name.encode())
        checksum.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _metadata(state, expected, torch):
    require(set(state) == set(expected), "Exact suffix/head tensor keys required")
    rows = []
    for name, shape in sorted(expected.items()):
        tensor = state[name]
        require(
            isinstance(tensor, torch.Tensor)
            and list(tensor.shape) == shape
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and bool(torch.isfinite(tensor).all()),
            "Finite exact CPU FP32 tensor shape/dtype required: " + name,
        )
        rows.append(
            {
                "name": name,
                "shape": shape,
                "dtype": str(tensor.dtype),
                "finite": True,
                "tensor_sha256": hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest(),
            }
        )
    return rows


def _reconstruct(context, loaded, repo, weights, *, fixture_only=False):
    """Only this private kernel permits toy factories, always under fixture scope."""
    import torch

    if not fixture_only:
        require(
            isinstance(loaded, sources.LoadedF009Sources) and loaded.receipt["production_F009_pins"] is True,
            "Actual source14 loader and producer identities required",
        )
    trainer = loaded.trainer
    require(
        trainer.native_source_hashes() == sources.CORE_SHA
        and trainer.NativeTripletMIL is loaded.model.NativeTripletMIL,
        "Pinned F009 trainer/model identity differs",
    )
    checkpoint = trainer.load_native_checkpoint(context["checkpoint_path"], allow_artificial=False)
    require(
        checkpoint["schema_version"] == "native392_generic_dino_suffix_head_fast16_v1"
        and type(checkpoint["fold"]) is int
        and checkpoint["fold"] == context["fold"]
        and checkpoint["epoch"] == 12
        and checkpoint["weak_bce"] == context["row"]["weak_bce"]
        and checkpoint["config"] == context["config"]
        and checkpoint["targets"] == _contracts().TARGETS
        and checkpoint["source_sha256"] == sources.CORE_SHA
        and checkpoint["encoder_provenance"] == context["provenance"]
        and checkpoint["frozen_prefix_sha256"] == context["prefix"]
        and checkpoint["artificial"] is False,
        "CPU checkpoint exact E12/run/config/source/provenance/prefix differs",
    )
    shapes = _contracts().expected_tensor_shapes()
    original = {part: _metadata(checkpoint[part], expected, torch) for part, expected in shapes.items()}
    # Exactly one audited generic factory; never use the legacy COMMON factory/class.
    model = trainer.new_native_model(repo, weights, context["provenance"], context["config"], device="cpu")
    require(
        type(model) is loaded.model.NativeTripletMIL
        and type(model).__module__ == loaded.namespace + ".finetune_model"
        and type(model.head) is loaded.modules["feature_model"].FrozenFeatureHead
        and type(model.head).__module__ == loaded.namespace + ".feature_model",
        "Exact isolated model/head class identity required; module-name spoofing is insufficient",
    )
    require(trainer.prefix_digest(model) == context["prefix"], "Generic CPU prefix/norm differs before apply")
    require(
        trainer.apply_native_checkpoint(model, checkpoint, context["provenance"]) is model,
        "Strict apply must preserve model identity",
    )
    model.eval().requires_grad_(False)
    suffix = {name: model.encoder.state_dict()[name] for name in trainer.suffix_keys(model)}
    head = model.head.state_dict()
    reconstructed = {"encoder_suffix": suffix, "head": head}
    metadata = {part: _metadata(state, shapes[part], torch) for part, state in reconstructed.items()}
    require(
        metadata == original
        and all(
            torch.equal(value, checkpoint[part][name])
            for part, state in reconstructed.items()
            for name, value in state.items()
        )
        and trainer.prefix_digest(model) == context["prefix"],
        "CPU exact suffix/head bytes or frozen prefix changed",
    )
    require(
        not model.training
        and not model.encoder.training
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()
        ),
        "CPU FP32 eval/no-grad required",
    )
    rng = torch.get_rng_state().clone()
    gamma_state = model._gamma_generator.get_state().clone()
    require(
        model._gamma_draws == 0
        and model.sample_training_gamma(1, "cpu") is None
        and model._gamma_draws == 0
        and torch.equal(rng, torch.get_rng_state())
        and torch.equal(gamma_state, model._gamma_generator.get_state()),
        "Eval gamma must perform no draw, power operation or RNG change",
    )
    return {
        "schema_version": _contracts().CPU_SCHEMA,
        "complete": True,
        "evidence_scope": "artificial_fixture_only" if fixture_only else "actual_CPU_tensor_reconstruction",
        "actual_CPU_strict_reconstruction": not fixture_only,
        "fold": context["fold"],
        "epoch": 12,
        "producer_sha256": context["producer_sha256"],
        "source_sha256": sources.CORE_SHA,
        "support_sha256": sources.SUPPORT_SHA,
        "producer_CLI_sha256": sha(ROOT / "scripts/check_f009_E12_cpu.py"),
        "namespace_helper_sha256": _contracts().HELPER_SHA,
        "CLI_sha256": sources.CLI_SHA,
        "config_sha256": sources.CONFIG_SHA,
        "namespace": loaded.namespace,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "head_class": type(model.head).__module__ + "." + type(model.head).__name__,
        "targets": _contracts().TARGETS,
        "run_pin": context["run_pin"],
        "retained_index_pin": context["retained_index_pin"],
        "checkpoint_pin": context["checkpoint_pin"],
        "checkpoint_schema": checkpoint["schema_version"],
        "weak_bce": checkpoint["weak_bce"],
        "encoder_provenance": context["provenance"],
        "frozen_prefix_sha256": context["prefix"],
        "tensor_metadata": metadata,
        **{part + "_tensor_sha256": tensor_digest(state) for part, state in reconstructed.items()},
        "generic_factory_calls": 1,
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "model_training": False,
        "encoder_training": False,
        "gamma_draws": 0,
        "eval_gamma_RNG_unchanged": True,
        "teacher_MRI_Gold_Report_read": False,
        "GPU_used": False,
        "checkpoint_loader": {
            "source_sha256": sources.CORE_SHA["native_training_fast16.py"],
            "map_location": "cpu",
            "weights_only": True,
            "allow_artificial": False,
        },
        "strict_apply_identity_and_tensor_bytes_exact": True,
        "runtime_or_adoption_qualified": False,
    }


def produce(spec_path, expected_spec_sha256, fold, output, repo, weights, expected_producer_sha256):
    c = _contracts()
    require(
        sha(__file__) == expected_producer_sha256 and sha(spec_path) == expected_spec_sha256,
        "Explicit reviewed producer/spec SHAs required",
    )
    spec = c.read(spec_path)
    context = prepare_inputs(ROOT, spec, fold)
    output = Path(output).resolve()
    require(
        output.is_relative_to(ROOT / "artifacts/qa") and not output.exists(),
        "Fresh artifacts/qa proof directory required",
    )
    # First model-body access occurs only after both complete positive evaluation guards.
    require(c.pin(context["checkpoint_path"]) == context["checkpoint_pin"], "Actual E12 checkpoint bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    loaded = None
    try:
        sources.stage_f009_sources(
            ROOT / "src/rsna_knee",
            ROOT / "scripts/train_native_dinov2_fast16.py",
            ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json",
            output / "source14",
        )
        loaded = sources.load_f009_sources(output / "source14", "_rsna_f009_strict_cpu_fold" + str(fold))
        result = _reconstruct(context, loaded, repo, weights)
        require(
            c.pin(context["checkpoint_path"]) == context["checkpoint_pin"]
            and c.pin(context["run_path"]) == context["run_pin"]
            and c.pin(context["index_path"]) == context["retained_index_pin"]
            and sha(spec_path) == expected_spec_sha256
            and sha(__file__) == expected_producer_sha256
            and loaded.trainer.native_source_hashes() == sources.CORE_SHA,
            "Input/source changed during reconstruction",
        )
        result.update(
            input_spec={"path": str(Path(spec_path).resolve()), **c.pin(spec_path)},
            local_source14_import=dict(loaded.receipt),
        )
        # A namespace cleanup failure must not leave a complete CPU receipt.
        loaded.close()
        loaded = None
        c.atomic_json(output / "CPU.json", result)
        return result
    except BaseException as error:
        c.atomic_json(
            output / "FAILED.json",
            {"complete": False, "actual_CPU_strict_reconstruction": False, "error_type": type(error).__name__},
        )
        raise
    finally:
        if loaded is not None:
            loaded.close()
