"""Conditional F009 CLS/old-REG four-role construction and inference bridge.

Standalone vendorable source: stdlib only at import, no canonical rsna_knee
imports. Actual entry requires explicit execute/root permission and two fully
qualified CLS bundles. It never uploads, starts training, or resolves MRI.
The two CLS folds keep independent source14 namespaces; old REG stays old.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

FAMILIES = ("CLS", "REG")
FOLDS = (0, 1)
PLAN_SHA = "37bb0c60826da6deddda0e3929fd8aeb4a2f56ad91371af9dab365835a69bba2"
PAIR_SCHEMA = "f009_CLS_E12_P039_REG_four_role_metadata_v1"
PERMISSION_SCHEMA = "f009_P039_four_role_CPU_reconstruction_permission_v1"
RECEIPT_SCHEMA = "f009_P039_four_role_CPU_reconstruction_v1"
NAMESPACES = {0: "_rsna_f009_runtime_cls_f0", 1: "_rsna_f009_runtime_cls_f1"}
STATIC_DEPENDENCIES = {
    "cls_transport": "8c613c2c2d24fc6b1d1c6222482c581b03ac5027c15b10f5e7116e7f7056af47",
    "core": "4e365e1c46110561fcccb050ef70871ea0ef4bb52b6723176d0cc1da156723d3",
    "regional": "f0110b7ef7f6f1290447e4e9230f55a579b1476de4c866ff1fa0738bfd65bb8c",
    "boundary": "ef32e762ec5732597119d13d04b2f683b7bad446f352bb6509f6adf1c6af56af",
    "namespace_helper": "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526",
    "regional_original": "cf3e61687512ab885ebc0517df1aa5f16fa130d3ca637acbe01b875ddf5502ec",
    "reg_model": "31f91d75d73b222c1e37d8864abfe584c6bd648203ef67293ec56d855eaa9589",
    "old_native_model": "6befcaa880e13581ef776fc52e3d47d0d4f6d4776e73ee67e2ed8e25ab70193f",
    "reg_head": "af85d8eaed6015ed2ddb7f0b2029eca446471876750ebde60bed403cf8ced0e1",
}
PLAN_CANONICAL_SHA = 'd7a80608b6adca6d3329b358cfe22fed75dc5d86c9edc502b709b4885cf3172a'
FIXED_REG_FINGERPRINTS = {'REG:0': {'model_class': 'rsna_knee.regional_triplet_model.RegionalTripletMIL', 'checkpoint_sha256': '8ff42302a8a54e8e8caa3e089f176c73144bc79c5577f0990c6efeba47d881ad', 'frozen_prefix_sha256': '67a05dff5f621b78dc8b16cee95a56aea314437806637950ad8336a4ab522347', 'encoder_suffix_tensor_sha256': '278f20295b1729faa80deea972685f038b3edffde2ed37e0523ffa183324818e', 'head_tensor_sha256': '1b5bcabba082ef67f197037214884162efae4195c62386dc102224b9f4583f8c', 'CPU_receipt_sha256': '3e441a7f6b712e02698ac6ffdd03af0e22188a9ccc1014c55b2bf941d1adb2fa', 'run_json_sha256': 'a46f9e82f0105727a35bd770c0b903f74f121e8f9e2193d488a60b427a042e36', 'selected_epoch': 8, 'weak_bce': 0.28030224999122455}, 'REG:1': {'model_class': 'rsna_knee.regional_triplet_model.RegionalTripletMIL', 'checkpoint_sha256': 'eb640d7662cfaeb7df2d998a5f0f219494ccc0494abea91d0e95d7bbb1ac0714', 'frozen_prefix_sha256': '67a05dff5f621b78dc8b16cee95a56aea314437806637950ad8336a4ab522347', 'encoder_suffix_tensor_sha256': 'c9ac9728d1145a253144c0383c3dd3a6cf742e541822eb5a2b10823b1ec981b6', 'head_tensor_sha256': '2785d46eb7be0164a3d1fdd245945210e10ee34fc60dc18835ec02478d16cd62', 'CPU_receipt_sha256': 'addb97d372643e848c858e39154cbc9e805a6efc1c6cbf7c963b8e189111a237', 'run_json_sha256': '9367b74d0346d18a7e8daee6508a9ba411233509529ef2e15f19901ceec06b70', 'selected_epoch': 10, 'weak_bce': 0.2779688899362838}}
FIXED_REG_TRANSPORT = {'REG:0': {'input_root': '/kaggle/input/datasets/rsraki/n392rg-f0-8e978296019046e3a9fff8fcb6a4901f', 'dataset_ref': 'rsraki/n392rg-f0-8e978296019046e3a9fff8fcb6a4901f/1', 'manifest_sha': '205cbc3c2ca006f80c6af0755c0dca22d332a4e1bdc6c1fd8f0555c6f8962de9', 'archive_sha': 'dcce9e2629fb1be4940b0dbaf96108cbd36779df97c6621212beb54b97be02ee', 'bundle_sha': '1520eba01499b6c456f88013bfde35ebd76574a87fa6381ffeeee441f3361c37', 'root_permission': 'C:\\Users\\Casper4\\Python\\ueki\\shibasaki\\pj-RSNA-Knee-Abnormality-Detection\\artifacts\\research\\20261012-p018-regional-bytes-root-v1\\fold0-transport-permission.json', 'root_permission_sha': '84afcab13cf48bf4424d6c23d77687099c1238ebceaf7c2dae288af6ac65acd8', 'peer_review': 'C:\\Users\\Casper4\\Python\\ueki\\shibasaki\\pj-RSNA-Knee-Abnormality-Detection\\artifacts\\research\\20261010-f005-regional-transport-independent-review-v1\\review.json', 'peer_sha': '6102ea4107e18243cf2d7f73acdfe92e56e59dce30f358ad2d2be321b516345b'}, 'REG:1': {'input_root': '/kaggle/input/datasets/rsraki/n392rg-f1-82087594dae443ab82a701d9750b661a', 'dataset_ref': 'rsraki/n392rg-f1-82087594dae443ab82a701d9750b661a/1', 'manifest_sha': 'fdf6f6f433659c586ee8a2b392a206fd1b54c1556cd84f4b3dd339db3a9a1c72', 'archive_sha': 'fcd685be3cb030b87892939b0b82b03fe5eb6c2ba44659bd88dfd8e4f1bd87f5', 'bundle_sha': '088df29f855afd0e69814b48e8b63383cac339e5376c09f6aeee8d1903fd7760', 'root_permission': 'C:\\Users\\Casper4\\Python\\ueki\\shibasaki\\pj-RSNA-Knee-Abnormality-Detection\\artifacts\\research\\20261012-p018-regional-bytes-root-v1\\fold1-transport-permission.json', 'root_permission_sha': '6f1b852cd7af7c3054adc9e55424ffd21f8636fca1e0417b9ab551cbef6277a4', 'peer_review': 'C:\\Users\\Casper4\\Python\\ueki\\shibasaki\\pj-RSNA-Knee-Abnormality-Detection\\artifacts\\research\\20261010-f005-regional-transport-independent-review-v1\\review.json', 'peer_sha': '6102ea4107e18243cf2d7f73acdfe92e56e59dce30f358ad2d2be321b516345b'}}


def require(value, message):
    if not value:
        raise ValueError(message)


p = SimpleNamespace(require=require, FAMILIES=FAMILIES)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular pinned member required")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _module(module, role, pins):
    require(role in pins and set(pins[role]) == {"path", "sha256"}, "Exact module dependency pin required: " + role)
    require(isinstance(module, ModuleType) and sys.modules.get(module.__name__) is module
            and module.__spec__ is not None and module.__spec__.origin is not None,
            "Actual imported module identity required: " + role)
    path = Path(module.__file__).resolve()
    require(Path(module.__spec__.origin).resolve() == path
            and path == Path(pins[role]["path"]).resolve() and sha(path) == pins[role]["sha256"]
            and (role not in STATIC_DEPENDENCIES or pins[role]["sha256"] == STATIC_DEPENDENCIES[role]),
            "Pinned dependency identity differs: " + role)
    return module


def _permission(permission, path, expected, pair, dependency_pins):
    require(path is not None and expected is not None, "Explicit root permission path/SHA required")
    path = Path(path)
    require(path.suffix == ".json" and 0 < path.stat().st_size <= 2*1024**2 and sha(path) == expected,
            "Root permission source SHA differs")
    require(json.loads(path.read_bytes()) == permission and permission["schema_version"] == PERMISSION_SCHEMA
            and permission["root_authorized_CPU_reconstruction"] is True
            and permission["conditional_plan_sha256"] == PLAN_SHA
            and permission["pair_fingerprint_sha256"] == pair["pair_fingerprint_sha256"]
            and permission["runtime_source_sha256"] == sha(__file__)
            and permission["dependency_pins"] == dependency_pins,
            "Explicit actual model permission/bindings required; source readiness is insufficient")


def _config_compatibility(left, right):
    for key in ("seed", "n_folds", "preprocess", "encoder", "head", "train"):
        a, b = copy.deepcopy(left[key]), copy.deepcopy(right[key])
        if key == "train":
            a.pop("lr_schedule", None)
            b.pop("lr_schedule", None)
        require(a == b, "Unchanged shared architecture/data config required: " + key)


def _shared_inputs(cls_runs, reg_runs, cls_bindings, reg_bindings):
    for key in ("manifest_sha256", "pixel_index_sha256"):
        reference = reg_runs[0]["audit"][key]
        require(all(cls_runs[fold]["audit"][key] == reg_runs[fold]["audit"][key] == reference
                    for fold in FOLDS), "Shared weak/group/pixel audit differs: " + key)
    identifiers = [bindings[fold]["checkpoint_sha256"]
                   for bindings in (cls_bindings, reg_bindings) for fold in FOLDS]
    require(len(set(identifiers)) == 4 and all(isinstance(value, str)
            and re.fullmatch("[0-9a-f]{64}", value) for value in identifiers),
            "Four distinct fixed selected checkpoint SHAs required")


def bind_pairs(core, regional, roots, manifests, *, cls_pair, conditional_plan, cls_transport,
               namespace_helper, dependency_pins, execute=False):
    """Concrete future router preflight; default refuses before asset access."""
    require(execute is True, "Explicit mixed metadata execution required")
    return _bind_pairs(core, regional, roots, manifests, cls_pair=cls_pair, conditional_plan=conditional_plan,
        cls_transport=cls_transport, namespace_helper=namespace_helper, dependency_pins=dependency_pins,
        fixture_only=False)


def _bind_pairs(core, regional, roots, manifests, *, cls_pair, conditional_plan, cls_transport,
                namespace_helper, dependency_pins, fixture_only):
    for name, module in (("core", core), ("regional", regional), ("cls_transport", cls_transport),
                         ("namespace_helper", namespace_helper)):
        _module(module, name, dependency_pins)
    require(cls_transport.sources is namespace_helper, "Transport must use the same pinned source14 helper")
    require(json_sha(conditional_plan) == PLAN_CANONICAL_SHA, "Frozen P039 comparison plan differs")
    require(set(roots) == set(manifests) == set(FAMILIES)
            and all(set(roots[f]) == set(manifests[f]) == set(FOLDS) for f in FAMILIES)
            and len({Path(roots[f][n]).resolve() for f in FAMILIES for n in FOLDS}) == 4,
            "Four distinct explicit role roots/manifests required")
    require(all(manifests["REG"][n] == FIXED_REG_TRANSPORT["REG:"+str(n)]["bundle_sha"] for n in FOLDS),
            "Exact P039 REG manifest refs required")
    reg_pair, reg_views, reg_documents = regional.bind_fold_seals(roots["REG"], manifests["REG"])
    cls_documents, bindings = {}, {}
    for fold in FOLDS:
        if fixture_only:
            value = cls_transport._consume(roots["CLS"][fold], expected_bundle_pin=manifests["CLS"][fold],
                pair=cls_pair, conditional_plan=conditional_plan, fold=fold, fixture_only=True)
            require(value["actual_transport"] is False, "Artificial transport must remain false")
        else:
            value = cls_transport.consume_f009_bundle(roots["CLS"][fold], expected_bundle_pin=manifests["CLS"][fold],
                pair=cls_pair, conditional_plan=conditional_plan, fold=fold)
            require(value["actual_transport"] is True and value["cpu"]["actual_CPU_strict_reconstruction"] is True,
                    "Actual qualified transport/CPU evidence required")
        require(Path(value["bundle_root"]).resolve() == Path(roots["CLS"][fold]).resolve()
                and value["cpu"]["fold"] == value["run"]["fold"] == fold
                and value["cpu"]["epoch"] == 12, "CLS role/source root/fold differs")
        cpu = value["cpu"]
        binding = {"fold": fold, "config": value["config"], "encoder_provenance": value["run"]["encoder_provenance"],
            "checkpoint_sha256": cpu["checkpoint_pin"]["sha256"], "checkpoint_pin": cpu["checkpoint_pin"],
            "CPU_receipt_sha256": sha(Path(value["bundle_root"])/"audit/CPU.json"),
            "run_json_sha256": cpu["run_pin"]["sha256"], "selected_epoch": 12, "weak_bce": cpu["weak_bce"],
            "frozen_prefix_sha256": cpu["frozen_prefix_sha256"],
            "encoder_suffix_tensor_sha256": cpu["encoder_suffix_tensor_sha256"],
            "head_tensor_sha256": cpu["head_tensor_sha256"],
            "producer_model_class": cpu["model_class"], "producer_head_class": cpu["head_class"],
            "runtime_model_class": NAMESPACES[fold]+".finetune_model.NativeTripletMIL",
            "runtime_head_class": NAMESPACES[fold]+".feature_model.FrozenFeatureHead",
            "source14_manifest_sha256": sha(Path(value["source14"])/"source-manifest.json")}
        bindings[fold], cls_documents[fold] = binding, value
        rb = reg_pair["bindings"][fold]
        require(all(rb.get(key) == fp for key, fp in FIXED_REG_FINGERPRINTS["REG:"+str(fold)].items()
                    if key != "model_class"), "Fixed P039 REG fingerprints differ")
        require(rb["encoder_provenance"] == binding["encoder_provenance"]
                and rb["frozen_prefix_sha256"] == binding["frozen_prefix_sha256"], "One common generic prefix/provenance required")
        _config_compatibility(binding["config"], rb["config"])
    require(bindings[0]["config"] == bindings[1]["config"] and bindings[0]["checkpoint_sha256"] != bindings[1]["checkpoint_sha256"],
            "Two distinct fixed F009 folds required")
    _shared_inputs({n:cls_documents[n]["run"] for n in FOLDS}, reg_views, bindings, reg_pair["bindings"])
    cls_descriptor = {"bindings": bindings, "qualification_pins": cls_pair["qualification_pins"],
        "qualification_binding_pins": cls_pair["qualification_binding_pins"],
        "bundle_pins": manifests["CLS"]}
    cls_descriptor["pair_fingerprint_sha256"] = json_sha(cls_descriptor)
    mixed = {"schema_version": PAIR_SCHEMA, "family_pairs": {"CLS": cls_descriptor, "REG": reg_pair},
        "family_fold_manifest_sha256": {"CLS": {n: manifests["CLS"][n]["sha256"] for n in FOLDS}, "REG": manifests["REG"]},
        "model_classes": {"CLS:"+str(n): bindings[n]["runtime_model_class"] for n in FOLDS} |
                         {"REG:"+str(n): FIXED_REG_FINGERPRINTS["REG:"+str(n)]["model_class"] for n in FOLDS},
        "targets": list(core.p.TARGETS), "family_order": list(FAMILIES), "fold_order": list(FOLDS),
        "FP32_aggregation": "CLS(p0+p1)*.5; REG(p0+p1)*.5; (CLSmean+REGmean)*.5",
        "conditional_plan_sha256": PLAN_SHA, "actual_runtime_qualified": False,
        "artificial_fixture_only": fixture_only, "qualification_transfer_from_old_receipts": False}
    mixed["pair_fingerprint_sha256"] = json_sha(mixed)
    return mixed, {"CLS": {n: cls_documents[n]["run"] for n in FOLDS}, "REG": reg_views}, {"CLS": cls_documents, "REG": reg_documents}


def model_fingerprints_from_pair(pair):
    """Expected router descriptor, not evidence that reconstruction occurred."""
    require(pair["schema_version"] == PAIR_SCHEMA, "New mixed pair schema required")
    result = {}
    for family in FAMILIES:
        for fold in FOLDS:
            role = family+":"+str(fold)
            binding = pair["family_pairs"][family]["bindings"][fold]
            result[role] = {"model_class": pair["model_classes"][role], **{key: binding[key] for key in (
                "checkpoint_sha256", "frozen_prefix_sha256", "encoder_suffix_tensor_sha256", "head_tensor_sha256",
                "CPU_receipt_sha256", "run_json_sha256", "selected_epoch", "weak_bce")}}
            if family == "CLS":
                result[role].update(producer_model_class=binding["producer_model_class"],
                    producer_head_class=binding["producer_head_class"],
                    source14_manifest_sha256=binding["source14_manifest_sha256"],
                    runtime_head_class=binding["runtime_head_class"])
    return result


class _NamespaceOwner:
    def __init__(self, handles):
        self.handles, self.references = list(handles), 1
    def acquire(self):
        require(self.references > 0, "Closed F009 namespaces")
        self.references += 1
    def release(self):
        require(self.references > 0, "Namespace lease already closed")
        if self.references == 1:
            # Keep failed handles owned and retryable; never claim successful close.
            while self.handles:
                self.handles[-1].close()
                self.handles.pop()
        self.references -= 1


class MixedModels(dict):
    """Dict-compatible serving models; close every CPU/GPU owner after use."""
    def __init__(self, values, handles=None, *, owner=None, classes=None):
        super().__init__(values)
        self.owner = _NamespaceOwner(handles) if owner is None else owner
        if owner is not None:
            self.owner.acquire()
        self.classes, self.closed = dict(classes or {}), False
    def close(self):
        require(not self.closed, "Mixed model container already closed")
        self.clear()
        self.owner.release()
        self.closed = True


def _tensor_digest(state):
    checksum = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        checksum.update(name.encode())
        checksum.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _alias_count(models):
    require(len(models) == len({id(model) for model in models.values()}), "Independent full model objects required")
    def storage(model):
        return {(str(t.device), t.untyped_storage().data_ptr()) for t in model.state_dict().values() if t.numel()}
    pairs = list(itertools.combinations(models, 2))
    require(all(not (storage(models[a]) & storage(models[b])) for a, b in pairs), "Full model storage alias detected")
    return len(pairs)


def _gamma_snapshot(models):
    return {key: (model._gamma_draws, model._gamma_generator.get_state().clone())
            for key, model in models.items() if hasattr(model, "_gamma_generator")}


def _gamma_equal(before, models, torch):
    return all(model._gamma_draws == before[key][0] and torch.equal(model._gamma_generator.get_state(), before[key][1])
               for key, model in models.items() if key in before)


def _eval_gamma(models, torch):
    rng = torch.get_rng_state().clone()
    states = _gamma_snapshot(models)
    for key, model in models.items():
        require(not model.training and not model.encoder.training, "Eval model and encoder required")
        if key in states:
            require(model._gamma_draws == 0 and model.sample_training_gamma(1, "cpu") is None,
                    "Eval gamma must be inactive with zero draws")
    require(torch.equal(rng, torch.get_rng_state()) and _gamma_equal(states, models, torch),
            "Eval gamma sampler changed CPU/global or owned RNG")


def _reg_identity(regional_original, dependency_pins):
    reg = _module(regional_original, "regional_original", dependency_pins)
    require(reg.__name__ == "rsna_knee.regional_training_fast16", "Canonical old REG trainer required")
    model_module = _module(sys.modules[reg.RegionalTripletMIL.__module__], "reg_model", dependency_pins)
    old = _module(sys.modules[model_module.NativeTripletMIL.__module__], "old_native_model", dependency_pins)
    head = _module(sys.modules[old.FrozenFeatureHead.__module__], "reg_head", dependency_pins)
    require(reg.RegionalTripletMIL is model_module.RegionalTripletMIL
            and reg.RegionalTripletMIL.__bases__ == (old.NativeTripletMIL,)
            and old.FrozenFeatureHead is head.FrozenFeatureHead,
            "Old REG/Native/head class identities or inheritance differ")
    return reg


def reconstruct_cpu_models(roots, pair, views, documents, permission, *, core, regional, boundary,
                           regional_original, cls_pair, conditional_plan, cls_transport, namespace_helper,
                           dependency_pins, permission_path=None, expected_permission_sha256=None,
                           mode="CPU_smoke", execute=False, **unused):
    """Router-compatible constructor; no hidden loader/factory defaults."""
    require(execute is True and mode == "CPU_smoke" and not unused, "Explicit fixed CPU reconstruction only")
    _permission(permission, permission_path, expected_permission_sha256, pair, dependency_pins)
    _module(boundary, "boundary", dependency_pins)
    reg = _reg_identity(regional_original, dependency_pins)
    expected_manifests = {"CLS": pair["family_pairs"]["CLS"]["bundle_pins"],
                          "REG": pair["family_fold_manifest_sha256"]["REG"]}
    fresh, fresh_views, fresh_documents = bind_pairs(core, regional, roots, expected_manifests, cls_pair=cls_pair,
        conditional_plan=conditional_plan, cls_transport=cls_transport, namespace_helper=namespace_helper,
        dependency_pins=dependency_pins, execute=True)
    require((fresh, fresh_views, fresh_documents) == (pair, views, documents),
            "Fresh qualified mixed pair/run views/documents differ")
    regional.verify_assets(roots["REG"], documents["REG"])
    handles, models = [], None
    try:
        for fold in FOLDS:
            loaded = namespace_helper.load_f009_sources(documents["CLS"][fold]["source14"], NAMESPACES[fold])
            require(loaded.receipt["production_F009_pins"] is True, "Actual pinned F009 closure required")
            handles.append(loaded)
        models, receipt = _construct_cpu_models(roots, pair, views, documents, core=core, reg=reg,
            boundary=boundary, loaded_by_fold=dict(zip(FOLDS, handles)), fixture_only=False)
        require(receipt["model_fingerprints"] == model_fingerprints_from_pair(pair),
                "Actual reconstructed per-role fingerprints differ from new pair")
        require(sha(permission_path) == expected_permission_sha256
                and sha(__file__) == permission["runtime_source_sha256"], "Permission/runtime changed during construction")
        return models, receipt
    except BaseException:
        if models is not None:
            models.close()
        else:
            for handle in reversed(handles):
                handle.close()
        raise


def _construct_cpu_models(roots, pair, views, documents, *, core, reg, boundary, loaded_by_fold,
                          fixture_only, artificial_factory=None):
    import numpy as np
    import torch
    require(set(loaded_by_fold) == set(FOLDS), "Independent source14 namespace per CLS fold required")
    require(len({loaded.namespace for loaded in loaded_by_fold.values()}) == 2
            and len({Path(loaded.model.__file__).resolve() for loaded in loaded_by_fold.values()}) == 2,
            "Distinct per-fold model namespace/source roots required")
    bindings = {family: pair["family_pairs"][family]["bindings"] for family in FAMILIES}
    checkpoints = {}
    originals = {}
    for family in FAMILIES:
        for fold in FOLDS:
            original = loaded_by_fold[fold].trainer if family == "CLS" else reg
            binding = bindings[family][fold]
            path = documents["CLS"][fold]["checkpoint_path"] if family == "CLS" else Path(roots["REG"][fold])/"trained/best.pt"
            require(sha(path) == binding["checkpoint_sha256"], "Bound checkpoint bytes changed before strict loader")
            cp = original.load_native_checkpoint(path, allow_artificial=False) if family == "CLS" else original.load_regional_checkpoint(path, allow_artificial=False)
            if family == "CLS":
                require(original.NativeTripletMIL is loaded_by_fold[fold].model.NativeTripletMIL
                        and original.native_source_hashes() == {
                            key: loaded_by_fold[fold].receipt["source_sha256"][key]
                            for key in original.native_source_hashes()}
                        and Path(loaded_by_fold[fold].model.__file__).resolve().parent
                            == (Path(documents["CLS"][fold]["source14"])/"package").resolve()
                        and sys.modules.get(loaded_by_fold[fold].model.__name__) is loaded_by_fold[fold].model,
                        "Actual F009 trainer/class/source identity differs")
                require(cp["config"] == binding["config"] and cp["fold"] == fold and cp["epoch"] == 12
                        and cp["encoder_provenance"] == binding["encoder_provenance"]
                        and cp["frozen_prefix_sha256"] == binding["frozen_prefix_sha256"]
                        and cp["weak_bce"] == binding["weak_bce"], "Exact F009 fixed E12 metadata differs")
            else:
                boundary.guard_checkpoint(cp, views["REG"][fold], binding)
            require(all(len(cp[part]) == count and all(isinstance(t,torch.Tensor) and t.device.type == "cpu"
                        and t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in cp[part].values())
                        for part,count in (("encoder_suffix",28),("head",12))), "Exact finite CPU FP32 suffix28/head12 required")
            require(_tensor_digest(cp["encoder_suffix"]) == binding["encoder_suffix_tensor_sha256"]
                    and _tensor_digest(cp["head"]) == binding["head_tensor_sha256"],
                    "Bound suffix/head bytes or shapes differ before factory")
            checkpoints[family,fold], originals[family,fold] = cp, original
    seed = bindings["REG"][0]["config"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root, binding = Path(roots["REG"][0]), bindings["REG"][0]
    require((artificial_factory is None) if not fixture_only else callable(artificial_factory),
            "Artificial factory permitted only in private artificial kernel")
    factory = reg.new_regional_model if not fixture_only else artificial_factory
    regional_base = factory(root/"generic/repo",root/"generic/dinov2_vits14_pretrain.pth",
                            binding["encoder_provenance"],binding["config"],device="cpu")
    require(type(regional_base) is reg.RegionalTripletMIL, "Exact old REG factory class required")
    values = {"CLS": {}, "REG": {fold: copy.deepcopy(regional_base) for fold in FOLDS}}
    classes = {}
    for fold in FOLDS:
        loaded, config = loaded_by_fold[fold], bindings["CLS"][fold]["config"]
        options = config["online"]
        values["CLS"][fold] = loaded.model.NativeTripletMIL(copy.deepcopy(regional_base.encoder),config,
            trainable_blocks=options["trainable_blocks"],window_chunk=options["window_chunk"],
            checkpoint_last2=options["checkpoint_last2"]).to("cpu")
    del regional_base
    fingerprints = {}
    for family in FAMILIES:
        for fold in FOLDS:
            model, original, binding = values[family][fold], originals[family,fold], bindings[family][fold]
            expected_class = loaded_by_fold[fold].model.NativeTripletMIL if family == "CLS" else reg.RegionalTripletMIL
            require(type(model) is expected_class and (model.trainable_blocks,model.window_chunk,model.checkpoint_last2)==(2,16,True)
                    and original.prefix_digest(model)==binding["frozen_prefix_sha256"], "Exact per-role constructor/prefix required")
            contract = model.parameter_contract()
            require(contract["trainable_encoder_parameters"]==3550464 and contract["trainable_head_parameters"]==83352,
                    "Original last2/head parameter counts required")
            apply = original.apply_native_checkpoint if family == "CLS" else original.apply_regional_checkpoint
            require(apply(model,checkpoints[family,fold],binding["encoder_provenance"]) is model, "Strict apply must retain full role object")
            model.eval().requires_grad_(False)
            suffix = {name:model.encoder.state_dict()[name] for name in original.suffix_keys(model)}
            require(original.prefix_digest(model)==binding["frozen_prefix_sha256"]
                    and _tensor_digest(suffix)==binding["encoder_suffix_tensor_sha256"]
                    and _tensor_digest(model.head.state_dict())==binding["head_tensor_sha256"]
                    and all(t.device.type=="cpu" and t.dtype==torch.float32 and not t.requires_grad for t in model.parameters()),
                    "Full strict per-role FP32 tensors/prefix differ")
            role = family+":"+str(fold)
            classes[role] = expected_class
            fingerprints[role] = {"model_class":type(model).__module__+"."+type(model).__name__,
                **{key:binding[key] for key in ("checkpoint_sha256","frozen_prefix_sha256","encoder_suffix_tensor_sha256",
                                              "head_tensor_sha256","CPU_receipt_sha256","run_json_sha256","selected_epoch","weak_bce")}}
            if family == "CLS":
                fingerprints[role].update(producer_model_class=binding["producer_model_class"],
                    producer_head_class=binding["producer_head_class"],source14_manifest_sha256=binding["source14_manifest_sha256"],
                    runtime_head_class=type(model.head).__module__+"."+type(model.head).__name__)
    flat={f+":"+str(n):values[f][n] for f in FAMILIES for n in FOLDS}
    checked=_alias_count(flat)
    _eval_gamma(flat,torch)
    container=MixedModels(values,[loaded_by_fold[n] for n in FOLDS],classes=classes)
    return container,{"schema_version":RECEIPT_SCHEMA,"complete":not fixture_only,"phase_contract_passed":not fixture_only,
        "actual_CPU_four_model_reconstruction":not fixture_only,"CPU_strict_four_model_reconstruction":not fixture_only,
        "artificial_stub_components":fixture_only,"generic_factory_calls":1,"CLS_constructor_calls":2,
        "logical_CPU_fold_models":4,"selected_checkpoint_loads":4,"strictapply_calls":4,
        "CPU_storage_alias_count":0,"CPU_storage_alias_pairs_checked":checked,"native_MRI_decode":0,"model_forward_calls":0,
        "model_fingerprints":fingerprints,"model_classes":{key:cls.__module__+"."+cls.__name__ for key,cls in classes.items()},
        "pair_fingerprint_sha256":pair["pair_fingerprint_sha256"],
        "family_fold_manifest_sha256":pair["family_fold_manifest_sha256"],
        "per_fold_source_roots":{str(n):str(Path(loaded_by_fold[n].model.__file__).resolve()) for n in FOLDS},
        "namespaces_kept_alive_until_models_closed":True,"eval_gamma_RNG_unchanged":True,
        "gamma_draws_by_role":{key:model._gamma_draws for key,model in flat.items() if hasattr(model,"_gamma_draws")},
        "construction_RNG_preserved_claimed":False,"actual_Kaggle_qualification":False,
        "serving_candidate_gate_passed":False,"runtime_or_accuracy_qualified":False}


def models_on_two_devices(cpu_models, core):
    """Future two-T4 source bridge; no CUDA execution during preparation."""
    import torch
    require(isinstance(cpu_models,MixedModels) and not cpu_models.closed and torch.cuda.is_initialized()
            and torch.cuda.device_count()==2,"Explicit live model owner and initialized two-CUDA context required")
    before=torch.get_rng_state().clone()
    cuda_before=[v.clone() for v in torch.cuda.get_rng_state_all()]
    original={f+":"+str(n):cpu_models[f][n] for f in FAMILIES for n in FOLDS}
    gamma_before=_gamma_snapshot(original)
    result={family:core.models_on_two_devices(cpu_models[family]) for family in FAMILIES}
    flat={f+":"+str(n)+":"+device:result[f][device][n] for f in FAMILIES for n in FOLDS for device in ("cuda:0","cuda:1")}
    for family in FAMILIES:
        for fold in FOLDS:
            role=family+":"+str(fold)
            for device in ("cuda:0","cuda:1"):
                model=result[family][device][fold]
                require(type(model) is cpu_models.classes[role] and not model.training and not model.encoder.training
                        and all(str(t.device)==device and t.dtype==torch.float32 and not t.requires_grad for t in model.parameters()),
                        "Exact per-role class/device/eval/frozen replica required")
                if role in gamma_before:
                    require(model._gamma_generator is not original[role]._gamma_generator
                            and model._gamma_draws==gamma_before[role][0]
                            and torch.equal(model._gamma_generator.get_state(),gamma_before[role][1]),
                            "Copied gamma generator ownership/state differs")
    _alias_count(flat)  # Check all 28 pairs, including cross-device ownership.
    checked = sum(_alias_count({f+":"+str(n):result[f][device][n]
                                for f in FAMILIES for n in FOLDS})
                  for device in ("cuda:0", "cuda:1"))
    require(checked == 12, "Preserve router six pairs per device, twice")
    require(torch.equal(before,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all()))
            and _gamma_equal(gamma_before,original,torch),"CPU/CUDA/gamma RNG changed across replica copies/transfers")
    return MixedModels(result,owner=cpu_models.owner,classes=cpu_models.classes),checked


def _consume_four_numeric(device, payload, *, models, predictions, receipts, lock, core, pair,
                 family_predictions, family_receipts, array_hashes):
    """Original twofold consumer twice; immutable five arrays and fixed fold/family order."""
    key, arrays, pixel_receipt = payload
    p.require(array_hashes(arrays) == pixel_receipt["array_sha256"], "Native arrays differ before four full forwards")
    for family in p.FAMILIES:
        core.consume_twofold(device, payload, models=models[family], predictions=family_predictions[family],
                             receipts=family_receipts[family], lock=lock,
                             pair_sha=pair["family_pairs"][family]["pair_fingerprint_sha256"])
        p.require(array_hashes(arrays) == pixel_receipt["array_sha256"], "Native arrays modified by a family forward")
    values = [family_predictions[f][key] for f in p.FAMILIES]
    for family in p.FAMILIES:
        receipt = family_receipts[family][key]
        p.require(receipt["fold_order"] == [0, 1] and receipt["logical_full_fold_model_forward_calls"] == 2
                  and receipt["fallback"] is False and receipt["device"] == device
                  and family_predictions[family][key] == core.probability_mean_reference(*receipt["fold_probabilities"]),
                  "Exact original full twofold family forward required")
    mean = core.probability_mean_reference(*values)
    with lock:
        p.require(key not in predictions, "Duplicate mixed UID consumed")
        predictions[key] = mean
        receipts[key] = {**pixel_receipt, "device": device, "family_order": list(p.FAMILIES), "fold_order": [0, 1],
                         "pair_fingerprint_sha256": pair["pair_fingerprint_sha256"],
                         "logical_full_fold_model_forward_calls": 4, "family_mean_probabilities": dict(zip(p.FAMILIES, values)),
                         "families": {f: family_receipts[f][key] for f in p.FAMILIES},
                         "FP32_aggregation": pair["FP32_aggregation"], "shared_CLS_or_shared_prefix_forward": False,
                         "inference_mode_in_consumer": True, "fallback": False}


def consume_four(device, payload, *, models, predictions, receipts, lock, core, pair,
                 family_predictions, family_receipts, array_hashes):
    """Unchanged four-forward FP32 kernel plus eval/RNG guards; no rerank."""
    import torch
    require(isinstance(models,MixedModels) and not models.closed,"Live owned mixed model container required")
    flat={f+":"+str(n):models[f][device][n] for f in FAMILIES for n in FOLDS}
    require(all(type(model) is models.classes[key] for key,model in flat.items()),"Consumer per-role class differs")
    before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state(device).clone() if str(device).startswith("cuda:") else None
    _eval_gamma(flat,torch)
    gamma_before=_gamma_snapshot(flat)
    _consume_four_numeric(device,payload,models=models,predictions=predictions,receipts=receipts,lock=lock,
        core=core,pair=pair,family_predictions=family_predictions,family_receipts=family_receipts,array_hashes=array_hashes)
    require(torch.equal(before,torch.get_rng_state()) and _gamma_equal(gamma_before,flat,torch)
            and (cuda_before is None or torch.equal(cuda_before,torch.cuda.get_rng_state(device))),
            "Inference changed CPU/CUDA/gamma RNG")
