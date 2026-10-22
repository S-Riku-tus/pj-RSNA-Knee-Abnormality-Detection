"""Hash-bound F009 producer imports in an isolated package; no model construction.

This transports trusted local Python source, not untrusted-code sandboxing. The
eight checkpoint producer files need six additional relative-import modules.
Staging/import alone never qualifies weights, an evaluation, or a submission.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

CORE_SHA = {
    "native_training_fast16.py": "9f5c3b490374201f72cc73ebf11a26f8a8163945387d4d493e7a04c08e729d57",
    "finetune_model.py": "c43bcedf1d97b713528b4157a0a9c8867741b785d58c5be084cd7072c2b2c624",
    "feature_model.py": "af85d8eaed6015ed2ddb7f0b2029eca446471876750ebde60bed403cf8ced0e1",
    "triplet_cache.py": "0b4c666407cb72d7ef3249f479af1e13cca1b4422b18f18cebf3eed2b768f26f",
    "feature_imaging.py": "de36a21929b482eee897b12e06bb9964cee59beab6a9bc2b2d44c15ace279546",
    "feature_contracts.py": "be76ecc28b14491f4d1a0a459794c7f94456e845ce0e46c46ced699bfaf54ed6",
    "imaging.py": "07a114888cde19ae0e5799f2de48c6751ec9c4b5637ec1f4f4c220f2911db9ff",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
}
SUPPORT_SHA = {
    "checkpoints.py": "4ffc94da42c34798c263360ecdb4af7340f417d653cb7e37ad1b6169937be541",
    "metrics.py": "c5f3944e55392ae8a98fd6ed79b140a95d7af58fc3d3537d41ceefcc4c3b4f1c",
    "runtime.py": "09c7ee998e1c3be02f0ed3f00db6acc92351ea374fcf8a469c8d4fcc57233f05",
    "diagnostics.py": "aa89c9f29a8c641d005405b306f060a782c864d3e35a3b3371c7d3ffb4c1428a",
    "inputs.py": "2399ebaaf1f1c9fe3e3bcc302c90d2afa8f81c5352e16f5af6ae03efe93ff87f",
    "model.py": "86d21b96701c7ca99231fce3ddbd8f8d1c0967d81059b42787ef831cf203a96a",
}
CLI_SHA = "3e005de950c67aef7b8813f633755ff49ad39d13578df55f15d09b0b48dd3674"
CONFIG_SHA = "1a0a1d396660c6e326fd0f78962867b8243a3ac0e9b87fe2bb98dbea5f59297f"
SCHEMA = "f009_isolated_source14_package_v1"
INIT = b'"""F009 isolated producer package; explicit source-only import."""\n'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 256 * 1024,
             "Bounded regular UTF8 source/metadata file required")
    raw = path.read_bytes()
    raw.decode("utf-8")
    return raw


def _sources(directory, expected):
    _require(not Path(directory).is_symlink(), "Source directory symlink refused")
    directory = Path(directory).resolve(strict=True)
    bodies = {}
    for name, digest in expected.items():
        _require(re.fullmatch(r"[a-z][a-z0-9_]*\.py", name), "Flat Python source member required")
        raw = _read(directory / name).decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        _require(_sha(raw) == digest, "Pinned source SHA differs: " + name)
        bodies[name] = raw
    for name, raw in bodies.items():
        tree = ast.parse(raw.decode("utf-8"), filename=str(directory / name))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                _require(node.level == 1 and node.module is not None and node.module + ".py" in bodies,
                         "Relative import escapes the pinned closure")
            if isinstance(node, ast.Import):
                _require(all(not n.name.startswith("rsna_knee") for n in node.names),
                         "Canonical old package import forbidden")
            if isinstance(node, ast.ImportFrom) and not node.level:
                _require(not (node.module or "").startswith("rsna_knee"), "Canonical old package import forbidden")
    return bodies


def _stage(source_directory, cli_path, config_path, output, expected):
    bodies = _sources(source_directory, expected)
    cli, config = _read(cli_path), _read(config_path)
    _require(_sha(cli) == CLI_SHA and _sha(config) == CONFIG_SHA, "Fixed F009 CLI/config SHA differs")
    output = Path(output)
    _require(not output.exists() and not output.is_symlink(), "New staging directory required")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "package"
    package.mkdir()
    (package / "__init__.py").write_bytes(INIT)
    for name, raw in bodies.items():
        (package / name).write_bytes(raw)
    (output / "producer-CLI.py").write_bytes(cli)
    (output / "config.json").write_bytes(config)
    manifest = {"schema_version": SCHEMA, "source_sha256": expected,
                "CLI_sha256": CLI_SHA, "config_sha256": CONFIG_SHA, "init_sha256": _sha(INIT),
                "production_F009_pins": expected == {**CORE_SHA, **SUPPORT_SHA},
                "source8_is_not_complete_import_closure": True,
                "model_constructed_or_weights_loaded": False, "runtime_or_adoption_qualified": False}
    (output / "source-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def stage_f009_sources(source_directory, cli_path, config_path, output):
    """Copy exact 14-source closure plus fixed CLI/config; import/weights/fit0."""
    return _stage(source_directory, cli_path, config_path, output, {**CORE_SHA, **SUPPORT_SHA})


class _PinnedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, namespace, directory, bodies):
        self.namespace, self.directory, self.bodies = namespace, directory, bodies

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.namespace:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if fullname.startswith(self.namespace + "."):
            name = fullname[len(self.namespace) + 1:] + ".py"
            if name not in self.bodies:
                raise ModuleNotFoundError("Module outside pinned F009 closure: " + fullname)
            return importlib.util.spec_from_loader(fullname, self, is_package=False)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = "__init__.py" if module.__name__ == self.namespace else module.__name__.split(".")[-1] + ".py"
        raw = INIT if name == "__init__.py" else self.bodies[name]
        module.__file__ = str(self.directory / name)
        if name == "__init__.py":
            module.__path__ = [str(self.directory)]
        # Execute verified in-memory bytes, avoiding pre-existing bytecode or path finders.
        exec(compile(raw.decode("utf-8"), module.__file__, "exec"), module.__dict__)


@dataclass(frozen=True)
class LoadedF009Sources:
    namespace: str
    package: object
    modules: object
    receipt: object

    @property
    def trainer(self):
        return self.modules["native_training_fast16"]

    @property
    def model(self):
        return self.modules["finetune_model"]

    def close(self):
        """Release only this loader's owned namespace; never remove old modules."""
        _require(sys.modules.get(self.namespace) is self.package
                 and {n for n in sys.modules if n == self.namespace or n.startswith(self.namespace + ".")}
                 == {self.namespace, *(self.namespace + "." + n for n in self.modules)},
                 "Owned package or module inventory changed after load")
        for name, module in self.modules.items():
            _require(sys.modules.get(self.namespace + "." + name) is module,
                     "Namespace changed after load")
        for name in self.modules:
            del sys.modules[self.namespace + "." + name]
        sys.modules.pop(self.namespace)


def _load(output, namespace, expected):
    _require(re.fullmatch(r"_rsna_f009_[a-z0-9_]+", namespace) is not None,
             "Dedicated F009 top-level namespace required")
    _require(not any(n == namespace or n.startswith(namespace + ".") for n in sys.modules),
             "Namespace or descendants already imported")
    output = Path(output).resolve(strict=True)
    manifest = json.loads(_read(output / "source-manifest.json").decode("utf-8"))
    _require(manifest.get("schema_version") == SCHEMA and manifest.get("source_sha256") == expected
             and manifest.get("production_F009_pins") == (expected == {**CORE_SHA, **SUPPORT_SHA})
             and manifest.get("init_sha256") == _sha(INIT)
             and manifest.get("CLI_sha256") == CLI_SHA and manifest.get("config_sha256") == CONFIG_SHA,
             "Staged source manifest differs")
    _require(_sha(_read(output / "producer-CLI.py")) == CLI_SHA
             and _sha(_read(output / "config.json")) == CONFIG_SHA, "Staged CLI/config changed")
    directory = output / "package"
    _require(not directory.is_symlink() and {p.name for p in directory.iterdir()} == {*expected, "__init__.py"},
             "Exact pinned package inventory required")
    _require(_read(directory / "__init__.py") == INIT, "Package initializer changed")
    bodies = _sources(directory, expected)
    original = {n: m for n, m in sys.modules.items() if n == "rsna_knee" or n.startswith("rsna_knee.")}
    finder = _PinnedFinder(namespace, directory, bodies)
    sys.meta_path.insert(0, finder)
    try:
        package = importlib.import_module(namespace)
        importlib.import_module(namespace + ".native_training_fast16")
        modules = {name[:-3]: importlib.import_module(namespace + "." + name[:-3]) for name in expected}
        for name, module in modules.items():
            _require(module.__name__ == namespace + "." + name and module.__package__ == namespace
                     and Path(module.__file__).resolve() == directory / (name + ".py"),
                     "Imported module origin/namespace differs")
        trainer, model, head = (modules[name] for name in ("native_training_fast16", "finetune_model", "feature_model"))
        _require(trainer.NativeTripletMIL is model.NativeTripletMIL
                 and model.NativeTripletMIL.__module__ == namespace + ".finetune_model"
                 and model.FrozenFeatureHead is head.FrozenFeatureHead
                 and head.FrozenFeatureHead.__module__ == namespace + ".feature_model",
                 "Trainer/model/head class resolved outside its F009 package")
        _require(all(sys.modules.get(n) is m for n, m in original.items())
                 and set(original) == {n for n in sys.modules if n == "rsna_knee" or n.startswith("rsna_knee.")},
                 "Canonical old package was modified")
        if expected == {**CORE_SHA, **SUPPORT_SHA}:
            _require(trainer.native_source_hashes() == CORE_SHA, "Actual F009 source8 identity differs")
        receipt = {"schema_version": "f009_isolated_source14_import_receipt_v1", "namespace": namespace,
                   "source_sha256": expected, "production_F009_pins": expected == {**CORE_SHA, **SUPPORT_SHA},
                   "module_count": len(modules), "canonical_old_modules_unchanged": True,
                   "trainer_model_head_class_identity_passed": True, "verified_bytes_imported": True,
                   "model_constructed_or_weights_loaded": False, "runtime_or_adoption_qualified": False}
        return LoadedF009Sources(namespace, package, MappingProxyType(modules), MappingProxyType(receipt))
    except BaseException:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise
    finally:
        sys.meta_path.remove(finder)


def load_f009_sources(output, namespace):
    """Import exact F009 source only. Does not build/load a model or call CUDA."""
    return _load(output, namespace, {**CORE_SHA, **SUPPORT_SHA})
