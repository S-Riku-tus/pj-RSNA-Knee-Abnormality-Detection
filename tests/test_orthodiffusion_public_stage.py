"""Stdlib-only fake-stream stage tests. No real URLs/model/MRI/science run."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("orthodiff_stage_builder", "prepare_orthodiffusion_public_stage.py")
runtime = load("orthodiff_stage_runtime", "orthodiffusion_public_stage_runtime.py")


class Response(io.BytesIO):
    def __init__(self, body, url, *, length=None, status=200):
        super().__init__(body)
        self.status = status
        self.url = url
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def open(self, request, timeout):
        self.calls.append(request.full_url)
        body = self.rows[request.full_url]
        return Response(body, request.full_url)


def synthetic_assets():
    rows, assets = {}, []
    for relative in sorted(runtime.SOURCE_PATHS):
        body = b"import math\n" if relative.endswith(".py") else b"synthetic source marker\n"
        url = "https://raw.githubusercontent.com/lt-0123/OrthoDiffusion/" + runtime.CODE_COMMIT + "/" + relative
        rows[url] = body
        assets.append({"kind": "source", "path": "upstream/" + relative, "url": url,
                       "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "git_blob_sha1": builder.git_blob(body)})
    body = b"synthetic checkpoint marker, not a model and never deserialized"
    rows[runtime.MODEL_URL] = body
    assets.append({"kind": "checkpoint", "path": "weights/sagittal_model.pt", "url": runtime.MODEL_URL,
                   "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "git_blob_sha1": None})
    return assets, rows


class PublicStageTests(unittest.TestCase):
    def test_offline_pack_exact_primary_pins_flags_and_feature_contract(self):
        before = "torch" in sys.modules
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pack"
            value = builder.prepare(output)
            payload = json.loads((output / "save-payload.json").read_bytes())
            self.assertEqual(payload["text"].encode(), (output / "source.ipynb").read_bytes())
            self.assertEqual(len(value["assets"]), 9)
            weight = value["assets"][-1]
            self.assertEqual((weight["bytes"], weight["sha256"]), (552919566, builder.MODEL_SHA))
            self.assertFalse(payload["enable_gpu"])
            self.assertTrue(payload["enable_internet"])
            self.assertTrue(payload["is_private"])
            self.assertFalse(any(payload[key] for key in ("dataset_data_sources", "competition_data_sources", "kernel_data_sources")))
            self.assertFalse(value["decision"]["authorize_download"])
            self.assertFalse(value["decision"]["pending_experiment_score_return_required"])
            self.assertEqual(value["feature_contract"]["model"]["feature_timestep"], 100)
            self.assertEqual(value["feature_contract"]["model"]["feature"], "mid_2")
            self.assertFalse(value["feature_contract"]["existing_sampled16_cache_is_faithful_contiguous16"])
            self.assertEqual(value["feature_contract"]["fixed_pooling"]["trainable_pool_parameters"], 0)
            proposal = (output / "feature-vendor-proposal.py").read_text(encoding="utf-8")
            ast.parse(proposal)
            self.assertNotIn("class Trainer", proposal)
            self.assertNotIn("import nibabel", proposal)
            self.assertNotIn("from apex", proposal)
            with self.assertRaises(FileExistsError):
                builder.prepare(output)
        self.assertEqual("torch" in sys.modules, before)

    def test_disabled_or_goal_already_met_never_opens_URL(self):
        assets, rows = synthetic_assets()
        for decision, expected in (({"authorize_download": False, "known_own_official_score": None}, "DISABLED_UNTIL_OPERATOR_ENABLES_DOWNLOAD"),
                                   ({"authorize_download": True, "known_own_official_score": .960}, "STOP_OWN_GOAL_ALREADY_MET"),
                                   ({"authorize_download": True, "known_own_official_score": .961}, "STOP_OWN_GOAL_ALREADY_MET")):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
                opener = Opener(rows)
                result = runtime.stage(assets, {}, decision, work=temporary, opener=opener)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["asset_request_attempts"], 0)
                self.assertEqual(opener.calls, [])
                self.assertFalse((Path(temporary) / "orthodiffusion-sagittal-stage").exists())

    def test_positive_fake_streams_publish_complete_bytes_and_AST_only(self):
        assets, rows = synthetic_assets()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            opener = Opener(rows)
            result = runtime.stage(assets, {"fake_contract": True}, {"authorize_download": True,
                                   "known_own_official_score": .957}, work=temporary, opener=opener)
            self.assertTrue(result["complete"])
            self.assertEqual(result["asset_request_attempts"], 9)
            root = Path(temporary) / "orthodiffusion-sagittal-stage"
            for asset in assets:
                self.assertEqual((root / asset["path"]).read_bytes(), rows[asset["url"]])
            package = json.loads((root / "asset-manifest.json").read_bytes())
            self.assertFalse(package["checkpoint_loaded_or_qualified"])
            self.assertTrue(result["source_import_inventory"])
            self.assertFalse(any(path.name.startswith(".orthodiff-owned-") for path in Path(temporary).iterdir()))
            with self.assertRaises(ValueError):
                runtime.stage(assets, {}, {"authorize_download": True}, work=temporary, opener=opener)

    def test_truncated_oversize_wrong_SHA_and_wrong_blob_never_publish(self):
        original, rows = synthetic_assets()
        for mode in ("SHA", "blob", "size", "truncated"):
            assets = copy.deepcopy(original)
            if mode == "SHA":
                assets[-1]["sha256"] = "0" * 64
            elif mode == "blob":
                assets[0]["git_blob_sha1"] = "0" * 40
            elif mode == "size":
                assets[-1]["bytes"] -= 1
            else:
                assets[-1]["bytes"] += 1
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
                result = runtime.stage(assets, {}, {"authorize_download": True}, work=temporary, opener=Opener(rows))
                self.assertFalse(result["complete"])
                self.assertEqual(result["status"], "FAILED_NO_MODEL_PACKAGE_PUBLISHED")
                self.assertFalse((Path(temporary) / "orthodiffusion-sagittal-stage").exists())
                self.assertFalse(any(path.name.startswith(".orthodiff-owned-") for path in Path(temporary).iterdir()))

    def test_stream_size_guard_works_without_length_header(self):
        assets, rows = synthetic_assets()
        asset = assets[-1]
        class Chunked:
            def open(self, request, timeout):
                response = Response(rows[request.full_url] + b"excess", request.full_url)
                response.headers = {}
                return response
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            runtime.stream_asset(asset, temporary, Chunked())
        class Truncated:
            def open(self, request, timeout):
                response = Response(rows[request.full_url][:-1], request.full_url)
                response.headers = {}
                return response
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            runtime.stream_asset(asset, temporary, Truncated())

    def test_url_redirect_name_and_digest_rejections(self):
        assets, _ = synthetic_assets()
        for url in ("http://huggingface.co/a", "https://user:secret@huggingface.co/a", "https://127.0.0.1/a",
                    "https://huggingface.co.evil.test/a", "https://other.example/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                runtime.public_https(url)
        runtime.public_https("https://cas-bridge.xethub.hf.co/public-file?temporary-signature=not-recorded")
        for key, value in (("path", "../outside"), ("url", "https://huggingface.co/other"), ("sha256", None)):
            bad = copy.deepcopy(assets[-1])
            bad[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                runtime.validate_asset(bad)
        with self.assertRaises(ValueError):
            runtime.decision_status({"authorize_download": True, "known_own_official_score": float("nan")})

    def test_existing_permission_allows_enabled_pack_without_pending_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = builder.prepare(Path(temporary) / "enabled", authorize_download=True, known_own_score=.957)
            self.assertEqual(runtime.decision_status(value["decision"]), "PROCEED")
            self.assertFalse(value["decision"]["pending_experiment_score_return_required"])
            self.assertFalse(value["actual_asset_download_or_execution_performed"])


if __name__ == "__main__":
    unittest.main()
