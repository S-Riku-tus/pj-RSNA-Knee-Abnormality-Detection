"""CPU-only public generic asset staging; importing this module does no I/O.

No torch/pickle/model import or deserialization, MRI, training, Gold, Kaggle API,
upload or Submit exists here. Only explicit later stage() streams public assets.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

CODE_COMMIT = "99d15186bb8face1728263023d8414d9405c8f7e"
WEIGHT_REVISION = "d6920d49e7d1b3dd0774eed6b8e47e386a54dd81"
SOURCE_PATHS = {"LICENSE", "requirements.txt", "configs/config_linear.yaml", "diffusion_model/unet.py",
                "diffusion_model/modules.py", "diffusion_model/fp16_util.py", "diffusion_model/logger.py",
                "diffusion_model/trainer.py"}
MODEL_URL = "https://huggingface.co/lanstat0123/orthodiffusion/resolve/" + WEIGHT_REVISION + "/sagittal_model.pt"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_https(url):
    value = urllib.parse.urlsplit(url)
    host = (value.hostname or "").lower()
    require(value.scheme == "https" and not value.username and not value.password
            and value.port in (None, 443) and (host in {"huggingface.co", "raw.githubusercontent.com"}
            or host.endswith(".huggingface.co") or host.endswith(".hf.co")), "Only official public HTTPS asset hosts allowed")
    return value


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def validate_asset(asset):
    name = asset["path"]
    path = PurePosixPath(name)
    require(isinstance(name, str) and not path.is_absolute() and not any(part in ("", ".", "..") for part in name.split("/"))
            and "\\" not in name and ":" not in name, "Safe fixed relative asset name required")
    require(type(asset["bytes"]) is int and 0 < asset["bytes"] <= 553_000_000, "Positive bounded expected asset size required")
    public_https(asset["url"])
    if asset["kind"] == "checkpoint":
        require(name == "weights/sagittal_model.pt" and asset["url"] == MODEL_URL
                and len(asset.get("sha256", "")) == 64 and asset.get("git_blob_sha1") is None,
                "One fixed sagittal checkpoint with positive LFS SHA required")
    else:
        require(asset["kind"] == "source" and name.startswith("upstream/"), "Only fixed upstream source assets allowed")
        relative = name.removeprefix("upstream/")
        require(relative in SOURCE_PATHS and asset["url"] == "https://raw.githubusercontent.com/lt-0123/OrthoDiffusion/" + CODE_COMMIT + "/" + relative
                and len(asset.get("git_blob_sha1", "")) == 40, "Fixed source commit/blob SHA required")
    for field, length in (("sha256", 64), ("git_blob_sha1", 40)):
        if asset.get(field) is not None:
            require(len(asset[field]) == length and all(char in "0123456789abcdef" for char in asset[field]), "Positive lowercase content digest required")


def stream_asset(asset, destination, opener):
    """Verify bounded bytes before publishing; no retries or alternate URLs."""
    validate_asset(asset)
    target = Path(destination) / PurePosixPath(asset["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = asset["bytes"]
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(expected).encode() + b"\0")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "RSNA-Knee-fixed-public-stage", "Accept-Encoding": "identity"})
    count = 0
    try:
        with opener.open(request, timeout=60) as response:
            require(response.status == 200, "Public asset HTTP200 required")
            public_https(response.geturl())
            length = response.headers.get("Content-Length")
            require(length is None or int(length) == expected, "HTTP expected asset size differs")
            with target.open("xb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    count += len(chunk)
                    require(count <= expected, "Asset exceeded its positive expected size")
                    sha256.update(chunk)
                    blob.update(chunk)
                    output.write(chunk)
            require(count == expected, "Asset truncated before expected size")
            require(asset.get("sha256") is None or sha256.hexdigest() == asset["sha256"], "Asset expected SHA256 differs")
            require(asset.get("git_blob_sha1") is None or blob.hexdigest() == asset["git_blob_sha1"], "Asset expected Git blob SHA1 differs")
            return {"path": asset["path"], "bytes": count, "sha256": sha256.hexdigest(),
                    "git_blob_sha1": blob.hexdigest() if asset["kind"] == "source" else None,
                    "fixed_request_url": asset["url"], "final_public_host": urllib.parse.urlsplit(response.geturl()).hostname,
                    "positive_expected_digests_passed": True}
    except Exception:
        target.unlink(missing_ok=True)
        raise


def source_imports(root):
    rows = {}
    for path in sorted((Path(root) / "upstream").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        rows[path.relative_to(root).as_posix()] = [
            {"module": node.module, "relative_level": node.level, "names": [name.name for name in node.names]}
            if isinstance(node, ast.ImportFrom) else {"module": None, "names": [name.name for name in node.names]}
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    return rows


def decision_status(decision):
    score = decision.get("known_own_official_score")
    if score is not None:
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, "Finite observed own official score required")
        if score >= .960:
            return "STOP_OWN_GOAL_ALREADY_MET"
    require(type(decision.get("authorize_download")) is bool, "Explicit stage execution flag required")
    return "PROCEED" if decision["authorize_download"] else "DISABLED_UNTIL_OPERATOR_ENABLES_DOWNLOAD"


def stage(assets, contract, decision, *, work="/kaggle/working", opener=None):
    work = Path(work)
    final = work / "orthodiffusion-sagittal-stage"
    receipt_path = work / "ORTHODIFF_STAGE_RECEIPT.json"
    require(work.is_dir() and not final.exists() and not receipt_path.exists(), "Fresh owned output directory required")
    names = [asset["path"] for asset in assets]
    require(len(names) == len(set(names)) == 9 and set(names) == {"weights/sagittal_model.pt", *("upstream/" + path for path in SOURCE_PATHS)},
            "Exactly one checkpoint and eight fixed source assets required")
    for asset in assets:
        validate_asset(asset)
    state = {"schema_version": "orthodiffusion_sagittal_public_CPU_stage_v2", "complete": False,
             "at_UTC": datetime.now(timezone.utc).isoformat(), "decision": decision,
             "status": decision_status(decision), "verified_assets": [], "asset_request_attempts": 0,
             "model_deserialization_forward_MRI_GPU_training_Gold_Save_Submit_calls": 0,
             "source_import_or_execution_calls": 0, "own_Public_gain_verified": False}
    try:
        if state["status"] == "PROCEED":
            active = opener if opener is not None else urllib.request.build_opener(PublicRedirects())
            with tempfile.TemporaryDirectory(prefix=".orthodiff-owned-", dir=work) as temporary:
                temporary = Path(temporary)
                for asset in assets:
                    state["asset_request_attempts"] += 1
                    state["verified_assets"].append(stream_asset(asset, temporary, active))
                state.update(complete=True, status="COMPLETE_VERIFIED_PUBLIC_ASSETS_ONLY", source_import_inventory=source_imports(temporary))
                package = {"schema_version": "orthodiffusion_sagittal_fixed_generic_assets_v2", "assets": state["verified_assets"],
                           "feature_contract": contract, "checkpoint_loaded_or_qualified": False,
                           "source_import_graph_inspected_by_AST_only": True}
                (temporary / "asset-manifest.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
                # Same filesystem; publish only the entirely verified package.
                os.replace(temporary, final)
    except Exception as error:
        state.update(complete=False, status="FAILED_NO_MODEL_PACKAGE_PUBLISHED", error_type=type(error).__name__, error=str(error)[:1000])
    with receipt_path.open("x", encoding="utf-8") as output:
        json.dump(state, output, indent=2, allow_nan=False)
        output.write("\n")
    print("ORTHODIFF_PUBLIC_STAGE", json.dumps({"complete": state["complete"], "status": state["status"],
                                               "verified_assets": len(state["verified_assets"]),
                                               "asset_request_attempts": state["asset_request_attempts"]}), flush=True)
    return state
