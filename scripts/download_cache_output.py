"""Download an audited cache export file by file; never upload, decode MRI or train."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

try:
    from .export_paths import comparable_path
except ImportError:  # Running directly as python scripts/download_cache_output.py.
    from export_paths import comparable_path


ROOT = Path(__file__).resolve().parents[1]


def safe_path(root, name):
    """Validate even Windows path syntax before resolving a remote output name."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe export path")
    parts = name.split("/")
    if any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in parts):
        raise ValueError("Unsafe export path")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.split(".")[0].upper() in reserved for part in parts):
        raise ValueError("Reserved Windows export path")
    root = Path(root).resolve()
    target = root.joinpath(*PurePosixPath(name).parts).resolve()
    comparable_path(target).relative_to(comparable_path(root))
    return target


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, metadata):
    return path.is_file() and path.stat().st_size == metadata["bytes"] and file_hash(path) == metadata["sha256"]


def validate_export(record, root):
    if record.get("schema_version") != 1 or record.get("complete") is not True:
        raise ValueError("A complete schema-1 export is required")
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing export file manifest")
    allowed_fixed = {
        "audit.json",
        "raw/train.csv",
        "raw/train_series.csv",
        "raw/sample_submission.csv",
        "train-v1/cache.json",
        "train-v1/coverage.json",
        "code/configs/baseline.json",
    }
    if not allowed_fixed <= files.keys():
        raise ValueError("Export is missing required metadata")
    total = 0
    resolved_paths = set()
    for name, metadata in files.items():
        path = safe_path(root, name)
        if path in resolved_paths:
            raise ValueError("Export names collide on this filesystem")
        resolved_paths.add(path)
        if name not in allowed_fixed and not (
            re.fullmatch(r"train-v1/[^/]+\.npz", name)
            or re.fullmatch(r"code/src/rsna_knee/[A-Za-z_][A-Za-z_0-9]*\.py", name)
        ):
            raise ValueError("Unexpected file type in cache export")
        if not isinstance(metadata, dict) or type(metadata.get("bytes")) is not int or metadata["bytes"] < 0:
            raise ValueError("Invalid export byte count")
        if not isinstance(metadata.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]):
            raise ValueError("Invalid export SHA-256")
        total += metadata["bytes"]
    if total != record.get("output_bytes") or total >= 18_000_000_000:
        raise ValueError("Export byte count differs from the notebook contract")
    return files, total


def transfer_file(url, target, metadata, get, retries=3):
    """Reuse hash-verified files and atomically replace only a completed download."""
    if matches(target, metadata):
        return "reused"
    if urlsplit(url).scheme != "https":
        raise ValueError("Expected HTTPS output URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError("Refuse to write through a partial-file symlink")
    for attempt in range(retries):
        try:
            with get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                digest, received = hashlib.sha256(), 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            received += len(chunk)
                            if received > metadata["bytes"]:
                                raise ValueError("Output exceeds manifest byte count")
                            digest.update(chunk)
                            stream.write(chunk)
                if received != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
                    raise ValueError("Downloaded output hash/size mismatch")
                partial.replace(target)
                return "downloaded"
        except Exception:
            if attempt + 1 == retries:
                # Signed URLs and HTTP response bodies must not appear in error output.
                raise ValueError(f"Transfer failed after retries: {target.name}") from None
            time.sleep(attempt + 1)


def list_output(client, owner, slug, version):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest

    def check_latest():
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        actual = client.get_kernel(request).metadata.current_version_number
        if actual != version:
            raise ValueError("Latest Notebook version differs from requested version; do not mix outputs")

    # Explicit version_label=1 returned 404 for this saved Notebook, while
    # latest output succeeded. Use latest only with matching metadata before
    # and after pagination. Refuse to resume once a newer version is saved.
    check_latest()

    result, token, seen_tokens = {}, None, set()
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.page_size = 200
        if token:
            request.page_token = token
        response = client.list_kernel_session_output(request)
        for item in response.files or []:
            if item.file_name in result:
                raise ValueError("Duplicate filename in paginated output")
            result[item.file_name] = item.url
        token = response.next_page_token
        if not token:
            check_latest()
            return result
        if token in seen_tokens:
            raise ValueError("Output pagination token repeated")
        seen_tokens.add(token)


def fetch_export(url, get):
    if urlsplit(url).scheme != "https":
        raise ValueError("Expected HTTPS export URL")
    chunks, size = [], 0
    with get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            size += len(chunk)
            if size > 16 * 1024 * 1024:
                raise ValueError("Export manifest is unexpectedly large")
            chunks.append(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/slug")
    parser.add_argument("--version", required=True, type=int, help="Notebook Version number, not scriptVersionId")
    parser.add_argument(
        "--script-version-id", required=True, type=int, help="URL ID, recorded as user-supplied context"
    )
    parser.add_argument("--dest", required=True, type=Path, help="New transfer directory under this repository's data/")
    parser.add_argument("--export-name", default="rsna-cache-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--metadata-only", action="store_true", help="Fetch and validate export.json before image transfer"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.kernel):
        parser.error("Expected owner/slug")
    if args.version < 1 or args.script_version_id < 1 or not 1 <= args.workers <= 8:
        parser.error("Positive version/ID and 1..8 workers required")
    dest = args.dest.resolve()
    data_root = (ROOT / "data").resolve()
    data_root.relative_to(ROOT.resolve())
    dest.relative_to(data_root)
    export = safe_path(dest, args.export_name)
    dest.mkdir(parents=True, exist_ok=True)
    transfer_path = dest / "transfer.json"
    context = {
        "kernel": args.kernel,
        "version_number": args.version,
        "script_version_id_user_reported": args.script_version_id,
        "export_name": args.export_name,
    }
    if transfer_path.exists():
        previous = json.loads(transfer_path.read_text(encoding="utf-8"))
        if previous["context"] != context:
            raise ValueError("Use a new destination for a different Notebook version")
    elif export.exists() and any(export.iterdir()):
        raise ValueError("Existing export has no transfer provenance; use a new destination")

    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # Normal CLI authentication; never print/copy credentials.
    owner, slug = args.kernel.split("/")
    with api.build_kaggle_client() as client:
        remote = list_output(client.kernels.kernels_api_client, owner, slug, args.version)
    export_key = f"{args.export_name}/export.json"
    if export_key not in remote:
        raise ValueError("Saved output does not contain the requested export.json")
    payload = fetch_export(remote[export_key], requests.get)
    record = json.loads(payload)
    files, total = validate_export(record, export)
    required = {f"{args.export_name}/{name}" for name in files}
    if not required <= remote.keys():
        raise ValueError("Saved Output is missing manifest files; do not train this export")
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if transfer_path.exists() and previous["export_sha256"] != manifest_hash:
        raise ValueError("Saved export changed since the previous transfer")
    export.mkdir(exist_ok=True)
    (export / "export.json").write_bytes(payload)
    summary = {
        "context": context,
        "version_selection": "latest version matched requested number before and after output pagination",
        "export_sha256": manifest_hash,
        "expected_files": len(files),
        "output_bytes": total,
        "transfer_complete": False,
    }
    transfer_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_valid": True, "files": len(files), "bytes": total}), flush=True)
    if args.metadata_only:
        return
    reusable = {name for name, metadata in files.items() if matches(safe_path(export, name), metadata)}
    remaining = sum(metadata["bytes"] for name, metadata in files.items() if name not in reusable)
    if shutil.disk_usage(dest).free < remaining + 2_000_000_000:
        raise ValueError("Insufficient space for remaining files and 2 GB margin")
    started, reused, downloaded = time.monotonic(), len(reusable), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {
            pool.submit(
                transfer_file, remote[f"{args.export_name}/{name}"], safe_path(export, name), metadata, requests.get
            ): name
            for name, metadata in files.items()
            if name not in reusable
        }
        try:
            for completed in as_completed(tasks):
                completed.result()
                downloaded += 1
                if downloaded % 50 == 0 or downloaded + reused == len(files):
                    print(
                        json.dumps(
                            {
                                "done": downloaded + reused,
                                "total": len(files),
                                "reused": reused,
                                "elapsed_seconds": round(time.monotonic() - started),
                            }
                        ),
                        flush=True,
                    )
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_cache_export import verify

    summary["verification"] = verify(export)
    summary["transfer_complete"] = True
    transfer_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"export_dir": str(export), **summary["verification"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not print exceptions from SDK/requests: they may contain signed URLs.
        print(f"Transfer stopped ({type(error).__name__}). No training was started.", file=sys.stderr)
        if isinstance(error, ValueError):
            print(str(error), file=sys.stderr)
        sys.exit(2)
