"""Transfer integrity, resumption and Notebook selection without network access."""

import hashlib
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import patch

from scripts.download_cache_output import comparable_path, list_output, safe_path, transfer_file, validate_export


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload[:2]
        yield self.payload[2:]


def metadata(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class TransferTests(unittest.TestCase):
    def test_windows_extended_paths_keep_containment_check(self):
        root = PureWindowsPath("C:/repo/data/export")
        target = comparable_path("\\\\?\\C:\\repo\\data\\export\\cache.npz")
        self.assertEqual(target.relative_to(root), PureWindowsPath("cache.npz"))
        with self.assertRaises(ValueError):
            comparable_path("\\\\?\\C:\\outside\\cache.npz").relative_to(root)
        unc_root = PureWindowsPath("//server/share/export")
        self.assertEqual(
            comparable_path("\\\\?\\UNC\\server\\share\\export\\cache.npz").relative_to(unc_root),
            PureWindowsPath("cache.npz"),
        )

    def test_remote_paths_cannot_escape_or_use_windows_devices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(safe_path(root, "raw/train.csv"), root.resolve() / "raw/train.csv")
            for name in ("../outside", "/absolute", "a//b", "a\\b", "a:stream", "CON.txt", "a./b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_path(root, name)

    def test_resume_checks_hash_and_replaces_corruption_atomically(self):
        payload = b"valid output"
        calls = []

        def get(*args, **kwargs):
            calls.append(True)
            return Response(payload)

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "cache.npz"
            target.write_bytes(b"wrong output")  # Same byte count, different SHA-256.
            self.assertEqual(transfer_file("https://example.test/file", target, metadata(payload), get), "downloaded")
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(target.with_name("cache.npz.part").exists())
            self.assertEqual(transfer_file("https://example.test/file", target, metadata(payload), get), "reused")
            self.assertEqual(len(calls), 1)

    def test_failed_download_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.download_cache_output.time.sleep"):
            target = Path(folder) / "cache.npz"
            target.write_bytes(b"previous")
            with self.assertRaises(ValueError):
                transfer_file(
                    "https://example.test/file", target, metadata(b"expected"), lambda *a, **k: Response(b"tampered")
                )
            self.assertEqual(target.read_bytes(), b"previous")

    def test_export_rejects_debug_missing_metadata_and_dicom(self):
        names = (
            "audit.json",
            "raw/train.csv",
            "raw/train_series.csv",
            "raw/sample_submission.csv",
            "train-v1/cache.json",
            "train-v1/coverage.json",
            "code/configs/baseline.json",
        )
        record = {
            "schema_version": 1,
            "complete": True,
            "files": {name: metadata(b"") for name in names},
            "output_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as folder:
            validate_export(record, folder)
            with self.assertRaises(ValueError):
                validate_export({**record, "complete": False}, folder)
            with self.assertRaises(ValueError):
                validate_export({**record, "files": {}}, folder)
            with self.assertRaises(ValueError):
                validate_export({**record, "files": {**record["files"], "raw/image.dcm": metadata(b"")}}, folder)


class OutputSelectionTests(unittest.TestCase):
    def client(self, versions):
        versions = iter(versions)
        pages = iter(
            [
                SimpleNamespace(
                    files=[SimpleNamespace(file_name="a", url="https://example.test/a")], next_page_token="next"
                ),
                SimpleNamespace(
                    files=[SimpleNamespace(file_name="b", url="https://example.test/b")], next_page_token=""
                ),
            ]
        )
        requests = []

        def output(request):
            requests.append(request)
            return next(pages)

        return SimpleNamespace(
            get_kernel=lambda request: SimpleNamespace(metadata=SimpleNamespace(current_version_number=next(versions))),
            list_kernel_session_output=output,
        ), requests

    def sdk(self):
        return patch.dict(
            "sys.modules",
            {
                "kagglesdk.kernels.types.kernels_api_service": SimpleNamespace(
                    ApiGetKernelRequest=SimpleNamespace,
                    ApiListKernelSessionOutputRequest=SimpleNamespace,
                ),
            },
        )

    def test_all_pages_and_matching_version_before_and_after(self):
        client, requests = self.client([1, 1])
        with self.sdk():
            self.assertEqual(set(list_output(client, "owner", "slug", 1)), {"a", "b"})
        self.assertEqual(requests[1].page_token, "next")

    def test_version_change_during_listing_is_rejected(self):
        client, _ = self.client([1, 2])
        with self.sdk(), self.assertRaises(ValueError):
            list_output(client, "owner", "slug", 1)


if __name__ == "__main__":
    unittest.main()
