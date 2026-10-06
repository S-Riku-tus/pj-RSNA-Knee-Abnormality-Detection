"""Artificial features/DICOM only: geometry, missingness, provenance, CUDA head and submission."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, sha256, write_csv
from rsna_knee.feature_contracts import (
    feature_contract,
    head_implementation_hash,
    json_hash,
    load_feature_config,
    selection_structure_hash,
    validate_encoder_provenance,
    validate_fold_audit,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "numpy", "PIL"))
HAS_IMAGING = HAS_TORCH and importlib.util.find_spec("pydicom") is not None


def configuration():
    result = load_feature_config(ROOT / "configs/experiments/f001-dinov2-frozen.json")
    result["preprocess"].update(image_size=28, max_series=2, slices_per_series=3)
    result["head"].update(hidden_dim=8, dropout=0)
    result["train"].update(epochs=1, batch_size=2)
    return result


def provenance():
    return {
        "source_url": "synthetic",
        "source_revision": "synthetic",
        "weights_url": "synthetic",
        "license": "synthetic test only",
        "reviewed_at": "2026-10-06",
        "exposure_audit": "artificial fixture",
        "weights_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "pretraining": "generic_dinov2_lvd142m",
        "competition_finetuned": False,
        "gold_used_for_tuning": False,
    }


def fake_cache(root, rows, config, source_csv, split="test", selection_csv=None, limited=False):
    import numpy as np

    root.mkdir()
    contract = feature_contract(config, provenance())
    fingerprint = json_hash(contract)
    write_csv(root / "studies.csv", (ID,), [{ID: row[ID]} for row in rows])
    p = config["preprocess"]
    shape = (p["max_series"], p["slices_per_series"])
    rng = np.random.default_rng(0)
    files = {}
    for row in rows:
        path = root / f"{row[ID]}.npz"
        np.savez_compressed(
            path,
            features=rng.normal(size=(*shape, 384)).astype(np.float16),
            mask=np.ones(shape, dtype=bool),
            positions=np.broadcast_to(np.linspace(0, 1, shape[1], dtype=np.float32), shape),
            planes=np.arange(shape[0], dtype=np.int64),
            flags=np.ones((shape[0], 2), dtype=np.float32),
            fingerprint=fingerprint,
        )
        files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    metadata = {
        "schema_version": "frozen_feature_export_v1",
        "complete": True,
        "limited": limited,
        "split": split,
        "requested": len(rows),
        "contract": contract,
        "fingerprint": fingerprint,
        "studies_csv_sha256": sha256(source_csv),
        "series_csv_sha256": "c" * 64,
        "selection_csv_sha256": sha256(selection_csv) if selection_csv else None,
        "selection_structure_sha256": selection_structure_hash(rows) if split == "train" else None,
        "study_ids_sha256": sha256(root / "studies.csv"),
        "files": files,
    }
    dump_json(root / "export.json", metadata)
    return metadata


class FeatureContractTests(unittest.TestCase):
    def test_notebook_generation_has_no_saved_outputs_and_compiles(self):
        spec = importlib.util.spec_from_file_location(
            "frozen_handoff", ROOT / "scripts/build_frozen_feature_handoff.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for name, cells in builder.notebooks().items():
            saved = json.loads((ROOT / "notebooks/public" / name).read_text(encoding="utf-8"))
            self.assertEqual(saved["cells"], cells)
            for index, cell in enumerate(cells):
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    compile("".join(cell["source"]), f"{name}:{index}", "exec")

    def test_stale_fold_audit_rejected_and_teacher_only_structure_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {"inputs_sha256": {"train": "a" * 64}}
            dump_json(root / "manifest.json", manifest)
            (root / "weak.csv").write_text("weak", encoding="utf-8")
            (root / "gold.csv").write_text("gold", encoding="utf-8")
            audit = {
                "integrity_valid": True,
                "ready_for_training": True,
                "gold_used_for_tuning": False,
                "inputs_sha256": manifest["inputs_sha256"],
                "manifest_sha256": {name: sha256(root / name) for name in ("manifest.json", "weak.csv", "gold.csv")},
            }
            dump_json(root / "audit.json", audit)
            validate_fold_audit(root, root / "audit.json", manifest)
            (root / "gold.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stale"):
                validate_fold_audit(root, root / "audit.json", manifest)
        rows = [{ID: "s", "group_id": "g", "fold": "0", "ACL": "0"}]
        changed = [{**rows[0], "ACL": "1"}]
        self.assertEqual(selection_structure_hash(rows), selection_structure_hash(changed))
        self.assertNotEqual(selection_structure_hash(rows), selection_structure_hash([{**rows[0], "fold": "1"}]))

    def test_pretrained_competition_or_placeholder_provenance_rejected(self):
        good = provenance()
        validate_encoder_provenance(good)
        for key, value in (
            ("competition_finetuned", True),
            ("gold_used_for_tuning", True),
            ("weights_sha256", "TODO"),
            ("exposure_audit", "TODO audit"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_encoder_provenance({**good, key: value})

    def test_pooling_only_config_and_shared_cache_identity(self):
        baseline = load_feature_config(ROOT / "configs/experiments/f001-dinov2-frozen.json")
        candidate = load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json")
        self.assertEqual(feature_contract(baseline, provenance()), feature_contract(candidate, provenance()))
        candidate["head"]["pooling"] = baseline["head"]["pooling"]
        candidate["reason"] = baseline["reason"]
        self.assertEqual(candidate, baseline)


@unittest.skipUnless(HAS_TORCH, "Optional torch unavailable")
class FeatureHeadTests(unittest.TestCase):
    def test_shard_merge_requires_exact_disjoint_coverage(self):
        from rsna_knee.feature_imaging import merge_feature_shards, verify_feature_cache

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = configuration()
            rows = [{ID: f"s{i}", "group_id": f"g{i}", "fold": str(i % 2)} for i in range(4)]
            write_csv(root / "weak.csv", (ID, "group_id", "fold"), rows)
            write_csv(root / "train.csv", (ID,), [{ID: row[ID]} for row in rows])
            for i in range(2):
                metadata = fake_cache(
                    root / f"shard{i}", rows[i::2], config, root / "train.csv", "train", root / "weak.csv"
                )
                metadata["selection_structure_sha256"] = selection_structure_hash(rows)
                dump_json(root / f"shard{i}" / "export.json", metadata)
            merged = merge_feature_shards(
                [root / "shard0", root / "shard1"], root / "weak.csv", root / "merged", config
            )
            self.assertEqual(merged["requested"], 4)
            verify_feature_cache(root / "merged", config)
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                merge_feature_shards([root / "shard0", root / "shard0"], root / "weak.csv", root / "bad", config)
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                merge_feature_shards([root / "shard0"], root / "weak.csv", root / "missing", config)

    def test_mask_and_variable_series_cpu_or_gpu_forward_backward(self):
        import torch

        from rsna_knee.feature_model import FrozenFeatureHead
        from rsna_knee.runtime import masked_bce

        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = configuration()
        features = torch.randn(2, 2, 3, 384, device=device)
        mask = torch.tensor(
            [[[True, True, False], [False, False, False]], [[True, False, False], [True, True, True]]], device=device
        )
        positions = torch.rand(2, 2, 3, device=device)
        planes = torch.tensor([[0, 1], [0, 2]], device=device)
        flags = torch.ones(2, 2, 2, device=device)
        for pooling in ("mean", "attention"):
            config["head"]["pooling"] = pooling
            model = FrozenFeatureHead(config).to(device).eval()
            actual = model(features, mask, positions, planes, flags)
            changed = features.clone()
            changed[~mask] = float("nan")
            torch.testing.assert_close(actual, model(changed, mask, positions, planes, flags))
            observed = torch.ones(2, 12, dtype=torch.bool, device=device)
            observed[:, 1] = False
            labels = torch.zeros_like(actual)
            loss = masked_bce(actual, labels, observed)
            changed_labels = labels.clone()
            changed_labels[:, 1] = 1
            torch.testing.assert_close(loss, masked_bce(actual, changed_labels, observed))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            with self.assertRaises(ValueError):
                model(features, torch.zeros_like(mask), positions, planes, flags)

    def test_cache_corruption_limited_and_encoder_mismatch_refused(self):
        from rsna_knee.feature_imaging import verify_feature_cache

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = configuration()
            rows = [{ID: "one"}]
            write_csv(root / "test.csv", (ID,), rows)
            cache = root / "features"
            metadata = fake_cache(cache, rows, config, root / "test.csv", limited=True)
            with self.assertRaisesRegex(ValueError, "limited"):
                verify_feature_cache(cache, config)
            verify_feature_cache(cache, config, allow_limited=True)
            metadata["limited"] = False
            dump_json(cache / "export.json", metadata)
            with (cache / "one.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "size"):
                verify_feature_cache(cache, config)

    def test_cuda_guard_before_io_and_gold_group_exclusion(self):
        from rsna_knee.feature_runtime import train_feature_head
        from rsna_knee.runtime import check_split

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "GPU terminal"):
                train_feature_head("missing", "missing", "missing", {}, 0)
        rows = [
            {ID: "one", "source": "report_weak", "group_id": "shared", "fold": 0},
            {ID: "two", "source": "report_weak", "group_id": "other", "fold": 1},
        ]
        with self.assertRaisesRegex(ValueError, "Gold group"):
            check_split(rows, [{ID: "gold", "group_id": "shared"}], 0, 5)

    def test_artificial_cuda_fit_checkpoint_and_report_free_submission(self):
        import torch

        from rsna_knee.feature_model import FrozenFeatureHead
        from rsna_knee.feature_runtime import load_head_checkpoint, predict_feature_cache, train_feature_head

        config = configuration()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest"
            manifest.mkdir()
            weak = [
                {
                    ID: f"s{i}",
                    **{target: str(i % 2) for target in TARGETS},
                    "source": "report_weak",
                    "group_id": f"g{i}",
                    "fold": str(i // 2),
                }
                for i in range(4)
            ]
            weak[0][TARGETS[1]] = ""  # observed-mask contract
            write_csv(root / "train.csv", SUBMISSION_COLUMNS, [{ID: r[ID], **{t: "" for t in TARGETS}} for r in weak])
            fields = (*SUBMISSION_COLUMNS, "source", "group_id", "fold")
            write_csv(manifest / "weak.csv", fields, weak)
            write_csv(
                manifest / "gold.csv",
                fields,
                [
                    {
                        ID: "gold",
                        **{t: "1" for t in TARGETS},
                        "source": "official_image_review",
                        "group_id": "gold_group",
                        "fold": -1,
                    }
                ],
            )
            dump_json(
                manifest / "manifest.json",
                {
                    "seed": config["seed"],
                    "n_folds": 5,
                    "label_provenance": {"gold_used_for_tuning": False},
                    "inputs_sha256": {"train": sha256(root / "train.csv")},
                },
            )
            train_meta = fake_cache(
                root / "train_features", weak, config, root / "train.csv", "train", manifest / "weak.csv"
            )
            audit = {
                "integrity_valid": True,
                "ready_for_training": True,
                "gold_used_for_tuning": False,
                "inputs_sha256": {"train": sha256(root / "train.csv")},
                "manifest_sha256": {
                    name: sha256(manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv")
                },
            }
            dump_json(manifest / "fold-audit-v2.json", audit)
            if torch.cuda.is_available():
                result = train_feature_head(manifest, root / "train_features", root / "run", config, 0)
                self.assertEqual(result["status"], "complete")
                self.assertFalse(result["gold_evaluated"])
                checkpoint = root / "run/best.pt"
                with self.assertRaisesRegex(ValueError, "new run"):
                    train_feature_head(manifest, root / "train_features", root / "run", config, 0)
            else:
                checkpoint = root / "synthetic.pt"
                torch.save(
                    {
                        "schema_version": "frozen_feature_head_v1",
                        "head_implementation_sha256": head_implementation_hash(),
                        "model": FrozenFeatureHead(config).state_dict(),
                        "targets": list(TARGETS),
                        "config": config,
                        "feature_contract": train_meta["contract"],
                        "feature_fingerprint": train_meta["fingerprint"],
                        "selection_data": "weak_validation",
                    },
                    checkpoint,
                )
            rows = [{ID: "test1"}, {ID: "test2"}]
            write_csv(root / "test.csv", (ID,), rows)
            fake_cache(root / "test_features", rows, config, root / "test.csv")
            result = predict_feature_cache(
                root / "test.csv", root / "test_features", checkpoint, root / "submission.csv", "cpu"
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["studies"], 2)
            damaged_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
            damaged_checkpoint["head_implementation_sha256"] = "c" * 64
            torch.save(damaged_checkpoint, root / "stale.pt")
            with self.assertRaisesRegex(ValueError, "Head implementation"):
                load_head_checkpoint(root / "stale.pt")
            with self.assertRaisesRegex(ValueError, "new prediction"):
                predict_feature_cache(
                    root / "test.csv", root / "test_features", checkpoint, root / "submission.csv", "cpu"
                )
            wrong = copy.deepcopy(config)
            wrong["preprocess"]["image_size"] = 42
            with self.assertRaisesRegex(ValueError, "mismatch"):
                from rsna_knee.feature_imaging import verify_feature_cache

                verify_feature_cache(root / "test_features", wrong)


@unittest.skipUnless(HAS_IMAGING, "Optional imaging unavailable")
class FeatureGeometryTests(unittest.TestCase):
    def test_native_physical_order_triplets_and_non_square_spacing(self):
        import numpy as np
        import pydicom
        from test_imaging_runtime import synthetic_dicom

        from rsna_knee.feature_imaging import encode_windows, letterbox, read_feature_windows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, z in enumerate([10, 0, 5]):
                path = root / f"{index}.dcm"
                synthetic_dicom(path, z, index)
                data = pydicom.dcmread(path)
                data.PixelSpacing = [1, 2]
                data.save_as(path, enforce_file_format=True)
            windows, positions, coverage = read_feature_windows(root, configuration()["preprocess"])
            self.assertEqual(windows.shape, (3, 3, 28, 28))
            np.testing.assert_array_equal(positions, [0, 0.5, 1])
            self.assertEqual(coverage["order"], "physical_position")
            self.assertEqual(coverage["pixel_spacing_mm"], [1, 2])
            np.testing.assert_array_equal(windows[0, 0], windows[0, 1])
            boxed = letterbox(np.ones((10, 10), np.uint8), 28, [1, 2])
            self.assertEqual(int(boxed.sum()), 14 * 28)
            import torch

            config = configuration()
            config["encoder"]["precision"] = "fp32"

            class ArtificialEncoder(torch.nn.Module):
                def forward(self, x):
                    return x.mean((1, 2, 3))[:, None].expand(-1, 384)

            encoded = encode_windows(ArtificialEncoder(), windows, config, "cpu")
            self.assertEqual(encoded.shape, (3, 384))
            self.assertEqual(encoded.dtype, np.float16)
            data = pydicom.dcmread(root / "0.dcm")
            del data.PixelSpacing
            data.save_as(root / "0.dcm", enforce_file_format=True)
            with self.assertRaisesRegex(ValueError, "PixelSpacing"):
                read_feature_windows(root, configuration()["preprocess"])


if __name__ == "__main__":
    unittest.main()
