"""Package manually downloaded DINOv2 assets for 06; no network, GPU or uploads."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from rsna_knee.contracts import dump_json, sha256
from rsna_knee.feature_contracts import json_hash, validate_encoder_provenance, validate_fold_audit

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_URL = f"https://github.com/facebookresearch/dinov2/archive/{REVISION}.zip"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
BUNDLE_SHA = "6c8731cfc34489c5ce1cccb60c163829d39e84c482dac64cffcaf0db882a6cce"
BUNDLE = ROOT / "artifacts/kaggle/frozen-features-v1-20261006-final/rsna-frozen-features-code.zip"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
NOTEBOOK = ROOT / "notebooks/public/06_prepare_frozen_features.ipynb"
MANIFEST_FILES = ("weak.csv", "gold.csv", "manifest.json", "fold-audit-v2.json")


def source_members(path):
    """Read the pinned archive safely, without importing or executing any source."""
    prefix = f"dinov2-{REVISION}"
    result = {}
    folded = set()
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 200 * 1024**2:
            raise ValueError("Unexpected source archive size")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (
                not name.parts
                or name.parts[0] != prefix
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in info.filename
                or ":" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("Unexpected source archive member")
            if info.is_dir():
                continue
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if relative == "." or relative.casefold() in folded:
                raise ValueError("Duplicate or invalid source archive member")
            folded.add(relative.casefold())
            result[relative] = archive.read(info)
    if not {"hubconf.py", "LICENSE", "MODEL_CARD.md", "dinov2/__init__.py"}.issubset(result):
        raise ValueError("Expected the complete official DINOv2 source ZIP")
    return result


def configured_notebook(template):
    notebook = json.loads(Path(template).read_text(encoding="utf-8"))
    replacements = {
        "CODE_INPUT = Path('/kaggle/input/your-frozen-feature-code')": (
            "# Change INPUT only if Kaggle displays a different mounted directory.\n"
            "INPUT = Path('/kaggle/input/datasets/rsraki/rsna-frozen-feature-inputs-v1')\n"
            "if not INPUT.is_dir():\n"
            "    INPUT = Path('/kaggle/input/rsna-frozen-feature-inputs-v1')\n"
            "CODE_INPUT = INPUT / 'code'"
        ),
        "ENCODER_REPO = Path('/kaggle/input/your-generic-dinov2/dinov2-source')": (
            "ENCODER_REPO = INPUT / 'encoder/dinov2-source'"
        ),
        "ENCODER_WEIGHTS = Path('/kaggle/input/your-generic-dinov2/dinov2_vits14_pretrain.pth')": (
            "ENCODER_WEIGHTS = INPUT / 'encoder/dinov2_vits14_pretrain.pth'"
        ),
        "PROVENANCE = Path('/kaggle/input/your-generic-dinov2/encoder-provenance.json')": (
            "PROVENANCE = INPUT / 'encoder/encoder-provenance.json'"
        ),
        "MANIFEST = Path('/kaggle/input/your-reviewed-manifest/v1-vmohitrao-research')": (
            "MANIFEST = INPUT / 'manifest'"
        ),
    }
    counts = dict.fromkeys(replacements, 0)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for old, new in replacements.items():
            counts[old] += source.count(old)
            source = source.replace(old, new)
        compile(source, "prepared-06", "exec")
        cell.update(source=source.splitlines(keepends=True), outputs=[], execution_count=None)
    if set(counts.values()) != {1}:
        raise ValueError("Notebook template changed; review its Input assignments")
    return notebook


def package(source_zip, weights, output):
    source_zip, weights, output = Path(source_zip), Path(weights), Path(output).resolve()
    if not output.is_relative_to((ROOT / "artifacts/kaggle").resolve()):
        raise ValueError("Use a new directory inside this repository's artifacts/kaggle")
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    members = source_members(source_zip)
    if weights.name != "dinov2_vits14_pretrain.pth" or weights.stat().st_size == 0:
        raise ValueError("Supply the official dinov2_vits14_pretrain.pth")
    if sha256(BUNDLE) != BUNDLE_SHA:
        raise ValueError("Frozen code bundle has changed")
    manifest = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8-sig"))
    validate_fold_audit(MANIFEST, MANIFEST / "fold-audit-v2.json", manifest)
    manifest_payloads = {name: (MANIFEST / name).read_bytes() for name in MANIFEST_FILES}
    provenance = {
        "source_url": "https://github.com/facebookresearch/dinov2",
        "source_revision": REVISION,
        "source_tree_sha256": json_hash(
            {
                name: hashlib.sha256(data).hexdigest()
                for name, data in members.items()
                if name == "hubconf.py" or (name.startswith("dinov2/") and name.endswith(".py"))
            }
        ),
        "weights_url": WEIGHTS_URL,
        "weights_sha256": sha256(weights),
        "license": "Apache-2.0 for generic DINOv2 code/weights; original archive notices retained",
        "pretraining": "generic_dinov2_lvd142m",
        "competition_finetuned": False,
        "gold_used_for_tuning": False,
        "reviewed_at": "2026-10-06 JST; official README, model card and pinned commit reviewed",
        "exposure_audit": (
            "Intended input is the unmodified generic ViT-S/14 checkpoint manually obtained from the official URL. "
            "Competition fine-tuning and gold selection are not part of the official generic recipe. "
            "LVD-142M overlap with competition patients/images is unknown. Local hashes pin supplied bytes; "
            "no publisher checksum or independent origin authentication was performed."
        ),
    }
    validate_encoder_provenance(provenance)
    notebook = configured_notebook(NOTEBOOK)
    output.mkdir(parents=True)
    bundle = output / "rsna-frozen-feature-inputs-v1.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        # Expand our own code in the Input so Kaggle need not retain a nested ZIP.
        with zipfile.ZipFile(BUNDLE) as code:
            for name in code.namelist():
                if not name.endswith("/"):
                    archive.writestr("code/" + name, code.read(name))
        for name, data in members.items():
            archive.writestr("encoder/dinov2-source/" + name, data)
        archive.write(weights, "encoder/dinov2_vits14_pretrain.pth")
        archive.writestr("encoder/encoder-provenance.json", json.dumps(provenance, indent=2) + "\n")
        for name, data in manifest_payloads.items():
            archive.writestr("manifest/" + name, data)
    # Check that the archived weight bytes match the provenance after the copy.
    with zipfile.ZipFile(bundle) as archive, archive.open("encoder/dinov2_vits14_pretrain.pth") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != provenance["weights_sha256"]:
            raise ValueError("Weights changed during packaging; do not use this output")
    dump_json(output / "06_prepare_frozen_features.ipynb", notebook)
    record = {
        "status": "packaged_not_executed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_url": SOURCE_URL,
        "source_archive_sha256": sha256(source_zip),
        "provenance": provenance,
        "code_bundle_sha256": BUNDLE_SHA,
        "input_zip_sha256": sha256(bundle),
        "manifest_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in manifest_payloads.items()},
        "notebook_sha256": sha256(output / "06_prepare_frozen_features.ipynb"),
        "dataset_visibility": "private; contains the existing study/fold manifest",
        "downloads_performed": False,
        "model_loaded_or_inference_or_training_run": False,
        "uploaded_or_submitted": False,
    }
    dump_json(output / "package.json", record)
    return {"status": record["status"], "upload_zip": str(bundle), "notebook": str(output / NOTEBOOK.name)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True, help=f"Manually download: {SOURCE_URL}")
    parser.add_argument("--weights", type=Path, required=True, help=f"Manually download: {WEIGHTS_URL}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source_zip, args.weights, args.output), indent=2))
