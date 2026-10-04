"""Explicit official ImageNet initialization download; never called by train or predict."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from torch.hub import download_url_to_file

from rsna_knee.contracts import RESNET18_V1_SHA256_PREFIX, RESNET18_V1_URL, dump_json, sha256


def fetch(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty directory; do not overwrite prior weights or provenance")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resnet18-f37072fd.pth"
    download_url_to_file(RESNET18_V1_URL, str(path), hash_prefix=RESNET18_V1_SHA256_PREFIX, progress=False)
    digest = sha256(path)
    if not digest.startswith(RESNET18_V1_SHA256_PREFIX):
        raise ValueError("Official weight prefix mismatch")
    record = {
        "name": "ResNet18 IMAGENET1K_V1",
        "source_url": RESNET18_V1_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "hash_verification": "Official filename SHA-256 prefix checked; full SHA-256 calculated locally",
        "weight_terms": "Torchvision code BSD-3-Clause does not by itself establish training-dataset/weight rights",
        "weight_terms_source": "https://github.com/pytorch/vision/blob/v0.25.0/README.md#pre-trained-model-license",
        "scope": "Explicit local research initialization; no MRI/data acquisition or Kaggle upload/submission",
    }
    dump_json(output_dir / "provenance.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(fetch(parser.parse_args().output_dir))
