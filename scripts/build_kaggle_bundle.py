"""Package an explicit allowlist only; never upload or submit anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "src" / "rsna_knee").glob("*.py")) + [ROOT / "configs" / "baseline.json"]
    bundle = output_dir / "rsna-knee-code.zip"
    manifest = {}
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            name = source.relative_to(ROOT).as_posix()
            data = source.read_bytes()
            # Fixed timestamps and permissions make builds reproducible.
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
    result = {"file": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(), "sources": manifest}
    (output_dir / "bundle-manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "kaggle")
    print(json.dumps(build(parser.parse_args().output_dir), indent=2))
