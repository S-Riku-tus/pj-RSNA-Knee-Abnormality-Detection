"""Extract the audited author224 numerical source into a fresh local folder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from p010_author224_runtime import (
    AUTHOR_NOTEBOOK_SHA256,
    CHECKPOINT_SHA256,
    STOCK_SOURCE_SHA256,
    SYMBOL_SHA256,
    build_stock_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = build_stock_source(args.author_notebook)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != STOCK_SOURCE_SHA256:
        raise ValueError("Extracted stock source differs from the fixed runtime contract")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "author224_stock.py").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    receipt = {
        "schema_version": "p010_author224_literal_extraction_v1", "status": "source_only",
        "author_notebook": str(args.author_notebook), "author_saved_version": 2,
        "author_script_version_id": 356179962, "author_notebook_sha256": AUTHOR_NOTEBOOK_SHA256,
        "stock_source_sha256": digest, "stock_source_bytes": len(source.encode()),
        "definition_source_sha256": SYMBOL_SHA256,
        "infer_probs_decorators": ["torch.no_grad()"], "defaults_preserved": True,
        "timm_import": "namespace-injected; no processwide import state mutation",
        "checkpoint_expected_sha256": CHECKPOINT_SHA256, "actual_checkpoint_read": False,
        "MRI_decode": False, "GPU_forward": False, "remote_operations": False,
    }
    with (args.output_dir / "extraction-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "source_only", "stock_source_sha256": digest}, ensure_ascii=True))


if __name__ == "__main__":
    main()
