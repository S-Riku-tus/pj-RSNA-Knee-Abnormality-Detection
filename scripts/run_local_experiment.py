"""Run one explicit local research experiment, preserving logs and immutable outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import torch

from rsna_knee.contracts import dump_json, load_config, sha256, source_hashes
from rsna_knee.runtime import ensure_cuda, train


def run(args):
    ensure_cuda()
    config = load_config(args.config)
    out = Path(args.run_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty run directory")
    launches = out.parent / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    record_path, console_path = launches / f"{out.name}.json", launches / f"{out.name}.log"
    if record_path.exists() or console_path.exists():
        raise ValueError("This experiment launch already exists; choose a new run name")
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hashes(),
        "config_path": str(args.config),
        "config_file_sha256": sha256(args.config),
        "config": config,
        "fold": args.fold,
        "reason": args.reason,
        "scope": "Local research only; no Kaggle upload/submission; source eligibility remains separately audited",
        "diagnostic_studies": args.diagnostic_studies,
        "status": "running",
        "elapsed_scope": "train function and final source verification; cache preflight included; Python imports excluded",
    }
    dump_json(record_path, record)
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        with console_path.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            result = train(
                args.manifest_dir, args.cache_dir, out, config, args.fold, diagnostic_studies=args.diagnostic_studies
            )
        record["result"] = result
        record["source_sha256_at_finish"] = source_hashes()
        if record["source_sha256_at_finish"] != record["source_sha256"]:
            record["status"] = "invalid_source_changed"
            raise ValueError("Source changed during the run; audit before accepting its result")
        record["status"] = "completed"
    except BaseException as error:
        if record["status"] != "invalid_source_changed":
            record["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record.update(
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        dump_json(record_path, record)
        if out.is_dir():
            dump_json(out / "execution.json", record)
    print(json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"], **result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--diagnostic-studies", type=int)
    parser.add_argument("--reason", required=True)
    run(parser.parse_args())
