"""Audit a saved progress checkpoint and its JSON evidence; never contact Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "experiments/F006-bothfold-formal-Submit57169291-Public956-checkpoint-20261014.json"
MAX_METADATA_BYTES = 2_000_000
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
CREDENTIAL_NAMES = {"kaggle.json", "credentials.json", "client_secret.json", "access_token"}
CREDENTIAL_DIRECTORIES = {".aws", ".codex", ".config", ".ssh"}


def reject_credentials(parts):
    if (parts and parts[-1].casefold() in CREDENTIAL_NAMES
            or any(part.casefold() in CREDENTIAL_DIRECTORIES for part in parts[:-1])):
        raise ValueError("Credential files and authentication directories must not be read")


def contained_path(root, relative):
    """Resolve only workspace-relative paths, checking links before any file read."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("A nonempty relative path is required")
    portable = relative.replace("\\", "/")
    windows = PureWindowsPath(relative)
    parts = PurePosixPath(portable).parts
    if windows.drive or windows.root or portable.startswith("/") or ".." in parts or ":" in portable:
        raise ValueError("Absolute paths and parent traversal are forbidden")
    reject_credentials(parts)
    resolved = root.joinpath(*parts).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("Path resolves outside the repository or to its root")
    reject_credentials(resolved.relative_to(root).parts)
    return resolved


def json_bytes(path):
    reject_credentials((path.name,))
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON metadata evidence may be read")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("JSON metadata exceeds the audit size limit")
    with path.open("rb") as stream:
        body = stream.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise ValueError("JSON metadata exceeds the audit size limit")
    return body


def descriptors(value, location="checkpoint"):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield location, value
        for key, child in value.items():
            yield from descriptors(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from descriptors(child, f"{location}[{index}]")


def audit_evidence(root, location, descriptor):
    result = {
        "location": location,
        "path": descriptor["path"],
        "expected_bytes": descriptor.get("bytes"),
        "expected_sha256": descriptor["sha256"],
    }
    expected_size, expected_hash = result["expected_bytes"], result["expected_sha256"]
    if (not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash)
            or (expected_size is not None and (type(expected_size) is not int or expected_size < 0))):
        return {**result, "status": "invalid_descriptor"}
    try:
        path = contained_path(root, descriptor["path"])
        if path.suffix.lower() != ".json":
            raise ValueError("Only JSON metadata evidence may be read")
        if not path.exists():
            return {**result, "status": "missing"}
        if not path.is_file():
            raise ValueError("Evidence is not a regular JSON file")
        body = json_bytes(path)
    except ValueError as error:
        return {**result, "status": "rejected", "reason": str(error)}
    except OSError as error:
        return {**result, "status": "unreadable", "reason": type(error).__name__}
    actual_hash = hashlib.sha256(body).hexdigest()
    size_matches = expected_size is None or len(body) == expected_size
    hash_matches = actual_hash == expected_hash.lower()
    return {
        **result,
        "status": "matched" if size_matches and hash_matches else "mismatch",
        "actual_bytes": len(body),
        "actual_sha256": actual_hash,
        "size_matches": size_matches,
        "sha256_matches": hash_matches,
        "scope": "Byte identity only; contents and Kaggle authenticity are not reverified",
    }


def probability(value, field):
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid {field}") from error
    if not number.is_finite() or not 0 <= number <= 1:
        raise ValueError(f"Invalid {field}")
    return number


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("An explicit recorded UTC observation timestamp is required")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Observation timestamp must include its timezone")
    return moment.astimezone(timezone.utc)


def audit_progress(root=ROOT, checkpoint=DEFAULT_CHECKPOINT, *, now=None):
    root = Path(root).resolve(strict=True)
    path = contained_path(root, checkpoint)
    body = json_bytes(path)
    saved = json.loads(body)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must be a JSON object")
    adopted = saved["adopted_current_best"]
    score = probability(saved["own_confirmed_Public_best"], "recorded Public score")
    if probability(adopted["own_official_Public"], "adopted Public score") != score:
        raise ValueError("Recorded best and adopted score disagree")
    target = probability(saved["goal_target"], "goal target")
    if saved["goal_operator"] != ">=":
        raise ValueError("This audit supports the recorded >= target only")
    observed = timestamp(saved["official_observation_at_UTC"])
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None:
        raise ValueError("Audit time must include its timezone")
    age = int((now - observed).total_seconds())
    evidence = [audit_evidence(root, location, item) for location, item in descriptors(saved)]
    official = next((item for item in evidence if item["location"] == "checkpoint.evidence.official"), None)
    quota = saved["quota_after_Submit_scope"]
    unresolved = saved["unresolved_formal_submissions"]
    return {
        "schema_version": "recorded_progress_evidence_audit_v1",
        "audit_at_UTC": now.astimezone(timezone.utc).isoformat(),
        "checkpoint": {
            "path": path.relative_to(root).as_posix(),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "recorded_basis_at_UTC": saved["checkpoint_basis_at_UTC"],
        },
        "recorded_Public": {
            "candidate": adopted["candidate"],
            "submission_ref": adopted["submission_ref"],
            "last_recorded_score": str(score),
            "last_recorded_official_observation_at_UTC": observed.isoformat(),
            "observation_age_seconds": age,
            "observation_timestamp_in_future": age < 0,
            "raw_official_evidence_file_status": official["status"] if official else "not_referenced",
            "evidence_interpretation": "Checkpoint claim; file hashes do not authenticate Kaggle or scores",
            "live_verified": False,
            "current_official_score": None,
        },
        "goal": {
            "operator": ">=",
            "target": str(target),
            "gap_from_recorded_score": str(max(Decimal(0), target - score)),
            "recorded_score_meets_target": score >= target,
            "current_achievement_verified": False,
            "achievement_probability": None,
            "scope": "Arithmetic on rounded recorded scores; no prediction of future Public improvement",
        },
        "unresolved_in_checkpoint": [
            {"candidate": name, "submission_ref": reference,
             "recorded_Public": saved["own_Public_by_target"].get(name), "current_status": None}
            for name, reference in unresolved.items()
        ],
        "historical_quota": {
            "derived_at_UTC": saved["checkpoint_basis_at_UTC"],
            "derived_today": quota["derived_today"],
            "derived_remaining": quota["derived_remaining"],
            "derived_total": quota["derived_total"],
            "post_Submit_official_limits_observed": quota["post_Submit_GetSubmissionLimits_performed_for_this_record"],
            "origin": "Historical same-session derivation; not current official limits",
            "current_remaining": None,
            "fresh_official_limits_required_before_external_action": True,
        },
        "evidence": evidence,
        "evidence_status_counts": dict(sorted(Counter(item["status"] for item in evidence).items())),
        "limits": {
            "network_or_authentication_used": False,
            "credential_files_or_authentication_directories_read": False,
            "model_deserialization_or_MRI_decode_or_training": False,
            "original_evidence_files_modified": False,
            "live_submission_state_or_private_accuracy_verified": False,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Repository-relative JSON path")
    parser.add_argument("--output-dir", required=True, help="New repository-relative directory; never overwrite")
    args = parser.parse_args(argv)
    try:
        root = args.repo_root.resolve(strict=True)
        output = contained_path(root, args.output_dir)
        if output.exists():
            raise ValueError("Preserve old results; use a new output directory")
        report = audit_progress(root, args.checkpoint)
        output.mkdir(parents=True, exist_ok=False)
        with (output / "audit.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps({
        "report": (output / "audit.json").relative_to(root).as_posix(),
        "last_recorded_Public": report["recorded_Public"]["last_recorded_score"],
        "live_verified": False,
        "evidence_status_counts": report["evidence_status_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
