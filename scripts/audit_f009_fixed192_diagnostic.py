"""Manual offline fixed192 JSON audit. The default opens no input body."""
import argparse
import json
from pathlib import Path

from rsna_knee.f009_diagnostic_audit import AUDIT_SCHEMA, audit_diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--new-result", type=Path)
    parser.add_argument("--expected-new-sha256")
    parser.add_argument("--old-result", type=Path)
    parser.add_argument("--expected-old-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "SOURCE_PREPARATION_ONLY", "input_bodies_read": 0,
                          "actual_audit_performed": False, "API_GPU_Model_MRI": 0}))
        return 0
    if any(getattr(args, key) is None for key in ("new_result", "expected_new_sha256", "old_result", "expected_old_sha256", "output")):
        parser.error("--execute requires both result paths/SHA pins and a fresh --output")
    if args.output.suffix != ".json" or args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("Fresh JSON output in an existing directory required")
    try:
        result = audit_diagnostic(args.new_result, args.expected_new_sha256, args.old_result, args.expected_old_sha256)
    except Exception as error:
        result = {"schema_version": AUDIT_SCHEMA, "passed": False, "decision": "ERROR", "error_type": type(error).__name__,
                  "error": str(error), "actual_result_files_audited": False,
                  "source_saved_Input_decoder_full_runtime_or_Submit_qualification_issued": False}
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "decision": result["decision"], "coverage": result["coverage"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
