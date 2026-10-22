"""Manual offline metadata assembly. No execution by default; no uploader."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee.f009_release import assemble_f009_release  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-assembler-sha")
    args = parser.parse_args(argv)
    result = assemble_f009_release(args.spec, args.spec_sha, args.output,
        expected_assembler_sha256=args.expected_assembler_sha, execute=args.execute)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
