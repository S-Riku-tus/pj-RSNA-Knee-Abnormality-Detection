"""Manual offline F009 pair pack/unpack. Default opens no input/model body."""
import argparse
import json
from pathlib import Path

from rsna_knee.f009_transport import pack_f009_pair, unpack_f009_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("pack", "unpack"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-sha256")
    parser.add_argument("--pair-bytes", type=int)
    parser.add_argument("--complete-sha256")
    parser.add_argument("--complete-bytes", type=int)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--fold", type=int, choices=(0, 1))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "SOURCE_PREPARATION_ONLY", "input_model_or_run_opened": False,
                          "actual_transport_created": False, "API_upload_Save_Submit": 0}))
        return
    if args.mode is None or args.input is None or args.output is None:
        parser.error("--execute requires --mode --input --output and explicit mode-specific pins")
    if args.mode == "pack":
        if any(getattr(args, key) is None for key in ("pair_sha256", "pair_bytes", "complete_sha256", "complete_bytes", "plan")):
            parser.error("pack requires --pair-sha256/--pair-bytes/--complete-sha256/--complete-bytes/--plan")
        pack_f009_pair(args.input, args.output, expected_pair_pin={"bytes": args.pair_bytes, "sha256": args.pair_sha256},
                       expected_complete_pin={"bytes": args.complete_bytes, "sha256": args.complete_sha256}, conditional_plan_path=args.plan)
    else:
        if args.manifest_sha256 is None or args.fold is None:
            parser.error("unpack requires --manifest-sha256 and --fold")
        unpack_f009_transport(args.input, args.output, expected_manifest_sha256=args.manifest_sha256, fold=args.fold)


if __name__ == "__main__":
    main()
