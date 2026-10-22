"""Manual future amended-qualification F009 strict CPU proof; default does not open any run/model."""

import argparse
import json
from pathlib import Path

from rsna_knee.f009_cpu_proof_v2 import produce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--expected-spec-sha256")
    parser.add_argument("--expected-producer-sha256")
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "SOURCE_ONLY_NOT_EXECUTED",
                    "actual_CPU_strict_reconstruction": False,
                    "run_or_checkpoint_or_encoder_binary_opened": False,
                }
            )
        )
        return
    if any(
        getattr(args, name) is None
        for name in ("spec", "expected_spec_sha256", "expected_producer_sha256", "fold", "output", "repo", "weights")
    ):
        parser.error("--execute requires all spec/fold/output/assets/source arguments")
    produce(
        args.spec,
        args.expected_spec_sha256,
        args.fold,
        args.output,
        args.repo,
        args.weights,
        args.expected_producer_sha256,
    )


if __name__ == "__main__":
    main()
