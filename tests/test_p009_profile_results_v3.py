"""Duplicate visible messages do not change strict actual inference counts."""

import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p009_result_verifier_v3",
    ROOT / "artifacts/research/20261008-p009-profile32-results-v2/verify_results_v3.py",
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ReaderLoggingContracts(unittest.TestCase):
    def test_duplicate_real_write_messages_do_not_count_as_duplicate_model_inference(self):
        state = {
            "mode": "plain",
            "complete": True,
            "model_count": 3,
            "load_grad_disabled": True,
            "mirror_calls": [],
            "forwards": [
                {
                    "model_index": index % 3,
                    "phase": "plain",
                    "shape": [4, 6, 16, 3, 384, 384],
                    "finite_logits": True,
                    "grad_disabled": True,
                }
                for index in range(24)
            ],
        }
        replay = {"reader_replayed": True, "reader_replay_calls": 1}
        start = "[ConvNeXt Reader] Starting inference on test set...\n"
        finish = "[ConvNeXt Reader] Finished inference in 1.0s\n"
        marker = "wrote /kaggle/working/_own.csv\n"
        for repeats in (1, 2):
            log = start + marker * repeats + finish + start + finish
            result = verifier.reader_logging_contract(log, state, replay)
            self.assertEqual(result["visible_reader_write_message_counts_by_fusion"], [repeats, 0])
            self.assertEqual(result["actual_reader_inference_executions"], 1)
            self.assertEqual(result["real_model_calls"], 24)
        for log in (
            start + finish + start + finish,
            start + marker + finish + start + marker + finish,
            marker + start + marker + finish + start + finish,
            start + marker + finish + marker + start + finish,
        ):
            with self.assertRaises(AssertionError):
                verifier.reader_logging_contract(log, state, replay)
        duplicated = copy.deepcopy(state)
        duplicated["forwards"] *= 2
        with self.assertRaises(AssertionError):
            verifier.reader_logging_contract(start + marker * 2 + finish + start + finish, duplicated, replay)
        with self.assertRaises(AssertionError):
            verifier.reader_logging_contract(
                start + marker + finish + start + finish, state, {"reader_replayed": True, "reader_replay_calls": 2}
            )


if __name__ == "__main__":
    unittest.main()
