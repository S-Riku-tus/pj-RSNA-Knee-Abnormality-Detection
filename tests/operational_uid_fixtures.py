"""Synthetic native-shaped preparation, including explicit source-bound record fixtures."""

import json
import os
import time


def prepare_operational_fixture(item):
    _, uid, mode, fields, _ = item
    if mode == "timeout":
        time.sleep(60)
    if mode == "exit":
        os._exit(71)
    if mode == "unexpected":
        raise ValueError("Artificial non-whitelisted reader exception")
    if mode == "structural":
        from uid_recovery_adapter import FailedUID

        return FailedUID(**json.loads(fields))
    return ("OTHER-UID" if mode == "wrong_uid" else uid, {"artificial": b"same-array-bytes"},
            {"force_nonfinite": mode == "nonfinite"})
