"""Diagnostics around the frozen gates; no relaxed model or submission checks."""

import time
import traceback

from p003_guard import RECEIPT_FILES, P003Guard, read_json, save_json, sha256
from p003_profile import P003ProfileGuard


class SpeedDiagnostics:
    def prepare(self):
        self.speed_phases = []
        super().prepare()
        save_json(self.work / "P003_SPEED_CONTRACT.json", self.contract["speed"])

    def run_cell(self, number, source, namespace):
        if number == 27:
            original = namespace["rsna_phase"]

            def phase(name, status, *args, **kwargs):
                result = original(name, status, *args, **kwargs)
                self.speed_phases.append(
                    {"phase": name, "status": status, "elapsed_seconds": time.monotonic() - self.started}
                )
                save_json(self.work / "P003_SPEED_PHASES.json", self.speed_phases)
                return result

            namespace["rsna_phase"] = phase
        try:
            super().run_cell(number, source, namespace)
        except BaseException:
            save_json(
                self.work / "P003_SPEED_FAILURE.json",
                {
                    "cell": number,
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - self.started,
                    "phases": self.speed_phases,
                    "note": "Unhandled exception is not automatically a nine-hour timeout.",
                },
            )
            raise

    def finish(self, namespace):
        runtime = namespace["_p003_speed_schedule"]
        runtime.flush()
        report = read_json(self.work / "P003_SPEED_RAPTOR.json")
        if self.contract["speed"]["mode"] == "profile" and report.get("parity_passed") is not True:
            self.reject("Missing exact native-Raptor profile parity")
            raise ValueError("Missing exact native-Raptor profile parity")
        # Original strict completion, full-cohort rank and fallback gates remain required.
        result = super().finish(namespace)
        summary = {
            "status": "completed_unscored",
            "contract": self.contract["speed"],
            "parent_completion": result,
            "raptor": report,
            "phase_events": self.speed_phases,
            "coat_receipts": {name: read_json(self.work / name) for name in RECEIPT_FILES},
            "original_source_cells": self.contract["speed"]["original_source_cells"],
            "executed_source_cells": self.contract["source_cells"],
            "preflight_sha256": sha256(self.work / "P003_PREFLIGHT.json"),
            "public_score": None,
            "hidden_test_runtime_verified": False,
            "speedup_verified": False,
        }
        save_json(self.work / "P003_SPEED_SUMMARY.json", summary)
        return summary


class P003SpeedGuard(SpeedDiagnostics, P003Guard):
    pass


class P003SpeedProfileGuard(SpeedDiagnostics, P003ProfileGuard):
    pass
