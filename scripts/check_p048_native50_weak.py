"""Fixed heldout-family50 CSV check before P048 execution; no Torch or APIs."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import analyze_native_ensemble_diversity as D
from diagnose_f013_f009_matched import matched_counters

from rsna_knee.ensemble_diversity import fixed_mean, pair_transitions
from rsna_knee.metrics import binary_auc, evaluate_rows

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/research/20261020-P048-native50-paired-weak-prospective-v1"
POLICY_SHA = "0f24b3625e4d555795cadfe4409ba81087d2c5dcdec5c5f17775993173c1842f"


def pin(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    D.require(not (RUN / "result.json").exists(), "Fresh fixed-comparison output")
    policy_path = RUN / "policy.json"
    D.require(pin(policy_path)["sha256"] == POLICY_SHA, "Frozen before-metric native50 policy")
    policy = json.loads(policy_path.read_bytes())
    for ref in policy["pins"].values():
        D.require(pin(ROOT / ref["path"]) == {k: ref[k] for k in ("bytes", "sha256")}, "Original source/proof/manifest pin")
    original = json.loads((ROOT / policy["pins"]["original_pair"]["path"]).read_bytes())
    qualified = json.loads((ROOT / policy["pins"]["F013_policy"]["path"]).read_bytes())
    arms = {}
    # Complete all six metadata/body-hash/epoch/heldout checks before reading
    # any saved prediction values or weak target values.
    for fold, items in policy["inputs"].items():
        arms[fold] = {}
        for name, ref in items.items():
            for role in ("run", "prediction", "checkpoint"):
                D.require(pin(ROOT / ref[role]["path"]) == {k: ref[role][k] for k in ("bytes", "sha256")}, "Fixed run/CSV/checkpoint hash")
            run, epoch, path = D.read_run(ROOT / ref["directory"], ref["rule"])
            D.require(run["fold"] == int(fold) and run["seed"] == 20261002
                      and path == ROOT / ref["prediction"]["path"], "Exact fold/seed/prediction path")
            expected = (qualified["fixed_checkpoints"][fold] if name == "candidate"
                        else original["family_pairs"]["REG" if name == "REG" else "CLS"]["bindings"][fold])
            wanted_SHA = expected["sha256"] if name == "candidate" else expected["checkpoint_sha256"]
            wanted_epoch = 12 if name != "REG" else expected["selected_epoch"]
            D.require(ref["checkpoint"]["sha256"] == wanted_SHA and epoch["epoch"] == wanted_epoch,
                      "Exact serving checkpoint/epoch, never BCE-selected CLS substitution")
            D.require(run["config"] == expected["config"] and run["encoder_provenance"] == expected["encoder_provenance"], "Exact serving config/generic provenance")
            D.require(run["audit"]["manifest_sha256"]["weak.csv"] == policy["pins"]["manifest"]["sha256"], "Fixed teacher manifest")
            arms[fold][name] = (run, epoch, path)
        control, candidate = (arms[fold][n][0] for n in ("control", "candidate"))
        for run, _, _ in arms[fold].values():
            D.require(run["audit"]["manifest_sha256"] == control["audit"]["manifest_sha256"]
                      and run["audit"]["pixel_index_sha256"] == control["audit"]["pixel_index_sha256"], "All three arms same heldout/group/pixel audit")
        if fold == "1":
            matched_counters(control["history"], candidate["history"])
        else:
            D.require(all(row["amp_skipped_steps"] == 0 for run in (control, candidate) for row in run["history"]), "Original fold0 no AMP skips")
    results = {}
    for fold, items in arms.items():
        truth, counts, groups = D.weak_partition(ROOT / policy["pins"]["manifest"]["path"], int(fold))
        predictions = {}
        for name, (run, epoch, path) in items.items():
            values = D.prediction_csv(path, epoch)
            D.require(values.keys() == truth.keys() and len(truth) == run["valid_studies"]
                      and sum(counts.values()) - len(truth) == run["train_studies"], "Same exact full heldout UID support")
            writer_truth = {uid: {t: "" if value is None else str(value) for t, value in row.items()} for uid, row in truth.items()}
            D.require(evaluate_rows(writer_truth, values) == epoch["metrics"], "Independent CSV metrics equal original writer")
            predictions[name] = values
        order, targets = sorted(truth), {}
        for target in D.TARGETS:
            uids = [uid for uid in order if truth[uid][target] in (0, 1)]
            labels = [truth[uid][target] for uid in uids]
            values = {name: [fixed_mean(predictions[name][uid][target], predictions["REG"][uid][target]) for uid in uids]
                      for name in ("control", "candidate")}
            au = {name: binary_auc(labels, v) for name, v in values.items()}
            transitions = pair_transitions(labels, values["control"], values["candidate"])
            delta = au["candidate"] - au["control"] if all(v is not None for v in au.values()) else None
            D.require(delta is None or abs(delta - transitions["auc_delta"]) < 1e-12, "Independent pair-transition AUC delta")
            targets[target] = {"auc": au, "delta": delta, "binary_N": len(uids), "positive": sum(labels),
                    "missing_excluded": sum(truth[uid][target] is None for uid in order),
                    "UID_mask_SHA": hashlib.sha256("\n".join(uids).encode()).hexdigest(), "pair_transitions": transitions}
        fixed = {name: math.fsum(targets[t]["auc"][name] for t in D.FIXED10) / 10 for name in ("control", "candidate")}
        results[fold] = {"heldout_studies": len(truth), "groups_all_folds": groups, "fixed10": fixed,
                         "fixed10_delta": fixed["candidate"] - fixed["control"], "per_target": targets,
                         "scope": "Single matching heldout CLS/REG fold family50; not full serving twofold OOF"}
    D.require("torch" not in sys.modules, "No tensor/model/GPU work")
    positive = all(result["fixed10_delta"] > 0 for result in results.values())
    result = {"schema_version": "P048_fixed_native50_heldout_weak_check_v1", "at_UTC": datetime.now(timezone.utc).isoformat(),
              "policy_SHA": POLICY_SHA, "complete": True, "folds": results,
              "bothfold_fixed10_positive": positive, "P048_weak_family_gate_passed": positive,
              "original_F009_HOLD_unchanged": True, "eligible_for_formal_Submit": False,
              "required_next": "Actual prior/current GPU, currentMRI and independent CSV proof; freshgoal/source/quota/rootreview",
              "limits": policy["uncertainty"] + " No P033 parent sameholdout predictions; no fullpipeline Public gain proof.",
              "model_GPU_MRI_API_Save_Submit_Gold_reads": 0, "source_SHA": pin(__file__)["sha256"]}
    with (RUN / "result.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"bothfold_positive": positive, "folds": {k: v["fixed10"] | {"delta": v["fixed10_delta"]} for k, v in results.items()}}))


if __name__ == "__main__":
    main()
