"""Isolated inner4 common-C selection; outer2 has metadata/predictions only.

No CLI here performs any upload, target release or outer2 evaluation. The old
fixed-C1 module remains immutable. See the pinned prospective declaration.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from . import orthodiff_weak_head as H
from .contracts import ID, TARGETS

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "experiments/orthodiff-regional-innerC-prospective-policy-20261017.json"
POLICY_SHA = "dd26d15c5b27c624c08eeee84e65d828b4c0df572854dc301af149954a1c2c6f"
REVIEW_PATH = ROOT / "artifacts/research/20261017-orthodiff-regional-inner4-C-prospective-v1/root-design-review.json"
REVIEW_SHA = "b496cdd2363b9d708fa5a1570833fa40473a0be7b8f2f58d13a21f289de33ea5"
H_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
LOADER_SHA = "1363055c2d9493fd9018b8d9c0f7f0f1da52dba6277a30bbc9c865e12add855d"
COHORT_SHA = "86d976f55bb67bd3a738679625c9281aead06b0c7f01d7c894ffe870ead65246"
BUNDLE_SHA = "d892f0f94a6b98a172c6644564e8769ad8395b3a2223093313b95062123340d5"
GRID = (0.001, 0.01, 0.1, 1.0)
INNER_FOLDS = (0, 1, 3, 4)
META = (ID, "group_id", "fold", "source")
VIEW_SCHEMA = "orthodiff_regional_inner4_training_only_labels_v1"
RUN_SCHEMA = "orthodiff_regional_inner4_C_outer2_sealed_run_v1"
require = H.require
predict_numeric = H.predict_numeric


class HoldRecipe(ValueError):
    """Expected qualification failure; no alternative recipe or retry allowed."""


def prepare_sources(policy_path, expected_policy_sha, expected_producer_sha,
                    cli_path, expected_cli_sha, expected_review_sha):
    for path, expected, label in (
        (policy_path, expected_policy_sha, "policy"),
        (__file__, expected_producer_sha, "producer"),
        (cli_path, expected_cli_sha, "CLI"),
        (REVIEW_PATH, expected_review_sha, "root design review"),
    ):
        require(type(expected) is str and len(expected) == 64
                and H.pin(path)["sha256"] == expected, label + " SHA differs")
    require(expected_policy_sha == POLICY_SHA and expected_review_sha == REVIEW_SHA,
            "Exact prospective policy and root design review required")
    require(H.pin(POLICY_PATH)["sha256"] == POLICY_SHA, "Tracked policy copy differs")
    require(H.pin(H.__file__)["sha256"] == H_SHA, "Immutable H source differs")
    policy, review = H.read_json(policy_path), H.read_json(REVIEW_PATH)
    require(policy["schema_version"] == "orthodiff_regional1024_inner4_common_C_outer2_policy_v1"
            and review["decision"] == "ALLOW_ISOLATED_IMPLEMENTATION_AND_ARTIFICIAL_CPU_QA_ONLY"
            and review["policy"]["sha256"] == POLICY_SHA, "Policy/review binding differs")
    for relative, expected in policy["immutable_before_pins"].items():
        H.check_pin(ROOT / relative, expected)
    require(audit_fit_AST(), "Parameterized fit changed more than its C argument")
    return policy


def project_training_rows(header, token_rows, cohort_rows):
    """Parse syntax tokens, then project metadata BEFORE touching target tokens.

    csv.reader can process all bytes; there is deliberately no whole-row target
    dict or target-array construction before the heldout branch.
    """
    require(len(header) == len(set(header)) and set((*META, *TARGETS)).issubset(header),
            "Unique required weak CSV columns missing")
    indices = {name: header.index(name) for name in (*META, *TARGETS)}
    training, heldout, group_folds = [], [], {}
    count = 0
    for count, tokens in enumerate(token_rows, 1):
        require(count <= len(cohort_rows) and len(tokens) == len(header), "Weak row count/width differs")
        metadata = {name: tokens[indices[name]] for name in META}
        require(metadata == cohort_rows[count - 1] and metadata["source"] == "report_weak",
                "Original UID/group/fold/source order differs")
        require(group_folds.setdefault(metadata["group_id"], metadata["fold"]) == metadata["fold"],
                "Supplied/report group crosses folds")
        if metadata["fold"] == "2":
            heldout.append(metadata)
            continue
        require(int(metadata["fold"]) in INNER_FOLDS, "Unexpected outer-training fold")
        labels = H.labels_from_cells([tokens[indices[name]] for name in TARGETS])
        training.append({**metadata, "targets": [None if np.isnan(x) else float(x) for x in labels]})
    require(count == len(cohort_rows) and len({r[ID] for r in cohort_rows}) == len(cohort_rows),
            "Exact unique metadata cohort required")
    return {"schema_version": VIEW_SCHEMA, "target_order": list(TARGETS),
            "training": training, "outer2_metadata": heldout,
            "outer2_target_cells_projected_converted_or_retained": False,
            "shared_CSV_syntax_bytes_may_include_heldout_targets": True}


def load_training_view(manifest_dir, cohort_path):
    require(H.pin(cohort_path)["sha256"] == COHORT_SHA, "Original metadata cohort SHA differs")
    cohort = H.read_json(cohort_path)
    require(cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0
            and len(cohort["requested"]) == 4207, "Original weak/no-Gold metadata differs")
    weak_path = Path(manifest_dir) / "weak.csv"
    H.check_pin(weak_path, cohort["manifest_pins"]["weak.csv"])
    # Do not open Gold target files or unrelated fields, including Report.
    with weak_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        view = project_training_rows(next(reader), reader, cohort["requested"])
    require(len(view["training"]) == 3319 and len(view["outer2_metadata"]) == 888,
            "Exact outer2/train population differs")
    require(Counter(r["fold"] for r in view["training"]) == Counter({"0": 821, "1": 891, "3": 778, "4": 829}),
            "Original inner4 populations differ")
    view["cohort"] = {"path": str(Path(cohort_path).resolve()), **H.pin(cohort_path)}
    view["weak_source"] = {"path": str(weak_path.resolve()), **H.pin(weak_path)}
    return view


def _uid_hash(rows):
    return hashlib.sha256("".join(r[ID] + "\n" for r in rows).encode()).hexdigest()


def _labels(rows):
    values = np.asarray([[np.nan if x is None else x for x in r["targets"]] for r in rows], dtype=np.float64)
    require(values.shape == (len(rows), len(TARGETS))
            and np.all(np.isnan(values) | np.isin(values, [0, 1])), "Observed0/1 or missing labels required")
    return values


def _arrays(rows, features):
    require(rows and len({r[ID] for r in rows}) == len(rows), "Nonempty unique training rows required")
    x = np.asarray([features[r[ID]] for r in rows], dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] > 0 and np.isfinite(x).all(), "Finite feature matrix required")
    return x, _labels(rows)


def _qualify_fixed10(model, probabilities):
    require([e["target"] for e in model["heads"]] == list(TARGETS), "Head order differs")
    require(probabilities.ndim == 2 and probabilities.shape[1] == len(TARGETS), "Prediction shape differs")
    reasons = []
    for i, name in enumerate(TARGETS):
        entry = model["heads"][i]
        if name in H.FIXED10 and (entry["status"] != "qualified" or entry.get("warnings")
                                  or not np.isfinite(probabilities[:, i]).all()):
            reasons.append(name + ":unqualified_warning_or_nonfinite")
    return reasons


def fit_at_C(train_x, train_y, valid_x, C):
    require(type(C) in (float, int) and float(C) in GRID, "C must belong to the frozen four-value grid")
    return _fit_numeric_C(train_x, train_y, valid_x, float(C))


def support_and_partitions(training_rows, features):
    require(len({r[ID] for r in training_rows}) == len(training_rows)
            and all(int(r["fold"]) in INNER_FOLDS for r in training_rows),
            "Training input must contain only unique outertrain original-fold rows")
    group_folds = {}
    for row in training_rows:
        require(group_folds.setdefault(row["group_id"], row["fold"]) == row["fold"], "Group crosses inner folds")
    successful = [r for r in training_rows if r[ID] in features]
    partitions, support = [], []
    for fold in INNER_FOLDS:
        train = [r for r in successful if int(r["fold"]) != fold]
        valid = [r for r in successful if int(r["fold"]) == fold]
        tx, ty = _arrays(train, features)
        vx, vy = _arrays(valid, features)
        require(tx.shape[1] == vx.shape[1], "Inner feature dimension differs")
        per_target = {}
        for i, name in enumerate(TARGETS):
            if name not in H.FIXED10:
                continue
            a, b = ty[np.isfinite(ty[:, i]), i], vy[np.isfinite(vy[:, i]), i]
            if len(np.unique(a)) != 2 or len(np.unique(b)) != 2:
                raise HoldRecipe("Fixed10 inner support unavailable: fold" + str(fold) + ":" + name)
            per_target[name] = {"train_observed": len(a), "valid_observed": len(b),
                                "valid_observed_UID_hash": _uid_hash([r for r in valid if r["targets"][i] is not None])}
        support.append({"inner_fold": fold, "train_count": len(train), "valid_count": len(valid),
                        "train_UID_hash": _uid_hash(train), "valid_UID_hash": _uid_hash(valid),
                        "fixed10_support": per_target})
        partitions.append((fold, tx, ty, vx, vy))
    return partitions, support


def choose_C(table):
    require([row["C"] for row in table] == list(GRID), "Exact declared C grid/order required")
    eligible = [r for r in table if r["eligible"]]
    if not eligible:
        raise HoldRecipe("All four Cs ineligible; no replacement grid or retries")
    require(all(math.isfinite(r["score"]) for r in eligible), "Eligible C has nonfinite score")
    maximum = max(r["score"] for r in eligible)
    return min(r["C"] for r in eligible if maximum - r["score"] <= 1e-12)


def select_common_C(training_rows, features, fit_function=None, progress=None):
    fitter = fit_at_C if fit_function is None else fit_function
    partitions, support = support_and_partitions(training_rows, features)
    table = []
    for C in GRID:
        result = {"C": C, "eligible": True, "score": None, "inner": [], "failure_reason": None}
        for fold, tx, ty, vx, vy in partitions:
            try:
                model, probabilities, numerical = fitter(tx, ty, vx, C)
                probabilities = np.asarray(probabilities, dtype=np.float64)
                require(probabilities.shape == vy.shape, "Inner predictions not aligned")
                reasons = _qualify_fixed10(model, probabilities)
                if reasons:
                    raise HoldRecipe(";".join(reasons))
                aucs = {name: H.metric(vy[:, i], probabilities[:, i])["auc"]
                        for i, name in enumerate(TARGETS) if name in H.FIXED10}
                require(len(aucs) == 10 and all(x is not None and math.isfinite(x) for x in aucs.values()),
                        "All40 fixed10 AUCs must be defined")
                row = {"fold": fold, "AUC": aucs, "macro10": sum(aucs.values()) / 10,
                       "numeric_roundtrip": numerical,
                       "auxiliary_status": {x["target"]: x["status"] for x in model["heads"] if x["target"] not in H.FIXED10}}
                result["inner"].append(row)
                if progress is not None:
                    progress({"event": "inner_fold_complete", "C": C, "fold": fold, "macro10": row["macro10"]})
            except (ValueError, FloatingPointError) as exc:
                result["eligible"] = False
                result["failure_reason"] = {"inner_fold": fold, "type": type(exc).__name__, "message": str(exc)}
                break
        if result["eligible"]:
            require(len(result["inner"]) == 4, "Exactly four scores required")
            result["score"] = sum(row["macro10"] for row in result["inner"]) / 4
        table.append(result)
        if progress is not None:
            progress({"event": "C_complete", "C": C, "eligible": result["eligible"], "score": result["score"]})
    try:
        selected = choose_C(table)
    except HoldRecipe as exc:
        exc.selection = {"schema_version": "orthodiff_regional_inner4_C_selection_v1", "table": table,
                         "selected_C": None, "support": support, "outer2_targets_used": False,
                         "inner_validation_labels_used_for_C_selection": True}
        raise
    return {"schema_version": "orthodiff_regional_inner4_C_selection_v1", "table": table,
            "selected_C": selected, "common_C_all12": True, "support": support,
            "inner_validation_labels_used_for_C_selection": True,
            "legacy_roundtrip_validation_label_flag_scope": "False refers only to each isolated fit_numeric_C call; this new outer algorithm explicitly uses inner-validation labels for C selection.",
            "outer2_target_or_prediction_used_for_selection": False,
            "tie_tolerance": 1e-12, "tie_preference": "smallest_C"}


def refit_selected(training_rows, outer2_metadata, features, selected_C, fit_function=None):
    require(selected_C in GRID and all(int(r["fold"]) in INNER_FOLDS for r in training_rows),
            "Selected C/outertrain membership differs")
    require(all(set(r) == set(META) and r["fold"] == "2" for r in outer2_metadata),
            "Outer2 input must be metadata only, without any target field")
    train_uids, valid_uids = {r[ID] for r in training_rows}, {r[ID] for r in outer2_metadata}
    require(len(train_uids) == len(training_rows) and len(valid_uids) == len(outer2_metadata)
            and not train_uids & valid_uids
            and not {r["group_id"] for r in training_rows} & {r["group_id"] for r in outer2_metadata},
            "Train/outer2 UID or group alignment differs")
    successful_train = [r for r in training_rows if r[ID] in features]
    successful_valid = [r for r in outer2_metadata if r[ID] in features]
    tx, ty = _arrays(successful_train, features)
    require(successful_valid, "No successful outer2 features")
    vx = np.asarray([features[r[ID]] for r in successful_valid], dtype=np.float64)
    fitter = fit_at_C if fit_function is None else fit_function
    try:
        model, probabilities, numerical = fitter(tx, ty, vx, selected_C)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        require(probabilities.shape == (len(successful_valid), len(TARGETS)), "Outer2 predictions not aligned")
        reasons = _qualify_fixed10(model, probabilities)
        if reasons:
            raise HoldRecipe(";".join(reasons))
    except (ValueError, FloatingPointError) as exc:
        raise HoldRecipe("Selected C refit failed; do not try second-best C: " + str(exc)) from exc
    model["selected_common_C"] = float(selected_C)
    return model, probabilities, numerical, successful_train, successful_valid


def write_predictions(path, uids, values, precision):
    require(precision in ("float64", "float32") and np.asarray(values).shape == (len(uids), len(TARGETS))
            and len(set(uids)) == len(uids)
            and np.all(np.isfinite(values) | np.isnan(values)), "Prediction output contract differs")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID, *TARGETS])
        for uid, row in zip(uids, np.asarray(values)):
            writer.writerow([uid] + ["" if np.isnan(x) else format(float(np.float32(x)) if precision == "float32" else float(x),
                                                                   ".9g" if precision == "float32" else ".17g") for x in row])


def audit_fit_AST():
    old = ast.parse(inspect.getsource(H.fit_numeric)).body[0]
    new = ast.parse(inspect.getsource(_fit_numeric_C)).body[0]
    new.name = old.name
    require(new.args.args[-1].arg == "chosen_C", "C argument missing")
    new.args.args.pop()
    changed = 0
    for node in ast.walk(new):
        if isinstance(node, ast.keyword) and node.arg == "C" and isinstance(node.value, ast.Name) and node.value.id == "chosen_C":
            node.value = ast.Constant(value=1.0)
            changed += 1
    return changed == 1 and ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def _fit_numeric_C(train_x, train_y, valid_x, chosen_C):
    """One fixed CPU fit per supported target; no validation selection or retries."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    train_x, train_y, valid_x = (np.asarray(value, dtype=np.float64) for value in (train_x, train_y, valid_x))
    require(train_x.ndim == valid_x.ndim == 2 and train_x.shape[1] == valid_x.shape[1]
            and train_y.shape == (len(train_x), len(TARGETS)) and len(train_x) > 0
            and np.isfinite(train_x).all() and np.isfinite(valid_x).all(), "Finite train/valid feature arrays required")
    require(np.all(np.isnan(train_y) | np.isin(train_y, [0, 1])), "Train labels must be observed0/1 or missing")
    with threadpool_limits(limits=1):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train_x)
        transformed_train, transformed_valid = scaler.transform(train_x), scaler.transform(valid_x)
        model = {"schema_version": "orthodiff_weak_head_numeric_model_v1", "targets": list(TARGETS),
                 "feature_dim": train_x.shape[1], "dtype": "float64", "seed": 20261002,
                 "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                            "variance": scaler.var_.tolist(), "training_samples": int(scaler.n_samples_seen_),
                            "fit_scope": "successful unchanged training-fold vectors only"}, "heads": []}
        sklearn_predictions = np.full((len(valid_x), len(TARGETS)), np.nan, dtype=np.float64)
        for i, target in enumerate(TARGETS):
            mask = np.isfinite(train_y[:, i])
            labels = train_y[mask, i]
            entry = {"target": target, "train_observed": int(mask.sum()),
                     "train_positive": int(np.count_nonzero(labels == 1)), "train_negative": int(np.count_nonzero(labels == 0)),
                     "train_missing": int((~mask).sum())}
            if len(np.unique(labels)) != 2:
                model["heads"].append({**entry, "status": "insufficient_training_classes", "coef": None, "intercept": None})
                continue
            estimator = LogisticRegression(solver="lbfgs", C=chosen_C, penalty="l2", fit_intercept=True,
                                           class_weight=None, max_iter=2000, tol=1e-4, random_state=20261002)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(transformed_train[mask], labels)
            messages = [{"category": row.category.__name__, "message": str(row.message)} for row in caught]
            converged = not any(row["category"] == "ConvergenceWarning" for row in messages)
            finite = np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all()
            if not converged or not finite:
                model["heads"].append({**entry, "status": "nonconverged_or_nonfinite", "warnings": messages, "coef": None, "intercept": None})
                continue
            require(np.array_equal(estimator.classes_, [0., 1.]), "Binary class order differs")
            prediction = estimator.predict_proba(transformed_valid)[:, 1] if len(valid_x) else np.empty(0)
            require(np.isfinite(prediction).all(), "Nonfinite estimator probability")
            sklearn_predictions[:, i] = prediction
            model["heads"].append({**entry, "status": "qualified", "coef": estimator.coef_[0].tolist(),
                                   "intercept": float(estimator.intercept_[0]), "iterations": int(estimator.n_iter_[0]), "warnings": messages})
    # JSON roundtrip is the deployed model representation, independent of sklearn objects.
    reloaded = json.loads(json.dumps(model, allow_nan=False))
    independent = predict_numeric(reloaded, valid_x)
    supported = np.isfinite(sklearn_predictions)
    require(np.array_equal(np.isfinite(independent), supported), "Numeric model supported-mask changed")
    require(np.allclose(independent[supported], sklearn_predictions[supported], rtol=1e-12, atol=2e-15), "Numeric predictor differs from sklearn")
    require(np.array_equal(independent[supported].astype(np.float32).view(np.uint32),
                           sklearn_predictions[supported].astype(np.float32).view(np.uint32)), "Numeric predictor/sklearn differ after FP32 cast")
    roundtrip = {"JSON_model_roundtrip": True, "sklearn_NumPy_max_abs_difference":
                 float(np.max(np.abs(independent[supported] - sklearn_predictions[supported]))) if supported.any() else 0.,
                 "sklearn_NumPy_FP32_bitwise_equal": True, "pickle_joblib_used": False,
                 "CPU_threads": 1, "validation_labels_used_for_fit_or_selection": False}
    return reloaded, independent, roundtrip

