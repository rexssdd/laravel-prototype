#!/usr/bin/env python3
"""
predict.py -- inference worker for the Laravel prototype.

Reads one CSV, runs all three models, prints JSON to stdout.
Depends on numpy, pandas and joblib only. No GPU, no sklearn, no imblearn,
no training-script classes.

    python3 predict.py --bundle inference_bundle.joblib --input traffic.csv

Exit codes: 0 ok, 2 bad input, 3 bad bundle, 1 anything else.
"""

import argparse
import json
import math
import sys
import traceback

import numpy as np
import pandas as pd
import joblib


EPS = 1e-9
MAX_ROWS = 300_000          # supports the complete 257,673-row UNSW-NB15 dataset


# ============================================================ isolation forest

def score_forest(forest, X):
    """Vectorized path-length scoring from flat node arrays."""

    n = X.shape[0]
    total = np.zeros(n, dtype=np.float64)
    max_depth = forest["max_depth"]

    for tree in forest["trees"]:

        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        c_adjust = tree["c_adjust"]

        node = np.zeros(n, dtype=np.int32)
        depth = np.zeros(n, dtype=np.float64)

        for _ in range(max_depth + 1):

            f = feature[node]
            internal = f >= 0

            if not internal.any():
                break

            safe_f = np.where(internal, f, 0).astype(np.int64)
            values = np.take_along_axis(X, safe_f[:, None], axis=1)[:, 0]

            go_left = values < threshold[node]
            nxt = np.where(go_left, left[node], right[node])

            node = np.where(internal, nxt, node).astype(np.int32)
            depth = depth + internal.astype(np.float64)

        total = total + depth + c_adjust[node]

    mean_depth = total / float(forest["n_trees"])
    return np.power(2.0, -(mean_depth / forest["c_n"]))


# ==================================================================== svm

def svm_decision(spec, frame):
    """RBF decision function rebuilt from support vectors."""

    X = frame[spec["columns"]].to_numpy(dtype=np.float64)
    X = (X - spec["scaler_mean"]) / spec["scaler_scale"]
    # RBFSVMClassifier.decision_function casts after StandardScaler.
    X = X.astype(np.float32).astype(np.float64)

    SV = spec["support_vectors"]
    gamma = spec["gamma"]
    support_norms = (SV ** 2).sum(1)[None, :]

    # chunk so a large upload cannot blow up memory on the kernel matrix
    out = np.empty(len(X), dtype=np.float64)
    chunk = max(1, int(2_000_000 / max(1, len(SV))))

    for start in range(0, len(X), chunk):
        stop = min(start + chunk, len(X))
        block = X[start:stop]

        sq = (
            (block ** 2).sum(1)[:, None]
            - 2.0 * block @ SV.T
            + support_norms
        )
        np.maximum(sq, 0.0, out=sq)

        out[start:stop] = np.exp(-gamma * sq) @ spec["dual_coef"]

    return spec["sign"] * (out + spec["intercept"])


# ================================================== feature reconstruction

def build_features(raw, bundle):
    """
    Must mirror create_enhanced_features() in the training script exactly.
    The feature_order check at the end is the guard against drift.
    """

    features = bundle["shukla_features"]
    X = raw[features].copy()

    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(bundle["train_medians"]))

    # min-max using the TRAINING range
    if "minmax_scale" in bundle and "minmax_offset" in bundle:
        scale = bundle["minmax_scale"]
        offset = bundle["minmax_offset"]
    else:
        ranges = np.asarray(bundle["minmax_range"], dtype=np.float64)
        scale = 1.0 / np.where(ranges < 10 * np.finfo(np.float64).eps, 1.0, ranges)
        offset = -np.asarray(bundle["minmax_min"]) * scale
    scaled = X[features].to_numpy(dtype=np.float64) * scale + offset

    # --- isolation forest scores, one per scale
    scale_columns = {}

    ordered_scales = [bundle["if_sample_size"]] + [
        scale for scale in bundle["if_scales"] if scale != bundle["if_sample_size"]
    ]
    for scale in ordered_scales:
        scores = score_forest(bundle["forests"][scale], scaled)
        name = "if_score" if scale == bundle["if_sample_size"] \
            else f"if_score_s{scale}"
        scale_columns[name] = scores
        X[name] = scores

    matrix = np.column_stack([scale_columns[k] for k in scale_columns])
    X["if_scale_spread"] = matrix.std(axis=1)
    X["if_scale_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    X["if_scale_mean"] = matrix.mean(axis=1)

    # --- engineered features
    for column in ["rate", "sload", "dload", "sbytes", "dmean", "ackdat"]:
        if column in X.columns:
            X[f"log_{column}"] = np.log1p(np.abs(X[column]))

    if {"sload", "dload"} <= set(X.columns):
        X["load_ratio"] = X["sload"] / (np.abs(X["dload"]) + EPS)
    if {"sbytes", "rate"} <= set(X.columns):
        X["byte_rate_ratio"] = X["sbytes"] / (np.abs(X["rate"]) + EPS)
    if {"sttl", "dttl"} <= set(X.columns):
        X["ttl_ratio"] = X["sttl"] / (np.abs(X["dttl"]) + EPS)
    if {"ackdat", "rate"} <= set(X.columns):
        X["ack_rate_ratio"] = X["ackdat"] / (np.abs(X["rate"]) + EPS)

    if {"sttl", "dttl"} <= set(X.columns):
        X["ttl_difference"] = X["sttl"] - X["dttl"]
    if {"sload", "dload"} <= set(X.columns):
        X["load_difference"] = X["sload"] - X["dload"]

    for column in [
        "rate", "sload", "dload", "sbytes", "dmean", "sttl", "dttl", "ackdat"
    ]:
        if column in X.columns:
            X[f"if_x_{column}"] = X["if_score"] * X[column]

    X["if_score_squared"] = X["if_score"] ** 2
    X["if_score_sqrt"] = np.sqrt(np.abs(X["if_score"]))

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(pd.Series(bundle["enhanced_medians"]))
    X = X.replace([np.inf, -np.inf], 0.0)

    missing = [c for c in bundle["feature_order"] if c not in X.columns]
    if missing:
        raise ValueError(
            "Feature reconstruction drifted from training. Missing: "
            + ", ".join(missing[:8])
        )

    return X[bundle["feature_order"]], scale_columns["if_score"]


# ================================================================= metrics

def metrics(y_true, y_pred, scores=None):
    """Confusion-matrix metrics, computed without sklearn."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    denom = math.sqrt(
        float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN)
    )
    mcc = ((TP * TN) - (FP * FN)) / denom if denom else 0.0

    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "f0_5": 1.25 * precision * recall / (0.25 * precision + recall)
        if (0.25 * precision + recall) else 0.0,
        "fpr": FP / (FP + TN) if (FP + TN) else 0.0,
        "fnr": FN / (FN + TP) if (FN + TP) else 0.0,
        "mcc": mcc,
        "roc_auc": None,
        "pr_auc": None,
        "tn": TN, "fp": FP, "fn": FN, "tp": TP,
    }

    # ROC-AUC via rank statistic -- no sklearn needed
    if scores is not None and 0 < y_true.sum() < len(y_true):
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)

        # average ranks within ties
        s_sorted = scores[order]
        i = 0
        while i < len(s_sorted):
            j = i
            while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1

        n_pos = float(y_true.sum())
        n_neg = float(len(y_true) - n_pos)
        result["roc_auc"] = (
            ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
        ) / (n_pos * n_neg)

    if scores is not None:
        scores = np.asarray(scores)
        order = np.argsort(-scores, kind="mergesort")
        labels = y_true[order]
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
        positives = np.cumsum(labels)[ends]
        precision_at_threshold = positives / (ends + 1)
        result["pr_auc"] = float(
            np.sum(np.diff(np.r_[0, positives]) * precision_at_threshold) / y_true.sum()
        ) if y_true.sum() else 0.0

    return result


def summarize(pred, scores):
    n = len(pred)
    attacks = int(pred.sum())
    return {
        "rows": n,
        "attacks": attacks,
        "normal": n - attacks,
        "attack_rate": attacks / n if n else 0.0,
        "mean_score": float(np.mean(scores)),
    }


# ==================================================================== main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    try:
        bundle = joblib.load(args.bundle)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"Could not load the model bundle: {exc}",
        }))
        return 3

    try:
        raw = pd.read_csv(args.input, low_memory=False)
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"Could not read the CSV: {exc}",
        }))
        return 2

    raw.columns = raw.columns.astype(str).str.strip().str.lower()

    if len(raw) > MAX_ROWS:
        raw = raw.head(MAX_ROWS)
        truncated = True
    else:
        truncated = False

    missing = [c for c in bundle["shukla_features"] if c not in raw.columns]
    if missing:
        print(json.dumps({
            "ok": False,
            "error": "CSV is missing required columns: " + ", ".join(missing),
            "required": bundle["shukla_features"],
            "found": list(raw.columns)[:60],
        }))
        return 2

    if raw.empty:
        print(json.dumps({"ok": False, "error": "CSV has no rows."}))
        return 2

    if "label" in raw.columns:
        labels = pd.to_numeric(raw["label"], errors="coerce")
        if not labels.isin([0, 1]).all():
            print(json.dumps({"ok": False, "error": "Labels must be 0 (normal) or 1 (attack)."}))
            return 2
        y = labels.to_numpy(dtype=int)

    enhanced, if_scores = build_features(raw, bundle)

    # ---- model 1: Shukla et al. (2023) close-replication baseline
    # Uses ONLY the nine Shukla features after train-fitted MinMax scaling,
    # T=100 and S=256 from the exported baseline forest. Extra IF scales and
    # engineered features below are never used to make this baseline decision.
    if bundle["if_label_switched"]:
        if_pred = np.where(if_scores > bundle["if_threshold"], 0, 1)
        if_rank = -if_scores
    else:
        if_pred = np.where(if_scores > bundle["if_threshold"], 1, 0)
        if_rank = if_scores

    # ---- model 2: standalone SVM
    svm_scores = svm_decision(bundle["svm_baseline"], enhanced)
    svm_pred = (svm_scores >= bundle["threshold_baseline"]).astype(int)

    # ---- model 3: hybrid IF-SVM
    hyb_scores = svm_decision(bundle["svm_hybrid"], enhanced)
    hyb_pred = (hyb_scores >= bundle["threshold_hybrid"]).astype(int)

    payload = {
        "ok": True,
        "file": args.name or args.input,
        "rows": int(len(raw)),
        "truncated": truncated,
        "labelled": "label" in raw.columns,
        "meta": bundle.get("meta", {}),
        "official_test_candidate": int(len(raw)) == int(bundle.get("meta", {}).get("official_test_rows", 82332)),
    }

    if "label" in raw.columns:
        payload["actual"] = {
            "attacks": int(y.sum()),
            "normal": int((y == 0).sum()),
            "attack_rate": float(y.mean()),
        }
        payload["baseline"] = metrics(y, if_pred, if_rank)
        payload["svm"] = metrics(y, svm_pred, svm_scores)
        payload["hybrid"] = metrics(y, hyb_pred, hyb_scores)
    else:
        payload["baseline"] = summarize(if_pred, if_scores)
        payload["svm"] = summarize(svm_pred, svm_scores)
        payload["hybrid"] = summarize(hyb_pred, hyb_scores)

    # first 50 rows for the preview table
    payload["preview"] = [
        {
            "row": int(i),
            "if_score": round(float(if_scores[i]), 6),
            "baseline": int(if_pred[i]),
            "svm": int(svm_pred[i]),
            "hybrid": int(hyb_pred[i]),
            "hybrid_score": round(float(hyb_scores[i]), 6),
            **(
                {"label": int(y[i])} if "label" in raw.columns else {}
            ),
        }
        for i in range(min(50, len(raw)))
    ]

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(json.dumps({
            "ok": False,
            "error": "Unhandled error in predict.py",
            "trace": traceback.format_exc(limit=4),
        }))
        sys.exit(1)
