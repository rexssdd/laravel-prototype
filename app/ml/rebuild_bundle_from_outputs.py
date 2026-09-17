#!/usr/bin/env python3
"""
rebuild_bundle_from_outputs.py
==============================

Builds inference_bundle.joblib from an EXISTING Kaggle output folder, so you
do not have to retrain. Run this in a fresh Kaggle notebook.

REQUIREMENTS
------------
  * GPU accelerator ON. The two SVM pickles wrap cuML objects; without cuml
    importable they will not unpickle at all. This is the one hard blocker.
  * The UNSW-NB15 dataset attached (needed for the training medians and to
    rebuild the multi-scale forests -- those were never saved).
  * The Kaggle output folder from your run, attached as a dataset or sitting
    in /kaggle/working.

WHAT IT REUSES vs REBUILDS
--------------------------
  reused from the pickles : the S=256 Isolation Forest, both fitted SVM
                            pipelines, the MinMaxScaler
  read from the CSVs      : both tuned decision thresholds, the IF
                            contamination cut, the feature order
  rebuilt deterministically: the S=64/128/512 forests (same seeds, same
                            training data -> identical trees), the training
                            and enhanced medians

If anything here drifts from the training run, the feature-order assertion
at the end will catch it before a bad bundle ships.

USAGE
-----
  python rebuild_bundle_from_outputs.py \
      --results  "/kaggle/input/kaggle-test-results" \
      --train    "/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv" \
      --out      "/kaggle/working/inference_bundle.joblib"
"""

import argparse
import math
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_selection import SelectKBest, mutual_info_classif


# ============================================================================
# GPU probe -- cuml only needs to be IMPORTABLE for the pickles to resolve
# ============================================================================

cp = None
try:
    import cupy as cp
    cp.zeros(1).sum()
except Exception:
    cp = None

try:
    import cuml  # noqa: F401  -- required for unpickling, not called directly
    HAS_CUML = True
except Exception:
    HAS_CUML = False


RANDOM_STATE = 42
N_TREES = 100
SAMPLE_SIZE = 256
IF_SCALES = [64, 128, 256, 512]
EPS = 1e-9

SHUKLA_FEATURES = [
    "sttl", "ct_state_ttl", "dload", "rate",
    "dmean", "dttl", "ackdat", "sload", "sbytes",
]

EXPECTED_TRAIN_SIZE = 175341
EXPECTED_TEST_SIZE = 82332


# ============================================================================
# PART 3 classes, verbatim -- pickle resolves against these names
# ============================================================================

@lru_cache(maxsize=None)
def harmonic_number(n):
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))


@lru_cache(maxsize=None)
def c_factor(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)


class FlatIsolationTree:

    def __init__(self, max_depth=8, random_state=0):
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
        self.feature = []
        self.threshold = []
        self.left = []
        self.right = []
        self.c_adjust = []

    def _new_node(self):
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.c_adjust.append(0.0)
        return len(self.feature) - 1

    def _make_leaf(self, node_id, n_samples):
        self.feature[node_id] = -1
        self.left[node_id] = -1
        self.right[node_id] = -1
        self.c_adjust[node_id] = (
            c_factor(int(n_samples)) if n_samples > 1 else 0.0
        )
        return node_id

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self._build(self._new_node(), X, depth=0)
        self.feature = np.asarray(self.feature, dtype=np.int32)
        self.threshold = np.asarray(self.threshold, dtype=np.float64)
        self.left = np.asarray(self.left, dtype=np.int32)
        self.right = np.asarray(self.right, dtype=np.int32)
        self.c_adjust = np.asarray(self.c_adjust, dtype=np.float64)
        return self

    def _build(self, node_id, X, depth):
        n_samples, n_features = X.shape

        if n_samples <= 1 or depth >= self.max_depth or n_features == 0:
            return self._make_leaf(node_id, n_samples)

        feature = int(self.rng.integers(0, n_features))
        values = X[:, feature]
        min_value, max_value = float(np.min(values)), float(np.max(values))

        if min_value == max_value:
            return self._make_leaf(node_id, n_samples)

        threshold = float(self.rng.uniform(min_value, max_value))
        left_mask = values < threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(node_id, n_samples)

        left_id = self._new_node()
        right_id = self._new_node()

        self.feature[node_id] = feature
        self.threshold[node_id] = threshold
        self.left[node_id] = left_id
        self.right[node_id] = right_id
        self.c_adjust[node_id] = 0.0

        self._build(left_id, X[left_mask], depth + 1)
        self._build(right_id, X[right_mask], depth + 1)
        return node_id


class CustomIsolationForest:

    def __init__(self, n_trees=100, sample_size=256, random_state=42,
                 use_gpu=False, verbose=True):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.max_depth = math.ceil(math.log2(self.sample_size))
        self.trees = []

        for i in range(self.n_trees):
            if n_samples > self.sample_size:
                indices = self.rng.choice(
                    n_samples, size=self.sample_size, replace=False
                )
            else:
                indices = np.arange(n_samples)

            tree = FlatIsolationTree(
                max_depth=self.max_depth,
                random_state=self.random_state + i,
            )
            tree.fit(X[indices])
            self.trees.append(tree)

        self._stage_arrays()
        return self

    def _stage_arrays(self):
        xp = cp if (self.use_gpu and cp is not None) else np
        self._xp = xp
        self._staged = [
            (xp.asarray(t.feature), xp.asarray(t.threshold),
             xp.asarray(t.left), xp.asarray(t.right), xp.asarray(t.c_adjust))
            for t in self.trees
        ]

    def anomaly_score(self, X, batch_size=200000):
        xp = getattr(self, "_xp", np)
        c_n = c_factor(self.sample_size)
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(len(X), dtype=np.float64)

        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            Xb = xp.asarray(X[start:stop])
            n = Xb.shape[0]
            total_depth = xp.zeros(n, dtype=xp.float64)

            for feature, threshold, left, right, c_adjust in self._staged:
                node = xp.zeros(n, dtype=xp.int32)
                depth = xp.zeros(n, dtype=xp.float64)

                for _ in range(self.max_depth + 1):
                    f = feature[node]
                    internal = f >= 0
                    if not bool(internal.any()):
                        break
                    safe_f = xp.where(internal, f, 0).astype(xp.int64)
                    values = xp.take_along_axis(
                        Xb, safe_f[:, None], axis=1
                    )[:, 0]
                    go_left = values < threshold[node]
                    nxt = xp.where(go_left, left[node], right[node])
                    node = xp.where(internal, nxt, node).astype(xp.int32)
                    depth = depth + internal.astype(xp.float64)

                total_depth = total_depth + depth + c_adjust[node]

            scores = xp.power(2.0, -((total_depth / float(self.n_trees)) / c_n))
            out[start:stop] = (
                cp.asnumpy(scores) if xp is not np else np.asarray(scores)
            )

        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_staged", None)
        state.pop("_xp", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.trees:
            self._stage_arrays()


# ============================================================================
# Shims -- only need to exist so pickle can restore the attributes
# ============================================================================

class RBFSVMClassifier:
    """Attribute carrier. We read _model, _gamma_value and _sign off it."""
    pass


class KeepColumnSelector(SelectKBest):
    """Needs the real mask logic so get_support() reports what was used."""

    def __init__(self, score_func=mutual_info_classif, k=10, keep_index=None):
        super().__init__(score_func=score_func, k=k)
        self.keep_index = keep_index

    def _get_support_mask(self):
        mask = super()._get_support_mask()
        if self.keep_index is None or self.keep_index >= len(mask):
            return mask
        if mask[self.keep_index]:
            return mask

        scores = np.nan_to_num(self.scores_, nan=-np.inf)
        selected = np.flatnonzero(mask)
        weakest = selected[np.argmin(scores[selected])]

        mask = mask.copy()
        mask[weakest] = False
        mask[self.keep_index] = True
        return mask


# ============================================================================
# Feature engineering -- must mirror the training script exactly
# ============================================================================

def create_enhanced_features(df):
    X = df.copy()

    for feature in ["rate", "sload", "dload", "sbytes", "dmean", "ackdat"]:
        if feature in X.columns:
            X[f"log_{feature}"] = np.log1p(np.abs(X[feature]))

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

    for feature in [
        "rate", "sload", "dload", "sbytes", "dmean", "sttl", "dttl", "ackdat"
    ]:
        if feature in X.columns:
            X[f"if_x_{feature}"] = X["if_score"] * X[feature]

    X["if_score_squared"] = X["if_score"] ** 2
    X["if_score_sqrt"] = np.sqrt(np.abs(X["if_score"]))

    return X.replace([np.inf, -np.inf], np.nan)


# ============================================================================
# Export helpers
# ============================================================================

def export_forest(forest):
    return {
        "n_trees": int(forest.n_trees),
        "sample_size": int(forest.sample_size),
        "max_depth": int(forest.max_depth),
        "c_n": float(c_factor(forest.sample_size)),
        "trees": [
            {
                "feature": np.asarray(t.feature, dtype=np.int32),
                "threshold": np.asarray(t.threshold, dtype=np.float64),
                "left": np.asarray(t.left, dtype=np.int32),
                "right": np.asarray(t.right, dtype=np.int32),
                "c_adjust": np.asarray(t.c_adjust, dtype=np.float64),
            }
            for t in forest.trees
        ],
    }


def export_svm(pipeline, input_columns, label):
    selector = pipeline.named_steps["selector"]
    scaler = pipeline.named_steps["scaler"]
    svc = pipeline.named_steps["svc"]
    inner = svc._model

    mask = selector.get_support()
    columns = [str(c) for c in np.asarray(list(input_columns))[mask]]

    spec = {
        "columns": columns,
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "support_vectors": np.asarray(
            inner.support_vectors_, dtype=np.float64
        ),
        "dual_coef": np.asarray(inner.dual_coef_, dtype=np.float64).ravel(),
        "intercept": float(np.asarray(inner.intercept_).ravel()[0]),
        "gamma": float(svc._gamma_value),
        "sign": float(getattr(svc, "_sign", 1.0)),
    }

    print(
        f"  {label}: {len(columns)} features, "
        f"{spec['support_vectors'].shape[0]} support vectors, "
        f"gamma={spec['gamma']:.6g}, sign={spec['sign']:+.0f}"
    )
    return spec


def read_threshold(path, criterion="MCC"):
    frame = pd.read_csv(path)
    row = frame.loc[frame["Criterion"] == criterion]
    if row.empty:
        raise ValueError(f"No '{criterion}' row in {path}")
    return float(row["Threshold"].iloc[0])


def read_config(path):
    frame = pd.read_csv(path)
    return dict(zip(frame["Parameter"], frame["Value"]))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="folder containing the Kaggle output files")
    parser.add_argument("--train", required=True,
                        help="UNSW_NB15_training-set.csv (175,341 rows)")
    parser.add_argument("--out", default="/kaggle/working/inference_bundle.joblib")
    args = parser.parse_args()

    R = args.results

    print("=" * 72)
    print("REBUILD INFERENCE BUNDLE FROM SAVED OUTPUTS")
    print("=" * 72)
    print(f"CuPy available : {cp is not None}")
    print(f"cuML importable: {HAS_CUML}")

    if not HAS_CUML:
        print(
            "\nSTOP. The SVM pickles wrap cuML objects and will not unpickle "
            "without cuml importable. Turn the GPU accelerator on, or re-run "
            "the training script with the export cell appended instead."
        )
        return 3

    # ---------------------------------------------------------------- load

    print("\nLoading saved models...")
    minmax_scaler = joblib.load(os.path.join(R, "minmax_scaler.joblib"))
    if_model = joblib.load(os.path.join(R, "isolation_forest_model.joblib"))
    svm_only = joblib.load(os.path.join(R, "standalone_svm_model.joblib"))
    svm_hybrid = joblib.load(os.path.join(R, "hybrid_if_svm_model.joblib"))

    print(f"  Isolation Forest: {len(if_model.trees)} trees, "
          f"S={if_model.sample_size}, depth={if_model.max_depth}")

    if_model.use_gpu = cp is not None
    if_model._stage_arrays()

    config = read_config(os.path.join(R, "experiment_configuration.csv"))

    if_threshold = float(config["IF_Best_Threshold"])
    if_contamination = float(config["IF_Best_Contamination"])
    label_switched = "NORMAL (0)" in str(config["IF_Label_Direction"])

    threshold_hybrid = read_threshold(
        os.path.join(R, "hybrid_threshold_criteria.csv"),
        str(config.get("Threshold_Criterion", "MCC")),
    )
    threshold_baseline = read_threshold(
        os.path.join(R, "svm_threshold_criteria.csv"),
        str(config.get("Threshold_Criterion", "MCC")),
    )

    print(f"\n  IF threshold        : {if_threshold:.6f} "
          f"(contamination {if_contamination})")
    print(f"  Label switched      : {label_switched}")
    print(f"  SVM threshold       : {threshold_baseline:.6f}")
    print(f"  Hybrid threshold    : {threshold_hybrid:.6f}")

    # ------------------------------------------------- training partition

    print("\nLoading the training partition (for medians and the extra scales)...")
    train_df = pd.read_csv(args.train, low_memory=False)
    train_df.columns = train_df.columns.astype(str).str.strip().str.lower()

    if len(train_df) == EXPECTED_TEST_SIZE:
        return fail(
            "That looks like the TESTING partition (82,332 rows). This "
            "public Kaggle copy has the filenames reversed -- pass the "
            "175,341-row file."
        )

    if len(train_df) != EXPECTED_TRAIN_SIZE:
        print(f"  WARNING: expected {EXPECTED_TRAIN_SIZE:,} rows, "
              f"got {len(train_df):,}. Medians will not match the training "
              f"run and the bundle will be subtly wrong.")

    X = train_df[SHUKLA_FEATURES].copy()
    for column in SHUKLA_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    train_medians = X.median()
    X_clean = X.fillna(train_medians)
    X_scaled = minmax_scaler.transform(X_clean)

    print(f"  {len(train_df):,} rows, scaled matrix {X_scaled.shape}")

    # ------------------------------------------------ multi-scale forests

    print("\nRebuilding the multi-scale forests (same seeds -> same trees)...")

    forests = {SAMPLE_SIZE: export_forest(if_model)}
    scale_columns = {"if_score": if_model.anomaly_score(X_scaled)}

    for scale in IF_SCALES:
        if scale == SAMPLE_SIZE:
            continue

        forest = CustomIsolationForest(
            n_trees=N_TREES,
            sample_size=scale,
            random_state=RANDOM_STATE + scale,
            use_gpu=cp is not None,
            verbose=False,
        )
        forest.fit(X_scaled)

        forests[scale] = export_forest(forest)
        scale_columns[f"if_score_s{scale}"] = forest.anomaly_score(X_scaled)
        print(f"  S={scale:4d} rebuilt and scored")

    # -------------------------------------------------- enhanced features

    print("\nRebuilding the enhanced feature frame...")

    frame = X_clean.copy()
    for name, values in scale_columns.items():
        frame[name] = values

    matrix = np.column_stack([scale_columns[k] for k in scale_columns])
    frame["if_scale_spread"] = matrix.std(axis=1)
    frame["if_scale_range"] = matrix.max(axis=1) - matrix.min(axis=1)
    frame["if_scale_mean"] = matrix.mean(axis=1)

    enhanced = create_enhanced_features(frame)
    enhanced_medians = enhanced.median()
    enhanced = enhanced.fillna(enhanced_medians).replace(
        [np.inf, -np.inf], 0.0
    )

    feature_order = [str(c) for c in enhanced.columns]

    # ---- the guard: does this match what training actually produced?
    expected = pd.read_csv(
        os.path.join(R, "hybrid_feature_list.csv")
    )["Feature"].astype(str).tolist()

    if feature_order != expected:
        only_here = [c for c in feature_order if c not in expected]
        only_there = [c for c in expected if c not in feature_order]
        return fail(
            "Feature order does not match hybrid_feature_list.csv.\n"
            f"  extra here : {only_here[:10]}\n"
            f"  missing    : {only_there[:10]}\n"
            "create_enhanced_features() has drifted from the training run."
        )

    print(f"  {len(feature_order)} features, order matches training exactly")

    # ------------------------------------------------------------- export

    print("\nExtracting SVM decision functions...")
    baseline_spec = export_svm(svm_only, SHUKLA_FEATURES, "standalone SVM")
    hybrid_spec = export_svm(svm_hybrid, feature_order, "hybrid IF-SVM")

    selected = pd.read_csv(
        os.path.join(R, "hybrid_selected_features.csv")
    )["Selected_Feature"].astype(str).tolist()

    if sorted(hybrid_spec["columns"]) != sorted(selected):
        print(
            "  WARNING: selected columns differ from "
            "hybrid_selected_features.csv. Check before shipping."
        )

    results = pd.read_csv(os.path.join(R, "test_results_all_models.csv"))
    row = lambda name: results.loc[
        results["Model"].str.contains(name, case=False)
    ].iloc[0]

    bundle = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "dataset": "UNSW-NB15",
        "rebuilt_from_outputs": True,

        "shukla_features": list(SHUKLA_FEATURES),
        "train_medians": {k: float(v) for k, v in train_medians.items()},
        "enhanced_medians": {
            k: float(v) for k, v in enhanced_medians.items()
        },
        "minmax_min": np.asarray(minmax_scaler.data_min_, dtype=np.float64),
        "minmax_range": np.asarray(
            minmax_scaler.data_range_, dtype=np.float64
        ),

        "if_scales": list(IF_SCALES),
        "if_sample_size": int(SAMPLE_SIZE),
        "forests": forests,
        "if_threshold": if_threshold,
        "if_contamination": if_contamination,
        "if_label_switched": bool(label_switched),

        "feature_order": feature_order,

        "svm_baseline": baseline_spec,
        "svm_hybrid": hybrid_spec,

        "threshold_baseline": threshold_baseline,
        "threshold_hybrid": threshold_hybrid,

        "meta": {
            "search_scoring": str(config.get("Search_Scoring", "")),
            "threshold_criterion": str(config.get("Threshold_Criterion", "")),
            "baseline_params": {
                "C": str(config.get("SVM_Baseline_C", "")),
                "gamma": str(config.get("SVM_Baseline_Gamma", "")),
                "k": str(config.get("SVM_Baseline_k", "")),
            },
            "hybrid_params": {
                "C": str(config.get("Hybrid_C", "")),
                "gamma": str(config.get("Hybrid_Gamma", "")),
                "k": str(config.get("Hybrid_k", "")),
            },
            "test_accuracy_if": float(row("Isolation Forest")["Accuracy"]),
            "test_accuracy_svm": float(row("Standalone SVM")["Accuracy"]),
            "test_accuracy_hybrid": float(row("Hybrid")["Accuracy"]),
            "test_mcc_if": float(row("Isolation Forest")["MCC"]),
            "test_mcc_svm": float(row("Standalone SVM")["MCC"]),
            "test_mcc_hybrid": float(row("Hybrid")["MCC"]),
        },
    }

    joblib.dump(bundle, args.out, compress=3)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)

    print("\n" + "=" * 72)
    print(f"Bundle written: {args.out}  ({size_mb:.2f} MB)")
    print("=" * 72)
    print(
        "\nVERIFY BEFORE SHIPPING\n"
        "Run predict.py on the UNSW-NB15 TESTING partition. The three\n"
        "accuracies it reports must match test_results_all_models.csv. If\n"
        "they do not, something in the rebuild drifted -- do not use the\n"
        "bundle until they agree."
    )
    return 0


def fail(message):
    print("\nFAILED: " + message)
    return 2


if __name__ == "__main__":
    sys.exit(main())
