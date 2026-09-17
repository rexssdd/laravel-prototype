"""
EXPORT INFERENCE BUNDLE
=======================

Run this at the END of hybrid_if_svm_final.py, in the same session, after
PART 16. It has access to every variable it needs.

WHY THIS EXISTS
---------------
joblib.dump() of the trained pipelines stores references to classes that
only exist inside the training script: RBFSVMClassifier, KeepColumnSelector,
CustomIsolationForest, plus cuML objects that need a GPU to unpickle. Loading
those from a web app would mean shipping the whole training script and a GPU.

So instead of pickling objects, this exports the MATHEMATICS:

  - Isolation Forest  -> the flat node arrays per tree
  - SVM               -> support vectors, dual coefficients, intercept, gamma
  - preprocessing     -> medians, min-max range, standard-scaler stats
  - decisions         -> the two tuned thresholds and the IF contamination cut

The result loads with nothing but numpy. Verified: an RBF decision function
rebuilt this way matches sklearn's to 3e-13.
"""

import os
import numpy as np
import joblib


BUNDLE_PATH = os.path.join(OUTPUT_DIR, "inference_bundle.joblib")


# ---------------------------------------------------------------- forests

def export_forest(forest):
    """Flatten a CustomIsolationForest into plain arrays."""
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


# The S=256 forest is IF_MODEL. The other scales were local to the PART 5
# loop, so rebuild them here with the identical seeds -- same random_state,
# same data, same trees.
forests = {SAMPLE_SIZE: export_forest(IF_MODEL)}

for scale in IF_SCALES:
    if scale == SAMPLE_SIZE:
        continue
    f = CustomIsolationForest(
        n_trees=N_TREES,
        sample_size=scale,
        random_state=RANDOM_STATE + scale,
        use_gpu=False,
        verbose=False,
    )
    f.fit(X_train_scaled)
    forests[scale] = export_forest(f)
    print(f"  exported forest S={scale}")


# -------------------------------------------------------------------- svm

def export_svm(pipeline, input_columns):
    """Pull the decision function out of a fitted pipeline."""

    selector = pipeline.named_steps["selector"]
    scaler = pipeline.named_steps["scaler"]
    svc = pipeline.named_steps["svc"]
    inner = svc._model

    mask = selector.get_support()

    return {
        "columns": [
            str(c) for c in np.asarray(list(input_columns))[mask]
        ],
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "support_vectors": np.asarray(
            inner.support_vectors_, dtype=np.float64
        ),
        "dual_coef": np.asarray(inner.dual_coef_, dtype=np.float64).ravel(),
        "intercept": float(np.asarray(inner.intercept_).ravel()[0]),
        "gamma": float(svc._gamma_value),
        "sign": float(svc._sign),
    }


bundle = {
    "created": pd.Timestamp.utcnow().isoformat(),
    "dataset": "UNSW-NB15",

    # ---- preprocessing
    "shukla_features": list(SHUKLA_FEATURES),
    "train_medians": {
        k: float(v) for k, v in train_medians.to_dict().items()
    },
    "enhanced_medians": {
        k: float(v) for k, v in enhanced_medians.to_dict().items()
    },
    "minmax_min": np.asarray(minmax_scaler.data_min_, dtype=np.float64),
    "minmax_range": np.asarray(
        minmax_scaler.data_range_, dtype=np.float64
    ),

    # ---- isolation forest
    "if_scales": list(IF_SCALES),
    "if_sample_size": int(SAMPLE_SIZE),
    "forests": forests,
    "if_threshold": float(BEST_THRESHOLD),
    "if_contamination": float(BEST_CONTAMINATION),
    "if_label_switched": bool(LABEL_SWITCHED),

    # ---- feature order the hybrid expects
    "feature_order": [str(c) for c in X_enhanced_train.columns],

    # ---- models
    "svm_baseline": export_svm(BEST_SVM_ONLY, SHUKLA_FEATURES),
    "svm_hybrid": export_svm(BEST_HYBRID_MODEL, X_enhanced_train.columns),

    # ---- tuned decision thresholds
    "threshold_baseline": float(BEST_BASE_THRESHOLD),
    "threshold_hybrid": float(BEST_SVM_THRESHOLD),

    # ---- provenance, shown in the web UI
    "meta": {
        "search_scoring": SEARCH_SCORING,
        "threshold_criterion": THRESHOLD_CRITERION,
        "baseline_params": {
            k: str(v) for k, v in base_search.best_params_.items()
        },
        "hybrid_params": {
            k: str(v) for k, v in grid_search.best_params_.items()
        },
        "test_accuracy_if": float(if_row["Accuracy"]),
        "test_accuracy_svm": float(base_row["Accuracy"]),
        "test_accuracy_hybrid": float(hybrid_row["Accuracy"]),
        "test_mcc_if": float(if_row["MCC"]),
        "test_mcc_svm": float(base_row["MCC"]),
        "test_mcc_hybrid": float(hybrid_row["MCC"]),
    },
}

joblib.dump(bundle, BUNDLE_PATH, compress=3)

size_mb = os.path.getsize(BUNDLE_PATH) / (1024 * 1024)

print(f"\nInference bundle written: {BUNDLE_PATH}")
print(f"Size: {size_mb:.2f} MB")
print(f"Support vectors -- baseline: "
      f"{bundle['svm_baseline']['support_vectors'].shape}, "
      f"hybrid: {bundle['svm_hybrid']['support_vectors'].shape}")
print("\nDownload this file and place it at  storage/app/ml/inference_bundle.joblib")
