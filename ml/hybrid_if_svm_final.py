# ============================================================================
# A HYBRID ISOLATION FOREST-SVM APPROACH FOR IMPROVED ANOMALY DETECTION
# AND CLASSIFICATION OF NETWORK INTRUSION INCIDENTS
#
# Lusica, Care, Cabanilla - University of Mindanao
# Adviser: Charisse P. Barbosa
#
# FINAL EDITION - GPU accelerated, MCC-selected, multi-scale fusion
#
# Dataset : UNSW-NB15 (Moustafa & Slay, 2015)
# Baseline: Shukla et al. (2023) UInDeSI4.0 - standalone Isolation Forest
#
# ----------------------------------------------------------------------------
# THREE MODELS, ONE HELD-OUT TEST PARTITION
#
#   1. Standalone Isolation Forest  (Shukla replication - unsupervised)
#   2. Standalone SVM               (9 Shukla features - supervised)
#   3. Hybrid IF-SVM                (9 features + IF scores + engineered)
#
# Model 2 is what makes the claim falsifiable. Without it, "the hybrid beats
# the 63% baseline" only proves that supervision beats no supervision.
#
# ----------------------------------------------------------------------------
# WHY SELECTION USES MCC AND NOT F1
#
# F1 has no term for true negatives. On this test partition, predicting
# "attack" for every single record scores F1 = 0.7102. The validation split
# is 68% attack, so F1 there rewards recall almost unconditionally -- which
# is how an earlier run ended at recall 0.997, specificity 0.586, and 15,325
# false positives, while the hybrid's ROC-AUC advantage of +0.086 over the
# standalone SVM never reached the reported accuracy at all.
#
# MCC uses all four cells of the confusion matrix and scores 0.0 for the
# predict-everything model. It is used for BOTH the hyperparameter search
# and the decision threshold, for BOTH supervised models.
#
# ----------------------------------------------------------------------------
# FAIRNESS CONTROLS - DO NOT BREAK THESE
#
# The standalone SVM and the hybrid share: the same rows, the same grid, the
# same CV strategy, the same SMOTE settings, the same scorer, the same
# threshold criterion. The ONLY difference is the feature matrix. That is
# what licenses the claim that any gap comes from feature-level fusion.
#
# PART 12c prints the ablation. If the hybrid does not beat the standalone
# SVM, that is a finding to report, not a bug to fix.
#
# ----------------------------------------------------------------------------
# LEAKAGE CONTROL
#   medians, MinMaxScaler, StandardScaler  -> fitted on TRAIN only
#   SelectKBest, SMOTE                     -> inside the CV pipeline
#   IF contamination threshold             -> selected on TRAIN accuracy
#   SVM decision threshold                 -> selected on VALIDATION
#   test labels                            -> PART 11 onward only
#                                             (PART 12d excepted and labelled)
# ============================================================================


# ============================================================================
# PART 0 - IMPORTS, GPU DETECTION, CONFIGURATION
# ============================================================================

import os
import math
import time
import shutil
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- GPU probe

print("=" * 80)
print("GPU DETECTION")
print("=" * 80)

HAS_CUPY = False
HAS_CUML = False
cp = None
cuSVC = None

try:
    import cupy as _cp

    _cp.zeros(1).sum()          # force context creation; fails without a GPU
    cp = _cp
    HAS_CUPY = True
    print("CuPy      : available")
except Exception as exc:
    print(f"CuPy      : not available ({type(exc).__name__})")

try:
    from cuml.svm import SVC as _cuSVC

    cuSVC = _cuSVC
    HAS_CUML = True
    print("cuML SVC  : available")
except Exception as exc:
    print(f"cuML SVC  : not available ({type(exc).__name__})")

USE_GPU_IF = HAS_CUPY
USE_GPU_SVM = HAS_CUML

if not (HAS_CUPY or HAS_CUML):
    print(
        "\nRunning fully on CPU. In Kaggle, set Settings -> Accelerator to a "
        "GPU (T4 x2 or P100). If cuML is still missing after switching:\n"
        "    !pip install -q cuml-cu12 --extra-index-url=https://pypi.nvidia.com\n"
        "The script is correct either way; CPU is just slower."
    )

print(f"\nIsolation Forest scoring on GPU : {USE_GPU_IF}")
print(f"SVM training on GPU             : {USE_GPU_SVM}")


# ---------------------------------------------------------------- reproducible
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ------------------------------------------------- Shukla baseline parameters
EXPECTED_TRAIN_SIZE = 175341          # paper Section 4.1
EXPECTED_TEST_SIZE = 82332

N_TREES = 100                         # paper Table 3 / Table 4: T = 100
SAMPLE_SIZE = 256                     # paper Table 3 / Table 4: S = 256

CONTAMINATIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

PAPER_BEST_CONTAMINATION = 0.25       # Table 3
PAPER_TRAIN_ACCURACY = 0.6897         # Table 4
PAPER_TEST_ACCURACY = 0.6261          # Table 4


# --------------------------------------------- nine RF-selected features (T5)
SHUKLA_FEATURES = [
    "sttl",
    "ct_state_ttl",
    "dload",
    "rate",
    "dmean",
    "dttl",
    "ackdat",
    "sload",
    "sbytes",
]


# ------------------------------------------------------------------ filenames
TRAIN_FILENAME_TARGET = "unsw_nb15_training-set.csv"
TEST_FILENAME_TARGET = "unsw_nb15_testing-set.csv"


# ------------------------------------------------- multi-scale IF fusion
#
# Isolation Forest is sensitive to sub-sample size S. Small S isolates
# coarse global outliers; large S resolves local structure. Running the
# forest at several scales gives the SVM a score VECTOR instead of a
# scalar, plus the disagreement between scales -- a record that looks
# anomalous at one resolution but not another is sitting on a boundary,
# and no single score can express that.
#
# S=256 remains the Shukla baseline and is reused, not retrained, so the
# replication numbers do not move. The extra scales feed only the hybrid.
#
IF_SCALES = [64, 128, 256, 512]


# ------------------------------------------------------- SVM / hybrid config
#
# ONE pool and ONE grid, used by BOTH supervised models. Giving the hybrid
# more rows or a wider grid than the baseline would make the comparison
# meaningless.
#
SVM_POOL_SIZE = 60000 if USE_GPU_SVM else 30000
VALIDATION_SIZE = 0.20

if USE_GPU_SVM:
    SELECT_K_VALUES = [15, 20, 25, 30]
    SVM_C_VALUES = [1, 10, 100, 1000]
    SVM_GAMMA_VALUES = ["scale", 0.01, 0.1]
else:
    SELECT_K_VALUES = [10, 15, 20, 25]
    SVM_C_VALUES = [1, 10, 100]
    SVM_GAMMA_VALUES = ["scale", 0.01]

CV_FOLDS = 3
SMOTE_K_NEIGHBORS = 5

# Matthews correlation coefficient, for the reasons in the header.
SEARCH_SCORING = "matthews_corrcoef"
THRESHOLD_CRITERION = "MCC"   # MCC | Balanced_Accuracy | Youden_J | F1

# ---------------------------------------------------------------------------
# FORCE_KEEP_IF_SCORE
#
# SelectKBest ranks by mutual information with the label and knows nothing
# about the research question, so it can discard every IF-derived feature
# and silently turn the "hybrid" into a plain SVM. With this on, the S=256
# anomaly score is always retained; SelectKBest fills the remaining k-1
# slots. This is a declared design decision -- Section 2.5.2 specifies the
# IF score AS an input feature, so this tests the model the manuscript
# describes. It costs one of the k slots, so the hybrid never gets more
# features than k. PART 9 reports where selection would have ranked it on
# its own; put that number in the paper.
# ---------------------------------------------------------------------------
FORCE_KEEP_IF_SCORE = True

OUTPUT_DIR = "/kaggle/working/hybrid_if_svm_experiment"
CACHE_DIR = "/kaggle/working/_pipeline_cache"


print("\n" + "=" * 80)
print("EXPERIMENT CONFIGURATION")
print("=" * 80)
print(f"Random state          : {RANDOM_STATE}")
print(f"IF trees / sample     : {N_TREES} / {SAMPLE_SIZE}")
print(f"IF fusion scales      : {IF_SCALES}")
print(f"SVM pool size         : {SVM_POOL_SIZE:,}  (shared by both models)")
print(f"Validation fraction   : {VALIDATION_SIZE}")
print(f"SelectKBest k values  : {SELECT_K_VALUES}")
print(f"SVM C values          : {SVM_C_VALUES}")
print(f"SVM gamma values      : {SVM_GAMMA_VALUES}")
print(f"CV folds              : {CV_FOLDS}")
print(f"Search scoring        : {SEARCH_SCORING}")
print(f"Threshold criterion   : {THRESHOLD_CRITERION}")
print(f"Force-keep IF score   : {FORCE_KEEP_IF_SCORE}")

n_candidates = (
    len(SELECT_K_VALUES) * len(SVM_C_VALUES) * len(SVM_GAMMA_VALUES)
)
print(f"Grid candidates       : {n_candidates}")
print(f"Fits per model        : {n_candidates * CV_FOLDS} (+1 refit)")


# ============================================================================
# PART 1 - LOCATE AND LOAD THE DATASET
# ============================================================================

print("\n" + "=" * 80)
print("PART 1 - LOCATING UNSW-NB15 PARTITIONED FILES")
print("=" * 80)

INPUT_ROOT = "/kaggle/input"

if not os.path.exists(INPUT_ROOT):
    raise FileNotFoundError(
        f"\n{INPUT_ROOT} does not exist. This script is intended to run "
        "inside a Kaggle notebook with an attached UNSW-NB15 dataset."
    )

train_path = None
test_path = None
all_found_files = []

for dirpath, _, filenames in os.walk(INPUT_ROOT):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)
        all_found_files.append(full_path)

        lower_name = filename.lower()
        if lower_name == TRAIN_FILENAME_TARGET.lower() and train_path is None:
            train_path = full_path
        if lower_name == TEST_FILENAME_TARGET.lower() and test_path is None:
            test_path = full_path

print(f"\nFiles discovered: {len(all_found_files)}")
for path in all_found_files:
    print(" ", path)

if train_path is None or test_path is None:
    raise FileNotFoundError(
        "\n\nCould not locate the partitioned UNSW-NB15 files.\n"
        f"Required training file: {TRAIN_FILENAME_TARGET}\n"
        f"Required testing file : {TEST_FILENAME_TARGET}\n"
    )

print(f"\nTraining file: {train_path}")
print(f"Testing file : {test_path}")

df_train_file = pd.read_csv(train_path)
df_test_file = pd.read_csv(test_path)

df_train_file.columns = (
    df_train_file.columns.astype(str).str.strip().str.lower()
)
df_test_file.columns = (
    df_test_file.columns.astype(str).str.strip().str.lower()
)

print(
    f"\nTraining file contents: {df_train_file.shape[0]:,} rows "
    f"x {df_train_file.shape[1]} columns"
)
print(
    f"Testing file contents : {df_test_file.shape[0]:,} rows "
    f"x {df_test_file.shape[1]} columns"
)


# ------------------------------------------------- verify partition assignment

print("\n" + "-" * 80)
print("VERIFYING PARTITION ASSIGNMENT")
print("-" * 80)

if (
    len(df_train_file) == EXPECTED_TRAIN_SIZE
    and len(df_test_file) == EXPECTED_TEST_SIZE
):
    train_df = df_train_file
    test_df = df_test_file
    print("Partition sizes match the values reported in the paper.")

elif (
    len(df_test_file) == EXPECTED_TRAIN_SIZE
    and len(df_train_file) == EXPECTED_TEST_SIZE
):
    print("Filenames appear reversed. Reassigning by row count.")
    train_df = df_test_file
    test_df = df_train_file

else:
    print(
        "WARNING: partition sizes do not match the paper's reported totals. "
        "Public UNSW-NB15 releases vary slightly. Proceeding with the files "
        "as given -- verify this is the official partitioned split and not a "
        "re-shuffled or subsampled copy."
    )
    train_df = df_train_file
    test_df = df_test_file

print(f"\nTraining records: {len(train_df):,}")
print(f"Testing records : {len(test_df):,}")


# ----------------------------------------------------- verify columns present

missing_train = [f for f in SHUKLA_FEATURES if f not in train_df.columns]
missing_test = [f for f in SHUKLA_FEATURES if f not in test_df.columns]

if missing_train:
    raise ValueError(
        "Missing features in training data: " + ", ".join(missing_train)
    )
if missing_test:
    raise ValueError(
        "Missing features in testing data: " + ", ".join(missing_test)
    )

for frame_name, frame in (("training", train_df), ("testing", test_df)):
    if "label" not in frame.columns:
        raise ValueError(f"'label' column not found in {frame_name} data.")

print("\nAll nine Shukla features and the label column are present.")

print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())
print("\nTesting label distribution:")
print(test_df["label"].value_counts().sort_index())

HAS_ATTACK_CAT = "attack_cat" in test_df.columns
print(f"\nattack_cat available for per-category analysis: {HAS_ATTACK_CAT}")


# ============================================================================
# PART 2 - CLEANING AND SCALING
# ============================================================================

print("\n" + "=" * 80)
print("PART 2 - CLEANING AND SCALING")
print("=" * 80)

y_train = (
    pd.to_numeric(train_df["label"], errors="raise").astype(int).to_numpy()
)
y_test = (
    pd.to_numeric(test_df["label"], errors="raise").astype(int).to_numpy()
)

X_train_raw = train_df[SHUKLA_FEATURES].copy()
X_test_raw = test_df[SHUKLA_FEATURES].copy()

train_attack_fraction = float(y_train.mean())
test_attack_fraction = float(y_test.mean())

print(f"\nTraining attack fraction: {train_attack_fraction:.4f}")
print(f"Testing attack fraction : {test_attack_fraction:.4f}")
print("(paper Table 7 reports attack as the majority class, ~0.678)")

prior_shift = abs(train_attack_fraction - test_attack_fraction)

if prior_shift > 0.05:
    print(
        f"\nPRIOR SHIFT of {prior_shift:.4f} between the training and testing "
        "partitions. This is inherent to UNSW-NB15's official split, not a "
        "bug, but it means any threshold tuned on validation transfers "
        "imperfectly to test. It is the reason selection uses MCC rather "
        "than F1, and it belongs in the limitations section."
    )

LABEL_SWITCHED = train_attack_fraction >= 0.5

if not LABEL_SWITCHED:
    print(
        "\nNOTE: attack is NOT the majority class in this split. The Table-7 "
        "label-switching direction assumes it is. The direction is flipped "
        "automatically, and both directions are reported in PART 12."
    )

for feature in SHUKLA_FEATURES:
    X_train_raw[feature] = pd.to_numeric(X_train_raw[feature], errors="coerce")
    X_test_raw[feature] = pd.to_numeric(X_test_raw[feature], errors="coerce")

X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

train_medians = X_train_raw.median()

if train_medians.isna().any():
    problematic = train_medians[train_medians.isna()].index.tolist()
    raise ValueError(
        "Unable to compute training median for: " + ", ".join(problematic)
    )

X_train_clean = X_train_raw.fillna(train_medians)
X_test_clean = X_test_raw.fillna(train_medians)

print("\nMissing values replaced using training-set medians.")

minmax_scaler = MinMaxScaler()
X_train_scaled = minmax_scaler.fit_transform(X_train_clean)
X_test_scaled = minmax_scaler.transform(X_test_clean)

print(f"Training scaled matrix: {X_train_scaled.shape}")
print(f"Testing scaled matrix : {X_test_scaled.shape}")


# ============================================================================
# PART 3 - ISOLATION FOREST (Algorithm 1) - VECTORIZED, GPU-CAPABLE
# ============================================================================
#
# Trees are built exactly as in Algorithm 1: random feature, random split in
# [min, max], depth cap ceil(log2(S)), sub-sample of S without replacement.
#
# The difference is representation. Each tree is flattened into arrays
#   feature[], threshold[], left[], right[], c_adjust[]
# and every record descends the tree simultaneously, one depth level per
# vectorized step. On CuPy that runs on the GPU. Scores are identical to a
# recursive per-row implementation; only the arithmetic order changes.
#
# c(n) is computed once per leaf at BUILD time. Recomputing a 256-term
# harmonic sum on every scoring call is what makes the naive version
# effectively never finish on 257,673 records x 100 trees.
# ============================================================================


@lru_cache(maxsize=None)
def harmonic_number(n):
    """H(n) = sum_{i=1..n} 1/i"""
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))


@lru_cache(maxsize=None)
def c_factor(n):
    """c(n) = 2H(n-1) - 2(n-1)/n  -- expected path length normalizer."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * harmonic_number(n - 1) - (2.0 * (n - 1) / n)


class FlatIsolationTree:
    """Algorithm 1 steps 3-4, stored as flat arrays instead of objects."""

    def __init__(self, max_depth, random_state):
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)

        self.feature = []       # -1 marks a leaf
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
        root = self._new_node()
        self._build(root, X, depth=0)

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

        min_value = float(np.min(values))
        max_value = float(np.max(values))

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
    """Algorithm 1 steps 1-6a. Score(x) = 2^(-E[h(x)] / c(n))."""

    def __init__(
        self,
        n_trees=100,
        sample_size=256,
        random_state=42,
        use_gpu=False,
        verbose=True,
    ):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.verbose = verbose

        self.trees = []
        self.rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------------- fit

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        self.max_depth = math.ceil(math.log2(self.sample_size))

        if self.verbose:
            print(f"\nMaximum tree depth: {self.max_depth}")

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

            if self.verbose and ((i + 1) % 20 == 0 or i == 0):
                print(f"Trees built: {i + 1}/{self.n_trees}")

        self._stage_arrays()
        return self

    def _stage_arrays(self):
        """Move the flattened trees onto the GPU once, if available."""
        xp = cp if (self.use_gpu and cp is not None) else np

        self._xp = xp
        self._staged = [
            (
                xp.asarray(t.feature),
                xp.asarray(t.threshold),
                xp.asarray(t.left),
                xp.asarray(t.right),
                xp.asarray(t.c_adjust),
            )
            for t in self.trees
        ]

    # -------------------------------------------------------------- scoring

    def anomaly_score(self, X, batch_size=200000):
        xp = getattr(self, "_xp", np)

        c_n = c_factor(self.sample_size)
        if c_n == 0:
            raise ValueError("c(n) cannot be zero.")

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

            mean_depth = total_depth / float(self.n_trees)
            scores = xp.power(2.0, -(mean_depth / c_n))

            out[start:stop] = (
                cp.asnumpy(scores) if xp is not np else np.asarray(scores)
            )

        return out

    # keep the object picklable without GPU handles
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
# PART 4 - BASELINE 1: STANDALONE ISOLATION FOREST (Shukla replication)
# ============================================================================

print("\n" + "=" * 80)
print("PART 4 - BASELINE 1: STANDALONE ISOLATION FOREST")
print("=" * 80)
print(f"Scoring backend: {'CuPy (GPU)' if USE_GPU_IF else 'NumPy (CPU)'}")

IF_MODEL = CustomIsolationForest(
    n_trees=N_TREES,
    sample_size=SAMPLE_SIZE,
    random_state=RANDOM_STATE,
    use_gpu=USE_GPU_IF,
)

t0 = time.time()
IF_MODEL.fit(X_train_scaled)
if_training_time = time.time() - t0
print(f"\nIsolation Forest training time: {if_training_time:.2f} s")

print("\nScoring training partition...")
t0 = time.time()
train_scores = IF_MODEL.anomaly_score(X_train_scaled)
if_train_score_time = time.time() - t0
print(f"Training scoring time: {if_train_score_time:.2f} s")

print("Scoring testing partition...")
t0 = time.time()
test_scores = IF_MODEL.anomaly_score(X_test_scaled)
if_test_score_time = time.time() - t0
print(f"Testing scoring time: {if_test_score_time:.2f} s")

print("\nTraining score summary:")
print(pd.Series(train_scores).describe())
print("\nTesting score summary:")
print(pd.Series(test_scores).describe())

assert len(train_scores) == len(train_df)
assert len(test_scores) == len(test_df)


# ---------------------------------------------------------------------------
# LABEL DIRECTION (paper Table 7 / Section 5, discussion point 1)
#
# Attack is the MAJORITY class here. Isolation Forest isolates the minority,
# so isolated points correspond to NORMAL traffic. The paper label-switches:
#
#   score >  threshold -> isolated     -> predicted NORMAL (0)
#   score <= threshold -> not isolated -> predicted ATTACK (1)
# ---------------------------------------------------------------------------

def if_predict(scores, threshold, switched=True):
    if switched:
        return np.where(scores > threshold, 0, 1)
    return np.where(scores > threshold, 1, 0)


print("\n" + "-" * 80)
print("CONTAMINATION EXPERIMENT (selection uses TRAIN accuracy only)")
print("-" * 80)

contamination_results = []

for contamination in CONTAMINATIONS:

    threshold = float(
        np.percentile(train_scores, 100.0 * (1.0 - contamination))
    )
    y_train_pred = if_predict(train_scores, threshold, LABEL_SWITCHED)

    contamination_results.append(
        {
            "Contamination": contamination,
            "Threshold": threshold,
            "Train_Accuracy": accuracy_score(y_train, y_train_pred),
            "Train_Precision": precision_score(
                y_train, y_train_pred, zero_division=0
            ),
            "Train_Recall": recall_score(
                y_train, y_train_pred, zero_division=0
            ),
            "Train_F1": f1_score(y_train, y_train_pred, zero_division=0),
            "Train_MCC": matthews_corrcoef(y_train, y_train_pred),
        }
    )

contamination_df = pd.DataFrame(contamination_results)
print(
    contamination_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
)

best_row = contamination_df.loc[contamination_df["Train_Accuracy"].idxmax()]

BEST_CONTAMINATION = float(best_row["Contamination"])
BEST_THRESHOLD = float(best_row["Threshold"])
BEST_TRAIN_ACCURACY = float(best_row["Train_Accuracy"])

print("\nSELECTED BASELINE CONFIGURATION")
print(
    f"Contamination  : {BEST_CONTAMINATION:.2f} "
    f"(paper Table 3 reports {PAPER_BEST_CONTAMINATION:.2f})"
)
print(f"Threshold      : {BEST_THRESHOLD:.6f}")
print(
    f"Train accuracy : {BEST_TRAIN_ACCURACY:.4f} "
    f"(paper Table 4 reports {PAPER_TRAIN_ACCURACY:.4f})"
)
print(
    "\n  Selection stays on TRAIN ACCURACY here, not MCC, because that is "
    "what Shukla et al. used. Changing it would stop this being a "
    "replication. MCC is reported alongside for context."
)

if abs(BEST_CONTAMINATION - PAPER_BEST_CONTAMINATION) > 1e-9:
    print(
        "\nNOTE: selected contamination differs from the paper's reported "
        "value. Investigate rather than override -- dataset version, RNG "
        "variance and Isolation Forest randomness all contribute. The paper "
        "itself averages over 100 iterations with non-zero SD."
    )

y_test_pred_if = if_predict(test_scores, BEST_THRESHOLD, LABEL_SWITCHED)
if_ranking_score = -test_scores if LABEL_SWITCHED else test_scores


# ============================================================================
# PART 5 - MULTI-SCALE IF SCORES AND HYBRID FEATURE CONSTRUCTION
# ============================================================================

print("\n" + "=" * 80)
print("PART 5 - HYBRID FEATURE CONSTRUCTION")
print("=" * 80)

EPS = 1e-9

X_hybrid_train = X_train_clean.copy()
X_hybrid_test = X_test_clean.copy()

# The S=256 score is the Shukla baseline score, reused as-is.
X_hybrid_train["if_score"] = train_scores
X_hybrid_test["if_score"] = test_scores


# ------------------------------------------------- multi-scale IF fusion

print("\nBuilding multi-scale Isolation Forest scores...")

scale_columns_train = {"if_score": train_scores}
scale_columns_test = {"if_score": test_scores}

multi_scale_time = 0.0

for scale in IF_SCALES:

    if scale == SAMPLE_SIZE:
        continue      # already have it as if_score; do not retrain

    t0 = time.time()

    forest = CustomIsolationForest(
        n_trees=N_TREES,
        sample_size=scale,
        random_state=RANDOM_STATE + scale,
        use_gpu=USE_GPU_IF,
        verbose=False,
    )
    forest.fit(X_train_scaled)

    column = f"if_score_s{scale}"
    scale_columns_train[column] = forest.anomaly_score(X_train_scaled)
    scale_columns_test[column] = forest.anomaly_score(X_test_scaled)

    X_hybrid_train[column] = scale_columns_train[column]
    X_hybrid_test[column] = scale_columns_test[column]

    multi_scale_time += time.time() - t0
    print(f"  S={scale:4d} scored  ({time.time() - t0:.2f} s)")

# Disagreement across scales: a record that looks anomalous at one
# resolution but not another is sitting on a decision boundary.
scale_matrix_train = np.column_stack(list(scale_columns_train.values()))
scale_matrix_test = np.column_stack(list(scale_columns_test.values()))

X_hybrid_train["if_scale_spread"] = scale_matrix_train.std(axis=1)
X_hybrid_test["if_scale_spread"] = scale_matrix_test.std(axis=1)

X_hybrid_train["if_scale_range"] = (
    scale_matrix_train.max(axis=1) - scale_matrix_train.min(axis=1)
)
X_hybrid_test["if_scale_range"] = (
    scale_matrix_test.max(axis=1) - scale_matrix_test.min(axis=1)
)

X_hybrid_train["if_scale_mean"] = scale_matrix_train.mean(axis=1)
X_hybrid_test["if_scale_mean"] = scale_matrix_test.mean(axis=1)

print(
    f"Multi-scale build time: {multi_scale_time:.2f} s "
    f"({len(IF_SCALES)} scales, {len(IF_SCALES) - 1} newly trained)"
)


# ------------------------------------------------------ feature engineering

def create_enhanced_features(df):
    """Deterministic, label-free feature engineering."""

    X = df.copy()

    # A. log transforms for heavily skewed volume features
    for feature in ["rate", "sload", "dload", "sbytes", "dmean", "ackdat"]:
        if feature in X.columns:
            X[f"log_{feature}"] = np.log1p(np.abs(X[feature]))

    # B. ratios
    if {"sload", "dload"} <= set(X.columns):
        X["load_ratio"] = X["sload"] / (np.abs(X["dload"]) + EPS)
    if {"sbytes", "rate"} <= set(X.columns):
        X["byte_rate_ratio"] = X["sbytes"] / (np.abs(X["rate"]) + EPS)
    if {"sttl", "dttl"} <= set(X.columns):
        X["ttl_ratio"] = X["sttl"] / (np.abs(X["dttl"]) + EPS)
    if {"ackdat", "rate"} <= set(X.columns):
        X["ack_rate_ratio"] = X["ackdat"] / (np.abs(X["rate"]) + EPS)

    # C. differences
    if {"sttl", "dttl"} <= set(X.columns):
        X["ttl_difference"] = X["sttl"] - X["dttl"]
    if {"sload", "dload"} <= set(X.columns):
        X["load_difference"] = X["sload"] - X["dload"]

    # D. IF-score interactions -- the part a voting ensemble cannot express.
    #    These let the SVM learn that the meaning of a network feature
    #    CHANGES with the anomaly level assigned by Isolation Forest.
    for feature in [
        "rate", "sload", "dload", "sbytes", "dmean", "sttl", "dttl", "ackdat"
    ]:
        if feature in X.columns:
            X[f"if_x_{feature}"] = X["if_score"] * X[feature]

    # E. non-linear IF-score terms
    X["if_score_squared"] = X["if_score"] ** 2
    X["if_score_sqrt"] = np.sqrt(np.abs(X["if_score"]))

    return X.replace([np.inf, -np.inf], np.nan)


X_enhanced_train = create_enhanced_features(X_hybrid_train)
X_enhanced_test = create_enhanced_features(X_hybrid_test)

enhanced_medians = X_enhanced_train.median()

X_enhanced_train = X_enhanced_train.fillna(enhanced_medians).replace(
    [np.inf, -np.inf], 0.0
)
X_enhanced_test = X_enhanced_test.fillna(enhanced_medians).replace(
    [np.inf, -np.inf], 0.0
)

X_enhanced_test = X_enhanced_test[X_enhanced_train.columns]

N_BASE = len(SHUKLA_FEATURES)
N_TOTAL = X_enhanced_train.shape[1]
N_ENGINEERED = N_TOTAL - N_BASE - 1

IF_SCORE_POSITION = int(list(X_enhanced_train.columns).index("if_score"))

print(f"\nBase features       : {N_BASE}")
print(f"IF score (S=256)    : 1 (column index {IF_SCORE_POSITION})")
print(f"Engineered features : {N_ENGINEERED}")
print(f"Total hybrid inputs : {N_TOTAL}")
print(f"\nTrain NaN remaining: {int(X_enhanced_train.isna().sum().sum())}")
print(f"Test  NaN remaining: {int(X_enhanced_test.isna().sum().sum())}")

# Standalone SVM baseline input: the nine features ONLY.
X_svmonly_train = X_enhanced_train[SHUKLA_FEATURES].copy()
X_svmonly_test = X_enhanced_test[SHUKLA_FEATURES].copy()


# ============================================================================
# PART 6 - STRATIFIED POOL AND TRAIN / VALIDATION SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("PART 6 - STRATIFIED POOL AND SPLIT")
print("=" * 80)

all_idx = np.arange(len(X_enhanced_train))

if len(all_idx) > SVM_POOL_SIZE:
    pool_idx, _ = train_test_split(
        all_idx,
        train_size=SVM_POOL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
else:
    pool_idx = all_idx

train_idx, val_idx = train_test_split(
    pool_idx,
    test_size=VALIDATION_SIZE,
    stratify=y_train[pool_idx],
    random_state=RANDOM_STATE,
)

X_hyb_train = X_enhanced_train.iloc[train_idx]
X_hyb_val = X_enhanced_train.iloc[val_idx]

X_base_train = X_svmonly_train.iloc[train_idx]
X_base_val = X_svmonly_train.iloc[val_idx]

y_fit = y_train[train_idx]
y_val = y_train[val_idx]

print(f"Full training partition : {len(X_enhanced_train):,}")
print(f"SVM pool                : {len(pool_idx):,}")
print(f"  fitting rows          : {len(train_idx):,}")
print(f"  validation rows       : {len(val_idx):,}")
print(f"Held-out test partition : {len(X_enhanced_test):,}")

print("\nFitting-set class distribution:")
print(pd.Series(y_fit).value_counts().sort_index())


# ============================================================================
# PART 7 - GPU SVM WRAPPER AND PIPELINE
# ============================================================================
#
# cuML's SVC is not a drop-in sklearn estimator: it wants float32, it does
# not resolve gamma="scale" the same way, and its decision_function sign
# convention is not guaranteed. If that sign flipped, every threshold in
# PART 10 would be tuned against an inverted score and the results would be
# quietly wrong. The wrapper probes it after each fit and normalizes.
# ============================================================================

print("\n" + "=" * 80)
print("PART 7 - PIPELINE DEFINITION")
print("=" * 80)


class RBFSVMClassifier(BaseEstimator, ClassifierMixin):
    """RBF C-SVC on GPU (cuML) with an sklearn fallback."""

    def __init__(
        self,
        C=1.0,
        gamma="scale",
        tol=1e-3,
        cache_size=2000,
        random_state=RANDOM_STATE,
        use_gpu=False,
    ):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.cache_size = cache_size
        self.random_state = random_state
        self.use_gpu = use_gpu

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            var = float(np.asarray(X).var())
            return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y = np.asarray(y).astype(np.int32).ravel()

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._gamma_value = self._resolve_gamma(X)

        if self.use_gpu and cuSVC is not None:
            self._model = cuSVC(
                C=float(self.C),
                kernel="rbf",
                gamma=self._gamma_value,
                tol=float(self.tol),
                cache_size=float(self.cache_size),
                output_type="numpy",
            )
            self._model.fit(X, y.astype(np.float32))
            self._backend = "cuml"
        else:
            self._model = SklearnSVC(
                C=float(self.C),
                kernel="rbf",
                gamma=self._gamma_value,
                tol=float(self.tol),
                cache_size=float(self.cache_size),
                random_state=self.random_state,
            )
            self._model.fit(X, y)
            self._backend = "sklearn"

        # ---- orientation check: positive decision must mean positive class
        probe = X[: min(4000, len(X))]
        raw = np.asarray(self._model.decision_function(probe)).ravel()
        probe_pred = np.asarray(self._model.predict(probe)).ravel()

        positive_mask = probe_pred == self.classes_[-1]

        if positive_mask.any() and (~positive_mask).any():
            self._sign = (
                1.0
                if raw[positive_mask].mean() > raw[~positive_mask].mean()
                else -1.0
            )
        else:
            self._sign = 1.0

        return self

    def decision_function(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        raw = np.asarray(self._model.decision_function(X)).ravel()
        return self._sign * raw

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(
            scores >= 0, self.classes_[-1], self.classes_[0]
        ).astype(int)


class KeepColumnSelector(SelectKBest):
    """
    SelectKBest that always retains a nominated column.

    The forced column consumes one of the k slots, so the hybrid never
    receives more features than k. See FORCE_KEEP_IF_SCORE in PART 0.
    """

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


if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def build_pipeline(n_available, keep_index=None):
    """Identical steps and order for both supervised models."""

    selector = (
        KeepColumnSelector(
            score_func=mutual_info_classif,
            k=min(SELECT_K_VALUES[0], n_available),
            keep_index=keep_index,
        )
        if keep_index is not None
        else SelectKBest(
            score_func=mutual_info_classif,
            k=min(SELECT_K_VALUES[0], n_available),
        )
    )

    return ImbPipeline(
        steps=[
            ("selector", selector),
            # scaler BEFORE smote: SMOTE interpolates on Euclidean distance,
            # so on unscaled inputs the large-magnitude features would
            # dominate every synthetic sample.
            ("scaler", StandardScaler()),
            # smote INSIDE the pipeline: refits per CV fold, so no synthetic
            # point ever crosses into a validation fold.
            (
                "smote",
                SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=SMOTE_K_NEIGHBORS,
                ),
            ),
            (
                "svc",
                RBFSVMClassifier(
                    random_state=RANDOM_STATE,
                    use_gpu=USE_GPU_SVM,
                ),
            ),
        ],
        memory=CACHE_DIR,
    )


def build_grid(n_available):
    ks = [k for k in SELECT_K_VALUES if k <= n_available]
    if not ks:
        ks = [n_available]
    return {
        "selector__k": ks,
        "svc__C": SVM_C_VALUES,
        "svc__gamma": SVM_GAMMA_VALUES,
    }


cv_strategy = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# One GPU means one fit at a time. Parallel workers contend for VRAM.
SEARCH_N_JOBS = 1 if USE_GPU_SVM else -1

print(f"SVM backend      : {'cuML (GPU)' if USE_GPU_SVM else 'sklearn (CPU)'}")
print(f"GridSearch n_jobs: {SEARCH_N_JOBS}")
print(f"Scoring          : {SEARCH_SCORING}")
print("Steps            :", [n for n, _ in build_pipeline(10).steps])


# ============================================================================
# PART 8 - BASELINE 2: STANDALONE SVM (nine features, no IF score)
# ============================================================================

print("\n" + "=" * 80)
print("PART 8 - BASELINE 2: STANDALONE SVM")
print("=" * 80)

base_grid = build_grid(X_base_train.shape[1])
print("Parameter grid:", base_grid)

base_search = GridSearchCV(
    estimator=build_pipeline(X_base_train.shape[1]),
    param_grid=base_grid,
    scoring=SEARCH_SCORING,
    cv=cv_strategy,
    n_jobs=SEARCH_N_JOBS,
    verbose=1,
    refit=True,
    return_train_score=True,
    error_score="raise",
)

t0 = time.time()
base_search.fit(X_base_train, y_fit)
svmonly_training_time = time.time() - t0

BEST_SVM_ONLY = base_search.best_estimator_

print(f"\nStandalone SVM training time: {svmonly_training_time:.2f} s")
print("Best parameters:", base_search.best_params_)
print(f"Best CV {SEARCH_SCORING}: {base_search.best_score_:.4f}")


# ============================================================================
# PART 9 - PROPOSED MODEL: HYBRID IF-SVM
# ============================================================================

print("\n" + "=" * 80)
print("PART 9 - PROPOSED MODEL: HYBRID IF-SVM")
print("=" * 80)

hybrid_grid = build_grid(X_hyb_train.shape[1])
print("Parameter grid:", hybrid_grid)
print("(identical to the standalone SVM grid -- only the features differ)")

hybrid_pipeline = build_pipeline(
    X_hyb_train.shape[1],
    keep_index=IF_SCORE_POSITION if FORCE_KEEP_IF_SCORE else None,
)

grid_search = GridSearchCV(
    estimator=hybrid_pipeline,
    param_grid=hybrid_grid,
    scoring=SEARCH_SCORING,
    cv=cv_strategy,
    n_jobs=SEARCH_N_JOBS,
    verbose=2,
    refit=True,
    return_train_score=True,
    error_score="raise",
)

t0 = time.time()
grid_search.fit(X_hyb_train, y_fit)
svm_training_time = time.time() - t0

BEST_HYBRID_MODEL = grid_search.best_estimator_

print(f"\nHybrid SVM training time: {svm_training_time:.2f} s")
print("Best parameters:", grid_search.best_params_)
print(f"Best CV {SEARCH_SCORING}: {grid_search.best_score_:.4f}")


# ---------------------------------------------------- selected feature names

selector_step = BEST_HYBRID_MODEL.named_steps["selector"]
selected_features = X_hyb_train.columns[selector_step.get_support()].tolist()

print(f"\nSelected features ({len(selected_features)}):")
for i, feature in enumerate(selected_features, start=1):
    print(f"  {i:02d}. {feature}")

if_derived_selected = [
    f for f in selected_features if f == "if_score" or f.startswith("if_")
]

print(
    f"\nIF-derived features retained: "
    f"{len(if_derived_selected)} of {len(selected_features)}"
)
for feature in if_derived_selected:
    print(f"  - {feature}")


# What would selection have chosen WITHOUT the forced column?

mi_scores = np.nan_to_num(selector_step.scores_, nan=-np.inf)
mi_ranking = pd.DataFrame(
    {
        "Feature": X_hyb_train.columns,
        "Mutual_Information": mi_scores,
        "IF_Derived": [
            f == "if_score" or f.startswith("if_")
            for f in X_hyb_train.columns
        ],
    }
).sort_values("Mutual_Information", ascending=False)

mi_ranking["Rank"] = np.arange(1, len(mi_ranking) + 1)

if_score_rank = int(
    mi_ranking.loc[mi_ranking["Feature"] == "if_score", "Rank"].iloc[0]
)

print(
    f"\nUnforced mutual-information rank of if_score: "
    f"{if_score_rank} of {len(mi_ranking)}"
)
print("\nTop 15 features by mutual information:")
print(
    mi_ranking.head(15).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    )
)

if (
    FORCE_KEEP_IF_SCORE
    and if_score_rank > grid_search.best_params_["selector__k"]
):
    print(
        "\n  Note for the manuscript: mutual information alone would NOT "
        "have selected if_score at this k. It is retained by design because "
        "the proposed architecture specifies it as an input. State this "
        "explicitly rather than letting a reader discover it."
    )


cv_results_df = pd.DataFrame(grid_search.cv_results_)

search_columns = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "param_selector__k",
    "param_svc__C",
    "param_svc__gamma",
]

search_results_display = cv_results_df[search_columns].sort_values(
    "rank_test_score"
)

print("\nHyperparameter search results (top 10):")
print(
    search_results_display.head(10).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    )
)


# ============================================================================
# PART 10 - VALIDATION THRESHOLD TUNING
# ============================================================================
#
# Selecting on MCC rather than F1, for the reason in the header. The
# comparison table shows what every criterion WOULD have chosen -- put it in
# the results chapter so the choice is visibly reasoned.
#
# Test labels remain unseen here.
# ============================================================================

print("\n" + "=" * 80)
print("PART 10 - VALIDATION THRESHOLD TUNING")
print("=" * 80)


def tune_threshold(model, X_validation, y_validation, name,
                   criterion=THRESHOLD_CRITERION):

    t_start = time.time()
    scores = model.decision_function(X_validation)
    elapsed = time.time() - t_start

    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 241)))

    rows = []
    for threshold in candidates:

        pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_validation, pred, labels=[0, 1])
        TN, FP, FN, TP = (
            int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        )

        sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
        specificity = TN / (TN + FP) if (TN + FP) else 0.0

        rows.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy_score(y_validation, pred),
                "Precision": precision_score(
                    y_validation, pred, zero_division=0
                ),
                "Recall": sensitivity,
                "Specificity": specificity,
                "F1": f1_score(y_validation, pred, zero_division=0),
                "F0.5": fbeta_score(
                    y_validation, pred, beta=0.5, zero_division=0
                ),
                "Balanced_Accuracy": 0.5 * (sensitivity + specificity),
                "Youden_J": sensitivity + specificity - 1.0,
                "MCC": matthews_corrcoef(y_validation, pred),
            }
        )

    frame = pd.DataFrame(rows)
    best = frame.loc[frame[criterion].idxmax()]

    comparison = pd.DataFrame(
        [
            {
                "Criterion": c,
                "Threshold": frame.loc[frame[c].idxmax(), "Threshold"],
                "Val_Accuracy": frame.loc[frame[c].idxmax(), "Accuracy"],
                "Val_Recall": frame.loc[frame[c].idxmax(), "Recall"],
                "Val_Specificity": frame.loc[
                    frame[c].idxmax(), "Specificity"
                ],
                "Val_F1": frame.loc[frame[c].idxmax(), "F1"],
                "Val_MCC": frame.loc[frame[c].idxmax(), "MCC"],
            }
            for c in ["F1", "MCC", "Balanced_Accuracy", "Youden_J"]
        ]
    )

    print(f"\n{name}  (selecting on {criterion})")
    print(f"  validation scoring time : {elapsed:.2f} s")
    print(
        "\n"
        + comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\n  chosen threshold: {best['Threshold']:.6f}")

    return float(best["Threshold"]), frame, comparison, elapsed


(
    BEST_SVM_THRESHOLD,
    threshold_df,
    hybrid_criteria_df,
    hybrid_val_time,
) = tune_threshold(BEST_HYBRID_MODEL, X_hyb_val, y_val, "Hybrid IF-SVM")

(
    BEST_BASE_THRESHOLD,
    base_threshold_df,
    base_criteria_df,
    base_val_time,
) = tune_threshold(BEST_SVM_ONLY, X_base_val, y_val, "Standalone SVM")


# ============================================================================
# PART 11 - FINAL TEST EVALUATION (test labels used from here only)
# ============================================================================

print("\n" + "=" * 80)
print("PART 11 - FINAL TEST EVALUATION")
print("=" * 80)


def evaluate(name, y_true, y_pred, ranking_score=None):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = (
        int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    )

    sensitivity = TP / (TP + FN) if (TP + FN) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": sensitivity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "Specificity": specificity,
        "Balanced_Accuracy": 0.5 * (sensitivity + specificity),
        "FPR": FP / (FP + TN) if (FP + TN) else 0.0,
        "FNR": FN / (FN + TP) if (FN + TP) else 0.0,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
    }

    if ranking_score is not None:
        row["ROC_AUC"] = roc_auc_score(y_true, ranking_score)
        row["PR_AUC"] = average_precision_score(y_true, ranking_score)

    return row, cm


t0 = time.time()
base_test_scores = BEST_SVM_ONLY.decision_function(X_svmonly_test)
svmonly_prediction_time = time.time() - t0
base_test_pred = (base_test_scores >= BEST_BASE_THRESHOLD).astype(int)

t0 = time.time()
hybrid_test_decision_scores = BEST_HYBRID_MODEL.decision_function(
    X_enhanced_test
)
svm_prediction_time = time.time() - t0
hybrid_test_pred = (
    hybrid_test_decision_scores >= BEST_SVM_THRESHOLD
).astype(int)

if_row, if_cm = evaluate(
    "Standalone Isolation Forest (Shukla)",
    y_test,
    y_test_pred_if,
    if_ranking_score,
)
base_row, base_cm = evaluate(
    "Standalone SVM", y_test, base_test_pred, base_test_scores
)
hybrid_row, hybrid_cm = evaluate(
    "Hybrid IF-SVM", y_test, hybrid_test_pred, hybrid_test_decision_scores
)

results_df = pd.DataFrame([if_row, base_row, hybrid_row])

print(
    "\n" + results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
)

print("\nConfusion matrices  [[TN FP] [FN TP]]")
for label, cm in (
    ("Standalone Isolation Forest", if_cm),
    ("Standalone SVM", base_cm),
    ("Hybrid IF-SVM", hybrid_cm),
):
    print(f"\n{label}:")
    print(cm)

print("\nHybrid IF-SVM classification report:")
print(
    classification_report(
        y_test,
        hybrid_test_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)

print(
    f"\nBaseline IF test accuracy : {if_row['Accuracy']:.4f}  "
    f"(paper Table 4 reports {PAPER_TEST_ACCURACY:.4f})"
)
print(
    f"Difference from paper     : "
    f"{if_row['Accuracy'] - PAPER_TEST_ACCURACY:+.4f}"
)


# ============================================================================
# PART 12 - SANITY CHECKS THE PANEL WILL ASK FOR
# ============================================================================

print("\n" + "=" * 80)
print("PART 12 - SANITY CHECKS")
print("=" * 80)


# ---- 12a. trivial baselines
majority_floor = max(test_attack_fraction, 1.0 - test_attack_fraction)
trivial_f1 = 2.0 * test_attack_fraction / (1.0 + test_attack_fraction)

print("\n12a. TRIVIAL BASELINES")
print(f"  Test attack prior           : {test_attack_fraction:.4f}")
print(f"  Predict-all-attack accuracy : {majority_floor:.4f}")
print(f"  Predict-all-attack F1       : {trivial_f1:.4f}   <-- note this")
print("  Predict-all-attack MCC      : 0.0000")

floor_df = pd.DataFrame(
    {
        "Model": results_df["Model"],
        "Accuracy": results_df["Accuracy"],
        "Above_Accuracy_Floor": results_df["Accuracy"] - majority_floor,
        "F1": results_df["F1"],
        "Above_F1_Floor": results_df["F1"] - trivial_f1,
        "MCC": results_df["MCC"],
    }
)

print("\n" + floor_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(
    "\n  Any model whose Above_F1_Floor is near zero is barely beating "
    "'call everything an attack'. Do not lead with F1 for that model -- "
    "its MCC is the number that means something."
)


# ---- 12b. IF score direction
print("\n12b. ISOLATION FOREST SCORE DIRECTION")

pred_switched = np.where(test_scores > BEST_THRESHOLD, 0, 1)
pred_standard = np.where(test_scores > BEST_THRESHOLD, 1, 0)

direction_df = pd.DataFrame(
    [
        {
            "Mapping": "high score -> Normal (Table 7, label-switched)",
            "Accuracy": accuracy_score(y_test, pred_switched),
            "F1": f1_score(y_test, pred_switched, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred_switched),
            "ROC_AUC": roc_auc_score(y_test, -test_scores),
        },
        {
            "Mapping": "high score -> Attack (standard IF convention)",
            "Accuracy": accuracy_score(y_test, pred_standard),
            "F1": f1_score(y_test, pred_standard, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred_standard),
            "ROC_AUC": roc_auc_score(y_test, test_scores),
        },
    ]
)

print(
    "\n"
    + direction_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
)
print(
    f"\n  Direction used: "
    f"{'label-switched' if LABEL_SWITCHED else 'standard'}, justified by "
    f"attack being the {'majority' if LABEL_SWITCHED else 'minority'} class "
    f"({train_attack_fraction:.1%} of the training partition)."
)


# ---- 12c. fusion ablation
print("\n12c. FUSION ABLATION")

ablation_df = pd.DataFrame(
    [
        {
            "Comparison": "Hybrid vs standalone SVM (fusion effect)",
            "Delta_Accuracy": hybrid_row["Accuracy"] - base_row["Accuracy"],
            "Delta_F1": hybrid_row["F1"] - base_row["F1"],
            "Delta_MCC": hybrid_row["MCC"] - base_row["MCC"],
            "Delta_ROC_AUC": hybrid_row["ROC_AUC"] - base_row["ROC_AUC"],
            "Delta_PR_AUC": hybrid_row["PR_AUC"] - base_row["PR_AUC"],
        },
        {
            "Comparison": "Hybrid vs standalone IF (headline claim)",
            "Delta_Accuracy": hybrid_row["Accuracy"] - if_row["Accuracy"],
            "Delta_F1": hybrid_row["F1"] - if_row["F1"],
            "Delta_MCC": hybrid_row["MCC"] - if_row["MCC"],
            "Delta_ROC_AUC": hybrid_row["ROC_AUC"] - if_row["ROC_AUC"],
            "Delta_PR_AUC": hybrid_row["PR_AUC"] - if_row["PR_AUC"],
        },
        {
            "Comparison": "Standalone SVM vs standalone IF (supervision)",
            "Delta_Accuracy": base_row["Accuracy"] - if_row["Accuracy"],
            "Delta_F1": base_row["F1"] - if_row["F1"],
            "Delta_MCC": base_row["MCC"] - if_row["MCC"],
            "Delta_ROC_AUC": base_row["ROC_AUC"] - if_row["ROC_AUC"],
            "Delta_PR_AUC": base_row["PR_AUC"] - if_row["PR_AUC"],
        },
    ]
)

print(
    "\n"
    + ablation_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}")
)

fusion_mcc = hybrid_row["MCC"] - base_row["MCC"]
fusion_auc = hybrid_row["ROC_AUC"] - base_row["ROC_AUC"]

print(
    f"\n  Thresholded gain (MCC): {fusion_mcc:+.4f}"
    f"\n  Ranking gain (ROC-AUC): {fusion_auc:+.4f}"
)

if fusion_auc > 0.02 and fusion_mcc < 0.02:
    print(
        "\n  The IF score is carrying real discriminative information -- the "
        "AUC gap proves it -- but the chosen operating point is not "
        "converting it into predictions. Check 12d before concluding "
        "anything about the fusion."
    )
elif fusion_mcc > 0.01:
    print(
        "\n  The hybrid beats the standalone SVM on the thresholded metrics "
        "AND on ranking. That gap IS the contribution -- lead with it, not "
        "with the comparison against the unsupervised baseline."
    )
else:
    print(
        "\n  The hybrid does not clearly beat the standalone SVM. Report it. "
        "A negative result cleanly reported is stronger than a buried one. "
        "Check PART 13 -- per-category gains on rare attack families are "
        "still a real contribution when aggregate metrics tie."
    )


# ---- 12d. threshold headroom (DIAGNOSTIC ONLY)
print("\n12d. THRESHOLD HEADROOM  (DIAGNOSTIC ONLY - uses test labels)")


def headroom(name, scores, y_true, chosen_pred):

    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 400)))

    best_acc, best_acc_mcc = -1.0, 0.0
    best_mcc, best_mcc_acc = -1.0, 0.0

    for t in grid:
        pred = (scores >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        mcc = matthews_corrcoef(y_true, pred)

        if acc > best_acc:
            best_acc, best_acc_mcc = acc, mcc
        if mcc > best_mcc:
            best_mcc, best_mcc_acc = mcc, acc

    achieved_acc = accuracy_score(y_true, chosen_pred)
    achieved_mcc = matthews_corrcoef(y_true, chosen_pred)

    return {
        "Model": name,
        "Achieved_Accuracy": achieved_acc,
        "Oracle_Accuracy": best_acc,
        "Accuracy_Left": best_acc - achieved_acc,
        "Achieved_MCC": achieved_mcc,
        "Oracle_MCC": best_mcc,
        "MCC_Left": best_mcc - achieved_mcc,
        "Oracle_MCC_Accuracy": best_mcc_acc,
    }


headroom_df = pd.DataFrame(
    [
        headroom("Standalone SVM", base_test_scores, y_test, base_test_pred),
        headroom(
            "Hybrid IF-SVM",
            hybrid_test_decision_scores,
            y_test,
            hybrid_test_pred,
        ),
    ]
)

print(
    "\n"
    + headroom_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
)
print(
    "\n  Oracle columns use TEST LABELS to find the best threshold that "
    "exists. They are a diagnostic and an upper bound -- NEVER report them "
    "as your model's performance.\n"
    "  Large *_Left values mean the model is fine and threshold transfer is "
    "the problem (usually prior shift between partitions); near-zero means "
    "the model itself is the ceiling. Either way, this belongs in the "
    "limitations discussion, not the results table."
)


# ============================================================================
# PART 13 - PER-CATEGORY ANALYSIS (Objective 1.4.2.5)
# ============================================================================

per_category_df = None

if HAS_ATTACK_CAT:

    print("\n" + "=" * 80)
    print("PART 13 - PER-ATTACK-CATEGORY DETECTION RATE")
    print("=" * 80)

    categories = test_df["attack_cat"].astype(str).to_numpy()
    rows = []

    for category in sorted(pd.unique(categories)):

        mask = categories == category
        target = 0 if category.strip().lower() == "normal" else 1

        rows.append(
            {
                "Category": category,
                "Records": int(mask.sum()),
                "IF_Correct_Rate": float(
                    (y_test_pred_if[mask] == target).mean()
                ),
                "SVM_Correct_Rate": float(
                    (base_test_pred[mask] == target).mean()
                ),
                "Hybrid_Correct_Rate": float(
                    (hybrid_test_pred[mask] == target).mean()
                ),
            }
        )

    per_category_df = pd.DataFrame(rows).sort_values(
        "Records", ascending=False
    )

    per_category_df["Hybrid_minus_SVM"] = (
        per_category_df["Hybrid_Correct_Rate"]
        - per_category_df["SVM_Correct_Rate"]
    )
    per_category_df["Hybrid_minus_IF"] = (
        per_category_df["Hybrid_Correct_Rate"]
        - per_category_df["IF_Correct_Rate"]
    )

    print(
        "\n"
        + per_category_df.to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    wins = int((per_category_df["Hybrid_minus_SVM"] > 0).sum())
    print(
        f"\n  Hybrid beats the standalone SVM on {wins} of "
        f"{len(per_category_df)} categories. For 'Normal' the rate is the "
        "proportion correctly kept as normal; for every attack family it is "
        "the detection rate. Rare-family gains are a real contribution even "
        "when aggregate metrics tie -- that is what Objective 1.4.2.5 was "
        "written to surface."
    )

else:
    print("\nPART 13 skipped: no 'attack_cat' column in the test partition.")


# ============================================================================
# PART 14 - TIMING COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("PART 14 - TRAINING AND INFERENCE TIME")
print("=" * 80)

timing_df = pd.DataFrame(
    [
        {
            "Model": "Standalone Isolation Forest",
            "Training_Time_s": if_training_time,
            "Inference_Time_s": if_test_score_time,
            "Inference_Rows": len(X_enhanced_test),
        },
        {
            "Model": "Standalone SVM",
            "Training_Time_s": svmonly_training_time,
            "Inference_Time_s": svmonly_prediction_time,
            "Inference_Rows": len(X_enhanced_test),
        },
        {
            "Model": "Hybrid IF-SVM",
            # includes every IF stage the hybrid depends on
            "Training_Time_s": (
                if_training_time + multi_scale_time + svm_training_time
            ),
            "Inference_Time_s": if_test_score_time + svm_prediction_time,
            "Inference_Rows": len(X_enhanced_test),
        },
    ]
)

timing_df["Rows_Per_Second"] = (
    timing_df["Inference_Rows"] / timing_df["Inference_Time_s"]
)
timing_df["Backend"] = [
    "GPU" if USE_GPU_IF else "CPU",
    "GPU" if USE_GPU_SVM else "CPU",
    (
        "GPU"
        if (USE_GPU_IF and USE_GPU_SVM)
        else ("mixed" if (USE_GPU_IF or USE_GPU_SVM) else "CPU")
    ),
]

print("\n" + timing_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(
    "\n  Hybrid timings INCLUDE every Isolation Forest stage, because the "
    "hybrid cannot run without them. Report the backend alongside the "
    "numbers -- a GPU timing is not comparable to the paper's CPU timing."
)


# ============================================================================
# PART 15 - CONSOLIDATED RESULTS AND IMPROVEMENT SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PART 15 - FINAL EXPERIMENT RESULTS")
print("=" * 80)

final_results_df = results_df.copy()

final_results_df["Dataset"] = "UNSW-NB15"
final_results_df["Base_Features"] = N_BASE
final_results_df["IF_Score_Feature"] = ["No", "No", "Yes"]
final_results_df["IF_Scales"] = [
    str(SAMPLE_SIZE), "N/A", str(IF_SCALES)
]
final_results_df["Engineered_Features"] = [0, 0, N_ENGINEERED]
final_results_df["Selected_Features"] = [
    N_BASE,
    base_search.best_params_["selector__k"],
    grid_search.best_params_["selector__k"],
]
final_results_df["IF_Trees"] = [N_TREES, np.nan, N_TREES]
final_results_df["IF_Contamination"] = [
    BEST_CONTAMINATION, np.nan, BEST_CONTAMINATION
]
final_results_df["SVM_C"] = [
    np.nan,
    base_search.best_params_["svc__C"],
    grid_search.best_params_["svc__C"],
]
final_results_df["SVM_Gamma"] = [
    "N/A",
    base_search.best_params_["svc__gamma"],
    grid_search.best_params_["svc__gamma"],
]
final_results_df["Decision_Threshold"] = [
    BEST_THRESHOLD, BEST_BASE_THRESHOLD, BEST_SVM_THRESHOLD
]
final_results_df["Training_Time_s"] = timing_df["Training_Time_s"].to_numpy()
final_results_df["Inference_Time_s"] = timing_df["Inference_Time_s"].to_numpy()
final_results_df["Backend"] = timing_df["Backend"].to_numpy()

print(
    "\n"
    + final_results_df.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    )
)

metrics_for_improvement = [
    "Accuracy", "Precision", "Recall", "F1", "F0.5",
    "Specificity", "Balanced_Accuracy", "MCC", "ROC_AUC", "PR_AUC",
]

improvement_rows = []

for metric in metrics_for_improvement:

    if_value = float(if_row[metric])
    base_value = float(base_row[metric])
    hybrid_value = float(hybrid_row[metric])

    improvement_rows.append(
        {
            "Metric": metric,
            "IF_Baseline": if_value,
            "SVM_Baseline": base_value,
            "Hybrid": hybrid_value,
            "Hybrid_minus_IF": hybrid_value - if_value,
            "Hybrid_minus_SVM": hybrid_value - base_value,
            "Relative_vs_IF_Percent": (
                (hybrid_value - if_value) / if_value * 100.0
                if if_value != 0.0 and not np.isnan(if_value)
                else np.nan
            ),
        }
    )

improvement_df = pd.DataFrame(improvement_rows)

print("\n" + "=" * 80)
print("IMPROVEMENT SUMMARY")
print("=" * 80)
print(
    "\n"
    + improvement_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
)


# ============================================================================
# PART 16 - SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 16 - SAVING OUTPUTS")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(frame, filename):
    if frame is not None:
        frame.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


save(contamination_df, "baseline_if_contamination_results.csv")
save(results_df, "test_results_all_models.csv")
save(final_results_df, "final_experiment_results.csv")
save(improvement_df, "improvement_summary.csv")
save(ablation_df, "fusion_ablation.csv")
save(direction_df, "if_score_direction_check.csv")
save(floor_df, "trivial_baseline_check.csv")
save(headroom_df, "threshold_headroom_DIAGNOSTIC.csv")
save(timing_df, "timing_comparison.csv")
save(threshold_df, "hybrid_validation_thresholds.csv")
save(base_threshold_df, "svm_validation_thresholds.csv")
save(hybrid_criteria_df, "hybrid_threshold_criteria.csv")
save(base_criteria_df, "svm_threshold_criteria.csv")
save(search_results_display, "hybrid_hyperparameter_results.csv")
save(mi_ranking, "mutual_information_ranking.csv")
save(per_category_df, "per_attack_category_results.csv")

save(
    pd.DataFrame({"Feature": X_enhanced_train.columns}),
    "hybrid_feature_list.csv",
)
save(
    pd.DataFrame({"Selected_Feature": selected_features}),
    "hybrid_selected_features.csv",
)
save(
    pd.DataFrame(
        {
            "True_Label": y_test,
            "IF_Anomaly_Score": test_scores,
            "IF_Predicted_Label": y_test_pred_if,
            "SVM_Decision_Score": base_test_scores,
            "SVM_Predicted_Label": base_test_pred,
            "Hybrid_Decision_Score": hybrid_test_decision_scores,
            "Hybrid_Predicted_Label": hybrid_test_pred,
        }
    ),
    "test_predictions_all_models.csv",
)

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Dataset", "Training_Records", "Testing_Records",
            "Train_Attack_Prior", "Test_Attack_Prior", "Prior_Shift",
            "Base_Features", "Engineered_Features", "Total_Hybrid_Features",
            "IF_Trees", "IF_Sample_Size", "IF_Fusion_Scales",
            "IF_Best_Contamination", "IF_Best_Threshold", "IF_Label_Direction",
            "IF_Backend", "SVM_Backend",
            "SVM_Pool_Size", "Validation_Fraction",
            "Search_Scoring", "Threshold_Criterion",
            "Force_Keep_IF_Score", "IF_Score_MI_Rank",
            "SVM_Kernel",
            "SVM_Baseline_C", "SVM_Baseline_Gamma", "SVM_Baseline_k",
            "Hybrid_C", "Hybrid_Gamma", "Hybrid_k",
            "Hybrid_Decision_Threshold",
            "SMOTE", "CV_Folds", "Random_State",
        ],
        "Value": [
            "UNSW-NB15", len(train_df), len(test_df),
            round(train_attack_fraction, 4), round(test_attack_fraction, 4),
            round(prior_shift, 4),
            N_BASE, N_ENGINEERED, N_TOTAL,
            N_TREES, SAMPLE_SIZE, str(IF_SCALES),
            BEST_CONTAMINATION, BEST_THRESHOLD,
            (
                "score > threshold => NORMAL (0); else ATTACK (1) "
                "[Table 7 label-switched]"
                if LABEL_SWITCHED
                else "score > threshold => ATTACK (1); else NORMAL (0) "
                "[standard IF]"
            ),
            "CuPy GPU" if USE_GPU_IF else "NumPy CPU",
            "cuML GPU" if USE_GPU_SVM else "sklearn CPU",
            SVM_POOL_SIZE, VALIDATION_SIZE,
            SEARCH_SCORING, THRESHOLD_CRITERION,
            FORCE_KEEP_IF_SCORE, if_score_rank,
            "RBF",
            base_search.best_params_["svc__C"],
            base_search.best_params_["svc__gamma"],
            base_search.best_params_["selector__k"],
            grid_search.best_params_["svc__C"],
            grid_search.best_params_["svc__gamma"],
            grid_search.best_params_["selector__k"],
            BEST_SVM_THRESHOLD,
            f"inside CV pipeline, k_neighbors={SMOTE_K_NEIGHBORS}",
            CV_FOLDS, RANDOM_STATE,
        ],
    }
)

save(configuration_df, "experiment_configuration.csv")

joblib.dump(IF_MODEL, os.path.join(OUTPUT_DIR, "isolation_forest_model.joblib"))
joblib.dump(
    BEST_SVM_ONLY, os.path.join(OUTPUT_DIR, "standalone_svm_model.joblib")
)
joblib.dump(
    BEST_HYBRID_MODEL, os.path.join(OUTPUT_DIR, "hybrid_if_svm_model.joblib")
)
joblib.dump(minmax_scaler, os.path.join(OUTPUT_DIR, "minmax_scaler.joblib"))

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

print(f"\nOutput directory: {OUTPUT_DIR}")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("  ", filename)


# ============================================================================
# PART 17 - THESIS SUMMARY
# ============================================================================

print("\n")
print("=" * 100)
print("FINAL THESIS EXPERIMENT SUMMARY")
print("=" * 100)

print(
    f"\nCompute: IF on {'GPU' if USE_GPU_IF else 'CPU'}, "
    f"SVM on {'GPU' if USE_GPU_SVM else 'CPU'}"
)
print(f"Selection: {SEARCH_SCORING} (search), {THRESHOLD_CRITERION} (threshold)")

print("\nBASELINE 1 - STANDALONE ISOLATION FOREST (Shukla et al., 2023)")
print("-" * 100)
print(f"  Features            : {N_BASE}")
print(f"  Trees / sample size : {N_TREES} / {SAMPLE_SIZE}")
print(f"  Contamination       : {BEST_CONTAMINATION:.2f}")
print(f"  Accuracy            : {if_row['Accuracy']:.4f}")
print(f"  F1                  : {if_row['F1']:.4f}")
print(f"  MCC                 : {if_row['MCC']:.4f}")
print(f"  Paper reported acc. : {PAPER_TEST_ACCURACY:.4f}")

print("\nBASELINE 2 - STANDALONE SVM")
print("-" * 100)
print(f"  Features            : {N_BASE} (no IF score)")
print(f"  Selected features   : {base_search.best_params_['selector__k']}")
print(
    f"  C / gamma           : {base_search.best_params_['svc__C']} / "
    f"{base_search.best_params_['svc__gamma']}"
)
print(f"  Accuracy            : {base_row['Accuracy']:.4f}")
print(f"  F1                  : {base_row['F1']:.4f}")
print(f"  MCC                 : {base_row['MCC']:.4f}")
print(f"  ROC-AUC             : {base_row['ROC_AUC']:.4f}")

print("\nPROPOSED - HYBRID ISOLATION FOREST-SVM")
print("-" * 100)
print(f"  Total input features: {N_TOTAL}")
print(f"  IF fusion scales    : {IF_SCALES}")
print(f"  Selected features   : {grid_search.best_params_['selector__k']}")
print(f"  IF-derived selected : {len(if_derived_selected)}")
print(f"  if_score MI rank    : {if_score_rank} of {N_TOTAL}")
print(
    f"  C / gamma           : {grid_search.best_params_['svc__C']} / "
    f"{grid_search.best_params_['svc__gamma']}"
)
print(f"  Accuracy            : {hybrid_row['Accuracy']:.4f}")
print(f"  Precision           : {hybrid_row['Precision']:.4f}")
print(f"  Recall              : {hybrid_row['Recall']:.4f}")
print(f"  Specificity         : {hybrid_row['Specificity']:.4f}")
print(f"  F1                  : {hybrid_row['F1']:.4f}")
print(f"  MCC                 : {hybrid_row['MCC']:.4f}")
print(f"  ROC-AUC             : {hybrid_row['ROC_AUC']:.4f}")
print(f"  PR-AUC              : {hybrid_row['PR_AUC']:.4f}")

print("\nHEADLINE NUMBERS FOR THE MANUSCRIPT")
print("-" * 100)
print(
    f"  Hybrid vs IF baseline  : "
    f"{hybrid_row['Accuracy'] - if_row['Accuracy']:+.4f} accuracy, "
    f"{hybrid_row['MCC'] - if_row['MCC']:+.4f} MCC"
)
print(
    f"  Hybrid vs SVM baseline : "
    f"{hybrid_row['Accuracy'] - base_row['Accuracy']:+.4f} accuracy, "
    f"{hybrid_row['MCC'] - base_row['MCC']:+.4f} MCC, "
    f"{hybrid_row['ROC_AUC'] - base_row['ROC_AUC']:+.4f} ROC-AUC"
    f"   <-- the contribution"
)
print(f"  Predict-all-attack floor: acc {majority_floor:.4f}, F1 {trivial_f1:.4f}")

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)