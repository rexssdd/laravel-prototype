"""Parity checks against the supplied experiment, without running its training job."""

import ast
from contextlib import redirect_stdout
from functools import lru_cache
import math
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
import predict


def reference_definitions():
    source = ROOT / "ml" / "hybrid_if_svm_final.py"
    names = {
        "harmonic_number", "c_factor", "FlatIsolationTree", "CustomIsolationForest",
        "create_enhanced_features", "if_predict", "RBFSVMClassifier",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {
        "np": np, "math": math, "lru_cache": lru_cache, "cp": None,
        "EPS": 1e-9, "RANDOM_STATE": 42,
        "BaseEstimator": type("BaseEstimator", (), {}),
        "ClassifierMixin": type("ClassifierMixin", (), {}),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


REFERENCE = reference_definitions()


class PredictionLogicTest(unittest.TestCase):
    def test_exported_forest_matches_reference_on_unseen_rows(self):
        rng = np.random.default_rng(42)
        forest = REFERENCE["CustomIsolationForest"](
            n_trees=7, sample_size=16, random_state=42, verbose=False,
        ).fit(rng.normal(size=(50, 9)))
        exported = {
            "n_trees": forest.n_trees, "max_depth": forest.max_depth,
            "c_n": REFERENCE["c_factor"](forest.sample_size),
            "trees": [{name: getattr(tree, name) for name in
                       ["feature", "threshold", "left", "right", "c_adjust"]}
                      for tree in forest.trees],
        }
        rows = rng.normal(size=(23, 9))

        actual = predict.score_forest(exported, rows)

        np.testing.assert_array_equal(actual, forest.anomaly_score(rows))

    def test_features_match_reference_and_use_training_preprocessing(self):
        columns = ["sttl", "ct_state_ttl", "dload", "rate", "dmean",
                   "dttl", "ackdat", "sload", "sbytes"]
        raw = pd.DataFrame([[2.0] * 9, [-3.0] * 9, [0.0] * 9], columns=columns)
        raw.loc[0, "rate"] = np.nan
        raw.loc[1, "sload"] = np.inf
        clean = raw.replace([np.inf, -np.inf], np.nan).fillna(7.0)
        scores = {
            256: np.array([0.2, 0.7, 0.3]), 64: np.array([0.9, 0.3, 0.4]),
            128: np.array([0.4, 0.6, 0.8]), 512: np.array([0.6, 0.5, 0.1]),
        }
        reference_input = clean.copy()
        for scale, values in scores.items():
            reference_input["if_score" if scale == 256 else f"if_score_s{scale}"] = values
        matrix = np.column_stack(list(scores.values()))
        reference_input["if_scale_spread"] = matrix.std(axis=1)
        reference_input["if_scale_range"] = matrix.max(axis=1) - matrix.min(axis=1)
        reference_input["if_scale_mean"] = matrix.mean(axis=1)
        expected = REFERENCE["create_enhanced_features"](reference_input)
        bundle = {
            "shukla_features": columns, "train_medians": dict.fromkeys(columns, 7.0),
            "enhanced_medians": {}, "minmax_scale": np.full(9, 0.5),
            "minmax_offset": np.full(9, -1.0), "if_sample_size": 256,
            "if_scales": [64, 128, 256, 512], "forests": {s: s for s in scores},
            "feature_order": list(expected.columns),
        }

        with patch.object(predict, "score_forest", side_effect=lambda s, x: scores[s]) as scorer:
            actual, baseline = predict.build_features(raw, bundle)

        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        np.testing.assert_array_equal(baseline, scores[256])
        np.testing.assert_array_equal(scorer.call_args_list[0].args[1], clean.to_numpy() * 0.5 - 1.0)
        self.assertEqual([call.args[0] for call in scorer.call_args_list], [256, 64, 128, 512])

        bundle.pop("minmax_scale")
        bundle.pop("minmax_offset")
        bundle.update(minmax_min=np.full(9, 2.0), minmax_range=np.full(9, 2.0))
        with patch.object(predict, "score_forest", side_effect=lambda s, x: scores[s]) as scorer:
            legacy, _ = predict.build_features(raw, bundle)
        pd.testing.assert_frame_equal(legacy, expected, check_exact=True)
        np.testing.assert_array_equal(scorer.call_args_list[0].args[1], clean.to_numpy() * 0.5 - 1.0)

    def test_svm_matches_reference_float32_input_and_orientation(self):
        support = np.array([[0.0, 1.0], [0.5, -0.25]])
        dual = np.array([-0.7, 0.4])

        class Kernel:
            def decision_function(self, values):
                values = values.astype(np.float64)
                distances = ((values[:, None, :] - support[None, :, :]) ** 2).sum(axis=2)
                return np.exp(-0.6 * distances) @ dual + 0.15

        reference = REFERENCE["RBFSVMClassifier"]()
        reference._model = Kernel()
        reference._sign = -1.0
        frame = pd.DataFrame({"a": [1.123456789, -0.99999999], "b": [0.22222222, 2.33333333]})
        spec = {
            "columns": ["a", "b"], "scaler_mean": np.array([0.2, -0.1]),
            "scaler_scale": np.array([1.3, 0.7]), "support_vectors": support,
            "dual_coef": dual, "intercept": 0.15, "gamma": 0.6, "sign": -1.0,
        }
        standardized = (frame.to_numpy() - spec["scaler_mean"]) / spec["scaler_scale"]

        actual = predict.svm_decision(spec, frame)

        np.testing.assert_allclose(actual, reference.decision_function(standardized), rtol=0, atol=1e-15)

    def test_metrics_include_average_precision_with_tied_scores(self):
        result = predict.metrics([0, 1, 0, 1], [0, 1, 1, 0], np.array([0.1, 0.8, 0.8, 0.2]))

        self.assertEqual({name: result[name] for name in ["tn", "fp", "fn", "tp"]},
                         {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        for name in ["accuracy", "precision", "recall", "f1", "f0_5", "specificity", "balanced_accuracy", "fpr", "fnr"]:
            self.assertEqual(result[name], 0.5)
        self.assertEqual(result["mcc"], 0.0)
        self.assertEqual(result["roc_auc"], 0.625)
        self.assertAlmostEqual(result["pr_auc"], 7 / 12)

    def test_single_class_metrics_remain_json_safe(self):
        result = predict.metrics([0, 0], [0, 0], np.array([0.1, 0.2]))

        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["pr_auc"], 0.0)
        self.assertEqual(result["mcc"], 0.0)

    def test_worker_uses_saved_thresholds_and_reference_label_direction(self):
        raw = pd.DataFrame({"rate": [1, 2, 3], "label": [0, 1, 0]})
        forest_scores = np.array([0.2, 0.5, 0.8])
        decisions = np.array([-0.1, 0.0, 0.1])
        for switched in [True, False]:
            with self.subTest(switched=switched):
                bundle = {
                    "shukla_features": ["rate"], "if_label_switched": switched,
                    "if_threshold": 0.5, "svm_baseline": {}, "svm_hybrid": {},
                    "threshold_baseline": 0.0, "threshold_hybrid": 0.1,
                }
                output = io.StringIO()
                with patch.object(sys, "argv", ["predict.py", "--bundle", "synthetic", "--input", "synthetic.csv"]), \
                     patch.object(predict.joblib, "load", return_value=bundle), \
                     patch.object(pd, "read_csv", return_value=raw.copy()), \
                     patch.object(predict, "build_features", return_value=(raw, forest_scores)), \
                     patch.object(predict, "svm_decision", return_value=decisions), \
                     redirect_stdout(output):
                    status = predict.main()

                result = json.loads(output.getvalue())
                self.assertEqual(status, 0)
                self.assertEqual([row["baseline"] for row in result["preview"]],
                                 REFERENCE["if_predict"](forest_scores, 0.5, switched).tolist())
                self.assertEqual([row["svm"] for row in result["preview"]], [0, 1, 1])
                self.assertEqual([row["hybrid"] for row in result["preview"]], [0, 0, 1])

    def test_invalid_labels_are_rejected_before_inference(self):
        for label in [None, "invalid", 0.5, 2, -1]:
            with self.subTest(label=label):
                output = io.StringIO()
                with patch.object(sys, "argv", ["predict.py", "--bundle", "synthetic", "--input", "synthetic.csv"]), \
                     patch.object(predict.joblib, "load", return_value={"shukla_features": ["rate"]}), \
                     patch.object(pd, "read_csv", return_value=pd.DataFrame({"rate": [1], "label": [label]})), \
                     patch.object(predict, "build_features") as build, redirect_stdout(output):
                    status = predict.main()

                self.assertEqual(status, 2)
                self.assertFalse(json.loads(output.getvalue())["ok"])
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
