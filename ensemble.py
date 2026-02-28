import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pickle
import mlflow

class FraudEnsemble:
    def __init__(self, xgb_weight=0.7, iso_weight=0.3, threshold=0.5):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=10, eval_metric='aucpr',
            tree_method='hist', enable_categorical=True,
        )
        self.iso_forest = IsolationForest(
            n_estimators=200, contamination=0.01,
            max_features=0.8, random_state=42,
        )
        self.xgb_weight = xgb_weight
        self.iso_weight = iso_weight
        self.threshold = threshold

    def train(self, X_train, y_train, X_val=None, y_val=None):
        with mlflow.start_run():
            # Train XGBoost
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.xgb_model.fit(
                X_train, y_train, eval_set=eval_set,
                verbose=False,
            )
            # Train Isolation Forest on normal transactions only
            normal_mask = y_train == 0
            self.iso_forest.fit(X_train[normal_mask])

            # Log metrics
            if X_val is not None:
                preds = self.predict(X_val)
                metrics = {
                    "precision": precision_score(y_val, preds["is_fraud"]),
                    "recall": recall_score(y_val, preds["is_fraud"]),
                    "f1": f1_score(y_val, preds["is_fraud"]),
                    "auc_roc": roc_auc_score(y_val, preds["fraud_probability"]),
                }
                mlflow.log_metrics(metrics)
                mlflow.log_params({
                    "xgb_weight": self.xgb_weight,
                    "threshold": self.threshold,
                })

    def predict(self, X) -> dict:
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        iso_scores = self.iso_forest.decision_function(X)
        iso_normalized = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
        combined = self.xgb_weight * xgb_proba + self.iso_weight * iso_normalized
        return {
            "fraud_probability": combined,
            "is_fraud": (combined > self.threshold).astype(int),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FraudEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)
