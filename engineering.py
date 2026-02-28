import pandas as pd
import numpy as np
from typing import Dict, Any

class FeatureEngineer:
    """Computes real-time and historical features for fraud detection."""

    VELOCITY_WINDOWS = [1, 6, 24]  # hours

    def compute_transaction_features(
        self, txn: Dict[str, Any], history: pd.DataFrame
    ) -> Dict[str, float]:
        features = {}
        amount = txn["amount"]
        merchant = txn["merchant_category"]
        
        # Basic features
        features["amount"] = amount
        features["hour_of_day"] = pd.Timestamp(txn["timestamp"]).hour
        features["day_of_week"] = pd.Timestamp(txn["timestamp"]).dayofweek
        features["is_weekend"] = int(features["day_of_week"] >= 5)
        features["is_night"] = int(features["hour_of_day"] < 6 or features["hour_of_day"] > 22)

        # Velocity features
        for window in self.VELOCITY_WINDOWS:
            cutoff = pd.Timestamp(txn["timestamp"]) - pd.Timedelta(hours=window)
            recent = history[history["timestamp"] > cutoff]
            features[f"txn_count_{window}h"] = len(recent)
            features[f"txn_sum_{window}h"] = recent["amount"].sum()
            features[f"txn_mean_{window}h"] = recent["amount"].mean() if len(recent) > 0 else 0
            features[f"txn_std_{window}h"] = recent["amount"].std() if len(recent) > 1 else 0

        # Deviation features
        avg_amount = history["amount"].mean() if len(history) > 0 else amount
        std_amount = history["amount"].std() if len(history) > 1 else 1
        features["amount_zscore"] = (amount - avg_amount) / max(std_amount, 1e-6)
        features["amount_ratio_to_avg"] = amount / max(avg_amount, 1e-6)

        # Merchant features
        merchant_history = history[history["merchant_category"] == merchant]
        features["merchant_txn_count"] = len(merchant_history)
        features["is_new_merchant"] = int(len(merchant_history) == 0)

        return features
