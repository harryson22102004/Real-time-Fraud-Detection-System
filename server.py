from flask import Flask, request, jsonify
from src.models.ensemble import FraudEnsemble
from src.features.engineering import FeatureEngineer
from src.features.store import FeatureStore
import numpy as np
import time

app = Flask(__name__)
model = FraudEnsemble.load("./models/fraud_ensemble.pkl")
engineer = FeatureEngineer()
store = FeatureStore()

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    txn = request.json

    # Get user history & compute features
    history_raw = store.get_user_history(txn["user_id"])
    import pandas as pd
    history = pd.DataFrame(history_raw) if history_raw else pd.DataFrame()
    features = engineer.compute_transaction_features(txn, history)

    # Predict
    feature_array = np.array([list(features.values())])
    result = model.predict(feature_array)

    # Store for future velocity calculations
    store.append_transaction(txn["user_id"], txn)
    store.cache_features(txn["transaction_id"], features)

    latency_ms = (time.perf_counter() - start) * 1000
    return jsonify({
        "transaction_id": txn["transaction_id"],
        "fraud_probability": float(result["fraud_probability"][0]),
        "is_fraud": bool(result["is_fraud"][0]),
        "latency_ms": round(latency_ms, 1),
    })

if __name__ == "__main__":
    app.run(port=5000)
