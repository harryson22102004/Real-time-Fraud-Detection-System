import redis
import json
from typing import Dict, Optional

class FeatureStore:
    def __init__(self, host="localhost", port=6379, ttl=86400):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def get_user_history(self, user_id: str) -> list:
        key = f"history:{user_id}"
        data = self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in data]

    def append_transaction(self, user_id: str, txn: dict):
        key = f"history:{user_id}"
        self.client.lpush(key, json.dumps(txn))
        self.client.ltrim(key, 0, 999)  # Keep last 1000
        self.client.expire(key, self.ttl)

    def cache_features(self, txn_id: str, features: Dict[str, float]):
        self.client.setex(f"features:{txn_id}", 3600, json.dumps(features))
