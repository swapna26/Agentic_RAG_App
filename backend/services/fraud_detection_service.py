"""
Fraud Detection Service
Loads a trained RandomForest model and provides fraud prediction
for credit card transactions. Logs every prediction to PostgreSQL
for drift monitoring.
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
import psycopg2

logger = logging.getLogger(__name__)

# 30 features the model expects (same order as training data)
FEATURE_NAMES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount_scaled", "Time_scaled",
]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ml_models", "fraud_model.joblib"
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fraud_prediction_log (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version VARCHAR(20) NOT NULL,
    features JSONB NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    fraud_probability FLOAT NOT NULL
);
"""


class FraudDetectionService:
    """Loads the fraud detection model and serves predictions."""

    def __init__(self, database_url: str = None):
        self._model = None
        self._model_version = "v1"
        self._database_url = database_url

    async def initialize(self):
        """Load the model from disk and create prediction log table."""
        try:
            self._model = joblib.load(MODEL_PATH)
            logger.info(
                f"Fraud detection model loaded: {type(self._model).__name__}"
            )
        except Exception as e:
            logger.error(f"Failed to load fraud detection model: {e}")
            raise

        # Create prediction log table if DB is configured
        if self._database_url:
            try:
                conn = psycopg2.connect(self._database_url)
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(CREATE_TABLE_SQL)
                conn.close()
                logger.info("Fraud prediction log table ready")
            except Exception as e:
                logger.warning(f"Could not create prediction log table: {e}")

    def predict(self, features: dict) -> dict:
        """
        Predict whether a transaction is fraudulent.

        Args:
            features: dict with keys V1-V28, Amount_scaled, Time_scaled

        Returns:
            dict with is_fraud, fraud_probability, model_version
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Build feature array in correct order
        values = []
        for name in FEATURE_NAMES:
            val = features.get(name)
            if val is None:
                raise ValueError(f"Missing feature: {name}")
            values.append(float(val))

        X = np.array([values])
        prediction = self._model.predict(X)[0]
        probability = self._model.predict_proba(X)[0][1]

        result = {
            "is_fraud": bool(prediction == 1),
            "fraud_probability": round(float(probability), 4),
            "model_version": self._model_version,
        }

        # Log prediction to PostgreSQL
        self._log_prediction(features, result)

        return result

    def _log_prediction(self, features: dict, result: dict):
        """Store prediction in fraud_prediction_log table for drift monitoring."""
        if not self._database_url:
            return
        try:
            conn = psycopg2.connect(self._database_url)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO fraud_prediction_log
                           (id, timestamp, model_version, features, is_fraud, fraud_probability)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            str(uuid.uuid4()),
                            datetime.now(timezone.utc),
                            result["model_version"],
                            json.dumps(features),
                            result["is_fraud"],
                            result["fraud_probability"],
                        ),
                    )
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

    def get_prediction_stats(self) -> dict:
        """Get prediction statistics for drift monitoring."""
        if not self._database_url:
            return {"error": "Database not configured"}
        try:
            conn = psycopg2.connect(self._database_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) AS total_predictions,
                        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_count,
                        AVG(fraud_probability) AS avg_probability,
                        MIN(timestamp) AS first_prediction,
                        MAX(timestamp) AS last_prediction
                    FROM fraud_prediction_log
                """)
                row = cur.fetchone()
            conn.close()
            return {
                "total_predictions": row[0],
                "fraud_count": row[1],
                "fraud_rate": round(row[1] / row[0], 4) if row[0] > 0 else 0,
                "avg_fraud_probability": round(float(row[2]), 4) if row[2] else 0,
                "first_prediction": str(row[3]) if row[3] else None,
                "last_prediction": str(row[4]) if row[4] else None,
            }
        except Exception as e:
            logger.warning(f"Failed to get prediction stats: {e}")
            return {"error": str(e)}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def cleanup(self):
        self._model = None
