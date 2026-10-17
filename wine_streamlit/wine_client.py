"""Helpers for calling the wine classifier: deployed Cloud Function or local model."""
import numpy as np
import requests
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

WINE = load_wine()
FEATURE_NAMES = list(WINE.feature_names)
CULTIVARS = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]

# One real sample of each class from the dataset, used as presets.
PRESETS = {
    "Sample of cultivar 1": WINE.data[0].tolist(),
    "Sample of cultivar 2": WINE.data[70].tolist(),
    "Sample of cultivar 3": WINE.data[150].tolist(),
}


def feature_ranges():
    """Min and max of each feature in the dataset, for input bounds."""
    return {
        name: (float(WINE.data[:, i].min()), float(WINE.data[:, i].max()))
        for i, name in enumerate(FEATURE_NAMES)
    }


def predict_cloud(url, features, token=None, timeout=60):
    """Sends features to the Cloud Function in the lab's request format.

    Returns (status_code, response_json).
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    resp = requests.post(url.strip(), json={"features": features},
                         headers=headers, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text[:500] or f"HTTP {resp.status_code}"}
    return resp.status_code, body


def train_local_model():
    """Same training as ML-Example/main.py, for running without GCP."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(WINE.data)
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    model.fit(X_scaled, WINE.target)
    return scaler, model


def predict_local(scaler, model, features):
    """Returns a response shaped exactly like the Cloud Function's."""
    X = scaler.transform(np.array(features, dtype=float).reshape(1, -1))
    return {
        "predicted_class": int(model.predict(X)[0]),
        "prediction_confidences": model.predict_proba(X).tolist(),
    }
