import os
import numpy as np
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from flask import request, jsonify
import joblib

MODEL_PATH = '/tmp/wine_model.joblib'

def train_and_save_model():
    """Trains the model and saves it to disk for future invocations."""
    wine = load_wine()
    X = wine.data
    y = wine.target


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump((scaler, model), MODEL_PATH)
    print("Model and scaler trained and saved.")

def load_model():
    """Loads the saved model and scaler."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

def validate_features(features):
    """Validates that the features are a list of 13 numeric values."""
    if not isinstance(features, list) or len(features) != 13:
        return False
    try:
        [float(x) for x in features]
        return True
    except ValueError:
        return False

def logfunction(request):
    """Predicts the Wine cultivar class based on input features."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request content-type must be application/json"}), 400

        request_json = request.get_json()

        if not request_json or "features" not in request_json:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        features = request_json['features']

        if not validate_features(features):
            return jsonify({"error": "Invalid input. 'features' must be a list of 13 numeric values."}), 400


        scaler, model = load_model()

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        predicted_class = model.predict(features_scaled)[0]
        prediction_probabilities = model.predict_proba(features_scaled).tolist()

        return jsonify({
            "predicted_class": int(predicted_class),
            "prediction_confidences": prediction_probabilities
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
