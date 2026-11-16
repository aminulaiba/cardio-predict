import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

THRESHOLD = 0.35
MODEL_FILE = "model.bin"

numerical = ["age", "resting_bp", "cholesterol", "max_hr", "oldpeak", "fasting_bs"]
categorical = ["sex", "chest_pain_type", "resting_ecg", "exercise_angina", "st_slope"]
features = numerical + categorical
fill_cols = ["resting_bp", "cholesterol"] 
required = [f for f in features if f not in fill_cols]

with open(MODEL_FILE, "rb") as f_in:
    medians, dv, model = pickle.load(f_in)

app = Flask("heart-disease")


def get_probability(patient):
    df = pd.DataFrame([patient]).reindex(columns=features)
    # same cleaning as in training: 0 means missing
    df[fill_cols] = df[fill_cols].astype(float).replace(0, np.nan)
    dicts = df[features].fillna(medians).to_dict(orient="records")
    X = dv.transform(dicts)
    return float(model.predict_proba(X)[0, 1])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json(silent=True)
    if not isinstance(patient, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    missing = [f for f in required if patient.get(f) is None]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    proba = get_probability(patient)
    return jsonify(
        {
            "heart_disease_probability": round(proba, 3),
            "heart_disease": bool(proba >= THRESHOLD),
            "threshold": THRESHOLD,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)