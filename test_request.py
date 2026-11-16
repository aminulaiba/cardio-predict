import json
import urllib.request

patient = {
    "age": 70,
    "sex": "M",
    "chest_pain_type": "ASY",
    "resting_bp": 170,
    "cholesterol": 192,
    "fasting_bs": 0,
    "resting_ecg": "ST",
    "max_hr": 129,
    "exercise_angina": "Y",
    "oldpeak": 3.0,
    "st_slope": "Down",
}

req = urllib.request.Request(
    "http://localhost:9696/predict",
    data=json.dumps(patient).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())