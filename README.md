## Cardio Predict (Heart Disease Prediction)

## Problem description
When a patient shows up with possible heart trouble, doctors check several factors such as age, blood pressure, cholesterol, heart rate during exercise, and ECG results and so on. This project builds a model that takes those measurements and predicts whether the patient is likely to have heart disease.

The model could be used to flag higher-risk patients for further tests. It is not a diagnosis and does not replace a doctor. This is a learning project.

This is a binary classification where 1 means heart disease and 0 means no heart disease.

## Dataset
- **Source:** "Heart Failure Prediction Dataset" by fedesoriano from Kaggle
  [link here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download)
  
- **Size:** 918 patients, 11 input columns and 1 target column
- **File:** `data/heart.csv` (original, unmodified)

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | `M` or `F` |
| `chest_pain_type` | `TA` typical angina, `ATA` atypical angina, `NAP` non-anginal pain, `ASY` no symptoms |
| `resting_bp` | Blood pressure at rest (mm Hg) |
| `cholesterol` | Serum cholesterol (mg/dl); 0 in the raw file means missing |
| `fasting_bs` | 1 if fasting blood sugar is above 120 mg/dl, otherwise 0 |
| `resting_ecg` | `Normal`, `ST` (ST-T wave abnormality) or `LVH` (probable left ventricular hypertrophy) |
| `max_hr` | Highest heart rate reached during exercise |
| `exercise_angina` | Chest pain during exercise (`Y`/`N`) |
| `oldpeak` | ST depression during exercise relative to rest |
| `st_slope` | Slope of the ST segment at peak exercise: `Up`, `Flat`, `Down` |
| `heart_disease` | Target: 1 = heart disease, 0 = none |



## EDA summary
- **Data:** 918 patients, 11 features. 172 cholesterol and 1 resting BP values were 0 which is not possible, so fill them with the training-set median.
- **Target:** 55% heart disease, 45% healthy, so fairly balanced.
- **Numeric features:** patients with heart disease have lower max heart rate (128 vs 148), higher oldpeak (1.27 vs 0.41) and are older (56 vs 51). Cholesterol and oldpeak are right-skewed with outliers.
- **Categorical features:** exercise angina, chest pain type, ST slope and fasting blood sugar show the biggest differences in disease rate. Resting ECG shows little.
- **Correlation and importance:** ST slope, exercise angina, oldpeak and max heart rate rank highest in both correlation and mutual information, while resting ECG and resting BP rank lowest.

## Modeling
I trained three models: logistic regression (linear), a decision tree and a random forest (tree-based). Missing values (zeros in cholesterol and resting BP) are filled with the training-set median and categorical features are one-hot encoded with DictVectorizer. Settings were tuned by trying several values and comparing AUC on a validation set.

**Metric:** the classes are fairly balanced (55% disease), so I use ROC AUC as the main metric and also report accuracy, recall and precision. Recall matters because missing a sick patient is worse than a false alarm.


| Model | Settings tried | Best | Validation AUC |
|---|---|---|---|
| Logistic regression | C = 0.01, 0.1, 1, 10 | C=1 | 0.903 |
| Decision tree | max_depth 2–8 | max_depth=5 | 0.862 |
| Random forest | max_depth 4/6/10, 50/100/200 trees | depth 10, 100 trees | 0.905 |

**Final model:** logistic regression (C=1). It performs about the same as the random forest but is simpler.

Test-set result: AUC 0.954, accuracy 0.924, recall 0.945, precision 0.929.

The service uses a threshold of 0.35 because missing a sick patient is worse than a false alarm. At 0.35 the test set gives accuracy 0.908, recall 0.955, precision 0.897. At 0.5 the test set gives AUC 0.954, accuracy 0.924, recall 0.945, precision 0.929.


## How to run locally
Requires [uv](https://docs.astral.sh/uv/).
1. Install dependencies and create the virtual environment (`.venv`): `uv sync`
2. Activate it (optional, since `uv run` does it for you): Linux/Mac `source .venv/bin/activate`, Windows `.venv\Scripts\activate`
3. Train the model: `uv run python train.py` (creates `model.bin`)
4. Start the service: `uv run python predict.py` (port 9696)


## How to run with Docker
`docker build -t heart-disease .`
`docker run -it --rm -p 9696:9696 heart-disease`


## API usage example
The service listens on port 9696 and has two endpoints:
- `GET /health` returns `{"status": "ok"}`
- `POST /predict` takes one patient as JSON and returns the predicted probability of heart disease

`resting_bp` and `cholesterol` are optional. If they are left out, `null` or 0, they are filled with the training median. All other fields are required, otherwise the service answers with a 400 error that lists the missing fields.

**Example request** (curl on Linux, Mac or Git Bash):
```
curl -X POST -H "Content-Type: application/json" \
  -d '{"age":70,"sex":"M","chest_pain_type":"ASY","resting_bp":170,"cholesterol":192,"fasting_bs":0,"resting_ecg":"ST","max_hr":129,"exercise_angina":"Y","oldpeak":3.0,"st_slope":"Down"}' \
  http://localhost:9696/predict
```

**Example response:**
```
{"heart_disease": true, "heart_disease_probability": 0.924, "threshold": 0.35}
```
`heart_disease` is true when the probability is at least the threshold (0.35).

**With Python** (works on every system, including Windows): `uv run python test_request.py`


## Limitations / next steps
- Small dataset (918 patients) about 79% are men, so results for women are less reliable.
- Zero cholesterol values are treated as missing, this is an assumption. Missing cholesterol may itself carry information, which I did not model.
- The test set has only 184 patients, so scores are rough, and the threshold was chosen on one validation split.
- Not clinically validated and not medical advice.
- Next steps: cross-validation, XGBoost, cloud deployment.