# 


# Cardio Predict (Heart Disease Prediction)

## Problem description
When a patient shows up with possible hert trouble, doctors check several factors such as age, blood pressure, cholesterol, heart rate during exercise, and ECG results and so on. This project builds a model that takes those measurements and predicts whether the patient is likely to have heart disease.

The model could be used to flag higher-risk patients for further tests. It is not a diagnosis and does not replace a doctor. This is a learning project.

This is a binary classification where 1 means heart disease and 0 means no hear disease.

## Dataset
- **Source:** "Heart Failure Prediction Dataset" by fedesoriano from Kaggle
  [link here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction?resource=download)
  
- **Size:** 918 patients, 11 input columns and 1 target column
- **File:** `data/heart.csv` (original, unmodified)

| Column | Meaning |
|---|---|
| `age` | Age of the patient (years) |
| `sex` | Sex of the patient (`M`: Male, `F`: Female) |
| `chest_pain_type` | Chest pain type (`TA`: Typical Angina, `ATA`: Atypical Angina, `NAP`: Non-Anginal Pain, `ASY`: Asymptomatic) |
| `resting_bp` | Resting blood pressure (mm Hg) |
| `cholesterol` | Serum cholesterol (mm/dl) |
| `fasting_bs` | Fasting blood sugar (`1`: if `fasting_bs` > 120 mg/dl, `0`: otherwise) |
| `resting_ecg` | Resting electrocardiogram results (`Normal`: Normal, `ST`: having ST-T wave abnormality with T wave inversions and/or ST elevation or depression of > 0.05 mV, `LVH`: showing probable or definite left ventricular hypertrophy by Estes' criteria) |
| `max_hr` | Maximum heart rate achieved (Numeric value between 60 and 202) |
| `exercise_angina` | Exercise-induced angina (`Y`: Yes, `N`: No) |
| `oldpeak` | Oldpeak = ST (Numeric value measured in depression) |
| `st_slope` | The slope of the peak exercise ST segment (`Up`: upsloping, `Flat`: flat, `Down`: downsloping) |
| `heart_disease` | Output class (`1`: heart disease, `0`: Normal) |


## EDA summary
## Modeling
## How to run locally
## How to run with Docker
## API usage example
## Cloud deployment (if done)
## Limitations / next steps