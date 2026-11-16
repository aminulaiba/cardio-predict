import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "heart.csv"
OUTPUT_FILE = "model.bin"
C = 1.0
THRESHOLD = 0.35

numerical = ["age", "resting_bp", "cholesterol", "max_hr", "oldpeak", "fasting_bs"]
categorical = ["sex", "chest_pain_type", "resting_ecg", "exercise_angina", "st_slope"]
features = numerical + categorical
fill_cols = ["resting_bp", "cholesterol"]


def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.replace(r"(?<=[a-z])(?=[A-Z])", "_", regex=True).str.lower()
    )
    # 0 is impossible for these two columns, so it means "missing"
    df[fill_cols] = df[fill_cols].replace(0, np.nan)
    return df


def train(df, y, model):
    medians = df[fill_cols].median()  # learned from training rows only
    dicts = df[features].fillna(medians).to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    model.fit(X, y)
    return medians, dv, model


def predict(df, medians, dv, model):
    dicts = df[features].fillna(medians).to_dict(orient="records")
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]


def main():
    df = load_data(DATA_PATH)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    y_full_train = df_full_train.heart_disease.values
    y_test = df_test.heart_disease.values

    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    medians, dv, model = train(df_full_train, y_full_train, model)

    proba = predict(df_test, medians, dv, model)
    pred = proba >= THRESHOLD
    print(f"Test AUC:       {roc_auc_score(y_test, proba):.3f}")
    print(f"Test accuracy:  {accuracy_score(y_test, pred):.3f}")
    print(f"Test recall:    {recall_score(y_test, pred):.3f}")
    print(f"Test precision: {precision_score(y_test, pred):.3f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((medians, dv, model), f_out)
    print(f"Model saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()