# src/modeling.py
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_ingestion import load_data, get_features_and_target
from src.data_processing import build_preprocessor


def train_and_select_model():
    df = load_data()
    X, y = get_features_and_target(df)

    categorical_features = ["tribunal", "grau", "classe_nome"]
    numeric_features = ["qtd_movimentos"]

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "log_reg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
    }

    best_model_name = None
    best_f1 = -1
    best_pipeline = None

    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds)

        print("\n===== Modelo:", name, "=====")
        print(classification_report(y_test, preds))
        print("F1:", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipe

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_pipeline.joblib")
    print("\nMelhor modelo salvo como best_pipeline.joblib")


if __name__ == "__main__":
    train_and_select_model()
