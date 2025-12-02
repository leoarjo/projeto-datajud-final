# modeling.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# IMPORTS CORRETOS
from src.data_ingestion import load_data, get_features_and_target
from src.data_processing import build_preprocessor

from sklearn.pipeline import Pipeline
import os


def train_and_select_model():
    print("📥 Carregando dados...")
    df = load_data()
    X, y = get_features_and_target(df)

    print("✂️ Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🔧 Criando pré-processador...")

    categorical_features = ["tribunal", "grau", "classe_nome"]
    numeric_features = ["qtd_movimentos"]

    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )


    # ==========================================================
    # MODELOS A SEREM TESTADOS
    # ==========================================================
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    metricas = []

    # ==========================================================
    # TREINAR E AVALIAR MODELOS
    # ==========================================================
    for nome, modelo in modelos.items():
        print(f"🤖 Treinando modelo: {nome}")

        pipe = Pipeline(steps=[
            ("preprocessamento", preprocessor),
            ("modelo", modelo)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        metricas.append({
            "modelo": nome,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        print(f"✔ {nome} — F1-score: {f1:.4f}")

    # ==========================================================
    # GERAR ARQUIVO model_metrics.csv
    # ==========================================================
    metrics_df = pd.DataFrame(metricas)
    os.makedirs("models", exist_ok=True)

    metrics_path = "models/model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"📊 Métricas salvas em: {metrics_path}")
    print(metrics_df)

    # ==========================================================
    # SELECIONAR MELHOR MODELO (USANDO F1)
    # ==========================================================
    melhor_modelo_nome = metrics_df.sort_values("f1_score", ascending=False).iloc[0]["modelo"]
    melhor_modelo = modelos[melhor_modelo_nome]

    print(f"🏆 Melhor modelo: {melhor_modelo_nome}")

    # Treinar novamente o melhor modelo para salvar o pipeline final
    melhor_pipeline = Pipeline(steps=[
        ("preprocessamento", preprocessor),
        ("modelo", melhor_modelo)
    ])

    melhor_pipeline.fit(X_train, y_train)

    # ==========================================================
    # SERIALIZAR O PIPELINE FINAL
    # ==========================================================
    model_path = "models/best_pipeline.joblib"
    joblib.dump(melhor_pipeline, model_path)

    print(f"💾 Pipeline final salvo em: {model_path}")


if __name__ == "__main__":
    train_and_select_model()
