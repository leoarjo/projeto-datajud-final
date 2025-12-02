# src/data_ingestion.py
import pandas as pd
import os

DATA_PATH = os.path.join("data", "raw", "datajud_amostra.csv")

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def get_features_and_target(df: pd.DataFrame):
    target_col = "foi_julgado"

    feature_cols = [
        "tribunal",
        "grau",
        "classe_nome",
        "qtd_movimentos"
    ]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    return X, y

if __name__ == "__main__":
    df = load_data()
    X, y = get_features_and_target(df)
    print(df.head())
