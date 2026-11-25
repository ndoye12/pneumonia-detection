"""Recrée les paramètres de prétraitement des modèles tabulaires.

Les notebooks d'entraînement (Colab) normalisaient les données avec scikit-learn,
mais les objets `scaler` n'ont pas été sauvegardés avec les modèles. Ce script
refait exactement les mêmes étapes que les notebooks et écrit les paramètres
dans des fichiers JSON lus par `app.py` (l'application n'a donc pas besoin
de scikit-learn ni de pandas).

Utilisation (nécessite pandas et scikit-learn) :

    python tools/build_preprocessing.py student chemin/vers/student-mat.csv
    python tools/build_preprocessing.py bank "chemin/vers/Copie de banking_dataset.csv"
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_student(csv_path):
    """Reproduit TP2_Student_Performance_Hyperparameter_Optimization.ipynb."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(csv_path, sep=";")
    X = df.drop("G3", axis=1)
    y = df["G3"]

    raw_features = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            raw_features.append({
                "name": col,
                "type": "numeric",
                "min": int(X[col].min()),
                "max": int(X[col].max()),
                "default": int(X[col].median()),
            })
        else:
            raw_features.append({
                "name": col,
                "type": "categorical",
                "choices": sorted(X[col].unique().tolist()),
                "default": X[col].mode()[0],
            })

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Correspondance colonne encodée -> (variable d'origine, modalité)
    categorical = {f["name"] for f in raw_features if f["type"] == "categorical"}
    encoded = []
    for column in X_encoded.columns:
        if column in X.columns:
            encoded.append({"column": column, "source": column})
        else:
            source = next(c for c in categorical if column.startswith(c + "_"))
            encoded.append({
                "column": column,
                "source": source,
                "value": column[len(source) + 1:],
            })

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)

    params = {
        "description": "TP2 - Student Performance (UCI student-mat.csv), cible G3",
        "raw_features": raw_features,
        "encoded": encoded,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    out = MODELS_DIR / "student_preprocessing.json"
    out.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Écrit : {out}")
    return X_test, y_test


BANK_CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "poutcome"]
BANK_FEATURES = ["age", "job", "marital", "education", "default", "housing", "loan",
                 "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                 "previous", "poutcome"]


def build_bank(csv_path):
    """Reproduit DNN_on_Bank_data.ipynb (LabelEncoder + KNNImputer + RobustScaler)."""
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder, RobustScaler

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"y": "souscription"})

    category_defaults = {col: df[col].mode()[0] for col in BANK_CATEGORICAL}

    categories = {}
    for col in BANK_CATEGORICAL + ["souscription"]:
        encoder = LabelEncoder().fit(df[col])
        df[col] = encoder.transform(df[col])
        categories[col] = encoder.classes_.tolist()

    defaults = {col: int(df[col].median()) for col in ["age", "duration", "campaign", "pdays", "previous"]}

    # Valeurs aberrantes (méthode IQR) remplacées par KNNImputer, comme dans le notebook
    for col in ["age", "duration", "campaign"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].astype(float)
        df.loc[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr), col] = np.nan
        df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=df.columns)

    scaler = RobustScaler().fit(df[BANK_FEATURES].values)

    params = {
        "description": "DNN Bank Marketing (LabelEncoder + RobustScaler)",
        "features": BANK_FEATURES,
        "categories": {c: categories[c] for c in BANK_CATEGORICAL},
        "category_defaults": category_defaults,
        "target_classes": categories["souscription"],
        "numeric_defaults": defaults,
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    out = MODELS_DIR / "bank_preprocessing.json"
    out.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Écrit : {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"student", "bank"}:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "student":
        build_student(sys.argv[2])
    else:
        build_bank(sys.argv[2])
