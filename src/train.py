"""
train.py
------------------------------------------------------------------
Entrena y evalúa los modelos, registrando los experimentos en MLflow.
Toda la configuración (características, hiperparámetros, semilla, folds)
se lee de config.yml a través de config/core.py.

Salida: models/model_bundle.joblib (modelos empaquetados para la API)
------------------------------------------------------------------
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.sklearn

from config.core import ROOT, config

CAT = config.model.cat_features
NUM = config.model.num_features
FEATURES = {
    "solo_marcador": config.model.features_benchmark,
    "marcador+dominio": config.model.features_full,
}


def make_pipe(model, cols, scale):
    num_cols = [c for c in cols if c in NUM]
    tr = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]
    if num_cols:
        tr.append(("num", StandardScaler() if scale else "passthrough", num_cols))
    return Pipeline([("prep", ColumnTransformer(tr)), ("clf", model)])


def build_models():
    rs = config.model.random_state
    return {
        "logistica": lambda cols: make_pipe(LogisticRegression(max_iter=2000, C=1.0), cols, True),
        "random_forest": lambda cols: make_pipe(
            RandomForestClassifier(**config.model.random_forest.model_dump(), random_state=rs), cols, False),
        "hist_gboost": lambda cols: make_pipe(
            HistGradientBoostingClassifier(**config.model.hist_gboost.model_dump(), random_state=rs), cols, False),
    }


def evaluate(pipe, X, y, groups, labels):
    proba = cross_val_predict(
        pipe, X, y, cv=GroupKFold(n_splits=config.model.cv_splits),
        groups=groups, method="predict_proba")
    ll = log_loss(y, proba, labels=labels)
    pred = np.array(labels)[proba.argmax(axis=1)]
    acc = accuracy_score(y, pred)
    precision = precision_score(
    y,
    pred,
    average="weighted"
    )
    
    recall = recall_score(
    y,
    pred,
    average="weighted"
    )
    f1 = f1_score(
    y,
    pred,
    average="weighted"
    )
    Y = pd.get_dummies(pd.Categorical(y, categories=labels)).values
    brier = float(np.mean(((proba - Y) ** 2).sum(1)))
    return ll, brier, acc, precision, recall, f1


def main():
    df = pd.read_csv(ROOT / config.app.data_file)
    y = df[config.model.target]
    groups = df[config.model.group_col]
    labels = sorted(y.unique())
    print(f"Datos: {len(df)} filas | clases {labels}\n")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("entretiempo-resultado")

    models = build_models()
    results, fitted = [], {}
    for feat_name, cols in FEATURES.items():
        X = df[cols]
        for model_name, factory in models.items():
            pipe = factory(cols)
            ll, brier, acc, precision, recall, f1 = evaluate(pipe,X,y,groups,labels)
            with mlflow.start_run(run_name=f"{model_name}__{feat_name}"):
                mlflow.log_params({"modelo": model_name, "features": feat_name,
                                   "cv_splits": config.model.cv_splits})
                mlflow.log_metrics({
                    "cv_logloss": ll,
                    "cv_brier": brier,
                    "cv_accuracy": acc,
                    "cv_precision": precision,
                    "cv_recall": recall,
                    "cv_f1": f1
                    })
                pipe.fit(X, y)
                mlflow.sklearn.log_model(pipe, name="model")
            fitted[(feat_name, model_name)] = pipe
            results.append({"features": feat_name, "modelo": model_name,
                            "logloss": ll, "brier": brier, "accuracy": acc})

    res = pd.DataFrame(results).sort_values(["features", "logloss"]).reset_index(drop=True)
    print("=== Resultados (GroupKFold, menor logloss = mejor) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_full = res[res.features == "marcador+dominio"].iloc[0]
    best_bench = res[res.features == "solo_marcador"].iloc[0]
    mfull = fitted[("marcador+dominio", best_full.modelo)]
    mbench = fitted[("solo_marcador", best_bench.modelo)]

    os.makedirs(ROOT / "models", exist_ok=True)
    bundle = {
        "model_full": mfull, "model_bench": mbench, "labels": labels,
        "features_full": config.model.features_full,
        "features_bench": config.model.features_benchmark,
        "meta": {"best_full": best_full.modelo, "best_bench": best_bench.modelo,
                 "deceptive_winrate": config.model.deceptive_winrate,
                 "threshold": config.model.threshold},
    }
    joblib.dump(bundle, ROOT / "models" / "model_bundle.joblib")
    print(f"\nMejor con dominio: {best_full.modelo} (logloss {best_full.logloss:.4f}) | "
          f"solo marcador: {best_bench.modelo} (logloss {best_bench.logloss:.4f})")
    print("OK -> models/model_bundle.joblib")


if __name__ == "__main__":
    main()
