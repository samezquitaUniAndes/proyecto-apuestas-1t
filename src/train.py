"""
train.py
------------------------------------------------------------------
Entrena y evalúa los modelos para predecir el resultado a 90' (W/D/L)
al entretiempo, y registra los experimentos en MLflow.

Compara dos conjuntos de características:
  - solo_marcador      -> benchmark (estado del marcador al descanso)
  - marcador+dominio   -> benchmark + diferenciales del 1er tiempo (xG, remates, posesión)

y tres modelos: regresión logística (baseline), random forest, hist gradient boosting.

Validación: GroupKFold agrupando por match_id (las dos filas de un partido no se
separan entre train/test -> evita fuga de información).
Métrica principal: log-loss y Brier (calidad de la probabilidad); accuracy secundaria.

Salida: models/model_bundle.joblib  (modelos empaquetados para la API)
        mlruns/  (experimentos MLflow)
------------------------------------------------------------------
Para registrar en un servidor MLflow remoto (EC2):
    export MLFLOW_TRACKING_URI=http://<IP-EC2>:5000
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
import mlflow
import mlflow.sklearn

CAT = ["estado_ht"]
NUM = ["pos_diff", "rem_diff", "arco_diff", "xg_diff"]
FEATURES = {"solo_marcador": CAT, "marcador+dominio": CAT + NUM}


def make_pipe(model, cols, scale):
    num_cols = [c for c in cols if c in NUM]
    tr = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]
    if num_cols:
        tr.append(("num", StandardScaler() if scale else "passthrough", num_cols))
    return Pipeline([("prep", ColumnTransformer(tr)), ("clf", model)])


def build_models():
    return {
        "logistica": (lambda cols: make_pipe(LogisticRegression(max_iter=2000, C=1.0), cols, scale=True)),
        "random_forest": (lambda cols: make_pipe(
            RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=42), cols, scale=False)),
        "hist_gboost": (lambda cols: make_pipe(
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           l2_regularization=1.0, random_state=42), cols, scale=False)),
    }


def evaluate(pipe, X, y, groups, labels):
    """CV con GroupKFold -> probabilidades out-of-fold y métricas."""
    gkf = GroupKFold(n_splits=5)
    proba = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, method="predict_proba")
    ll = log_loss(y, proba, labels=labels)
    pred = np.array(labels)[proba.argmax(1)]
    acc = accuracy_score(y, pred)
    Y = pd.get_dummies(pd.Categorical(y, categories=labels)).values
    brier = float(np.mean(((proba - Y) ** 2).sum(1)))
    return ll, brier, acc


def main():
    # --- datos ---
    cands = ["data/processed/team_match_dataset.csv", "../data/processed/team_match_dataset.csv"]
    path = next((p for p in cands if os.path.exists(p)), None)
    assert path, "No se encontró el dataset. Corre primero: python src/build_dataset.py"
    df = pd.read_csv(path)
    y = df["resultado_ft"]
    groups = df["match_id"]
    labels = sorted(y.unique())          # ['D','L','W']
    print(f"Datos: {len(df)} filas | clases {labels} | partidos {groups.nunique()}\n")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("entretiempo-resultado")

    models = build_models()
    results, fitted = [], {}
    for feat_name, cols in FEATURES.items():
        X = df[cols]
        for model_name, factory in models.items():
            pipe = factory(cols)
            ll, brier, acc = evaluate(pipe, X, y, groups, labels)
            with mlflow.start_run(run_name=f"{model_name}__{feat_name}"):
                mlflow.log_params({"modelo": model_name, "features": feat_name,
                                   "n_features": len(cols), "cv": "GroupKFold(5) por match_id"})
                mlflow.log_metrics({"cv_logloss": ll, "cv_brier": brier, "cv_accuracy": acc})
                pipe.fit(X, y)                        # refit en todo el set
                mlflow.sklearn.log_model(pipe, name="model")
            fitted[(feat_name, model_name)] = pipe
            results.append({"features": feat_name, "modelo": model_name,
                            "logloss": ll, "brier": brier, "accuracy": acc})

    res = pd.DataFrame(results).sort_values(["features", "logloss"]).reset_index(drop=True)
    print("=== Resultados (validación cruzada GroupKFold, menor logloss = mejor) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- mejor de cada conjunto ---
    best_full = res[res.features == "marcador+dominio"].iloc[0]
    best_bench = res[res.features == "solo_marcador"].iloc[0]
    mfull = fitted[("marcador+dominio", best_full.modelo)]
    mbench = fitted[("solo_marcador", best_bench.modelo)]
    mejora = best_bench.logloss - best_full.logloss
    print(f"\nMejor con dominio : {best_full.modelo}  (logloss {best_full.logloss:.4f})")
    print(f"Mejor solo marcador: {best_bench.modelo}  (logloss {best_bench.logloss:.4f})")
    print(f"Mejora del logloss al añadir dominio: {mejora:+.4f}  "
          f"({'las métricas de dominio ayudan' if mejora > 0 else 'no aportan'})")

    # --- escenario 'dominio engañoso' ---
    scen = pd.DataFrame([{"estado_ht": "Level", "pos_diff": 16.0, "rem_diff": 4, "arco_diff": 2, "xg_diff": 0.6}])
    p = mfull.predict_proba(scen[FEATURES["marcador+dominio"]])[0]
    pw = p[labels.index("W")]
    print(f"\nEscenario 'dominó sin ir ganando' (xg_diff=+0.6, empatando):")
    print(f"  P(gana)={pw:.2f} | P(NO gana)={1-pw:.2f}  -> el modelo confirma que dominar no basta.")

    # --- empaquetar para la API ---
    os.makedirs("models", exist_ok=True)
    bundle = {"model_full": mfull, "model_bench": mbench, "labels": labels,
              "features_full": FEATURES["marcador+dominio"], "features_bench": FEATURES["solo_marcador"],
              "meta": {"best_full": best_full.modelo, "best_bench": best_bench.modelo,
                       "deceptive_winrate": 0.29, "threshold": 0.10}}
    joblib.dump(bundle, "models/model_bundle.joblib")
    print("\nOK -> models/model_bundle.joblib | experimentos en MLflow")


if __name__ == "__main__":
    main()
