"""
api.py
------------------------------------------------------------------
API (FastAPI) que sirve las inferencias del modelo y aloja el tablero.

Endpoints:
  GET  /          -> el tablero (dashboard/index.html)
  GET  /health    -> estado del servicio
  POST /predict   -> probabilidades V/E/D de los dos modelos + alerta de dominio engañoso

Correr:
  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
  Abrir http://localhost:8000
------------------------------------------------------------------
"""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")
LABELS = BUNDLE["labels"]  # ['D','L','W']

app = FastAPI(title="Prediccion al entretiempo - Copa del Mundo", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class EstadoPrimerTiempo(BaseModel):
    goles_a: int = 0
    goles_b: int = 0
    pos_a: float = 50.0          # posesion del equipo A (%); B = 100 - A
    rem_a: int = 0
    rem_b: int = 0
    arco_a: int = 0
    arco_b: int = 0
    xg_a: float = 0.0
    xg_b: float = 0.0
    fase: str = "Grupos"         # informativo (el modelo actual no lo usa)


def _proba(model, feats, X):
    p = model.predict_proba(X[feats])[0]
    return {c: round(float(p[LABELS.index(c)]), 4) for c in ["W", "D", "L"]}


@app.get("/")
def home():
    return FileResponse(BASE / "dashboard" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok",
            "modelo_con_dominio": BUNDLE["meta"]["best_full"],
            "modelo_solo_marcador": BUNDLE["meta"]["best_bench"]}


@app.post("/predict")
def predict(e: EstadoPrimerTiempo):
    estado = "Lead" if e.goles_a > e.goles_b else ("Trail" if e.goles_a < e.goles_b else "Level")
    feats = {
        "estado_ht": estado,
        "pos_diff": round(2 * e.pos_a - 100, 1),
        "rem_diff": e.rem_a - e.rem_b,
        "arco_diff": e.arco_a - e.arco_b,
        "xg_diff": round(e.xg_a - e.xg_b, 3),
    }
    X = pd.DataFrame([feats])
    dominando = (feats["xg_diff"] > 0.3) or (e.pos_a >= 55 and feats["rem_diff"] >= 2)
    return {
        "con_dominio": _proba(BUNDLE["model_full"], BUNDLE["features_full"], X),
        "solo_marcador": _proba(BUNDLE["model_bench"], BUNDLE["features_bench"], X),
        "features": feats,
        "dominio_enganoso": bool(estado != "Lead" and dominando),
        "deceptive_winrate": BUNDLE["meta"]["deceptive_winrate"],
    }
