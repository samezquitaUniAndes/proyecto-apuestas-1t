"""
test_prediction.py — el modelo empaquetado y la API predicen correctamente.
"""
from api import EstadoPrimerTiempo, predict


def test_bundle_tiene_ambos_modelos(bundle):
    assert "model_full" in bundle
    assert "model_bench" in bundle
    assert bundle["labels"] == ["D", "L", "W"]


def test_probabilidades_suman_uno(sample_input):
    # Given / When
    r = predict(EstadoPrimerTiempo(**sample_input))
    # Then
    p = r["con_dominio"]
    assert abs(p["W"] + p["D"] + p["L"] - 1.0) < 0.01
    assert all(0.0 <= v <= 1.0 for v in p.values())


def test_transformacion_de_features(sample_input):
    r = predict(EstadoPrimerTiempo(**sample_input))
    f = r["features"]
    # 0-0 -> empatando ; posesión 58% -> pos_diff = +16
    assert f["estado_ht"] == "Level"
    assert f["pos_diff"] == 16.0
    assert f["xg_diff"] == 0.8  # 1.1 - 0.3


def test_alerta_dominio_enganoso(sample_input):
    r = predict(EstadoPrimerTiempo(**sample_input))
    assert r["dominio_enganoso"] is True


def test_va_ganando_detecta_lead():
    r = predict(EstadoPrimerTiempo(goles_a=1, goles_b=0))
    assert r["features"]["estado_ht"] == "Lead"
    assert r["dominio_enganoso"] is False
