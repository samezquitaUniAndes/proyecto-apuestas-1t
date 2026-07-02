"""
conftest.py — fixtures compartidas por las pruebas.
"""
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def bundle():
    """Modelos empaquetados que sirve la API."""
    return joblib.load(ROOT / "models" / "model_bundle.joblib")


@pytest.fixture()
def sample_input():
    """Escenario de 'dominio engañoso': domina el xG pero no va ganando."""
    return {
        "goles_a": 0, "goles_b": 0, "pos_a": 58,
        "rem_a": 7, "rem_b": 2, "arco_a": 4, "arco_b": 1,
        "xg_a": 1.1, "xg_b": 0.3,
    }
