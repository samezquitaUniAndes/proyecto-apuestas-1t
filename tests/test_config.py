"""
test_config.py — la configuración se carga y se valida correctamente.
"""
from config.core import config


def test_config_carga_valores_clave():
    assert config.model.target == "resultado_ft"
    assert config.model.cv_splits == 5
    assert "xg_diff" in config.model.features_full
    assert config.app.data_file.endswith("team_match_dataset.csv")


def test_benchmark_es_subconjunto_del_completo():
    # el benchmark (solo marcador) debe estar contenido en el conjunto completo
    assert set(config.model.features_benchmark).issubset(set(config.model.features_full))


def test_hiperparametros_tienen_tipos_correctos():
    assert isinstance(config.model.random_forest.n_estimators, int)
    assert isinstance(config.model.hist_gboost.learning_rate, float)
    assert 0.0 < config.model.threshold < 1.0
