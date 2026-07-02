"""
config/core.py
------------------------------------------------------------------
Configuración del proyecto, validada con pydantic.

Sigue la metodología de empaquetamiento vista en clase: los hiperparámetros
y las características del modelo viven en config.yml (no incrustados en el
código) y se validan por tipo al cargar la aplicación.
------------------------------------------------------------------
"""
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

# Ubicación del proyecto
PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # -> src/
ROOT = PACKAGE_ROOT.parent                              # -> raíz del repositorio
CONFIG_FILE_PATH = ROOT / "config.yml"


class AppConfig(BaseModel):
    """Configuración a nivel de aplicación."""
    package_name: str
    data_file: str


class RFParams(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_leaf: int


class HGBParams(BaseModel):
    max_depth: int
    learning_rate: float
    max_iter: int
    l2_regularization: float


class ModelConfig(BaseModel):
    """Configuración de entrenamiento y de características."""
    target: str
    group_col: str
    cat_features: List[str]
    num_features: List[str]
    features_benchmark: List[str]
    features_full: List[str]
    cv_splits: int
    random_state: int
    random_forest: RFParams
    hist_gboost: HGBParams
    deceptive_winrate: float
    threshold: float


class Config(BaseModel):
    """Objeto de configuración maestro."""
    app: AppConfig
    model: ModelConfig


def find_config_file() -> Path:
    """Ubica el archivo de configuración."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"config.yml no encontrado en {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> dict:
    """Lee y parsea el YAML de configuración."""
    cfg_path = cfg_path or find_config_file()
    with open(cfg_path, "r") as conf_file:
        return yaml.safe_load(conf_file)


def create_and_validate_config(parsed: dict = None) -> Config:
    """Valida los valores de configuración contra los tipos definidos."""
    parsed = parsed if parsed is not None else fetch_config_from_yaml()
    # pydantic ignora campos extra: cada sub-config toma solo los suyos
    return Config(app=AppConfig(**parsed), model=ModelConfig(**parsed))


config = create_and_validate_config()
