# Predicción del resultado al entretiempo — ¿dominio real o engañoso? (MIIA · Proyecto)

Modelo y tablero que estiman el resultado final (1X2) de un partido de Copa del Mundo
a partir del estado del primer tiempo, y señalan situaciones de "dominio engañoso".

## Datos
StatsBomb Open Data (eventos) — Mundial 2022, Euro 2024, Euro 2020 y Copa América 2024
(198 partidos, 396 filas equipo-partido). Versionados con DVC.

## Reproducir
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/build_dataset.py     # -> data/processed/team_match_dataset.csv
python src/eda.py               # -> reports/figures/*.png
```

## Estructura
- `src/build_dataset.py` — descarga StatsBomb y deriva métricas del 1er tiempo.
- `src/eda.py` — exploración y figuras del reporte.
- `data/processed/` — dataset (gestionado por DVC).
- `reports/figures/` — figuras.
