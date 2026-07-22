# Tablero + API de predicción al entretiempo (Entrega 2)
FROM python:3.12-slim

WORKDIR /app

# Solo las dependencias necesarias para SERVIR (imagen liviana)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código y artefactos del modelo
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY models/ ./models/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
