# Image de production de l'API de scoring « Prêt à dépenser ».
# Build :  docker build -t pret-a-depenser-api .
# Run :    docker run -d -p 8000:8000 pret-a-depenser-api
FROM python:3.12-slim

# Métadonnées de l'image.
LABEL org.opencontainers.image.title="pret-a-depenser-api" \
      org.opencontainers.image.description="API FastAPI de scoring crédit (LightGBM, 402 features)"

# Python sans buffer (logs en temps réel) et sans .pyc.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dépendances minimales de serving (couche cachée par Docker).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# curl est requis par le HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Code de l'API et artefacts du modèle (ordre des couches : le code change
# plus souvent que le modèle).
COPY models/ ./models/
COPY api/ ./api/

# Le dossier de monitoring est créé au runtime par l'API ; on le déclare
# en volume pour pouvoir persister les logs de production hors conteneur.
VOLUME ["/app/monitoring/data"]

EXPOSE 8000

# Vérification de santé : l'API doit répondre sur /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Serveur de production (1 worker : le modèle est chargé en mémoire une
# fois par worker ; scaler via réplicas plutôt que workers).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
