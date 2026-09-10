"""Application FastAPI « Prêt à dépenser » — API de scoring crédit.

Endpoints :
- GET  /health   : état de santé de l'API et du modèle.
- GET  /info     : métadonnées du modèle (version, seuil, features).
- POST /predict  : score de défaut + décision métier (accorde/refuse).

Chaque appel (succès ou erreur gérée) est journalisé dans SQLite et JSONL
(voir api/storage.py). Le modèle est chargé une seule fois au démarrage
via le lifespan.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model import ServiceModele
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from api import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage, initialise le stockage."""
    app.state.service_modele = ServiceModele()
    storage.init_stockage()
    yield


app = FastAPI(
    title="Prêt à dépenser — API de scoring crédit",
    description=(
        "API de scoring du risque de défaut client pour « Prêt à dépenser ».\n\n"
        "Le modèle (LightGBM, 402 features) retourne la probabilité de défaut "
        "et une décision métier basée sur un seuil optimisé en coût "
        "(FN = 10 x FP).\n\n"
        "Décision : `score >= seuil` → **refuse**, sinon **accorde**."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _maintenant_iso() -> str:
    """Horodatage UTC au format ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Healthcheck : l'API répond et le modèle est chargé."""
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(app.state, "service_modele"),
    )


@app.get("/info", tags=["monitoring"])
def info() -> dict:
    """Métadonnées du modèle chargé."""
    service: ServiceModele = app.state.service_modele
    return {
        "nom_modele": "LGBMClassifier",
        "version_api": app.version,
        "moteur": service.moteur,
        "version_moteur": service.version_moteur,
        "seuil": service.seuil,
        "nb_features": service.nb_features,
        "chemin_modele": service.chemin_modele,
    }


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(requete: PredictRequest) -> PredictResponse:
    """Score un dossier client et applique le seuil métier de décision."""
    debut = perf_counter()
    request_id = str(uuid.uuid4())
    service: ServiceModele = app.state.service_modele

    score, inference_ms = service.predire(requete)
    decision = service.decider(score)
    latence_ms = (perf_counter() - debut) * 1000.0

    # Journalisation structurée de l'appel (succès).
    storage.log_appel(
        timestamp_utc=_maintenant_iso(),
        request_id=request_id,
        http_status=200,
        score=round(score, 6),
        decision=decision,
        seuil=service.seuil,
        latence_ms=latence_ms,
        inference_ms=inference_ms,
        inputs=requete.model_dump(),
    )

    return PredictResponse(
        request_id=request_id,
        score=round(score, 6),
        decision=decision,
        seuil=service.seuil,
        inference_ms=round(inference_ms, 3),
        latence_ms=round(latence_ms, 3),
    )


@app.exception_handler(RequestValidationError)
async def handler_validation(request: Request, exc: RequestValidationError):
    """Logue les erreurs de validation (422) comme les appels réussis."""
    request_id = str(uuid.uuid4())
    try:
        corps = await request.json()
    except Exception:
        corps = None
    storage.log_appel(
        timestamp_utc=_maintenant_iso(),
        request_id=request_id,
        http_status=422,
        erreur=str(exc.errors())[:2000],
        inputs=corps,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": _erreurs_serialisables(exc)},
    )


def _erreurs_serialisables(exc: RequestValidationError) -> list[dict]:
    """Convertit les erreurs Pydantic en dicts JSON-sérialisables.

    Le champ `ctx` peut contenir l'instance d'exception d'origine
    (ValueError...), qui n'est pas sérialisable : on la convertit en texte.
    """
    erreurs = []
    for erreur in exc.errors():
        erreur = dict(erreur)
        if "ctx" in erreur:
            erreur["ctx"] = {cle: str(val) for cle, val in erreur["ctx"].items()}
        erreurs.append(erreur)
    return erreurs


@app.exception_handler(Exception)
async def handler_exception_globale(request: Request, exc: Exception):
    """Logue toute erreur serveur non gérée (500)."""
    request_id = str(uuid.uuid4())
    storage.log_appel(
        timestamp_utc=_maintenant_iso(),
        request_id=request_id,
        http_status=500,
        erreur=f"{type(exc).__name__}: {exc}",
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur."},
    )
