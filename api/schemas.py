"""Schémas Pydantic v2 de l'API « Prêt à dépenser ».

Ce module définit le contrat d'entrée de POST /predict :
- les champs métier obligatoires, avec validation de plage métier ;
- un dictionnaire optionnel `features` pour toutes les autres features
  engineered, dont les clés inconnues sont rejetées (422).
"""

from pathlib import Path
import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Racine du projet (dossier parent du package api/), robuste au cwd.
RACINE_PROJET = Path(__file__).resolve().parent.parent

# Colonnes connues du modèle (ordre exact attendu par LightGBM).
with open(RACINE_PROJET / "models" / "features.json", encoding="utf-8") as _f:
    COLONNES_CONNUES = set(json.load(_f)["columns"])

# Constante métier : anomalie documentée du dataset Home Credit
# (valeur sentinelle signifiant « non renseigné »).
DAYS_EMPLOYED_ANOMALIE = 365243


class PredictRequest(BaseModel):
    """Requête de scoring crédit pour un client."""

    # --- Champs métier obligatoires ---
    AMT_INCOME_TOTAL: float = Field(
        gt=0, le=100_000_000,
        description="Revenu annuel total du client (strictement positif).",
    )
    AMT_CREDIT: float = Field(
        gt=0, le=100_000_000,
        description="Montant du crédit demandé (strictement positif).",
    )
    AMT_ANNUITY: Optional[float] = Field(
        default=None, ge=0, le=10_000_000,
        description="Annuité du crédit (optionnelle).",
    )
    DAYS_BIRTH: int = Field(
        ge=-30000, le=-6575,
        description="Âge du client en jours négatifs (18 à 82 ans).",
    )
    DAYS_EMPLOYED: int = Field(
        description="Ancienneté d'emploi en jours négatifs, "
                    "ou 365243 (anomalie documentée = non renseigné).",
    )
    CNT_CHILDREN: int = Field(
        ge=0, le=20,
        description="Nombre d'enfants du client.",
    )
    EXT_SOURCE_1: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Score externe 1, normalisé entre 0 et 1.",
    )
    EXT_SOURCE_2: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Score externe 2, normalisé entre 0 et 1.",
    )
    EXT_SOURCE_3: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Score externe 3, normalisé entre 0 et 1.",
    )

    # --- Features engineered supplémentaires (optionnelles) ---
    features: Optional[dict[str, float]] = Field(
        default=None,
        description="Autres features du modèle (clés = noms exacts de "
                    "models/features.json). Les features absentes restent NaN.",
    )

    @field_validator("DAYS_EMPLOYED")
    @classmethod
    def _valider_days_employed(cls, v: int) -> int:
        """Accepte -25000 <= v <= 0 ou l'anomalie documentée 365243."""
        if v == DAYS_EMPLOYED_ANOMALIE:
            return v
        if -25000 <= v <= 0:
            return v
        raise ValueError(
            "DAYS_EMPLOYED doit être compris entre -25000 et 0, "
            f"ou valoir {DAYS_EMPLOYED_ANOMALIE} (anomalie documentée)."
        )

    @field_validator("features")
    @classmethod
    def _valider_cles_features(
        cls, v: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        """Rejette toute clé inconnue du schéma du modèle (→ 422)."""
        if v is None:
            return v
        inconnues = sorted(set(v) - COLONNES_CONNUES)
        if inconnues:
            raise ValueError(
                f"Clés de features inconnues : {inconnues}. "
                "Les clés autorisées sont celles de models/features.json."
            )
        return v


class PredictResponse(BaseModel):
    """Réponse de scoring crédit."""

    request_id: str
    score: float = Field(description="Probabilité de défaut (6 décimales).")
    decision: str = Field(description="'accorde' ou 'refuse'.")
    seuil: float = Field(description="Seuil métier de décision.")
    inference_ms: float = Field(description="Temps de predict_proba seul (ms).")
    latence_ms: float = Field(description="Temps total de l'endpoint (ms).")


class HealthResponse(BaseModel):
    """Réponse du healthcheck."""

    status: str
    model_loaded: bool
