"""Fixtures pytest pour l'API « Prêt à dépenser ».

- `client` : TestClient FastAPI avec lifespan déclenché (modèle chargé).
- `payload_valide` : requête valide construite à partir de la première
  ligne de `artifacts/serving_input.json` (champs métier + features).
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Rend la racine du projet importable quel que soit le cwd de pytest.
RACINE_PROJET = Path(__file__).resolve().parent.parent
if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))

from api.main import app  # noqa: E402

# Champs métier portés en première classe par la requête.
CHAMPS_METIER = (
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)

# Champs métier typés int dans le schéma Pydantic.
CHAMPS_INT = ("CNT_CHILDREN", "DAYS_BIRTH", "DAYS_EMPLOYED")


@pytest.fixture(scope="session")
def client():
    """TestClient avec lifespan déclenché (chargement du modèle)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def payload_valide() -> dict:
    """Payload client valide, issu de la 1re ligne de serving_input.json."""
    chemin = RACINE_PROJET / "artifacts" / "serving_input.json"
    with open(chemin, encoding="utf-8") as f:
        donnees = json.load(f)["dataframe_split"]
    colonnes = donnees["columns"]
    ligne = dict(zip(colonnes, donnees["data"][0]))

    payload: dict = {}
    features: dict = {}
    for cle, val in ligne.items():
        if val is None:
            continue  # les features absentes restent NaN côté modèle
        if cle in CHAMPS_METIER:
            # Caste les champs int (le JSON les donne parfois en float).
            payload[cle] = int(val) if cle in CHAMPS_INT else float(val)
        else:
            features[cle] = float(val)
    payload["features"] = features
    return payload
