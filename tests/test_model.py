"""Tests du comportement du modèle et de la journalisation.

Le stockage est redirigé vers un dossier temporaire (tmp_path) via la
variable d'environnement PAD_MONITORING_DIR afin de ne pas polluer
monitoring/data pendant les tests.
"""

import copy
import json
import sqlite3

from api.main import app
from api.model import ServiceModele


def test_modele_charge_une_seule_fois(client, payload_valide):
    """Le modèle n'est chargé qu'une fois : même objet entre requêtes."""
    nb_avant = ServiceModele.nb_chargements
    client.post("/predict", json=payload_valide)
    service_1 = app.state.service_modele
    client.post("/predict", json=payload_valide)
    service_2 = app.state.service_modele

    # Identité d'objet : pas de rechargement entre deux requêtes.
    assert service_1 is service_2
    assert ServiceModele.nb_chargements == nb_avant


def test_features_manquantes_nan(client, payload_valide):
    """Les features manquantes restent NaN : prédiction sans erreur."""
    payload = {cle: payload_valide[cle] for cle in (
        "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "DAYS_BIRTH", "DAYS_EMPLOYED",
    )}
    # Aucune feature engineered fournie : tout le reste sera NaN.
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 200
    assert 0.0 <= reponse.json()["score"] <= 1.0


def test_coherence_seuil_decision(client, payload_valide):
    """score >= seuil <=> 'refuse' (et réciproquement)."""
    reponse = client.post("/predict", json=payload_valide)
    corps = reponse.json()
    attendu = "refuse" if corps["score"] >= corps["seuil"] else "accorde"
    assert corps["decision"] == attendu


def test_log_succes_en_base(client, payload_valide, tmp_path, monkeypatch):
    """Un appel réussi est écrit dans la SQLite (redirigée vers tmp_path)."""
    monkeypatch.setenv("PAD_MONITORING_DIR", str(tmp_path))
    reponse = client.post("/predict", json=payload_valide)
    assert reponse.status_code == 200
    corps = reponse.json()

    with sqlite3.connect(tmp_path / "prod_logs.db") as conn:
        lignes = conn.execute(
            "SELECT request_id, http_status, score, decision, seuil "
            "FROM predictions"
        ).fetchall()

    assert len(lignes) == 1
    request_id, http_status, score, decision, seuil = lignes[0]
    assert request_id == corps["request_id"]
    assert http_status == 200
    assert abs(score - corps["score"]) < 1e-9
    assert decision == corps["decision"]
    assert seuil == corps["seuil"]


def test_log_erreur_validation_en_base(client, payload_valide, tmp_path,
                                       monkeypatch):
    """Une erreur de validation 422 est aussi journalisée (score NULL)."""
    monkeypatch.setenv("PAD_MONITORING_DIR", str(tmp_path))
    payload = copy.deepcopy(payload_valide)
    payload["AMT_CREDIT"] = "abc"
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422

    with sqlite3.connect(tmp_path / "prod_logs.db") as conn:
        lignes = conn.execute(
            "SELECT http_status, score, decision, erreur FROM predictions"
        ).fetchall()

    assert len(lignes) == 1
    http_status, score, decision, erreur = lignes[0]
    assert http_status == 422
    assert score is None
    assert decision is None
    assert erreur  # message d'erreur non vide


def test_miroir_jsonl(client, payload_valide, tmp_path, monkeypatch):
    """Le miroir JSONL contient une ligne par appel, mêmes champs."""
    monkeypatch.setenv("PAD_MONITORING_DIR", str(tmp_path))
    client.post("/predict", json=payload_valide)

    chemin = tmp_path / "api_logs.jsonl"
    assert chemin.exists()
    lignes = [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lignes) == 1
    assert lignes[0]["http_status"] == 200
    assert lignes[0]["score"] is not None
    assert lignes[0]["decision"] in ("accorde", "refuse")


def test_parite_onnx_sklearn(client, payload_valide):
    """Parité ONNX vs sklearn : scores quasi identiques (tolérance 1e-4)
    et même décision au seuil métier sur le payload de référence.

    Le modèle skops n'est chargé que pour ce test de non-régression
    (jamais dans le chemin nominal de l'API quand ONNX est disponible).
    """
    import pandas as pd
    import skops.io

    from api.model import RACINE_PROJET, TRUSTED
    from api.schemas import PredictRequest

    # Score de référence : predict_proba sklearn sur le même payload.
    modele_ref = skops.io.load(
        RACINE_PROJET / "models" / "model.skops", trusted=TRUSTED
    )
    service = app.state.service_modele
    requete = PredictRequest(**payload_valide)
    valeurs = service._dict_valeurs(requete)
    df = pd.DataFrame([valeurs], columns=service.colonnes)
    df = df.apply(pd.to_numeric, errors="coerce")
    score_ref = float(modele_ref.predict_proba(df)[0, 1])

    # Score servi par l'API (moteur actif, ONNX attendu).
    reponse = client.post("/predict", json=payload_valide)
    assert reponse.status_code == 200
    corps = reponse.json()

    assert abs(corps["score"] - score_ref) < 1e-4
    # Même décision au seuil métier.
    decision_ref = "refuse" if score_ref >= corps["seuil"] else "accorde"
    assert corps["decision"] == decision_ref


def test_fallback_skops_fonctionnel(payload_valide, monkeypatch):
    """Le repli skops fonctionne quand model.onnx est simulé absent."""
    import api.model as module_modele

    # Simule l'absence du fichier ONNX (sans toucher au vrai fichier).
    monkeypatch.setattr(module_modele, "CHEMIN_ONNX",
                        module_modele.RACINE_PROJET / "models" / "absent.onnx")

    service = ServiceModele()
    assert service.moteur == "lightgbm_sklearn"

    from api.schemas import PredictRequest
    requete_payload = PredictRequest(**payload_valide)
    score, inference_ms = service.predire(requete_payload)
    assert 0.0 <= score <= 1.0
    assert inference_ms > 0
