"""Tests des endpoints /health, /info et /predict (cas nominal)."""


def test_health(client):
    """GET /health retourne 200 et confirme que le modèle est chargé."""
    reponse = client.get("/health")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] == "ok"
    assert corps["model_loaded"] is True


def test_info(client):
    """GET /info expose les métadonnées du modèle et le moteur actif."""
    reponse = client.get("/info")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["seuil"] == 0.45
    assert corps["nb_features"] == 402
    assert corps["moteur"] in ("onnx", "lightgbm_sklearn")
    assert corps["version_moteur"]  # version non vide
    assert "chemin_modele" in corps
    assert corps["chemin_modele"].endswith(("model.onnx", "model.skops"))


def test_predict_valide(client, payload_valide):
    """POST /predict avec un payload valide retourne un score cohérent."""
    reponse = client.post("/predict", json=payload_valide)
    assert reponse.status_code == 200
    corps = reponse.json()

    # Structure de la réponse.
    for champ in ("request_id", "score", "decision", "seuil",
                  "inference_ms", "latence_ms"):
        assert champ in corps

    # Le score est une probabilité.
    assert 0.0 <= corps["score"] <= 1.0

    # Cohérence seuil / décision : score >= seuil <=> refuse.
    if corps["score"] >= corps["seuil"]:
        assert corps["decision"] == "refuse"
    else:
        assert corps["decision"] == "accorde"

    # Latences positives.
    assert corps["inference_ms"] > 0
    assert corps["latence_ms"] >= corps["inference_ms"]
