"""Tests de validation métier de POST /predict (réponses 422 attendues)."""

import copy


def test_champ_requis_manquant(client, payload_valide):
    """Un champ métier obligatoire manquant est rejeté."""
    payload = copy.deepcopy(payload_valide)
    del payload["AMT_CREDIT"]
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_type_incorrect(client, payload_valide):
    """Un type incorrect (texte pour un montant) est rejeté."""
    payload = copy.deepcopy(payload_valide)
    payload["AMT_CREDIT"] = "abc"
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_revenu_nul(client, payload_valide):
    """AMT_INCOME_TOTAL = 0 est impossible (revenu strictement positif)."""
    payload = copy.deepcopy(payload_valide)
    payload["AMT_INCOME_TOTAL"] = 0
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_age_trop_jeune(client, payload_valide):
    """DAYS_BIRTH = -1825 (environ 5 ans) est rejeté."""
    payload = copy.deepcopy(payload_valide)
    payload["DAYS_BIRTH"] = -1825
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_days_birth_positif(client, payload_valide):
    """DAYS_BIRTH positif (date de naissance dans le futur) est rejeté."""
    payload = copy.deepcopy(payload_valide)
    payload["DAYS_BIRTH"] = 1000
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_ext_source_hors_plage(client, payload_valide):
    """EXT_SOURCE_2 = 1.5 dépasse la plage [0, 1]."""
    payload = copy.deepcopy(payload_valide)
    payload["EXT_SOURCE_2"] = 1.5
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_cle_feature_inconnue(client, payload_valide):
    """Une clé inconnue dans `features` est rejetée avec message clair."""
    payload = copy.deepcopy(payload_valide)
    payload["features"]["CLE_INCONNUE_XYZ"] = 1.0
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422
    assert "CLE_INCONNUE_XYZ" in str(reponse.json())


def test_cnt_children_negatif(client, payload_valide):
    """CNT_CHILDREN = -1 est rejeté."""
    payload = copy.deepcopy(payload_valide)
    payload["CNT_CHILDREN"] = -1
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422


def test_days_employed_anomalie_acceptee(client, payload_valide):
    """DAYS_EMPLOYED = 365243 (anomalie documentée) reste accepté."""
    payload = copy.deepcopy(payload_valide)
    payload["DAYS_EMPLOYED"] = 365243
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 200


def test_days_employed_hors_plage(client, payload_valide):
    """DAYS_EMPLOYED = 10 (positif non sentinelle) est rejeté."""
    payload = copy.deepcopy(payload_valide)
    payload["DAYS_EMPLOYED"] = 10
    reponse = client.post("/predict", json=payload)
    assert reponse.status_code == 422
