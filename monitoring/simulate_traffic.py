"""Simulation de trafic de production contre l'API « Prêt à dépenser ».

Étape 3 (partie 2) de la consigne : alimenter le stockage de données de
production (SQLite + JSONL, écrit par l'API elle-même via api/storage.py).

Fonctionnement :
1. Échantillonne N lignes de data/processed/test.parquet (random_state=42).
2. Pour chaque ligne, construit le payload POST /predict :
   - champs métier obligatoires (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
     DAYS_BIRTH, DAYS_EMPLOYED, CNT_CHILDREN, EXT_SOURCE_1/2/3) ;
   - toutes les autres features non-NaN dans le dict `features`.
3. Structure la simulation en 2 périodes pour créer une DÉRIVE OBSERVABLE :
   - Période 1 (les ~60 % premiers) : clients « normaux », tels quels.
   - Période 2 (les ~40 % restants) : DÉRIVE ARTIFICIELLE documentée :
       * AMT_CREDIT       × 1.40  (crédits plus élevés)
       * AMT_INCOME_TOTAL × 0.85  (revenus plus faibles)
       * EXT_SOURCE_2     - 0.08  (plafonné dans [0, 1])
       * EXT_SOURCE_3     - 0.05  (plafonné dans [0, 1])
       * DAYS_BIRTH       + 1500  (clients plus jeunes, borné à [-30000, -6576])
     L'analyse de drift (drift_analysis.py) doit RETROUVER ces features :
     c'est la validation du système de détection.
4. Injecte ~5 % de requêtes volontairement invalides (réparties sur les deux
   périodes) pour obtenir un taux d'erreur 422 réaliste :
   champ requis manquant / AMT_INCOME_TOTAL=0 / DAYS_BIRTH=-1000 /
   EXT_SOURCE_2=1.7 / AMT_CREDIT en texte.
5. Mesure la latence côté client (perf_counter autour du POST) en complément
   de la latence loguée par l'API.

Pré-requis : l'API doit TOURNER AVANT de lancer ce script :
    python -m uvicorn api.main:app --port 8002
Le script vérifie GET /health et s'arrête si l'API ne répond pas.

Un manifeste monitoring/data/simulation_manifest.json est écrit : il
mémorise le découpage période 1 / période 2 (par ordre d'insertion) pour
que l'analyse de drift puisse comparer les deux périodes.

Usage :
    python monitoring/simulate_traffic.py [--url http://127.0.0.1:8002]
                                          [--n 2000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests

RACINE_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_TEST = RACINE_PROJET / "data" / "processed" / "test.parquet"
CHEMIN_MANIFESTE = RACINE_PROJET / "monitoring" / "data" / "simulation_manifest.json"
CHEMIN_DB = RACINE_PROJET / "monitoring" / "data" / "prod_logs.db"

# Champs métier de premier niveau du schéma Pydantic (api/schemas.py).
CHAMPS_METIER = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
CHAMPS_ENTIERS = {"DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_CHILDREN"}

# Fraction de requêtes volontairement invalides (~5 %, réparties).
TAUX_INVALIDES = 0.05


def verifier_sante(url: str) -> None:
    """Vérifie GET /health ; interrompt le script si l'API est injoignable."""
    try:
        rep = requests.get(f"{url}/health", timeout=5)
        rep.raise_for_status()
        corps = rep.json()
    except Exception as exc:  # noqa: BLE001 - on veut tout message d'échec
        print(
            f"[ERREUR] API injoignable sur {url} ({exc}).\n"
            "Lancez d'abord : python -m uvicorn api.main:app --port 8002"
        )
        sys.exit(1)
    if not corps.get("model_loaded"):
        print("[ERREUR] L'API répond mais le modèle n'est pas chargé.")
        sys.exit(1)
    print(f"[OK] API en ligne sur {url} : {corps}")


def charger_echantillon(n: int, seed: int) -> pd.DataFrame:
    """Échantillonne n lignes de test.parquet et force le numérique."""
    df = pd.read_parquet(CHEMIN_TEST)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    echantillon = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    print(f"[OK] Échantillon : {echantillon.shape[0]} lignes x {echantillon.shape[1]} colonnes")
    return echantillon


def _assainir(nom: str) -> str:
    """Reproduit l'assainissement des noms de colonnes du pipeline d'entraînement.

    Le parquet brut contient des noms avec espaces (« FONDKAPREMONT_MODE_not
    specified ») alors que models/features.json attend des identifiants
    nettoyés (« FONDKAPREMONT_MODE_not_specified »).
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", nom)


def construire_payload(ligne: pd.Series) -> dict:
    """Construit le payload POST /predict à partir d'une ligne du parquet."""
    payload: dict = {}
    for champ in CHAMPS_METIER:
        valeur = ligne.get(champ)
        if pd.isna(valeur):
            if champ == "DAYS_EMPLOYED":
                # Champ obligatoire mais souvent NaN dans le parquet brut :
                # on utilise l'anomalie documentée Home Credit (365243 =
                # « non renseigné »), acceptée par le schéma de l'API.
                payload[champ] = 365243
            continue  # champs optionnels absents → None côté schéma
        if champ in CHAMPS_ENTIERS:
            payload[champ] = int(valeur)
        else:
            payload[champ] = float(valeur)
    features = {}
    for col, valeur in ligne.items():
        if col == "SK_ID_CURR" or col in CHAMPS_METIER or pd.isna(valeur):
            continue
        features[_assainir(col)] = float(valeur)
    if features:
        payload["features"] = features
    return payload


def appliquer_derive(payload: dict) -> dict:
    """Applique la dérive artificielle DOCUMENTÉE de la période 2.

    - AMT_CREDIT       × 1.40
    - AMT_INCOME_TOTAL × 0.85
    - EXT_SOURCE_2     - 0.08 (plafonné dans [0, 1])
    - EXT_SOURCE_3     - 0.05 (plafonné dans [0, 1])
    - DAYS_BIRTH       + 1500 jours, borné à [-30000, -6576] pour rester
      dans la plage métier valide (les 422 sont injectés séparément).
    """
    payload = dict(payload)
    if "AMT_CREDIT" in payload:
        payload["AMT_CREDIT"] = float(payload["AMT_CREDIT"]) * 1.40
    if "AMT_INCOME_TOTAL" in payload:
        payload["AMT_INCOME_TOTAL"] = float(payload["AMT_INCOME_TOTAL"]) * 0.85
    for champ, delta in (("EXT_SOURCE_2", -0.08), ("EXT_SOURCE_3", -0.05)):
        if champ in payload:
            payload[champ] = float(np.clip(float(payload[champ]) + delta, 0.0, 1.0))
    if "DAYS_BIRTH" in payload:
        payload["DAYS_BIRTH"] = int(np.clip(int(payload["DAYS_BIRTH"]) + 1500, -30000, -6576))
    return payload


def rendre_invalide(payload: dict, variante: int) -> dict:
    """Injecte une erreur de validation (→ 422 attendu)."""
    payload = dict(payload)
    variante = variante % 5
    if variante == 0:
        payload.pop("AMT_CREDIT", None)  # champ requis manquant
    elif variante == 1:
        payload["AMT_INCOME_TOTAL"] = 0  # doit être > 0
    elif variante == 2:
        payload["DAYS_BIRTH"] = -1000  # hors plage (-30000, -6575)
    elif variante == 3:
        payload["EXT_SOURCE_2"] = 1.7  # hors [0, 1]
    else:
        payload["AMT_CREDIT"] = "pas_un_nombre"  # type invalide
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation de trafic de production.")
    parser.add_argument("--url", default="http://127.0.0.1:8002", help="URL de base de l'API.")
    parser.add_argument("--n", type=int, default=2000, help="Nombre de requêtes à simuler.")
    parser.add_argument("--seed", type=int, default=42, help="Graine d'échantillonnage.")
    args = parser.parse_args()

    verifier_sante(args.url)
    echantillon = charger_echantillon(args.n, args.seed)
    n_total = len(echantillon)
    n_periode1 = int(round(n_total * 0.60))  # ~1200 pour n=2000

    # Indices des requêtes invalides : ~5 %, réparties uniformément.
    pas = max(1, int(round(1.0 / TAUX_INVALIDES)))
    indices_invalides = set(range(pas // 2, n_total, pas))

    session = requests.Session()
    compteurs = {"200": 0, "422": 0, "autres": 0}
    latences_client_ms: list[float] = []
    debut_global = perf_counter()

    for i, (_, ligne) in enumerate(echantillon.iterrows()):
        payload = construire_payload(ligne)
        periode = 1 if i < n_periode1 else 2
        if periode == 2:
            payload = appliquer_derive(payload)
        if i in indices_invalides:
            payload = rendre_invalide(payload, i)

        t0 = perf_counter()
        try:
            rep = session.post(f"{args.url}/predict", json=payload, timeout=30)
            statut = rep.status_code
        except Exception as exc:  # noqa: BLE001
            statut = -1
            print(f"[AVERTISSEMENT] Requête {i} en échec réseau : {exc}")
        latences_client_ms.append((perf_counter() - t0) * 1000.0)

        if statut == 200:
            compteurs["200"] += 1
        elif statut == 422:
            compteurs["422"] += 1
        else:
            compteurs["autres"] += 1

        if (i + 1) % 250 == 0:
            print(f"  ... {i + 1}/{n_total} requêtes envoyées")

    duree_s = perf_counter() - debut_global
    lat = np.array(latences_client_ms)
    manifeste = {
        "url": args.url,
        "n_total": n_total,
        "n_periode1": n_periode1,
        "n_periode2": n_total - n_periode1,
        "taux_invalides_cible": TAUX_INVALIDES,
        "nb_requetes_invalides_ciblees": len(indices_invalides),
        "derive_periode2": {
            "AMT_CREDIT": "x1.40",
            "AMT_INCOME_TOTAL": "x0.85",
            "EXT_SOURCE_2": "-0.08 (clip [0,1])",
            "EXT_SOURCE_3": "-0.05 (clip [0,1])",
            "DAYS_BIRTH": "+1500 jours (borne [-30000, -6576])",
        },
        "resultats": {
            "nb_200": compteurs["200"],
            "nb_422": compteurs["422"],
            "nb_autres": compteurs["autres"],
            "duree_totale_s": round(duree_s, 1),
            "latence_client_moy_ms": round(float(lat.mean()), 2),
            "latence_client_p95_ms": round(float(np.percentile(lat, 95)), 2),
        },
    }
    CHEMIN_MANIFESTE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHEMIN_MANIFESTE, "w", encoding="utf-8") as f:
        json.dump(manifeste, f, ensure_ascii=False, indent=2)

    print("\n===== RÉSUMÉ DE LA SIMULATION =====")
    print(f"Requêtes envoyées : {n_total} en {duree_s:.1f} s")
    print(f"Période 1 (normale) : {n_periode1} | Période 2 (dérivée) : {n_total - n_periode1}")
    print(f"HTTP 200 : {compteurs['200']} | HTTP 422 : {compteurs['422']} | autres : {compteurs['autres']}")
    print(f"Taux d'erreur 422 : {compteurs['422'] / n_total * 100:.1f} %")
    print(f"Latence client moyenne : {lat.mean():.1f} ms | p95 : {np.percentile(lat, 95):.1f} ms")
    print(f"Manifeste écrit : {CHEMIN_MANIFESTE}")

    # Vérification croisée avec la base SQLite.
    if CHEMIN_DB.exists():
        with sqlite3.connect(CHEMIN_DB) as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            par_statut = dict(
                conn.execute(
                    "SELECT http_status, COUNT(*) FROM predictions GROUP BY http_status"
                ).fetchall()
            )
        print(f"\nBase {CHEMIN_DB.name} : {total} lignes, répartition {par_statut}")
    else:
        print(f"\n[AVERTISSEMENT] Base introuvable : {CHEMIN_DB}")

    time.sleep(0.2)  # laisse le serveur finir d'écrire


if __name__ == "__main__":
    main()
