"""Analyse automatique du data drift — « Prêt à dépenser ».

Étape 3 (partie 2) de la consigne : comparer les données de PRODUCTION
stockées (SQLite monitoring/data/prod_logs.db, remplies par l'API) avec les
données de RÉFÉRENCE (échantillon de data/processed/train.parquet).

Méthodologie :
- Référence : 5 000 lignes de train.parquet (random_state=42), noms de
  colonnes assainis comme dans models/features.json.
- Production : requêtes http_status=200 uniquement, inputs_json re-parsé
  en DataFrame (features absentes → NaN). Les requêtes sont ordonnées par
  id d'insertion ; le manifeste de simulation permet de distinguer la
  période 1 (normale) de la période 2 (dérive artificielle injectée).
- Périmètre : les 30 features les plus importantes
  (artifacts/feature_importance.csv) + le score du modèle.
- Deux comparaisons Evidently (API nouvelle génération 0.7.x :
  Report + DataDriftPreset) :
    1. RÉFÉRENCE (train) vs PRODUCTION (global)  → drift_report.html
    2. PÉRIODE 1 vs PÉRIODE 2 (features + score) → drift_report_periodes.html
       (dérive de la sortie du modèle incluse via la colonne score)
- Résumé structuré → monitoring/drift_metrics.json
  (par feature : drift_score, statut_drift, methode ; nb driftées / total ;
  métriques opérationnelles : taux d'erreur, latences moyenne/p95).

Validation du système : la dérive artificielle injectée en période 2
(AMT_CREDIT ×1.4, AMT_INCOME_TOTAL ×0.85, EXT_SOURCE_2 −0.08,
EXT_SOURCE_3 −0.05, DAYS_BIRTH +1500) doit être RETROUVÉE par Evidently.

Usage : python monitoring/drift_analysis.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

RACINE_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_TRAIN = RACINE_PROJET / "data" / "processed" / "train.parquet"
CHEMIN_IMPORTANCE = RACINE_PROJET / "artifacts" / "feature_importance.csv"
CHEMIN_DB = RACINE_PROJET / "monitoring" / "data" / "prod_logs.db"
CHEMIN_MANIFESTE = RACINE_PROJET / "monitoring" / "data" / "simulation_manifest.json"
CHEMIN_JSON = RACINE_PROJET / "monitoring" / "drift_metrics.json"
CHEMIN_HTML_GLOBAL = RACINE_PROJET / "monitoring" / "drift_report.html"
CHEMIN_HTML_PERIODES = RACINE_PROJET / "monitoring" / "drift_report_periodes.html"

NB_REFERENCE = 5000
NB_TOP_FEATURES = 30
SEUIL = 0.45  # seuil métier de décision

# Features volontairement shiftées en période 2 par simulate_traffic.py
# (la détection de ces features valide le système de monitoring).
FEATURES_SHIFTEES = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
]


def _assainir(nom: str) -> str:
    """Assainit les noms de colonnes comme le pipeline d'entraînement."""
    return re.sub(r"[^A-Za-z0-9_]", "_", nom)


def charger_reference() -> pd.DataFrame:
    """Charge 5 000 lignes de train.parquet (référence d'entraînement)."""
    df = pd.read_parquet(CHEMIN_TRAIN)
    df = df.drop(columns=[c for c in ("TARGET", "SK_ID_CURR") if c in df.columns])
    df.columns = [_assainir(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    echantillon = df.sample(n=min(NB_REFERENCE, len(df)), random_state=42)
    print(f"[OK] Référence : {echantillon.shape[0]} lignes x {echantillon.shape[1]} colonnes")
    return echantillon


def charger_production() -> pd.DataFrame:
    """Reconstruit un DataFrame des inputs de production (http 200).

    Chaque inputs_json contient les champs métier de premier niveau et un
    dict `features`. Toutes les clés sont aplaties en colonnes ; les
    features absentes d'une requête restent NaN.
    """
    if not CHEMIN_DB.exists():
        raise FileNotFoundError(
            f"Base introuvable : {CHEMIN_DB}. Lancez d'abord monitoring/simulate_traffic.py."
        )
    with sqlite3.connect(CHEMIN_DB) as conn:
        lignes = conn.execute(
            "SELECT id, timestamp_utc, score, decision, latence_ms, "
            "inference_ms, inputs_json FROM predictions "
            "WHERE http_status = 200 ORDER BY id"
        ).fetchall()
    enregistrements = []
    for id_, ts, score, decision, lat, inf, inputs_json in lignes:
        entree: dict = {
            "id": id_,
            "timestamp_utc": ts,
            "score": score,
            "decision": decision,
            "latence_ms": lat,
            "inference_ms": inf,
        }
        if inputs_json:
            brut = json.loads(inputs_json)
            features = brut.pop("features", None) or {}
            for cle, val in brut.items():
                if cle != "features" and val is not None:
                    entree[cle] = val
            for cle, val in features.items():
                if val is not None:
                    entree[cle] = val
        enregistrements.append(entree)
    df = pd.DataFrame(enregistrements)
    for col in df.columns:
        if col not in {"decision", "timestamp_utc"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[OK] Production : {df.shape[0]} requêtes valides (http 200)")
    return df


def decouper_periodes(prod: pd.DataFrame, total_brut: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare période 1 / période 2 par ordre d'insertion (id).

    Le manifeste de simulation donne n_periode1 = nombre de requêtes
    (200 ET 422 confondues) de la période 1 : comme l'insertion est
    séquentielle, les n_periode1 premiers ids appartiennent à la période 1.
    """
    if not CHEMIN_MANIFESTE.exists():
        print("[INFO] Manifeste absent : analyse globale uniquement.")
        return prod, prod.iloc[0:0]
    with open(CHEMIN_MANIFESTE, encoding="utf-8") as f:
        n_periode1 = json.load(f)["n_periode1"]
    with sqlite3.connect(CHEMIN_DB) as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM predictions ORDER BY id").fetchall()]
    ids_p1 = set(ids[:n_periode1])
    p1 = prod[prod["id"].isin(ids_p1)]
    p2 = prod[~prod["id"].isin(ids_p1)]
    print(f"[OK] Période 1 : {len(p1)} requêtes valides | Période 2 : {len(p2)}")
    return p1, p2


def selectionner_features(ref: pd.DataFrame, prod: pd.DataFrame) -> list[str]:
    """Top 30 features par importance + les 5 features artificiellement shiftées.

    AMT_INCOME_TOTAL n'entre pas dans le top 30 : on l'ajoute explicitement
    pour que la comparaison période 1 vs période 2 puisse VALIDER la
    détection des 5 dérives injectées (c'est le test du système).
    """
    importance = pd.read_csv(CHEMIN_IMPORTANCE)
    top = importance["feature"].head(NB_TOP_FEATURES).tolist()
    candidats = list(dict.fromkeys(top + FEATURES_SHIFTEES))
    communes = [f for f in candidats if f in ref.columns and f in prod.columns]
    print(f"[OK] Features analysées : {len(communes)} (top {NB_TOP_FEATURES} + shiftées)")
    return communes


def extraire_resultats(snapshot_dict: dict) -> dict[str, dict]:
    """Extrait {colonne: {drift_score, statut_drift, methode}} d'un snapshot.

    API Evidently 0.7.x : snapshot.dict() → liste de métriques ; chaque
    ValueDrift porte la colonne, la méthode et le seuil dans sa config.
    Règle de décision : méthodes à p-value → drift si score < seuil ;
    méthodes à distance (Jensen-Shannon, Wasserstein...) → drift si
    score >= seuil.
    """
    resultats: dict[str, dict] = {}
    for metrique in snapshot_dict.get("metrics", []):
        config = metrique.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        colonne = config["column"]
        methode = config.get("method", "?")
        seuil_met = config.get("threshold")
        score = metrique.get("value")
        if score is None:
            continue
        est_pvalue = "p_value" in methode or "p-value" in methode
        if seuil_met is not None:
            statut = bool(score < seuil_met) if est_pvalue else bool(score >= seuil_met)
        else:
            statut = None
        resultats[colonne] = {
            "drift_score": round(float(score), 6),
            "statut_drift": statut,
            "methode": methode,
        }
    return resultats


def lancer_evidently(
    reference: pd.DataFrame, courant: pd.DataFrame, chemin_html: Path, titre: str
) -> dict[str, dict]:
    """Exécute un Report Evidently (DataDriftPreset) et sauvegarde le HTML."""
    rapport = Report(metrics=[DataDriftPreset()])
    snapshot = rapport.run(reference_data=reference, current_data=courant)
    snapshot.save_html(str(chemin_html))
    resultats = extraire_resultats(snapshot.dict())
    nb_drift = sum(1 for r in resultats.values() if r["statut_drift"])
    print(f"[OK] {titre} : {nb_drift}/{len(resultats)} colonnes en dérive → {chemin_html.name}")
    return resultats


def metriques_operationnelles() -> dict:
    """Taux d'erreur, latences, scores et décisions depuis la base."""
    with sqlite3.connect(CHEMIN_DB) as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        par_statut = dict(
            conn.execute(
                "SELECT http_status, COUNT(*) FROM predictions GROUP BY http_status"
            ).fetchall()
        )
        lat = pd.read_sql_query(
            "SELECT latence_ms FROM predictions WHERE latence_ms IS NOT NULL", conn
        )["latence_ms"]
        scores = pd.read_sql_query(
            "SELECT score FROM predictions WHERE score IS NOT NULL", conn
        )["score"]
        decisions = dict(
            conn.execute(
                "SELECT decision, COUNT(*) FROM predictions "
                "WHERE decision IS NOT NULL GROUP BY decision"
            ).fetchall()
        )
    nb_succes = par_statut.get(200, 0)
    return {
        "nb_requetes_total": total,
        "nb_succes_200": nb_succes,
        "nb_erreurs_422": par_statut.get(422, 0),
        "taux_erreur_pct": round((total - nb_succes) / total * 100, 2) if total else None,
        "latence_moyenne_ms": round(float(lat.mean()), 2) if len(lat) else None,
        "latence_p95_ms": round(float(np.percentile(lat, 95)), 2) if len(lat) else None,
        "score_moyen": round(float(scores.mean()), 4) if len(scores) else None,
        "pct_refuses": round(decisions.get("refuse", 0) / nb_succes * 100, 2) if nb_succes else None,
        "decisions": decisions,
    }


def main() -> None:
    ref = charger_reference()
    prod = charger_production()
    p1, p2 = decouper_periodes(prod, len(prod))
    features = selectionner_features(ref, prod)

    # --- Comparaison 1 : référence (train) vs production (globale) ---
    res_global = lancer_evidently(
        ref[features], prod[features], CHEMIN_HTML_GLOBAL, "Référence vs production globale"
    )

    # --- Comparaison 2 : période 1 vs période 2 (features + score) ---
    res_periodes: dict[str, dict] = {}
    if len(p2) > 0:
        colonnes = features + ["score"]
        res_periodes = lancer_evidently(
            p1[colonnes], p2[colonnes], CHEMIN_HTML_PERIODES, "Période 1 vs période 2"
        )

    # --- Validation : la dérive injectée est-elle retrouvée ? ---
    validation = {}
    for feat in FEATURES_SHIFTEES:
        detecte_periodes = res_periodes.get(feat, {}).get("statut_drift")
        detecte_global = res_global.get(feat, {}).get("statut_drift")
        validation[feat] = {
            "detectee_p1_vs_p2": detecte_periodes,
            "detectee_ref_vs_prod": detecte_global,
        }
    nb_retrouvees = sum(1 for v in validation.values() if v["detectee_p1_vs_p2"])

    nb_drift_global = sum(1 for r in res_global.values() if r["statut_drift"])
    nb_drift_periodes = sum(1 for r in res_periodes.values() if r["statut_drift"])
    conclusion = (
        f"{nb_drift_global}/{len(res_global)} features en dérive entre la référence "
        f"d'entraînement et la production globale ; {nb_drift_periodes}/{len(res_periodes)} "
        "colonnes (features + score) en dérive entre la période 1 et la période 2. "
        f"{nb_retrouvees}/{len(FEATURES_SHIFTEES)} features artificiellement shiftées "
        "retrouvées par la comparaison période 1 vs période 2."
    )

    synthese = {
        "date_analyse_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {"source": "data/processed/train.parquet", "nb_lignes": len(ref)},
        "production": {
            "source": "monitoring/data/prod_logs.db (http_status=200)",
            "nb_requetes_valides": len(prod),
            "nb_periode1": len(p1),
            "nb_periode2": len(p2),
        },
        "seuil_metier": SEUIL,
        "ref_vs_prod_globale": {
            "fichier_html": CHEMIN_HTML_GLOBAL.name,
            "features": res_global,
            "nb_features_driftees": nb_drift_global,
            "nb_features_total": len(res_global),
        },
        "periode1_vs_periode2": {
            "fichier_html": CHEMIN_HTML_PERIODES.name if res_periodes else None,
            "colonnes": res_periodes,
            "nb_colonnes_driftees": nb_drift_periodes,
            "nb_colonnes_total": len(res_periodes),
            "drift_score_modele": res_periodes.get("score"),
        },
        "features_shiftées_attendues": FEATURES_SHIFTEES,
        "validation_derive_injectee": validation,
        "nb_features_shiftees_retrouvees": nb_retrouvees,
        "conclusion": conclusion,
        "metriques_operationnelles": metriques_operationnelles(),
    }
    with open(CHEMIN_JSON, "w", encoding="utf-8") as f:
        json.dump(synthese, f, ensure_ascii=False, indent=2)

    print(f"\n===== SYNTHÈSE =====\n{conclusion}")
    print("Features shiftées retrouvées :")
    for feat, v in validation.items():
        print(f"  - {feat}: p1_vs_p2={v['detectee_p1_vs_p2']}, ref_vs_prod={v['detectee_ref_vs_prod']}")
    print(f"\nJSON écrit : {CHEMIN_JSON}")


if __name__ == "__main__":
    main()
