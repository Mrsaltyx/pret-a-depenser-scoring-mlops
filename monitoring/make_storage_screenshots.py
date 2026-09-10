"""Génère les captures « solution de stockage » (livrable de la consigne).

Produit dans docs/screenshots/ (matplotlib, dpi=150, sans navigateur) :
- storage_schema.png   : schéma de la table predictions (PRAGMA table_info),
                         comptage de lignes et répartition des http_status ;
- storage_extrait.png  : extrait de 10 lignes réelles de la base
                         (colonnes principales, inputs_json tronqué) ;
- monitoring_apercu.png: planche de 4 mini-graphiques (distribution des
                         scores, latence dans le temps, décisions, top
                         features en dérive) = aperçu du dashboard.

Usage : python monitoring/make_storage_screenshots.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_DB = RACINE_PROJET / "monitoring" / "data" / "prod_logs.db"
CHEMIN_DRIFT = RACINE_PROJET / "monitoring" / "drift_metrics.json"
DOSSIER_SORTIE = RACINE_PROJET / "docs" / "screenshots"
SEUIL = 0.45
DPI = 150

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold"})


def charger() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with sqlite3.connect(CHEMIN_DB) as conn:
        schema = pd.read_sql_query("PRAGMA table_info(predictions)", conn)
        logs = pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)
    drift = {}
    if CHEMIN_DRIFT.exists():
        with open(CHEMIN_DRIFT, encoding="utf-8") as f:
            drift = json.load(f)
    return schema, logs, drift


def fig_schema(schema: pd.DataFrame, logs: pd.DataFrame, chemin: Path) -> None:
    """Schéma de la table predictions + volumétrie."""
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.25)

    # --- Tableau du schéma (PRAGMA table_info) ---
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")
    ax_t.set_title("Schéma de la table SQLite « predictions »\n(PRAGMA table_info)", loc="left")
    colonnes = ["cid", "name", "type", "notnull", "pk"]
    tableau = ax_t.table(
        cellText=schema[colonnes].values,
        colLabels=["#", "Colonne", "Type", "NOT NULL", "PK"],
        colWidths=[0.08, 0.34, 0.20, 0.20, 0.10],
        loc="center",
        cellLoc="left",
    )
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(9)
    tableau.scale(1, 1.35)
    for j in range(len(colonnes)):
        tableau[0, j].set_facecolor("#4C78A8")
        tableau[0, j].set_text_props(color="white", weight="bold")
    ax_t.text(
        0, -0.06,
        "Double écriture : SQLite prod_logs.db + miroir api_logs.jsonl "
        "(voir api/storage.py).",
        transform=ax_t.transAxes, fontsize=8, style="italic",
    )

    # --- Volumétrie ---
    ax_v = fig.add_subplot(gs[1])
    ax_v.axis("off")
    ax_v.set_title("Volumétrie", loc="left")
    par_statut = logs["http_status"].value_counts().sort_index()
    lignes = [
        f"Nombre total de lignes : {len(logs):,}".replace(",", " "),
        "",
        "Répartition par http_status :",
    ] + [f"  • {s} : {n} ({n / len(logs) * 100:.1f} %)" for s, n in par_statut.items()] + [
        "",
        f"Premier appel : {logs['timestamp_utc'].iloc[0][:19]} UTC",
        f"Dernier appel  : {logs['timestamp_utc'].iloc[-1][:19]} UTC",
        "",
        "Décisions (succès) :",
    ]
    dec = logs["decision"].value_counts()
    lignes += [f"  • {d} : {n} ({n / dec.sum() * 100:.1f} %)" for d, n in dec.items()]
    ax_v.text(0, 0.95, "\n".join(lignes), transform=ax_v.transAxes,
              va="top", fontsize=8.5, family="monospace",
              bbox=dict(boxstyle="round", facecolor="#F2F2F2", edgecolor="#CCCCCC"))
    fig.suptitle("Solution de stockage des données de production — « Prêt à dépenser »",
                 fontsize=13, fontweight="bold")
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {chemin}")


def fig_extrait(logs: pd.DataFrame, chemin: Path) -> None:
    """Extrait de 10 lignes réelles (inputs_json tronqué)."""
    extrait = logs.head(10).copy()
    extrait["timestamp_utc"] = extrait["timestamp_utc"].str[:19]
    extrait["request_id"] = extrait["request_id"].str[:8] + "…"
    extrait["inputs_json"] = extrait["inputs_json"].fillna("").str[:42] + "…"
    extrait["erreur"] = extrait["erreur"].fillna("").str[:40]
    extrait["erreur"] = extrait["erreur"].where(extrait["erreur"] != "", "")
    colonnes = ["id", "timestamp_utc", "http_status", "score", "decision",
                "latence_ms", "erreur", "inputs_json"]

    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    ax.axis("off")
    tableau = ax.table(
        cellText=extrait[colonnes].round(3).values,
        colLabels=colonnes,
        colWidths=[0.03, 0.13, 0.07, 0.06, 0.07, 0.07, 0.10, 0.30],
        loc="center",
        cellLoc="left",
    )
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(7.5)
    tableau.scale(1, 1.4)
    for j in range(len(colonnes)):
        tableau[0, j].set_facecolor("#4C78A8")
        tableau[0, j].set_text_props(color="white", weight="bold")
    ax.set_title(
        "Extrait de 10 lignes réelles — table « predictions » "
        "(inputs_json tronqué à 42 caractères)",
        fontsize=12, fontweight="bold", pad=18,
    )
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {chemin}")


def fig_apercu(logs: pd.DataFrame, drift: dict, chemin: Path) -> None:
    """Planche de 4 mini-graphiques = aperçu du dashboard."""
    succes = logs[logs["http_status"] == 200]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Distribution des scores + seuil
    ax = axes[0, 0]
    ax.hist(succes["score"].dropna(), bins=40, color="#4C78A8", edgecolor="white")
    ax.axvline(SEUIL, color="red", linestyle="--", linewidth=2,
               label=f"Seuil métier = {SEUIL}")
    ax.set_title("Distribution des scores prédits")
    ax.set_xlabel("Probabilité de défaut")
    ax.set_ylabel("Dossiers")
    ax.legend(fontsize=8)

    # 2. Latence dans le temps (moyenne glissante)
    ax = axes[0, 1]
    lat = logs.dropna(subset=["latence_ms"])
    ax.plot(lat["id"], lat["latence_ms"], color="#BBBBBB", linewidth=0.6,
            label="par requête")
    ax.plot(lat["id"], lat["latence_ms"].rolling(50, min_periods=1).mean(),
            color="#E15759", linewidth=2, label="moyenne glissante (50)")
    ax.set_title("Latence API dans le temps")
    ax.set_xlabel("Requête n°")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)

    # 3. Répartition des décisions
    ax = axes[1, 0]
    dec = succes["decision"].value_counts()
    couleurs = ["#59A14F" if d == "accorde" else "#E15759" for d in dec.index]
    ax.bar(dec.index, dec.values, color=couleurs)
    for i, v in enumerate(dec.values):
        ax.text(i, v, f"{v} ({v / len(succes) * 100:.1f} %)", ha="center", va="bottom",
                fontsize=9)
    ax.set_title("Répartition des décisions (succès)")
    ax.set_ylabel("Dossiers")

    # 4. Top features en dérive (comparaison période 1 vs période 2)
    ax = axes[1, 1]
    colonnes_drift = drift.get("periode1_vs_periode2", {}).get("colonnes", {})
    driftees = sorted(
        ((k, v["drift_score"]) for k, v in colonnes_drift.items() if v["statut_drift"]),
        key=lambda kv: kv[1], reverse=True,
    )[:8]
    if driftees:
        noms = [k for k, _ in driftees][::-1]
        valeurs = [v for _, v in driftees][::-1]
        ax.barh(noms, valeurs, color="#F28E2B")
        ax.set_title("Colonnes en dérive — période 1 vs période 2\n(drift score Evidently)")
        for i, v in enumerate(valeurs):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Aucune donnée de drift\n(lancer drift_analysis.py)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Colonnes en dérive")

    taux = (len(logs) - len(succes)) / len(logs) * 100
    fig.suptitle(
        f"Aperçu du dashboard de monitoring — {len(logs)} requêtes, "
        f"{taux:.1f} % d'erreurs 422, "
        f"latence moy. {logs['latence_ms'].mean():.2f} ms",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(chemin, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {chemin}")


def main() -> None:
    if not CHEMIN_DB.exists():
        raise FileNotFoundError(
            f"Base introuvable : {CHEMIN_DB}. Lancez d'abord monitoring/simulate_traffic.py."
        )
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    schema, logs, drift = charger()
    fig_schema(schema, logs, DOSSIER_SORTIE / "storage_schema.png")
    fig_extrait(logs, DOSSIER_SORTIE / "storage_extrait.png")
    fig_apercu(logs, drift, DOSSIER_SORTIE / "monitoring_apercu.png")


if __name__ == "__main__":
    main()
