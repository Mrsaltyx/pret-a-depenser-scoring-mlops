"""Dashboard de monitoring « Prêt à dépenser » — Streamlit.

Étape 3 (partie 2) de la consigne : visualiser les données de production
stockées (SQLite) et les résultats de l'analyse de drift (Evidently).

Sources :
- monitoring/data/prod_logs.db   : journal des appels API (table predictions)
- monitoring/drift_metrics.json  : synthèse produite par drift_analysis.py

Lancement :  streamlit run monitoring/dashboard.py
Le bouton « Recharger les données » vide le cache (TTL court de 30 s).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RACINE_PROJET = Path(__file__).resolve().parent.parent
CHEMIN_DB = RACINE_PROJET / "monitoring" / "data" / "prod_logs.db"
CHEMIN_DRIFT = RACINE_PROJET / "monitoring" / "drift_metrics.json"
SEUIL = 0.45

st.set_page_config(
    page_title="Monitoring — Prêt à dépenser",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=30)
def charger_logs() -> pd.DataFrame:
    """Charge le journal de production depuis SQLite."""
    if not CHEMIN_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(CHEMIN_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id", conn)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    return df


@st.cache_data(ttl=30)
def charger_drift() -> dict:
    """Charge la synthèse de drift produite par drift_analysis.py."""
    if not CHEMIN_DRIFT.exists():
        return {}
    with open(CHEMIN_DRIFT, encoding="utf-8") as f:
        return json.load(f)


st.title("📊 Dashboard de monitoring — API de scoring « Prêt à dépenser »")
st.caption(
    "Journal de production SQLite + analyse de data drift Evidently. "
    "Décision métier : score ≥ 0,45 → **refusé**."
)

if st.button("🔄 Recharger les données"):
    st.cache_data.clear()

logs = charger_logs()
drift = charger_drift()

if logs.empty:
    st.warning(
        "Aucune donnée de production. Lancez d'abord "
        "`python monitoring/simulate_traffic.py` (API démarrée sur le port 8002)."
    )
    st.stop()

succes = logs[logs["http_status"] == 200]

# ---------------------------------------------------------------- KPIs
st.subheader("Indicateurs clés")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Requêtes totales", f"{len(logs):,}".replace(",", " "))
taux_erreur = (len(logs) - len(succes)) / len(logs) * 100
c2.metric("Taux d'erreur", f"{taux_erreur:.1f} %")
lat = logs["latence_ms"].dropna()
c3.metric("Latence moyenne", f"{lat.mean():.2f} ms" if len(lat) else "—")
c4.metric("Latence p95", f"{np.percentile(lat, 95):.2f} ms" if len(lat) else "—")
scores = succes["score"].dropna()
c5.metric("Score moyen", f"{scores.mean():.4f}" if len(scores) else "—")
pct_refus = (succes["decision"] == "refuse").mean() * 100 if len(succes) else 0
c6.metric("% refusés", f"{pct_refus:.1f} %")

st.divider()

# ------------------------------------------------- distribution des scores
st.subheader("Distribution des scores prédits")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(scores, bins=40, color="#4C78A8", edgecolor="white")
ax.axvline(SEUIL, color="red", linestyle="--", linewidth=2, label=f"Seuil métier = {SEUIL}")
ax.set_xlabel("Probabilité de défaut prédite")
ax.set_ylabel("Nombre de dossiers")
ax.legend()
st.pyplot(fig)
plt.close(fig)

col_a, col_b = st.columns(2)

# --------------------------------------------------- latence dans le temps
with col_a:
    st.subheader("Latence API dans le temps")
    df_lat = logs.dropna(subset=["latence_ms"])[["id", "latence_ms"]].set_index("id")
    df_lat["moyenne_glissante"] = df_lat["latence_ms"].rolling(50, min_periods=1).mean()
    st.line_chart(df_lat, y=["latence_ms", "moyenne_glissante"])
    st.caption("Latence par requête (ms) et moyenne glissante sur 50 requêtes.")

# ------------------------------------------------- répartition décisions
with col_b:
    st.subheader("Répartition des décisions")
    if len(succes):
        nb_dec = succes["decision"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.bar(
            nb_dec.index,
            nb_dec.values,
            color=["#59A14F" if d == "accorde" else "#E15759" for d in nb_dec.index],
        )
        for i, v in enumerate(nb_dec.values):
            ax2.text(i, v, f"{v} ({v / len(succes) * 100:.1f} %)", ha="center", va="bottom")
        ax2.set_ylabel("Nombre de dossiers")
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Aucune requête réussie.")

# ---------------------------------------------- taux d'erreur dans le temps
st.subheader("Taux d'erreur dans le temps")
st.caption(
    "Une moyenne glissante requête par requête est plate par construction "
    "(erreurs injectées de façon régulière) : on agrège donc par **paquets "
    "de requêtes**, bien plus lisible pour repérer les périodes à problème."
)
taille = st.slider("Taille d'un paquet (nombre de requêtes)", 50, 500, 100, step=50)
df_err = logs[["id", "http_status"]].copy()
df_err["erreur"] = (df_err["http_status"] != 200).astype(int)
df_err["paquet"] = (df_err["id"] - 1) // taille
par_paquet = (
    df_err.groupby("paquet")
    .agg(debut=("id", "min"), fin=("id", "max"), total=("erreur", "size"), erreurs=("erreur", "sum"))
    .reset_index()
)
par_paquet["taux_pct"] = par_paquet["erreurs"] / par_paquet["total"] * 100
fig3, ax3 = plt.subplots(figsize=(9.5, 3.2))
couleurs = ["#E15759" if t > 5 else "#4C78A8" for t in par_paquet["taux_pct"]]
ax3.bar(range(len(par_paquet)), par_paquet["taux_pct"], color=couleurs)
ax3.axhline(5, color="gray", linestyle="--", linewidth=1.5, label="cible injectée : 5 %")
ax3.set_xticks(range(len(par_paquet)))
ax3.set_xticklabels(
    [f"{int(r.debut)}–{int(r.fin)}" for r in par_paquet.itertuples()],
    rotation=45, ha="right", fontsize=8,
)
ax3.set_ylabel("% d'erreurs (HTTP ≠ 200)")
ax3.set_xlabel("Requêtes (par ordre d'arrivée)")
ax3.legend()
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)
dernier = par_paquet.iloc[-1]
if dernier["total"] < taille and dernier["taux_pct"] > 10:
    st.caption(
        f"🔎 Pic final normal : les requêtes {int(dernier['debut'])}–{int(dernier['fin'])} "
        f"sont des tests manuels (Swagger/démo) avec des payloads volontairement invalides — "
        f"{dernier['taux_pct']:.0f} % d'erreurs sur ce petit paquet."
    )

st.divider()

# ------------------------------------------------------ analyse de drift
st.subheader("Data drift — synthèse Evidently")
if not drift:
    st.info(
        "`drift_metrics.json` absent : lancez `python monitoring/drift_analysis.py` "
        "pour produire l'analyse."
    )
else:
    st.write(f"**Conclusion :** {drift.get('conclusion', '—')}")
    c1, c2 = st.columns(2)
    g = drift.get("ref_vs_prod_globale", {})
    with c1:
        st.markdown(
            f"**Référence vs production** : "
            f"{g.get('nb_features_driftees', '?')}/{g.get('nb_features_total', '?')} "
            "features en dérive"
        )
        lignes_g = [
            {
                "Feature": k,
                "Méthode": v["methode"],
                "Drift score": v["drift_score"],
                "Dérive": "🔴 oui" if v["statut_drift"] else "🟢 non",
            }
            for k, v in g.get("features", {}).items()
        ]
        st.dataframe(pd.DataFrame(lignes_g), width="stretch", hide_index=True)
    p = drift.get("periode1_vs_periode2", {})
    with c2:
        st.markdown(
            f"**Période 1 vs période 2 (dérive injectée)** : "
            f"{p.get('nb_colonnes_driftees', '?')}/{p.get('nb_colonnes_total', '?')} "
            "colonnes en dérive (score inclus)"
        )
        lignes_p = [
            {
                "Colonne": k,
                "Méthode": v["methode"],
                "Drift score": v["drift_score"],
                "Dérive": "🔴 oui" if v["statut_drift"] else "🟢 non",
            }
            for k, v in p.get("colonnes", {}).items()
        ]
        st.dataframe(pd.DataFrame(lignes_p), width="stretch", hide_index=True)

    val = drift.get("validation_derive_injectee", {})
    if val:
        retrouvees = sum(1 for v in val.values() if v.get("detectee_p1_vs_p2"))
        st.success(
            f"Validation du système : {retrouvees}/{len(val)} features artificiellement "
            f"shiftées retrouvées ({', '.join(val)})."
        )
    liens = []
    if (RACINE_PROJET / "monitoring" / "drift_report.html").exists():
        liens.append("`monitoring/drift_report.html`")
    if (RACINE_PROJET / "monitoring" / "drift_report_periodes.html").exists():
        liens.append("`monitoring/drift_report_periodes.html`")
    if liens:
        st.caption("Rapports Evidently détaillés : " + " et ".join(liens))

st.divider()
st.caption(
    f"Base : `{CHEMIN_DB.name}` — {len(logs)} lignes | "
    "Détail méthodologique : `monitoring/ANALYSE_DRIFT.md`"
)
