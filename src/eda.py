"""EDA — Projet « Prêt à dépenser ».

Génère des figures d'analyse exploratoire dans artifacts/figures/ :
1. Distribution de la cible TARGET (déséquilibre des classes).
2. Top 30 colonnes par taux de valeurs manquantes (train).
3. Distributions de DAYS_BIRTH (âge) et AMT_CREDIT.

Utilise matplotlib en backend Agg (aucun affichage).
Exécution : python src/eda.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import APPLICATION_TRAIN, FIGURES_DIR, TRAIN_PARQUET  # noqa: E402


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # On relit le CSV brut pour la cible + variables d'origine,
    # et le parquet traité pour les valeurs manquantes.
    raw = pd.read_csv(
        APPLICATION_TRAIN,
        usecols=["SK_ID_CURR", "TARGET", "DAYS_BIRTH", "AMT_CREDIT"],
    )
    train = pd.read_parquet(TRAIN_PARQUET)

    # 1. Distribution de la cible
    counts = raw["TARGET"].value_counts().sort_index()
    ratio = counts[1] / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["0 — Prêt remboursé", "1 — Défaut"], counts.values,
           color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum() * 100:.1f} %)",
                ha="center", va="bottom")
    ax.set_title(f"Distribution de la cible TARGET (défauts : {ratio:.1f} %)")
    ax.set_ylabel("Nombre de clients")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Top 30 colonnes par taux de valeurs manquantes
    missing = train.isna().mean().sort_values(ascending=False).head(30) * 100
    fig, ax = plt.subplots(figsize=(9, 8))
    missing.iloc[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Top 30 colonnes par taux de valeurs manquantes (train)")
    ax.set_xlabel("Valeurs manquantes (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missing_values_top30.png", dpi=150)
    plt.close(fig)

    # 3. Distributions DAYS_BIRTH et AMT_CREDIT
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    age = -raw["DAYS_BIRTH"] / 365.25
    axes[0].hist(age, bins=50, color="#4C72B0")
    axes[0].set_title("Distribution de l'âge des clients")
    axes[0].set_xlabel("Âge (années)")
    axes[0].set_ylabel("Nombre de clients")
    axes[1].hist(raw["AMT_CREDIT"].dropna() / 1e3, bins=50, color="#55A868")
    axes[1].set_title("Distribution du montant du crédit (AMT_CREDIT)")
    axes[1].set_xlabel("Montant du crédit (milliers)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions_age_credit.png", dpi=150)
    plt.close(fig)

    print("Figures sauvegardées dans", FIGURES_DIR)
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
