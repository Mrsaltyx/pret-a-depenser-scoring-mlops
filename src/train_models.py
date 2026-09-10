"""Comparaison de modèles avec tracking MLflow — Phase 2.

Modèles comparés : DummyClassifier (baseline), LogisticRegression,
RandomForest, LightGBM.

Protocole :
- Split stratifié 80 % train_dev / 20 % holdout (holdout = évaluation finale
  honnête uniquement).
- Probabilités OOF via cross_val_predict (StratifiedKFold 5) sur train_dev.
- Seuil optimal par coût métier (FN = 10 x FP), métriques AUC/accuracy/
  recall/F1, courbes ROC et coût-vs-seuil.
- Réentraînement sur tout train_dev puis évaluation sur holdout.

Exécution : python src/train_models.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from business import cout_metier, trouver_seuil_optimal  # noqa: E402
from config import (  # noqa: E402
    ARTIFACTS,
    FIGURES_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TRAIN_PARQUET,
    nettoyer_noms_features,
)

EXPERIMENT_NAME = "pret_a_depenser_scoring"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plot_roc_oof(y_true, proba, nom: str, auc: float) -> Path:
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC OOF = {auc:.4f}", color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(f"Courbe ROC (OOF) — {nom}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"roc_oof_{nom}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cout_seuil(tableau, nom: str, seuil: float, cout_min: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tableau["seuil"], tableau["cout"], color="#C44E52")
    ax.axvline(seuil, color="k", ls="--",
               label=f"Seuil optimal = {seuil:.2f} (coût = {cout_min:,.0f})")
    ax.set_xlabel("Seuil de classification")
    ax.set_ylabel("Coût métier (FN=10, FP=1)")
    ax.set_title(f"Coût métier vs seuil — {nom}")
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / f"cout_seuil_{nom}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    log("Chargement train.parquet ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X.columns = nettoyer_noms_features(X.columns)
    y = df["TARGET"].astype(int)

    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(f"train_dev : {X_dev.shape}, holdout : {X_hold.shape}")

    ratio = float((y_dev == 0).sum() / (y_dev == 1).sum())
    log(f"ratio nég/pos = {ratio:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    modeles = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE)),
        ]),
        # n_estimators réduit 100 -> 60 par précaution budgétaire (documenté
        # dans JOURNAL_ACTIONS.md) : ~5-6 fits sur 246k x 402 sinon trop long.
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=60, max_depth=15,
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "lightgbm": LGBMClassifier(
            scale_pos_weight=ratio, n_estimators=500, learning_rate=0.05,
            num_leaves=31, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        ),
    }

    resultats = []

    for nom, modele in modeles.items():
        tm = time.time()
        log(f"=== {nom} : cross_val_predict OOF (5 folds) ...")
        proba_oof = cross_val_predict(
            modele, X_dev, y_dev, cv=cv, method="predict_proba"
        )[:, 1]

        auc_oof = roc_auc_score(y_dev, proba_oof)
        seuil, cout_min, tableau = trouver_seuil_optimal(y_dev, proba_oof)
        pred_opt = (proba_oof >= seuil).astype(int)
        acc = accuracy_score(y_dev, pred_opt)
        rec = recall_score(y_dev, pred_opt)
        f1 = f1_score(y_dev, pred_opt)

        log(f"  {nom} : réentraînement sur train_dev + éval holdout ...")
        modele.fit(X_dev, y_dev)
        proba_hold = modele.predict_proba(X_hold)[:, 1]
        auc_hold = roc_auc_score(y_hold, proba_hold)
        pred_hold = (proba_hold >= seuil).astype(int)
        cout_hold = cout_metier(y_hold, pred_hold)
        acc_hold = accuracy_score(y_hold, pred_hold)

        roc_path = plot_roc_oof(y_dev, proba_oof, nom, auc_oof)
        cout_path = plot_cout_seuil(tableau, nom, seuil, cout_min)

        with mlflow.start_run(run_name=f"comparaison_{nom}"):
            mlflow.set_tag("etape", "comparaison_modeles")
            mlflow.set_tag("modele", nom)
            try:
                mlflow.log_params(modele.get_params())
            except Exception:
                pass
            mlflow.log_metrics({
                "auc_oof": auc_oof,
                "seuil_optimal": seuil,
                "cout_metier_oof": cout_min,
                "accuracy_oof": acc,
                "recall_oof": rec,
                "f1_oof": f1,
                "holdout_auc": auc_hold,
                "holdout_cout_metier": float(cout_hold),
                "holdout_accuracy": acc_hold,
            })
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(cout_path))

        resultats.append({
            "modele": nom,
            "auc_oof": round(auc_oof, 4),
            "cout_oof": int(cout_min),
            "seuil_optimal": seuil,
            "accuracy_oof": round(acc, 4),
            "recall_oof": round(rec, 4),
            "f1_oof": round(f1, 4),
            "auc_holdout": round(auc_hold, 4),
            "cout_holdout": int(cout_hold),
            "accuracy_holdout": round(acc_hold, 4),
            "duree_s": round(time.time() - tm, 1),
        })
        log(f"  {nom} terminé en {time.time() - tm:.0f}s — "
            f"AUC OOF={auc_oof:.4f}, coût OOF={cout_min:,.0f} "
            f"(seuil {seuil:.2f}), AUC holdout={auc_hold:.4f}, "
            f"coût holdout={cout_hold:,.0f}")

    res_df = pd.DataFrame(resultats)
    csv_path = ARTIFACTS / "comparaison_modeles.csv"
    res_df.to_csv(csv_path, index=False)
    log(f"Résultats sauvegardés : {csv_path}")
    log(f"TERMINÉ en {time.time() - t0:.0f}s")
    print(res_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
