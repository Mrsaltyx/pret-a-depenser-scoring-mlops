"""Modèle final + registry MLflow + interprétabilité — Phase 2.

- Réentraîne LGBMClassifier avec les meilleurs hyperparamètres Optuna sur
  TOUT train_dev.
- Seuil optimal via probabilités OOF (5-fold) sur train_dev.
- Évaluation finale sur le holdout (AUC, accuracy, coût métier).
- Enregistrement dans le model registry MLflow
  ("credit_scoring_pret_a_depenser") avec signature et exemple d'entrée.
- Feature importance globale (gain) + SHAP local (beeswarm + 2 waterfall).
- Sauvegarde artifacts/seuil_optimal.json et artifacts/meilleurs_params.json.

Exécution : python src/finalize_model.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from mlflow.models.signature import infer_signature  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402

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
REGISTERED_NAME = "credit_scoring_pret_a_depenser"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with open(ARTIFACTS / "meilleurs_params.json", encoding="utf-8") as f:
        opt = json.load(f)
    best_params = opt["best_params"]
    ratio = opt["scale_pos_weight"]
    log(f"Meilleurs params Optuna : {best_params}")

    log("Chargement train.parquet ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X.columns = nettoyer_noms_features(X.columns)
    y = df["TARGET"].astype(int)
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    params = {
        **best_params,
        "scale_pos_weight": ratio,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    # --- Seuil optimal via OOF 5-fold sur train_dev ----------------------
    log("Probas OOF (5 folds) pour le seuil optimal ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba_oof = cross_val_predict(
        LGBMClassifier(**params), X_dev, y_dev, cv=cv,
        method="predict_proba",
    )[:, 1]
    seuil, cout_oof, _ = trouver_seuil_optimal(y_dev, proba_oof)
    auc_oof = roc_auc_score(y_dev, proba_oof)
    log(f"OOF : AUC={auc_oof:.4f}, seuil optimal={seuil:.2f}, "
        f"coût={cout_oof:,.0f}")

    # --- Entraînement final + holdout ------------------------------------
    log("Entraînement du modèle final sur tout train_dev ...")
    model = LGBMClassifier(**params)
    model.fit(X_dev, y_dev)
    proba_hold = model.predict_proba(X_hold)[:, 1]
    pred_hold = (proba_hold >= seuil).astype(int)
    auc_hold = roc_auc_score(y_hold, proba_hold)
    acc_hold = accuracy_score(y_hold, pred_hold)
    cout_hold = cout_metier(y_hold, pred_hold)
    seuil_hold, cout_hold_min, tab_hold = trouver_seuil_optimal(
        y_hold, proba_hold)
    log(f"Holdout : AUC={auc_hold:.4f}, accuracy={acc_hold:.4f}, "
        f"coût (seuil OOF {seuil:.2f})={cout_hold:,.0f}")

    # Figures holdout
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_hold, proba_hold)
    ax.plot(fpr, tpr, label=f"AUC holdout = {auc_hold:.4f}", color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC (holdout) — LightGBM final")
    ax.legend()
    fig.tight_layout()
    roc_path = FIGURES_DIR / "roc_holdout_final.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tab_hold["seuil"], tab_hold["cout"], color="#C44E52")
    ax.axvline(seuil, color="k", ls="--",
               label=f"Seuil retenu (OOF) = {seuil:.2f}")
    ax.set_xlabel("Seuil de classification")
    ax.set_ylabel("Coût métier (FN=10, FP=1)")
    ax.set_title("Coût métier vs seuil (holdout) — LightGBM final")
    ax.legend()
    fig.tight_layout()
    cout_path = FIGURES_DIR / "cout_seuil_holdout_final.png"
    fig.savefig(cout_path, dpi=150)
    plt.close(fig)

    # --- Feature importance globale (gain) --------------------------------
    log("Feature importance (gain) ...")
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.booster_.feature_importance(
            importance_type="gain"),
    }).sort_values("importance", ascending=False)
    imp.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    imp.head(30).iloc[::-1].plot.barh(
        x="feature", y="importance", ax=ax, color="#4C72B0", legend=False)
    ax.set_title("Top 30 features — importance (gain) LightGBM final")
    ax.set_xlabel("Gain total")
    fig.tight_layout()
    imp_path = FIGURES_DIR / "feature_importance_globale.png"
    fig.savefig(imp_path, dpi=150)
    plt.close(fig)

    # --- MLflow : run final + registry ------------------------------------
    log("Run MLflow 'final_lightgbm' + enregistrement registry ...")
    X_sample = X_dev.head(100)
    proba_sample = model.predict_proba(X_sample)[:, 1]
    signature = infer_signature(X_sample, proba_sample)
    input_example = X_dev.head(5)

    with mlflow.start_run(run_name="final_lightgbm"):
        mlflow.set_tag("etape", "modele_final")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_param("seuil_optimal", seuil)
        mlflow.log_metrics({
            "seuil_optimal": seuil,
            "auc_oof": auc_oof,
            "cout_metier_oof": float(cout_oof),
            "holdout_auc": auc_hold,
            "holdout_accuracy": acc_hold,
            "holdout_cout_metier": float(cout_hold),
        })
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(cout_path))
        mlflow.log_artifact(str(imp_path))
        mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_NAME,
        )

    # Vérification registry
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    for v in versions:
        log(f"Registry : modèle '{v.name}' version {v.version} "
            f"(statut {v.status})")

    # --- SHAP local ---------------------------------------------------------
    log("SHAP : TreeExplainer sur 500 lignes du holdout ...")
    echant = X_hold.sample(n=500, random_state=RANDOM_STATE)
    y_echant = y_hold.loc[echant.index]
    proba_echant = model.predict_proba(echant)[:, 1]

    explainer = shap.TreeExplainer(model.booster_)
    shap_values = explainer(echant)
    # Binaire : certaines versions renvoient (n, features, 2)
    if len(shap_values.values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    fig = plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=25, show=False)
    plt.title("SHAP summary — modèle final (échantillon holdout)")
    plt.tight_layout()
    shap_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # 2 clients : un vrai défaut bien prédit (proba haute),
    #             un non-défaut bien prédit (proba basse)
    idx_defaut = echant.index[
        (y_echant == 1) & (proba_echant >= seuil)
    ]
    idx_bon = echant.index[
        (y_echant == 0) & (proba_echant < seuil)
    ]
    choix = []
    if len(idx_defaut) > 0:
        i = echant.index.get_loc(
            idx_defaut[np.argmax(proba_echant[echant.index.isin(idx_defaut)])])
        choix.append((i, "client_1_defaut_bien_predit"))
    if len(idx_bon) > 0:
        i = echant.index.get_loc(
            idx_bon[np.argmin(proba_echant[echant.index.isin(idx_bon)])])
        choix.append((i, "client_2_non_defaut_bien_predit"))

    for n, (i, etiquette) in enumerate(choix, start=1):
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(shap_values[i], max_display=20, show=False)
        plt.title(f"Explication locale SHAP — {etiquette} "
                  f"(proba={proba_echant[i]:.3f}, réel={y_echant.iloc[i]})")
        plt.tight_layout()
        p = FIGURES_DIR / f"shap_client_{n}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close("all")
        log(f"  {p.name} sauvegardé")

    # --- Artefacts JSON ------------------------------------------------------
    with open(ARTIFACTS / "seuil_optimal.json", "w", encoding="utf-8") as f:
        json.dump({
            "seuil_optimal_oof": seuil,
            "cout_metier_oof": int(cout_oof),
            "cout_metier_holdout_au_seuil_oof": int(cout_hold),
            "seuil_optimal_holdout_a_posteriori": seuil_hold,
            "cout_min_holdout_a_posteriori": int(cout_hold_min),
            "cout_fn": 10,
            "cout_fp": 1,
        }, f, indent=2, ensure_ascii=False)
    with open(ARTIFACTS / "meilleurs_params.json", "w", encoding="utf-8") as f:
        json.dump(opt, f, indent=2, ensure_ascii=False)

    log(f"TERMINÉ en {time.time() - t0:.0f}s")
    print(json.dumps({
        "auc_oof": round(auc_oof, 4),
        "seuil_optimal": seuil,
        "cout_oof": int(cout_oof),
        "holdout_auc": round(auc_hold, 4),
        "holdout_accuracy": round(acc_hold, 4),
        "holdout_cout": int(cout_hold),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
