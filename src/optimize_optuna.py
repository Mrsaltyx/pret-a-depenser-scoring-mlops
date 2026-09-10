"""Optimisation des hyperparamètres LightGBM avec Optuna — Phase 2.

- 8 essais (réduit de 10 par précaution budgétaire, documenté), 3-fold
  StratifiedKFold sur train_dev (même split 80/20 que train_models.py).
- Objectif : coût métier moyen sur les folds (pour chaque fold, le seuil
  optimal est cherché sur les prédictions de validation du fold, puis le
  coût minimal est calculé). Direction : minimize.
- Tracking MLflow : run parent "optuna_lgbm" + un run enfant par essai.

Exécution : python src/optimize_optuna.py
"""

import sys
import time
from pathlib import Path

import json

import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from business import trouver_seuil_optimal  # noqa: E402
from config import (  # noqa: E402
    ARTIFACTS,
    PROJECT_ROOT,
    RANDOM_STATE,
    TRAIN_PARQUET,
    nettoyer_noms_features,
)

N_TRIALS = 8
EXPERIMENT_NAME = "pret_a_depenser_scoring"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT.as_posix()}/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    log("Chargement train.parquet ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X.columns = nettoyer_noms_features(X.columns)
    y = df["TARGET"].astype(int)
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    ratio = float((y_dev == 0).sum() / (y_dev == 1).sum())
    log(f"train_dev : {X_dev.shape}, ratio nég/pos = {ratio:.2f}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=100),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": ratio,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        couts = []
        for tr_idx, val_idx in cv.split(X_dev, y_dev):
            X_tr = X_dev.iloc[tr_idx]
            y_tr = y_dev.iloc[tr_idx]
            X_val = X_dev.iloc[val_idx]
            y_val = y_dev.iloc[val_idx]
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            proba_val = model.predict_proba(X_val)[:, 1]
            _, cout_min, _ = trouver_seuil_optimal(y_val, proba_val)
            couts.append(cout_min)
        cout_moyen = sum(couts) / len(couts)

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}",
                              nested=True):
            mlflow.set_tag("etape", "optimisation_optuna")
            mlflow.log_params(params)
            mlflow.log_metric("cout_metier_moyen_cv", cout_moyen)
        log(f"  essai {trial.number} : coût moyen CV = {cout_moyen:,.0f}")
        return cout_moyen

    with mlflow.start_run(run_name="optuna_lgbm"):
        mlflow.set_tag("etape", "optimisation_optuna")
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", 3)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=N_TRIALS)

        mlflow.log_params({f"best_{k}": v
                           for k, v in study.best_params.items()})
        mlflow.log_metric("best_cout_metier_cv", study.best_value)

    log(f"Meilleur coût CV : {study.best_value:,.0f}")
    log(f"Meilleurs paramètres : {study.best_params}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "meilleurs_params.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": study.best_params,
            "best_cout_metier_cv": study.best_value,
            "scale_pos_weight": ratio,
            "n_trials": N_TRIALS,
        }, f, indent=2, ensure_ascii=False)
    log(f"Sauvegardé : {out}")
    log(f"TERMINÉ en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
