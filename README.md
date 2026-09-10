# Prêt à dépenser — Scoring crédit & MLOps

Projet OpenClassrooms MLOps : construire et industrialiser un **modèle de
scoring crédit** pour la société financière « Prêt à dépenser », à partir du
dataset Kaggle **Home Credit Default Risk**. Le modèle estime la probabilité
de défaut d'un client (cible `TARGET`, 8,07 % de défauts — classes
déséquilibrées).

**Particularité métier** : le coût d'erreur est **asymétrique** — un faux
négatif (crédit accordé à un client défaillant) coûte **10 ×** un faux
positif (bon client refusé). Le seuil de classification est donc **optimisé**
sur ce coût (et non fixé à 0,5).

La démarche MLOps couvre : feature engineering, EDA, comparaison de modèles,
optimisation d'hyperparamètres (Optuna), **tracking MLflow + model registry**,
interprétabilité (importance globale + SHAP local) et **test de serving**.

## Structure du projet

```
Initiez_vous_MLOPS/
├── Projet+Mise+en+prod+-+home-credit-default-risk/   # données brutes (CSV Kaggle)
├── data/processed/            # train.parquet / test.parquet (feature engineering)
├── src/
│   ├── config.py              # chemins, constantes (COUT_FN=10, COUT_FP=1)
│   ├── data_preparation.py    # feature engineering -> parquets
│   ├── eda.py                 # figures d'analyse exploratoire
│   ├── business.py            # coût métier + seuil optimal
│   ├── train_models.py        # comparaison 4 modèles + tracking MLflow
│   ├── optimize_optuna.py     # optimisation hyperparamètres LightGBM
│   ├── finalize_model.py      # modèle final, registry, SHAP
│   └── test_serving.py        # test de serving (repli batch)
├── artifacts/
│   ├── figures/               # EDA, ROC, coût-vs-seuil, importance, SHAP
│   ├── comparaison_modeles.csv
│   ├── meilleurs_params.json / seuil_optimal.json
│   └── serving_input.json / serving_output.json
├── mlflow.db + mlartifacts/   # backend MLflow (créés à l'exécution)
├── JOURNAL_ACTIONS.md         # journal détaillé des phases 1 et 2
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Ordre d'exécution

```bash
python src/data_preparation.py   # 1. feature engineering (~40 s)
python src/eda.py                # 2. figures EDA
python src/train_models.py       # 3. comparaison de modèles (~4 min)
python src/optimize_optuna.py    # 4. optimisation Optuna (~5 min)
python src/finalize_model.py     # 5. modèle final + registry + SHAP (~3 min)
python src/test_serving.py       # 6. test de serving (repli batch)
```

## Résultats principaux

Comparaison (probas OOF 5-fold sur train_dev 80 %, évaluation sur holdout
20 % stratifié) — coût métier = 10 × FN + 1 × FP :

| Modèle | AUC OOF | Coût OOF | Seuil optimal | AUC holdout | Coût holdout |
|---|---|---|---|---|---|
| Dummy (baseline) | 0,5000 | 198 600 | 0,09 | 0,5000 | 49 650 |
| Régression logistique | 0,7669 | 126 464 | 0,53 | 0,7709 | 31 661 |
| RandomForest | 0,7355 | 137 122 | 0,35 | 0,7409 | 34 063 |
| **LightGBM optimisé (final)** | **0,7852** | **120 791** | **0,45** | **0,7900** | **29 591** |

Modèle final : LightGBM optimisé par Optuna (8 essais, objectif = coût
métier en CV) — `num_leaves=91`, `learning_rate=0,0174`, `n_estimators=800`,
`min_child_samples=160`, `subsample=0,976`, `colsample_bytree=0,958`,
`reg_alpha=0,246`, `reg_lambda=4,87`, `scale_pos_weight=11,39`.
Enregistré dans le model registry MLflow :
`credit_scoring_pret_a_depenser` **version 1**.

## MLflow : UI et serving

Interface de suivi des expérimentations :

- **Le plus simple : double-cliquer sur `lancer_mlflow_ui.bat`** (ouvre le
  navigateur automatiquement sur http://127.0.0.1:5000).
- En ligne de commande, avec le Python qui contient MLflow (environnement
  géré Kimi) :

```bash
"C:\Users\Salty\AppData\Roaming\kimi-desktop\daimon-share\daimon\runtime\python\.venv\Scripts\python.exe" \
    -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# http://127.0.0.1:5000 — expérience « pret_a_depenser_scoring »
```

⚠️ Le Python système (`C:\Python314`) ne contient **pas** MLflow — utiliser
le `.bat` ou le chemin complet ci-dessus.

Serving HTTP du modèle enregistré :

```bash
mlflow models serve -m "models:/credit_scoring_pret_a_depenser/1" -p 5002 --no-conda
curl -X POST http://127.0.0.1:5002/invocations \
    -H "Content-Type: application/json" \
    --data-binary @artifacts/serving_input.json
# -> {"predictions": [0, 0, 1]}
```

Repli batch (sans serveur) :

```bash
mlflow models predict -m "models:/credit_scoring_pret_a_depenser/1" \
    -i artifacts/serving_input.json -o artifacts/serving_output.json \
    --env-manager local
```

⚠️ L'endpoint pyfunc renvoie la classe au seuil 0,5. Pour la décision
métier, utiliser les probabilités et le seuil optimisé (0,45, voir
`artifacts/seuil_optimal.json`).

## Limites et pistes d'amélioration

- **AUC ~0,79** : correct mais perfectible — kernels Kaggle de référence
  atteignent ~0,80+ avec plus d'essais Optuna, du feature engineering
  supplémentaire (agrégations temporelles, ratios croisés) et de
  l'early stopping.
- Le seuil optimal (0,45) est estimé sur OOF train_dev ; le seuil a
  posteriori sur holdout (0,48) est proche — stabilité correcte, à
  surveiller en production (data drift).
- Les NaN structurels (clients sans historique carte/POS) sont laissés à
  LightGBM ; tester des indicateurs « a un historique X ».
- Serving testé localement ; industrialisation à prévoir : API dédiée
  (FastAPI) exposant `predict_proba` + seuil métier, tests unitaires,
  conteneurisation, CI/CD, monitoring de dérive.
- Coût métier supposé constant (FN=10, FP=1) ; à recalibrer avec les
  équipes métier et à intégrer dans une surveillance continue.
