# Prêt à dépenser — Scoring crédit & MLOps (Partie 2 : mise en production)

Projet OpenClassrooms MLOps. La **Partie 1** (initiation : feature engineering,
comparaison de modèles, optimisation Optuna, tracking MLflow + registry,
seuil métier optimisé) a produit un **LightGBM final** (AUC holdout 0,7900,
coût métier 29 591, seuil 0,45 pour un coût FN=10 × FP=1).

La **Partie 2** (ce dépôt) prend ce modèle versionné et le met en production :
**API FastAPI conteneurisée (Docker), tests automatisés, pipeline CI/CD,
stockage des données de production, analyse de data drift (Evidently),
dashboard de monitoring (Streamlit) et optimisation d'inférence (ONNX)**.

## Démarrage rapide

**Windows — le plus simple** : double-cliquer sur `lancer_projet.bat`
(menu : API + dashboard, tests, simulation, analyse de drift, MLflow, arrêt).

Sinon, manuellement :

```bash
pip install -r requirements.txt

# Lancer l'API (charge le modèle ONNX une seule fois au démarrage)
python -m uvicorn api.main:app --port 8000
# Documentation interactive Swagger : http://127.0.0.1:8000/docs
```

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 22000,
  "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -2500, "CNT_CHILDREN": 1,
  "EXT_SOURCE_1": 0.55, "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.50,
  "features": {"REGION_RATING_CLIENT": 2}
}'
# -> {"score": 0.31..., "decision": "accorde", "seuil": 0.45, "inference_ms": 0.6, ...}
```

- `GET /health` — santé du service ; `GET /info` — version du modèle, moteur
  d'inférence actif (`onnx` ou repli `lightgbm_sklearn`), seuil, nb de features.
- Les 402 features engineered ne sont pas toutes obligatoires : les 9 champs
  métier sont validés (plages, types) et les autres features passent par le
  dict optionnel `features` (clés inconnues rejetées → 422). Les features
  absentes restent NaN, gérés nativement par LightGBM/ONNX.

## Docker

```bash
docker build -t pret-a-depenser-api .
docker run -p 8000:8000 -v ${PWD}/monitoring/data:/app/monitoring/data pret-a-depenser-api
```

Le `Dockerfile` (python:3.12-slim) n'embarque que `api/`, `models/` et
`requirements-api.txt` ; healthcheck intégré sur `/health`.

## Tests et CI/CD

```bash
python -m pytest tests/ -v    # 21 tests : API, validation (422), moteur, parité ONNX
```

Le pipeline **GitHub Actions** (`.github/workflows/ci-cd.yml`) s'exécute sur
push/PR sur `main` : **test** (pytest) → **docker-build** (image de l'API) →
**deploy** (main uniquement : conteneur lancé + smoke tests `/health` et
`/predict`). Les credentials de registry sont attendus en secrets GitHub.

## Monitoring et data drift

L'API **logge chaque appel** (succès et erreurs 422) de façon structurée :
SQLite `monitoring/data/prod_logs.db` (table `predictions` : timestamp,
request_id, http_status, score, decision, latence/inference_ms, inputs_json)
+ miroir JSONL `monitoring/data/api_logs.jsonl`.

```bash
python monitoring/simulate_traffic.py      # simule 2000 requêtes (dont 5 % d'erreurs, dérive injectée en période 2)
python monitoring/drift_analysis.py        # rapport Evidently -> drift_report.html + drift_metrics.json
streamlit run monitoring/dashboard.py      # dashboard : distribution scores, latence, erreurs, drift
```

- **`monitoring/ANALYSE_DRIFT.md`** — étude de dérive : référence (données
  d'entraînement) vs production, les 5 features artificiellement dérivées sont
  toutes détectées (validation du système), points de vigilance.
- **`monitoring/dashboard.py`** — KPIs (taux d'erreur, latence p95, % refusés),
  distribution des scores avec seuil 0,45, latence dans le temps, table des
  features en dérive.
- **`docs/screenshots/`** — captures de la solution de stockage (schéma de la
  table, extrait de lignes, aperçu monitoring).

## Optimisation de l'inférence (étape 4)

Voir **`optimization/RAPPORT_OPTIMISATION.md`** (chiffres mesurés) :

| Chemin (1 prédiction) | Latence médiane |
|---|---|
| API naïve (DataFrame + coercion pandas) | 31,4 ms |
| predict_proba sklearn seul | 3,1 ms |
| **ONNX Runtime + numpy direct (retenu, intégré à l'API)** | **0,016 ms** |

Parité vérifiée (max |diff| = 2,3e-7, 0 décision divergente au seuil 0,45).
Repli automatique LightGBM si le modèle ONNX est indisponible. Rejouable via
`python optimization/profile_inference.py`.

## Structure du dépôt

```
├── api/                    # API FastAPI (schemas Pydantic, moteur ONNX, storage SQLite/JSONL)
├── tests/                  # 21 tests pytest (API, validation, modèle, parité)
├── monitoring/             # simulation trafic, analyse drift Evidently, dashboard Streamlit
├── optimization/           # profiling cProfile, benchmarks, conversion ONNX, rapport
├── models/                 # model.onnx + model.skops + seuil.json + features.json
├── src/                    # code Partie 1 (feature engineering, entraînement, MLflow)
├── artifacts/              # résultats Partie 1 (comparaison, seuil, figures EDA/SHAP)
├── docs/screenshots/       # captures de la solution de stockage
├── Dockerfile + .dockerignore
├── .github/workflows/ci-cd.yml
├── requirements.txt        # environnement complet
├── requirements-api.txt    # deps minimales de serving (Docker)
└── JOURNAL_ACTIONS.md      # journal détaillé des deux parties
```

Les données (`data/`, CSV bruts), `mlruns/`, `mlflow.db` et les logs de
production sont exclus de Git (`.gitignore`) — aucune donnée sensible commitée.

## Partie 1 (rappel)

Pipeline complet rejouable (nécessite de restaurer les CSV Kaggle bruts) :

```bash
python src/data_preparation.py && python src/eda.py && python src/train_models.py \
  && python src/optimize_optuna.py && python src/finalize_model.py && python src/test_serving.py
```

Modèle final : LightGBM optimisé Optuna, registry MLflow
`credit_scoring_pret_a_depenser` v1 (UI : double-cliquer `lancer_mlflow_ui.bat`).
