# Journal des actions — Projet « Prêt à dépenser » (MLOps)

## Objectif du projet

Projet OpenClassrooms MLOps à partir du dataset Kaggle **Home Credit Default Risk** :

- Construire un **modèle de scoring crédit** qui estime la probabilité de défaut
  d'un client (variable cible `TARGET`), à partir des données de la demande de
  crédit et de l'historique de crédits (bureau, crédits précédents, POS, cartes,
  échéanciers).
- Le **coût métier est asymétrique** : un faux négatif (défaut non détecté,
  crédit accordé à un client défaillant) coûte **10 fois** un faux positif
  (bon client refusé) — constantes `COUT_FN = 10` / `COUT_FP = 1` dans
  `src/config.py`. Une métrique métier dédiée et un seuil de décision optimisé
  seront construits en phase 2.
- Démarche **MLOps complète** : tracking des expérimentations avec **MLflow**
  (tracking, registry), optimisation des hyperparamètres, interprétabilité
  (SHAP), puis mise en production (serving API + interface).

---

## Phase 1 — Préparation des données et EDA (2026-09-08)

### Actions réalisées

1. **Structure du projet** créée :
   - `src/config.py` : centralise tous les chemins (`DATA_RAW`,
     `DATA_PROCESSED`, `ARTIFACTS`, `FIGURES_DIR`) et les constantes
     (`RANDOM_STATE = 42`, `COUT_FN = 10`, `COUT_FP = 1`,
     `DAYS_EMPLOYED_ANOMALY = 365243`).
   - `src/data_preparation.py` : feature engineering complet, exécutable via
     `python src/data_preparation.py` (~36 s).
   - `src/eda.py` : génération des figures EDA (matplotlib backend `Agg`).
   - `requirements.txt` à la racine.

2. **Feature engineering** inspiré des kernels Kaggle de référence
   (« Start Here: A Gentle Introduction » de willkoehrsen et
   « LightGBM with Simple Features » de jsaguiar, URL dans les fichiers `.url`) :

   - **application_train / application_test** :
     - anomalie documentée `DAYS_EMPLOYED == 365243` remplacée par `NaN` ;
     - 5 features métier (kernel willkoehrsen) : `CREDIT_INCOME_PERCENT`,
       `ANNUITY_INCOME_PERCENT`, `CREDIT_TERM`, `DAYS_EMPLOYED_PERCENT`,
       `INCOME_PER_PERSON` ;
     - one-hot encoding (`pd.get_dummies`) des variables catégorielles puis
       alignement train/test (`align(join="inner")`), `TARGET` restaurée dans
       le train après alignement.
   - **bureau + bureau_balance** : `bureau_balance` (27 M lignes) agrégé par
     `SK_ID_BUREAU` (`MONTHS_BALANCE` : min/max/size ; `STATUS` : dummies +
     somme), fusionné dans `bureau`, puis agrégation par `SK_ID_CURR`
     (mean/max/min/sum sur les colonnes bureau d'origine, mean/max sur les
     colonnes `BB_*`, + nombre de crédits `BUREAU_CREDIT_COUNT`).
   - **previous_application** : mean/max/min sur `AMT_APPLICATION`,
     `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_DOWN_PAYMENT`, `DAYS_DECISION`,
     `CNT_PAYMENT` + comptages des statuts `NAME_CONTRACT_STATUS` (dummies) +
     `PREV_APP_COUNT`.
   - **POS_CASH_balance** : mean/max/min sur `MONTHS_BALANCE`,
     `CNT_INSTALMENT`, `CNT_INSTALMENT_FUTURE`, `SK_DPD`, `SK_DPD_DEF` +
     `POS_COUNT`.
   - **credit_card_balance** : mean/max/min/sum sur `AMT_BALANCE`,
     `AMT_CREDIT_LIMIT_ACTUAL`, `AMT_DRAWINGS_CURRENT`, `AMT_PAYMENT_CURRENT`,
     `SK_DPD`, `SK_DPD_DEF`, `CNT_DRAWINGS_CURRENT` + `CC_COUNT`.
   - **installments_payments** : création de `PAYMENT_DIFF` et `DAYS_LATE`
     (comportement de paiement), puis mean/max/min/sum sur `AMT_INSTALMENT`,
     `AMT_PAYMENT`, `PAYMENT_DIFF`, `DAYS_LATE` + `INST_COUNT`.
   - **Assemblage** : left join de toutes les agrégations sur
     `application_train` et `application_test` via `SK_ID_CURR`.

3. **Sauvegarde** des datasets traités en Parquet (pyarrow) :
   - `data/processed/train.parquet` : **307 511 lignes × 404 colonnes**
     (111,5 Mo) — dont `TARGET` ;
   - `data/processed/test.parquet` : **48 744 lignes × 403 colonnes**
     (21,5 Mo) — sans `TARGET` ;
   - soit **402 features modélisables** (hors `SK_ID_CURR` et `TARGET`).
   - Vérifications : seule différence de colonnes train/test = `TARGET` ;
     `SK_ID_CURR` unique dans les deux fichiers ; aucune valeur 365243
     résiduelle ; taux de défauts = **8,07 %** (déséquilibre confirmé).

4. **EDA** — figures sauvegardées dans `artifacts/figures/` :
   - `target_distribution.png` : déséquilibre des classes (8,1 % de défauts,
     282 686 vs 24 825) ;
   - `missing_values_top30.png` : top 30 colonnes par taux de valeurs
     manquantes (dominé par les agrégations `CC_*`, ~70-80 % de NaN, car
     peu de clients ont un historique carte de crédit — NaN structurels) ;
   - `distributions_age_credit.png` : distribution de l'âge (20-70 ans,
     pic ~35-40 ans) et du montant de crédit (asymétrique, queue longue).

### Décisions techniques et justifications

- **Pandas 3.0 / Copy-on-Write** : le premier run a révélé un
  `ChainedAssignmentError` — `df["col"].replace(..., inplace=True)` ne met
  plus à jour le DataFrame avec pandas 3.0. Correction :
  `df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(...)`. Vérifié ensuite :
  0 valeur aberrante résiduelle.
- **Nombre de features (402) légèrement au-dessus de la cible indicative
  200-350** : la table application seule produit 247 colonnes après one-hot,
  et la consigne d'agrégation (mean/max/min/sum) génère beaucoup de colonnes.
  Compromis choisi : les colonnes issues de `bureau_balance` (`BB_*`) ne sont
  agrégées qu'en mean/max au niveau client (au lieu de mean/max/min/sum),
  ce qui ramène le bloc bureau de 93 à 71 features sans perdre l'essentiel
  de l'information. Le reste suit strictement la spécification des kernels.
- **Agrégations 100 % vectorisées** (`groupby().agg()` avec agrégations
  nommées, `get_dummies` + `sum`) : aucune boucle ligne à ligne, y compris
  sur `bureau_balance` (27 M lignes). Durée totale du pipeline : ~36 s.
- **NaN structurels** conservés tels quels : les valeurs manquantes des
  blocs `CC_*` / `BUREAU_*` signifient « pas d'historique de ce type » ;
  LightGBM gère nativement les NaN, pas d'imputation à ce stade.
- **Pas de scaling** : inutile pour les modèles d'arbres (LightGBM) prévus
  en phase 2.

### Fichiers créés

| Fichier | Rôle |
|---|---|
| `src/config.py` | Chemins et constantes du projet |
| `src/data_preparation.py` | Feature engineering → parquets |
| `src/eda.py` | Figures EDA |
| `data/processed/train.parquet` | Train traité (307 511 × 404) |
| `data/processed/test.parquet` | Test traité (48 744 × 403) |
| `artifacts/figures/target_distribution.png` | Déséquilibre des classes |
| `artifacts/figures/missing_values_top30.png` | Valeurs manquantes |
| `artifacts/figures/distributions_age_credit.png` | Distributions âge / crédit |
| `requirements.txt` | Dépendances Python |
| `JOURNAL_ACTIONS.md` | Ce journal |

---

## Phase 2 — Modélisation et MLOps (2026-09-08)

### Protocole expérimental

- Split stratifié **80 % train_dev (246 008 lignes) / 20 % holdout
  (61 503 lignes)** (`random_state=42`). Les validations croisées se font
  uniquement sur train_dev ; le holdout sert exclusivement à l'évaluation
  finale honnête.
- **Coût métier** : `cout = 10 × FN + 1 × FP` (`src/business.py`).
  Le **seuil de classification est optimisé** par balayage 0.01 → 0.99
  (pas 0.01) sur les probabilités OOF — jamais 0.5.
- **Déséquilibre de classes** (8,07 % de défauts) géré par pondération :
  `class_weight='balanced'` (logreg), `balanced_subsample` (RF),
  `scale_pos_weight = 11,39` (LightGBM).
- **Tracking MLflow** : `sqlite:///mlflow.db` à la racine, expérience
  « pret_a_depenser_scoring ». Chaque run logge paramètres, métriques
  (OOF + holdout), tags (`etape`) et figures en artifacts.

### Comparaison de modèles (src/train_models.py)

Probas OOF via `cross_val_predict` (StratifiedKFold 5) sur train_dev, puis
réentraînement sur tout train_dev et évaluation holdout :

| Modèle | AUC OOF | Coût OOF | Seuil | AUC holdout | Coût holdout |
|---|---|---|---|---|---|
| Dummy (baseline) | 0,5000 | 198 600 | 0,09 | 0,5000 | 49 650 |
| LogisticRegression (balanced) | 0,7669 | 126 464 | 0,53 | 0,7709 | 31 661 |
| RandomForest (60 arbres) | 0,7355 | 137 122 | 0,35 | 0,7409 | 34 063 |
| **LightGBM (scale_pos_weight)** | **0,7820** | **121 965** | **0,47** | **0,7880** | **30 145** |

- LightGBM l'emporte sur toutes les métriques → retenu pour la suite.
- **AUC < 0,82** : pas de signal de surapprentissage selon la consigne.
- Les pipelines sklearn non-LGBM incluent un `SimpleImputer(median)`
  (LightGBM gère les NaN nativement).
- Résultats complets : `artifacts/comparaison_modeles.csv` ; courbes ROC OOF
  et coût-vs-seuil par modèle dans `artifacts/figures/`.

### Optimisation des hyperparamètres (Optuna, src/optimize_optuna.py)

- **8 essais** (réduit de 10 par précaution budgétaire), 3-fold
  StratifiedKFold sur train_dev ; objectif = **coût métier moyen** (seuil
  optimal recherché sur les prédictions de validation de chaque fold),
  direction `minimize`. Durée : 300 s.
- Run MLflow parent `optuna_lgbm` + un run enfant par essai.
- Meilleur coût CV : **40 448**. Meilleurs hyperparamètres
  (`artifacts/meilleurs_params.json`) :
  - `num_leaves = 91`, `learning_rate = 0,0174`, `n_estimators = 800`,
    `min_child_samples = 160`, `subsample = 0,976`,
    `colsample_bytree = 0,958`, `reg_alpha = 0,246`, `reg_lambda = 4,870`,
    `scale_pos_weight = 11,39` (fixe).

### Modèle final (src/finalize_model.py)

- LGBMClassifier réentraîné avec les meilleurs paramètres sur **tout
  train_dev** ; seuil optimal déterminé sur probas OOF 5-fold.
- Résultats :
  - OOF : **AUC = 0,7852**, **seuil optimal = 0,45**, coût = 120 791 ;
  - Holdout (au seuil OOF 0,45) : **AUC = 0,7900**, **accuracy = 0,7390**,
    **coût métier = 29 591** (vs 30 145 avant optimisation, −1,8 % ; le
    meilleur seuil a posteriori sur holdout serait 0,48 pour 29 514 —
    écart faible, pas de fuite : le seuil 0,45 est retenu).
  - Run MLflow `final_lightgbm` (tag `etape = modele_final`), seuil loggé
    en métrique ET en paramètre ; courbes `roc_holdout_final.png` et
    `cout_seuil_holdout_final.png` en artifacts.
- `artifacts/seuil_optimal.json` sauvegardé.

### Tracking MLflow et registry

- **Model registry** : modèle enregistré sous
  `credit_scoring_pret_a_depenser`, **version 1 (statut READY)**, avec
  signature (`infer_signature` sur 100 lignes + probas) et exemple d'entrée
  (5 lignes). Vérifié via `MlflowClient.search_model_versions`.
- **UI MLflow** — commande (vérifiée : HTTP 200) :
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```
  puis ouvrir http://127.0.0.1:5000 (expérience « pret_a_depenser_scoring »).

### Interprétabilité

- **Globale** : importance par gain du booster LightGBM —
  `artifacts/figures/feature_importance_globale.png` (top 30) +
  `artifacts/feature_importance.csv`. Dominée par `EXT_SOURCE_3/2/1`
  (scores externes), puis `CREDIT_TERM`, `DAYS_EMPLOYED`, `DAYS_BIRTH`.
- **Locale (SHAP)** : `shap.TreeExplainer` sur `model.booster_`, échantillon
  de 500 lignes du holdout :
  - `shap_summary.png` (beeswarm, 25 features) — valeurs élevées de
    `EXT_SOURCE_*` poussent vers le non-défaut, cohérent métier ;
  - `shap_client_1.png` : vrai défaut bien prédit (explication individuelle) ;
  - `shap_client_2.png` : non-défaut bien prédit.

### Déploiement / serving

- Entrée de test : `artifacts/serving_input.json` (3 vraies lignes du
  holdout, format `dataframe_split`, NaN → null).
- **Serving HTTP — fonctionne sous Windows** (testé et arrêté proprement) :
  ```bash
  mlflow models serve -m "models:/credit_scoring_pret_a_depenser/1" -p 5002 --no-conda
  curl -X POST http://127.0.0.1:5002/invocations \
      -H "Content-Type: application/json" \
      --data-binary @artifacts/serving_input.json
  # -> {"predictions": [0, 0, 1]}   (GET /health -> 200)
  ```
- **Repli batch** (également testé, même résultat [0, 0, 1]) :
  ```bash
  mlflow models predict -m "models:/credit_scoring_pret_a_depenser/1" \
      -i artifacts/serving_input.json -o artifacts/serving_output.json \
      --env-manager local
  ```
  (script `src/test_serving.py` fourni pour rejouer ce repli).
- Note : l'endpoint renvoie la **classe** (seuil sklearn 0,5 côté pyfunc) ;
  pour appliquer le seuil métier 0,45 en production, exposer les probabilités
  (`predict_proba`) et comparer à `artifacts/seuil_optimal.json`.

### Problèmes rencontrés et contournements (phase 2)

1. **LightGBM refusait les noms de features** issus du one-hot (espaces,
   virgules, guillemets — « Do not support special JSON characters in
   feature name »). Correctif : `nettoyer_noms_features()` dans
   `src/config.py` (regex `[^A-Za-z0-9_] -> _`) appliqué au chargement dans
   tous les scripts de modélisation et au fichier de serving. Comparaison
   relancée (~4 min perdues).
2. **RandomForest réduit à 60 estimateurs** (au lieu de 100) par précaution
   budgétaire : 77 s pour 6 fits, impact négligeable sur un modèle de toute
   façon dominé par LightGBM.
3. **Optuna réduit à 8 essais** (au lieu de 10) pour tenir le budget temps.
4. `pkill` absent de Git Bash : arrêt des serveurs via
   `netstat -ano` + `taskkill //PID <pid> //F`.

### Fichiers créés en phase 2

| Fichier | Rôle |
|---|---|
| `src/business.py` | Coût métier + recherche du seuil optimal |
| `src/train_models.py` | Comparaison 4 modèles + tracking MLflow |
| `src/optimize_optuna.py` | Optimisation Optuna LightGBM |
| `src/finalize_model.py` | Modèle final, registry, SHAP |
| `src/test_serving.py` | Repli batch de serving documenté |
| `mlflow.db` + `mlartifacts/` | Backend MLflow (runs, registry, artifacts) |
| `artifacts/comparaison_modeles.csv` | Tableau comparatif des modèles |
| `artifacts/meilleurs_params.json` | Meilleurs hyperparamètres Optuna |
| `artifacts/seuil_optimal.json` | Seuil optimal + coûts associés |
| `artifacts/feature_importance.csv` | Importances (gain) du modèle final |
| `artifacts/serving_input.json` / `serving_output.json` | Test de serving |
| `artifacts/figures/roc_oof_*.png`, `cout_seuil_*.png` | Courbes par modèle |
| `artifacts/figures/roc_holdout_final.png`, `cout_seuil_holdout_final.png` | Évaluation finale |
| `artifacts/figures/feature_importance_globale.png` | Importance globale |
| `artifacts/figures/shap_summary.png`, `shap_client_1.png`, `shap_client_2.png` | Interprétabilité SHAP |
| `README.md` | Documentation du projet |
