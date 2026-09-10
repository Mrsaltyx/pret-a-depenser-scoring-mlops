# Analyse de data drift — « Prêt à dépenser »

**Date de l'analyse** : 2026-09-10 (données de production simulées le même jour, entre 15:29 et 15:30 UTC)
**Outil** : Evidently 0.7.21 (API nouvelle génération : `Report` + `DataDriftPreset`)
**Livrables associés** : `drift_report.html`, `drift_report_periodes.html`, `drift_metrics.json`

---

## 1. Méthodologie

### Données comparées

| Jeu de données | Source | Volume |
|---|---|---|
| **Référence** | `data/processed/train.parquet` (échantillon `random_state=42`) | 5 000 lignes |
| **Production globale** | `monitoring/data/prod_logs.db`, requêtes `http_status=200` uniquement | 1 900 requêtes |
| dont période 1 (clients « normaux ») | ids d'insertion 1 à 1 200 | 1 140 requêtes valides |
| dont période 2 (dérive artificielle) | ids d'insertion 1 201 à 2 000 | 760 requêtes valides |

Les entrées de production sont re-parsées depuis la colonne `inputs_json` du
journal SQLite : les champs métier de premier niveau et le dict `features`
sont aplatis en colonnes ; les features absentes d'une requête restent `NaN`
(comme côté modèle). Les noms de colonnes du parquet sont assainis
(`[^A-Za-z0-9_]` → `_`) pour correspondre à `models/features.json`.

### Périmètre et méthodes statistiques

L'analyse porte sur les **30 features les plus importantes**
(`artifacts/feature_importance.csv`) **auxquelles s'ajoutent les 5 features
volontairement shiftées** (AMT_INCOME_TOTAL n'entre pas dans le top 30 mais
est nécessaire à la validation), soit **31 features**, plus la colonne
**score** (dérive de la sortie du modèle) pour la comparaison entre périodes.

Evidently choisit automatiquement le test statistique selon le type et la
taille des données :

- **Distance de Wasserstein normée** (features numériques à ≥ 1 000
  observations) : dérive si distance ≥ 0,1 ;
- **Test de Kolmogorov-Smirnov** (p-value, features numériques plus petites
  ou binaires) : dérive si p-value < 0,05 ;
- **Distance de Jensen-Shannon** (features catégorielles) : dérive si
  distance ≥ 0,1.

Deux rapports sont produits :

1. `drift_report.html` — **référence (train) vs production globale** ;
2. `drift_report_periodes.html` — **période 1 vs période 2**, score du
   modèle inclus : c'est cette comparaison qui isole la dérive injectée,
   les deux périodes provenant de la même population d'échantillonnage.

## 2. Résultats — référence vs production globale

**10 features sur 31 en dérive** (toutes par distance de Wasserstein normée,
seuil 0,1) :

| Feature | Drift score | Lecture |
|---|---|---|
| DAYS_EMPLOYED | **28,94** | Artefact attendu : les `NaN` du parquet brut sont convertis en valeur sentinelle 365243 par le simulateur (conforme au contrat de l'API), ce qui modifie fortement la distribution par rapport au train brut. |
| CREDIT_TERM | 0,566 | Forte dérive, conséquence directe du ×1,40 sur AMT_CREDIT en période 2 (CREDIT_TERM = annuité / crédit). |
| AMT_GOODS_PRICE | 0,222 | Corrélée à AMT_CREDIT, dérive en cascade. |
| EXT_SOURCE_3 | 0,202 | Dérive injectée (−0,05) — retrouvée. |
| EXT_SOURCE_2 | 0,148 | Dérive injectée (−0,08) — retrouvée. |
| INST_AMT_PAYMENT_SUM | 0,132 | Écart structurel train/test, non injecté. |
| AMT_ANNUITY | 0,130 | Corrélée à AMT_CREDIT. |
| DAYS_ID_PUBLISH | 0,119 | Écart structurel train/test, non injecté. |
| BUREAU_DAYS_CREDIT_ENDDATE_MAX | 0,111 | Écart structurel train/test, non injecté. |
| INST_DAYS_LATE_SUM | 0,120 | Écart structurel train/test, non injecté. |

La comparaison référence vs production mélange dérive injectée et écarts
structurels train/test : elle est utile en exploitation réelle, mais c'est la
comparaison entre périodes qui valide le système.

## 3. Validation du système — la dérive injectée est retrouvée

La simulation a appliqué en période 2 une dérive artificielle documentée
(`simulate_traffic.py`). La comparaison période 1 vs période 2 détecte
**6 colonnes sur 32 en dérive**, et **retrouve les 5 features shiftées** :

| Feature shiftée | Transformation injectée | Méthode Evidently | Drift score | Détectée ? |
|---|---|---|---|---|
| AMT_CREDIT | × 1,40 | Wasserstein normée | 0,456 | ✅ |
| AMT_INCOME_TOTAL | × 0,85 | Wasserstein normée | 0,302 | ✅ |
| EXT_SOURCE_2 | −0,08 (clip [0,1]) | Wasserstein normée | 0,436 | ✅ |
| EXT_SOURCE_3 | −0,05 (clip [0,1]) | K-S p-value | 0,026 | ✅ |
| DAYS_BIRTH | +1 500 jours | Wasserstein normée | 0,310 | ✅ |
| **score (sortie modèle)** | conséquence des 5 shifts | Wasserstein normée | 0,217 | ✅ |

Le système de détection est donc **validé de bout en bout** : les données
loguées en production permettent de retrouver exactement les dérives
injectées, y compris la dérive de la sortie du modèle (score).

## 4. Métriques opérationnelles (base de production)

| Indicateur | Valeur |
|---|---|
| Requêtes totales | 2 000 |
| Succès (HTTP 200) | 1 900 |
| Erreurs de validation (HTTP 422) | 100 (**5,0 %**, taux injecté volontairement) |
| Latence API moyenne (loguée) | 0,32 ms |
| Latence API p95 (loguée) | 0,44 ms |
| Latence client moyenne (mesurée côté simulateur) | 5,2 ms |
| Latence client p95 | 5,7 ms |
| Score moyen | 0,3364 |
| Dossiers refusés (score ≥ 0,45) | 579 (30,5 %) |

## 5. Points de vigilance pour la production

- **Seuil métier 0,45** : la dérive du score (Wasserstein 0,217 entre
  périodes) montre que la distribution des sorties bouge quand les entrées
  bougent. Le taux de refus (30,5 % ici) est l'indicateur métier le plus
  simple à surveiller au quotidien : une dérive durable de ce taux doit
  déclencher une investigation avant même l'analyse fine des features.
- **Déclencheurs de réentraînement (retraining triggers) proposés** :
  1. plus de 20 % des features surveillées en dérive sur une fenêtre
     glissante (ici 10/31, seuil déjà atteint → investigation) ;
  2. dérive du score (sortie modèle) détectée sur la fenêtre courante ;
  3. hausse anormale du taux d'erreur 422 (> 10 % sur une journée) signe
     d'un changement de contrat côté appelant ;
  4. dégradation du taux de défaut réel observé a posteriori (quand la
     vérité terrain arrive, typiquement à échéance des crédits).
- **Qualité des entrées** : la dérive extrême de DAYS_EMPLOYED (28,9) est un
  artefact du remplissage des valeurs manquantes en 365243. En production,
  tracer explicitement le taux de valeurs sentinelles par feature plutôt que
  la distribution brute.
- **Cadence recommandée** : analyse Evidently hebdomadaire sur fenêtre
  glissante de 7 jours + alerte quotidienne sur les KPIs du dashboard.

## 6. RGPD et stockage

- Les données journalisées (`prod_logs.db` + `api_logs.jsonl`) contiennent
  des **données à caractère personnel financières** (revenus, montants de
  crédit, âge en jours) : le dossier `monitoring/data/` est exclu du
  versioning (`.gitignore`) et doit rester sur un stockage chiffré, à accès
  restreint (principe de minimisation et limitation d'accès, RGPD art. 5
  et 32).
- **Durée de conservation** à définir avec le DPO : les logs détaillés
  (inputs complets) ne devraient être conservés que le temps nécessaire au
  debugging et à l'analyse de drift (par ex. 90 jours), puis agrégés ou
  anonymisés ; seuls les agrégats (drift scores, KPIs) sont conservés long
  terme.
- Pas d'identifiant client (`SK_ID_CURR`) dans le payload API : le simulateur
  l'exclut volontairement, ce qui limite la ré-identification. Conserver
  cette règle en production (pseudonymisation par conception).
- Toute fuite des logs exposerait les profils financiers des demandeurs :
  journaliser les accès à la base et chiffrer les sauvegardes.

## 7. Rejouer l'analyse

```bash
python -m uvicorn api.main:app --port 8002          # terminal 1
python monitoring/simulate_traffic.py --n 2000      # terminal 2
python monitoring/drift_analysis.py                 # régénère HTML + JSON
streamlit run monitoring/dashboard.py               # dashboard de monitoring
```
