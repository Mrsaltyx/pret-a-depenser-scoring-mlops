# Script de soutenance — Prêt à Dépenser · MLOps Partie 2

**Format : 30 min au total** — Présentation 15 min (slides 1→14 + 2 démos) · Discussion avec « Chloé » 10 min · Débrief 5 min.

**Avant la soutenance, ouvrir et tester :**

```bash
cd "C:\Users\Salty\Downloads\Initiez_vous_MLOPS _2"
python -m uvicorn api.main:app --port 8000     # terminal 1 : l'API tourne
streamlit run monitoring/dashboard.py           # terminal 2 : le dashboard (optionnel)
```
Vérifier http://127.0.0.1:8000/docs et l'onglet **Actions** du dépôt GitHub ouvert dans le navigateur.

---

## Partie 1 — Présentation des livrables (15 min)

### Slide 1 — Titre (30 s)
> « Bonjour Chloé. Suite à ton message, j'ai piloté la mise en production du modèle de scoring validé lors de la phase précédente. Je te présente le livrable : une API conteneurisée, testée, monitorée, et optimisée. »

### Slide 2 — Contexte (1 min)
> « Rappel de la mission : le département Crédit Express veut traiter les demandes en quasi temps réel. Point de départ : le LightGBM versionné dans MLflow, avec son seuil métier 0,45 — un faux négatif coûte 10 fois un faux positif. Quatre objectifs : exposer, industrialiser, monitorer, optimiser. »

### Slide 3 — Le modèle déployé (1 min)
> « Ce que l'on déploie : AUC 0,79 sur le holdout, coût métier 29 591 — le meilleur des quatre modèles comparés. 402 features engineered, NaN gérés nativement. Important : la décision (accord/refus) est prise côté API avec le seuil 0,45, pas côté client. Le modèle n'est pas réentraîné : on industrialise le modèle validé. »

### Slide 4 — Architecture (1 min 30)
> « Vue d'ensemble. Le client Crédit Express envoie du JSON à l'API FastAPI ; le moteur ONNX — avec repli LightGBM automatique — renvoie score et décision. Chaque appel, succès ou erreur, est loggé dans SQLite avec miroir JSONL ; Evidently analyse la dérive depuis cette base, et le dashboard Streamlit la visualise. En haut : la chaîne CI/CD GitHub Actions qui teste, construit l'image Docker et déploie à chaque push sur main. »

### Slide 5 — API (1 min 30)
> « Contrat d'entrée strict : les champs métier sont validés par Pydantic — plages de valeurs, types — et toute entrée invalide renvoie un 422 explicite. Les 402 features engineered passent par un dictionnaire optionnel ; une clé inconnue est rejetée. Point crucial de la consigne : le modèle est chargé **une seule fois au démarrage**, via le lifespan FastAPI. »

**DÉMO 1 — API en direct (1 min)** : dans http://127.0.0.1:8000/docs, « Try it out » sur `/predict`, envoyer le payload d'exemple → montrer `score`, `decision`, `inference_ms`. Puis envoyer un payload invalide (ex. `AMT_INCOME_TOTAL: 0`) → montrer le **422** et son détail.

### Slide 6 — Tests (1 min)
> « 21 tests automatisés, tous verts en 2,8 secondes. Trois couches : nominal, validation — les cas critiques de la consigne : champ manquant, type incorrect, revenu nul, âge de 5 ans, EXT_SOURCE hors [0,1] — et le moteur : chargement unique, parité ONNX/sklearn, repli, écriture des logs. La gestion d'erreurs couvre le 422, le 500 loggé, et le repli si le modèle ONNX est indisponible. »

### Slide 7 — CI/CD (1 min 30)
> « Trois jobs enchaînés : les tests d'abord ; l'image Docker n'est construite que si les tests passent ; le déploiement — conteneur lancé avec smoke tests réels sur /health et /predict — uniquement sur main. Aucun secret dans le code : les credentials de registry iraient en secrets GitHub. »

**DÉMO 2 — Pipeline en direct (1 min)** : onglet Actions du dépôt → montrer le run déclenché par le dernier push, dérouler les 3 jobs. (Si tu veux le rejouer live : `git commit --allow-empty -m "demo: declenche le pipeline" && git push`, puis rafraîchir l'onglet Actions.)

### Slide 8 — Stockage production (1 min)
> « Chaque appel est traçable : horodatage, request_id, inputs complets, score, décision, latence, statut HTTP. Double écriture SQLite + JSONL. Les erreurs 422 sont aussi loggées — c'est ce qui permet de surveiller la qualité des données envoyées. Et les inputs stockés rendent l'analyse de dérive rejouable. Côté RGPD : aucune identité client stockée, seulement les variables de scoring. »

### Slide 9 — Simulation (1 min)
> « Pour valider le dispositif sans attendre des semaines de production : 2 000 vraies requêtes HTTP contre l'API, issues du jeu de test. 5 % d'erreurs 422 volontaires. Et surtout : une deuxième période avec une dérive artificielle documentée — crédit plus élevé, revenus plus faibles, clients plus jeunes. Latence API mesurée : 0,32 ms en moyenne. »

### Slide 10 — Data drift (1 min 30)
> « Résultat clé : le système **retrouve les 5 dérives injectées**, plus la dérive du score lui-même — 6 colonnes en alerte sur 32 suivies. Si la détection n'avait pas fonctionné sur une dérive connue, elle serait inopérante en vraie production. Point de vigilance honnête : DAYS_EMPLOYED apparaît en dérive technique à cause de la sentinelle 365243 — identifié et documenté. Le signal d'action proposé : dérive persistante du score → ré-entraînement et recalibration du seuil. »

### Slide 11 — Dashboard (45 s)
> « Tout est consultable dans le dashboard Streamlit, alimenté directement par la base SQLite de production : distribution des scores avec le seuil 0,45, latence, décisions, taux d'erreur et table des dérives. Un choix de lecture à noter : le taux d'erreur est agrégé **par paquets de requêtes** plutôt qu'en moyenne glissante — avec 5 % d'erreurs injectées de façon régulière, la moyenne glissante était une ligne plate par construction, inexploitable. Par paquets, la cible à 5 % est lisible, tout dépassement passe en rouge, et le pic final s'explique : ce sont mes propres tests manuels Swagger avec payloads volontairement invalides — le monitoring les distingue visuellement d'un régime normal. » (Montrer le dashboard live si ouvert au terminal 2.)

### Slide 12 — Optimisation (1 min 30)
> « Le profiling cProfile a donné un résultat contre-intuitif : le goulot n'était **pas le modèle**, mais la manipulation pandas — 84 % du temps. J'ai donc benchmarké quatre chemins : du chemin naïf à 31 ms jusqu'à ONNX Runtime à 0,016 ms — ×201. Avec vérification de non-régression : écart max des probabilités de 2×10⁻⁷, zéro décision divergente. La version ONNX est **intégrée à l'API et déployée via le pipeline** : ~0,8 ms de bout en bout à chaud, ×9. Et le modèle est 32 % plus léger. »

### Slide 13 — Dépôt GitHub (45 s)
> « Le dépôt public est structuré : api, tests, monitoring, optimization, models, Dockerfile, workflow CI/CD, README qui explique comment lancer l'API et lire le monitoring. L'historique de commits retrace la construction — et le .gitignore garantit qu'aucune donnée n'est commitée. » (Naviguer rapidement dans le dépôt en live.)

### Slide 14 — Bilan (30 s)
> « Pour conclure : l'API est Docker Ready, testée, observable, et neuf fois plus rapide qu'une intégration naïve. Les limites sont connues et documentées : déploiement cloud réel, authentification, alerting sur la dérive. Je te laisse avec tes questions, Chloé. »

---

## Partie 2 — Discussion avec Chloé (10 min) : questions probables et réponses

### Robustesse et fiabilité

**« Que se passe-t-il si l'entrée est malformée ? »**
Pydantic valide types et plages → 422 avec le détail du champ fautif, et l'erreur est loggée en base (elle alimente le taux d'erreur du monitoring). Erreur interne → 500 loggé. Les 11 tests de validation couvrent les cas critiques : champ manquant, texte au lieu d'un nombre, revenu nul, âge impossible, EXT_SOURCE > 1, clé de feature inconnue.

**« Et si le modèle ne charge pas au démarrage ? »**
Le modèle ONNX est embarqué dans l'image ; s'il est illisible, l'API bascule automatiquement sur le modèle LightGBM/skops (warning tracé, `/info` expose le moteur actif). Le healthcheck Docker sur `/health` empêche un conteneur cassé de recevoir du trafic.

**« Le modèle est-il rechargé à chaque requête ? »**
Non — chargé une fois au démarrage via le lifespan FastAPI, réutilisé ensuite. C'est testé explicitement (identité d'objet entre requêtes).

### Monitoring et maintenance

**« Comment détectes-tu la dérive en continu ? »**
Référence = données d'entraînement ; courant = inputs rejoués depuis la base de logs. Evidently compare les distributions (Wasserstein pour les numériques) sur les ~30 features les plus importantes + le score. Le rapport HTML + métriques JSON alimentent le dashboard. La preuve que ça marche : les 5 dérives artificielles injectées ont toutes été retrouvées.

**« Pourquoi ton taux d'erreur est-il plat puis grimpe à la fin ? »**
Ce n'est pas un incident : la base contient 2 000 requêtes simulées avec 5 % d'erreurs 422 injectées de façon strictement régulière — d'où la stabilité — puis 16 requêtes de tests manuels Swagger, dont plus de la moitié volontairement invalides, qui forment le pic final. C'est exactement pour rendre cette distinction visible que le graphique agrège par paquets de requêtes avec une cible à 5 % : une moyenne glissante lissait tout en une ligne plate inexploitable. Axe d'amélioration identifié : ajouter un champ `source` (simulation / manuel / production) dans la table de logs pour filtrer proprement.

**« Que fais-tu quand une dérive est détectée ? »**
Politique proposée : alerte si dérive persistante du score sur plusieurs fenêtres → réévaluation du modèle sur données récentes, recalibration du seuil 0,45, ré-entraînement si nécessaire via le pipeline de la partie 1. L'artefact DAYS_EMPLOYED montre qu'il faut distinguer dérive technique et dérive métier avant de réagir.

**« RGPD ? »**
La base ne stocke que les variables du scoring — pas de nom ni d'identifiant client. Le PoC est 100 % local, aucun coût cloud.

### Optimisation et scalabilité

**« Pourquoi ONNX plutôt que X ? »**
Le profiling a montré que le goulot était pandas (84 % du temps), pas le modèle. ONNX Runtime CPU + construction numpy directe : ×201 sur l'inférence, modèle 32 % plus léger, sans dépendance GPU — compatible avec l'image slim. La parité des prédictions a été vérifiée (2×10⁻⁷, 0 décision divergente). Le repli LightGBM reste disponible.

**« Et sous forte charge ? »**
À 0,8 ms par requête, un seul worker traite >1 000 req/s théoriques ; la simulation a fait 2 000 requêtes en 11,6 s (~170 req/s en séquentiel client). Pour aller plus loin : uvicorn multi-workers, tests de charge (Locust/k6), et déploiement cloud avec autoscaling — listés en prochaines étapes.

**« Pourquoi SQLite et pas PostgreSQL/Elasticsearch ? »**
Choix assumé pour un PoC local sans coût : SQLite suffit pour tracer et rejouer, le schéma (11 colonnes) est pensé pour migrer tel quel vers PostgreSQL — c'est le chemin documenté en prochaine étape.

---

## Partie 3 — Débrief (5 min)

Points forts à rappeler si demandé : chaîne complète de bout en bout (API → CI/CD → monitoring → optimisation), détection de dérive **validée sur vérité terrain**, optimisation mesurée et intégrée, hygiène Git.

Axes d'amélioration assumés : déploiement cloud réel, authentification, alerting automatisé, champ `source` (simulation / manuel / production) dans les logs pour filtrer le taux d'erreur, AUC du modèle perfectible (partie 1).
