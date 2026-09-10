"""Chargement du modèle et inférence pour l'API « Prêt à dépenser ».

Moteur d'inférence optimisé (partie 2, étape 4) :
- chemin nominal : ONNX Runtime (CPUExecutionProvider) sur
  `models/model.onnx`, alimenté par un vecteur numpy float32 (1, 402)
  construit directement depuis la requête — sans DataFrame ni coercion
  pandas (cette plomberie représentait 84 % du temps du chemin naïf) ;
- repli (fallback) : si `models/model.onnx` est absent ou illisible,
  chargement du modèle skops et inférence bas niveau via
  `booster_.predict(num_threads=1)` sur le même vecteur numpy
  (probabilités strictement identiques à sklearn, mesuré).

Le moteur est chargé UNE SEULE FOIS au démarrage (lifespan). Les features
manquantes restent NaN : LightGBM/ONNX les gèrent nativement.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Racine du projet (dossier parent du package api/), robuste au cwd.
RACINE_PROJET = Path(__file__).resolve().parent.parent

CHEMIN_ONNX = RACINE_PROJET / "models" / "model.onnx"
CHEMIN_MODELE_SKOPS = RACINE_PROJET / "models" / "model.skops"
CHEMIN_FEATURES = RACINE_PROJET / "models" / "features.json"
CHEMIN_SEUIL = RACINE_PROJET / "models" / "seuil.json"

# Types autorisés pour le chargement skops (désérialisation sécurisée).
TRUSTED = [
    "lightgbm.basic.Booster",
    "collections.OrderedDict",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
]

# Champs métier portés par la requête (le reste vient de `features`).
CHAMPS_METIER = (
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)


class ServiceModele:
    """Service de scoring : encapsule le moteur chargé une seule fois.

    Attributs publics :
    - moteur : "onnx" ou "lightgbm_sklearn" (repli) ;
    - version_moteur : version de onnxruntime ou lightgbm selon le moteur.
    """

    #: compteur de chargements effectifs (utile aux tests et au debug).
    nb_chargements = 0

    def __init__(self) -> None:
        # Ordre exact des 402 features attendu par le modèle.
        with open(CHEMIN_FEATURES, encoding="utf-8") as f:
            self.colonnes: list[str] = json.load(f)["columns"]

        # Seuil métier (coût FN = 10 x FP = 1).
        with open(CHEMIN_SEUIL, encoding="utf-8") as f:
            self.seuil: float = float(json.load(f)["seuil_optimal_oof"])

        self.nb_features = len(self.colonnes)

        # --- Chargement du moteur : ONNX d'abord, skops en repli ---
        self._session = None  # onnxruntime.InferenceSession
        self._nom_entree: str | None = None
        self._nom_probas: str | None = None
        self._booster = None  # lightgbm.basic.Booster (repli)

        if self._charger_onnx():
            self.moteur = "onnx"
            self.chemin_modele = str(CHEMIN_ONNX)
        else:
            self._charger_repli_skops()
            self.moteur = "lightgbm_sklearn"
            self.chemin_modele = str(CHEMIN_MODELE_SKOPS)

        type(self).nb_chargements += 1

    # ------------------------------------------------------------------
    # Chargement des moteurs
    # ------------------------------------------------------------------
    def _charger_onnx(self) -> bool:
        """Tente de charger la session ONNX. Retourne True si succès."""
        if not CHEMIN_ONNX.exists():
            logger.warning(
                "models/model.onnx absent : repli sur le modèle skops."
            )
            return False
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            # Scoring unitaire : le multithreading intra-op nuit à la
            # latence d'une prédiction seule (mesuré au benchmark).
            options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(CHEMIN_ONNX),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._nom_entree = self._session.get_inputs()[0].name
            # Sortie probabilités d'un classifier zipmap=False :
            # [label (n,), probabilities (n, 2)] — on repère la sortie
            # dont la dernière dimension vaut 2 (ou nommée 'probabilities').
            self._nom_probas = None
            for sortie in self._session.get_outputs():
                forme = sortie.shape
                if sortie.name == "probabilities" or (
                    isinstance(forme, list) and len(forme) == 2 and forme[-1] == 2
                ):
                    self._nom_probas = sortie.name
                    break
            if self._nom_probas is None:
                raise RuntimeError(
                    "Sortie de probabilités introuvable dans le graphe ONNX "
                    f"(sorties: {[o.name for o in self._session.get_outputs()]})."
                )
            self.version_moteur = ort.__version__
            return True
        except Exception as exc:
            logger.warning(
                "Échec du chargement ONNX (%s) : repli sur le modèle skops.",
                exc,
            )
            self._session = None
            return False

    def _charger_repli_skops(self) -> None:
        """Charge le modèle skops et extrait le booster LightGBM."""
        import lightgbm  # noqa: F401 (version exposée ci-dessous)
        import skops.io

        modele = skops.io.load(CHEMIN_MODELE_SKOPS, trusted=TRUSTED)
        # Inférence bas niveau : booster_.predict accepte directement
        # numpy (probas strictement identiques à predict_proba, mesuré).
        self._booster = modele.booster_
        self.version_moteur = lightgbm.__version__

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def _dict_valeurs(self, requete) -> dict:
        """Dict des 402 valeurs : champs métier + features, manquantes → None."""
        valeurs = {col: None for col in self.colonnes}

        # Features engineered fournies explicitement.
        if requete.features:
            for cle, val in requete.features.items():
                if cle in valeurs:
                    valeurs[cle] = val

        # Champs métier de la requête.
        for champ in CHAMPS_METIER:
            if champ in valeurs:
                valeurs[champ] = getattr(requete, champ)

        return valeurs

    def construire_vecteur(self, requete) -> np.ndarray:
        """Vecteur numpy float32 (1, 402) dans l'ordre exact des colonnes.

        Les features non fournies sont np.nan (gérées nativement par
        LightGBM/ONNX). Aucun passage par pandas.
        """
        valeurs = self._dict_valeurs(requete)
        ligne = [
            np.nan if valeurs[col] is None else valeurs[col]
            for col in self.colonnes
        ]
        return np.array([ligne], dtype=np.float32)

    def predire(self, requete) -> tuple[float, float]:
        """Retourne (probabilité de défaut, temps d'inférence en ms)."""
        vecteur = self.construire_vecteur(requete)
        debut = time.perf_counter()
        if self._session is not None:
            # Sortie [label, probabilities (1, 2)] : proba de la classe 1.
            probas = self._session.run(
                [self._nom_probas], {self._nom_entree: vecteur}
            )[0]
            score = float(probas[0, 1])
        else:
            # Repli LightGBM bas niveau : proba de la classe positive.
            score = float(
                self._booster.predict(
                    vecteur, raw_score=False, num_threads=1
                )[0]
            )
        inference_ms = (time.perf_counter() - debut) * 1000.0
        return score, inference_ms

    def decider(self, score: float) -> str:
        """Applique le seuil métier : proba >= seuil → 'refuse'."""
        return "refuse" if score >= self.seuil else "accorde"
