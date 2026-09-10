"""Coût métier et seuil optimal — Projet « Prêt à dépenser ».

Coût asymétrique : un faux négatif (défaut non détecté) coûte COUT_FN fois
un faux positif (bon client refusé). Le seuil de classification est donc
optimisé sur ce coût, et non fixé à 0.5.
"""

import numpy as np
import pandas as pd

from config import COUT_FN, COUT_FP


def cout_metier(y_true, y_pred, cout_fn: int = COUT_FN, cout_fp: int = COUT_FP) -> int:
    """Coût métier total = cout_fn * FN + cout_fp * FP."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return cout_fn * fn + cout_fp * fp


def trouver_seuil_optimal(
    y_true, y_proba, cout_fn: int = COUT_FN, cout_fp: int = COUT_FP
) -> tuple[float, float, pd.DataFrame]:
    """Balaye les seuils 0.01 -> 0.99 (pas 0.01) et renvoie
    (seuil_optimal, cout_min, tableau seuils/coûts)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    seuils = np.round(np.arange(0.01, 1.00, 0.01), 2)
    couts = [
        cout_metier(y_true, (y_proba >= s).astype(int), cout_fn, cout_fp)
        for s in seuils
    ]
    tableau = pd.DataFrame({"seuil": seuils, "cout": couts})
    best = tableau.loc[tableau["cout"].idxmin()]
    return float(best["seuil"]), float(best["cout"]), tableau
