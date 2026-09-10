"""Configuration centrale du projet « Prêt à dépenser » (scoring crédit).

Chemins et constantes partagés entre les scripts du projet.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
# Racine du projet = parent du dossier src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Données brutes (attention au '+' dans le nom du dossier)
DATA_RAW = PROJECT_ROOT / "Projet+Mise+en+prod+-+home-credit-default-risk"

# Données traitées (parquet)
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Artefacts (figures, modèles, rapports MLflow locaux, ...)
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS / "figures"

# Fichiers sources
APPLICATION_TRAIN = DATA_RAW / "application_train.csv"
APPLICATION_TEST = DATA_RAW / "application_test.csv"
BUREAU = DATA_RAW / "bureau.csv"
BUREAU_BALANCE = DATA_RAW / "bureau_balance.csv"
PREVIOUS_APPLICATION = DATA_RAW / "previous_application.csv"
POS_CASH_BALANCE = DATA_RAW / "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE = DATA_RAW / "credit_card_balance.csv"
INSTALLMENTS_PAYMENTS = DATA_RAW / "installments_payments.csv"

# Fichiers produits
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
TEST_PARQUET = DATA_PROCESSED / "test.parquet"

# ---------------------------------------------------------------------------
# Constantes métier / techniques
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Coût métier : un faux négatif (défaut non détecté) coûte 10x
# un faux positif (bon client refusé). Utilisé en phase 2 pour le
# choix du seuil de décision et la métrique métier.
COUT_FN = 10
COUT_FP = 1

# Valeur aberrante documentée de DAYS_EMPLOYED (anomalie du dataset)
DAYS_EMPLOYED_ANOMALY = 365243


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nettoyer_noms_features(columns) -> list[str]:
    """Remplace les caractères spéciaux des noms de colonnes par '_'.

    Nécessaire pour LightGBM, qui refuse les caractères JSON spéciaux
    (espaces, virgules, guillemets...) présents dans les noms issus du
    one-hot encoding (ex. « NAME_CONTRACT_TYPE_Cash loans »).
    """
    import re

    return [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in columns]
