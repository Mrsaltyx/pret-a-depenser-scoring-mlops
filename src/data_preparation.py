"""Préparation des données — Projet « Prêt à dépenser ».

Feature engineering inspiré des kernels Kaggle de référence :
- « Start Here: A Gentle Introduction » (willkoehrsen) : features métier
  sur application_train/test, one-hot encoding, alignement train/test.
- « LightGBM with Simple Features » (jsaguiar) : agrégations vectorisées
  des tables annexes (bureau, bureau_balance, previous_application,
  POS_CASH_balance, credit_card_balance, installments_payments).

Produit :
- data/processed/train.parquet  (307 511 lignes, avec TARGET)
- data/processed/test.parquet   (48 744 lignes, sans TARGET)

Exécution : python src/data_preparation.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    APPLICATION_TEST,
    APPLICATION_TRAIN,
    BUREAU,
    BUREAU_BALANCE,
    CREDIT_CARD_BALANCE,
    DATA_PROCESSED,
    DAYS_EMPLOYED_ANOMALY,
    INSTALLMENTS_PAYMENTS,
    POS_CASH_BALANCE,
    PREVIOUS_APPLICATION,
    TEST_PARQUET,
    TRAIN_PARQUET,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. application_train / application_test
# ---------------------------------------------------------------------------
def build_application() -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Chargement application_train / application_test ...")
    train = pd.read_csv(APPLICATION_TRAIN)
    test = pd.read_csv(APPLICATION_TEST)
    log(f"  application_train : {train.shape}, application_test : {test.shape}")

    for df in (train, test):
        # Anomalie documentée : DAYS_EMPLOYED == 365243 -> NaN
        # (pas d'inplace : Copy-on-Write de pandas 3.0 l'ignorerait)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY, np.nan
        )

        # Features métier (kernel willkoehrsen)
        df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
        df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # One-hot encoding des variables catégorielles + alignement train/test
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    train, test = train.align(test, join="inner", axis=1)
    # align() peut supprimer TARGET si absente du test : on la restaure
    if "TARGET" not in train.columns:
        target = pd.read_csv(APPLICATION_TRAIN, usecols=["SK_ID_CURR", "TARGET"])
        train = train.merge(target, on="SK_ID_CURR", how="left")
    log(f"  Après one-hot + align : train {train.shape}, test {test.shape}")
    return train, test


# ---------------------------------------------------------------------------
# 2. bureau.csv + bureau_balance.csv
# ---------------------------------------------------------------------------
def build_bureau_features() -> pd.DataFrame:
    log("Traitement bureau_balance (27M lignes) ...")
    bb = pd.read_csv(BUREAU_BALANCE)
    # Encodage simple du statut : dummies puis somme par crédit
    bb = pd.get_dummies(bb, columns=["STATUS"], prefix="BB_STATUS")
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_BALANCE_MIN=("MONTHS_BALANCE", "min"),
        BB_MONTHS_BALANCE_MAX=("MONTHS_BALANCE", "max"),
        BB_MONTHS_BALANCE_SIZE=("MONTHS_BALANCE", "size"),
        **{
            f"{c}_SUM": (c, "sum")
            for c in bb.columns
            if c.startswith("BB_STATUS_")
        },
    )
    bb_agg.columns = [c.upper() for c in bb_agg.columns]
    log(f"  bureau_balance agrégé par SK_ID_BUREAU : {bb_agg.shape}")

    log("Chargement bureau.csv et merge avec bureau_balance ...")
    bureau = pd.read_csv(BUREAU)
    bureau = bureau.merge(
        bb_agg, how="left", left_on="SK_ID_BUREAU", right_index=True
    )

    num_cols = bureau.select_dtypes(include=[np.number]).columns.tolist()
    for key in ("SK_ID_CURR", "SK_ID_BUREAU"):
        if key in num_cols:
            num_cols.remove(key)

    # Pour rester proche de l'objectif ~200-350 features finales :
    # - colonnes bureau d'origine : mean/max/min/sum (spec kernel jsaguiar)
    # - colonnes issues de bureau_balance (BB_*) : mean/max seulement
    agg_spec = {}
    for c in num_cols:
        funcs = ["mean", "max"] if c.upper().startswith("BB_") else [
            "mean", "max", "min", "sum"
        ]
        for f in funcs:
            agg_spec[f"BUREAU_{c.upper()}_{f.upper()}"] = (c, f)

    log("  Agrégation bureau par SK_ID_CURR ...")
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"), **agg_spec
    )
    log(f"  Features bureau : {bureau_agg.shape}")
    return bureau_agg


# ---------------------------------------------------------------------------
# 3. previous_application.csv
# ---------------------------------------------------------------------------
def build_previous_features() -> pd.DataFrame:
    log("Traitement previous_application ...")
    prev = pd.read_csv(PREVIOUS_APPLICATION)
    # Comptages des statuts de contrat
    prev = pd.get_dummies(
        prev, columns=["NAME_CONTRACT_STATUS"], prefix="PREV_STATUS"
    )
    status_cols = [c for c in prev.columns if c.startswith("PREV_STATUS_")]

    num_agg_cols = [
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
    ]
    agg_spec = {}
    for c in num_agg_cols:
        for f in ("mean", "max", "min"):
            agg_spec[f"PREV_{c}_{f.upper()}"] = (c, f)
    for c in status_cols:
        agg_spec[f"{c}_SUM"] = (c, "sum")
    agg_spec["PREV_APP_COUNT"] = ("SK_ID_PREV", "count")

    prev_agg = prev.groupby("SK_ID_CURR").agg(**agg_spec)
    log(f"  Features previous_application : {prev_agg.shape}")
    return prev_agg


# ---------------------------------------------------------------------------
# 4. POS_CASH_balance.csv
# ---------------------------------------------------------------------------
def build_pos_features() -> pd.DataFrame:
    log("Traitement POS_CASH_balance ...")
    pos = pd.read_csv(POS_CASH_BALANCE)
    agg_spec = {}
    for c in [
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]:
        for f in ("mean", "max", "min"):
            agg_spec[f"POS_{c}_{f.upper()}"] = (c, f)
    agg_spec["POS_COUNT"] = ("SK_ID_PREV", "count")

    pos_agg = pos.groupby("SK_ID_CURR").agg(**agg_spec)
    log(f"  Features POS_CASH : {pos_agg.shape}")
    return pos_agg


# ---------------------------------------------------------------------------
# 5. credit_card_balance.csv
# ---------------------------------------------------------------------------
def build_cc_features() -> pd.DataFrame:
    log("Traitement credit_card_balance ...")
    cc = pd.read_csv(CREDIT_CARD_BALANCE)
    agg_spec = {}
    for c in [
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_CURRENT",
        "SK_DPD",
        "SK_DPD_DEF",
        "CNT_DRAWINGS_CURRENT",
    ]:
        for f in ("mean", "max", "min", "sum"):
            agg_spec[f"CC_{c}_{f.upper()}"] = (c, f)
    agg_spec["CC_COUNT"] = ("SK_ID_PREV", "count")

    cc_agg = cc.groupby("SK_ID_CURR").agg(**agg_spec)
    log(f"  Features credit_card : {cc_agg.shape}")
    return cc_agg


# ---------------------------------------------------------------------------
# 6. installments_payments.csv
# ---------------------------------------------------------------------------
def build_installments_features() -> pd.DataFrame:
    log("Traitement installments_payments ...")
    inst = pd.read_csv(INSTALLMENTS_PAYMENTS)
    # Features de comportement de paiement (kernel jsaguiar)
    inst["PAYMENT_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    agg_spec = {}
    for c in ["AMT_INSTALMENT", "AMT_PAYMENT", "PAYMENT_DIFF", "DAYS_LATE"]:
        for f in ("mean", "max", "min", "sum"):
            agg_spec[f"INST_{c}_{f.upper()}"] = (c, f)
    agg_spec["INST_COUNT"] = ("SK_ID_PREV", "count")

    inst_agg = inst.groupby("SK_ID_CURR").agg(**agg_spec)
    log(f"  Features installments : {inst_agg.shape}")
    return inst_agg


# ---------------------------------------------------------------------------
# 7. Assemblage final
# ---------------------------------------------------------------------------
def merge_all(base: pd.DataFrame, features: list[pd.DataFrame]) -> pd.DataFrame:
    out = base
    for feat in features:
        out = out.merge(feat, how="left", on="SK_ID_CURR")
    return out


def main() -> None:
    t0 = time.time()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    train, test = build_application()

    features = [
        build_bureau_features(),
        build_previous_features(),
        build_pos_features(),
        build_cc_features(),
        build_installments_features(),
    ]

    log("Merge final des agrégations sur application (train & test) ...")
    train = merge_all(train, features)
    test = merge_all(test, features)

    # Cohérence des dtypes pour parquet
    for df in (train, test):
        df.columns = df.columns.astype(str)

    log(f"Sauvegarde {TRAIN_PARQUET} ...")
    train.to_parquet(TRAIN_PARQUET, engine="pyarrow", index=False)
    log(f"Sauvegarde {TEST_PARQUET} ...")
    test.to_parquet(TEST_PARQUET, engine="pyarrow", index=False)

    log(f"TERMINÉ en {time.time() - t0:.1f}s")
    log(f"  train.parquet : {train.shape}")
    log(f"  test.parquet  : {test.shape}")


if __name__ == "__main__":
    main()
