"""Journalisation structurée de chaque appel à l'API « Prêt à dépenser ».

Double écriture :
- SQLite `monitoring/data/prod_logs.db` (table `predictions`) ;
- miroir JSONL `monitoring/data/api_logs.jsonl` (une ligne JSON par appel).

Les succès ET les erreurs (validation 422, erreur serveur 500) sont logués.
Le dossier et la table sont créés à la volée si absents.

Pour les tests, le dossier de stockage peut être redirigé via la variable
d'environnement `PAD_MONITORING_DIR` (lue à chaque appel, donc compatible
avec monkeypatch et tmp_path).
"""

import json
import os
import sqlite3
import threading
from pathlib import Path

# Racine du projet (dossier parent du package api/), robuste au cwd.
RACINE_PROJET = Path(__file__).resolve().parent.parent

ENV_MONITORING_DIR = "PAD_MONITORING_DIR"

# Verrou global : sqlite3 stdlib n'est pas thread-safe en écriture
# concurrente avec une connexion partagée ; on ouvre une connexion courte
# par écriture, protégée par ce verrou.
_LOCK = threading.Lock()


def _dossier_monitoring() -> Path:
    """Dossier de stockage des logs (redirigeable par variable d'env)."""
    brut = os.environ.get(ENV_MONITORING_DIR)
    dossier = Path(brut) if brut else RACINE_PROJET / "monitoring" / "data"
    dossier.mkdir(parents=True, exist_ok=True)
    return dossier


def _chemin_db() -> Path:
    return _dossier_monitoring() / "prod_logs.db"


def _chemin_jsonl() -> Path:
    return _dossier_monitoring() / "api_logs.jsonl"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    request_id TEXT NOT NULL,
    http_status INTEGER NOT NULL,
    score REAL,
    decision TEXT,
    seuil REAL,
    latence_ms REAL,
    inference_ms REAL,
    erreur TEXT,
    inputs_json TEXT
)
"""


def init_stockage() -> None:
    """Crée le dossier et la table si absents (idempotent)."""
    with _LOCK:
        with sqlite3.connect(_chemin_db()) as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()


def log_appel(
    *,
    timestamp_utc: str,
    request_id: str,
    http_status: int,
    score: float | None = None,
    decision: str | None = None,
    seuil: float | None = None,
    latence_ms: float | None = None,
    inference_ms: float | None = None,
    erreur: str | None = None,
    inputs: dict | None = None,
) -> None:
    """Écrit un appel dans SQLite et dans le miroir JSONL.

    Toute erreur d'écriture est interceptée : la journalisation ne doit
    jamais faire échouer une requête métier.
    """
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str) if inputs is not None else None
    enregistrement = {
        "timestamp_utc": timestamp_utc,
        "request_id": request_id,
        "http_status": http_status,
        "score": score,
        "decision": decision,
        "seuil": seuil,
        "latence_ms": latence_ms,
        "inference_ms": inference_ms,
        "erreur": erreur,
        "inputs_json": inputs_json,
    }
    try:
        with _LOCK:
            with sqlite3.connect(_chemin_db()) as conn:
                conn.execute(_SCHEMA_SQL)
                conn.execute(
                    """
                    INSERT INTO predictions (
                        timestamp_utc, request_id, http_status, score,
                        decision, seuil, latence_ms, inference_ms,
                        erreur, inputs_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp_utc,
                        request_id,
                        http_status,
                        score,
                        decision,
                        seuil,
                        latence_ms,
                        inference_ms,
                        erreur,
                        inputs_json,
                    ),
                )
                conn.commit()
            with open(_chemin_jsonl(), "a", encoding="utf-8") as f:
                f.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
    except Exception:
        # La journalisation ne doit jamais casser le service.
        pass
