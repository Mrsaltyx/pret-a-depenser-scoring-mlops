"""Test de serving MLflow — Projet « Prêt à dépenser ».

Script de documentation/rejeu des commandes de serving. Le test effectif a
été réalisé en ligne de commande (voir JOURNAL_ACTIONS.md, Phase 2) :

1) Serving HTTP (peut échouer sous Windows selon l'environnement) :
   mlflow models serve -m "models:/credit_scoring_pret_a_depenser/1" \
       -p 5002 --no-conda
   curl -X POST http://127.0.0.1:5002/invocations \
       -H "Content-Type: application/json" \
       -d @artifacts/serving_input.json

2) Repli batch (fonctionne partout, sans serveur) :
   mlflow models predict -m "models:/credit_scoring_pret_a_depenser/1" \
       -i artifacts/serving_input.json \
       -o artifacts/serving_output.json --env-manager local

Ce script exécute le REPLI batch (chemin garanti) et affiche les prédictions.
Exécution : python src/test_serving.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, PROJECT_ROOT  # noqa: E402


def main() -> None:
    entree = ARTIFACTS / "serving_input.json"
    sortie = ARTIFACTS / "serving_output.json"
    cmd = [
        "mlflow", "models", "predict",
        "-m", "models:/credit_scoring_pret_a_depenser/1",
        "-i", str(entree),
        "-o", str(sortie),
        "--env-manager", "local",
    ]
    print("Commande :", " ".join(cmd))
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                         text=True, timeout=600)
    print("code retour :", res.returncode)
    if res.returncode != 0:
        print("STDERR :", res.stderr[-2000:])
        return
    with open(sortie, encoding="utf-8") as f:
        predictions = json.load(f)
    print("Prédictions :", predictions)


if __name__ == "__main__":
    main()
