"""Analyse et optimisation des performances d'inférence — « Prêt à dépenser ».

Étape 4 de la partie 2 du projet MLOps OpenClassrooms :
profilage cProfile du chemin d'inférence, benchmarks de latence
(sklearn predict_proba vs LightGBM bas niveau vs ONNX Runtime),
effet du parallélisme (n_jobs / num_threads), latence par taille de
batch, décomposition du temps par étape, empreinte mémoire et
vérification de parité des prédictions.

Exécution : python optimization/profile_inference.py  (depuis la racine du projet)

Sorties :
  - optimization/cprofile_report.txt  : top 20 des fonctions (temps cumulé)
  - optimization/benchmarks.json      : tous les chiffres mesurés
  - optimization/model.onnx           : modèle converti au format ONNX
  - optimization/figures/*.png        : graphiques de synthèse
"""

import cProfile
import gc
import json
import os
import platform
import pstats
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import skops.io

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parent.parent
MODELE_SKOPS = RACINE / "models" / "model.skops"
FEATURES_JSON = RACINE / "models" / "features.json"
TEST_PARQUET = RACINE / "data" / "processed" / "test.parquet"
SORTIE = RACINE / "optimization"
FIGURES = SORTIE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TRUSTED = [
    "lightgbm.basic.Booster",
    "collections.OrderedDict",
    "lightgbm.sklearn.LGBMRegressor",
    "lightgbm.sklearn.LGBMClassifier",
]

SEUIL = 0.45          # seuil métier de décision de crédit
WARMUP = 10
N_REP = 50            # répétitions standard (médiane + p95)
N_REP_GROS = 20       # répétitions réduites pour les très gros batches
N_PROFIL = 200        # itérations du profiling cProfile

resultats = {"machine": {
    "systeme": platform.platform(),
    "processeur": platform.processor(),
    "coeurs_physiques": psutil.cpu_count(logical=False),
    "coeurs_logiques": psutil.cpu_count(logical=True),
    "ram_totale_go": round(psutil.virtual_memory().total / 1e9, 1),
    "python": platform.python_version(),
}}


def bench(fn, n_rep=N_REP, warmup=WARMUP):
    """Chronomètre fn : retourne médiane / p95 / moyenne / min / max (ms)."""
    for _ in range(warmup):
        fn()
    temps = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        temps.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(temps)
    return {
        "mediane_ms": round(float(np.median(a)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "moyenne_ms": round(float(a.mean()), 4),
        "min_ms": round(float(a.min()), 4),
        "max_ms": round(float(a.max()), 4),
        "n_repetitions": n_rep,
    }


def nettoyer_noms(columns):
    """Même règle de nettoyage que src/config.py (caractères spéciaux -> '_')."""
    return [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in columns]


# ---------------------------------------------------------------------------
# 1. Chargement des données et du modèle (+ empreinte mémoire)
# ---------------------------------------------------------------------------
print("Chargement des données de test ...")
with open(FEATURES_JSON, encoding="utf-8") as f:
    COLS = json.load(f)["columns"]

df_brut = pd.read_parquet(TEST_PARQUET)
df_brut.columns = nettoyer_noms(df_brut.columns)
X = df_brut[COLS].apply(pd.to_numeric, errors="coerce")
X_np32 = X.to_numpy(dtype=np.float32)
print(f"  Données : {X.shape[0]} lignes x {X.shape[1]} features")

proc = psutil.Process(os.getpid())
gc.collect()
rss_avant = proc.memory_info().rss
print("Chargement du modèle skops ...")
model = skops.io.load(MODELE_SKOPS, trusted=TRUSTED)
gc.collect()
rss_apres = proc.memory_info().rss

taille_skops_mo = os.path.getsize(MODELE_SKOPS) / 1e6
resultats["modele"] = {
    "type": type(model).__name__,
    "n_estimators": int(model.n_estimators),
    "n_features": int(model.n_features_in_),
    "taille_fichier_skops_mo": round(taille_skops_mo, 3),
    "rss_avant_chargement_mo": round(rss_avant / 1e6, 1),
    "rss_apres_chargement_mo": round(rss_apres / 1e6, 1),
    "delta_rss_chargement_mo": round((rss_apres - rss_avant) / 1e6, 1),
}
print(f"  Modèle : {model.n_estimators} arbres, {model.n_features_in_} features, "
      f"fichier {taille_skops_mo:.1f} Mo, RSS +{(rss_apres - rss_avant) / 1e6:.0f} Mo")

# Ligne témoin brute (dict) simulant une requête JSON reçue par l'API
row_dict = df_brut.iloc[0][COLS].to_dict()
X1 = X.iloc[[0]]            # DataFrame 1 ligne déjà coercé
np1 = X_np32[:1]            # même ligne en numpy float32


def chemin_api():
    """Chemin complet côté API : dict JSON -> DataFrame -> coercion -> predict."""
    df1 = pd.DataFrame([row_dict])
    df1 = df1.reindex(columns=COLS)
    df1 = df1.apply(pd.to_numeric, errors="coerce")
    return model.predict_proba(df1)


def chemin_api_optimise():
    """Chemin optimisé : dict JSON -> numpy float32 -> booster.predict."""
    arr = np.array([[row_dict.get(c, np.nan) for c in COLS]], dtype=np.float32)
    return model.booster_.predict(arr, raw_score=False)


# ---------------------------------------------------------------------------
# 2. Profiling cProfile du chemin d'inférence
# ---------------------------------------------------------------------------
print(f"Profiling cProfile ({N_PROFIL} itérations du chemin API) ...")
profiler = cProfile.Profile()
t0 = time.perf_counter()
profiler.enable()
for _ in range(N_PROFIL):
    chemin_api()
profiler.disable()
t_profil = time.perf_counter() - t0

rapport_txt = SORTIE / "cprofile_report.txt"
with open(rapport_txt, "w", encoding="utf-8") as f:
    f.write("PROFILING cProfile — chemin d'inférence API « Prêt à dépenser »\n")
    f.write(f"Chemin profilé : construction DataFrame 1 ligne (dict JSON) "
            f"+ coercion pandas + predict_proba\n")
    f.write(f"Itérations : {N_PROFIL} | Temps total : {t_profil:.3f} s "
            f"| Temps moyen par appel : {t_profil / N_PROFIL * 1000:.3f} ms\n")
    f.write("=" * 78 + "\n\n")
    f.write("Top 20 des fonctions par temps CUMULÉ :\n\n")
    stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
    stats.print_stats(20)
    f.write("\n" + "=" * 78 + "\n\n")
    f.write("Top 20 des fonctions par temps PROPRE (tottime) :\n\n")
    stats.sort_stats("tottime").print_stats(20)

    # Extraction des goulots pour le JSON et le rapport Markdown
    # (directement depuis les stats, sans re-parser le texte)
    stats_json = pstats.Stats(profiler)
    goulots = []
    for func, (cc, nc, tt, ct, _callers) in stats_json.stats.items():
        goulots.append({
            "fonction": pstats.func_std_string(func),
            "nb_appels": nc,
            "temps_propre_s": round(tt, 4),
            "temps_cumule_s": round(ct, 4),
            "pct_temps_total": round(100 * ct / t_profil, 1),
        })
    goulots.sort(key=lambda g: g["temps_cumule_s"], reverse=True)
    f.write("\n" + "=" * 78 + "\n\n")
    f.write("GOULOTS D'ÉTRANGLEMENT IDENTIFIÉS :\n\n")
    for g in goulots[:8]:
        f.write(f"  - {g['fonction']} : {g['pct_temps_total']} % du temps cumulé "
                f"({g['nb_appels']} appels)\n")

resultats["profiling"] = {
    "nb_iterations": N_PROFIL,
    "temps_total_s": round(t_profil, 3),
    "temps_moyen_par_appel_ms": round(t_profil / N_PROFIL * 1000, 3),
    "top_fonctions_temps_cumule": goulots[:10],
    "fichier": "cprofile_report.txt",
}
print(f"  Temps moyen par appel profilé : {t_profil / N_PROFIL * 1000:.2f} ms "
      f"(surcharge cProfile incluse)")


# ---------------------------------------------------------------------------
# 3. Latence 1 ligne : sklearn predict_proba vs booster.predict (bas niveau)
# ---------------------------------------------------------------------------
print("Benchmark latence 1 ligne : sklearn vs booster ...")
res_1l = {}
res_1l["sklearn_predict_proba"] = bench(lambda: model.predict_proba(X1))
res_1l["booster_predict"] = bench(
    lambda: model.booster_.predict(np1, raw_score=False))

# Parité sklearn vs booster sur 1000 lignes
proba_sk = model.predict_proba(X.iloc[:1000])[:, 1]
proba_booster = model.booster_.predict(X_np32[:1000], raw_score=False)
diff_booster = float(np.max(np.abs(proba_sk - proba_booster)))
resultats["latence_1_ligne"] = res_1l
resultats["parite_sklearn_vs_booster"] = {
    "n_lignes": 1000,
    "max_abs_diff": diff_booster,
    "identique": bool(diff_booster < 1e-12),
}
print(f"  sklearn : {res_1l['sklearn_predict_proba']['mediane_ms']:.3f} ms | "
      f"booster : {res_1l['booster_predict']['mediane_ms']:.3f} ms | "
      f"max|diff| = {diff_booster:.2e}")


# ---------------------------------------------------------------------------
# 4. Effet du parallélisme (n_jobs sklearn / num_threads booster)
# ---------------------------------------------------------------------------
print("Benchmark effet n_jobs / num_threads (batches 1, 100, 1000) ...")
res_threads = {}
for n in (1, 100, 1000):
    nrep = N_REP
    Xn = X.iloc[:n]
    npn = X_np32[:n]
    model.set_params(n_jobs=1)
    sk_njobs1 = bench(lambda: model.predict_proba(Xn), n_rep=nrep)
    model.set_params(n_jobs=-1)
    sk_njobs_m1 = bench(lambda: model.predict_proba(Xn), n_rep=nrep)
    model.set_params(n_jobs=-1)  # configuration finale retenue par défaut
    b_t1 = bench(lambda: model.booster_.predict(npn, raw_score=False,
                                                num_threads=1), n_rep=nrep)
    b_tm1 = bench(lambda: model.booster_.predict(npn, raw_score=False,
                                                 num_threads=-1), n_rep=nrep)
    res_threads[str(n)] = {
        "sklearn_n_jobs_1": sk_njobs1,
        "sklearn_n_jobs_-1": sk_njobs_m1,
        "booster_num_threads_1": b_t1,
        "booster_num_threads_-1": b_tm1,
    }
    print(f"  batch {n:>5} : sklearn nj1 {sk_njobs1['mediane_ms']:.3f} ms / "
          f"nj-1 {sk_njobs_m1['mediane_ms']:.3f} ms | "
          f"booster t1 {b_t1['mediane_ms']:.3f} ms / t-1 {b_tm1['mediane_ms']:.3f} ms")
resultats["effet_parallelisme"] = res_threads


# ---------------------------------------------------------------------------
# 5. Latence par taille de batch
# ---------------------------------------------------------------------------
print("Benchmark latence par taille de batch (1, 10, 100, 1000, 10000) ...")
res_batch = {}
for n in (1, 10, 100, 1000, 10000):
    nrep = N_REP if n <= 1000 else N_REP_GROS
    Xn = X.iloc[:n]
    npn = X_np32[:n]
    sk = bench(lambda: model.predict_proba(Xn), n_rep=nrep)
    bo = bench(lambda: model.booster_.predict(npn, raw_score=False), n_rep=nrep)
    res_batch[str(n)] = {
        "n_repetitions": nrep,
        "sklearn_total": sk,
        "sklearn_par_ligne_ms": round(sk["mediane_ms"] / n, 4),
        "booster_total": bo,
        "booster_par_ligne_ms": round(bo["mediane_ms"] / n, 4),
    }
    print(f"  batch {n:>6} : sklearn {sk['mediane_ms']:8.3f} ms "
          f"({sk['mediane_ms']/n:.4f} ms/lig) | "
          f"booster {bo['mediane_ms']:8.3f} ms ({bo['mediane_ms']/n:.4f} ms/lig)")
resultats["latence_par_batch"] = res_batch


# ---------------------------------------------------------------------------
# 6. Décomposition du temps par étape (chemin 1 ligne)
# ---------------------------------------------------------------------------
print("Décomposition du temps par étape (1 ligne) ...")
nrep_dec = 200
etapes = {}
etapes["construction_dataframe"] = bench(
    lambda: pd.DataFrame([row_dict]).reindex(columns=COLS),
    n_rep=nrep_dec, warmup=20)
_df1 = pd.DataFrame([row_dict]).reindex(columns=COLS)
etapes["coercion_pandas"] = bench(
    lambda: _df1.apply(pd.to_numeric, errors="coerce"),
    n_rep=nrep_dec, warmup=20)
etapes["predict_proba_seul"] = bench(
    lambda: model.predict_proba(X1), n_rep=nrep_dec, warmup=20)
etapes["chemin_api_complet"] = bench(chemin_api, n_rep=nrep_dec, warmup=20)
etapes["construction_numpy_optimisee"] = bench(
    lambda: np.array([[row_dict.get(c, np.nan) for c in COLS]],
                     dtype=np.float32),
    n_rep=nrep_dec, warmup=20)
etapes["chemin_api_optimise_complet"] = bench(
    chemin_api_optimise, n_rep=nrep_dec, warmup=20)
resultats["decomposition_1_ligne"] = etapes
for k, v in etapes.items():
    print(f"  {k:32s} : {v['mediane_ms']:.3f} ms (médiane)")


# ---------------------------------------------------------------------------
# 7. Conversion ONNX + benchmark + parité
# ---------------------------------------------------------------------------
print("Conversion ONNX ...")
onnx_ok = False
onnx_info = {}
try:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    import onnxruntime as ort

    t0 = time.perf_counter()
    onnx_model = convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, X_np32.shape[1]]))],
        zipmap=False,
    )
    t_conv = time.perf_counter() - t0
    onnx_path = SORTIE / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # masque les warnings de shape cosmétiques
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])

    def onnx_predict(arr):
        return sess.run(None, {"input": arr})[1][:, 1]

    # Parité des probabilités sur 1000 lignes
    proba_onnx = onnx_predict(X_np32[:1000])
    diff_onnx = float(np.max(np.abs(proba_sk - proba_onnx)))
    mismatch_sk_onnx = int(np.sum((proba_sk >= SEUIL) != (proba_onnx >= SEUIL)))
    mismatch_sk_booster = int(np.sum((proba_sk >= SEUIL)
                                     != (proba_booster >= SEUIL)))

    res_onnx = {
        "latence_1_ligne": bench(lambda: onnx_predict(np1)),
        "latence_batch_1000": bench(lambda: onnx_predict(X_np32[:1000])),
    }
    onnx_ok = True
    onnx_info = {
        "conversion_reussie": True,
        "temps_conversion_s": round(t_conv, 2),
        "taille_fichier_onnx_mo": round(os.path.getsize(onnx_path) / 1e6, 3),
        "providers": sess.get_providers(),
        "latence": res_onnx,
        "parite_vs_sklearn": {
            "n_lignes": 1000,
            "max_abs_diff": diff_onnx,
            "tolerance": 1e-4,
            "parite_ok": bool(diff_onnx < 1e-4),
            "decisions_seuil_045_divergentes": mismatch_sk_onnx,
        },
        "parite_booster_vs_sklearn_decisions": mismatch_sk_booster,
    }
    print(f"  ONNX : 1 ligne {res_onnx['latence_1_ligne']['mediane_ms']:.3f} ms | "
          f"batch 1000 {res_onnx['latence_batch_1000']['mediane_ms']:.3f} ms | "
          f"max|diff| = {diff_onnx:.2e} | décisions divergentes : "
          f"{mismatch_sk_onnx}/1000")
except Exception as exc:  # conversion ou runtime en échec : documenter
    import traceback
    onnx_info = {
        "conversion_reussie": False,
        "erreur": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=5),
    }
    print(f"  ÉCHEC ONNX : {type(exc).__name__}: {exc}")
resultats["onnx"] = onnx_info


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
print("Génération des figures ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1 : latence 1 ligne par méthode
methodes = ["sklearn\npredict_proba", "booster_.predict\n(bas niveau)"]
med = [res_1l["sklearn_predict_proba"]["mediane_ms"],
       res_1l["booster_predict"]["mediane_ms"]]
if onnx_ok:
    methodes.append("ONNX Runtime\n(CPU)")
    med.append(onnx_info["latence"]["latence_1_ligne"]["mediane_ms"])
fig, ax = plt.subplots(figsize=(8, 5))
couleurs = ["#4878CF", "#6ACC65", "#D65F5F"][:len(methodes)]
bars = ax.bar(methodes, med, color=couleurs)
for b, v in zip(bars, med):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f} ms",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Latence médiane (ms)")
ax.set_title("Latence d'une prédiction unitaire par méthode\n"
             f"(médiane sur {N_REP} répétitions, échelle log)")
ax.set_yscale("log")
fig.savefig(FIGURES / "latence_1_ligne.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Fig 2 : latence vs taille de batch (totale + par ligne)
tailles = [1, 10, 100, 1000, 10000]
sk_tot = [res_batch[str(n)]["sklearn_total"]["mediane_ms"] for n in tailles]
bo_tot = [res_batch[str(n)]["booster_total"]["mediane_ms"] for n in tailles]
sk_pl = [res_batch[str(n)]["sklearn_par_ligne_ms"] for n in tailles]
bo_pl = [res_batch[str(n)]["booster_par_ligne_ms"] for n in tailles]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(tailles, sk_tot, "o-", label="sklearn predict_proba", color="#4878CF")
ax1.plot(tailles, bo_tot, "s-", label="booster_.predict", color="#6ACC65")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Taille du batch (lignes)")
ax1.set_ylabel("Latence totale médiane (ms)")
ax1.set_title("Latence totale vs taille de batch")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)
ax2.plot(tailles, sk_pl, "o-", label="sklearn predict_proba", color="#4878CF")
ax2.plot(tailles, bo_pl, "s-", label="booster_.predict", color="#6ACC65")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Taille du batch (lignes)")
ax2.set_ylabel("Latence par ligne (ms)")
ax2.set_title("Latence par ligne vs taille de batch")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
fig.savefig(FIGURES / "latence_vs_batch.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Fig 3 : décomposition du temps (chemin 1 ligne)
labels = ["Construction\nDataFrame", "Coercion\npandas",
          "predict_proba\nseul", "Chemin API\ncomplet",
          "Construction\nnumpy (optimisé)", "Chemin optimisé\ncomplet"]
cles = ["construction_dataframe", "coercion_pandas", "predict_proba_seul",
        "chemin_api_complet", "construction_numpy_optimisee",
        "chemin_api_optimise_complet"]
vals = [etapes[c]["mediane_ms"] for c in cles]
fig, ax = plt.subplots(figsize=(10, 5))
cols_bar = ["#4878CF", "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
bars = ax.bar(labels, vals, color=cols_bar)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Temps médian (ms)")
ax.set_title("Décomposition du temps d'inférence unitaire par étape\n"
             f"(médiane sur {nrep_dec} répétitions)")
fig.savefig(FIGURES / "decomposition_1_ligne.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Sauvegarde des résultats
# ---------------------------------------------------------------------------
with open(SORTIE / "benchmarks.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

print("\nTerminé. Fichiers produits :")
print(f"  - {rapport_txt}")
print(f"  - {SORTIE / 'benchmarks.json'}")
print(f"  - {FIGURES / 'latence_1_ligne.png'}")
print(f"  - {FIGURES / 'latence_vs_batch.png'}")
print(f"  - {FIGURES / 'decomposition_1_ligne.png'}")
if onnx_ok:
    print(f"  - {SORTIE / 'model.onnx'}")
