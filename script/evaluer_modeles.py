"""Entraîne l'AutoML sur chaque jeu et affiche le score de chaque modèle essayé.

Sert de référence avant / après l'optimisation des hyperparamètres : on lance ce
script, on note les scores, on modifie automl/models.py, on relance et on compare.

Usage (depuis la racine du projet) :
    python script/evaluer_modeles.py                       # les 6 jeux de dataset/
    python script/evaluer_modeles.py data_B data_D         # seulement certains jeux
    python script/evaluer_modeles.py --budget 60 --csv resultats/etape3.csv

Le budget temps (secondes par jeu) et le nombre de cœurs passent par les mêmes
variables d'environnement que lors de l'évaluation officielle.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    racine = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("jeux", nargs="*", help="noms des jeux (par défaut : tous)")
    parser.add_argument("--dossier", default=os.path.join(racine, "dataset"))
    parser.add_argument("--budget", type=float, default=120, help="AUTOML_TIME_BUDGET par jeu (s)")
    parser.add_argument("--n-jobs", type=int, default=2, help="AUTOML_N_JOBS")
    parser.add_argument("--csv", help="fichier CSV où enregistrer les résultats")
    args = parser.parse_args()

    # À fixer avant l'import d'automl (limitation des threads natifs)
    os.environ["AUTOML_TIME_BUDGET"] = str(args.budget)
    os.environ["AUTOML_N_JOBS"] = str(args.n_jobs)
    import automl

    dossier = os.path.abspath(args.dossier)
    noms = args.jeux or sorted(
        n for n in os.listdir(dossier)
        if os.path.isdir(os.path.join(dossier, n))
        and any(f.endswith(".data") for f in os.listdir(os.path.join(dossier, n)))
    )

    lignes = []
    for nom in noms:
        print(f"\n=== {nom} === (budget {args.budget:.0f} s, {args.n_jobs} cœurs)")
        t = time.perf_counter()
        m = automl.AutoML()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(os.path.join(dossier, nom))
        except Exception as e:
            print(f"  ❌ fit a planté : {type(e).__name__}: {e}")
            continue
        duree = time.perf_counter() - t

        print(f"  Tâche : {m.task.task}")
        print(f"  {'modèle':24s} {'score valid':>11s} {'durée':>8s}")
        for r in m.leaderboard:
            marque = "  <- retenu" if r["modèle"] == m.metrics["model"] else ""
            print(f"  {r['modèle']:24s} {r['score']:11.4f} {r['durée (s)']:7.1f}s{marque}")
        print(f"  Test interne du modèle retenu : score {m.metrics['test_score']:.4f}"
              f"  ({', '.join(f'{k[5:]}={v:.4f}' for k, v in m.metrics.items() if k.startswith('test_') and k != 'test_score')})")
        print(f"  fit total : {duree:.1f} s")

        for r in m.leaderboard:
            lignes.append({
                "jeu": nom, "tache": m.task.task, "modele": r["modèle"],
                "score_valid": round(r["score"], 4), "duree_s": round(r["durée (s)"], 1),
                "retenu": r["modèle"] == m.metrics["model"],
                "score_test_retenu": round(m.metrics["test_score"], 4),
            })

    print("\n" + "=" * 60 + "\nRÉSUMÉ (modèle retenu par jeu)")
    for l in lignes:
        if l["retenu"]:
            print(f"  {l['jeu']:8s} {l['tache']:20s} {l['modele']:24s} valid={l['score_valid']:.4f} test={l['score_test_retenu']:.4f}")

    if args.csv and lignes:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0]))
            w.writeheader()
            w.writerows(lignes)
        print(f"\nRésultats enregistrés dans {args.csv}")


if __name__ == "__main__":
    main()
