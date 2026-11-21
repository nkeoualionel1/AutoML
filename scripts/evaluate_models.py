"""Entraîne l'AutoML sur chaque jeu et affiche le score de chaque modèle essayé.

Rôle : mesurer les performances de bout en bout (étape 3 et suivantes). Pour
chaque jeu, affiche les modèles essayés, leur score de validation, leur durée
réelle et estimée, le nombre de lignes utilisées et le modèle retenu.

Quand l'utiliser :
- avant / après une modification de automl/models.py ou de la sélection du
  modèle (hyperparamètres, Optuna, ensembling) : on enregistre les scores dans
  benchmarks/, on modifie, on relance et on compare les deux CSV ;
- sur Skinner, pour vérifier que le budget temps est respecté sur la vraie machine.

Usage (depuis la racine du projet) :
    python scripts/evaluate_models.py                       # les 6 jeux de dataset/
    python scripts/evaluate_models.py data_B data_D         # seulement certains jeux
    python scripts/evaluate_models.py --budget 60 --csv benchmarks/step4.csv

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
    root = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="noms des jeux (par défaut : tous)")
    parser.add_argument("--folder", default=os.path.join(root, "dataset"))
    parser.add_argument("--budget", type=float, default=120, help="AUTOML_TIME_BUDGET par jeu (s)")
    parser.add_argument("--n-jobs", type=int, default=2, help="AUTOML_N_JOBS")
    parser.add_argument("--csv", help="fichier CSV où enregistrer les résultats")
    args = parser.parse_args()

    # À fixer avant l'import d'automl (limitation des threads natifs)
    os.environ["AUTOML_TIME_BUDGET"] = str(args.budget)
    os.environ["AUTOML_N_JOBS"] = str(args.n_jobs)
    import automl

    folder = os.path.abspath(args.folder)
    names = args.datasets or sorted(
        n for n in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, n))
        and any(f.endswith(".data") for f in os.listdir(os.path.join(folder, n)))
    )

    rows = []
    for name in names:
        print(f"\n=== {name} === (budget {args.budget:.0f} s, {args.n_jobs} cœurs)")
        start = time.perf_counter()
        m = automl.AutoML()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.fit(os.path.join(folder, name))
        except Exception as e:
            print(f"  ❌ fit a planté : {type(e).__name__}: {e}")
            continue
        elapsed = time.perf_counter() - start

        print(f"  Tâche : {m.task.task}")
        print(f"  {'modèle':24s} {'score valid':>11s} {'durée':>8s} {'estimée':>8s} {'lignes':>8s}")
        for r in m.leaderboard:
            mark = "  <- retenu" if r["model"] == m.metrics["model"] else ""
            print(f"  {r['model']:24s} {r['score']:11.4f} {r['duration_s']:7.1f}s "
                  f"{r['estimated_s']:7.1f}s {r['rows']:>8}{mark}")
        details = ", ".join(f"{k[5:]}={v:.4f}" for k, v in m.metrics.items()
                            if k.startswith("test_") and k != "test_score")
        print(f"  Test interne du modèle retenu : score {m.metrics['test_score']:.4f}  ({details})")
        print(f"  fit total : {elapsed:.1f} s")

        for r in m.leaderboard:
            rows.append({
                "dataset": name, "task": m.task.task, "model": r["model"],
                "valid_score": round(r["score"], 4), "duration_s": round(r["duration_s"], 1),
                "estimated_s": round(r["estimated_s"], 1), "rows": r["rows"],
                "selected": r["model"] == m.metrics["model"],
                "selected_test_score": round(m.metrics["test_score"], 4),
            })

    print("\n" + "=" * 60 + "\nRÉSUMÉ (modèle retenu par jeu)")
    for r in rows:
        if r["selected"]:
            print(f"  {r['dataset']:8s} {r['task']:20s} {r['model']:24s} "
                  f"valid={r['valid_score']:.4f} test={r['selected_test_score']:.4f}")

    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRésultats enregistrés dans {args.csv}")


if __name__ == "__main__":
    main()
