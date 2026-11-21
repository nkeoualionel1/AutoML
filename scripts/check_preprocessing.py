"""Vérifie les étapes 1 (inspection) et 2 (prétraitement) sur des jeux complets.

Rôle : charger chaque jeu, détecter la tâche, prétraiter, puis contrôler le
résultat (aucun NaN ni infini, format creux conservé, même largeur train / valid /
test, décodage fidèle de la cible). Aucun modèle n'est entraîné : c'est rapide.

Quand l'utiliser :
- après toute modification de automl/data.py, inspection.py ou preprocessing.py ;
- sur un nouveau jeu de données, pour vérifier qu'il est bien lu ;
- avant de fusionner une branche (en plus de `pytest`).

Usage (depuis la racine du projet) :
    python scripts/check_preprocessing.py                     # tous les jeux de test_datasets/
    python scripts/check_preprocessing.py test_adult ...      # seulement certains jeux
    python scripts/check_preprocessing.py --folder dataset    # nos 6 jeux d'origine
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")  # affichage des ✅ / ❌ sous Windows
import automl  # noqa: E402
from automl.data import load_dataset  # noqa: E402
from automl.preprocessing import MAX_DENSE_FEATURES  # noqa: E402


def check(results, ok, message):
    results.append(bool(ok))
    print(("    ✅ " if ok else "    ❌ ") + message)


def check_matrix(results, Xt, X_raw, label):
    values = Xt.data if sp.issparse(Xt) else Xt
    check(results, not np.isnan(values).any(), f"{label} : aucun NaN")
    check(results, np.isfinite(values).all(), f"{label} : aucune valeur infinie")
    check(results, sp.issparse(Xt) == sp.issparse(X_raw),
          f"{label} : format {'creux' if sp.issparse(Xt) else 'dense'} conservé")


def check_dataset(path, name, expected):
    results = []
    is_trap = os.path.isdir(os.path.join(path, "train"))
    train_dir = os.path.join(path, "train") if is_trap else path
    print(f"\n=== {name} === {expected.get('note', '')}")

    start = time.time()
    try:
        # Étapes 1 et 2 seulement : aucun modèle n'est entraîné, le test reste rapide
        with contextlib.redirect_stdout(io.StringIO()):
            m = automl.AutoML().prepare(train_dir)
    except Exception as e:
        check(results, False, f"prepare a planté : {type(e).__name__}: {e}")
        return results

    if "task" in expected:
        check(results, m.task.task == expected["task"],
              f"tâche détectée : {m.task.task} (attendue : {expected['task']})")
    print("    " + m.profile.summary().replace("\n", "\n    "))

    Xt_train, _ = m.splits["train"]
    Xt_valid, _ = m.splits["valid"]
    ds = load_dataset(train_dir)
    check_matrix(results, Xt_train, ds.X, "train")
    check(results, Xt_train.shape[1] == Xt_valid.shape[1], f"train et valid : {Xt_train.shape[1]} colonnes")
    check(results, np.array_equal(m.task.decode_y(m.task.encode_y(ds.y)), ds.y), "decode_y(encode_y(y)) == solution")
    if (not sp.issparse(ds.X) and ds.X.shape[1] > MAX_DENSE_FEATURES
            and m.task.task not in ("multilabel", "multioutput_regression")):
        check(results, Xt_train.shape[1] <= MAX_DENSE_FEATURES,
              f"sélection de variables : {Xt_train.shape[1]} ≤ {MAX_DENSE_FEATURES}")

    if is_trap:
        try:
            # Relu comme dans predict : même largeur, même codage qu'au train
            test = load_dataset(os.path.join(path, "test"), meta=m.feature_meta)
            Xt_test = m.preprocessor.transform(test.X)
            check(results, test.y is None, "le dossier test n'a pas de solution")
            check(results, Xt_test.shape == (test.X.shape[0], Xt_train.shape[1]),
                  f"test prétraité : {Xt_test.shape} (même largeur que le train)")
            check_matrix(results, Xt_test, test.X, "test")
        except Exception as e:
            check(results, False, f"transform du test a planté : {type(e).__name__}: {e}")

    print(f"    ({time.time() - start:.1f} s)")
    return results


def is_dataset_dir(path):
    """Un dossier de jeu contient un .data, ou un sous-dossier train/ (pièges)."""
    if not os.path.isdir(path):
        return False
    return os.path.isdir(os.path.join(path, "train")) or any(f.endswith(".data") for f in os.listdir(path))


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="noms des jeux (par défaut : tous)")
    parser.add_argument("--folder", default=os.path.join(root, "test_datasets"))
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    expected_path = os.path.join(folder, "expected.json")
    expected = json.load(open(expected_path, encoding="utf-8")) if os.path.exists(expected_path) else {}
    names = args.datasets or sorted(n for n in os.listdir(folder) if is_dataset_dir(os.path.join(folder, n)))

    summary = {}
    for name in names:
        summary[name] = all(check_dataset(os.path.join(folder, name), name, expected.get(name, {})))

    print("\n" + "=" * 50 + "\nBILAN")
    for name, ok in summary.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"{sum(summary.values())}/{len(summary)} jeux OK")
    sys.exit(0 if all(summary.values()) else 1)


if __name__ == "__main__":
    main()
