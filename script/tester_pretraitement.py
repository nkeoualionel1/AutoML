"""Vérifie les étapes 1 (inspection) et 2 (prétraitement) sur des jeux de test.

Usage (depuis la racine du projet) :
    python script/tester_pretraitement.py                  # tous les jeux de dataset_test/
    python script/tester_pretraitement.py test_adult ...   # seulement certains jeux
    python script/tester_pretraitement.py --dossier dataset   # nos 6 jeux d'origine
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


def verifier(resultats, ok, message):
    resultats.append(bool(ok))
    print(("    ✅ " if ok else "    ❌ ") + message)


def verifier_matrice(res, Xt, X_brut, nom_jeu):
    valeurs = Xt.data if sp.issparse(Xt) else Xt
    verifier(res, not np.isnan(valeurs).any(), f"{nom_jeu} : aucun NaN")
    verifier(res, np.isfinite(valeurs).all(), f"{nom_jeu} : aucune valeur infinie")
    verifier(res, sp.issparse(Xt) == sp.issparse(X_brut), f"{nom_jeu} : format {'creux' if sp.issparse(Xt) else 'dense'} conservé")


def tester(chemin, nom, attendu):
    res = []
    est_piege = os.path.isdir(os.path.join(chemin, "train"))
    dossier_train = os.path.join(chemin, "train") if est_piege else chemin
    print(f"\n=== {nom} === {attendu.get('note', '')}")

    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # masque les messages de fit
            m = automl.AutoML().fit(dossier_train)
    except Exception as e:
        verifier(res, False, f"fit a planté : {type(e).__name__}: {e}")
        return res

    if "tache" in attendu:
        verifier(res, m.task.task == attendu["tache"], f"tâche détectée : {m.task.task} (attendue : {attendu['tache']})")
    print("    " + m.profile.summary().replace("\n", "\n    "))

    Xt_tr, y_tr = m.splits["train"]
    Xt_va, _ = m.splits["valid"]
    ds = load_dataset(dossier_train)
    verifier_matrice(res, Xt_tr, ds.X, "train")
    verifier(res, Xt_tr.shape[1] == Xt_va.shape[1], f"train et valid : {Xt_tr.shape[1]} colonnes")
    verifier(res, np.array_equal(m.task.decode_y(m.task.encode_y(ds.y)), ds.y), "decode_y(encode_y(y)) == solution")
    if (not sp.issparse(ds.X) and ds.X.shape[1] > MAX_DENSE_FEATURES
            and m.task.task not in ("multilabel", "multioutput_regression")):
        verifier(res, Xt_tr.shape[1] <= MAX_DENSE_FEATURES, f"sélection de variables : {Xt_tr.shape[1]} ≤ {MAX_DENSE_FEATURES}")

    if est_piege:
        try:
            test = load_dataset(os.path.join(chemin, "test"))
            Xt_te = m.preprocessor.transform(test.X)
            verifier(res, test.y is None, "le dossier test n'a pas de solution")
            verifier(res, Xt_te.shape == (test.X.shape[0], Xt_tr.shape[1]),
                     f"test prétraité : {Xt_te.shape} (même largeur que le train)")
            verifier_matrice(res, Xt_te, test.X, "test")
        except Exception as e:
            verifier(res, False, f"transform du test a planté : {type(e).__name__}: {e}")

    print(f"    ({time.time() - t:.1f} s)")
    return res


def main():
    racine = os.path.join(os.path.dirname(__file__), "..")
    parser = argparse.ArgumentParser()
    parser.add_argument("jeux", nargs="*", help="noms des jeux (par défaut : tous)")
    parser.add_argument("--dossier", default=os.path.join(racine, "dataset_test"))
    args = parser.parse_args()

    dossier = os.path.abspath(args.dossier)
    chemin_attendu = os.path.join(dossier, "attendu.json")
    attendus = json.load(open(chemin_attendu, encoding="utf-8")) if os.path.exists(chemin_attendu) else {}
    noms = args.jeux or sorted(n for n in os.listdir(dossier) if os.path.isdir(os.path.join(dossier, n)))

    bilan = {}
    for nom in noms:
        r = tester(os.path.join(dossier, nom), nom, attendus.get(nom, {}))
        bilan[nom] = all(r)

    print("\n" + "=" * 50 + "\nBILAN")
    for nom, ok in bilan.items():
        print(f"  {'✅' if ok else '❌'} {nom}")
    print(f"{sum(bilan.values())}/{len(bilan)} jeux OK")
    sys.exit(0 if all(bilan.values()) else 1)


if __name__ == "__main__":
    main()
