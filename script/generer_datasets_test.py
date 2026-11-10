"""Génère des jeux de test au même format que ceux du challenge.

Chaque jeu est un dossier <nom>/ avec <nom>.data, <nom>.type, <nom>.solution.
Les jeux « piege_* » ont un sous-dossier train/ (avec solution) et un
sous-dossier test/ (sans solution), pour tester le prétraitement au predict.

Usage (depuis la racine du projet) :
    python script/generer_datasets_test.py

Certains jeux réels sont téléchargés via scikit-learn (accès réseau) : ce
script est un outil de test, il ne fait pas partie du package automl/.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import datasets as skd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from automl.data import load_dataset  # noqa: E402

SEED = 42
ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCE = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "dataset_test")
rng = np.random.RandomState(SEED)
attendu = {}


# ---------------------------------------------------------------------------
# Écriture au format du challenge
# ---------------------------------------------------------------------------

def write(folder, name, X, types, y=None):
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, name)
    if sp.issparse(X):
        X = sp.csr_matrix(X)
        with open(base + ".data", "w") as f:
            for i in range(X.shape[0]):
                start, end = X.indptr[i], X.indptr[i + 1]
                f.write(" ".join(f"{j + 1}:{v:g}" for j, v in zip(X.indices[start:end], X.data[start:end])) + "\n")
    else:
        np.savetxt(base + ".data", np.asarray(X, dtype=float), fmt="%.6g")
    with open(base + ".type", "w") as f:
        f.write("\n".join(types) + "\n")
    if y is not None:
        np.savetxt(base + ".solution", np.asarray(y), fmt="%g")


def onehot(labels):
    classes, idx = np.unique(labels, return_inverse=True)
    out = np.zeros((len(labels), len(classes)), dtype=int)
    out[np.arange(len(labels)), idx] = 1
    return out


def simple(name, X, types, y, task, note):
    write(os.path.join(OUT, name), name, X, types, y)
    attendu[name] = {"tache": task, "note": note}
    print(f"  ✓ {name:28s} {X.shape} -> {task}")


def piege(name, X_train, y_train, X_test, types, task, note):
    write(os.path.join(OUT, name, "train"), name, X_train, types, y_train)
    write(os.path.join(OUT, name, "test"), name, X_test, types)
    attendu[name] = {"tache": task, "note": note}
    print(f"  ✓ {name:28s} train {X_train.shape} / test {X_test.shape} -> {task}")


def frame_to_challenge(df):
    """DataFrame mixte -> matrice numérique (catégories codées en entiers, NaN gardés)."""
    cols, types = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            cols.append(s.astype(float).to_numpy())
            types.append("Binary" if s.dropna().nunique() == 2 else "Numerical")
        else:
            codes = s.astype("category").cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
            types.append("Binary" if s.dropna().nunique() == 2 else "Categorical")
    return np.column_stack(cols), types


def run(label, fn):
    try:
        fn()
    except Exception as e:  # un téléchargement raté ne doit pas tout arrêter
        print(f"  ✗ {label} ignoré : {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 1. Jeux réels proches des nôtres
# ---------------------------------------------------------------------------

def adult():
    d = skd.fetch_openml("adult", version=2, as_frame=True)
    idx = rng.choice(len(d.frame), 15000, replace=False)
    df = d.data.iloc[idx]
    X, types = frame_to_challenge(df)
    y = (d.target.iloc[idx] == ">50K").astype(int).to_numpy()
    simple("test_adult", X, types, y, "binary", "comme data_A : catégorielles + manquants")


def titanic():
    # Le parseur pandas plante sur ce fichier ARFF : on force liac-arff
    d = skd.fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff")
    df = d.data.drop(columns=["boat", "body"])  # fuites de la cible
    X, types = frame_to_challenge(df)
    simple("test_titanic", X, types, d.target.astype(int).to_numpy(), "binary",
           "colonne presque vide (cabin), forte cardinalité (name, ticket)")


def california():
    d = skd.fetch_california_housing()
    simple("test_california", d.data, ["Numerical"] * d.data.shape[1], d.target,
           "regression", "comme data_B : régression, variables asymétriques")


def mnist():
    d = skd.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    idx = rng.choice(len(d.target), 8000, replace=False)
    simple("test_mnist", d.data[idx], ["Numerical"] * 784, onehot(d.target[idx].astype(int)),
           "multiclass_onehot", "comme data_C : pixels, colonnes constantes au bord")


def news20():
    from sklearn.feature_extraction.text import CountVectorizer
    d = skd.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), random_state=SEED)
    X = CountVectorizer(min_df=3).fit_transform(d.data)
    simple("test_20news", X, ["Numerical"] * X.shape[1], onehot(d.target),
           "multiclass_onehot", "comme data_F : comptages de mots creux, 20 classes")


# ---------------------------------------------------------------------------
# 2. Jeux inclus dans scikit-learn et jeux générés (sans réseau)
# ---------------------------------------------------------------------------

def inclus_et_generes():
    d = skd.load_diabetes()
    simple("test_diabetes", d.data, ["Numerical"] * 10, d.target, "regression", "petite régression déjà centrée")

    d = skd.load_digits()
    simple("test_digits", d.data, ["Numerical"] * 64, onehot(d.target), "multiclass_onehot",
           "mini data_C, colonnes constantes")

    d = skd.load_breast_cancer()
    simple("test_breast_cancer", d.data, ["Numerical"] * 30, d.target, "binary", "échelles très différentes")

    d = skd.load_wine()
    labels = np.array([3, 7, 12])[d.target]
    simple("test_wine_labels", d.data, ["Numerical"] * 13, labels, "multiclass",
           "multiclasse 1-D non one-hot (étiquettes 3, 7, 12)")

    X, y = skd.make_classification(4000, 20, weights=[0.95, 0.05], random_state=SEED)
    simple("synth_desequilibre", X, ["Numerical"] * 20, y, "binary", "95 % / 5 %")

    X, y = skd.make_multilabel_classification(3000, 15, n_classes=3, random_state=SEED)
    X = X.astype(float)
    X[:, :5] = rng.randint(0, 12, size=(3000, 5))         # 5 colonnes catégorielles
    X[rng.rand(*X.shape) < 0.08] = np.nan                 # 8 % de manquants
    types = ["Categorical"] * 5 + ["Numerical"] * 10
    simple("synth_multilabel", X, types, y, "multilabel", "comme data_A : multi-label + catégorielles + NaN")

    X, y = skd.make_regression(2000, 12, n_targets=3, noise=5, random_state=SEED)
    simple("synth_multisortie", X, ["Numerical"] * 12, y, "multioutput_regression", "3 cibles continues")

    X, y = skd.make_classification(500, 5500, n_informative=30, random_state=SEED)
    simple("synth_tres_large", np.round(X, 2), ["Numerical"] * 5500, y, "binary",
           "5 500 variables : déclenche SelectKBest (max 5 000)")

    X, y = skd.make_multilabel_classification(3000, 2000, n_classes=5, sparse=True,
                                              return_indicator="dense", random_state=SEED)
    X = sp.csr_matrix(X, dtype=float) * -0.5
    simple("synth_creux_negatif", X, ["Numerical"] * 2000, y, "multilabel",
           "creux avec valeurs négatives décimales -> MaxAbsScaler")


# ---------------------------------------------------------------------------
# 3. Pièges construits à partir de nos propres jeux
# ---------------------------------------------------------------------------

def pieges():
    A = load_dataset(os.path.join(SOURCE, "data_A"))
    B = load_dataset(os.path.join(SOURCE, "data_B"))
    F = load_dataset(os.path.join(SOURCE, "data_F"))
    cut_A, cut_B, cut_F = 30000, 4500, 12000

    cat = A.types.index("Categorical")
    X_test = A.X[cut_A:].copy()
    X_test[::3, cat] = 999  # modalité jamais vue à l'entraînement
    piege("piege_categorie_inconnue", A.X[:cut_A], A.y[:cut_A], X_test, A.types, "multilabel",
          "modalité 999 seulement dans le test")

    X_test = B.X[cut_B:].copy()
    X_test[::4, 0] = np.nan  # la colonne 0 n'a aucun manquant au train
    piege("piege_nan_test", B.X[:cut_B], B.y[:cut_B], X_test, B.types, "regression",
          "NaN dans le test sur une colonne sans manquants")

    X_test = B.X[cut_B:].copy()
    X_test[::5, :] = -np.abs(X_test[::5, :]) - 1  # négatifs, y compris dans les colonnes passées en log
    piege("piege_log_negatif", B.X[:cut_B], B.y[:cut_B], X_test, B.types, "regression",
          "valeurs négatives dans des colonnes passées en log")

    piege("piege_une_ligne", A.X[:cut_A], A.y[:cut_A], A.X[cut_A:cut_A + 1], A.types, "multilabel",
          "jeu de test d'une seule ligne")

    X = A.X.copy()
    X[:, 1] = np.nan
    simple("piege_colonne_vide", X, A.types, A.y, "multilabel", "colonne 1 entièrement vide")

    # Test creux dont aucune ligne n'atteint les dernières colonnes
    X_test = F.X[cut_F:].tocsc()[:, : F.X.shape[1] - 5000].tocsr()
    X_test = sp.hstack([X_test, sp.csr_matrix((X_test.shape[0], 5000))]).tocsr()
    piege("piege_creux_derniere_colonne", F.X[:cut_F], F.y[:cut_F], X_test, F.types, "multiclass_onehot",
          "les 5 000 dernières colonnes sont vides dans le test")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("1. Jeux réels (téléchargés)")
    for label, fn in [("adult", adult), ("titanic", titanic), ("california", california),
                      ("mnist", mnist), ("20news", news20)]:
        run(label, fn)
    print("2. Jeux inclus et générés")
    run("inclus_et_generes", inclus_et_generes)
    print("3. Pièges")
    run("pieges", pieges)
    with open(os.path.join(OUT, "attendu.json"), "w", encoding="utf-8") as f:
        json.dump(attendu, f, ensure_ascii=False, indent=2)
    print(f"\n{len(attendu)} jeux écrits dans {os.path.abspath(OUT)}")
