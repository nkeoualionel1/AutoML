"""Tests de automl/preprocessing.py : prétraitement face aux valeurs extrêmes.

Rôle : vérifier que le préprocesseur ajusté sur le train produit toujours une
matrice propre (ni NaN ni infini, même largeur) au moment de transformer un
test contenant des pièges : colonne vide, colonne constante, `inf`, modalité
inconnue, valeurs négatives dans des données de comptage.

Quand le lancer : après toute modification de automl/preprocessing.py ou de
automl/inspection.py, et avant chaque fusion de branche (`python -m pytest tests`).
Pour vérifier le prétraitement sur des jeux complets : scripts/check_preprocessing.py.
"""

import numpy as np
import scipy.sparse as sp

from automl.inspection import inspect_data
from automl.preprocessing import build_preprocessor


def test_dense_extreme_values():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 5)
    X[:, 0] = np.nan  # colonne vide
    X[:, 1] = 3.0  # colonne constante
    X[:, 4] = rng.choice([1.0, 2.0, 3.0], 200)
    types = ["Numerical"] * 4 + ["Categorical"]
    y = (X[:, 2] > 0).astype(float)
    profile = inspect_data(X, types, y)
    prep = build_preprocessor(profile)
    Xt = prep.fit_transform(X, y)
    X_test = rng.randn(3, 5)
    X_test[0, 2] = np.inf  # infini
    X_test[1, 4] = 99.0  # modalité inconnue
    out = prep.transform(X_test)
    assert out.shape[1] == Xt.shape[1] and np.isfinite(out).all()


def test_sparse_negative_counts_at_test_time():
    X = sp.random(100, 30, density=0.2, random_state=2, format="csr")
    X.data = np.ceil(X.data * 5)  # comptages entiers
    y = np.arange(100) % 3
    profile = inspect_data(X, ["Numerical"] * 30, y.astype(float))
    prep = build_preprocessor(profile)
    prep.fit(X, y)
    X_neg = X[:5].copy()
    X_neg.data = -X_neg.data
    assert np.isfinite(prep.transform(X_neg).data).all()
