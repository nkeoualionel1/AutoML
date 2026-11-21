"""Tests de automl/data.py : chargement des fichiers et découpage train / valid / test.

Rôle : vérifier que les pièges d'un jeu de test secret ne font pas planter le
chargement (`?`, `inf`, texte, modalité inconnue, colonne en moins, indices creux
commençant à 0 ou hors bornes, ligne vide) et que le test est relu exactement
comme le train (même largeur, même codage du texte).

Quand le lancer : après toute modification de automl/data.py, et avant chaque
fusion de branche (`python -m pytest tests`).
"""

import numpy as np

from automl.data import adaptive_split_sizes, load_dataset


def test_dense_text_tokens_and_column_alignment(tmp_path, write_dataset):
    train = write_dataset(
        tmp_path / "tr", "d",
        ["rouge 1.5 ?", "bleu inf 2", "rouge 0.5 NaN", "vert 2 3"],
        ["Categorical", "Numerical", "Numerical"], ["0", "1", "0", "1"],
    )
    ds = load_dataset(train)
    assert ds.X.shape == (4, 3)
    assert np.isnan(ds.X[0, 2]) and np.isnan(ds.X[1, 1])  # `?` et `inf` -> NaN

    # Test : modalité inconnue, texte parasite dans une colonne numérique, colonne en moins
    test = write_dataset(tmp_path / "te", "d", ["violet oops", "bleu 1"], ["Categorical", "Numerical"])
    te = load_dataset(test, meta=ds.meta)
    assert te.X.shape == (2, 3)
    assert np.isnan(te.X[0, 0]) and np.isnan(te.X[0, 1]) and np.isnan(te.X[:, 2]).all()
    assert te.X[1, 0] == ds.X[1, 0]  # "bleu" a le même code qu'au train


def test_sparse_zero_based_out_of_bounds_and_empty_line(tmp_path, write_dataset):
    train = write_dataset(tmp_path / "tr", "s", ["0:1 2:3", "1:2"], ["Numerical"] * 3, ["0", "1"])
    ds = load_dataset(train)
    assert ds.meta.sparse_offset == 0 and ds.X.shape == (2, 3)
    test = write_dataset(tmp_path / "te", "s", ["0:1 99:4", ""], ["Numerical"] * 3)
    te = load_dataset(test, meta=ds.meta)
    assert te.X.shape == (2, 3) and te.X.nnz == 1


def test_adaptive_split_sizes():
    v_small, t_small = adaptive_split_sizes(1000)
    v_big, t_big = adaptive_split_sizes(100000)
    assert v_small > v_big and 0 < t_big <= v_big
    assert 100000 * v_big > 1000 * v_small
