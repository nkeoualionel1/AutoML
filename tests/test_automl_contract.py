"""Tests du contrat imposé par le sujet : fit / eval / predict de bout en bout.

Rôle : vérifier, sur de petits jeux synthétiques couvrant toutes les tâches, que
l'interface officielle (`automl.fit`, `automl.eval`, `automl.predict` et la classe
`automl.AutoML`) fonctionne et que `predict` renvoie exactement la forme et les
valeurs du fichier .solution (étiquettes d'origine, one-hot qui somme à 1,
multi-label en 0/1), y compris sur un test creux contenant des pièges.

Quand le lancer : après toute modification de automl/ (en particulier
__init__.py, models.py, metrics.py), avant chaque fusion de branche et avant la
soumission finale (`python -m pytest tests`). C'est le test le plus important :
si le contrat est cassé, l'évaluation automatique échoue.
"""

import numpy as np
import pytest

import automl
from automl import AutoML

RNG = np.random.RandomState(0)
X_DENSE = RNG.randn(150, 5)


@pytest.fixture(autouse=True)
def _small_budget(monkeypatch):
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "8")
    monkeypatch.setenv("AUTOML_N_JOBS", "2")
    monkeypatch.setenv("AUTOML_N_TRIALS", "3")  # quelques essais Optuna suffisent pour tester le contrat


@pytest.fixture
def fit_predict(tmp_path, write_dataset):
    """Entraîne sur un jeu, puis prédit sur ses n_test premières lignes (sans .solution)."""
    def run(X_lines, types, y_lines, n_test=20, folder=tmp_path):
        train = write_dataset(folder / "train", "jeu", X_lines, types, y_lines)
        test = write_dataset(folder / "test", "jeu", X_lines[:n_test], types)
        m = AutoML().fit(train)
        return m, m.eval(), m.predict(test)
    return run


@pytest.mark.parametrize(
    "y_lines, shape, values",
    [
        ([str(int(v > 0)) for v in X_DENSE[:, 0]], (20,), {0, 1}),
        (["chat" if v > 0 else "chien" for v in X_DENSE[:, 0]], (20,), {"chat", "chien"}),
        ([str(int(v)) for v in np.digitize(X_DENSE[:, 0], [-0.5, 0.5]) * 4 + 3], (20,), {3, 7, 11}),
    ],
)
def test_classification_1d(fit_predict, dense_lines, y_lines, shape, values):
    _, metrics, preds = fit_predict(dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    assert preds.shape == shape
    assert set(preds.tolist()) <= values
    assert "valid_accuracy" in metrics and "test_score" in metrics


def test_onehot_rows_sum_to_one(fit_predict, dense_lines):
    Y = np.eye(3, dtype=int)[np.argmax(X_DENSE[:, :3], axis=1)]
    _, _, preds = fit_predict(dense_lines(X_DENSE), ["Numerical"] * 5,
                              [" ".join(map(str, r)) for r in Y])
    assert preds.shape == (20, 3) and np.all(preds.sum(axis=1) == 1)


def test_multilabel(fit_predict, dense_lines):
    Y = (X_DENSE[:, :3] > 0).astype(int)
    _, metrics, preds = fit_predict(dense_lines(X_DENSE), ["Numerical"] * 5,
                                    [" ".join(map(str, r)) for r in Y])
    assert preds.shape == (20, 3) and set(np.unique(preds)) <= {0, 1}
    assert "valid_subset_accuracy" in metrics


def test_regression_is_reproducible(tmp_path, fit_predict, dense_lines):
    y = 3 * X_DENSE[:, 0] ** 2 - X_DENSE[:, 1] + 0.1 * RNG.randn(150)
    y_lines = [f"{v:.5f}" for v in y]
    _, metrics, p1 = fit_predict(dense_lines(X_DENSE), ["Numerical"] * 5, y_lines, folder=tmp_path / "a")
    _, _, p2 = fit_predict(dense_lines(X_DENSE), ["Numerical"] * 5, y_lines, folder=tmp_path / "b")
    assert p1.shape == (20,) and metrics["valid_r2"] > 0.5
    assert np.allclose(p1, p2)  # même graine -> mêmes prédictions


def test_sparse_with_traps(tmp_path, write_dataset):
    lines, y = [], []
    for i in range(120):
        k = i % 3
        lines.append(f"{k * 5 + 1}:3 {k * 5 + 2}:2 {RNG.randint(20, 40)}:1")
        y.append(" ".join("1" if j == k else "0" for j in range(3)))
    train = write_dataset(tmp_path / "train", "sp", lines, ["Numerical"] * 50, y)
    test = write_dataset(tmp_path / "test", "sp", lines[:5] + ["999:2 1:3", ""], ["Numerical"] * 50)
    m = AutoML().fit(train)
    preds = m.predict(test)
    assert preds.shape == (7, 3) and np.all(preds.sum(axis=1) == 1)


def test_module_interface(tmp_path, write_dataset, dense_lines):
    y_lines = [str(int(v > 0)) for v in X_DENSE[:, 0]]
    train = write_dataset(tmp_path / "train", "jeu", dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    automl.fit(train)
    assert isinstance(automl.eval(), dict)
    assert automl.predict(train).shape == (150,)
