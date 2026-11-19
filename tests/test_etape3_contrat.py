"""Tests du contrat imposé par le sujet : fit / eval / predict sur toutes les tâches."""

import numpy as np
import pytest

import automl
from automl import AutoML


def _write(folder, name, X_lines, types, y_lines=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.data").write_text("\n".join(X_lines) + "\n")
    (folder / f"{name}.type").write_text("\n".join(types) + "\n")
    if y_lines is not None:
        (folder / f"{name}.solution").write_text("\n".join(y_lines) + "\n")
    return str(folder)


def _dense_lines(X):
    return [" ".join(f"{v:.4f}" for v in row) for row in X]


@pytest.fixture(autouse=True)
def _petit_budget(monkeypatch):
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "20")
    monkeypatch.setenv("AUTOML_N_JOBS", "2")


def _fit_predict(tmp_path, X_lines, types, y_lines, n_test=20):
    train = _write(tmp_path / "train", "jeu", X_lines, types, y_lines)
    test = _write(tmp_path / "test", "jeu", X_lines[:n_test], types)  # sans .solution
    m = AutoML().fit(train)
    return m, m.eval(), m.predict(test)


RNG = np.random.RandomState(0)
X_DENSE = RNG.randn(150, 5)


@pytest.mark.parametrize(
    "y_lines, shape, values",
    [
        ([str(int(v > 0)) for v in X_DENSE[:, 0]], (20,), {0, 1}),
        (["chat" if v > 0 else "chien" for v in X_DENSE[:, 0]], (20,), {"chat", "chien"}),
        ([str(int(v)) for v in np.digitize(X_DENSE[:, 0], [-0.5, 0.5]) * 4 + 3], (20,), {3, 7, 11}),
    ],
)
def test_classification_1d(tmp_path, y_lines, shape, values):
    _, metrics, preds = _fit_predict(tmp_path, _dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    assert preds.shape == shape
    assert set(preds.tolist()) <= values
    assert "valid_accuracy" in metrics and "test_score" in metrics


def test_onehot_somme_a_un(tmp_path):
    Y = np.eye(3, dtype=int)[np.argmax(X_DENSE[:, :3], axis=1)]
    _, _, preds = _fit_predict(tmp_path, _dense_lines(X_DENSE), ["Numerical"] * 5,
                               [" ".join(map(str, r)) for r in Y])
    assert preds.shape == (20, 3) and np.all(preds.sum(axis=1) == 1)


def test_multilabel(tmp_path):
    Y = (X_DENSE[:, :3] > 0).astype(int)
    _, metrics, preds = _fit_predict(tmp_path, _dense_lines(X_DENSE), ["Numerical"] * 5,
                                     [" ".join(map(str, r)) for r in Y])
    assert preds.shape == (20, 3) and set(np.unique(preds)) <= {0, 1}
    assert "valid_subset_accuracy" in metrics


def test_regression_et_reproductibilite(tmp_path):
    y = 3 * X_DENSE[:, 0] ** 2 - X_DENSE[:, 1] + 0.1 * RNG.randn(150)
    y_lines = [f"{v:.5f}" for v in y]
    m1, metrics, p1 = _fit_predict(tmp_path / "a", _dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    _, _, p2 = _fit_predict(tmp_path / "b", _dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    assert p1.shape == (20,) and metrics["valid_r2"] > 0.5
    assert np.allclose(p1, p2)  # même graine -> mêmes prédictions


def test_creux_avec_pieges(tmp_path):
    lines, y = [], []
    for i in range(120):
        k = i % 3
        lines.append(f"{k * 5 + 1}:3 {k * 5 + 2}:2 {RNG.randint(20, 40)}:1")
        y.append(" ".join("1" if j == k else "0" for j in range(3)))
    train = _write(tmp_path / "train", "sp", lines, ["Numerical"] * 50, y)
    test = _write(tmp_path / "test", "sp", lines[:5] + ["999:2 1:3", ""], ["Numerical"] * 50)
    m = AutoML().fit(train)
    preds = m.predict(test)
    assert preds.shape == (7, 3) and np.all(preds.sum(axis=1) == 1)


def test_interface_module(tmp_path):
    y_lines = [str(int(v > 0)) for v in X_DENSE[:, 0]]
    train = _write(tmp_path / "train", "jeu", _dense_lines(X_DENSE), ["Numerical"] * 5, y_lines)
    automl.fit(train)
    assert isinstance(automl.eval(), dict)
    assert automl.predict(train).shape == (150,)
