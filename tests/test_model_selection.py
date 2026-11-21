"""Tests du choix du modèle : budget, reproductibilité, Optuna, ensembling, gains rapides.

Rôle : vérifier les exigences du sujet et les réglages automatiques de l'étape
« choix du modèle » (automl/models.py, selection.py, __init__.py) :
- `fit` respecte AUTOML_TIME_BUDGET ;
- le choix est reproductible : deux `fit` sur les mêmes données essaient les mêmes
  modèles et les mêmes réglages Optuna, et donnent les mêmes prédictions (le plan
  d'entraînement dépend des coûts estimés, pas du chronomètre) ;
- Optuna ne dépasse pas AUTOML_N_TRIALS essais ;
- l'ensemble moyenné renvoie des prédictions au bon format ;
- Optuna peut essayer `class_weight="balanced"`, et la cible est apprise en log, quand l'inspection
  détecte des classes déséquilibrées ou une cible asymétrique.

Quand le lancer : après toute modification des modèles candidats, des formules
de coût, d'Optuna, de l'ensembling ou de `_select_model`, et avant chaque fusion
de branche (`python -m pytest tests`).
"""

import time

import numpy as np
import optuna
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression

from automl import AutoML
from automl.inspection import detect_task, inspect_data
from automl.models import Candidate, DataShape, forest_cost, get_candidates, rows_within
from automl.selection import AveragingEnsemble


def _medium_binary(tmp_path, write_dataset, dense_lines):
    rng = np.random.RandomState(1)
    X = rng.randn(4000, 40)
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
    train = write_dataset(tmp_path / "train", "medium", dense_lines(X), ["Numerical"] * 40, [str(v) for v in y])
    test = write_dataset(tmp_path / "test", "medium", dense_lines(X[:50]), ["Numerical"] * 40)
    return train, test


def test_budget_respected_and_choice_reproducible(tmp_path, monkeypatch, write_dataset, dense_lines):
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "15")
    monkeypatch.setenv("AUTOML_N_JOBS", "2")
    monkeypatch.setenv("AUTOML_N_TRIALS", "4")
    train, test = _medium_binary(tmp_path, write_dataset, dense_lines)

    runs = []
    for _ in range(2):
        start = time.perf_counter()
        m = AutoML().fit(train)
        runs.append((time.perf_counter() - start, m))
    for elapsed, _ in runs:
        assert elapsed < 15 * 1.2, f"fit a duré {elapsed:.1f}s pour un budget de 15s"
    plans = [[(r["model"], r["rows"]) for r in m.leaderboard] for _, m in runs]
    assert plans[0] == plans[1]
    assert runs[0][1].metrics["model"] == runs[1][1].metrics["model"]
    assert np.array_equal(runs[0][1].predict(test), runs[1][1].predict(test))


def test_optuna_respects_n_trials(tmp_path, monkeypatch, write_dataset, dense_lines):
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "60")
    monkeypatch.setenv("AUTOML_N_JOBS", "2")
    monkeypatch.setenv("AUTOML_N_TRIALS", "3")
    rng = np.random.RandomState(2)
    X = rng.randn(600, 8)  # petit jeu : les essais Optuna tiennent largement dans le budget
    y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    train = write_dataset(tmp_path / "train", "small", dense_lines(X), ["Numerical"] * 8, [str(v) for v in y])
    m = AutoML().fit(train)
    optuna_runs = [r for r in m.leaderboard if "optuna" in r["model"]]
    assert 1 <= len(optuna_runs) <= 3


def test_rows_within_matches_cost_estimate():
    cand = Candidate("rf", lambda p: None, forest_cost, {"n_estimators": 300})
    shape = DataShape(d=100, nnz_per_row=100, n_classes=1, n_outputs=1, n_jobs=2)

    def cost(n):
        return cand.estimate(shape, n)

    n = rows_within(cost, 100000, seconds=10, n_min=500)
    assert 500 <= n < 100000 and cost(n) <= 10 < cost(n + 1)
    assert rows_within(cost, 100000, seconds=0.01, n_min=500) == 0


def test_imbalanced_classes_let_optuna_try_balanced_class_weight():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 5)
    y = (rng.rand(500) < 0.05).astype(float)  # 5 % de positifs
    profile = inspect_data(X, ["Numerical"] * 5, y)
    profile.task = detect_task(y)
    assert profile.is_imbalanced
    shape = DataShape(d=5, nnz_per_row=5, n_classes=1, n_outputs=1, n_jobs=1)
    for cand in get_candidates(profile, n_jobs=1):
        params = cand.space(optuna.create_study().ask(), shape)  # une étude par modèle, comme dans fit
        assert params["class_weight"] in (None, "balanced"), cand.name
        cand.build(params)  # le modèle accepte bien ce réglage


def test_skewed_target_is_learned_in_log():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 5)
    y = np.exp(rng.randn(500) * 1.5)  # cible positive très asymétrique
    profile = inspect_data(X, ["Numerical"] * 5, y)
    profile.task = detect_task(y)
    assert profile.target_skewed
    for cand in get_candidates(profile, n_jobs=1):
        assert isinstance(cand.build(), TransformedTargetRegressor), cand.name


def test_averaging_ensemble_returns_class_indices():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = np.argmax(X[:, :3], axis=1)  # 3 classes, encodées 0..2
    task = detect_task(np.eye(3)[y])
    ens = AveragingEnsemble([LogisticRegression(C=c) for c in (0.1, 1.0, 10.0)], task).fit(X, y)
    pred = ens.predict(X)
    assert pred.shape == (300,) and set(np.unique(pred)) <= {0, 1, 2}
    assert np.mean(pred == y) > 0.8
