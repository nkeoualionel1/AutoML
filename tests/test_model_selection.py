"""Tests de la sélection du modèle sous budget temps (automl/models.py, __init__.py).

Rôle : vérifier deux exigences du sujet sur l'étape « choix du modèle » :
- `fit` respecte AUTOML_TIME_BUDGET ;
- le choix est reproductible : deux `fit` sur les mêmes données essaient les mêmes
  modèles sur le même nombre de lignes et donnent les mêmes prédictions (le plan
  d'entraînement dépend des coûts estimés, pas du chronomètre).

Quand le lancer : après toute modification des modèles candidats, des formules
de coût ou de `_select_model` (et donc pendant tout le travail sur Optuna et
l'ensembling), avant chaque fusion de branche (`python -m pytest tests`).
"""

import time

import numpy as np

from automl import AutoML
from automl.models import Candidate, DataShape, forest_cost, rows_within


def test_budget_respected_and_choice_reproducible(tmp_path, monkeypatch, write_dataset, dense_lines):
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "15")
    monkeypatch.setenv("AUTOML_N_JOBS", "2")
    rng = np.random.RandomState(1)
    X = rng.randn(4000, 40)
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
    train = write_dataset(tmp_path / "train", "medium", dense_lines(X), ["Numerical"] * 40, [str(v) for v in y])
    test = write_dataset(tmp_path / "test", "medium", dense_lines(X[:50]), ["Numerical"] * 40)

    runs = []
    for _ in range(2):
        start = time.perf_counter()
        m = AutoML().fit(train)
        runs.append((time.perf_counter() - start, m))
    for elapsed, _ in runs:
        assert elapsed < 15 * 1.2, f"fit a duré {elapsed:.1f}s pour un budget de 15s"
    plans = [[(r["model"], r["rows"]) for r in m.leaderboard] for _, m in runs]
    assert plans[0] == plans[1]
    assert np.array_equal(runs[0][1].predict(test), runs[1][1].predict(test))


def test_rows_within_matches_cost_estimate():
    cand = Candidate("rf", lambda: None, forest_cost(300))
    shape = DataShape(d=100, nnz_per_row=100, n_classes=1, n_outputs=1, n_jobs=2)
    n = rows_within(cand, shape, 100000, seconds=10, n_min=500)
    assert 500 <= n < 100000 and cand.cost(shape, n) <= 10 < cand.cost(shape, n + 1)
    assert rows_within(cand, shape, 100000, seconds=0.01, n_min=500) == 0
