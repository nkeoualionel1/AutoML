"""Tests de automl/config.py : lecture des ressources imposées par le sujet.

Rôle : vérifier que AUTOML_TIME_BUDGET, AUTOML_N_JOBS et AUTOML_N_TRIALS sont
lus au moment du fit (et non à l'import), et que les valeurs interdites
(n_jobs = -1 ou 0) sont ramenées à une valeur valide.

Quand le lancer : après toute modification de automl/config.py, et avant chaque
fusion de branche (`python -m pytest tests`).
"""

from automl.config import ResourceConfig


def test_resources_read_at_fit_time(monkeypatch):
    monkeypatch.setenv("AUTOML_N_JOBS", "-1")
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "45")
    cfg = ResourceConfig.from_env()
    assert cfg.n_jobs == 1 and cfg.time_budget == 45.0
