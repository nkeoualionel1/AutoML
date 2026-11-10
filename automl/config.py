"""Paramètres globaux et ressources imposées par le challenge."""

import os

# Graine unique utilisée partout pour garantir la reproductibilité
RANDOM_STATE = 42


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Ressources transmises par variables d'environnement lors de l'évaluation
TIME_BUDGET = _env_float("AUTOML_TIME_BUDGET", 600.0)
N_JOBS = max(1, _env_int("AUTOML_N_JOBS", 1))  # jamais n_jobs=-1
N_TRIALS = max(1, _env_int("AUTOML_N_TRIALS", 20))
