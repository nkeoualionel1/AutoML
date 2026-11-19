"""Paramètres globaux, ressources imposées par le challenge et reproductibilité.

Variables d'environnement lues au moment de `fit` (pas à l'import, pour que
l'évaluateur puisse les fixer à tout moment) :
- AUTOML_TIME_BUDGET : temps maximal (secondes) pour un appel à `fit`
- AUTOML_N_JOBS      : nombre de cœurs autorisés (jamais -1)
- AUTOML_N_TRIALS    : nombre maximal d'essais d'optimisation
"""

import os
import random
import time
from dataclasses import dataclass

import numpy as np

# Graine unique utilisée partout pour garantir la reproductibilité
RANDOM_STATE = 42


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def limit_native_threads(n_jobs: int) -> None:
    """Limite les threads OpenMP / BLAS à `n_jobs`.

    HistGradientBoosting n'a pas de paramètre n_jobs : sans cette limite il
    utilise tous les cœurs de la machine, ce qui viole AUTOML_N_JOBS. Une valeur
    déjà fixée par l'utilisateur ou le cluster est conservée.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(1, n_jobs)))


def set_seed(seed: int = RANDOM_STATE) -> int:
    """Fixe toutes les graines (Python, NumPy, hachage) pour la reproductibilité."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@dataclass(frozen=True)
class ResourceConfig:
    """Ressources allouées au pipeline, lues depuis l'environnement."""

    time_budget: float
    n_jobs: int
    n_trials: int

    @classmethod
    def from_env(cls) -> "ResourceConfig":
        return cls(
            time_budget=max(1.0, _env_float("AUTOML_TIME_BUDGET", 600.0)),
            n_jobs=max(1, _env_int("AUTOML_N_JOBS", 1)),  # -1 ou 0 -> 1 cœur
            n_trials=max(0, _env_int("AUTOML_N_TRIALS", 20)),
        )


class TimeBudget:
    """Chronomètre du budget temps de `fit`."""

    def __init__(self, total: float) -> None:
        self.total = max(1.0, float(total))
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed)

    def has_time(self, needed: float = 0.0, margin: float = 2.0) -> bool:
        """Vrai s'il reste au moins `needed` secondes en plus de la marge de sécurité."""
        return self.remaining - margin >= needed
