"""Gestion des ressources (temps, CPU, essais HPO) et de la reproductibilité.

Ce module lit les variables d'environnement imposées lors de l'évaluation sur Skinner :
- AUTOML_TIME_BUDGET : temps maximal (en secondes) pour un appel à `fit`
- AUTOML_N_JOBS      : nombre de cœurs autorisés (jamais -1 !)
- AUTOML_N_TRIALS    : nombre maximal d'essais Optuna pour l'optimisation
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_SEED = 42


def _lire_entier_env(nom: str, defaut: int, minimum: int = 1) -> int:
    """Lit une variable d'environnement entière avec repli sécurisé."""
    val_brute = os.environ.get(nom)
    if val_brute is None or not val_brute.strip():
        return defaut
    try:
        val = int(float(val_brute.strip()))
        return max(minimum, val)
    except ValueError:
        return defaut


def _lire_flottant_env(nom: str, defaut: float, minimum: float = 1.0) -> float:
    """Lit une variable d'environnement flottante avec repli sécurisé."""
    val_brute = os.environ.get(nom)
    if val_brute is None or not val_brute.strip():
        return defaut
    try:
        val = float(val_brute.strip())
        return max(minimum, val)
    except ValueError:
        return defaut


def fixer_graine(seed: int = DEFAULT_SEED) -> int:
    """Fixe toutes les graines aléatoires pour garantir une reproductibilité stricte."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


@dataclass(frozen=True)
class ConfigRessources:
    """Configuration des ressources allouées au pipeline AutoML."""

    time_budget: float = 120.0
    n_jobs: int = 1
    n_trials: int = 25
    seed: int = DEFAULT_SEED

    @classmethod
    def depuis_env(
        cls,
        time_budget: Optional[float] = None,
        n_jobs: Optional[int] = None,
        n_trials: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "ConfigRessources":
        """Construit la configuration en priorité depuis les arguments puis les variables d'env."""
        budget = (
            float(time_budget)
            if time_budget is not None
            else _lire_flottant_env("AUTOML_TIME_BUDGET", 120.0, minimum=1.0)
        )

        # Règle Skinner : ne jamais utiliser n_jobs=-1
        if n_jobs is not None:
            jobs = max(1, int(n_jobs))
        else:
            raw_jobs = os.environ.get("AUTOML_N_JOBS")
            if raw_jobs is not None and raw_jobs.strip():
                try:
                    parsed = int(raw_jobs.strip())
                    # Si -1 ou <= 0 est passé dans l'env, on force prudemment 1 cœur
                    jobs = parsed if parsed >= 1 else 1
                except ValueError:
                    jobs = 1
            else:
                jobs = 1

        trials = (
            max(0, int(n_trials))
            if n_trials is not None
            else _lire_entier_env("AUTOML_N_TRIALS", 25, minimum=0)
        )

        graine = (
            int(seed)
            if seed is not None
            else _lire_entier_env("AUTOML_SEED", DEFAULT_SEED, minimum=0)
        )

        return cls(
            time_budget=budget,
            n_jobs=jobs,
            n_trials=trials,
            seed=graine,
        )


class ChronoBudget:
    """Suivi en temps réel du budget temps restant pendant `fit()`."""

    def __init__(self, total_budget: float) -> None:
        self.total_budget = max(1.0, float(total_budget))
        self.debut = time.perf_counter()

    @property
    def ecoule(self) -> float:
        """Temps écoulé en secondes depuis le démarrage de `fit()`."""
        return time.perf_counter() - self.debut

    @property
    def restant(self) -> float:
        """Temps restant estimé en secondes."""
        return max(0.0, self.total_budget - self.ecoule)

    @property
    def progression(self) -> float:
        """Fraction du budget déjà consommée (entre 0.0 et 1.0+)."""
        return self.ecoule / self.total_budget

    def a_du_temps(self, marge_securite: float = 2.0, minimum_requis: float = 0.5) -> bool:
        """Vérifie s'il reste suffisamment de temps pour lancer une nouvelle étape."""
        return (self.restant - marge_securite) >= minimum_requis
