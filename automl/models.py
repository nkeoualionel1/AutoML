"""Étape 3 : modèles candidats selon la tâche et le format des données.

Version de base : quelques modèles scikit-learn bien choisis, avec leurs
réglages par défaut. L'optimisation des hyperparamètres (Optuna) viendra
ensuite, en s'appuyant sur cette liste.
"""

from dataclasses import dataclass
from typing import Any, Callable, List

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE
from .inspection import MULTILABEL, MULTIOUTPUT_REGRESSION, DataProfile


@dataclass
class Candidate:
    """Un modèle candidat : un nom et une fonction qui construit un modèle neuf."""

    name: str
    build: Callable[[], Any]


def _for_task(model: Any, task: str, native_multi_output: bool) -> Any:
    """Adapte un modèle mono-sortie aux tâches à plusieurs sorties."""
    if native_multi_output:
        return model
    if task == MULTILABEL:
        return MultiOutputClassifier(model)
    if task == MULTIOUTPUT_REGRESSION:
        return MultiOutputRegressor(model)
    return model


def get_candidates(profile: DataProfile, n_jobs: int) -> List[Candidate]:
    """Liste des modèles à essayer, du moins coûteux au plus coûteux.

    Le premier modèle est toujours entraîné : il doit être rapide pour que `fit`
    ne dépasse jamais le budget, même sur un très gros jeu.
    """
    task = profile.task.task
    rs = RANDOM_STATE

    def c(name: str, factory: Callable[[], Any], native: bool = False) -> Candidate:
        return Candidate(name, lambda: _for_task(factory(), task, native))

    if profile.task.is_classification:
        if profile.is_sparse:
            # Texte / sac de mots : les modèles linéaires sont rapides et très bons
            cands = [
                c("linear_svc", lambda: LinearSVC(C=0.5, random_state=rs)),
                c("logistic_regression", lambda: LogisticRegression(C=4.0, max_iter=300, random_state=rs)),
            ]
            if profile.nonnegative_integer:  # Naive Bayes exige des valeurs >= 0
                cands.append(c("complement_nb", lambda: ComplementNB(alpha=0.3)))
            return cands
        return [
            c("logistic_regression", lambda: LogisticRegression(max_iter=500, random_state=rs)),
            c("random_forest", lambda: RandomForestClassifier(
                n_estimators=300, n_jobs=n_jobs, random_state=rs), native=True),
            c("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, early_stopping=True, random_state=rs)),
        ]

    if profile.is_sparse:
        return [c("ridge", lambda: Ridge(alpha=1.0), native=True)]
    return [
        c("ridge", lambda: Ridge(alpha=1.0), native=True),
        c("random_forest", lambda: RandomForestRegressor(
            n_estimators=300, max_features=0.7, n_jobs=n_jobs, random_state=rs), native=True),
        c("hist_gradient_boosting", lambda: HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.06, early_stopping=True, n_iter_no_change=30, random_state=rs)),
    ]
