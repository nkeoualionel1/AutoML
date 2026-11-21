"""Étape 3 : modèles candidats selon la tâche et le format des données.

Version de base : quelques modèles scikit-learn bien choisis, avec leurs
réglages par défaut. L'optimisation des hyperparamètres (Optuna) viendra
ensuite, en s'appuyant sur cette liste.
"""

import math
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
import scipy.sparse as sp

from .inspection import MULTICLASS, MULTICLASS_ONEHOT, MULTILABEL, MULTIOUTPUT_REGRESSION, DataProfile


@dataclass
class DataShape:
    """Taille des données prétraitées : seule base des estimations de coût."""

    d: int                  # nombre de colonnes
    nnz_per_row: float      # valeurs non nulles par ligne (= d pour du dense)
    n_classes: int          # classes en multiclasse, 1 sinon
    n_outputs: int          # sorties en multi-label / multi-sorties, 1 sinon
    n_jobs: int


# Coût estimé (secondes) = constante x « unités de travail ». Constantes mesurées
# sur les 6 jeux (PC portable, 2 cœurs), puis multipliées par SAFETY : il vaut
# mieux surestimer (un modèle de moins) que dépasser le budget.
SAFETY = 1.5
OVERHEAD = 0.05


def linear_cost(k_unit: float, per_class: bool = False) -> Callable[[DataShape, int], float]:
    """Modèles linéaires : proportionnel à lignes x non-nuls x sorties.

    `per_class` : le modèle entraîne un sous-modèle par classe (un-contre-tous,
    comme LinearSVC). La régression logistique traite toutes les classes d'un coup.
    """
    def cost(s: DataShape, n: int) -> float:
        k = s.n_outputs * (s.n_classes if per_class else 1)
        return SAFETY * (OVERHEAD + k_unit * n * s.nnz_per_row * k)
    return cost


def forest_cost(n_estimators: int, feature_frac: float = None) -> Callable[[DataShape, int], float]:
    """Forêt : arbres x n log n x variables testées par nœud, réparti sur les cœurs."""
    def cost(s: DataShape, n: int) -> float:
        m = s.d * feature_frac if feature_frac else math.sqrt(s.d)
        return SAFETY * (OVERHEAD + 5e-8 * n_estimators * n * math.log2(max(n, 2)) * m / s.n_jobs)
    return cost


def boosting_cost(max_iter: int) -> Callable[[DataShape, int], float]:
    """HistGradientBoosting : itérations x arbres par itération x lignes x colonnes.

    Un arbre par classe à chaque itération, et un modèle par sortie en multi-label.
    """
    return lambda s, n: SAFETY * (
        OVERHEAD + 6e-8 * max_iter * s.n_classes * s.n_outputs * n * s.d / s.n_jobs)


@dataclass
class Candidate:
    """Un modèle candidat : un nom, une fonction qui construit un modèle neuf,
    et une estimation de son temps d'entraînement sur n lignes."""

    name: str
    build: Callable[[], Any]
    cost: Callable[[DataShape, int], float]


def _for_task(model: Any, task: str, native_multi_output: bool) -> Any:
    """Adapte un modèle mono-sortie aux tâches à plusieurs sorties."""
    if native_multi_output:
        return model
    if task == MULTILABEL:
        return MultiOutputClassifier(model)
    if task == MULTIOUTPUT_REGRESSION:
        return MultiOutputRegressor(model)
    return model


def data_shape(X, task_info, n_jobs: int) -> DataShape:
    """Mesure la taille des données prétraitées pour les estimations de coût."""
    n, d = X.shape
    nnz_per_row = X.nnz / max(n, 1) if sp.issparse(X) else float(d)
    multiclass = task_info.task in (MULTICLASS, MULTICLASS_ONEHOT)
    multi_output = task_info.task in (MULTILABEL, MULTIOUTPUT_REGRESSION)
    return DataShape(
        d=d, nnz_per_row=nnz_per_row,
        n_classes=max(1, len(task_info.classes)) if multiclass else 1,
        n_outputs=max(1, task_info.n_outputs) if multi_output else 1,
        n_jobs=max(1, n_jobs),
    )


def rows_within(cand: Candidate, shape: DataShape, n_max: int, seconds: float, n_min: int) -> int:
    """Plus grand nombre de lignes (<= n_max) dont l'entraînement estimé tient en `seconds`.

    Renvoie 0 si même `n_min` lignes ne tiennent pas. Le résultat ne dépend que de
    la taille des données et du budget : il est identique d'une exécution à l'autre.
    """
    if cand.cost(shape, n_max) <= seconds:
        return n_max
    lo, hi = min(n_min, n_max), n_max
    if cand.cost(shape, lo) > seconds:
        return 0
    while hi - lo > 1:  # le coût croît avec n : recherche dichotomique
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cand.cost(shape, mid) <= seconds else (lo, mid)
    return lo


def get_candidates(profile: DataProfile, n_jobs: int) -> List[Candidate]:
    """Liste des modèles à essayer, du moins coûteux au plus coûteux.

    Le premier modèle est toujours entraîné : il doit être rapide pour que `fit`
    ne dépasse jamais le budget, même sur un très gros jeu.
    """
    task = profile.task.task
    rs = RANDOM_STATE

    def c(name: str, factory: Callable[[], Any], cost, native: bool = False) -> Candidate:
        return Candidate(name, lambda: _for_task(factory(), task, native), cost)

    if profile.task.is_classification:
        if profile.is_sparse:
            # Texte / sac de mots : les modèles linéaires sont rapides et très bons
            cands = [
                c("linear_svc", lambda: LinearSVC(C=0.5, random_state=rs), linear_cost(6e-8, per_class=True)),
                c("logistic_regression", lambda: LogisticRegression(C=4.0, max_iter=300, random_state=rs),
                  linear_cost(1.5e-5)),
            ]
            if profile.nonnegative_integer:  # Naive Bayes exige des valeurs >= 0
                cands.append(c("complement_nb", lambda: ComplementNB(alpha=0.3),
                               linear_cost(1e-8, per_class=True)))
            return cands
        return [
            c("logistic_regression", lambda: LogisticRegression(max_iter=500, random_state=rs),
              linear_cost(2e-7)),
            c("random_forest", lambda: RandomForestClassifier(
                n_estimators=300, n_jobs=n_jobs, random_state=rs), forest_cost(300), native=True),
            c("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, early_stopping=True, random_state=rs), boosting_cost(300)),
        ]

    if profile.is_sparse:
        return [c("ridge", lambda: Ridge(alpha=1.0), linear_cost(1e-7), native=True)]
    return [
        c("ridge", lambda: Ridge(alpha=1.0), linear_cost(2e-7), native=True),
        c("random_forest", lambda: RandomForestRegressor(
            n_estimators=300, max_features=0.7, n_jobs=n_jobs, random_state=rs),
          forest_cost(300, feature_frac=0.7), native=True),
        c("hist_gradient_boosting", lambda: HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.06, early_stopping=True, n_iter_no_change=30, random_state=rs),
          boosting_cost(700)),
    ]
