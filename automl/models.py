"""Étapes 3 et 4 : modèles candidats, leurs hyperparamètres et leur coût estimé.

Chaque candidat fournit :
- des réglages par défaut (essayés en premier) ;
- un espace de recherche pour Optuna (`space`) ;
- une formule de coût estimé, qui ne dépend que de la taille des données et des
  réglages : c'est elle qui permet de planifier l'entraînement dans le budget
  temps de façon reproductible.

Les « gains rapides » sont appliqués ici selon le profil des données :
- quand les classes sont déséquilibrées, Optuna essaie aussi
  `class_weight="balanced"` (c'est la validation qui décide : sur data_A,
  l'imposer faisait perdre 7 points à la régression logistique) ;
- transformation log de la cible (`TransformedTargetRegressor`) quand une cible
  de régression positive est asymétrique.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np
import scipy.sparse as sp
from sklearn.compose import TransformedTargetRegressor
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
from .inspection import (
    MULTICLASS,
    MULTICLASS_ONEHOT,
    MULTILABEL,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    DataProfile,
)

Params = Dict[str, Any]


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

CostFn = Callable[[DataShape, int, Params], float]


def linear_cost(k_unit: float, per_class: bool = False) -> CostFn:
    """Modèles linéaires : proportionnel à lignes x non-nuls x sorties.

    `per_class` : le modèle entraîne un sous-modèle par classe (un-contre-tous,
    comme LinearSVC). La régression logistique traite toutes les classes d'un coup.
    Un C élevé (peu de régularisation) demande plus d'itérations au solveur.
    """
    def cost(s: DataShape, n: int, p: Params) -> float:
        k = s.n_outputs * (s.n_classes if per_class else 1)
        iterations = 1.0 + max(0.0, math.log10(p.get("C", 1.0)))
        return SAFETY * (OVERHEAD + k_unit * n * s.nnz_per_row * k * iterations)
    return cost


def forest_cost(s: DataShape, n: int, p: Params) -> float:
    """Forêt : arbres x n log n x variables testées par nœud, réparti sur les cœurs."""
    frac = p.get("max_features", "sqrt")
    m = math.sqrt(s.d) if frac == "sqrt" else max(1.0, s.d * frac)
    return SAFETY * (OVERHEAD + 5e-8 * p["n_estimators"] * n * math.log2(max(n, 2)) * m / s.n_jobs)


def boosting_cost(s: DataShape, n: int, p: Params) -> float:
    """HistGradientBoosting : itérations x arbres par itération x lignes x colonnes.

    Un arbre par classe à chaque itération, et un modèle par sortie en multi-label.
    Plus de feuilles par arbre = un peu plus de calcul.
    """
    leaves = math.sqrt(p.get("max_leaf_nodes", 31) / 31)
    return SAFETY * (OVERHEAD + 6e-8 * p["max_iter"] * s.n_classes * s.n_outputs
                     * n * s.d * leaves / s.n_jobs)


def no_space(trial, shape: DataShape) -> Params:
    return {}


@dataclass
class Candidate:
    """Un modèle candidat.

    - `factory(params)` construit le modèle scikit-learn de base ;
    - `defaults` : réglages essayés en premier ;
    - `space(trial, shape)` : réglages proposés par Optuna ;
    - `cost(shape, n, params)` : durée d'entraînement estimée sur n lignes ;
    - `has_proba` : le modèle donne des probabilités (utile pour l'ensembling).
    """

    name: str
    factory: Callable[[Params], Any]
    cost: CostFn
    defaults: Params = field(default_factory=dict)
    space: Callable[[Any, DataShape], Params] = no_space
    has_proba: bool = True
    wrap: Callable[[Any], Any] = lambda m: m

    def build(self, params: Params = None) -> Any:
        """Modèle neuf, adapté à la tâche (multi-sorties, cible log)."""
        return self.wrap(self.factory({**self.defaults, **(params or {})}))

    def estimate(self, shape: DataShape, n: int, params: Params = None) -> float:
        return self.cost(shape, n, {**self.defaults, **(params or {})})


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


def rows_within(cost: Callable[[int], float], n_max: int, seconds: float, n_min: int) -> int:
    """Plus grand nombre de lignes (<= n_max) dont le coût estimé `cost(n)` tient en `seconds`.

    Renvoie 0 si même `n_min` lignes ne tiennent pas. Le résultat ne dépend que de
    la taille des données et du budget : il est identique d'une exécution à l'autre.
    """
    if cost(n_max) <= seconds:
        return n_max
    lo, hi = min(n_min, n_max), n_max
    if cost(lo) > seconds:
        return 0
    while hi - lo > 1:  # le coût croît avec n : recherche dichotomique
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cost(mid) <= seconds else (lo, mid)
    return lo


# ---------------------------------------------------------------------------
# Espaces de recherche Optuna
# ---------------------------------------------------------------------------

def _space_C(low: float, high: float):
    return lambda trial, s: {"C": trial.suggest_float("C", low, high, log=True)}


def _space_alpha(low: float, high: float):
    return lambda trial, s: {"alpha": trial.suggest_float("alpha", low, high, log=True)}


def _space_forest(regression: bool):
    def space(trial, s: DataShape) -> Params:
        if regression:
            frac = trial.suggest_float("max_features", 0.3, 1.0)
        else:
            # Autour de sqrt(d) : de la moitié à trois fois plus de variables par nœud
            base = math.sqrt(s.d) / s.d
            frac = trial.suggest_float("max_features", max(0.5 * base, 1.0 / s.d), min(1.0, 3 * base), log=True)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_features": frac,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    return space


def _space_boosting(trial, s: DataShape) -> Params:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# Liste des candidats
# ---------------------------------------------------------------------------

def _task_wrapper(profile: DataProfile, native_multi_output: bool) -> Callable[[Any], Any]:
    """Adapte un modèle mono-sortie aux tâches à plusieurs sorties, et applique
    la transformation log quand la cible de régression est asymétrique."""
    task = profile.task.task

    def wrap(model):
        if not native_multi_output:
            if task == MULTILABEL:
                model = MultiOutputClassifier(model)
            elif task == MULTIOUTPUT_REGRESSION:
                model = MultiOutputRegressor(model)
        if task == REGRESSION and profile.target_skewed:
            # La cible positive et asymétrique est apprise en log, puis remise à l'échelle
            model = TransformedTargetRegressor(regressor=model, func=np.log1p,
                                               inverse_func=np.expm1, check_inverse=False)
        return model
    return wrap


def _with_class_weight(space):
    """Ajoute le choix class_weight (aucun / équilibré) à un espace de recherche."""
    def combined(trial, s: DataShape) -> Params:
        return {**space(trial, s),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    return combined


def get_candidates(profile: DataProfile, n_jobs: int) -> List[Candidate]:
    """Liste des modèles à essayer, du moins coûteux au plus coûteux.

    Le premier modèle est toujours entraîné : il doit être rapide pour que `fit`
    ne dépasse jamais le budget, même sur un très gros jeu.
    """
    rs = RANDOM_STATE
    wrap = _task_wrapper(profile, native_multi_output=False)
    wrap_native = _task_wrapper(profile, native_multi_output=True)

    if profile.task.is_classification:
        # Gain rapide : classes déséquilibrées -> Optuna essaie aussi de les rééquilibrer
        cw = _with_class_weight if profile.is_imbalanced else (lambda space: space)
        if profile.is_sparse:
            # Texte / sac de mots : les modèles linéaires sont rapides et très bons
            cands = [
                Candidate("linear_svc", lambda p: LinearSVC(random_state=rs, **p),
                          linear_cost(6e-8, per_class=True), {"C": 0.5},
                          cw(_space_C(1e-2, 10.0)), has_proba=False, wrap=wrap),
                Candidate("logistic_regression", lambda p: LogisticRegression(max_iter=300, random_state=rs, **p),
                          linear_cost(1.5e-5), {"C": 4.0}, cw(_space_C(1e-1, 100.0)), wrap=wrap),
            ]
            if profile.nonnegative_integer:  # Naive Bayes exige des valeurs >= 0
                cands.append(Candidate("complement_nb", lambda p: ComplementNB(**p),
                                       linear_cost(1e-8, per_class=True), {"alpha": 0.3},
                                       _space_alpha(1e-2, 3.0), wrap=wrap))
            return cands
        return [
            Candidate("logistic_regression", lambda p: LogisticRegression(max_iter=500, random_state=rs, **p),
                      linear_cost(2e-7), {"C": 1.0}, cw(_space_C(1e-2, 100.0)), wrap=wrap),
            Candidate("random_forest", lambda p: RandomForestClassifier(n_jobs=n_jobs, random_state=rs, **p),
                      forest_cost, {"n_estimators": 300, "max_features": "sqrt"},
                      cw(_space_forest(regression=False)), wrap=wrap_native),
            Candidate("hist_gradient_boosting",
                      lambda p: HistGradientBoostingClassifier(early_stopping=True, random_state=rs, **p),
                      boosting_cost, {"max_iter": 300, "learning_rate": 0.06},
                      cw(_space_boosting), wrap=wrap),
        ]

    if profile.is_sparse:
        return [Candidate("ridge", lambda p: Ridge(**p), linear_cost(1e-7), {"alpha": 1.0},
                          _space_alpha(1e-2, 1e3), wrap=wrap_native)]
    return [
        Candidate("ridge", lambda p: Ridge(**p), linear_cost(2e-7), {"alpha": 1.0},
                  _space_alpha(1e-2, 1e3), wrap=wrap_native),
        Candidate("random_forest", lambda p: RandomForestRegressor(n_jobs=n_jobs, random_state=rs, **p),
                  forest_cost, {"n_estimators": 300, "max_features": 0.7},
                  _space_forest(regression=True), wrap=wrap_native),
        Candidate("hist_gradient_boosting",
                  lambda p: HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=30,
                                                          random_state=rs, **p),
                  boosting_cost, {"max_iter": 700, "learning_rate": 0.06},
                  _space_boosting, wrap=wrap),
    ]
