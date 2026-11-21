"""Évaluation des modèles (validation simple ou croisée) et ensembling.

Le même évaluateur sert aux trois phases du choix du modèle : comparaison des
candidats, optimisation Optuna et ensembling. Les scores sont donc comparables
entre eux.

- **Grand jeu** : on entraîne sur le train et on note sur la validation.
- **Petit jeu** (<= CV_MAX_ROWS lignes train + valid) : validation croisée en
  CV_FOLDS plis sur train + valid. Avec ~500 lignes de validation, un score
  simple varie de plusieurs points selon le hasard du découpage : la moyenne
  sur 5 plis est bien plus fiable pour départager deux modèles proches.

Pour l'ensembling, l'évaluateur garde les « prédictions douces » de chaque
modèle (probabilités, ou valeurs en régression) sur les lignes de validation
(ou hors-pli en validation croisée) : un ensemble se note alors en moyennant
ces prédictions, sans réentraîner aucun modèle.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold, StratifiedKFold

from .config import RANDOM_STATE
from .inspection import BINARY, MULTICLASS, MULTICLASS_ONEHOT, MULTILABEL, TaskInfo
from .metrics import all_metrics
from .models import Candidate, DataShape, Params

CV_MAX_ROWS = 5000
CV_FOLDS = 5


def stack_rows(a, b):
    """Empile deux matrices (denses ou creuses) ligne par ligne."""
    return sp.vstack([a, b], format="csr") if sp.issparse(a) else np.vstack([a, b])


def _n_classes(task: TaskInfo) -> int:
    return len(task.classes) if task.classes is not None else 2


def soft_predict(model, X, task: TaskInfo) -> Optional[np.ndarray]:
    """Prédictions « douces » d'un modèle entraîné, alignées sur toutes les classes.

    - binaire / multiclasse : probabilités (n, K) ;
    - multi-label : probabilité de chaque label à 1, (n, L) ;
    - régression : les prédictions elles-mêmes.
    Renvoie None si le modèle ne donne pas de probabilités (LinearSVC).
    """
    t = task.task
    if t in (BINARY, MULTICLASS, MULTICLASS_ONEHOT):
        if not hasattr(model, "predict_proba"):
            return None
        out = np.zeros((X.shape[0], _n_classes(task)))
        # Un modèle entraîné sur un sous-échantillon peut ne pas avoir vu toutes les classes
        out[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
        return out
    if t == MULTILABEL:
        if not hasattr(model, "predict_proba"):
            return None
        out = np.zeros((X.shape[0], task.n_outputs))
        for j, (p, classes) in enumerate(zip(model.predict_proba(X), model.classes_)):
            classes = list(np.asarray(classes).astype(int))
            out[:, j] = p[:, classes.index(1)] if 1 in classes else 0.0
        return out
    return np.asarray(model.predict(X), dtype=np.float64)


def hard_from_soft(soft: np.ndarray, task: TaskInfo) -> np.ndarray:
    """Prédictions douces -> prédictions au format encodé (classe, 0/1, valeur)."""
    if task.task in (BINARY, MULTICLASS, MULTICLASS_ONEHOT):
        return np.argmax(soft, axis=1)
    if task.task == MULTILABEL:
        return (soft >= 0.5).astype(int)
    return soft


def evaluation_cost(cand: Candidate, shape: DataShape, rows: int, params: Params, use_cv: bool) -> float:
    if use_cv:
        return CV_FOLDS * cand.estimate(shape, int(rows * (CV_FOLDS - 1) / CV_FOLDS), params)
    return cand.estimate(shape, rows, params)


def choose_cv(candidates: List[Candidate], shape: DataShape, n_all: int, seconds: float) -> bool:
    """Validation croisée seulement pour un petit jeu, et si tous les candidats y tiennent
    dans `seconds` (estimation) : sinon la validation simple permet d'essayer plus de modèles."""
    if n_all > CV_MAX_ROWS:
        return False
    return sum(evaluation_cost(c, shape, n_all, None, True) for c in candidates) <= seconds


@dataclass
class Result:
    """Un modèle évalué : réglages, score, prédictions douces, modèle entraîné."""

    name: str
    cand: Candidate
    params: Params
    rows: int
    score: float
    metrics: dict
    soft: Optional[np.ndarray]
    model: Any = None          # modèle entraîné (validation simple seulement)
    estimated_s: float = 0.0
    duration_s: float = 0.0


class Evaluator:
    """Note un candidat avec des réglages donnés, en validation simple ou croisée."""

    def __init__(self, task: TaskInfo, X_train, y_train, X_valid, y_valid, use_cv: bool):
        self.task = task
        self.use_cv = use_cv
        if self.use_cv:
            self.X = stack_rows(X_train, X_valid)
            self.y = np.concatenate([y_train, y_valid])
            self.folds = self._make_folds()
        else:
            self.X, self.y = X_train, y_train
            self.X_valid, self.y_valid = X_valid, y_valid
        self.n_max = self.X.shape[0]

    @property
    def y_eval(self) -> np.ndarray:
        """Cible des lignes sur lesquelles les modèles sont notés."""
        return self.y if self.use_cv else self.y_valid

    def _make_folds(self):
        labels = self.task.stratify_labels(self.y)
        if labels is not None and np.bincount(np.unique(labels, return_inverse=True)[1]).min() >= CV_FOLDS:
            splitter = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            return list(splitter.split(np.zeros(len(self.y)), labels))
        splitter = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(np.zeros(len(self.y))))

    def cost(self, cand: Candidate, shape: DataShape, rows: int, params: Params = None) -> float:
        """Coût estimé d'une évaluation (CV_FOLDS entraînements en validation croisée)."""
        return evaluation_cost(cand, shape, rows, params, self.use_cv)

    def evaluate(self, cand: Candidate, params: Params, rows: int) -> Result:
        task = self.task.task
        if self.use_cv:
            # Prédictions hors-pli : chaque ligne est prédite par un modèle qui ne l'a pas vue
            soft_parts, hard = [], np.empty_like(self.y)
            soft = None
            for train_idx, test_idx in self.folds:
                model = cand.build(params).fit(self.X[train_idx], self.y[train_idx])
                hard[test_idx] = model.predict(self.X[test_idx])
                s = soft_predict(model, self.X[test_idx], self.task) if cand.has_proba else None
                soft_parts.append((test_idx, s))
            if all(s is not None for _, s in soft_parts):
                first = soft_parts[0][1]
                soft = np.zeros((len(self.y),) + first.shape[1:])
                for idx, s in soft_parts:
                    soft[idx] = s
            metrics = all_metrics(task, self.y, hard)
            return Result(cand.name, cand, params, rows, metrics["score"], metrics, soft)

        model = cand.build(params).fit(self.X[:rows], self.y[:rows])
        hard = model.predict(self.X_valid)
        soft = soft_predict(model, self.X_valid, self.task) if cand.has_proba else None
        metrics = all_metrics(task, self.y_valid, hard)
        return Result(cand.name, cand, params, rows, metrics["score"], metrics, soft, model=model)

    def score_ensemble(self, members: List[Result]) -> dict:
        """Note la moyenne des prédictions douces de plusieurs modèles, sans réentraîner."""
        soft = np.mean([m.soft for m in members], axis=0)
        return all_metrics(self.task.task, self.y_eval, hard_from_soft(soft, self.task))


class AveragingEnsemble:
    """Ensemble final : moyenne des probabilités (ou des prédictions en régression)."""

    def __init__(self, models: List[Any], task: TaskInfo):
        self.models = models
        self.task = task

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        soft = np.mean([soft_predict(m, X, self.task) for m in self.models], axis=0)
        return hard_from_soft(soft, self.task)
