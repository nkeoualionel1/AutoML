"""Étape 3 : métriques d'évaluation adaptées à chaque tâche.

Chaque tâche a un **score principal** (plus grand = meilleur) qui sert à choisir
le modèle, et un dictionnaire de métriques détaillées renvoyé par `eval()`.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .inspection import BINARY, MULTICLASS, MULTICLASS_ONEHOT, MULTILABEL


def primary_score(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score unique pour comparer les modèles (y au format encodé par TaskInfo)."""
    if task in (BINARY, MULTICLASS, MULTICLASS_ONEHOT):
        # Moyenne accuracy / F1 macro : l'accuracy seule favorise la classe majoritaire
        return 0.5 * accuracy_score(y_true, y_pred) + 0.5 * f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
    if task == MULTILABEL:
        return 0.5 * f1_score(y_true, y_pred, average="macro", zero_division=0) + 0.5 * float(
            np.mean(y_true == y_pred)
        )
    return float(r2_score(y_true, y_pred))


def all_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Métriques détaillées affichées par `eval()`."""
    m: Dict[str, float] = {"score": primary_score(task, y_true, y_pred)}
    if task in (BINARY, MULTICLASS, MULTICLASS_ONEHOT):
        m["accuracy"] = float(accuracy_score(y_true, y_pred))
        m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    elif task == MULTILABEL:
        m["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
        m["label_accuracy"] = float(np.mean(y_true == y_pred))
        m["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        m["r2"] = float(r2_score(y_true, y_pred))
        m["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        m["mae"] = float(mean_absolute_error(y_true, y_pred))
    return m
