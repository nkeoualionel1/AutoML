"""Étape 1 : inspection des données et détection automatique de la tâche."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

# Noms des tâches reconnues
BINARY = "binary"
MULTICLASS = "multiclass"
MULTICLASS_ONEHOT = "multiclass_onehot"
MULTILABEL = "multilabel"
REGRESSION = "regression"
MULTIOUTPUT_REGRESSION = "multioutput_regression"

CLASSIFICATION_TASKS = (BINARY, MULTICLASS, MULTICLASS_ONEHOT, MULTILABEL)

# Seuils de décision
MAX_CLASSES = 50          # au-delà, une cible entière est vue comme une régression
SKEW_THRESHOLD = 1.0      # |asymétrie| au-delà de laquelle on applique log1p
IMBALANCE_THRESHOLD = 0.2 # ratio classe minoritaire / majoritaire
HIGH_CARDINALITY = 20     # nombre max de modalités gardées par l'encodage one-hot


# ---------------------------------------------------------------------------
# Détection de la tâche
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    """Nature du problème et conversion de y vers / depuis le format scikit-learn."""

    task: str
    n_outputs: int
    classes: Optional[np.ndarray] = None

    @property
    def is_classification(self) -> bool:
        return self.task in CLASSIFICATION_TASKS

    def encode_y(self, y: np.ndarray) -> np.ndarray:
        """Format de la solution -> format attendu par scikit-learn."""
        if self.task == MULTICLASS_ONEHOT:
            return np.argmax(y, axis=1)
        if self.task == MULTILABEL:
            return y.astype(int)
        if self.task in (BINARY, MULTICLASS):
            return y.ravel()
        return y.ravel() if self.n_outputs == 1 else y

    def decode_y(self, y_pred: np.ndarray) -> np.ndarray:
        """Prédictions scikit-learn -> même forme que le fichier .solution."""
        y_pred = np.asarray(y_pred)
        if self.task == MULTICLASS_ONEHOT:
            onehot = np.zeros((len(y_pred), self.n_outputs), dtype=int)
            onehot[np.arange(len(y_pred)), y_pred.astype(int)] = 1
            return onehot
        return y_pred


def detect_task(y: np.ndarray) -> TaskInfo:
    """Déduit la tâche uniquement à partir de la forme et des valeurs de y."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    if y.ndim == 2:
        n_outputs = y.shape[1]
        if np.all((y == 0) | (y == 1)):
            if np.all(y.sum(axis=1) == 1):
                return TaskInfo(MULTICLASS_ONEHOT, n_outputs, np.arange(n_outputs))
            return TaskInfo(MULTILABEL, n_outputs, np.array([0, 1]))
        return TaskInfo(MULTIOUTPUT_REGRESSION, n_outputs)

    values = np.unique(y)
    if len(values) == 2:
        return TaskInfo(BINARY, 1, values)
    is_integer = np.all(np.mod(values, 1) == 0)
    if is_integer and len(values) <= MAX_CLASSES:
        return TaskInfo(MULTICLASS, 1, values)
    return TaskInfo(REGRESSION, 1)


# ---------------------------------------------------------------------------
# Profil des données
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Résumé de tout ce que l'inspection a appris sur un jeu de données."""

    n_samples: int
    n_features: int
    is_sparse: bool
    density: float
    numerical: List[int] = field(default_factory=list)
    categorical: List[int] = field(default_factory=list)
    binary: List[int] = field(default_factory=list)
    constant: List[int] = field(default_factory=list)
    skewed: List[int] = field(default_factory=list)
    missing: Dict[int, float] = field(default_factory=dict)
    cardinality: Dict[int, int] = field(default_factory=dict)
    nonnegative_integer: bool = False
    task: Optional[TaskInfo] = None
    class_counts: Optional[Dict] = None
    imbalance_ratio: Optional[float] = None
    target_skewed: bool = False

    @property
    def has_missing(self) -> bool:
        return bool(self.missing)

    @property
    def is_imbalanced(self) -> bool:
        return self.imbalance_ratio is not None and self.imbalance_ratio < IMBALANCE_THRESHOLD

    def summary(self) -> str:
        lines = [
            f"  Exemples x variables : {self.n_samples} x {self.n_features}",
            f"  Format               : {'creux' if self.is_sparse else 'dense'}"
            f" (densité {self.density:.1%})",
            f"  Types de colonnes    : {len(self.numerical)} num., "
            f"{len(self.categorical)} catég., {len(self.binary)} binaires",
            f"  Colonnes constantes  : {len(self.constant)}",
            f"  Colonnes asymétriques: {len(self.skewed)}",
            f"  Valeurs manquantes   : {len(self.missing)} colonnes",
        ]
        if self.task is not None:
            lines.append(f"  Tâche                : {self.task.task} ({self.task.n_outputs} sortie(s))")
        if self.imbalance_ratio is not None:
            flag = " -> déséquilibré" if self.is_imbalanced else ""
            lines.append(f"  Ratio min/max classes: {self.imbalance_ratio:.2f}{flag}")
        if self.target_skewed:
            lines.append("  Cible asymétrique    : oui (transformation log conseillée)")
        return "\n".join(lines)


def _skewness(col: np.ndarray) -> float:
    std = col.std()
    if std == 0:
        return 0.0
    return float(np.mean(((col - col.mean()) / std) ** 3))


def _profile_dense(X: np.ndarray, types: List[str], profile: DataProfile) -> None:
    nan_mask = np.isnan(X)
    for j, t in enumerate(types):
        col = X[~nan_mask[:, j], j]
        miss = nan_mask[:, j].mean()
        if miss > 0:
            profile.missing[j] = float(miss)
        n_unique = len(np.unique(col))
        if n_unique <= 1:
            profile.constant.append(j)
            continue
        if t == "Categorical":
            profile.categorical.append(j)
            profile.cardinality[j] = n_unique
        elif t == "Binary" or n_unique == 2:
            profile.binary.append(j)
        else:
            profile.numerical.append(j)
            if col.min() >= 0 and abs(_skewness(col)) > SKEW_THRESHOLD:
                profile.skewed.append(j)
    profile.density = float(np.count_nonzero(X[~nan_mask]) / max(X.size, 1))


def _profile_sparse(X: sp.csr_matrix, profile: DataProfile) -> None:
    Xc = X.tocsc()
    nnz = np.diff(Xc.indptr)
    profile.constant = [int(j) for j in np.where(nnz == 0)[0]]
    profile.numerical = [int(j) for j in np.where(nnz > 0)[0]]
    profile.density = X.nnz / max(X.shape[0] * X.shape[1], 1)
    data = X.data
    profile.nonnegative_integer = bool(
        data.size and np.all(data >= 0) and np.all(np.mod(data, 1) == 0)
    )


def _profile_target(y: np.ndarray, profile: DataProfile) -> None:
    task = detect_task(y)
    profile.task = task
    y_enc = task.encode_y(np.asarray(y, dtype=np.float64))

    if task.task in (BINARY, MULTICLASS, MULTICLASS_ONEHOT):
        values, counts = np.unique(y_enc, return_counts=True)
        profile.class_counts = {v.item(): int(c) for v, c in zip(values, counts)}
        profile.imbalance_ratio = float(counts.min() / counts.max())
    elif task.task == MULTILABEL:
        positives = y_enc.sum(axis=0)
        profile.class_counts = {i: int(c) for i, c in enumerate(positives)}
        rates = positives / len(y_enc)
        # Un label est déséquilibré si sa classe positive (ou négative) est rare
        profile.imbalance_ratio = float(np.min(np.minimum(rates, 1 - rates) / np.maximum(rates, 1 - rates)))
    elif task.task == REGRESSION:
        profile.target_skewed = bool(y_enc.min() >= 0 and _skewness(y_enc) > SKEW_THRESHOLD)


def inspect_data(X, types: List[str], y: Optional[np.ndarray] = None) -> DataProfile:
    """Analyse X (et y si fourni) et renvoie un DataProfile."""
    is_sparse = sp.issparse(X)
    profile = DataProfile(
        n_samples=X.shape[0], n_features=X.shape[1], is_sparse=is_sparse, density=1.0,
    )
    if is_sparse:
        _profile_sparse(X, profile)
    else:
        _profile_dense(X, types, profile)
    if y is not None:
        _profile_target(y, profile)
    return profile
