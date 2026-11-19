"""Chargement des jeux de données et découpage train/valid/test.

Le chargement doit survivre aux pièges d'un jeu de test secret : jetons `?`,
texte dans une colonne, indices creux 0-based ou hors bornes, nombre de colonnes
différent du train. Les informations apprises au `fit` (largeur, décalage des
indices creux, encodage du texte) sont gardées dans `FeatureMeta` et réutilisées
au `predict`, pour que le test soit lu exactement comme le train.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE

Matrix = Union[np.ndarray, sp.csr_matrix]

# Jetons lus comme valeurs manquantes (pandas ne reconnaît pas `?` par défaut)
NA_TOKENS = ["NaN", "nan", "NAN", "?", "NA", "na", "null", "NULL", "None", "none", ""]


@dataclass
class FeatureMeta:
    """Ce qu'il faut retenir du train pour relire un test de la même façon."""

    n_features: int
    is_sparse: bool
    sparse_offset: int = 1  # 1 : indices LibSVM (à partir de 1) ; 0 : à partir de 0
    string_encoders: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    X: Matrix
    types: List[str]
    y: Optional[np.ndarray] = None
    meta: Optional[FeatureMeta] = None


def _find_file(folder: str, ext: str, required: bool = True) -> Optional[str]:
    # On préfère `<dossier>/<dossier>.<ext>`, sinon le premier fichier trouvé
    preferred = os.path.join(folder, f"{os.path.basename(os.path.normpath(folder))}.{ext}")
    if os.path.exists(preferred):
        return preferred
    matches = sorted(glob.glob(os.path.join(folder, f"*.{ext}")))
    if not matches:
        if required:
            raise FileNotFoundError(f"Aucun fichier .{ext} dans {folder}")
        return None
    return matches[0]


def _is_sparse_file(path: str) -> bool:
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                return ":" in line.split()[0]
    return False


def _load_sparse(
    path: str, n_features: Optional[int], offset: Optional[int]
) -> Tuple[sp.csr_matrix, int, int]:
    """Format creux `indice:valeur`. Renvoie (matrice, n_features, décalage)."""
    rows, cols, vals = [], [], []
    n_rows = 0
    with open(path, "r") as f:
        for r, line in enumerate(f):
            n_rows = r + 1  # une ligne vide reste une ligne (exemple sans valeur)
            for item in line.split():
                if ":" not in item:
                    continue
                c, v = item.split(":", 1)
                rows.append(r)
                cols.append(int(c))
                vals.append(float(v))
    cols = np.asarray(cols, dtype=np.int64)
    if offset is None:
        # Au fit : un indice 0 prouve que la numérotation commence à 0
        offset = 0 if cols.size and cols.min() == 0 else 1
    cols = cols - offset
    if n_features is None:
        n_features = int(cols.max()) + 1 if cols.size else 1
    # Un test peut contenir un indice jamais vu (hors bornes) : on l'ignore
    keep = (cols >= 0) & (cols < n_features)
    X = sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64)[keep], (np.asarray(rows)[keep], cols[keep])),
        shape=(n_rows, n_features),
    )
    X.data[~np.isfinite(X.data)] = 0.0
    return X, n_features, offset


def _load_dense(path: str, encoders: Dict[int, Dict[str, float]], fitting: bool) -> np.ndarray:
    """Format dense. Le texte est codé en nombres ; le même codage est réutilisé au test."""
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=NA_TOKENS, keep_default_na=True)
    X = np.empty(df.shape, dtype=np.float64)
    for j in range(df.shape[1]):
        col = df.iloc[:, j]
        numeric = pd.to_numeric(col, errors="coerce")
        has_text = bool((col.notna() & numeric.isna()).any())
        if fitting and has_text:
            encoders[j] = {s: float(i) for i, s in enumerate(sorted(set(col.dropna().astype(str))))}
        if j in encoders:
            mapping = encoders[j]  # modalité inconnue au test -> NaN
            X[:, j] = col.map(lambda v: np.nan if pd.isna(v) else mapping.get(str(v), np.nan)).to_numpy(dtype=np.float64)
        else:
            # Colonne numérique au train : un texte parasite au test devient NaN
            X[:, j] = numeric.to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan
    return X


def _align_columns(X: np.ndarray, n_features: int) -> np.ndarray:
    """Complète avec des NaN (ou tronque) pour retrouver la largeur du train."""
    if X.shape[1] == n_features:
        return X
    if X.shape[1] > n_features:
        return X[:, :n_features]
    pad = np.full((X.shape[0], n_features - X.shape[1]), np.nan)
    return np.hstack([X, pad])


def load_types(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_solution(path: str) -> np.ndarray:
    """Lit `.solution` : nombres (float64) ou étiquettes texte."""
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=NA_TOKENS)
    try:
        y = df.to_numpy(dtype=np.float64)
    except ValueError:
        y = df.to_numpy().astype(str)
    return y.ravel() if y.shape[1] == 1 else y


def load_dataset(folder: str, meta: Optional[FeatureMeta] = None) -> Dataset:
    """Charge `<nom>.data`, `<nom>.type` et, s'il existe, `<nom>.solution`.

    `meta` (issu du train) est passé au moment du `predict` pour relire le test
    avec la même largeur, le même décalage creux et le même codage du texte.
    """
    data_path = _find_file(folder, "data")
    name = os.path.splitext(os.path.basename(data_path))[0]
    types = load_types(_find_file(folder, "type", required=False))
    fitting = meta is None

    sparse = meta.is_sparse if meta is not None else _is_sparse_file(data_path)
    if sparse:
        n_expected = meta.n_features if meta is not None else (len(types) or None)
        X, n_features, offset = _load_sparse(
            data_path, n_expected, meta.sparse_offset if meta is not None else None
        )
        new_meta = FeatureMeta(n_features, True, offset)
    else:
        encoders = {} if fitting else meta.string_encoders
        X = _load_dense(data_path, encoders, fitting)
        if not fitting:
            X = _align_columns(X, meta.n_features)
        new_meta = FeatureMeta(X.shape[1], False, string_encoders=encoders)

    if len(types) != X.shape[1]:
        types = (types + ["Numerical"] * X.shape[1])[: X.shape[1]]

    sol_path = _find_file(folder, "solution", required=False)
    y = load_solution(sol_path) if sol_path is not None else None
    return Dataset(name=name, X=X, types=types, y=y, meta=meta or new_meta)


def adaptive_split_sizes(n_samples: int) -> Tuple[float, float]:
    """Parts (validation, test) selon la taille N du jeu.

    Petit jeu : on garde une validation assez grande pour que son score soit
    stable. Grand jeu : quelques milliers de lignes suffisent, le reste sert à
    l'entraînement.
    """
    if n_samples < 2000:
        valid, test = 0.20, 0.10
    elif n_samples < 10000:
        valid, test = 0.15, 0.10
    elif n_samples < 50000:
        valid, test = 0.12, 0.08
    else:
        valid, test = 0.08, 0.05
    if n_samples > 0:
        valid = min(valid, max(0.05, 8000.0 / n_samples))
        test = min(test, max(0.03, 5000.0 / n_samples))
    return valid, test


def split_data(X, y, stratify=None, valid_size=None, test_size=None):
    """Découpe en train / valid / test (stratifié si possible, tailles adaptatives par défaut)."""
    auto_valid, auto_test = adaptive_split_sizes(X.shape[0])
    valid_size = auto_valid if valid_size is None else valid_size
    test_size = auto_test if test_size is None else test_size
    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = _split(X, y, stratify, test_size)
    rel_valid = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid, _, _ = _split(X_tmp, y_tmp, s_tmp, rel_valid)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _split(X, y, stratify, size):
    s = stratify if stratify is not None else np.zeros(len(y))
    try:
        return train_test_split(
            X, y, s, test_size=size, random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        # Classe trop rare pour stratifier : découpage aléatoire simple
        return train_test_split(X, y, s, test_size=size, random_state=RANDOM_STATE)
