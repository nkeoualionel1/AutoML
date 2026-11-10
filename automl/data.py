"""Chargement des jeux de données et découpage train/valid/test."""

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE

Matrix = Union[np.ndarray, sp.csr_matrix]


@dataclass
class Dataset:
    name: str
    X: Matrix
    types: List[str]
    y: Optional[np.ndarray] = None


def _find_file(folder: str, ext: str, required: bool = True) -> Optional[str]:
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
                return ":" in line
    return False


def _load_sparse(path: str, n_features: Optional[int]) -> sp.csr_matrix:
    """Format creux `indice:valeur` (indices à partir de 1, comme LibSVM)."""
    rows, cols, vals = [], [], []
    n_rows = 0
    with open(path, "r") as f:
        for r, line in enumerate(f):
            n_rows = r + 1
            for item in line.split():
                c, v = item.split(":")
                rows.append(r)
                cols.append(int(c) - 1)
                vals.append(float(v))
    # Le nombre de colonnes vient du fichier .type : un jeu de test peut ne pas
    # contenir la dernière colonne, il faut quand même garder la même largeur
    if n_features is None:
        n_features = max(cols) + 1 if cols else 0
    return sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64), (rows, cols)),
        shape=(n_rows, n_features),
    )


def _load_dense(path: str) -> np.ndarray:
    # read_csv est bien plus rapide que np.genfromtxt et lit les `nan` de data_A
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64).to_numpy()


def load_types(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_dataset(folder: str) -> Dataset:
    """Charge `<nom>.data`, `<nom>.type` et, s'il existe, `<nom>.solution`."""
    data_path = _find_file(folder, "data")
    name = os.path.splitext(os.path.basename(data_path))[0]
    types = load_types(_find_file(folder, "type", required=False))

    if _is_sparse_file(data_path):
        X = _load_sparse(data_path, len(types) or None)
    else:
        X = _load_dense(data_path)

    if not types:
        types = ["Numerical"] * X.shape[1]

    y = None
    sol_path = _find_file(folder, "solution", required=False)
    if sol_path is not None:
        y = np.loadtxt(sol_path, dtype=np.float64)

    return Dataset(name=name, X=X, types=types, y=y)


def split_data(X, y, stratify=None, valid_size=0.15, test_size=0.15):
    """Découpe en train / valid / test (stratifié si possible)."""
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
