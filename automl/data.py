"""Chargement des données, détection automatique de tâche et découpage stratifié adaptatif.

Ce module reprend et industrialise la logique prototypée dans le notebook
`02_baseline_automl.ipynb` afin de gérer de manière robuste :
- Les formats denses (avec gestion des valeurs manquantes `NaN`/`?` et chaînes)
- Les formats creux `indice:valeur` (0-based ou 1-based, sans conversion dense)
- La détection automatique des 6 types de tâches possibles
- Le découpage stratifié Train / Val / Test adaptatif selon la taille N du dataset
- Le reformatage strict des prédictions pour respecter le contrat de `.solution`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

# Jetons considérés comme valeurs manquantes dans les fichiers .data / .solution
TOKENS_NA = ["NaN", "nan", "NAN", "?", "NA", "na", "null", "NULL", "None", "none", ""]


@dataclass
class CheminsDataset:
    """Chemins résolus vers les fichiers `<nom>.data`, `<nom>.type` et `<nom>.solution`."""

    nom: str
    data_path: Path
    type_path: Optional[Path] = None
    solution_path: Optional[Path] = None


@dataclass
class MetaFeatures:
    """Métadonnées sur les variables d'entrée X (indispensable pour aligner train et test)."""

    n_features: int
    column_types: List[str]
    is_sparse: bool
    sparse_index_offset: int = 1
    string_encoders: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def indices_numeriques(self) -> List[int]:
        return [i for i, t in enumerate(self.column_types) if t == "Numerical"]

    @property
    def indices_categorielles(self) -> List[int]:
        return [i for i, t in enumerate(self.column_types) if t == "Categorical"]

    @property
    def indices_binaires(self) -> List[int]:
        return [i for i, t in enumerate(self.column_types) if t == "Binary"]


@dataclass
class MetaCible:
    """Métadonnées sur la cible y et la tâche détectée automatiquement."""

    task_type: str  # 'binary', 'multiclass', 'multiclass_onehot', 'multilabel', 'regression', 'multioutput_regression'
    n_outputs: int
    is_1d: bool
    original_dtype: np.dtype
    classes_: Optional[np.ndarray] = None
    n_classes: int = 0

    @property
    def est_classification(self) -> bool:
        return self.task_type in (
            "binary",
            "multiclass",
            "multiclass_onehot",
            "multilabel",
        )


@dataclass
class JeuDecoupe:
    """Sous-ensembles Train, Validation et Test interne."""

    X_train: Union[np.ndarray, sparse.csr_matrix]
    y_train: np.ndarray
    X_val: Union[np.ndarray, sparse.csr_matrix]
    y_val: np.ndarray
    X_test: Union[np.ndarray, sparse.csr_matrix]
    y_test: np.ndarray


def resoudre_fichiers_dataset(data_dest: Union[str, Path]) -> CheminsDataset:
    """Localise `<nom>.data`, `<nom>.type` et `<nom>.solution` depuis un dossier ou un fichier."""
    p = Path(data_dest).expanduser().resolve()

    if p.is_file():
        stem = p.stem
        parent = p.parent
        data_file = parent / f"{stem}.data"
        if not data_file.exists() and p.suffix == ".data":
            data_file = p
        if not data_file.exists():
            raise FileNotFoundError(f"Fichier .data introuvable pour : {data_dest}")
        type_file = parent / f"{stem}.type"
        sol_file = parent / f"{stem}.solution"
        return CheminsDataset(
            nom=stem,
            data_path=data_file,
            type_path=type_file if type_file.exists() else None,
            solution_path=sol_file if sol_file.exists() else None,
        )

    if not p.is_dir():
        raise FileNotFoundError(f"Dossier de dataset introuvable : {data_dest}")

    data_files = sorted(p.glob("*.data"))
    if not data_files:
        raise FileNotFoundError(f"Aucun fichier .data trouvé dans : {p}")

    # Si un fichier porte le même nom que le dossier, on le privilégie
    df = data_files[0]
    for cand in data_files:
        if cand.stem == p.name:
            df = cand
            break

    stem = df.stem
    tf = p / f"{stem}.type"
    sf = p / f"{stem}.solution"
    if not tf.exists():
        cands = sorted(p.glob("*.type"))
        if cands:
            tf = cands[0]
    if not sf.exists():
        cands = sorted(p.glob("*.solution"))
        if cands:
            sf = cands[0]

    return CheminsDataset(
        nom=stem,
        data_path=df,
        type_path=tf if tf.exists() else None,
        solution_path=sf if sf.exists() else None,
    )


def normaliser_type_colonne(type_brut: str) -> str:
    """Normalise un libellé de `.type` vers Numerical, Categorical ou Binary."""
    t = type_brut.strip().lower()
    if t in ("categorical", "cat", "nominal", "ordinal", "string", "object"):
        return "Categorical"
    if t in ("binary", "bin", "bool", "boolean"):
        return "Binary"
    return "Numerical"


def charger_types_colonnes(
    type_path: Optional[Path], n_cols: Optional[int] = None
) -> List[str]:
    """Lit le fichier `<nom>.type` et ajuste sa longueur à `n_cols` si précisé."""
    if type_path is not None and type_path.exists():
        with open(type_path, "r", encoding="utf-8", errors="ignore") as f:
            types = [normaliser_type_colonne(line) for line in f if line.strip()]
        if n_cols is not None and len(types) != n_cols:
            if len(types) < n_cols:
                types.extend(["Numerical"] * (n_cols - len(types)))
            else:
                types = types[:n_cols]
        return types
    if n_cols is not None:
        return ["Numerical"] * n_cols
    return []


def _est_fichier_creux(data_path: Path) -> bool:
    """Détecte si `<nom>.data` est au format creux `indice:valeur` (LibSVM)."""
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(25):
            ligne = f.readline()
            if not ligne:
                break
            propre = ligne.strip()
            if not propre:
                continue
            premier_jeton = propre.split()[0]
            if ":" in premier_jeton:
                parties = premier_jeton.split(":")
                if len(parties) == 2:
                    try:
                        int(parties[0])
                        float(parties[1])
                        return True
                    except ValueError:
                        pass
            return False
    return False


def _charger_matrice_creuse(
    data_path: Path,
    n_features: Optional[int] = None,
    index_offset: Optional[int] = None,
) -> Tuple[sparse.csr_matrix, int, int]:
    """Charge un fichier `idx:val` directement en scipy.sparse.csr_matrix."""
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    n_rows = 0
    min_idx = 2**31 - 1
    max_idx = 0

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for row_idx, ligne in enumerate(f):
            n_rows = row_idx + 1
            propre = ligne.strip()
            if not propre:
                continue
            for jeton in propre.split():
                if ":" not in jeton:
                    continue
                idx_str, val_str = jeton.split(":", 1)
                idx = int(idx_str)
                val = float(val_str)
                if idx < min_idx:
                    min_idx = idx
                if idx > max_idx:
                    max_idx = idx
                rows.append(row_idx)
                cols.append(idx)
                vals.append(val)

    if len(cols) == 0:
        offset = index_offset if index_offset is not None else 1
        n_feat = n_features if n_features is not None else 1
        return sparse.csr_matrix((n_rows, n_feat), dtype=np.float64), n_feat, offset

    # Détection automatique de l'indexation (0-based ou 1-based)
    if index_offset is None:
        if min_idx == 0:
            offset = 0
        elif n_features is not None and max_idx == n_features and min_idx >= 1:
            offset = 1
        elif min_idx >= 1:
            offset = 1
        else:
            offset = 0
    else:
        offset = index_offset

    cols_arr = np.asarray(cols, dtype=np.int64) - offset
    rows_arr = np.asarray(rows, dtype=np.int64)
    vals_arr = np.asarray(vals, dtype=np.float64)

    n_feat = n_features if n_features is not None else (int(cols_arr.max()) + 1)

    # Protection indispensable : ignore d'éventuels indices hors bornes en test
    valide = (cols_arr >= 0) & (cols_arr < n_feat)
    if not np.all(valide):
        rows_arr = rows_arr[valide]
        cols_arr = cols_arr[valide]
        vals_arr = vals_arr[valide]

    mat = sparse.csr_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(n_rows, n_feat),
        dtype=np.float64,
    )
    return mat, n_feat, offset


def _charger_matrice_dense(
    data_path: Path,
    string_encoders: Optional[Dict[int, Dict[str, float]]] = None,
    is_train: bool = True,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Charge un fichier `.data` dense en tableau NumPy float64."""
    encoders: Dict[int, Dict[str, float]] = (
        {} if string_encoders is None else dict(string_encoders)
    )
    try:
        df = pd.read_csv(
            data_path,
            sep=r"\s+",
            header=None,
            na_values=TOKENS_NA,
            dtype=np.float64,
            engine="c",
        )
        return df.to_numpy(dtype=np.float64, copy=False), encoders
    except (ValueError, pd.errors.ParserError):
        # Fallback si le fichier contient des chaînes de caractères ou des virgules
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            premiere = f.readline()
        sep = "," if ("," in premiere and " " not in premiere.strip()) else r"\s+"
        df = pd.read_csv(
            data_path,
            sep=sep,
            header=None,
            na_values=TOKENS_NA,
            engine="c" if sep == "," else "python",
        )
        arr = np.empty(df.shape, dtype=np.float64)
        for col_idx in range(df.shape[1]):
            serie = df.iloc[:, col_idx]
            num_serie = pd.to_numeric(serie, errors="coerce")
            non_null = serie.notna()
            echoue = non_null & num_serie.isna()
            if echoue.any():
                str_vals = serie.astype(str)
                if is_train and col_idx not in encoders:
                    uniques = sorted(set(str_vals[non_null]))
                    encoders[col_idx] = {s: float(i) for i, s in enumerate(uniques)}
                if col_idx not in encoders:
                    arr[:, col_idx] = num_serie.to_numpy(dtype=np.float64)
                    continue
                mapping = encoders[col_idx]
                mapped = serie.map(
                    lambda x: np.nan if pd.isna(x) else mapping.get(str(x), np.nan)
                )
                arr[:, col_idx] = mapped.to_numpy(dtype=np.float64)
            else:
                arr[:, col_idx] = num_serie.to_numpy(dtype=np.float64)
        return arr, encoders


def _aligner_nb_colonnes(X: np.ndarray, n_attendu: int) -> np.ndarray:
    """Aligne une matrice dense de test sur le nombre exact de colonnes vues à l'entraînement."""
    n_cols = X.shape[1]
    if n_cols == n_attendu:
        return X
    if n_cols > n_attendu:
        return X[:, :n_attendu]
    pad = np.full((X.shape[0], n_attendu - n_cols), np.nan, dtype=np.float64)
    return np.hstack([X, pad])


def charger_features(
    chemins: CheminsDataset,
    feature_meta: Optional[MetaFeatures] = None,
) -> Tuple[Union[np.ndarray, sparse.csr_matrix], MetaFeatures]:
    """Charge la matrice X et construit/réutilise les métadonnées associées."""
    is_sparse = (
        feature_meta.is_sparse
        if feature_meta is not None
        else _est_fichier_creux(chemins.data_path)
    )

    if is_sparse:
        types_prelim = (
            feature_meta.column_types
            if feature_meta is not None
            else charger_types_colonnes(chemins.type_path)
        )
        n_attendu = len(types_prelim) if types_prelim else None
        offset = feature_meta.sparse_index_offset if feature_meta is not None else None
        X, n_features, offset_infere = _charger_matrice_creuse(
            chemins.data_path, n_features=n_attendu, index_offset=offset
        )
        col_types = (
            feature_meta.column_types
            if feature_meta is not None
            else charger_types_colonnes(chemins.type_path, n_cols=n_features)
        )
        meta = MetaFeatures(
            n_features=n_features,
            column_types=col_types,
            is_sparse=True,
            sparse_index_offset=offset_infere,
        )
        return X, meta

    X_dense, encoders = _charger_matrice_dense(
        chemins.data_path,
        string_encoders=feature_meta.string_encoders if feature_meta else None,
        is_train=(feature_meta is None),
    )
    if feature_meta is not None:
        X_dense = _aligner_nb_colonnes(X_dense, feature_meta.n_features)
    n_features = X_dense.shape[1]
    col_types = (
        feature_meta.column_types
        if feature_meta is not None
        else charger_types_colonnes(chemins.type_path, n_cols=n_features)
    )
    meta = MetaFeatures(
        n_features=n_features,
        column_types=col_types,
        is_sparse=False,
        string_encoders=encoders,
    )
    return X_dense, meta


def charger_cible_et_detecter_tache(
    solution_path: Path,
) -> Tuple[np.ndarray, MetaCible]:
    """Lit `.solution`, détecte automatiquement la tâche et retourne la cible encodée."""
    df = pd.read_csv(
        solution_path,
        sep=r"\s+",
        header=None,
        na_values=TOKENS_NA,
        engine="c",
    )
    raw_arr = df.to_numpy()
    _, n_cols = raw_arr.shape

    try:
        float_arr = raw_arr.astype(np.float64)
    except (ValueError, TypeError):
        # Labels textuels (chaînes)
        labels = np.asarray(raw_arr[:, 0]).astype(str)
        classes, y_enc = np.unique(labels, return_inverse=True)
        meta = MetaCible(
            task_type="binary" if len(classes) == 2 else "multiclass",
            n_outputs=1,
            is_1d=True,
            original_dtype=classes.dtype,
            classes_=classes,
            n_classes=len(classes),
        )
        return y_enc.astype(np.int64), meta

    est_entier = np.all(np.isfinite(float_arr)) and np.all(
        np.equal(np.mod(float_arr, 1), 0)
    )
    orig_dtype = np.int64 if est_entier else np.float64

    # 1. Cible multi-colonnes (2D)
    if n_cols > 1:
        uniques = np.unique(float_arr)
        est_binaire = np.all(np.isin(uniques, [0.0, 1.0]))
        if est_binaire:
            sommes_lignes = float_arr.sum(axis=1)
            if np.all(sommes_lignes == 1.0):
                # Classification multiclasse encodée en One-Hot (ex: data_C, data_F)
                y_interne = np.argmax(float_arr, axis=1).astype(np.int64)
                meta = MetaCible(
                    task_type="multiclass_onehot",
                    n_outputs=n_cols,
                    is_1d=False,
                    original_dtype=np.int64,
                    classes_=np.arange(n_cols, dtype=np.int64),
                    n_classes=n_cols,
                )
                return y_interne, meta
            else:
                # Classification multi-label (ex: data_A)
                y_interne = float_arr.astype(np.int64)
                meta = MetaCible(
                    task_type="multilabel",
                    n_outputs=n_cols,
                    is_1d=False,
                    original_dtype=np.int64,
                    n_classes=n_cols,
                )
                return y_interne, meta
        else:
            # Régression multi-sorties
            meta = MetaCible(
                task_type="multioutput_regression",
                n_outputs=n_cols,
                is_1d=False,
                original_dtype=np.float64,
            )
            return float_arr, meta

    # 2. Cible mono-colonne (1D)
    y_1d = float_arr.ravel()
    uniques = np.unique(y_1d[~np.isnan(y_1d)])
    n_unique = len(uniques)

    if n_unique == 2:
        classes = uniques.astype(orig_dtype)
        y_enc = np.searchsorted(uniques, y_1d).astype(np.int64)
        meta = MetaCible(
            task_type="binary",
            n_outputs=1,
            is_1d=True,
            original_dtype=orig_dtype,
            classes_=classes,
            n_classes=2,
        )
        return y_enc, meta

    if (
        est_entier
        and 2 < n_unique <= 50
        and (uniques.max() - uniques.min() <= n_unique + 5)
    ):
        classes = uniques.astype(np.int64)
        y_enc = np.searchsorted(uniques, y_1d).astype(np.int64)
        meta = MetaCible(
            task_type="multiclass",
            n_outputs=1,
            is_1d=True,
            original_dtype=np.int64,
            classes_=classes,
            n_classes=n_unique,
        )
        return y_enc, meta

    # Régression continue (ex: data_B)
    meta = MetaCible(
        task_type="regression",
        n_outputs=1,
        is_1d=True,
        original_dtype=np.float64,
    )
    return y_1d.astype(np.float64), meta


def formater_predictions(preds: np.ndarray, target_meta: MetaCible) -> np.ndarray:
    """Formate les prédictions pour correspondre exactement au contrat `.solution`."""
    if target_meta.task_type == "multiclass_onehot":
        n_samples = preds.shape[0]
        n_classes = target_meta.n_outputs
        if preds.ndim == 2 and preds.shape[1] == n_classes:
            indices = np.argmax(preds, axis=1)
        else:
            indices = np.asarray(preds, dtype=np.int64).ravel()
        indices = np.clip(indices, 0, n_classes - 1)
        onehot = np.zeros((n_samples, n_classes), dtype=np.int64)
        onehot[np.arange(n_samples), indices] = 1
        return onehot

    if target_meta.task_type == "multilabel":
        out = np.asarray(preds)
        if out.ndim == 1:
            out = out.reshape(-1, target_meta.n_outputs)
        return (
            (out >= 0.5).astype(np.int64)
            if out.dtype.kind == "f"
            else out.astype(np.int64)
        )

    if target_meta.task_type in ("binary", "multiclass"):
        idx = np.asarray(preds, dtype=np.int64).ravel()
        if target_meta.classes_ is not None:
            idx = np.clip(idx, 0, len(target_meta.classes_) - 1)
            return target_meta.classes_[idx].astype(target_meta.original_dtype)
        return idx.astype(target_meta.original_dtype)

    if target_meta.task_type == "multioutput_regression":
        out = np.asarray(preds, dtype=np.float64)
        if out.ndim == 1:
            out = out.reshape(-1, target_meta.n_outputs)
        return out

    return np.asarray(preds, dtype=np.float64).ravel()


def construire_etiquettes_stratification(
    y: np.ndarray, target_meta: MetaCible
) -> Optional[np.ndarray]:
    """Construit un vecteur 1D pour stratifier le découpage même en multi-label ou régression."""
    if target_meta.task_type in ("binary", "multiclass", "multiclass_onehot"):
        y_1d = y.ravel().astype(np.int64)
        _, counts = np.unique(y_1d, return_counts=True)
        return y_1d if counts.min() >= 2 else None

    if target_meta.task_type == "multilabel":
        if y.ndim == 2 and y.shape[1] <= 10:
            puissances = (1 << np.arange(y.shape[1], dtype=np.int64))[::-1]
            combo_ids = (y.astype(np.int64) @ puissances).ravel()
            _, counts = np.unique(combo_ids, return_counts=True)
            if counts.min() >= 3:
                return combo_ids
        return None

    if target_meta.task_type == "regression":
        try:
            bins = pd.qcut(y.ravel(), q=10, labels=False, duplicates="drop")
            bins_arr = np.asarray(bins, dtype=np.int64)
            _, counts = np.unique(bins_arr, return_counts=True)
            if counts.min() >= 3:
                return bins_arr
        except Exception:
            pass
    return None


def tailles_split_adaptatif(n_samples: int) -> Tuple[float, float]:
    """Calcule (val_size, test_size) dynamiquement selon la taille N du dataset.

    - Petit dataset (< 2000) : on garde 20% val et 10% test pour stabiliser l'estimation.
    - Grand dataset (>= 50000) : quelques milliers de lignes suffisent pour valider,
      donc on réduit le pourcentage de validation et on laisse jusqu'à 87% pour l'entraînement.
    """
    if n_samples < 2000:
        val_size, test_size = 0.20, 0.10
    elif n_samples < 10000:
        val_size, test_size = 0.15, 0.10
    elif n_samples < 50000:
        val_size, test_size = 0.12, 0.08
    else:
        val_size, test_size = 0.08, 0.05

    if n_samples > 0:
        val_size = min(val_size, max(0.05, 8000.0 / n_samples))
        test_size = min(test_size, max(0.03, 5000.0 / n_samples))
    return float(val_size), float(test_size)


def decouper_train_val_test(
    X: Union[np.ndarray, sparse.csr_matrix],
    y: np.ndarray,
    target_meta: MetaCible,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    seed: int = 42,
) -> JeuDecoupe:
    """Découpe en Train / Val / Test interne avec stratification adaptée et ratio dynamique."""
    if val_size is None or test_size is None:
        v_adapt, t_adapt = tailles_split_adaptatif(X.shape[0])
        val_size = v_adapt if val_size is None else val_size
        test_size = t_adapt if test_size is None else test_size

    strat = construire_etiquettes_stratification(y, target_meta)
    total_holdout = val_size + test_size

    try:
        if strat is not None:
            X_train, X_temp, y_train, y_temp, _, strat_temp = train_test_split(
                X,
                y,
                strat,
                test_size=total_holdout,
                random_state=seed,
                stratify=strat,
            )
            rel_test = test_size / total_holdout
            _, counts = np.unique(strat_temp, return_counts=True)
            strat_second = strat_temp if counts.min() >= 2 else None
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=rel_test,
                random_state=seed + 1,
                stratify=strat_second,
            )
        else:
            raise ValueError("Repli vers découpage non stratifié")
    except ValueError:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=total_holdout, random_state=seed
        )
        rel_test = test_size / total_holdout
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_test, random_state=seed + 1
        )

    return JeuDecoupe(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
