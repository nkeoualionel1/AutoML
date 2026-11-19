"""Tests des étapes 1 (chargement / tâche) et 2 (prétraitement) sur des cas pièges."""

import ast
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from automl.config import ResourceConfig
from automl.data import adaptive_split_sizes, load_dataset
from automl.inspection import detect_task, inspect_data
from automl.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parent.parent


def _write(folder: Path, name: str, lines, types, y_lines=None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.data").write_text("\n".join(lines) + "\n")
    (folder / f"{name}.type").write_text("\n".join(types) + "\n")
    if y_lines is not None:
        (folder / f"{name}.solution").write_text("\n".join(y_lines) + "\n")
    return str(folder)


# --------------------------------------------------------------- étape 1
@pytest.mark.parametrize(
    "y, task",
    [
        (np.array([0.0, 1.0, 1.0, 0.0]), "binary"),
        (np.array([3.0, 7.0, 12.0, 3.0]), "multiclass"),
        (np.eye(3)[[0, 1, 2, 1]], "multiclass_onehot"),
        (np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float), "multilabel"),
        (np.linspace(0, 1, 100), "regression"),
        (np.array(["chat", "chien", "chat"]), "binary"),
    ],
)
def test_detection_et_aller_retour(y, task):
    info = detect_task(y)
    assert info.task == task
    assert np.array_equal(info.decode_y(info.encode_y(y)), y)


def test_decode_onehot_somme_a_un():
    info = detect_task(np.eye(4)[[0, 1, 2, 3]])
    out = info.decode_y(np.array([3, 0, 9]))  # 9 hors bornes : ramené dans [0, 3]
    assert out.shape == (3, 4) and np.all(out.sum(axis=1) == 1)


def test_dense_texte_jetons_et_alignement(tmp_path):
    train = _write(
        tmp_path / "tr", "d",
        ["rouge 1.5 ?", "bleu inf 2", "rouge 0.5 NaN", "vert 2 3"],
        ["Categorical", "Numerical", "Numerical"], ["0", "1", "0", "1"],
    )
    ds = load_dataset(train)
    assert ds.X.shape == (4, 3)
    assert np.isnan(ds.X[0, 2]) and np.isnan(ds.X[1, 1])  # `?` et `inf` -> NaN

    # Test : modalité inconnue, texte parasite dans une colonne numérique, colonne en moins
    test = _write(tmp_path / "te", "d", ["violet oops", "bleu 1"], ["Categorical", "Numerical"])
    te = load_dataset(test, meta=ds.meta)
    assert te.X.shape == (2, 3)
    assert np.isnan(te.X[0, 0]) and np.isnan(te.X[0, 1]) and np.isnan(te.X[:, 2]).all()
    assert te.X[1, 0] == ds.X[1, 0]  # "bleu" a le même code qu'au train


def test_creux_offset_hors_bornes_et_ligne_vide(tmp_path):
    train = _write(tmp_path / "tr", "s", ["0:1 2:3", "1:2"], ["Numerical"] * 3, ["0", "1"])
    ds = load_dataset(train)
    assert ds.meta.sparse_offset == 0 and ds.X.shape == (2, 3)
    test = _write(tmp_path / "te", "s", ["0:1 99:4", ""], ["Numerical"] * 3)
    te = load_dataset(test, meta=ds.meta)
    assert te.X.shape == (2, 3) and te.X.nnz == 1


def test_split_adaptatif():
    v_small, t_small = adaptive_split_sizes(1000)
    v_big, t_big = adaptive_split_sizes(100000)
    assert v_small > v_big and 0 < t_big <= v_big
    assert 100000 * v_big > 1000 * v_small


def test_stratification_multilabel_et_regression():
    rng = np.random.RandomState(0)
    y_ml = rng.randint(0, 2, (300, 3)).astype(float)
    info = detect_task(y_ml)
    assert info.stratify_labels(info.encode_y(y_ml)) is not None
    y_reg = rng.randn(300)
    info = detect_task(y_reg)
    assert len(np.unique(info.stratify_labels(info.encode_y(y_reg)))) == 10


# --------------------------------------------------------------- étape 2
def test_pretraitement_dense_valeurs_extremes():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 5)
    X[:, 0] = np.nan  # colonne vide
    X[:, 1] = 3.0  # colonne constante
    X[:, 4] = rng.choice([1.0, 2.0, 3.0], 200)
    types = ["Numerical"] * 4 + ["Categorical"]
    y = (X[:, 2] > 0).astype(float)
    profile = inspect_data(X, types, y)
    prep = build_preprocessor(profile)
    Xt = prep.fit_transform(X, y)
    X_test = rng.randn(3, 5)
    X_test[0, 2] = np.inf  # infini
    X_test[1, 4] = 99.0  # modalité inconnue
    out = prep.transform(X_test)
    assert out.shape[1] == Xt.shape[1] and np.isfinite(out).all()


def test_pretraitement_creux_negatif_au_test():
    X = sp.random(100, 30, density=0.2, random_state=2, format="csr")
    X.data = np.ceil(X.data * 5)  # comptages entiers
    y = np.arange(100) % 3
    profile = inspect_data(X, ["Numerical"] * 30, y.astype(float))
    prep = build_preprocessor(profile)
    prep.fit(X, y)
    X_neg = X[:5].copy()
    X_neg.data = -X_neg.data
    assert np.isfinite(prep.transform(X_neg).data).all()


# --------------------------------------------------------------- conformité
def test_ressources_lues_au_fit(monkeypatch):
    monkeypatch.setenv("AUTOML_N_JOBS", "-1")
    monkeypatch.setenv("AUTOML_TIME_BUDGET", "45")
    cfg = ResourceConfig.from_env()
    assert cfg.n_jobs == 1 and cfg.time_budget == 45.0


def test_conformite_imports_et_taille_fichiers():
    interdits = {"torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost",
                 "socket", "requests", "subprocess", "urllib"}
    for f in (ROOT / "automl").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for n in names:
                assert n.split(".")[0] not in interdits, f"{n} interdit dans {f}"
        assert "n_jobs=-1" not in f.read_text()
    ignores = {"dataset", "dataset_test", "automl_env", ".venv", ".git"}
    for f in ROOT.rglob("*"):
        if f.is_file() and not ignores & set(f.relative_to(ROOT).parts):
            assert f.stat().st_size <= 5 * 1024 * 1024, f"{f} dépasse 5 Mo"
            assert f.suffix not in {".pkl", ".joblib", ".h5"}
