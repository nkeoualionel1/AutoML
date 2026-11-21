"""Tests de automl/inspection.py : détection de la tâche et conversion de la cible.

Rôle : vérifier que chaque type de cible (binaire, multiclasse quelconque, one-hot,
multi-label, régression, étiquettes texte) est bien reconnu, que la conversion
format .solution -> scikit-learn -> format .solution est sans perte, et que la
stratification du découpage est possible pour chaque tâche.

Quand le lancer : après toute modification de automl/inspection.py, et avant
chaque fusion de branche (`python -m pytest tests`).
"""

import numpy as np
import pytest

from automl.inspection import detect_task


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
def test_task_detection_and_round_trip(y, task):
    info = detect_task(y)
    assert info.task == task
    assert np.array_equal(info.decode_y(info.encode_y(y)), y)


def test_onehot_decoding_sums_to_one():
    info = detect_task(np.eye(4)[[0, 1, 2, 3]])
    out = info.decode_y(np.array([3, 0, 9]))  # 9 hors bornes : ramené dans [0, 3]
    assert out.shape == (3, 4) and np.all(out.sum(axis=1) == 1)


def test_stratification_multilabel_and_regression():
    rng = np.random.RandomState(0)
    y_ml = rng.randint(0, 2, (300, 3)).astype(float)
    info = detect_task(y_ml)
    assert info.stratify_labels(info.encode_y(y_ml)) is not None
    y_reg = rng.randn(300)
    info = detect_task(y_reg)
    assert len(np.unique(info.stratify_labels(info.encode_y(y_reg)))) == 10
