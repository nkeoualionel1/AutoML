"""Outils partagés par tous les tests (chargé automatiquement par pytest).

Rôle : fournir des fixtures pour écrire de petits jeux de données temporaires
au format du challenge (<nom>.data, <nom>.type, <nom>.solution) dans un dossier
jetable (`tmp_path`), sans dépendre de dataset/.

Quand l'utiliser : dans n'importe quel test, en ajoutant `write_dataset` ou
`dense_lines` aux paramètres de la fonction de test.
"""

from pathlib import Path

import pytest


def _write_dataset(folder: Path, name: str, X_lines, types, y_lines=None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.data").write_text("\n".join(X_lines) + "\n")
    (folder / f"{name}.type").write_text("\n".join(types) + "\n")
    if y_lines is not None:
        (folder / f"{name}.solution").write_text("\n".join(y_lines) + "\n")
    return str(folder)


def _dense_lines(X):
    return [" ".join(f"{v:.4f}" for v in row) for row in X]


@pytest.fixture
def write_dataset():
    """write_dataset(dossier, nom, lignes_X, types, lignes_y=None) -> chemin du dossier."""
    return _write_dataset


@pytest.fixture
def dense_lines():
    """dense_lines(X) -> lignes texte d'un fichier .data dense."""
    return _dense_lines
