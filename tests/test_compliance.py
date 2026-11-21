"""Tests de conformité aux règles du sujet (risque de disqualification).

Rôle : vérifier automatiquement les règles du challenge qui ne concernent pas un
module précis :
- aucun import interdit dans automl/ (torch, xgboost, subprocess, socket…) ;
- jamais `n_jobs=-1` ;
- aucun fichier de plus de 5 Mo ni modèle sauvegardé (.pkl, .joblib, .h5) dans
  le dépôt (les dossiers ignorés par git sont exclus).

Quand le lancer : avant chaque fusion de branche et avant la soumission finale
(`python -m pytest tests`).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_IMPORTS = {"torch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost",
                     "socket", "requests", "subprocess", "urllib"}
# Dossiers ignorés par git (.gitignore) : ils ne sont pas dans le dépôt
IGNORED_DIRS = {"dataset", "test_datasets", "automl_env", ".venv", ".git", "build", ".pytest_cache"}


def test_no_forbidden_imports_and_no_all_cores():
    for f in (ROOT / "automl").rglob("*.py"):
        source = f.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                names = []
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN_IMPORTS, f"{n} interdit dans {f}"
        assert "n_jobs=-1" not in source


def test_no_large_files_or_saved_models():
    for f in ROOT.rglob("*"):
        if f.is_file() and not IGNORED_DIRS & set(f.relative_to(ROOT).parts):
            assert f.stat().st_size <= 5 * 1024 * 1024, f"{f} dépasse 5 Mo"
            assert f.suffix not in {".pkl", ".joblib", ".h5"}
