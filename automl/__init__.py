import os as _os

# Doit passer avant l'import de numpy / scikit-learn : leurs pools de threads
# natifs (OpenMP, BLAS) lisent ces variables à l'initialisation.
if _os.environ.get("AUTOML_N_JOBS", "").strip().isdigit():
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_var, str(max(1, int(_os.environ["AUTOML_N_JOBS"]))))

import numpy as np  # noqa: E402

from .config import ResourceConfig, TimeBudget, limit_native_threads, set_seed  # noqa: E402
from .data import load_dataset, split_data  # noqa: E402
from .inspection import DataProfile, TaskInfo, detect_task, inspect_data  # noqa: E402
from .preprocessing import build_preprocessor  # noqa: E402


class AutoML:
    """Pipeline AutoML : inspection -> prétraitement -> (modèles à venir)."""

    def __init__(self):
        self.model = None
        self.task = None
        self.profile = None
        self.preprocessor = None
        self.splits = None
        self.config = None
        self.feature_meta = None

    def fit(self, data_dest: str) -> "AutoML":
        print(f"[AutoML] Entraînement sur {data_dest}...")
        # Ressources lues à chaque fit (l'évaluateur peut les changer entre deux appels)
        self.config = ResourceConfig.from_env()
        limit_native_threads(self.config.n_jobs)
        set_seed()
        self.timer = TimeBudget(self.config.time_budget)
        ds = load_dataset(data_dest)
        self.feature_meta = ds.meta  # pour relire le test exactement comme le train

        # Étape 1 : inspection et détection de la tâche
        self.task = detect_task(ds.y)
        stratify = self.task.stratify_labels(self.task.encode_y(ds.y))
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(ds.X, ds.y, stratify)

        # Le profil est calculé sur le train seulement (pas de fuite de données)
        self.profile = inspect_data(X_train, ds.types, y_train)
        self.profile.task = self.task
        y_train, y_valid, y_test = (self.task.encode_y(y) for y in (y_train, y_valid, y_test))
        print(f"[AutoML] Jeu {ds.name} :\n{self.profile.summary()}")

        # Étape 2 : prétraitement ajusté sur le train, appliqué au valid / test
        self.preprocessor = build_preprocessor(self.profile)
        Xt_train = self.preprocessor.fit_transform(X_train, y_train)
        self.splits = {
            "train": (Xt_train, y_train),
            "valid": (self.preprocessor.transform(X_valid), y_valid),
            "test": (self.preprocessor.transform(X_test), y_test),
        }
        print(f"[AutoML] Après prétraitement : {Xt_train.shape[1]} variables")
        return self

    def eval(self) -> dict:
        print("[AutoML] Évaluation...")
        return {}

    def predict(self, test_dest: str) -> np.ndarray:
        print(f"[AutoML] Prédiction sur {test_dest}...")
        return np.array([])


# Instance singleton pour l'interface module (automl.fit, automl.eval, automl.predict)
_default_instance = AutoML()

def fit(data_dest: str) -> AutoML:
    return _default_instance.fit(data_dest)

def eval() -> dict:
    return _default_instance.eval()

def predict(test_dest: str) -> np.ndarray:
    return _default_instance.predict(test_dest)

__all__ = ["AutoML", "fit", "eval", "predict", "DataProfile", "TaskInfo"]
