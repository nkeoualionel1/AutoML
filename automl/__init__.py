import os as _os

# Doit passer avant l'import de numpy / scikit-learn : leurs pools de threads
# natifs (OpenMP, BLAS) lisent ces variables à l'initialisation.
if _os.environ.get("AUTOML_N_JOBS", "").strip().isdigit():
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_var, str(max(1, int(_os.environ["AUTOML_N_JOBS"]))))

import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from .config import ResourceConfig, TimeBudget, limit_native_threads, set_seed  # noqa: E402
from .data import load_dataset, split_data  # noqa: E402
from .inspection import DataProfile, TaskInfo, detect_task, inspect_data  # noqa: E402
from .metrics import all_metrics, primary_score  # noqa: E402
from .models import get_candidates  # noqa: E402
from .preprocessing import build_preprocessor  # noqa: E402


def _stack(a, b):
    """Empile deux matrices (denses ou creuses) ligne par ligne."""
    return sp.vstack([a, b], format="csr") if sp.issparse(a) else np.vstack([a, b])


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
        self.leaderboard = []  # (nom du modèle, score validation, durée d'entraînement)
        self.metrics = {}

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

        # Étape 3 : choix du modèle sur la validation, puis réentraînement sur train + valid
        self._select_model()
        return self

    def _select_model(self) -> None:
        """Essaie chaque candidat tant que le budget temps le permet, garde le meilleur."""
        Xt_train, y_train = self.splits["train"]
        Xt_valid, y_valid = self.splits["valid"]
        Xt_test, y_test = self.splits["test"]
        task = self.task.task
        best = None  # (score, nom, modèle, durée, candidat)
        self.leaderboard = []

        slowest = 0.0  # durée du modèle le plus lent entraîné jusqu'ici
        for cand in get_candidates(self.profile, self.config.n_jobs):
            # Toujours au moins un modèle. Ensuite, les candidats sont de plus en plus
            # coûteux : on estime le suivant à 4x le plus lent déjà vu (prudent) et on
            # ne le lance que s'il tient dans le budget restant.
            if best is not None and not self.timer.has_time(needed=4.0 * slowest):
                print(f"[AutoML] Budget temps : {cand.name} non essayé (trop long estimé)")
                continue
            start = time.perf_counter()
            try:
                model = cand.build().fit(Xt_train, y_train)
                score = primary_score(task, y_valid, model.predict(Xt_valid))
            except Exception as exc:  # un modèle qui échoue ne doit pas arrêter fit()
                print(f"[AutoML] {cand.name} a échoué : {exc}")
                continue
            duration = time.perf_counter() - start
            slowest = max(slowest, duration)
            self.leaderboard.append((cand.name, score, duration))
            print(f"[AutoML]   {cand.name:24s} score valid = {score:.4f} ({duration:.1f}s)")
            if best is None or score > best[0]:
                best = (score, cand.name, model, duration, cand)

        if best is None:
            raise RuntimeError("Aucun modèle n'a pu être entraîné.")
        score, name, model, duration, cand = best

        # Métriques honnêtes : le test interne n'a servi à aucun choix
        self.metrics = {f"valid_{k}": v for k, v in all_metrics(task, y_valid, model.predict(Xt_valid)).items()}
        self.metrics.update({f"test_{k}": v for k, v in all_metrics(task, y_test, model.predict(Xt_test)).items()})
        self.metrics["model"] = name

        # Plus de données = meilleur modèle final, si le temps le permet
        self.model = model
        if self.timer.has_time(needed=1.5 * duration):
            try:
                self.model = cand.build().fit(_stack(Xt_train, Xt_valid), np.concatenate([y_train, y_valid]))
            except Exception as exc:
                print(f"[AutoML] Réentraînement final impossible ({exc}) : on garde le modèle du train")
        print(f"[AutoML] Modèle retenu : {name} (fit total {self.timer.elapsed:.1f}s)")

    def eval(self) -> dict:
        """Affiche et renvoie les métriques de validation et du test interne."""
        if self.model is None:
            raise RuntimeError("Appelez fit() avant eval().")
        print("[AutoML] Évaluation...")
        for k, v in self.metrics.items():
            print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
        return dict(self.metrics)

    def predict(self, test_dest: str) -> np.ndarray:
        """Prédit sur `<test_dest>/<nom>.data`, au format exact du fichier `.solution`."""
        if self.model is None:
            raise RuntimeError("Appelez fit() avant predict().")
        print(f"[AutoML] Prédiction sur {test_dest}...")
        ds = load_dataset(test_dest, meta=self.feature_meta)  # relu comme le train
        return self.task.decode_y(self.model.predict(self.preprocessor.transform(ds.X)))


# Instance singleton pour l'interface module (automl.fit, automl.eval, automl.predict)
_default_instance = AutoML()

def fit(data_dest: str) -> AutoML:
    return _default_instance.fit(data_dest)

def eval() -> dict:
    return _default_instance.eval()

def predict(test_dest: str) -> np.ndarray:
    return _default_instance.predict(test_dest)

__all__ = ["AutoML", "fit", "eval", "predict", "DataProfile", "TaskInfo"]
