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
from .metrics import all_metrics  # noqa: E402
from .models import data_shape, get_candidates, rows_within  # noqa: E402
from .preprocessing import build_preprocessor  # noqa: E402

# Répartition du budget temps : le reste couvre chargement, prétraitement et marge
MODEL_SHARE = 0.50   # coût estimé total des modèles essayés
REFIT_SHARE = 0.75   # coût estimé total, réentraînement final sur train + valid compris
MIN_ROWS = 500       # en dessous, un modèle n'est pas essayé (sauf le premier)


def _stack(a, b):
    """Empile deux matrices (denses ou creuses) ligne par ligne."""
    return sp.vstack([a, b], format="csr") if sp.issparse(a) else np.vstack([a, b])


class AutoML:
    """Pipeline AutoML : inspection -> prétraitement -> choix du modèle sous budget temps."""

    def __init__(self):
        self.model = None
        self.task = None
        self.profile = None
        self.preprocessor = None
        self.splits = None
        self.config = None
        self.feature_meta = None
        self.leaderboard = []  # un dict par modèle essayé : model, rows, estimated_s, duration_s + métriques de validation
        self.metrics = {}

    def fit(self, data_dest: str) -> "AutoML":
        print(f"[AutoML] Entraînement sur {data_dest}...")
        self.prepare(data_dest)
        # Étape 3 : choix du modèle sur la validation, puis réentraînement sur train + valid
        self._select_model()
        return self

    def prepare(self, data_dest: str) -> "AutoML":
        """Étapes 1 et 2 seules : chargement, inspection, découpage, prétraitement.

        Utile pour tester le prétraitement sans entraîner de modèle.
        """
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

    def _select_model(self) -> None:
        """Entraîne les candidats selon un plan fixé à l'avance, garde le meilleur.

        Le plan (quels modèles, sur combien de lignes) est calculé à partir des
        coûts estimés, qui ne dépendent que de la taille des données et du budget :
        deux exécutions font donc exactement les mêmes choix (reproductibilité).
        L'horloge ne sert que de filet de sécurité si la machine est bien plus
        lente que prévu.
        """
        Xt_train, y_train = self.splits["train"]
        Xt_valid, y_valid = self.splits["valid"]
        Xt_test, y_test = self.splits["test"]
        task = self.task.task
        n_train = Xt_train.shape[0]
        shape = data_shape(Xt_train, self.task, self.config.n_jobs)
        budget = self.config.time_budget
        best = None  # (score, nom, modèle, lignes utilisées, candidat)
        self.leaderboard = []

        planned = 0.0  # somme des coûts estimés des modèles déjà lancés
        for cand in get_candidates(self.profile, self.config.n_jobs):
            # Le premier candidat est toujours entraîné (au besoin sur moins de lignes)
            left = MODEL_SHARE * budget - planned
            n_fit = rows_within(cand, shape, n_train, left, MIN_ROWS)
            if n_fit == 0:
                if best is not None:
                    print(f"[AutoML] Budget temps : {cand.name} non essayé "
                          f"(estimé {cand.cost(shape, min(MIN_ROWS, n_train)):.0f}s au minimum)")
                    continue
                n_fit = min(MIN_ROWS, n_train)
            estimate = cand.cost(shape, n_fit)
            planned += estimate
            if best is not None and not self.timer.has_time(needed=estimate):
                print(f"[AutoML] Arrêt de sécurité : plus assez de temps pour {cand.name}")
                break

            start = time.perf_counter()
            try:
                model = cand.build().fit(Xt_train[:n_fit], y_train[:n_fit])
                valid_metrics = all_metrics(task, y_valid, model.predict(Xt_valid))
                score = valid_metrics["score"]
            except Exception as exc:  # un modèle qui échoue ne doit pas arrêter fit()
                print(f"[AutoML] {cand.name} a échoué : {exc}")
                continue
            duration = time.perf_counter() - start
            self.leaderboard.append({"model": cand.name, "rows": n_fit, "estimated_s": estimate,
                                     "duration_s": duration, **valid_metrics})
            sous_ech = "" if n_fit == n_train else f", {n_fit}/{n_train} lignes"
            print(f"[AutoML]   {cand.name:24s} score valid = {score:.4f} "
                  f"({duration:.1f}s, estimé {estimate:.1f}s{sous_ech})")
            if best is None or score > best[0]:
                best = (score, cand.name, model, n_fit, cand)

        if best is None:
            raise RuntimeError("Aucun modèle n'a pu être entraîné.")
        score, name, model, n_fit, cand = best

        # Métriques honnêtes : le test interne n'a servi à aucun choix
        self.metrics = {f"valid_{k}": v for k, v in all_metrics(task, y_valid, model.predict(Xt_valid)).items()}
        self.metrics.update({f"test_{k}": v for k, v in all_metrics(task, y_test, model.predict(Xt_test)).items()})
        self.metrics["model"] = name

        # Plus de données = meilleur modèle final. Décision prise sur l'estimation
        # (reproductible) ; l'horloge ne sert que de filet de sécurité.
        self.model = model
        refit_cost = cand.cost(shape, n_train + Xt_valid.shape[0])
        if (n_fit == n_train and planned + refit_cost <= REFIT_SHARE * budget
                and self.timer.has_time(needed=refit_cost)):
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
