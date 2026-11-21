import os as _os

# Doit passer avant l'import de numpy / scikit-learn : leurs pools de threads
# natifs (OpenMP, BLAS) lisent ces variables à l'initialisation.
if _os.environ.get("AUTOML_N_JOBS", "").strip().isdigit():
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ.setdefault(_var, str(max(1, int(_os.environ["AUTOML_N_JOBS"]))))

import time  # noqa: E402
from typing import List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402

from .config import RANDOM_STATE, ResourceConfig, TimeBudget, limit_native_threads, set_seed  # noqa: E402
from .data import load_dataset, split_data  # noqa: E402
from .inspection import DataProfile, TaskInfo, detect_task, inspect_data  # noqa: E402
from .metrics import all_metrics  # noqa: E402
from .models import data_shape, get_candidates, no_space, rows_within  # noqa: E402
from .preprocessing import build_preprocessor  # noqa: E402
from .selection import CV_FOLDS, AveragingEnsemble, Evaluator, Result, choose_cv, stack_rows  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Répartition du budget temps (coûts estimés cumulés). Le reste couvre le
# chargement, le prétraitement et une marge de sécurité.
SCREEN_SHARE = 0.45  # phase 1 : chaque candidat avec ses réglages par défaut
HPO_SHARE = 0.65     # phase 2 : + essais Optuna
FINAL_SHARE = 0.80   # phase 4 : + réentraînement final sur train + valid
MIN_ROWS = 500       # en dessous, un modèle n'est pas essayé (sauf le premier)
MAX_TUNED = 2        # nombre de candidats optimisés par Optuna (les meilleurs)
ENSEMBLE_MAX = 3     # nombre maximal de modèles moyennés dans un ensemble


class AutoML:
    """Pipeline AutoML : inspection -> prétraitement -> modèles, Optuna et ensembling sous budget temps."""

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
        # Étapes 3 à 5 : candidats, Optuna, ensembling, puis réentraînement sur train + valid
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
        """Étapes 3 à 5 : comparaison des candidats, Optuna, ensembling, modèle final.

        Tout le plan (quels modèles, sur combien de lignes, combien d'essais Optuna)
        est décidé à partir des coûts estimés, qui ne dépendent que de la taille
        des données et du budget : deux exécutions font exactement les mêmes choix
        (reproductibilité). L'horloge ne sert que de filet de sécurité si la
        machine est bien plus lente que prévu.
        """
        Xt_train, y_train = self.splits["train"]
        Xt_valid, y_valid = self.splits["valid"]
        Xt_test, y_test = self.splits["test"]
        budget = self.config.time_budget
        shape = data_shape(Xt_train, self.task, self.config.n_jobs)
        candidates = get_candidates(self.profile, self.config.n_jobs)
        n_all = Xt_train.shape[0] + Xt_valid.shape[0]
        use_cv = choose_cv(candidates, shape, n_all, SCREEN_SHARE * budget)
        ev = Evaluator(self.task, Xt_train, y_train, Xt_valid, y_valid, use_cv)
        print(f"[AutoML] Validation : {f'croisée en {CV_FOLDS} plis' if ev.use_cv else 'simple'}")
        self.leaderboard = []
        self._planned = 0.0  # somme des coûts estimés déjà engagés

        results = self._screen_candidates(ev, shape, budget, candidates)
        if not results:
            raise RuntimeError("Aucun modèle n'a pu être entraîné.")
        results += self._tune_with_optuna(ev, shape, budget, results)
        winner = self._try_ensembles(ev, results)
        self._fit_final(ev, shape, budget, winner, Xt_test, y_test)

    def _run(self, ev: Evaluator, cand, params, rows, estimate, label) -> Optional[Result]:
        """Évalue un candidat, l'ajoute au leaderboard, renvoie None en cas d'échec."""
        start = time.perf_counter()
        try:
            res = ev.evaluate(cand, params, rows)
        except Exception as exc:  # un modèle qui échoue ne doit pas arrêter fit()
            print(f"[AutoML] {label} a échoué : {exc}")
            return None
        res.name, res.estimated_s, res.duration_s = label, estimate, time.perf_counter() - start
        self.leaderboard.append({"model": label, "rows": rows, "estimated_s": estimate,
                                 "duration_s": res.duration_s, **res.metrics})
        sub = "" if rows == ev.n_max else f", {rows}/{ev.n_max} lignes"
        print(f"[AutoML]   {label:34s} score = {res.score:.4f} "
              f"({res.duration_s:.1f}s, estimé {estimate:.1f}s{sub})")
        return res

    def _screen_candidates(self, ev: Evaluator, shape, budget, candidates) -> List[Result]:
        """Phase 1 : chaque candidat avec ses réglages par défaut."""
        results = []
        for cand in candidates:
            left = SCREEN_SHARE * budget - self._planned
            rows = rows_within(lambda n: ev.cost(cand, shape, n), ev.n_max, left, MIN_ROWS)
            if ev.use_cv and rows < ev.n_max:
                rows = 0  # en validation croisée, on garde toutes les lignes ou on saute
            if rows == 0:
                if results:
                    print(f"[AutoML] Budget temps : {cand.name} non essayé")
                    continue
                rows = min(MIN_ROWS, ev.n_max)  # le premier candidat est toujours essayé
            estimate = ev.cost(cand, shape, rows)
            if results and not self.timer.has_time(needed=estimate):
                print(f"[AutoML] Arrêt de sécurité : plus assez de temps pour {cand.name}")
                break
            self._planned += estimate
            res = self._run(ev, cand, {}, rows, estimate, cand.name)
            if res is not None:
                results.append(res)
        return results

    def _tune_with_optuna(self, ev: Evaluator, shape, budget, screened: List[Result]) -> List[Result]:
        """Phase 2 : Optuna sur les meilleurs candidats, dans la limite de N_TRIALS et du budget.

        Avant chaque essai, son coût estimé est comparé au budget restant : la suite
        des essais ne dépend donc pas du chronomètre (reproductible).
        """
        n_trials = self.config.n_trials
        tunable = [r for r in sorted(screened, key=lambda r: -r.score) if r.cand.space is not no_space]
        tunable = tunable[:MAX_TUNED]
        if n_trials <= 0 or not tunable:
            return []
        # Les essais sont répartis entre les candidats optimisés (le meilleur d'abord)
        per_model = [n_trials // len(tunable) + (1 if i < n_trials % len(tunable) else 0)
                     for i in range(len(tunable))]
        tuned = []
        for base, trials in zip(tunable, per_model):
            if trials == 0:
                continue
            cand, rows = base.cand, base.rows
            best: List[Result] = []

            def objective(trial):
                params = cand.space(trial, shape)
                estimate = ev.cost(cand, shape, rows, params)
                if (self._planned + estimate > HPO_SHARE * budget
                        or not self.timer.has_time(needed=estimate)):
                    trial.study.stop()  # plus de budget : on arrête cette étude
                    raise optuna.TrialPruned()
                self._planned += estimate
                res = self._run(ev, cand, params, rows, estimate, f"{cand.name} (optuna #{trial.number})")
                if res is None:
                    raise optuna.TrialPruned()
                if not best or res.score > best[0].score:
                    best[:] = [res]
                return res.score

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study.optimize(objective, n_trials=trials)
            if best and best[0].score > base.score:
                tuned.append(best[0])
                print(f"[AutoML] Optuna améliore {cand.name} : {base.score:.4f} -> {best[0].score:.4f}")
        return tuned

    def _try_ensembles(self, ev: Evaluator, results: List[Result]):
        """Phase 3 : moyenne des 2 ou 3 meilleurs modèles différents, gardée si elle gagne."""
        best_single = max(results, key=lambda r: r.score)
        winner = (best_single.score, best_single.name, [best_single], best_single.metrics)
        # Le meilleur réglage de chaque famille de modèles, s'il donne des probabilités
        per_family = {}
        for r in sorted(results, key=lambda r: -r.score):
            if r.soft is not None and r.cand.name not in per_family:
                per_family[r.cand.name] = r
        ranked = list(per_family.values())
        for k in range(2, min(ENSEMBLE_MAX, len(ranked)) + 1):
            members = ranked[:k]
            metrics = ev.score_ensemble(members)
            label = "ensemble(" + " + ".join(m.cand.name for m in members) + ")"
            self.leaderboard.append({"model": label, "rows": max(m.rows for m in members),
                                     "estimated_s": 0.0, "duration_s": 0.0, **metrics})
            print(f"[AutoML]   {label:34s} score = {metrics['score']:.4f}")
            if metrics["score"] > winner[0]:
                winner = (metrics["score"], label, members, metrics)
        return winner

    def _fit_final(self, ev: Evaluator, shape, budget, winner, Xt_test, y_test) -> None:
        """Phase 4 : réentraîne le gagnant sur train + valid si le budget le permet."""
        score, name, members, valid_metrics = winner
        X_all = stack_rows(self.splits["train"][0], self.splits["valid"][0])
        y_all = np.concatenate([self.splits["train"][1], self.splits["valid"][1]])
        n_all = X_all.shape[0]

        def refit_cost(n):
            return sum(m.cand.estimate(shape, n, m.params) for m in members)

        left = max(FINAL_SHARE * budget - self._planned, 0.0)
        rows = rows_within(refit_cost, n_all, left, MIN_ROWS)
        have_models = all(m.model is not None for m in members)
        if rows == n_all and self.timer.has_time(needed=refit_cost(n_all)):
            models = [m.cand.build(m.params).fit(X_all, y_all) for m in members]
        elif have_models:
            models = [m.model for m in members]  # déjà entraînés sur le train : on les garde
        else:
            # Validation croisée sans modèle gardé : entraînement sur ce qui tient dans le budget
            rows = rows or min(MIN_ROWS, n_all)
            models = [m.cand.build(m.params).fit(X_all[:rows], y_all[:rows]) for m in members]

        self.model = models[0] if len(models) == 1 else AveragingEnsemble(models, self.task)
        # Métriques honnêtes : le test interne n'a servi à aucun choix
        self.metrics = {f"valid_{k}": v for k, v in valid_metrics.items()}
        self.metrics.update({f"test_{k}": v for k, v in
                             all_metrics(self.task.task, y_test, self.model.predict(Xt_test)).items()})
        self.metrics["model"] = name
        self.metrics["validation"] = f"croisée ({CV_FOLDS} plis)" if ev.use_cv else "simple"
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
