"""Étape 2 : prétraitement et feature engineering.

Le préprocesseur est construit à partir du DataProfile (étape 1) et ne doit
être ajusté que sur le train pour éviter toute fuite de données.
"""

from typing import Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from .inspection import (
    BINARY,
    HIGH_CARDINALITY,
    MULTICLASS,
    MULTICLASS_ONEHOT,
    REGRESSION,
    DataProfile,
)

# Au-delà de ce nombre de variables (après nettoyage), on garde les plus informatives
MAX_DENSE_FEATURES = 5000


def _log1p_positive(X):
    # Les valeurs imputées restent >= 0, mais on protège quand même le log
    return np.log1p(np.clip(X, 0, None))


def _numerical_pipeline(log: bool, add_indicator: bool) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=add_indicator))]
    if log:
        steps.append(("log", FunctionTransformer(_log1p_positive, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Les modalités rares sont regroupées, les modalités inconnues ignorées
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=HIGH_CARDINALITY,
            min_frequency=5,
            sparse_output=False,
        )),
    ])


def _binary_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", MinMaxScaler()),
    ])


def _dense_preprocessor(profile: DataProfile) -> ColumnTransformer:
    skewed = set(profile.skewed)
    num_plain = [j for j in profile.numerical if j not in skewed]
    num_skewed = [j for j in profile.numerical if j in skewed]

    def has_missing(cols):
        return any(j in profile.missing for j in cols)

    transformers = []
    if num_plain:
        transformers.append(("num", _numerical_pipeline(False, has_missing(num_plain)), num_plain))
    if num_skewed:
        transformers.append(("num_log", _numerical_pipeline(True, has_missing(num_skewed)), num_skewed))
    if profile.categorical:
        transformers.append(("cat", _categorical_pipeline(), profile.categorical))
    if profile.binary:
        transformers.append(("bin", _binary_pipeline(), profile.binary))

    # Les colonnes constantes ne sont listées nulle part : elles sont supprimées
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _sparse_preprocessor(profile: DataProfile) -> Pipeline:
    steps = [("drop_constant", VarianceThreshold(0.0))]
    if profile.nonnegative_integer:
        # Données de comptage (type sac de mots) : pondération TF-IDF, reste creux
        steps.append(("tfidf", TfidfTransformer(sublinear_tf=True)))
    else:
        # MaxAbsScaler conserve les zéros, donc la matrice reste creuse
        steps.append(("scale", MaxAbsScaler()))
    return Pipeline(steps)


def build_preprocessor(profile: DataProfile, max_features: Optional[int] = None) -> Pipeline:
    """Construit la pipeline de prétraitement adaptée au profil des données."""
    if profile.is_sparse:
        steps = [("prep", _sparse_preprocessor(profile))]
    else:
        steps = [("prep", _dense_preprocessor(profile)), ("drop_constant", VarianceThreshold(0.0))]

    # Sélection de variables (filtre univarié) pour les jeux denses très larges.
    # Elle n'est possible qu'avec une cible 1-D (pas en multi-label).
    if max_features is None and not profile.is_sparse:
        max_features = MAX_DENSE_FEATURES
    task = profile.task.task if profile.task is not None else None
    if (max_features and _n_output_features(profile) > max_features
            and task in (BINARY, MULTICLASS, MULTICLASS_ONEHOT, REGRESSION)):
        score = f_regression if task == REGRESSION else f_classif
        steps.append(("select", SelectKBest(score, k=max_features)))

    return Pipeline(steps)


def _n_output_features(profile: DataProfile) -> int:
    """Estimation du nombre de colonnes produites par le prétraitement."""
    if profile.is_sparse:
        return len(profile.numerical)
    n_cat = sum(min(c, HIGH_CARDINALITY) for c in profile.cardinality.values())
    return len(profile.numerical) + len(profile.binary) + n_cat + len(profile.missing)
