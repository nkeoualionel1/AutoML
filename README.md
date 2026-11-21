# AutoML – Machine Learning Challenge

Projet AAG (Méthodologie IA et méthodes classiques) – M1 Informatique parcours IA, Le Mans Université.

Pipeline AutoML qui prépare les données, compare plusieurs modèles scikit-learn sous un budget temps, choisit le meilleur et prédit sur des jeux de données inconnus.

## Équipe

- `NKEOUA MBEMBA Lionel Sachat`
- `AMINE MOULAI`
- `FARACH Zakaria`

**Enseignants référents :** Aghilas Sini, Théo Mariotte, Nicolas Dugué

## Installation

```bash
pip install .
```

Ou, pour créer l'environnement complet (venv + packages autorisés + kernel Jupyter) :

#### Linux / Skinner

```bash
bash scripts/setup_autoML.bash
source ./automl_env/bin/activate
python check_autoML.py   # vérifie que l'environnement est conforme
```

#### Windows (PowerShell)

```powershell
python -m venv automl_env
.\automl_env\Scripts\python.exe -m pip install . pytest
```

Pour taper simplement `python` au lieu de `.\automl_env\Scripts\python.exe`, activez le venv avec `.\automl_env\Scripts\Activate.ps1`. Si PowerShell bloque l'activation, lancez une fois `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Utilisation

```python
import automl

automl.fit("/path/to/data")      # charge, inspecte, prétraite, choisit et entraîne le modèle
automl.eval()                    # affiche et renvoie les métriques (validation + test interne)
preds = automl.predict("/path/to/test")   # même forme que le fichier .solution
```

Ressources lues à chaque `fit` (variables d'environnement imposées par le sujet) :

| Variable | Rôle | Défaut |
| --- | --- | --- |
| `AUTOML_TIME_BUDGET` | temps maximal d'un `fit` (secondes) | 600 |
| `AUTOML_N_JOBS` | nombre de cœurs (`-1` est ramené à 1) | 1 |
| `AUTOML_N_TRIALS` | nombre maximal d'essais d'optimisation (étape 4, à venir) | 20 |

## Données

Sur le cluster Skinner : `/info/corpus/ChallengeMachineLearning`

Pour travailler en local, copiez ce dossier dans `dataset/` à la racine du projet. `dataset/` est **ignoré par git** : `data_C.data` fait 65 Mo, alors que le sujet interdit tout fichier de plus de 5 Mo dans le dépôt.

Chaque jeu contient trois fichiers :

- `<nom>.data` : les variables d'entrée X (format dense, ou creux `indice:valeur` pour `data_F`)
- `<nom>.type` : le type de chaque colonne (`Numerical`, `Categorical`, `Binary`)
- `<nom>.solution` : les cibles y (absent du dossier de test au moment de `predict`)

| Jeu      | Lignes | Variables | Types de colonnes               | Tâche                               |
|----------|-------:|----------:|---------------------------------|-------------------------------------|
| `data_A` | 34 190 |        24 | 11 numériques, 13 catégorielles | Multi-label (3 labels binaires)     |
| `data_B` |  5 000 |        16 | 16 numériques                   | Régression                          |
| `data_C` | 15 000 |     1 568 | 1 568 numériques                | Multiclasse one-hot (10 classes)    |
| `data_D` |  2 984 |       144 | 8 numériques, 136 binaires      | Classification binaire (équilibrée) |
| `data_E` |  3 140 |       259 | 259 numériques                  | Classification binaire (équilibrée) |
| `data_F` | 13 142 |    61 188 | 61 188 numériques (creux)       | Multiclasse one-hot (20 classes)    |

---

## Lancer les tests et voir les métriques

Toutes les commandes se lancent **depuis la racine du projet**. Sous Windows, remplacez `python` par `.\automl_env\Scripts\python.exe` si le venv n'est pas activé.

### 1. Tests automatiques (`pytest`) : est-ce que rien n'est cassé ?

```bash
python -m pytest tests          # résumé : "26 passed"
python -m pytest tests -v       # le détail, test par test
python -m pytest tests/test_automl_contract.py   # un seul fichier
```

À lancer **avant chaque commit et chaque fusion de branche**. Les tests utilisent de petits jeux créés à la volée : ils ne demandent pas `dataset/` et prennent environ 40 s. Chaque fichier explique en en-tête son rôle et quand le lancer :

| Fichier | Ce qu'il vérifie |
| --- | --- |
| `tests/test_data.py` | chargement et pièges du test (`?`, texte, colonnes en moins, indices creux), découpage |
| `tests/test_inspection.py` | détection des 6 types de tâches, aller-retour de la cible, stratification |
| `tests/test_preprocessing.py` | prétraitement face aux valeurs extrêmes (colonne vide, `inf`, modalité inconnue) |
| `tests/test_config.py` | lecture de `AUTOML_*` au moment du `fit` |
| `tests/test_compliance.py` | règles du sujet : imports interdits, `n_jobs=-1`, fichiers > 5 Mo |
| `tests/test_automl_contract.py` | `fit` / `eval` / `predict` de bout en bout sur toutes les tâches |
| `tests/test_model_selection.py` | budget temps respecté et choix du modèle reproductible |

### 2. Vérifier le prétraitement (étapes 1 et 2) sur des jeux complets

```bash
python scripts/check_preprocessing.py --folder dataset   # nos 6 jeux -> "6/6 jeux OK"
python scripts/generate_test_datasets.py                 # une fois : crée test_datasets/ (20 jeux)
python scripts/check_preprocessing.py                    # les 20 jeux -> "20/20 jeux OK"
python scripts/check_preprocessing.py test_adult         # un seul jeu
```

Pour chaque jeu, le script affiche le profil détecté (tâche, types de colonnes, valeurs manquantes, colonnes asymétriques, déséquilibre), puis une série de ✅ / ❌ : aucun NaN ni infini, format creux conservé, même largeur entre train, valid et test, décodage fidèle de la cible. Aucun modèle n'est entraîné, donc c'est rapide (environ 1 min).

### 3. Entraîner les modèles et voir les métriques

```bash
python scripts/evaluate_models.py --budget 60                                   # les 6 jeux, 60 s chacun
python scripts/evaluate_models.py data_E --budget 60                            # un seul jeu
python scripts/evaluate_models.py --budget 60 --csv benchmarks/mon_essai.csv    # + sauvegarde CSV
```

Options : `--budget` (secondes par jeu, défaut 120), `--n-jobs` (cœurs, défaut 2), `--folder` (défaut `dataset`).

Exemple de sortie pour un jeu :

```
=== data_E === (budget 60 s, 2 cœurs)
  Tâche : binary
  modèle                   score valid    durée  estimée   lignes
  logistic_regression           0.5562     0.1s     0.3s     2355
  random_forest                 0.7388     4.4s     4.9s     2355
  hist_gradient_boosting        0.8194     2.2s     8.3s     2355  <- retenu
  Test interne du modèle retenu : score 0.8597  (accuracy=0.8599, balanced_accuracy=0.8596, f1_macro=0.8596)
  fit total : 9.9 s
```

**Comment lire ces colonnes :**

| Colonne | Signification |
| --- | --- |
| `score valid` | score principal sur la **validation** : c'est lui qui sert à choisir le modèle |
| `durée` / `estimée` | temps d'entraînement réel / estimé à l'avance par la formule de coût du modèle |
| `lignes` | nombre de lignes du train utilisées (moins que le total si le modèle est trop coûteux pour le budget) |
| `<- retenu` | le modèle gardé, puis réentraîné sur train + valid si le budget le permet |
| `Test interne` | score sur une partie **jamais utilisée pour choisir** : c'est l'estimation honnête de la performance |

**Le score principal** (plus grand = meilleur) dépend de la tâche, voir [automl/metrics.py](automl/metrics.py) :

| Tâche | Score principal | Métriques détaillées (`eval()`) |
| --- | --- | --- |
| binaire, multiclasse | moyenne de l'accuracy et du F1 macro | accuracy, balanced_accuracy, f1_macro |
| multi-label | moyenne du F1 macro et de l'accuracy par label | subset_accuracy, label_accuracy, f1_macro |
| régression | R² | r2, rmse, mae |

Pour **comparer avant / après** une modification (par exemple Optuna), enregistrez les deux CSV dans `benchmarks/` et comparez-les. Références actuelles : [benchmarks/step3_reference.csv](benchmarks/step3_reference.csv) (avant la correction du budget) et [benchmarks/step3_deterministic_budget.csv](benchmarks/step3_deterministic_budget.csv) (actuelle).

### 4. Dans un notebook

```python
import automl

m = automl.AutoML().fit("../dataset/data_E")
m.eval()            # métriques valid_* et test_*
m.leaderboard       # une ligne par modèle essayé : model, rows, estimated_s, duration_s, score…
m.profile.summary() # ce que l'inspection a détecté
```

Les notebooks [04_steps_1_2_robustness.ipynb](notebooks/04_steps_1_2_robustness.ipynb) et [05_step3_models.ipynb](notebooks/05_step3_models.ipynb) présentent ces résultats sous forme de tableaux et de graphiques.

---

## Avancement du projet

### 1. Mise en place de l'environnement

- [scripts/setup_autoML.bash](scripts/setup_autoML.bash) : crée le venv `automl_env`, installe uniquement les packages autorisés (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, plotext, visidata, ipykernel), vérifie l'absence des bibliothèques interdites et enregistre le kernel Jupyter.
- [check_autoML.py](check_autoML.py) : script de contrôle qui affiche la version de chaque package autorisé et alerte si un package interdit est présent (torch, tensorflow, keras, xgboost, lightgbm, catboost, requests, urllib3).

### 2. Package `automl` installable

- [pyproject.toml](pyproject.toml) : configuration du package installable (`pip install .`), Python ≥ 3.9. Seul le dossier `automl/` est installé (`include = ["automl", "automl.*"]`), et non le venv.
- [automl/\_\_init\_\_.py](automl/__init__.py) : classe `AutoML` avec `fit`, `eval`, `predict`, exposées au niveau du module via une instance unique (`automl.fit(...)`, etc.). Une méthode `prepare()` exécute seulement les étapes 1 et 2, sans entraîner de modèle.
- [.gitignore](.gitignore) : exclut le venv `automl_env/`, les données `dataset/` et `test_datasets/`, les caches et les dumps de modèles.

### 3. Exploration des données – [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)

- Fonction de chargement universelle qui gère les matrices **denses** (`np.loadtxt`) et **creuses** au format LibSVM (`scipy.sparse.csr_matrix`).
- Tableau récapitulatif des 6 jeux (`analyze_dataset`) : dimensions, format, forme de la solution, type de problème.
- Visualisation des cibles : distribution des valeurs (`data_B`), répartition des classes (`data_D`, `data_C`), fréquence des labels (`data_A`).
- Points clés retenus :
  - il faut manipuler `data_F` en creux sans le convertir en tableau dense (61 188 variables) ;
  - deux grandes familles de modèles : régression et classification ;
  - si la solution est en one-hot, `predict` doit renvoyer du one-hot.

### 4. Baseline V1 – [notebooks/02_automl_baseline.ipynb](notebooks/02_automl_baseline.ipynb)

- **Détection automatique de la tâche** (`detect_task_baseline`) à partir de la forme et des valeurs de y.
- **Baseline Random Forest** (`train_baseline`) : sous-échantillonnage à 1 000 lignes, choix automatique de `RandomForestClassifier` ou `RandomForestRegressor`, calcul du score.
- **Scores de référence à battre :**

| Jeu      | Tâche                  | Métrique |  Score |
|----------|------------------------|----------|-------:|
| `data_D` | Classification binaire | Accuracy | 78,0 % |
| `data_C` | Multiclasse (10)       | Accuracy | 89,0 % |
| `data_B` | Régression             | R²       | 0,6605 |

### 5. Architecture de la pipeline

- [crafted_automl_architecture.md](crafted_automl_architecture.md) : description de la pipeline cible et du lien avec le cours.
- Schémas dans [media/](media/) : [pipeline_automl.png](media/pipeline_automl.png) et [automl_workflow_graph.png](media/automl_workflow_graph.png).

Étapes prévues :

1. Inspection et détection (type de tâche, types de variables, valeurs manquantes, déséquilibre)
2. Prétraitement et feature engineering (imputation, encodage, mise à l'échelle, sélection de variables)
3. Sélection de modèles sous budget temps
4. Optimisation des hyperparamètres (Optuna)
5. Combinaison de modèles (blending / stacking)

### 6. Étape 1 de la pipeline : inspection et détection

- [automl/data.py](automl/data.py) : chargement d'un dossier `<nom>.data` / `.type` / `.solution`.
  - Robuste aux pièges d'un jeu de test secret : `?` et `inf` lus comme manquants, texte codé en nombres (même code au test qu'au train), colonnes en moins ou en trop, indices creux commençant à 0 ou hors bornes, ligne vide, étiquettes texte dans `.solution`. Ce qui est appris au train est gardé dans `FeatureMeta` et réutilisé par `predict`.
  - Découpage train / valid / test **adaptatif** : la validation représente 20 % des petits jeux et 8 % des très grands, stratifiée en classification, en multi-label et en régression (par déciles).
- [automl/inspection.py](automl/inspection.py) :
  - `detect_task(y)` reconnaît la tâche : `binary`, `multiclass`, `multiclass_onehot`, `multilabel`, `regression`, `multioutput_regression`. L'objet `TaskInfo` convertit y vers le format scikit-learn (`encode_y`), puis les prédictions vers le format `.solution` (`decode_y`, qui reconstruit le one-hot et les étiquettes d'origine).
  - `inspect_data(X, types, y)` renvoie un `DataProfile` : colonnes numériques, catégorielles, binaires et constantes, taux de valeurs manquantes, cardinalité des catégorielles, variables asymétriques, densité, données de comptage creuses, répartition des classes et déséquilibre, asymétrie de la cible.
  - Le profil est calculé **sur le train uniquement**, pour éviter les fuites de données.

### 7. Étape 2 de la pipeline : prétraitement et feature engineering

[automl/preprocessing.py](automl/preprocessing.py) : `build_preprocessor(profile)` construit une pipeline scikit-learn adaptée au profil. Elle est ajustée sur le train, puis appliquée au valid et au test.

| Type de colonne                 | Traitement                                                                              |
|---------------------------------|-----------------------------------------------------------------------------------------|
| Numérique                       | imputation par la médiane (+ indicateur de manquant) puis `StandardScaler`              |
| Numérique asymétrique (≥ 0)     | imputation, `log1p`, puis `StandardScaler`                                              |
| Catégorielle                    | imputation par le mode, puis one-hot (20 modalités max, modalités rares regroupées)     |
| Binaire                         | imputation par le mode, puis `MinMaxScaler`                                             |
| Constante                       | supprimée                                                                               |
| Creux, données de comptage      | suppression des colonnes vides puis TF-IDF (la matrice reste creuse)                    |
| Creux, autres données           | suppression des colonnes vides puis `MaxAbsScaler`                                      |
| Dense très large (> 5 000 var.) | sélection des meilleures variables (`SelectKBest`)                                      |

Résultat sur les 6 jeux (partie train) :

| Jeu      | Tâche détectée           | Variables avant | Variables après | Remarques                                       |
|----------|--------------------------|----------------:|----------------:|-------------------------------------------------|
| `data_A` | multilabel (3)           |              24 |             173 | 6 colonnes avec manquants, labels déséquilibrés |
| `data_B` | regression               |              16 |              16 | 8 variables asymétriques passées en log         |
| `data_C` | multiclass_onehot (10)   |           1 568 |           1 568 | 4 colonnes binaires                             |
| `data_D` | binary                   |             144 |             143 | 1 colonne constante supprimée                   |
| `data_E` | binary                   |             259 |             259 | classes équilibrées                             |
| `data_F` | multiclass_onehot (20)   |          61 188 |          53 580 | creux, TF-IDF, 7 608 colonnes vides retirées    |

### 8. Vérification du prétraitement sur 20 jeux supplémentaires

- [scripts/generate_test_datasets.py](scripts/generate_test_datasets.py) crée `test_datasets/` (ignoré par git) au même format que le challenge :
  - **jeux réels proches des nôtres** : Adult, Titanic, California Housing, MNIST, 20 Newsgroups, Digits, Breast Cancer, Wine avec étiquettes 3/7/12, Diabetes ;
  - **jeux synthétiques** : classes déséquilibrées, multi-label avec catégorielles et NaN, régression à 3 sorties, 5 500 variables, creux avec valeurs négatives ;
  - **jeux pièges** `trap_*`, avec un sous-dossier `train/` et un sous-dossier `test/` sans solution : catégorie inconnue, NaN au test seulement, valeurs négatives dans une colonne passée en log, test d'une seule ligne, colonne entièrement vide, dernières colonnes creuses absentes du test.
- [scripts/check_preprocessing.py](scripts/check_preprocessing.py) vérifie les étapes 1 et 2 sur chaque jeu : **20/20 jeux OK** et **6/6** sur nos données.
- [notebooks/03_inspection_preprocessing.ipynb](notebooks/03_inspection_preprocessing.ipynb) : vérification détaillée, jeu par jeu, sur les 6 jeux d'origine.

### 9. Étape 3 de la pipeline : choix du modèle sous budget temps

- [automl/models.py](automl/models.py) : modèles candidats selon la tâche et le format, du moins coûteux au plus coûteux.

| Données | Tâche | Candidats |
| --- | --- | --- |
| Denses | classification | régression logistique → forêt aléatoire → HistGradientBoosting |
| Denses | régression | Ridge → forêt aléatoire → HistGradientBoosting |
| Creuses (texte) | classification | LinearSVC → régression logistique → Complement Naive Bayes |
| Creuses | régression | Ridge |

- [automl/metrics.py](automl/metrics.py) : score principal et métriques détaillées par tâche (voir plus haut).
- `fit` entraîne chaque candidat sur le train, le note sur la validation, garde le meilleur, mesure ses performances sur le test interne, puis le réentraîne sur train + valid.
- **Budget temps et reproductibilité.** Chaque candidat a une **formule de coût estimé** (`linear_cost`, `forest_cost`, `boosting_cost`), calibrée sur nos 6 jeux et qui ne dépend que de la taille des données. Le plan d'entraînement (quels modèles, sur combien de lignes) est fixé à l'avance : 50 % du budget pour les modèles, 75 % avec le réentraînement final. Un modèle trop coûteux est entraîné sur un sous-échantillon plutôt que sauté. Le chronomètre ne sert que de filet de sécurité.
  - Avant : `fit` dépassait le budget (112 s pour 60 s sur `data_A`) et pouvait choisir un modèle différent d'une exécution à l'autre.
  - Après : le budget est respecté sur les 6 jeux, et deux exécutions donnent exactement les mêmes choix et les mêmes prédictions (vérifié par `tests/test_model_selection.py`).

**Résultats actuels** (budget 60 s, 2 cœurs, [benchmarks/step3_deterministic_budget.csv](benchmarks/step3_deterministic_budget.csv)) :

| Jeu | Tâche | Modèle retenu | Score valid | Test interne | Baseline (notebook 02) | `fit` |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `data_A` | multi-label | forêt aléatoire | 0,881 | 0,877 | – | 16 s |
| `data_B` | régression | HistGradientBoosting | 0,799 | R² = 0,816 | R² = 0,661 | 8 s |
| `data_C` | multiclasse (10) | forêt aléatoire | 0,942 | 0,926 (acc. 92,7 %) | acc. 89,0 % | 35 s |
| `data_D` | binaire | forêt aléatoire | 0,804 | 0,793 (acc. 79,6 %) | acc. 78,0 % | 4 s |
| `data_E` | binaire | HistGradientBoosting | 0,819 | 0,860 | – | 10 s |
| `data_F` | multiclasse (20) | LinearSVC | 0,899 | 0,892 | – | 38 s |

### 10. Tests automatiques et organisation du code

- Dossier [tests/](tests/) : un fichier de test par module (voir le tableau plus haut), 26 tests. Les outils partagés sont dans `tests/conftest.py`.
- Noms de fichiers, de fonctions et de tests en anglais ; commentaires et messages en français pour l'équipe.
- [PROJECT_JOURNAL.md](PROJECT_JOURNAL.md) : journal des décisions techniques (qui a fait quoi, et pourquoi), pour le poster et la soutenance.
- [docs/STEPS_1_2_ROBUSTNESS.md](docs/STEPS_1_2_ROBUSTNESS.md) et [docs/STEP3_FIRST_MODELS.md](docs/STEP3_FIRST_MODELS.md) : explications détaillées des étapes 1 à 3.

### Structure du dépôt

```
automl/        package installé par pip : config, data, inspection, preprocessing, models, metrics
tests/         tests pytest (un fichier par module)
scripts/       outils : installation, génération de jeux de test, vérifications, évaluation
notebooks/     01 exploration, 02 baseline, 03 à 05 visualisation des étapes 1 à 3
benchmarks/    scores enregistrés par scripts/evaluate_models.py (comparaison avant / après)
docs/          explications détaillées pour l'équipe
dataset/       données du challenge (local, ignoré par git)
test_datasets/ jeux de test générés (local, ignoré par git)
```

---

## Prochaines étapes

- [x] Transférer la détection de tâche et le chargement des données dans le package `automl/`
- [x] Ajouter le prétraitement (gestion des variables catégorielles de `data_A`, données creuses de `data_F`)
- [x] Vérifier le prétraitement sur des jeux supplémentaires et des cas pièges
- [x] Implémenter le modèle dans `fit`, puis `eval` et `predict` (avec sortie one-hot quand nécessaire)
- [x] Respecter le budget temps et rendre le choix du modèle reproductible
- [x] Battre les scores de la baseline (`data_B`, `data_C`, `data_D`)
- [ ] Gains rapides : `class_weight="balanced"` quand les classes sont déséquilibrées, transformation log de la cible asymétrique
- [ ] Optimiser les hyperparamètres avec Optuna (`AUTOML_N_TRIALS`), avec validation croisée sur les petits jeux
- [ ] Combiner les meilleurs modèles (ensembling)
- [ ] Vérifier le budget temps et les scores sur Skinner
- [ ] Préparer le poster et la soutenance
