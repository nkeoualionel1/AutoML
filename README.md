# AutoML – Machine Learning Challenge

Projet AAG (Méthodologie IA et méthodes classiques) – M1 Informatique parcours IA, Le Mans Université.

Pipeline AutoML qui prépare les données, compare plusieurs modèles scikit-learn, optimise leurs hyperparamètres et prédit sur des jeux de données inconnus.

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

```bash
bash script/setup_autoML.bash
source ./automl_env/bin/activate
python check_autoML.py   # vérifie que l'environnement est conforme
```

## Utilisation

```python
import automl

automl.fit("/path/to/data")
automl.eval()
preds = automl.predict("/path/to/test")
```

## Données

Sur le cluster Skinner : `/info/corpus/ChallengeMachineLearning`

Une copie locale est aussi disponible dans [dataset/](dataset/). Chaque jeu contient trois fichiers :

- `<nom>.data` : les variables d'entrée X (format dense, ou creux `indice:valeur` pour `data_F`)
- `<nom>.type` : le type de chaque colonne (`Numerical`, `Categorical`, `Binary`)
- `<nom>.solution` : les cibles y

| Jeu      | Lignes | Variables | Types de colonnes               | Tâche                               |
|----------|-------:|----------:|---------------------------------|-------------------------------------|
| `data_A` | 34 190 |        24 | 11 numériques, 13 catégorielles | Multi-label (3 labels binaires)     |
| `data_B` |  5 000 |        16 | 16 numériques                   | Régression                          |
| `data_C` | 15 000 |     1 568 | 1 568 numériques                | Multiclasse one-hot (10 classes)    |
| `data_D` |  2 984 |       144 | 8 numériques, 136 binaires      | Classification binaire (équilibrée) |
| `data_E` |  3 140 |       259 | 259 numériques                  | Classification binaire (équilibrée) |
| `data_F` | 13 142 |    61 188 | 61 188 numériques (creux)       | Multiclasse one-hot (20 classes)    |

---

## Avancement du projet

### 1. Mise en place de l'environnement

- [script/setup_autoML.bash](script/setup_autoML.bash) : crée le venv `automl_env`, installe uniquement les packages autorisés (numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, plotext, visidata, ipykernel), vérifie l'absence des bibliothèques interdites et enregistre le kernel Jupyter.
- [check_autoML.py](check_autoML.py) : script de contrôle qui affiche la version de chaque package autorisé et alerte si un package interdit est présent (torch, tensorflow, keras, xgboost, lightgbm, catboost, requests, urllib3).

### 2. Squelette du package `automl`

- [pyproject.toml](pyproject.toml) : configuration du package installable (`pip install .`), Python ≥ 3.9.
- [automl/\_\_init\_\_.py](automl/__init__.py) : classe `AutoML` avec les méthodes `fit`, `eval` et `predict`, exposées au niveau du module via une instance unique (`automl.fit(...)`, etc.). Pour l'instant les méthodes sont des **stubs** : elles affichent un message mais n'entraînent encore aucun modèle.
- [.gitignore](.gitignore) : exclut les environnements virtuels, caches et dumps de modèles.

### 3. Exploration des données – [notebooks/01_exploration_donnees.ipynb](notebooks/01_exploration_donnees.ipynb)

- Fonction de chargement universelle qui gère les matrices **denses** (`np.loadtxt`) et **creuses** au format LibSVM (`scipy.sparse.csr_matrix`).
- Tableau récapitulatif des 6 jeux : dimensions, format, forme de la solution, type de problème.
- Visualisation des cibles : distribution des valeurs (`data_B`), répartition des classes (`data_D`, `data_C`), fréquence des labels (`data_A`).
- Points clés retenus :
  - il faut manipuler `data_F` en creux sans le convertir en tableau dense (61 188 variables) ;
  - deux grandes familles de modèles : régression et classification ;
  - si la solution est en one-hot, `predict` doit renvoyer du one-hot.

### 4. Baseline V1 – [notebooks/02_baseline_automl.ipynb](notebooks/02_baseline_automl.ipynb)

- **Détection automatique de la tâche** (`detecter_tache`) à partir de la forme et des valeurs de y : régression, classification binaire, multiclasse, multiclasse one-hot, multi-label.
- **Baseline Random Forest** (`entrainer_baseline`) : chargement, sous-échantillonnage à 1 000 lignes, choix automatique de `RandomForestClassifier` ou `RandomForestRegressor`, split train/test, calcul du score.
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
3. Sélection de modèles et optimisation des hyperparamètres
4. Combinaison de modèles (blending / stacking)
5. Évaluation par validation croisée et métriques adaptées

### 6. Étape 1 de la pipeline : inspection et détection

- [automl/data.py](automl/data.py) : chargement d'un dossier `<nom>.data` / `.type` / `.solution` (dense avec valeurs manquantes, ou creux `indice:valeur`). La largeur des matrices creuses est fixée par le fichier `.type`, pour que le jeu de test ait toujours le même nombre de colonnes que le train. Découpage train / valid / test (70 / 15 / 15), stratifié en classification.
- [automl/inspection.py](automl/inspection.py) :
  - `detect_task(y)` reconnaît la tâche : `binary`, `multiclass`, `multiclass_onehot`, `multilabel`, `regression`, `multioutput_regression`. L'objet `TaskInfo` renvoyé convertit y vers le format scikit-learn (`encode_y`), puis les prédictions vers le format `.solution` (`decode_y`, qui reconstruit le one-hot).
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

`automl.fit(...)` enchaîne maintenant ces deux étapes et garde les jeux train / valid / test prétraités dans `self.splits`, prêts pour l'étape 3.
Les paramètres globaux (graine aléatoire, `AUTOML_TIME_BUDGET`, `AUTOML_N_JOBS`, `AUTOML_N_TRIALS`) sont dans [automl/config.py](automl/config.py).

---

## Prochaines étapes

- [x] Transférer la détection de tâche et le chargement des données dans le package `automl/`
- [x] Ajouter le prétraitement (gestion des variables catégorielles de `data_A`, données creuses de `data_F`)
- [ ] Implémenter le modèle dans `fit`, puis `eval` et `predict` (avec sortie one-hot quand nécessaire)
- [ ] Comparer plusieurs modèles et optimiser leurs hyperparamètres
- [ ] Évaluer en validation croisée sur les 6 jeux et battre les scores de la baseline
- [ ] Préparer le poster et la soutenance
