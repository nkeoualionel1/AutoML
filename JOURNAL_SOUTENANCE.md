# Journal de Suivi Technique & Préparation Soutenance — Projet AutoML

**Équipe :** NKEOUA MBEMBA Lionel Sachat, MOULAI Amine, FARACH Zakaria  
**Formation :** M1 Informatique – Parcours IA (Le Mans Université)  
**Objectif de ce document :** Tracer chronologiquement chaque décision technique, expliquer **ce qui a été fait** et **pourquoi**, afin de structurer le poster et de répondre avec précision aux questions du jury lors de la soutenance orale.

---

## Vue d'ensemble du Fil Conducteur (De l'Exploration au Système Final)

| Phase | Statut | Module / Livrable Clé | Problème Résolu |
| :--- | :---: | :--- | :--- |
| **Phase 0 : Environnement & Conformité** | Terminé | `setup_autoML.bash`, `check_autoML.py` | Garantir le respect strict des bibliothèques autorisées sur Skinner |
| **Phase 1 : Exploration (EDA)** | Terminé | `01_exploration_donnees.ipynb` | Comprendre la diversité des formats (dense, creux 61k cols) et cibles |
| **Phase 2 : Baseline V1 (Prototype)** | Terminé | `02_baseline_automl.ipynb` | Valider la faisabilité sur un sous-échantillon ($1\,000$ lignes) |
| **Phase 3 : Socle Config & Données (Étape 1)** | Terminé | `automl/config.py`, `automl/data.py`, `03_etape1_...ipynb` | Industrialiser le chargement, la détection de tâche et le split adaptatif |
| **Phase 4 : Prétraitement Hybride (Étape 2)** | À faire | `automl/preprocessing.py` | Gérer les `NaN`, catégorielles (`data_A`) et haute dimension creuse (`data_F`) |
| **Phase 5 : Choix Auto des Modèles & HPO (Étape 3)** | À faire | `automl/models.py`, `automl/metrics.py` | Sélectionner la bonne famille de modèles scikit-learn + Optuna sous budget temps |
| **Phase 6 : Ensembling & Interface Finale (Étape 4)** | À faire | `automl/ensemble.py`, `automl/__init__.py` | Combiner les meilleurs modèles et exposer `fit`, `eval`, `predict` |

---

## Détail Chronologique des Réalisations

### Phase 0 — Mise en place de l'environnement & Conformité
- **Ce qu'on a fait :** Création du script `script/setup_autoML.bash` et du vérificateur `check_autoML.py`.
- **Pourquoi (Justification Soutenance) :**
  - Le challenge impose une contrainte stricte : **aucune librairie de Deep Learning ou de Gradient Boosting externe** (`torch`, `tensorflow`, `xgboost`, `lightgbm`, `catboost` sont interdits), et **aucun accès réseau**.
  - Nous avons automatisé la vérification dès le jour 1 pour éviter toute disqualification lors de l'évaluation sur le cluster Skinner.

---

### Phase 1 — Exploration des 6 jeux de données (`01_exploration_donnees.ipynb`)
- **Ce qu'on a fait :** Analyse systématique des fichiers `.data`, `.type` et `.solution` de `data_A` à `data_F`.
- **Ce qu'on a découvert :**
  1. **Hétérogénéité des formats :** `data_A` à `data_E` sont des matrices **denses**, tandis que `data_F` est une matrice **creuse (sparse)** au format `indice:valeur` avec **$61\,188$ variables** pour $13\,142$ lignes. Convertir `data_F` en tableau dense consommerait plus de $6\text{ Go}$ de RAM inutilement.
  2. **Hétérogénéité des types de colonnes :** Présence de variables `Categorical` et `Numerical` mélangées (`data_A`), ainsi que des variables `Binary` (`data_D`).
  3. **Hétérogénéité des cibles :** 4 types de tâches coexistent dans les 6 jeux : régression (`data_B`), classification binaire (`data_D`, `data_E`), multiclasse encodée en One-Hot (`data_C`, `data_F`), et multi-label (`data_A`).

---

### Phase 2 — Prototype Baseline V1 (`02_baseline_automl.ipynb`)
- **Ce qu'on a fait :** Première preuve de concept d'un aiguilleur automatique (`detecter_tache`) couplé à un `RandomForest` par défaut sur un échantillon de $1\,000$ lignes.
- **Scores de référence obtenus (à battre impérativement) :**
  - `data_D` (Binaire) : **78.0 % Accuracy**
  - `data_C` (Multiclasse 10 classes) : **89.0 % Accuracy**
  - `data_B` (Régression) : **$R^2$ = 0.6605**
- **Limites identifiées :**
  - Le code était encore dans un notebook (impossible à importer par le script du prof `import automl`).
  - Les méthodes de `automl/__init__.py` étaient des coquilles vides (*stubs*).
  - Pas de gestion des valeurs manquantes, pas de gestion des variables catégorielles, pas de respect des variables d'environnement (`AUTOML_TIME_BUDGET`, `AUTOML_N_JOBS`).

---

### Phase 3 (Étape 1) — Socle Ressources & Données (`automl/config.py`, `automl/data.py`)
**Date :** Étape actuelle — *Validé sur les 6 jeux dans `03_etape1_donnees_et_taches.ipynb`*

#### 1. Gestion des ressources et reproductibilité (`automl/config.py`)
- **Ce qu'on a fait :**
  - Lecture automatique de `AUTOML_TIME_BUDGET`, `AUTOML_N_JOBS` et `AUTOML_N_TRIALS` via `ConfigRessources.depuis_env()`.
  - Création d'un chronomètre `ChronoBudget` permettant de connaître à tout instant le temps écoulé et le temps restant.
  - Centralisation de la graine aléatoire (`fixer_graine(42)` : Python, NumPy, `PYTHONHASHSEED`).
- **Pourquoi (Argumentaire Soutenance) :**
  - Sur le cluster Skinner, utiliser `n_jobs=-1` est strictement interdit car cela saturerait les nœuds partagés. Notre parseur intercepte explicitement tout `n_jobs <= 0` et le remplace par `1`.
  - Le `ChronoBudget` garantit que notre pipeline ne dépassera jamais le budget temps imparti par le jury, peu importe la taille d'un jeu de test secret.

#### 2. Chargement universel & Contrats (`automl/data.py`)
- **Ce qu'on a fait :**
  - **Chargement Sparse robuste :** Détection automatique 0-based vs 1-based (`_charger_matrice_creuse`), stockage direct en `scipy.sparse.csr_matrix`, et filtrage des indices hors bornes qui pourraient apparaître dans des données de test inconnues.
  - **Chargement Dense résilient :** Gestion native des jetons manquants (`NaN`, `?`, `NA`), encodage reproductible si des chaînes apparaissent, et alignement automatique du nombre de colonnes au test (`_aligner_nb_colonnes`).
  - **Détection autonome de tâche (`charger_cible_et_detecter_tache`) :** Identifie mathématiquement la tâche parmi `binary`, `multiclass`, `multiclass_onehot`, `multilabel`, `regression`, `multioutput_regression` sans aucun nom de dataset codé en dur.
  - **Contrat de `predict` (`formater_predictions`) :** Garantit qu'un problème `multiclass_onehot` restitue une matrice binaire stricte dont chaque ligne somme exactement à `1`, et qu'une cible binaire/multiclasse retrouve ses étiquettes et types d'origine.

#### 3. Découpage Stratifié Adaptatif (`decouper_train_val_test` & `tailles_split_adaptatif`)
- **Ce qu'on a fait :** Remplacement du split fixe (ex: 70/15/15 ou 80/20) par un ratio dynamique selon la taille $N$ du dataset, avec une stratification intelligente adaptée à chaque tâche :
  - *Binaire / Multiclasse* : stratification directe sur la classe.
  - *Multi-label (`data_A`)* : stratification sur la combinaison binaire des labels (encodée en entier par décalage binaire) afin de préserver la distribution jointe des étiquettes.
  - *Régression (`data_B`)* : découpage de la cible continue en 10 quantiles (`pd.qcut`) pour garantir que train, val et test couvrent la même plage de valeurs extrêmes.
- **Pourquoi un split adaptatif ? (Argumentaire Soutenance) :**
  - Un pourcentage fixe de 15 % de validation pénalise les petits jeux (qui manquent d'exemples d'entraînement) et gaspille des données sur les grands jeux (car au-delà de quelques milliers d'exemples, la variance de l'estimateur de validation est déjà très faible).
  - En passant de $20\%$ de validation pour $N < 2\,000$ à $12\%$ (plafonné à $8\,000$ lignes) pour les grands datasets, on maximise la quantité d'informations disponibles pour l'apprentissage.

#### Résultats de validation de l'Étape 1 sur les 6 jeux réels :
| Jeu | Dimensions | Format | Tâche détectée automatiquement | Split Adaptatif (Train / Val / Test) |
| :--- | :---: | :---: | :--- | :---: |
| `data_A` | $34\,190 \times 24$ | Dense | `multilabel` | $27\,352$ / $4\,102$ / $2\,736$ (`val=12%`, `test=8%`) |
| `data_B` | $5\,000 \times 16$ | Dense | `regression` | $3\,750$ / $750$ / $500$ (`val=15%`, `test=10%`) |
| `data_C` | $15\,000 \times 1\,568$ | Dense | `multiclass_onehot` | $12\,000$ / $1\,800$ / $1\,200$ (`val=12%`, `test=8%`) |
| `data_D` | $2\,984 \times 144$ | Dense | `binary` | $2\,238$ / $447$ / $299$ (`val=15%`, `test=10%`) |
| `data_E` | $3\,140 \times 259$ | Dense | `binary` | $2\,355$ / $471$ / $314$ (`val=15%`, `test=10%`) |
| `data_F` | $13\,142 \times 61\,188$ | Creux CSR | `multiclass_onehot` | $10\,513$ / $1\,577$ / $1\,052$ (`val=12%`, `test=8%`) |

#### 4. Décisions d'Architecture & Justifications Détaillées (Pour la Soutenance)

##### A. Pourquoi `MetaFeatures` est indispensable entre `fit()` et `predict()` ?
Lors de l'évaluation automatique, il y a **deux appels séparés dans le temps** :
1. **Moment 1 (`automl.fit(train_dir)`)** : le code lit `train.data` et entraîne un modèle qui attend exactement $p$ colonnes (ex : $61\,188$ pour `data_F`).
2. **Moment 2 (`automl.predict(test_dir)`)** : le code lit un nouveau fichier `test.data` (sans `.solution`) pour reconstruire une matrice `X_test` et l'envoyer dans `model.predict(X_test)`.

**Sans `MetaFeatures`, deux pièges critiques casseraient le pipeline au Moment 2 :**
- **Piège 1 (Dimensions manquantes en format creux) :** Dans un fichier creux `indice:valeur`, les valeurs nulles ne sont pas écrites. Si le jeu de test secret contient peu de lignes et que l'indice maximal observé dans le fichier est `58 400` au lieu de `61 188`, un lecteur naïf construirait `X_test` avec seulement `58 400` colonnes, provoquant un crash immédiat (`ValueError: shape mismatch`).
- **Piège 2 (Incohérence d'encodage et d'indexation) :** Si une colonne dense contient du texte (`"Bleu" -> 0`, `"Rouge" -> 1`) et que le test ne contient que `"Rouge"`, un lecteur sans mémoire attribuerait `0` à `"Rouge"`, inversant la prédiction en silence. De même, si les premières colonnes d'un test creux sont nulles, l'offset (`0-based` vs `1-based`) pourrait être mal réinféré.

**Rôle de chaque attribut mémorisé dans `MetaFeatures` :**
- `n_features` : force `X_test` à avoir exactement le même nombre de colonnes que `X_train`.
- `sparse_index_offset` : garantit que les indices de colonnes du test s'alignent au même décalage (`0` ou `1`) que le train.
- `is_sparse` : évite de réinférer le format et assure un traitement homogène.
- `column_types` : conserve le typage `.type` (`Numerical`, `Categorical`, `Binary`) pour le prétraitement à venir.
- `string_encoders` : réapplique exactement le même dictionnaire de traduction texte $\rightarrow$ float découvert à l'entraînement.

##### B. Pourquoi une Stratification Avancée sur Multi-label et Régression ?
- **Rôle de la validation (`Val`) :** Dans notre AutoML, le sous-ensemble `Val` sert de **juge interne** pour sélectionner le meilleur modèle, ajuster les hyperparamètres et optimiser les seuils. Si `Val` n'est pas représentatif du jeu total, on sélectionne le mauvais modèle pour le test final.
- **Limite native de scikit-learn :** `train_test_split(..., stratify=y)` ne fonctionne nativement que pour une cible 1D de classes discrètes. Il plante si `y` est une matrice 2D (multi-label comme `data_A`) ou s'il s'agit de valeurs continues uniques (régression comme `data_B`).
- **Notre solution méthodologique (`construire_etiquettes_stratification`) :**
  - *Multi-label* : encodage de chaque combinaison de labels en un entier unique par décalage binaire (`(1 << arange)`), préservant la fréquence exacte des co-occurrences entre labels dans Train, Val et Test.
  - *Régression* : discrétisation temporaire de la cible continue en 10 quantiles (`pd.qcut`) pour garantir que les valeurs faibles, moyennes et extrêmes sont réparties équitablement.
  - *Sécurité (`try / except`)* : si un jeu secret possède des combinaisons trop rares ($< 3$ exemples), le système rebascule proprement sur un découpage non stratifié sans jamais planter.
- **Intérêt pour le jury :** Démontre une réflexion méthodologique au-delà des appels standards de `scikit-learn` et stabilise fortement les scores en validation et en test.

##### C. Généricité complète du contrat de sortie (`formater_predictions`)
Contrairement aux tests unitaires qui peuvent afficher des exemples sur un ou deux datasets (comme `data_C` pour illustrer le One-Hot et `data_D` pour le binaire), la fonction `formater_predictions(preds, target_meta)` est **100 % agnostique du nom du dataset**. Elle s'appuie uniquement sur `MetaCible` (inféré dynamiquement) pour restituer la forme exacte attendue par `.solution` : matrice One-Hot stricte (somme par ligne = 1), matrice binaire multi-label, vecteur 1D dans les types d'origine (`int64`, `float64`, `str`) ou matrice continue multi-sorties.

---

## Questions Probables du Jury & Réponses Clés (Antisèche Soutenance)

1. **« Comment votre système sait-il quel modèle utiliser sur un jeu de données secret dont il ne connaît rien ? »**  
   $\rightarrow$ *Il inspecte d'abord la structure de `.solution` (nombre de colonnes, valeurs discrètes vs continues, somme des lignes = 1) pour inférer la tâche mathématique exacte, puis regarde si $X$ est dense ou creux (`csr_matrix`), puis aiguille vers les familles scikit-learn compatibles.*
2. **« Que se passe-t-il si le jeu de test contient une colonne manquante ou un indice sparse jamais vu à l'entraînement ? »**  
   $\rightarrow$ *Nos métadonnées `MetaFeatures` mémorisent `n_features` et `sparse_index_offset` durant `fit()`. Au `predict()`, `_charger_matrice_creuse` filtre les indices hors bornes et `_aligner_nb_colonnes` ajuste les matrices denses sans jamais provoquer de crash.*
3. **« Comment garantissez-vous de ne pas dépasser le temps alloué sur Skinner ? »**  
   $\rightarrow$ *Grâce à `ConfigRessources` qui lit `AUTOML_TIME_BUDGET` et à `ChronoBudget` qui interrompt les boucles d'entraînement ou d'optimisation dès qu'on approche de la marge de sécurité.*
