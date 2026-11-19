# Étapes 1 et 2 : ce que Zakaria a renforcé

> Branche `feat/etape1-2-robustesse`, par **Zakaria**, 19/11/2026.
> Elle **améliore le travail déjà fusionné de Lionel** (`inspection.py`, `preprocessing.py`) et y
> **intègre ma branche `feat/etape1-data-config`**, qui n'avait jamais été fusionnée.

## En une phrase

Le jeu de test secret peut contenir des pièges (texte inattendu, `?`, colonnes en moins, indices
inconnus…). Cette branche fait en sorte que **rien de tout ça ne fasse planter `fit` ou `predict`**,
sans changer l'organisation du code de Lionel.

## Pourquoi ne pas avoir simplement fusionné ma branche ?

Ma branche `feat/etape1-data-config` partait d'un ancien `main`. Entre-temps, Lionel a écrit
`inspection.py`, qui détecte déjà la tâche. Fusionner tel quel aurait donné **deux détections de
tâche concurrentes** et des conflits. J'ai donc gardé la structure de `main` (noms anglais, comme le
reste du code) et j'y ai **porté uniquement ce qui manquait**. La branche `feat/etape1-data-config`
devient inutile une fois celle-ci fusionnée.

## Ce qui change, fichier par fichier

### `config.py` — ressources et reproductibilité

| Avant | Après | Pourquoi |
| :--- | :--- | :--- |
| Variables d'environnement lues **à l'import** | Lues **à chaque `fit`** (`ResourceConfig.from_env()`) | L'évaluateur peut fixer `AUTOML_TIME_BUDGET` après `import automl` : avant, la valeur était ignorée |
| Pas de chronomètre | `TimeBudget` (`elapsed`, `remaining`, `has_time`) | Indispensable pour l'étape 3 : arrêter d'entraîner avant de dépasser le budget |
| Seule `np.random` était fixée | `set_seed()` fixe aussi `random` et `PYTHONHASHSEED` | Reproductibilité exigée par le sujet |
| HistGradientBoosting utilisait **tous** les cœurs | Threads OpenMP/BLAS limités à `AUTOML_N_JOBS` | Ce modèle n'a pas de paramètre `n_jobs` : sans ça, on viole la règle des cœurs |

### `data.py` — chargement robuste

| Piège dans un fichier de test | Avant | Après |
| :--- | :--- | :--- |
| `?` comme valeur manquante | plantage | lu comme `NaN` |
| Texte dans une colonne (`rouge`, `bleu`) | plantage | codé en nombre ; **même code au test qu'au train** |
| Modalité jamais vue au train (`violet`) | — | `NaN` (le prétraitement l'impute) |
| Texte parasite dans une colonne numérique | — | seule la valeur fautive devient `NaN` |
| `inf` | plantage du `StandardScaler` | lu comme `NaN` |
| Colonne en moins / en trop au test | plantage | complétée par des `NaN` / tronquée |
| Indices creux commençant à 0 | décalés d'une colonne | décalage détecté au train et réutilisé |
| Indice creux hors bornes au test | plantage | ignoré |
| `.solution` en texte (`chat` / `chien`) | plantage | classification sur ces étiquettes |

Ce qui est appris au train (largeur, décalage creux, codage du texte) est rangé dans
`FeatureMeta` (`ds.meta`) et sera passé à `load_dataset(dossier_test, meta=...)` au moment du `predict`.

**Découpage adaptatif.** Avant : 70 / 15 / 15 pour tous les jeux. Après, la part de validation
dépend de la taille N :

| N | Validation | Test interne |
| :--- | :---: | :---: |
| < 2 000 | 20 % | 10 % |
| < 10 000 | 15 % | 10 % |
| < 50 000 | 12 % | 8 % |
| ≥ 50 000 | 8 % | 5 % |

Petit jeu : il faut assez de lignes de validation pour que le score soit fiable. Grand jeu :
quelques milliers suffisent, le reste sert à l'entraînement.

### `inspection.py` — tâche et stratification

- `encode_y` transforme les classes en indices `0..k-1` et `decode_y` fait l'inverse : on retrouve
  exactement les valeurs du `.solution` (y compris du texte). En one-hot, chaque ligne somme à 1.
- Nouvelle méthode `stratify_labels` : avant, seuls le binaire et le multiclasse étaient stratifiés.
  Maintenant aussi :
  - **multi-label** : chaque combinaison de labels devient un entier (`[1,0,1]` → 5) ;
  - **régression** : on stratifie sur le décile de la cible.

  Résultat : train, validation et test ont la même répartition, donc le score de validation est plus fiable.

### `preprocessing.py` — deux protections

- Dense : les `inf` sont transformés en `NaN` avant l'imputation.
- Creux (comptages) : une valeur négative au test est ramenée à 0 avant le TF-IDF (sinon `log` négatif → `NaN`).

## Comment vérifier

```bash
python -m pytest -q tests                      # 15 tests (cas pièges + conformité)
python script/generer_datasets_test.py         # les 20 jeux pièges de Lionel
python script/tester_pretraitement.py          # doit n'afficher aucun ❌
```

Le banc de Lionel a d'ailleurs attrapé une erreur pendant mon travail : j'avais ajouté une règle
« des entiers non consécutifs = régression », qui classait à tort `test_wine_labels` (étiquettes 3, 7, 12).
Je l'ai retirée.
