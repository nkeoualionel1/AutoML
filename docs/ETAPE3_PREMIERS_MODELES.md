# Étape 3 : premiers modèles — `fit`, `eval` et `predict` fonctionnent

> Branche `feat/etape3-premiers-modeles`, par **Zakaria**, 19/11/2026.
> Elle part de `feat/etape1-2-robustesse` : il faut fusionner cette dernière d'abord.

## En une phrase

Avant, `fit` s'arrêtait après le prétraitement, `eval()` renvoyait `{}` et `predict()` un tableau
vide : **le contrat du sujet n'était pas rempli**. Maintenant, le pipeline entraîne plusieurs
modèles, garde le meilleur et prédit au format exact du `.solution`, sur les 6 jeux.

## Ce qui se passe dans `fit`

```
chargement → détection de tâche → découpage → prétraitement   (étapes 1-2)
     → pour chaque modèle candidat (du moins cher au plus cher) :
           s'il reste assez de temps : entraîner sur train, noter sur valid
     → garder le meilleur score de validation
     → mesurer ses performances sur le test interne (jamais utilisé pour choisir)
     → le réentraîner sur train + valid (plus de données), si le temps le permet
```

## Les nouveaux fichiers

### `automl/models.py` — quels modèles pour quelle tâche

| Données | Tâche | Candidats (dans l'ordre d'essai) |
| :--- | :--- | :--- |
| Dense | classification | régression logistique → forêt aléatoire → HistGradientBoosting |
| Dense | régression | Ridge → forêt aléatoire → HistGradientBoosting |
| Creuses (texte) | classification | LinearSVC → régression logistique → Complement Naive Bayes* |
| Creuses | régression | Ridge |

\* seulement si les données sont des comptages positifs (Naive Bayes l'exige).

- **Multi-label / multi-sorties** : les modèles qui ne gèrent qu'une sortie sont emballés dans
  `MultiOutputClassifier` / `MultiOutputRegressor` (un modèle par label). La forêt aléatoire et Ridge
  gèrent nativement plusieurs sorties.
- **Pourquoi du moins cher au plus cher ?** Le premier modèle est toujours entraîné, quel que soit le
  budget. Il doit donc être rapide. Ensuite, avant chaque modèle, on estime son coût à **4× le plus
  lent déjà vu** et on le saute s'il ne tient pas dans le temps restant.
  Exemple réel sur `data_C` : HistGradientBoosting met 51 s à s'entraîner ; la forêt aléatoire fait
  le même score (0,9514 contre 0,9515) en 12 s. Avec un petit budget, le boosting est donc sauté
  sans perte de score ; avec un grand budget, il est essayé normalement.

### `automl/metrics.py` — comment on note un modèle

| Tâche | Score principal (sert à choisir) | Pourquoi |
| :--- | :--- | :--- |
| Binaire / multiclasse | moyenne de l'accuracy et du F1 macro | l'accuracy seule favorise la classe majoritaire |
| Multi-label | moyenne du F1 macro et de l'accuracy par label | tient compte de chaque label |
| Régression | R² | indépendant de l'échelle de la cible |

`eval()` affiche aussi les métriques détaillées, préfixées `valid_` (validation) et `test_`
(test interne, jamais utilisé pour choisir → estimation honnête).

### `predict(dossier)`

Le dossier de test est relu **avec les informations du train** (`FeatureMeta` : même largeur, même
codage du texte, même décalage creux), passe dans le même préprocesseur, puis `decode_y` remet les
prédictions au format du `.solution` (étiquettes d'origine, one-hot qui somme à 1, etc.).

## Résultats (mesures de test, 4 cœurs)

Ces mesures ont été faites avec un petit budget pour tester vite. Le budget réel est fixé par
l'évaluateur via `AUTOML_TIME_BUDGET` (600 s par défaut si la variable est absente) ; le code s'y adapte.

| Jeu | Tâche | Modèle retenu | Score valid | Score test interne | Durée `fit` |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `data_A` | multi-label | HistGradientBoosting | 0,888 | 0,888 | 19 s |
| `data_B` | régression | HistGradientBoosting | 0,799 | 0,816 | 2 s |
| `data_C` | multiclasse one-hot | forêt aléatoire | 0,951 | 0,944 | 32 s |
| `data_D` | binaire | forêt aléatoire | 0,804 | 0,797 | 1 s |
| `data_E` | binaire | HistGradientBoosting | 0,819 | 0,860 | 2 s |
| `data_F` | multiclasse one-hot (creux) | LinearSVC | 0,899 | 0,892 | 9 s |

Pour rappel, la baseline du notebook 02 faisait 78 % d'accuracy sur `data_D`, 89 % sur `data_C`
et R² = 0,66 sur `data_B`.

## Comment vérifier

```bash
python -m pytest -q tests     # 23 tests, dont 8 sur le contrat fit / eval / predict
```

```python
import automl
automl.fit("dataset/data_D")
automl.eval()
preds = automl.predict("dataset/data_D")   # forme (2984,), valeurs 0 / 1
```

## Ce qu'il reste à faire (à se répartir)

Ces étapes sont indépendantes ; chacune peut faire l'objet d'une branche :

1. **Optimisation des hyperparamètres (Optuna)** — `feat/etape3-hpo`.
   Les modèles utilisent leurs réglages par défaut. Il faut un espace de recherche par modèle,
   limité par `AUTOML_N_TRIALS` et par le chronomètre `self.timer` (`has_time`).
2. **Ensemble des meilleurs modèles** — `feat/etape4-ensemble`.
   Combiner les prédictions des meilleurs modèles plutôt que n'en garder qu'un (moyenne pondérée
   choisie sur la validation), et optimiser les seuils de décision en binaire / multi-label.
3. **Exploiter le profil des données** — `feat/cible-asymetrique`.
   `profile.target_skewed` est calculé mais pas utilisé : pour une régression à cible positive et
   asymétrique, essayer un modèle sur `log(1 + y)`.
4. **Une représentation adaptée aux arbres** — `feat/pretraitement-arbres`.
   Aujourd'hui tous les modèles reçoivent les données standardisées + one-hot. Les arbres n'en ont
   pas besoin et seraient plus rapides avec les catégories codées en entiers.
