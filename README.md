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

## Utilisation

```python
import automl

automl.fit("/path/to/data")
automl.eval()
preds = automl.predict("/path/to/test")
```

## Données

Sur le cluster Skinner : `/info/corpus/ChallengeMachineLearning`
