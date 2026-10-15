# Crafted AutoML Challenge

Ce dépôt contient le code source et la documentation pour le projet **Crafted AutoML Challenge**, réalisé dans le cadre des modules *Méthodologie pour l'Intelligence Artificielle* et *Méthodes Classiques d'IA*.

L'objectif principal est de concevoir et d'implémenter une pipeline d'apprentissage automatique entièrement automatisée (AutoML) capable d'analyser, de prétraiter, d'optimiser et d'évaluer des modèles sur différents jeux de données tabulaires.

---

## 📐 Architecture Générale de la Pipeline

La pipeline AutoML suit les principes méthodologiques et théoriques vus en cours :

```text
[Données Brutes] 
       │
       ▼
 1. Inspection & Détection
       │
       ▼
 2. Prétraitement & Feature Engineering
       │
       ▼
 3. Sélection de Modèles & Hyperparamétrage (HPO)
       │
       ▼
 4. Combinaison de Modèles (Ensembling)
       │
       ▼
 5. Évaluation Statistique & Validation
       │
       ▼
 [Prédictions & Export]
```

---

## 📚 Lien avec les Enseignements du Cours

### 1. Inspection & Analyse Exploratoire des Données (EDA)
- **Tâche & Cible :** Détection automatique de la nature du problème (classification binaire, multiclasse ou régression).
- **Typologie des Variables :** Identification des colonnes numériques, catégorielles et des données manquantes.
- **Contrôle de Qualité :** Analyse des distributions et détection du déséquilibre de classes (*class imbalance*).

### 2. Prétraitement & Ingénierie des Données (*Data Preprocessing*)
> *"80 % de la réussite d'un projet ML repose sur la qualité et la préparation des données"*.
- **Nettoyage & Imputation :** Traitement des valeurs manquantes (imputation simple/avancée) sans fuite de données (*data leakage*).
- **Encodage & Scaling :** Transformation des variables qualitatives (ex. *One-Hot Encoding*) et mise à l'échelle des descripteurs numériques (*StandardScaler*, *RobustScaler*).
- **Feature Engineering & Selection :** Construction de nouveaux descripteurs et réduction de dimensionnalité (méthodes *Filter*, *Wrapper*, *Embedded* comme Lasso/PCA).

### 3. Sélection de Modèles & Optimisation des Hyperparamètres (HPO)
- **Algorithmes Classiques :** Exploration de modèles de référence et avancés (Arbres de décision, Random Forest, Gradient Boosting, SVM, Linear/Logistic Regression).
- **Recherche d'Hyperparamètres :** Automatisation du réglage des paramètres (*GridSearch*, *RandomSearch*, ou optimisation bayésienne avec *Optuna*) pour équilibrer le compromis biais-variance.

### 4. Combinaison de Modèles (*Ensembling*)
- Agrégation de plusieurs modèles performants (*Blending* / *Stacking*) pour améliorer la généralisation et maximiser les performances globales.

### 5. Évaluation & Validation Statistique Rigoureuse
- **Protocole de Validation :** Utilisation de la validation croisée (*Stratified K-Fold*) pour garantir une évaluation robuste sans surapprentissage (*overfitting*).
- **Métriques Adaptées :**
  - *Classification :* Accuracy, F1-Score (macro/weighted), ROC-AUC.
  - *Régression :* MSE, RMSE, MAE, $R^2$.

---

## 🛠️ Interface Recommandée

Le code s'appuie sur une interface unifiée respectant les contraintes du challenge :

```python
class AutoMLPipeline:
    def fit(self, X_train, y_train):
        """Inspecte les données, prétraite, sélectionne et optimise le meilleur modèle."""
        pass

    def predict(self, X_test):
        """Génère les prédictions sur le jeu de test."""
        pass

    def evaluate(self, X_val, y_val):
        """Calcule et affiche les métriques d'évaluation sur le jeu de validation."""
        pass
```

---

## 📅 Livrables du Projet

1. **Code Source AutoML :** Pipeline fonctionnelle et automatisée.
2. **Crafted AutoML Poster :** Synthèse visuelle de la démarche méthodologique et des résultats.
3. **Soutenance Orale :** Présentation individuelle de 10 minutes.