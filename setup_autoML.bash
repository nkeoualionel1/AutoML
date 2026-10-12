#!/usr/bin/env bash
set -e

ENV_NAME="automl_env"

echo "=== 1. Création du venv ($ENV_NAME) ==="
python3 -m venv $ENV_NAME

echo "=== 2. Activation du venv ==="
source ./$ENV_NAME/bin/activate

echo "=== 3. Mise à jour de pip ==="
pip install --upgrade pip

echo "=== 4. Installation des packages autorisés ==="
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotext \
    visidata \
    ipykernel

echo "=== 5. Vérification de l'absence des bibliothèques interdites ==="
python -c "
import sys

# Frameworks lourds et bibliothèques d'accès réseau tierces interdites
forbidden_packages = ['torch', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'catboost', 'requests', 'urllib3']
found = []

for pkg in forbidden_packages:
    try:
        __import__(pkg)
        found.append(pkg)
    except ImportError:
        pass

if found:
    print('❌ ERREUR : Packages interdits détectés :', found)
    sys.exit(1)
else:
    print('✅ CONFORME : Aucun package tiers interdit détecté.')
"

echo "=== 6. Enregistrement du kernel pour Jupyter ==="
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo ""
echo "=== Initialisation terminée avec succès ! ==="
echo "Pour activer l'environnement dans votre terminal principal, lancez :"
echo "source ./$ENV_NAME/bin/activate"
