import sys
import importlib

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotext",
    "visidata",
    "ipykernel"
]

FORBIDDEN_PACKAGES = [
    "torch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "catboost",
    "requests",
    "urllib3"
]

def check_env():
    print("=" * 55)
    print("   VÉRIFICATION DES DÉPENDANCES AUTOM L")
    print("=" * 55)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}\n")

    print("[1] Vérification des packages AUTORISÉS :")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "version inconnue")
            print(f"  [✓] {pkg:<15} : {version}")
        except ImportError:
            print(f"  [✗] {pkg:<15} : NON INSTALLÉ")
            all_ok = False

    print("\n[2] Vérification des packages INTERDITS :")
    forbidden_found = False
    for pkg in FORBIDDEN_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            print(f"  [!] ALERTE : {pkg} est DÉTECTÉ !")
            forbidden_found = True
        except ImportError:
            print(f"  [✓] {pkg:<15} : Non présent (Conforme)")

    print("\n" + "=" * 55)
    if all_ok and not forbidden_found:
        print("  RESULTAT : Environnement parfaitement CONFORME !")
    else:
        print("  RESULTAT : Des problèmes ont été détectés.")
    print("=" * 55)

if __name__ == "__main__":
    check_env()
