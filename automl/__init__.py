import numpy as np

class AutoML:
    """Baseline minimale AutoML."""

    def __init__(self):
        self.model = None

    def fit(self, data_dest: str) -> "AutoML":
        print(f"[AutoML] Entraînement sur {data_dest}...")
        return self

    def eval(self) -> dict:
        print("[AutoML] Évaluation...")
        return {}

    def predict(self, test_dest: str) -> np.ndarray:
        print(f"[AutoML] Prédiction sur {test_dest}...")
        return np.array([])

# Instance singleton pour l'interface module (automl.fit, automl.eval, automl.predict)
_default_instance = AutoML()

def fit(data_dest: str) -> AutoML:
    return _default_instance.fit(data_dest)

def eval() -> dict:
    return _default_instance.eval()

def predict(test_dest: str) -> np.ndarray:
    return _default_instance.predict(test_dest)

__all__ = ["AutoML", "fit", "eval", "predict"]
