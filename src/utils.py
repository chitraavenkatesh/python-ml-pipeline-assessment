import os
import joblib


def save_model(model, file_path: str):
    """
    Save a trained machine learning model using joblib.

    Args:
        model: Trained machine learning model.
        file_path: Path where the model should be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)