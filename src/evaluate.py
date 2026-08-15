from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained classification model.

    Args:
        model: Trained classification model.
        X_test: Test feature data.
        y_test: True test labels.

    Returns:
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    return metrics