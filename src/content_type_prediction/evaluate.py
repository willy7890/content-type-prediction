"""Evaluation utilities."""
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred):
    return {
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["Movie", "Tv Show"]).tolist(),
    }
