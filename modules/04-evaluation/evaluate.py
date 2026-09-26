"""Reusable evaluation entry point for Module 4."""

from sklearn.metrics import classification_report


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)

