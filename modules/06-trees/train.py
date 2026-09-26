"""Module 6 tree and ensemble experiments."""

from sklearn.ensemble import RandomForestClassifier


def build_baseline(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

