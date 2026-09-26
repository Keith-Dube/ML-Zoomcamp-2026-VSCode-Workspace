from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, *, required_columns: set[str] | None = None) -> pd.DataFrame:
    """Load a CSV and fail early if required columns are missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

    frame = pd.read_csv(csv_path)
    missing = (required_columns or set()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame

