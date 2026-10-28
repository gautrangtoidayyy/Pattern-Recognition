from pathlib import Path

import numpy as np

from pattern_recognition.config import DATA_PATH


def load_processed_mnist(path: str | Path = DATA_PATH) -> dict[str, np.ndarray]:
    """Load the arrays produced by the preprocessing notebook."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed MNIST file not found: {data_path}. "
            "Run notebook/preprocessing.ipynb first."
        )

    with np.load(data_path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
