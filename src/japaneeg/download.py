"""Download the Japan EEG dataset from OpenNeuro."""

from __future__ import annotations

import os
from pathlib import Path

import openneuro as on
from dotenv import load_dotenv


DATASET_ID = "ds007600"
DEFAULT_TARGET_DIR = Path("data") / DATASET_ID


def download(
    target_dir: str | Path = DEFAULT_TARGET_DIR,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Path:
    """Download the ds007600 dataset from OpenNeuro.

    Parameters
    ----------
    target_dir : str or Path
        Directory to save the dataset.
    include : list of str, optional
        Glob patterns of files to include.
    exclude : list of str, optional
        Glob patterns of files to exclude.

    Returns
    -------
    Path
        Path to the downloaded dataset.
    """
    load_dotenv()

    api_key = os.getenv("OPENNEURO_API_KEY")
    if api_key:
        _ensure_openneuro_config(api_key)

    target_dir = Path(target_dir)
    on.download(
        dataset=DATASET_ID,
        target_dir=str(target_dir),
        include=include or [],
        exclude=exclude or [],
    )
    return target_dir


def _ensure_openneuro_config(api_key: str) -> None:
    """Create the openneuro-py config file if it doesn't exist."""
    import json

    import platformdirs

    config_dir = Path(
        platformdirs.user_config_dir(
            appname="openneuro-py", appauthor=False, roaming=True
        )
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"

    if not config_path.exists():
        config = {"endpoint": "https://openneuro.org/", "apikey": api_key}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        config_path.chmod(0o600)


if __name__ == "__main__":
    download()
