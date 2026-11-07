"""Hydra configs, composed without changing the working directory (works in notebooks too)."""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def config_dir() -> Path:
    """``configs/`` shipped inside the installed package, or the repository's in a source tree."""
    packaged = Path(__file__).parent / "configs"
    return packaged if packaged.is_dir() else Path(__file__).parents[2] / "configs"


def load_config(name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose ``configs/{name}.yaml`` with Hydra overrides such as ``model=efficientnet_b3``."""
    with initialize_config_dir(config_dir=str(config_dir().resolve()), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))
