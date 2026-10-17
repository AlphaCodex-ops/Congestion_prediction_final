from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and report configuration errors explicitly."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return loaded
