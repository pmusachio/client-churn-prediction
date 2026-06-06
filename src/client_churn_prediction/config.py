"""Project configuration helpers."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return project_root() / "configs" / "project.toml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a TOML config file as a plain dictionary."""
    config_path = Path(path) if path else default_config_path()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root())
    return config


def section(config: dict, name: str, default: dict | None = None) -> dict:
    return config.get(name, default or {})


def resolve_project_path(config: dict, value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.get("_project_root", project_root())) / path
