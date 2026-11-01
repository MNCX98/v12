"""Chemins de l'application (identiques en développement et dans le .exe)."""
import sys
from pathlib import Path


def app_dir() -> Path:
    """Dossier contenant l'application (à côté du .exe une fois compilée)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def monsters_dir() -> Path:
    return app_dir() / "assets" / "monsters"


def ui_dir() -> Path:
    return app_dir() / "assets" / "ui"


def logs_dir() -> Path:
    d = app_dir() / "logs"
    d.mkdir(exist_ok=True)
    return d


def captures_dir() -> Path:
    d = app_dir() / "captures"
    d.mkdir(exist_ok=True)
    return d


def config_path() -> Path:
    return app_dir() / "config.json"
