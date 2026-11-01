"""Contexte partagé par toutes les phases."""
from dataclasses import dataclass, field
from typing import Callable

from core.capture import WindowCapture
from core.config import Config
from core.vision import Template


@dataclass
class Context:
    config: Config
    capture: WindowCapture
    templates: list[Template] = field(default_factory=list)      # monstres
    ui_templates: list[Template] = field(default_factory=list)   # boutons / icônes du jeu
    log: Callable[[str], None] = print
    me_hint: tuple[int, int] | None = None                       # dernière position connue de mon personnage
