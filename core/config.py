"""Réglages de l'application, stockés dans config.json à côté du .exe."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    window_title_contains: str = "BlueStacks"   # partie du titre de la fenêtre à observer
    threshold: float = 0.80                     # confiance minimale (0.50 à 0.99)
    scan_interval_ms: int = 500                 # pause entre deux observations
    cooldown_after_click_s: float = 4.0         # pause après un clic (laisse le jeu réagir)
    click_count: int = 2                        # 1 = simple clic, 2 = double clic
    click_interval_ms: int = 200                # délai entre deux clics d'un double clic
    detect_combat: bool = True                  # phase 2 : attendre le combat après le clic
    combat_wait_s: int = 20                     # attente maximale du bouton « Prêt »
    ui_threshold: float = 0.85                  # confiance minimale pour les boutons du jeu
    do_placement: bool = True                   # phase 3 : se placer puis cliquer sur « Prêt »
    placement_rule: str = "closest"             # closest = plus proche des monstres, farthest, stay
    do_turns: bool = True                       # phases 4-5 : jouer les tours (sorts + fin du tour)
    casts_per_turn: int = 2                     # nombre de lancers de Coagulation par tour
    pause_between_casts_ms: int = 500           # pause supplémentaire entre deux lancers d'un même tour
    heal_enabled: bool = True                   # avant chaque combat : vérifier la vie et boire des potions si besoin
    min_hp_percent: int = 70                    # vie minimale (%) pour lancer un combat
    heal_max_potions: int = 15                  # nombre maximum de potions bues d'affilée
    opening_spell: bool = True                  # 1er tour du combat uniquement : lancer le sort d'ouverture sur soi, avant Coagulation
    second_spell: bool = True                   # après les Coagulations réussies : lancer le Pied du Sacrieur
    move_when_out_of_range: bool = True         # personne à portée : se rapprocher du monstre le plus proche
    max_move_cells: int = 4                     # nombre maximum de cases parcourues pour se rapprocher
    turn_wait_s: int = 120                      # attente maximale de mon tour
    turn_start_delay_s: float = 2.5             # attente après le début du tour (bandeau « À toi de jouer »)
    range_min_fraction: float = 0.05            # part de violet autour d'un monstre pour le dire « à portée »
    loop_combats: bool = True                   # phase 6 : enchaîner les combats sans fin (False = un seul combat)
    sequence_seconds: int = 40                  # durée de l'enregistrement de séquence
    save_diagnostics: bool = True               # enregistre une image annotée à chaque clic


def load_config(path: Path) -> Config:
    cfg = Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
    except (FileNotFoundError, ValueError):
        pass
    return cfg


def save_config(cfg: Config, path: Path) -> None:
    path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
