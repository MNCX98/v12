"""Lecture de la vie du personnage hors combat.

Le cœur du HUD se vide par le haut, avec une coupure horizontale nette : la hauteur de rouge donne le
pourcentage de vie. Les positions sont relatives à l'icône repère « œil » (assets/ui/hud_oeil.png),
à l'échelle d'origine des captures du programme."""
import numpy as np

HEART_X0, HEART_X1 = -92, -30     # colonnes examinées (centre du cœur, hors contour)
HEART_Y0 = -22                    # première ligne examinée (au-dessus du cœur)
HEART_BOTTOM = 49                 # ligne située juste sous la pointe du cœur
HEART_HEIGHT = 64                 # hauteur du cœur plein, en pixels


def read_hp_percent(scene, eye_x: int, eye_y: int) -> float | None:
    """Pourcentage de vie (0-100) d'après la hauteur de rouge du cœur, ou None si la zone sort de l'image."""
    h, w = scene.shape[:2]
    x0, x1 = eye_x + HEART_X0, eye_x + HEART_X1
    y0, y1 = eye_y + HEART_Y0, eye_y + HEART_BOTTOM
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    roi = scene[y0:y1, x0:x1].astype(int)
    b, g, r = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    red = (r > 130) & (r > g + 60) & (r > b + 60)
    rows = red.sum(axis=1) >= 3                         # lignes contenant du rouge
    for i in range(len(rows) - 1):
        if rows[i] and rows[i + 1]:                     # 2 lignes de suite : évite un pixel isolé
            return float(max(0, min(100, 100 * (y1 - (y0 + i)) / HEART_HEIGHT)))
    return 0.0
