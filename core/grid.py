"""Cases de la grille isométrique (détection par couleur) et distances en cases."""
from dataclasses import dataclass

import cv2
import numpy as np

# Pas entre deux cases voisines (en pixels), mesuré pour BlueStacks en 1920×1020.
PITCH_X = 44.0
PITCH_Y = 22.0


@dataclass
class Cell:
    cx: int
    cy: int
    area: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return self.cx, self.cy


def grid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance en cases (déplacement sur la grille, comme dans Dofus)."""
    gx = abs(a[0] - b[0]) / PITCH_X
    gy = abs(a[1] - b[1]) / PITCH_Y
    return max(gx, gy)


def find_cells(scene: np.ndarray, color: str, min_area: int = 400, merged: list | None = None) -> list[Cell]:
    """Cases de placement visibles : 'red' = les miennes, 'blue' = celles des monstres.

    `merged` (facultatif) reçoit les grandes taches de couleur qui ne sont pas une case seule
    (par exemple une case de monstre entourée de ses flèches de sélection, collée à sa voisine).
    """
    h, w = scene.shape[:2]
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    if color == "red":
        m = ((hh < 8) | (hh > 172)) & (ss > 150) & (vv > 150)
    elif color == "blue":
        m = (hh > 98) & (hh < 108) & (ss > 150) & (vv > 150)
    else:
        raise ValueError(color)
    roi = np.zeros((h, w), bool)                      # zone de jeu : on exclut barre du haut et menus
    roi[int(0.28 * h):int(0.955 * h), int(0.17 * w):int(0.83 * w)] = True
    mask = (m & roi).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    cells = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area or bh == 0:
            continue
        ratio = bw / bh
        if not (1.4 <= ratio <= 3.0) or bw > 80 or bh > 42:      # forme de losange, taille d'une case
            if merged is not None and area >= 1200 and bw > 80:
                merged.append(Cell(int(cents[i][0]), int(cents[i][1]), area, bw, bh))
            continue
        cells.append(Cell(int(cents[i][0]), int(cents[i][1]), area, bw, bh))
    return cells


# --- Coordonnées de grille (déplacements en combat) -------------------------------------------
# Une case voisine est décalée de (±44, ±22) px à l'écran. On repère une case par (u, v) par
# rapport à une case de référence : u avance de (+44, +22), v avance de (+44, -22).
# Le nombre de cases à parcourir entre deux cases vaut |u| + |v|.

def to_grid(p: tuple[float, float], ref: tuple[float, float]) -> tuple[int, int]:
    a = (p[0] - ref[0]) / PITCH_X
    b = (p[1] - ref[1]) / PITCH_Y
    return round((a + b) / 2), round((a - b) / 2)


def to_screen(uv: tuple[int, int], ref: tuple[float, float]) -> tuple[int, int]:
    u, v = uv
    return int(round(ref[0] + (u + v) * PITCH_X)), int(round(ref[1] + (u - v) * PITCH_Y))


def move_candidates(me: tuple[int, int], target: tuple[int, int], occupied: list[tuple[int, int]],
                    max_cells: int, shorter: bool = True, top: int = 2) -> list[tuple[int, tuple[int, int]]]:
    """Cases où aller pour se rapprocher de `target`, les meilleures d'abord.

    Retourne [(nombre de cases parcourues, (x, y) écran)]. On ne va jamais sur une case occupée
    et on s'arrête au contact du monstre. Pour chaque longueur (max_cells, puis moins), la case la
    plus proche du monstre, en suivant au mieux la ligne droite vers lui.
    """
    tu, tv = to_grid(target, me)
    dist = abs(tu) + abs(tv)
    busy = {to_grid(o, me) for o in occupied}
    out = []
    start = min(max_cells, dist - 1)
    stop = 0 if shorter else max(start - 1, 0)       # shorter=False : uniquement des déplacements de `start` cases
    for k in range(start, stop, -1):
        cells = []
        for u in range(-k, k + 1):
            for v in {k - abs(u), -(k - abs(u))}:
                if (u, v) in busy:
                    continue
                left = abs(tu - u) + abs(tv - v)
                if not shorter and left >= dist:         # case qui n'approche pas du monstre : inutile de l'essayer
                    continue
                straight = abs(u * tv - v * tu)          # écart à la ligne droite vers le monstre
                cells.append((left, straight, (u, v)))
        cells.sort()
        out.extend((k, to_screen(c[2], me)) for c in cells[:top])
    return out
