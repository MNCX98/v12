"""Vision du combat : anneaux sous les combattants, portée du sort (violet), distances."""
from dataclasses import dataclass

import cv2
import numpy as np

from core.grid import grid_distance


@dataclass
class Fighter:
    cx: int
    cy: int
    pixels: int
    w: int = 0
    h: int = 0

    @property
    def center(self) -> tuple[int, int]:
        return self.cx, self.cy


def _game_roi(h: int, w: int) -> np.ndarray:
    roi = np.zeros((h, w), bool)
    roi[int(0.05 * h):int(0.955 * h), int(0.17 * w):int(0.83 * w)] = True
    roi[:int(0.27 * h), :int(0.48 * w)] = False        # barre d'ordre de jeu (barres bleues/rouges des portraits)
    roi[:int(0.5 * h), :int(0.215 * w)] = False        # colonne d'icônes orange/rouges à gauche
    return roi


def _hsv(scene):
    return cv2.split(cv2.cvtColor(scene, cv2.COLOR_BGR2HSV))


def _rings(mask: np.ndarray, min_pixels: int = 35, merge=(31, 15)) -> list[tuple[Fighter, float]]:
    """Regroupe les morceaux d'anneau voisins ; retourne (combattant, taux de remplissage)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, merge)
    joined = cv2.dilate(mask, k)
    n, lab, _, _ = cv2.connectedComponentsWithStats(joined)
    out = []
    for i in range(1, n):
        ys, xs = np.where((lab == i) & (mask > 0))
        if len(xs) < min_pixels:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        fill = len(xs) / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        out.append((Fighter(int((x0 + x1) // 2), int((y0 + y1) // 2), len(xs),
                            int(x1 - x0 + 1), int(y1 - y0 + 1)), fill))
    return out



# Anneau sous un combattant : ellipse d'environ 47×22 px (identique en mode normal et en mode créature).
RING_W, RING_H = 47, 22


def _ring_template() -> np.ndarray:
    t = np.zeros((RING_H + 2, RING_W + 2), np.float32)
    cv2.ellipse(t, ((RING_W + 2) // 2, (RING_H + 2) // 2), (RING_W // 2, RING_H // 2), 0, 0, 360, 1.0, 1)
    return t


_RING = _ring_template()


def _find_rings(mask: np.ndarray, min_cover: float = 0.55, max_fill: float = 2.6) -> list[Fighter]:
    """Cherche des anneaux par leur forme : sépare les combattants sur des cases voisines.

    min_cover : part de l'ellipse retrouvée dans le masque de couleur (un anneau à demi caché reste accepté).
    max_fill  : écarte les formes pleines (cases, icônes, chiffres de dégâts), qui ne sont pas des anneaux fins.
    """
    th, tw = _RING.shape
    if mask.shape[0] < th or mask.shape[1] < tw:
        return []
    m = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    n_ring = float(_RING.sum())
    cover = cv2.matchTemplate(m, _RING, cv2.TM_CCORR) / n_ring
    raw = (mask > 0).astype(np.float32)
    area = cv2.boxFilter(raw, -1, (tw, th), normalize=False, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where(cover >= min_cover)
    order = np.argsort(-cover[ys, xs])
    found: list[Fighter] = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        cx, cy = x + tw // 2, y + th // 2
        if any(abs(cx - f.cx) < 20 and abs(cy - f.cy) < 10 for f in found):
            continue
        inside = float(area[y + th // 2, x + tw // 2])
        if inside > max_fill * n_ring:
            continue
        found.append(Fighter(cx, cy, int(inside), tw, th))
        if len(found) >= 24:
            break
    return found


def find_enemies(scene: np.ndarray) -> list[Fighter]:
    """Adversaires : anneau bleu sous chaque monstre.

    Méthode validée : on regroupe les morceaux d'anneau. Si un groupe est trop grand pour être un seul
    anneau (monstres sur des cases voisines), on le découpe en cherchant la forme de l'anneau.
    """
    h, w = scene.shape[:2]
    hh, ss, vv = _hsv(scene)
    m = (((hh > 98) & (hh < 112) & (ss > 140) & (vv > 140)) & _game_roi(h, w)).astype(np.uint8) * 255
    out: list[Fighter] = []
    for f, _ in _rings(m):
        if f.w <= RING_W + 10 and f.h <= RING_H + 8:
            out.append(f)                                   # un seul anneau
            continue
        x0, y0 = max(0, f.cx - f.w // 2 - 6), max(0, f.cy - f.h // 2 - 6)
        x1, y1 = min(w, f.cx + f.w // 2 + 7), min(h, f.cy + f.h // 2 + 7)
        parts = _find_rings(m[y0:y1, x0:x1])
        if parts:
            for p in parts:
                out.append(Fighter(p.cx + x0, p.cy + y0, p.pixels, p.w, p.h))
        else:
            out.append(f)                                   # découpage impossible : on garde le groupe
    return out


def find_me(scene: np.ndarray, hint: tuple[int, int] | None = None) -> Fighter | None:
    """Mon personnage : anneau rouge (ellipse d'environ 48×25 px, en mode normal comme en mode créature).

    On écarte les chiffres de dégâts rouges (plus gros), les cases rouges et les icônes rouges.
    `hint` : position connue (mon personnage ne bouge pas pendant le combat) pour départager.
    """
    h, w = scene.shape[:2]
    hh, ss, vv = _hsv(scene)
    m = ((((hh < 8) | (hh > 172)) & (ss > 120) & (vv > 120)) & _game_roi(h, w)).astype(np.uint8) * 255
    cands = [f for f, fill in _rings(m, min_pixels=120)
             if 120 <= f.pixels <= 650 and 40 <= f.w <= 60 and 15 <= f.h <= 40 and f.w / f.h >= 1.3 and fill < 0.6]
    if not cands:
        return None
    if hint is not None:
        near = [f for f in cands if abs(f.cx - hint[0]) <= 20 and abs(f.cy - hint[1]) <= 20]
        if near:
            return near[0]
    return max(cands, key=lambda f: f.pixels)


def range_mask(scene: np.ndarray) -> np.ndarray:
    """Cases où le sort sélectionné peut être lancé (violet affiché par le jeu)."""
    h, w = scene.shape[:2]
    hh, ss, vv = _hsv(scene)
    m = ((hh > 118) & (hh < 127) & (ss > 110) & (vv > 150) & _game_roi(h, w)).astype(np.uint8)
    return m


def range_visible(scene: np.ndarray, min_pixels: int = 3000) -> bool:
    return int(range_mask(scene).sum()) >= min_pixels


def range_fraction(mask: np.ndarray, center: tuple[int, int], half=(34, 15)) -> float:
    """Part de violet autour d'un point (case d'un adversaire)."""
    x, y = center
    h, w = mask.shape
    patch = mask[max(0, y - half[1]):min(h, y + half[1]), max(0, x - half[0]):min(w, x + half[0])]
    return float(patch.mean()) if patch.size else 0.0


def sort_by_distance(fighters: list[Fighter], origin: tuple[int, int]) -> list[Fighter]:
    return sorted(fighters, key=lambda f: grid_distance(f.center, origin))
