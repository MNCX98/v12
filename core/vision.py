"""Reconnaissance d'images (OpenCV, template matching avec masque).

Chaque sous-dossier de assets/monsters = un monstre.
Chaque PNG du dossier = une variante (orientation).
Si le PNG a de la transparence, elle sert de masque : seuls les pixels opaques
(le monstre) sont comparés, le décor autour est ignoré.
"""
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Template:
    monster: str
    variant: str
    image: np.ndarray            # BGR uint8
    mask: np.ndarray | None      # 3 canaux float32 0/1, ou None (rectangle complet)
    small_gray: np.ndarray | None = None    # demi-résolution, niveaux de gris (balayage rapide)
    small_mask: np.ndarray | None = None    # masque 2D correspondant (ou None)


@dataclass
class Detection:
    monster: str
    variant: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass
class ScanResult:
    detections: list[Detection] = field(default_factory=list)   # triées par score décroissant
    best_scores: dict[str, float] = field(default_factory=dict)  # meilleur score par variante


def _read_image(path: Path) -> np.ndarray | None:
    # np.fromfile + imdecode : fonctionne même avec des accents dans le chemin (Windows)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except OSError:
        return None


def _make_template(path: Path, monster: str, variant: str) -> Template | None:
    img = _read_image(path)
    if img is None or img.size == 0:
        return None
    mask = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3].copy()
        if alpha.min() < 255:
            m = (alpha > 127).astype(np.float32)
            mask = cv2.merge([m, m, m])
    img = np.ascontiguousarray(img)
    t = Template(monster, variant, img, mask)
    h, w = img.shape[:2]
    if min(h, w) >= 24:     # assez grande pour être réduite sans perdre sa forme
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        t.small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            sm = cv2.resize(mask[:, :, 0], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            t.small_mask = (sm > 0.5).astype(np.float32)
    return t


def load_templates(root: Path) -> list[Template]:
    """Monstres : un sous-dossier par monstre, un PNG par variante."""
    templates: list[Template] = []
    if not root.is_dir():
        return templates
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(folder.glob("*.png")):
            t = _make_template(f, folder.name, f.stem)
            if t:
                templates.append(t)
    return templates


def load_ui_templates(folder: Path) -> list[Template]:
    """Éléments d'interface du jeu (boutons, icônes) : un PNG = un élément."""
    out: list[Template] = []
    if folder.is_dir():
        for f in sorted(folder.glob("*.png")):
            t = _make_template(f, "ui", f.stem)
            if t:
                out.append(t)
    return out


def _match(scene, image, mask) -> np.ndarray:
    if mask is None:
        res = cv2.matchTemplate(scene, image, cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(scene, image, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _peaks(res: np.ndarray, thr: float, min_dist: int, limit: int = 30) -> list[tuple[int, int, float]]:
    ys, xs = np.where(res >= thr)
    if len(xs) == 0:
        return []
    scores = res[ys, xs]
    order = np.argsort(-scores)[:2000]
    peaks: list[tuple[int, int, float]] = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - px) >= min_dist or abs(y - py) >= min_dist for px, py, _ in peaks):
            peaks.append((x, y, float(scores[i])))
            if len(peaks) >= limit:
                break
    return peaks


_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1)))
    return _POOL


def _scan_template(t: Template, scene, small_gray_scene, threshold):
    """Retourne (clé, meilleur score, détections) pour une image de référence."""
    sh, sw = scene.shape[:2]
    th, tw = t.image.shape[:2]
    key = f"{t.monster}/{t.variant}"
    dets: list[Detection] = []
    if th > sh or tw > sw:
        return key, 0.0, dets
    if t.small_gray is None:                       # petite image : comparaison directe
        full = _match(scene, t.image, t.mask)
        best = float(full.max())
        for x, y, sc in _peaks(full, threshold, max(3, min(th, tw) // 2)):
            dets.append(Detection(t.monster, t.variant, sc, x, y, tw, th))
        return key, best, dets
    # Temps 1 : balayage rapide (demi-résolution, gris, seuil abaissé)
    coarse = _match(small_gray_scene, t.small_gray, t.small_mask)
    best = float(coarse.max())
    coarse_thr = max(0.55, threshold - 0.15)
    hits = _peaks(coarse, coarse_thr, max(3, min(t.small_gray.shape) // 2), limit=6)
    # Temps 2 : vérification précise (couleur, pleine résolution) autour de chaque candidat
    m = 4
    for hx, hy, _ in hits:
        cx, cy = 2 * hx, 2 * hy
        x0, y0 = max(0, cx - m), max(0, cy - m)
        x1, y1 = min(sw, cx + tw + m), min(sh, cy + th + m)
        if x1 - x0 < tw or y1 - y0 < th:
            continue
        res = _match(scene[y0:y1, x0:x1], t.image, t.mask)
        _, score, _, loc = cv2.minMaxLoc(res)
        best = max(best, float(score)) if dets or score >= threshold else best
        if score >= threshold:
            dets.append(Detection(t.monster, t.variant, float(score), x0 + loc[0], y0 + loc[1], tw, th))
    return key, best, dets


def scan(scene: np.ndarray, templates: list[Template], threshold: float) -> ScanResult:
    """Cherche toutes les variantes dans la scène (image BGR).

    Balayage rapide en demi-résolution puis vérification précise autour des candidats.
    Les images de référence sont traitées en parallèle sur plusieurs cœurs.
    """
    result = ScanResult()
    sh, sw = scene.shape[:2]
    small_gray = cv2.cvtColor(cv2.resize(scene, (sw // 2, sh // 2), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2GRAY)
    futures = [_pool().submit(_scan_template, t, scene, small_gray, threshold) for t in templates]
    candidates: list[Detection] = []
    for f in futures:
        key, best, dets = f.result()
        result.best_scores[key] = best
        candidates.extend(dets)
    # Suppression des doublons (même monstre détecté par plusieurs variantes/positions voisines)
    candidates.sort(key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in candidates:
        cx, cy = d.center
        close = False
        for k in kept:
            kx, ky = k.center
            if abs(cx - kx) < 0.5 * max(d.w, k.w) and abs(cy - ky) < 0.5 * max(d.h, k.h):
                close = True
                break
        if not close:
            kept.append(d)
    result.detections = kept
    return result


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def annotate(scene: np.ndarray, detections: list[Detection], clicked: tuple[int, int] | None = None) -> np.ndarray:
    out = scene.copy()
    for i, d in enumerate(detections):
        color = (0, 255, 0) if i == 0 else (0, 200, 255)
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        label = f"{_ascii(d.monster)}/{_ascii(d.variant)} {d.score:.2f}"
        cv2.putText(out, label, (d.x, max(12, d.y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if clicked:
        cv2.drawMarker(out, clicked, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return out
