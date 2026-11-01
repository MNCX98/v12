"""Phase 2 : détection du combat (bouton « Prêt » = combat chargé, écran de placement)."""
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from core.imgio import save_image
from core.paths import logs_dir
from core.vision import Detection, ScanResult, annotate, scan
from phases.base import Context

GREEN_SATURATION_MIN = 90     # bouton « Fin du tour » : vert ≈ 170, grisé ≈ 13


@dataclass
class UiReading:
    found: dict[str, Detection] = field(default_factory=dict)
    turn_active: bool = False          # « Fin du tour » visible ET vert = c'est mon tour
    raw: ScanResult | None = None


def read_ui(scene: np.ndarray, ui_templates, threshold: float) -> UiReading:
    """Repère les éléments d'interface connus dans l'écran."""
    result = scan(scene, ui_templates, threshold)
    reading = UiReading(raw=result)
    for d in result.detections:                 # triées par score décroissant
        reading.found.setdefault(d.variant, d)
    fin = reading.found.get("fin_du_tour")
    if fin:
        patch = scene[fin.y:fin.y + fin.h, fin.x:fin.x + fin.w]
        sat = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        reading.turn_active = sat >= GREEN_SATURATION_MIN
    return reading


def find_ui(ctx: Context, scene: np.ndarray, name: str) -> Detection | None:
    """Cherche un élément d'interface précis (bouton, titre de fenêtre…) dans l'écran."""
    templates = [t for t in ctx.ui_templates if t.variant == name]
    if not templates:
        return None
    r = scan(scene, templates, ctx.config.ui_threshold)
    return r.detections[0] if r.detections else None


def _save_diag(ctx: Context, frame: np.ndarray, prefix: str) -> None:
    if ctx.config.save_diagnostics:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_image(logs_dir() / f"{prefix}_{stamp}.png", frame)


def wait_for_combat(ctx: Context, should_stop, on_frame=None) -> bool:
    """Attend l'apparition du bouton « Prêt ». True si le combat est chargé."""
    cfg = ctx.config
    pret = [t for t in ctx.ui_templates if t.variant == "pret"]
    if not pret:
        ctx.log("⚠ Image « pret.png » introuvable dans assets/ui : phase 2 impossible.")
        return False
    ctx.log(f"Phase 2 : attente du combat (maximum {cfg.combat_wait_s} s)…")
    t0 = time.time()
    while time.time() - t0 < cfg.combat_wait_s and not should_stop():
        scene, _ = ctx.capture.grab()
        result = scan(scene, pret, cfg.ui_threshold)
        if result.detections:
            d = result.detections[0]
            frame = annotate(scene, [d])
            if on_frame:
                on_frame(frame)
            ctx.log(f"✔ Combat chargé : bouton « Prêt » détecté (score {d.score:.2f}) "
                    f"après {time.time() - t0:.1f} s")
            _save_diag(ctx, frame, "combat")
            return True
        time.sleep(0.3)
    if not should_stop():
        ctx.log(f"Aucun combat détecté après {cfg.combat_wait_s} s : on reprend la recherche.")
    return False


def test_ui(ctx: Context) -> np.ndarray:
    """Diagnostic (sans clic) : quels éléments du jeu le programme voit-il ?"""
    scene, _ = ctx.capture.grab()
    r = read_ui(scene, ctx.ui_templates, ctx.config.ui_threshold)
    f = r.found
    ctx.log("Bouton « Prêt » (placement) : " +
            (f"visible (score {f['pret'].score:.2f})" if "pret" in f else "non visible"))
    ctx.log("Bouton « Fin du tour » vert (c'est ton tour) : " +
            ("OUI" if r.turn_active else "non" + (" (bouton visible mais grisé)" if "fin_du_tour" in f else "")))
    ctx.log("Icône de sort (combat en cours) : " +
            (f"visible (score {f['sort_attaque'].score:.2f})" if "sort_attaque" in f else "non visible"))
    ctx.log("Icône du Pied du Sacrieur : " +
            (f"visible (score {f['pied_du_sacrieur'].score:.2f})" if "pied_du_sacrieur" in f else "non visible"))
    ctx.log("Fenêtre « Résultat du combat » (fin de combat) : " +
            (f"visible, bouton de fermeture " + ("trouvé" if "fermer_resultat" in f else "INTROUVABLE")
             if "resultat_titre" in f else "non visible"))
    return annotate(scene, list(f.values()))
