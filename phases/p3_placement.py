"""Phase 3 : placement du personnage sur une case rouge, puis clic sur « Prêt »."""
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from core import input as inp
from core.grid import Cell, find_cells, grid_distance
from core.imgio import save_image
from core.paths import logs_dir
from core.vision import annotate, scan
from phases.base import Context

RULES = {"closest": "Case la plus proche des monstres",
         "farthest": "Case la plus éloignée des monstres",
         "stay": "Ne pas bouger"}


@dataclass
class Plan:
    mine: list[Cell] = field(default_factory=list)       # cases rouges cliquables
    enemies: list[Cell] = field(default_factory=list)    # cases bleues
    occupied: list[Cell] = field(default_factory=list)   # cases bleues où se tient un monstre
    chosen: Cell | None = None
    distance: float | None = None


def occupied_cells(cells: list[Cell]) -> list[Cell]:
    """Cases bleues sur lesquelles se tient un monstre.

    Le monstre (ou son pion en mode créature) cache le haut du losange bleu : la case paraît
    nettement plus petite qu'une case libre (aire, hauteur ou largeur réduites).
    """
    if not cells:
        return []
    ref_area = sorted(c.area for c in cells)[len(cells) * 3 // 4]
    ref_w = max(c.w for c in cells)
    ref_h = max(c.h for c in cells)
    return [c for c in cells
            if c.area < 0.85 * ref_area or c.h < 0.88 * ref_h or c.w < 0.85 * ref_w]


def make_plan(scene: np.ndarray, rule: str) -> Plan:
    merged: list[Cell] = []
    plan = Plan(mine=find_cells(scene, "red", min_area=600),
                enemies=find_cells(scene, "blue", min_area=400, merged=merged))
    plan.occupied = occupied_cells(plan.enemies) + merged      # tache fusionnée = un monstre sélectionné
    if not plan.mine or rule == "stay":
        return plan
    if not plan.enemies:
        return plan                                       # pas de repère : on ne bouge pas
    # Repères : les cases où se tiennent vraiment les monstres (toutes les cases bleues si on ne les voit pas)
    marks = plan.occupied or plan.enemies
    def nearest(c: Cell) -> float:
        return min(grid_distance(c.center, e.center) for e in marks)
    ex = sum(e.cx for e in marks) / len(marks)
    ey = sum(e.cy for e in marks) / len(marks)
    def key(c: Cell):
        return (nearest(c), grid_distance(c.center, (ex, ey)), c.cx)
    plan.chosen = min(plan.mine, key=key) if rule == "closest" else max(plan.mine, key=key)
    plan.distance = nearest(plan.chosen)
    return plan


def draw_plan(scene: np.ndarray, plan: Plan, clicked: bool = False) -> np.ndarray:
    out = scene.copy()
    for c in plan.mine:
        cv2.circle(out, c.center, 6, (0, 255, 255), -1)
    for c in plan.enemies:
        cv2.circle(out, c.center, 6, (255, 255, 0), -1)
    for c in plan.occupied:
        cv2.circle(out, c.center, 11, (0, 140, 255), 3)          # orange : un monstre est sur cette case
    if plan.chosen:
        cv2.circle(out, plan.chosen.center, 18, (0, 255, 0), 3)
        if clicked:
            cv2.drawMarker(out, plan.chosen.center, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    return out


def _describe(ctx: Context, plan: Plan, rule: str) -> None:
    ctx.log(f"Cases : {len(plan.mine)} à moi (rouges), {len(plan.enemies)} côté monstres (bleues), "
            f"dont {len(plan.occupied)} occupée(s) par un monstre.")
    if rule == "stay":
        ctx.log("Règle « ne pas bouger » : aucun déplacement.")
    elif plan.chosen:
        ctx.log(f"Case choisie ({RULES[rule].lower()}) en {plan.chosen.center}, "
                f"à {plan.distance:.1f} case(s) du monstre le plus proche"
                f"{'' if plan.occupied else ' (monstres non repérés : je vise la case bleue la plus proche)'}.")
    else:
        ctx.log("Aucune case choisie (cases rouges ou bleues introuvables).")


def test_placement(ctx: Context) -> np.ndarray:
    """Diagnostic sans clic : montre la case que le programme choisirait."""
    scene, _ = ctx.capture.grab()
    plan = make_plan(scene, ctx.config.placement_rule)
    _describe(ctx, plan, ctx.config.placement_rule)
    return draw_plan(scene, plan)


def _stable_plan(ctx: Context, should_stop, timeout: float = 6.0):
    """Attend que le bandeau « Prépare-toi » disparaisse (cases identiques à deux lectures de suite)."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout and not should_stop():
        scene, origin = ctx.capture.grab()
        plan = make_plan(scene, ctx.config.placement_rule)
        sig = (sorted((c.cx // 4, c.cy // 4) for c in plan.mine), sorted((c.cx // 4, c.cy // 4) for c in plan.enemies))
        if plan.mine and sig == last:
            return scene, origin, plan
        last = sig
        time.sleep(0.4)
    scene, origin = ctx.capture.grab()
    return scene, origin, make_plan(scene, ctx.config.placement_rule)


def run_placement(ctx: Context, should_stop, on_frame=None) -> bool:
    """Se place, clique sur « Prêt », vérifie que le combat démarre. True si réussi."""
    cfg = ctx.config
    rule = cfg.placement_rule
    ctx.log(f"Phase 3 : placement ({RULES[rule].lower()})…")
    scene, (ox, oy), plan = _stable_plan(ctx, should_stop)
    if should_stop():
        return False
    _describe(ctx, plan, rule)
    if rule != "stay" and plan.chosen is None:
        ctx.log("⚠ Placement impossible : je continue sans me déplacer.")
    moved = rule != "stay" and plan.chosen is not None
    if moved:
        inp.click(ox + plan.chosen.cx, oy + plan.chosen.cy)
        ctx.log(f"Clic sur la case {plan.chosen.center}.")
        ctx.me_hint = plan.chosen.center                  # ma position de départ pour le combat
        time.sleep(0.8)
    frame = draw_plan(scene, plan, clicked=moved)
    if on_frame:
        on_frame(frame)
    if cfg.save_diagnostics:
        save_image(logs_dir() / f"placement_{datetime.now():%Y%m%d_%H%M%S}.png", frame)

    # Clic sur « Prêt »
    pret = [t for t in ctx.ui_templates if t.variant == "pret"]
    scene2, (ox, oy) = ctx.capture.grab()
    res = scan(scene2, pret, cfg.ui_threshold)
    if not res.detections:
        ctx.log("⚠ Bouton « Prêt » introuvable : placement non validé.")
        return False
    bx, by = res.detections[0].center
    inp.click(ox + bx, oy + by)
    ctx.log("Clic sur « Prêt ».")

    # Vérification : le bouton disparaît (ou l'icône de sort apparaît) = le combat a démarré
    sort = [t for t in ctx.ui_templates if t.variant == "sort_attaque"]
    t0, gone = time.time(), 0
    while time.time() - t0 < 12 and not should_stop():
        time.sleep(0.5)
        s, _ = ctx.capture.grab()
        pret_here = bool(scan(s, pret, cfg.ui_threshold).detections)
        sort_here = bool(sort) and bool(scan(s, sort, cfg.ui_threshold).detections)
        gone = 0 if pret_here else gone + 1
        if sort_here or gone >= 2:
            ctx.log(f"✔ Combat lancé (vérifié après {time.time() - t0:.1f} s).")
            return True
    ctx.log("⚠ Le combat ne semble pas avoir démarré (bouton « Prêt » toujours là).")
    return False
