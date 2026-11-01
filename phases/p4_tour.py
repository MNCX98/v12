"""Phases 4 et 5 : jouer mon tour (lancer le sort sur l'adversaire le plus proche à portée,
puis finir le tour) et recommencer à chaque tour."""
import time
from datetime import datetime

import cv2
import numpy as np

from core import input as inp
from core.combat import (find_enemies, find_me, range_fraction, range_mask, range_visible,
                         sort_by_distance)
from core.combat import _game_roi
from core.grid import grid_distance, move_candidates
from core.imgio import save_image
from core.paths import logs_dir
from core.vision import scan
from phases.base import Context
from phases.p2_combat import find_ui, read_ui


def _find(ctx: Context, scene, name: str):
    return find_ui(ctx, scene, name)


def wait_my_turn(ctx: Context, should_stop) -> str:
    """Attend mon tour. Retourne 'turn', 'over' (fenêtre de résultat), 'timeout' ou 'stopped'."""
    t0 = time.time()
    while time.time() - t0 < ctx.config.turn_wait_s and not should_stop():
        scene, _ = ctx.capture.grab()
        r = read_ui(scene, ctx.ui_templates, ctx.config.ui_threshold)
        if "resultat_titre" in r.found:
            return "over"
        if r.turn_active:
            return "turn"
        time.sleep(0.4)
    return "stopped" if should_stop() else "timeout"


def _diag(ctx: Context, scene, enemies, me, target=None, label="tour"):
    if not ctx.config.save_diagnostics:
        return
    out = scene.copy()
    for f in enemies:
        cv2.circle(out, f.center, 8, (0, 255, 255), -1)
    if me:
        cv2.circle(out, me.center, 10, (0, 0, 255), -1)
    if target:
        cv2.circle(out, target.center, 22, (0, 255, 0), 3)
    save_image(logs_dir() / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}.png", out)


def _wait_confirm(ctx: Context, timeout: float = 1.6):
    t0 = time.time()
    while time.time() - t0 < timeout:
        scene, origin = ctx.capture.grab()
        d = _find(ctx, scene, "valider")
        if d:
            return d, origin
        time.sleep(0.2)
    return None, None


def _select_spell(ctx: Context) -> bool:
    scene, (ox, oy) = ctx.capture.grab()
    if range_visible(scene):
        return True                                   # déjà sélectionné
    icon = _find(ctx, scene, "sort_attaque")
    if not icon:
        ctx.log("⚠ Icône du sort introuvable.")
        return False
    inp.click(ox + icon.center[0], oy + icon.center[1])
    t0 = time.time()
    while time.time() - t0 < 1.8:
        time.sleep(0.3)
        s, _ = ctx.capture.grab()
        if range_visible(s):
            return True
    ctx.log("Le sort ne s'affiche pas (plus assez de PA ?).")
    return False


def _deselect_spell(ctx: Context) -> bool:
    """Le jeu refuse de déplacer le personnage tant que le sort est sélectionné : on reclique dessus."""
    scene, (ox, oy) = ctx.capture.grab()
    if not range_visible(scene):
        return True
    icon = _find(ctx, scene, "sort_attaque")
    if not icon:
        ctx.log("⚠ Icône du sort introuvable : impossible de désélectionner le sort.")
        return False
    inp.click(ox + icon.center[0], oy + icon.center[1])
    t0 = time.time()
    while time.time() - t0 < 1.8:
        time.sleep(0.3)
        s, _ = ctx.capture.grab()
        if not range_visible(s):
            return True
    ctx.log("⚠ Le sort reste sélectionné : pas de déplacement.")
    return False


def plan_move(ctx: Context, scene, enemies, me):
    """Monstre visé et cases candidates (les meilleures d'abord) pour se rapprocher."""
    target = sort_by_distance(enemies, me.center)[0]
    h, w = scene.shape[:2]
    roi = _game_roi(h, w)
    cands = [(k, p) for k, p in move_candidates(me.center, target.center, [e.center for e in enemies],
                                               ctx.config.max_move_cells)
             if 0 <= p[0] < w and 0 <= p[1] < h and roi[p[1], p[0]]]
    return target, cands


def _wait_moved(ctx: Context, start: tuple[int, int], timeout: float = 4.0):
    """Attend que mon personnage ait changé de case et soit immobile. Retourne sa position ou None."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        time.sleep(0.4)
        s, _ = ctx.capture.grab()
        me = find_me(s)
        if not me or grid_distance(me.center, start) < 0.75:
            last = None
            continue
        if last and grid_distance(me.center, last) < 0.5:
            return me.center
        last = me.center
    return last


def move_closer(ctx: Context, enemies, me, should_stop) -> bool:
    """Se rapproche du monstre le plus proche (double clic sur une case), au plus max_move_cells cases."""
    if not _deselect_spell(ctx):
        return False
    scene, (ox, oy) = ctx.capture.grab()
    target, cands = plan_move(ctx, scene, enemies, me)
    dist = grid_distance(target.center, me.center)
    if not cands:
        ctx.log(f"Monstre le plus proche à {dist:.0f} case(s) : déjà au contact ou aucune case possible.")
        return False
    for k, (x, y) in cands[:4]:
        if should_stop():
            return False
        ctx.log(f"Déplacement de {k} case(s) vers le monstre en {target.center} (à {dist:.0f} case(s)) : "
                f"double clic en ({x}, {y}).")
        _diag(ctx, scene, enemies, me, target, label="deplacement")
        for i in range(2):
            inp.click(ox + x, oy + y)
            if i == 0:
                time.sleep(ctx.config.click_interval_ms / 1000)
        pos = _wait_moved(ctx, me.center)
        if pos:
            ctx.me_hint = pos
            ctx.log(f"✔ Personnage déplacé en {pos}.")
            return True
        ctx.log("Le personnage n'a pas bougé (case inaccessible ?), j'essaie une autre case.")
    ctx.log("⚠ Déplacement impossible : je finis le tour.")
    return False


def play_turn(ctx: Context, should_stop) -> int:
    """Joue un tour. Retourne le nombre de lancers réussis."""
    cfg = ctx.config
    ctx.log("Mon tour.")
    time.sleep(cfg.turn_start_delay_s)
    casts = 0
    while casts < cfg.casts_per_turn and not should_stop():
        scene0, _ = ctx.capture.grab()
        if _find(ctx, scene0, "resultat_titre"):
            ctx.log("Le combat est terminé.")
            break
        if not _select_spell(ctx):
            break
        scene, (ox, oy) = ctx.capture.grab()
        enemies, me = find_enemies(scene), find_me(scene, ctx.me_hint)
        if me:
            ctx.me_hint = me.center
        if not enemies:
            ctx.log("Aucun adversaire visible.")
            break
        mask = range_mask(scene)
        if me:
            origin = me.center
        else:
            ys, xs = np.where(mask > 0)
            origin = (int(xs.mean()), int(ys.mean())) if len(xs) else enemies[0].center
        in_range = [f for f in sort_by_distance(enemies, origin)
                    if range_fraction(mask, f.center) >= cfg.range_min_fraction]
        ctx.log(f"{len(enemies)} adversaire(s), {len(in_range)} à portée.")
        if not in_range:
            if cfg.move_when_out_of_range and me:
                ctx.log("Aucun adversaire à portée : je me rapproche.")
                move_closer(ctx, enemies, me, should_stop)
            else:
                ctx.log("Aucun adversaire à portée : je passe le tour.")
            break
        done = False
        for target in in_range[:3]:
            dist = grid_distance(target.center, origin)
            ctx.log(f"Cible à {dist:.1f} case(s) en {target.center}.")
            _diag(ctx, scene, enemies, me, target)
            inp.click(ox + target.cx, oy + target.cy - 4)
            ok, origin2 = _wait_confirm(ctx)
            if ok:
                ox2, oy2 = origin2
                inp.click(ox2 + ok.center[0], oy2 + ok.center[1])
                casts += 1
                ctx.log(f"✔ Lancer {casts}/{cfg.casts_per_turn} confirmé.")
                time.sleep(1.5)                       # animation + dégâts
                if casts < cfg.casts_per_turn and cfg.pause_between_casts_ms > 0:
                    ctx.log(f"Pause de {cfg.pause_between_casts_ms} ms avant le lancer suivant.")
                    time.sleep(cfg.pause_between_casts_ms / 1000)
                done = True
                break
            ctx.log("Pas de confirmation sur cette cible.")
            if not _select_spell(ctx):
                break
            scene, (ox, oy) = ctx.capture.grab()
        if not done:
            break
    return casts


def end_turn(ctx: Context) -> bool:
    scene, (ox, oy) = ctx.capture.grab()
    r = read_ui(scene, ctx.ui_templates, ctx.config.ui_threshold)
    fin = r.found.get("fin_du_tour")
    if "resultat_titre" in r.found:
        ctx.log("Le combat est terminé.")
        return False
    if not fin or not r.turn_active:
        ctx.log("⚠ Bouton « Fin du tour » vert introuvable.")
        return False
    inp.click(ox + fin.center[0], oy + fin.center[1])
    t0 = time.time()
    while time.time() - t0 < 6:
        time.sleep(0.4)
        s, _ = ctx.capture.grab()
        if not read_ui(s, ctx.ui_templates, ctx.config.ui_threshold).turn_active:
            ctx.log("Tour terminé.")
            return True
    ctx.log("⚠ Le tour ne semble pas terminé.")
    return False


def run_turns(ctx: Context, should_stop) -> str:
    """Boucle des tours. Retourne 'over' (combat terminé), 'timeout' ou 'stopped'."""
    ctx.log("Phases 4-5 : je joue les tours (F12 pour arrêter).")
    n = 0
    while not should_stop():
        status = wait_my_turn(ctx, should_stop)
        if status == "over":
            ctx.log("Le combat est terminé (fenêtre de résultat).")
            return "over"
        if status == "timeout":
            ctx.log(f"Aucun tour à jouer pendant {ctx.config.turn_wait_s} s : le programme s'arrête.")
            return "timeout"
        if status == "stopped":
            break
        n += 1
        ctx.log(f"── Tour {n} ──")
        play_turn(ctx, should_stop)
        if should_stop():
            break
        end_turn(ctx)
    return "stopped"


def test_combat(ctx: Context) -> np.ndarray:
    """Diagnostic sans clic : adversaires, mon personnage, portée."""
    scene, _ = ctx.capture.grab()
    enemies, me = find_enemies(scene), find_me(scene, ctx.me_hint)
    if me:
        ctx.me_hint = me.center
    mask = range_mask(scene)
    visible = range_visible(scene)
    ctx.log(f"Mon personnage : {me.center if me else 'introuvable'} · {len(enemies)} adversaire(s) · "
            f"portée du sort {'affichée' if visible else 'non affichée (sélectionne d’abord le sort)'}")
    origin = me.center if me else None
    for f in (sort_by_distance(enemies, origin) if origin else enemies):
        frac = range_fraction(mask, f.center)
        dist = f"{grid_distance(f.center, origin):.1f} case(s)" if origin else "?"
        ctx.log(f"  adversaire {f.center} : {dist}" + (f" · {'À PORTÉE' if frac >= ctx.config.range_min_fraction else 'hors portée'}"
                                                        if visible else ""))
    out = scene.copy()
    move_to = None
    if me and enemies and visible and ctx.config.move_when_out_of_range and not any(
            range_fraction(mask, f.center) >= ctx.config.range_min_fraction for f in enemies):
        target, cands = plan_move(ctx, scene, enemies, me)
        if cands:
            move_to = cands[0]
            ctx.log(f"Personne à portée : je me déplacerais de {move_to[0]} case(s) en {move_to[1]} "
                    f"vers le monstre en {target.center}.")
        else:
            ctx.log("Personne à portée et aucune case de déplacement possible.")
    out[mask > 0] = (0.55 * out[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    for f in enemies:
        cv2.circle(out, f.center, 9, (0, 255, 255), -1)
    if me:
        cv2.circle(out, me.center, 11, (0, 0, 255), -1)
    if move_to:
        cv2.drawMarker(out, move_to[1], (255, 0, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
    return out
