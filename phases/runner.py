"""Enchaîne toutes les phases en boucle : recherche (1), combat (2), placement (3),
tours (4-5), fin du combat (6), puis retour à la recherche."""
import time

from phases.base import Context
from phases.p1_recherche import scan_once
from phases.p2_combat import wait_for_combat
from phases.p3_placement import run_placement
from phases.p4_tour import run_turns
from phases.p6_fin import close_result
from phases.p_vie import ensure_health


def _sleep(seconds: float, should_stop) -> None:
    end = time.time() + seconds
    while time.time() < end and not should_stop():
        time.sleep(0.05)


def _fight(ctx: Context, should_stop, on_frame) -> str:
    """Déroule un combat après le clic sur le monstre. Retourne 'next' (reprendre la recherche),
    'done' (combat terminé et fenêtre fermée) ou 'stop' (le programme doit s'arrêter)."""
    cfg = ctx.config
    if not wait_for_combat(ctx, should_stop, on_frame):            # phase 2
        return "next"
    if not cfg.do_placement:
        ctx.log("Phase 2 validée. Le programme s'arrête ici.")
        return "stop"
    if not run_placement(ctx, should_stop, on_frame):              # phase 3
        ctx.log("Phase 3 non aboutie : le programme s'arrête, reprends la main.")
        return "stop"
    if not cfg.do_turns:
        ctx.log("Phase 3 validée. Le programme s'arrête ici.")
        return "stop"
    status = run_turns(ctx, should_stop)                           # phases 4-5
    if status != "over":
        return "stop"
    if not close_result(ctx, should_stop):                         # phase 6
        return "stop"
    return "done"


def run_phases(ctx: Context, should_stop, on_frame=None) -> None:
    cfg = ctx.config
    ctx.log("Démarré. Appuie sur F12 pour arrêter.")
    combats = 0
    need_heal = True                                               # contrôle de la vie avant chaque combat
    while not should_stop():
        pause = cfg.scan_interval_ms / 1000
        try:
            if need_heal and cfg.heal_enabled:
                if ensure_health(ctx, should_stop) == "fail":
                    if not should_stop():
                        ctx.log("Le programme s'arrête : vie insuffisante, reprends la main.")
                    return
                need_heal = False
            out = scan_once(ctx, do_click=True)                    # phase 1
            if on_frame:
                on_frame(out.frame)
            if out.clicked:
                if cfg.detect_combat:
                    result = _fight(ctx, should_stop, on_frame)
                    if result == "stop":
                        return
                    if result == "done":
                        combats += 1
                        ctx.me_hint = None
                        need_heal = True
                        ctx.log(f"✔ Combat n°{combats} terminé.")
                        if not cfg.loop_combats:
                            ctx.log("Arrêt après un combat (option « enchaîner les combats » décochée).")
                            return
                        ctx.log("Je reprends la recherche d'un monstre.")
                        pause = 2.0
                else:
                    pause = cfg.cooldown_after_click_s
        except Exception as e:                                     # fenêtre introuvable, etc.
            ctx.log(f"⚠ {e}")
            pause = 2.0
        _sleep(pause, should_stop)
    if combats:
        ctx.log(f"Arrêté après {combats} combat(s) terminé(s).")
    else:
        ctx.log("Arrêté.")
