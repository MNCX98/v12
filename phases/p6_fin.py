"""Phase 6 : fin du combat. Fermer la fenêtre « Résultat du combat » et attendre le retour à la carte."""
import time

from core import input as inp
from phases.base import Context
from phases.p2_combat import find_ui


def close_result(ctx: Context, should_stop) -> bool:
    """Clique sur la croix de la fenêtre de résultat, puis attend le retour à l'écran de recherche."""
    ctx.log("Phase 6 : fermeture de la fenêtre de résultat…")
    closed = False
    for attempt in range(3):
        scene, (ox, oy) = ctx.capture.grab()
        title = find_ui(ctx, scene, "resultat_titre")
        if not title:
            closed = True                                  # déjà fermée
            break
        btn = find_ui(ctx, scene, "fermer_resultat")
        if btn:
            inp.click(ox + btn.center[0], oy + btn.center[1])
        else:
            ctx.log("⚠ Croix de fermeture introuvable.")
        t0 = time.time()
        while time.time() - t0 < 3 and not should_stop():
            time.sleep(0.4)
            s, _ = ctx.capture.grab()
            if not find_ui(ctx, s, "resultat_titre"):
                closed = True
                break
        if closed or should_stop():
            break
        ctx.log(f"La fenêtre est toujours là (essai {attempt + 1}/3).")
    if not closed:
        ctx.log("⚠ Impossible de fermer la fenêtre de résultat.")
        return False
    # Retour à la carte : plus de fenêtre de résultat, plus de grille de sorts de combat
    t0 = time.time()
    while time.time() - t0 < 10 and not should_stop():
        s, _ = ctx.capture.grab()
        if not find_ui(ctx, s, "resultat_titre") and not find_ui(ctx, s, "sort_attaque"):
            time.sleep(1.5)                                # le personnage réapparaît sur la carte
            ctx.log("Retour à la carte.")
            return True
        time.sleep(0.5)
    ctx.log("⚠ Le retour à la carte n'est pas confirmé.")
    return False
