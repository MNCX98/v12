"""Phase 1 : recherche des monstres et clic sur la meilleure correspondance."""
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from core import input as inp
from core.imgio import save_image
from core.paths import logs_dir
from core.vision import Detection, annotate, scan
from phases.base import Context


@dataclass
class ScanOutcome:
    frame: np.ndarray              # image annotée à afficher
    detection: Detection | None    # cible choisie (ou None)
    clicked: bool


def scan_once(ctx: Context, do_click: bool) -> ScanOutcome:
    """Une observation : capture -> recherche -> (clic éventuel)."""
    t0 = time.time()
    scene, (ox, oy) = ctx.capture.grab()
    t1 = time.time()
    result = scan(scene, ctx.templates, ctx.config.threshold)
    ms_cap, ms_scan = (t1 - t0) * 1000, (time.time() - t1) * 1000

    if not result.detections:
        best = max(result.best_scores.values(), default=0.0)
        ctx.log(f"Aucune cible (meilleur score {best:.2f}, seuil {ctx.config.threshold:.2f}) "
                f"· image {scene.shape[1]}×{scene.shape[0]} · capture {ms_cap:.0f} ms, analyse {ms_scan:.0f} ms")
        return ScanOutcome(annotate(scene, []), None, False)

    target = result.detections[0]          # la plus fiable
    cx, cy = target.center
    ctx.log(f"Cible : {target.monster}/{target.variant} score {target.score:.2f} "
            f"en ({cx}, {cy}) — {len(result.detections)} détection(s) · analyse {ms_scan:.0f} ms")
    clicked = False
    if do_click:
        n = max(1, int(ctx.config.click_count))
        for i in range(n):
            inp.click(ox + cx, oy + cy)
            if i < n - 1:
                time.sleep(ctx.config.click_interval_ms / 1000)
        clicked = True
        ctx.log(f"{'Double clic' if n == 2 else str(n) + ' clic(s)'} effectué en ({cx}, {cy})")
    frame = annotate(scene, result.detections, (cx, cy) if clicked else None)
    if ctx.config.save_diagnostics:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_image(logs_dir() / f"diag_{stamp}.png", frame)
    return ScanOutcome(frame, target, clicked)
