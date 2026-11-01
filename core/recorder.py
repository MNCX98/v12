"""Enregistrement d'une séquence de captures + planche-contact (pour analyser les transitions)."""
import time
from datetime import datetime

import cv2
import numpy as np

from core.imgio import save_image


def _contact_sheet(frames: list[np.ndarray], cols: int = 6, thumb_w: int = 320) -> np.ndarray:
    h0, w0 = frames[0].shape[:2]
    thumb_h = max(1, round(thumb_w * h0 / w0))
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 30, np.uint8)
    for i, f in enumerate(frames):
        t = cv2.resize(f, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        label = f"{i + 1}"
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = t
    return sheet


def record_sequence(capture, out_root, seconds: int, log, should_stop, on_frame=None, interval: float = 1.0):
    """Une capture par seconde. Crée un dossier avec les images numérotées et planche.jpg."""
    folder = out_root / datetime.now().strftime("sequence_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    log(f"Enregistrement de {seconds} s : fais maintenant ton action dans le jeu (F12 pour arrêter).")
    for i in range(int(seconds / interval)):
        if should_stop():
            break
        t0 = time.time()
        img, _ = capture.grab()
        frames.append(img)
        save_image(folder / f"{i + 1:02d}.png", img)
        if on_frame:
            on_frame(img)
        while time.time() - t0 < interval and not should_stop():
            time.sleep(0.03)
    if frames:
        save_image(folder / "planche.jpg", _contact_sheet(frames), ".jpg")
        log(f"Séquence enregistrée : {len(frames)} images dans captures/{folder.name}. "
            f"Envoie-moi planche.jpg (et les images demandées).")
    else:
        log("Aucune image enregistrée.")
