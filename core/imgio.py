"""Enregistrement d'images (compatible avec les chemins accentués sous Windows)."""
import cv2


def save_image(path, img, ext: str = ".png") -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, 88] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
