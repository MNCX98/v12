"""Capture de la fenêtre BlueStacks (zone client, sans les bordures)."""
import sys
import threading

import numpy as np

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class WindowNotFound(Exception):
    pass


class WindowCapture:
    def __init__(self, title_contains: str):
        self.title_contains = title_contains
        self._local = threading.local()   # mss doit vivre dans le thread qui l'utilise
        self.last_title = ""

    def _find_window(self):
        if not IS_WINDOWS:
            return None
        found = []
        needle = self.title_contains.lower()

        def callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n == 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            low = title.lower()
            if needle in low and "dofusauto" not in low:
                r = wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                found.append(((r.right - r.left) * (r.bottom - r.top), hwnd, title))
            return True

        user32.EnumWindows(_EnumProc(callback), 0)
        if not found:
            return None
        found.sort(reverse=True)
        return found[0][1], found[0][2]

    def grab(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Retourne (image BGR, (x, y) du coin haut-gauche à l'écran)."""
        if not IS_WINDOWS:
            raise WindowNotFound("La capture ne fonctionne que sous Windows.")
        import mss

        win = self._find_window()
        if win is None:
            raise WindowNotFound(f"Fenêtre « {self.title_contains} » introuvable (ou réduite).")
        hwnd, title = win
        self.last_title = title
        r = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w < 50 or h < 50:
            raise WindowNotFound("La fenêtre est trop petite ou réduite.")
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = self._local.sct = mss.mss()
        shot = sct.grab({"left": pt.x, "top": pt.y, "width": w, "height": h})
        img = np.ascontiguousarray(np.array(shot)[:, :, :3])   # BGRA -> BGR
        return img, (pt.x, pt.y)
