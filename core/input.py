"""Souris et clavier (Windows, via l'API native : aucune dépendance)."""
import sys
import time

IS_WINDOWS = sys.platform == "win32"
VK_F12 = 0x7B

if IS_WINDOWS:
    import ctypes

    _user32 = ctypes.windll.user32


def click(x: int, y: int, hold_s: float = 0.06) -> None:
    """Clic gauche à la position écran (x, y)."""
    if not IS_WINDOWS:
        raise RuntimeError("Les clics ne fonctionnent que sous Windows.")
    _user32.SetCursorPos(int(x), int(y))
    time.sleep(0.05)
    _user32.mouse_event(0x0002, 0, 0, 0, 0)   # bouton gauche enfoncé
    time.sleep(hold_s)
    _user32.mouse_event(0x0004, 0, 0, 0, 0)   # bouton gauche relâché


def is_key_down(vk: int) -> bool:
    if not IS_WINDOWS:
        return False
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)
