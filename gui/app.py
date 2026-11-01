"""Fenêtre principale (PySide6)."""
import os
import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import input as inp
from core.capture import WindowCapture
from core.config import load_config, save_config
from core.paths import captures_dir, config_path, logs_dir, monsters_dir, ui_dir
from core.vision import load_templates, load_ui_templates
from phases.base import Context
from core.imgio import save_image
from core.recorder import record_sequence
from phases.p1_recherche import scan_once
from phases.p2_combat import test_ui
from phases.p3_placement import RULES, test_placement
from phases.p4_tour import test_combat
from phases.runner import run_phases

VERSION = "0.8.1"

# Modes pendant lesquels DofusAuto se réduit pour ne pas cacher le jeu, puis revient.
HIDE_MODES = {"test", "testui", "testplace", "testcombat", "snapshot", "sequence"}
HIDE_DELAY_MS = 700      # laisse à Windows le temps de réduire la fenêtre avant la capture


class Worker(QThread):
    """Exécute une tâche hors du thread de l'interface."""
    log = Signal(str)
    frame = Signal(object)

    def __init__(self, ctx: Context, mode: str):
        super().__init__()
        self.ctx, self.mode = ctx, mode
        self._stop = False
        ctx.log = self.log.emit

    def request_stop(self):
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop or inp.is_key_down(inp.VK_F12)

    def run(self):
        try:
            if self.mode == "loop":
                run_phases(self.ctx, self._should_stop, self.frame.emit)
            elif self.mode == "test":
                self.frame.emit(scan_once(self.ctx, do_click=False).frame)
            elif self.mode == "testcombat":
                self.frame.emit(test_combat(self.ctx))
            elif self.mode == "testplace":
                self.frame.emit(test_placement(self.ctx))
            elif self.mode == "testui":
                self.frame.emit(test_ui(self.ctx))
            elif self.mode == "sequence":
                record_sequence(self.ctx.capture, captures_dir(), self.ctx.config.sequence_seconds,
                                self.log.emit, self._should_stop, self.frame.emit)
            elif self.mode == "snapshot":
                img, _ = self.ctx.capture.grab()
                name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.png")
                save_image(captures_dir() / name, img)
                self.log.emit(f"Capture enregistrée : captures/{name}  ({img.shape[1]}×{img.shape[0]} px)")
                self.frame.emit(img)
        except Exception as e:
            self.log.emit(f"⚠ {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DofusAuto {VERSION} — Phases 1 à 6 : combats en boucle")
        self.resize(1000, 700)
        self.cfg = load_config(config_path())
        self.ctx = Context(self.cfg, WindowCapture(self.cfg.window_title_contains))
        self.worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.info = QLabel()
        root.addWidget(self.info)

        self.preview = QLabel("L'aperçu de ce que le programme voit apparaîtra ici.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 340)
        self.preview.setStyleSheet("background:#222;color:#aaa;")
        root.addWidget(self.preview, 1)

        row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Démarrer")
        self.btn_stop = QPushButton("■ Arrêter (F12)")
        self.btn_test = QPushButton("Tester : monstres (sans clic)")
        self.btn_testui = QPushButton("Tester : boutons du jeu")
        self.btn_testplace = QPushButton("Tester : placement (sans clic)")
        self.btn_testcombat = QPushButton("Tester : combat (sans clic)")
        self.btn_snap = QPushButton("Enregistrer une capture")
        self.btn_seq = QPushButton("Enregistrer une séquence")
        for b in (self.btn_start, self.btn_stop, self.btn_test, self.btn_testui, self.btn_testplace,
                  self.btn_testcombat, self.btn_snap, self.btn_seq):
            row.addWidget(b)
        root.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Seuil de confiance :"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.50, 0.99)
        self.spin.setSingleStep(0.01)
        self.spin.setDecimals(2)
        self.spin.setValue(self.cfg.threshold)
        self.spin.valueChanged.connect(self._threshold_changed)
        row2.addWidget(self.spin)
        row2.addWidget(QLabel("Clics :"))
        self.clicks = QSpinBox()
        self.clicks.setRange(1, 3)
        self.clicks.setValue(self.cfg.click_count)
        self.clicks.valueChanged.connect(lambda v: setattr(self.cfg, "click_count", int(v)))
        row2.addWidget(self.clicks)
        self.chk_combat = QCheckBox("Phase 2 : attendre le combat")
        self.chk_combat.setChecked(self.cfg.detect_combat)
        self.chk_combat.toggled.connect(lambda v: setattr(self.cfg, "detect_combat", bool(v)))
        row2.addWidget(self.chk_combat)
        self.chk_place = QCheckBox("Phase 3 : placement")
        self.chk_place.setChecked(self.cfg.do_placement)
        self.chk_place.toggled.connect(lambda v: setattr(self.cfg, "do_placement", bool(v)))
        row2.addWidget(self.chk_place)
        self.rule = QComboBox()
        for key, label in RULES.items():
            self.rule.addItem(label, key)
        idx = self.rule.findData(self.cfg.placement_rule)
        self.rule.setCurrentIndex(idx if idx >= 0 else 0)
        self.rule.currentIndexChanged.connect(lambda _: setattr(self.cfg, "placement_rule", self.rule.currentData()))
        row2.addWidget(self.rule)
        self.chk_turns = QCheckBox("Phases 4-5 : jouer les tours")
        self.chk_turns.setChecked(self.cfg.do_turns)
        self.chk_turns.toggled.connect(lambda v: setattr(self.cfg, "do_turns", bool(v)))
        row2.addWidget(self.chk_turns)
        row2.addWidget(QLabel("Lancers :"))
        self.casts = QSpinBox()
        self.casts.setRange(1, 3)
        self.casts.setValue(self.cfg.casts_per_turn)
        self.casts.valueChanged.connect(lambda v: setattr(self.cfg, "casts_per_turn", int(v)))
        row2.addWidget(self.casts)
        self.chk_loop = QCheckBox("Phase 6 : enchaîner les combats")
        self.chk_loop.setChecked(self.cfg.loop_combats)
        self.chk_loop.toggled.connect(lambda v: setattr(self.cfg, "loop_combats", bool(v)))
        row2.addWidget(self.chk_loop)
        row2.addStretch(1)
        self.btn_reload = QPushButton("Recharger les images")
        self.btn_imgs = QPushButton("Dossier des images")
        self.btn_logs = QPushButton("Dossier des journaux")
        self.btn_caps = QPushButton("Dossier des captures")
        for b in (self.btn_reload, self.btn_imgs, self.btn_logs, self.btn_caps):
            row2.addWidget(b)
        root.addLayout(row2)

        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        self.logbox.setMaximumHeight(160)
        root.addWidget(self.logbox)

        self.btn_start.clicked.connect(lambda: self._launch("loop"))
        self.btn_test.clicked.connect(lambda: self._launch("test"))
        self.btn_testui.clicked.connect(lambda: self._launch("testui"))
        self.btn_testplace.clicked.connect(lambda: self._launch("testplace"))
        self.btn_testcombat.clicked.connect(lambda: self._launch("testcombat"))
        self.btn_snap.clicked.connect(lambda: self._launch("snapshot"))
        self.btn_seq.clicked.connect(lambda: self._launch("sequence"))
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reload.clicked.connect(self._reload_templates)
        self.btn_imgs.clicked.connect(lambda: self._open(monsters_dir()))
        self.btn_logs.clicked.connect(lambda: self._open(logs_dir()))
        self.btn_caps.clicked.connect(lambda: self._open(captures_dir()))

        self._reload_templates()
        self._set_running(False)
        QTimer.singleShot(0, lambda: self.log("Prêt. Place BlueStacks bien visible (sans le recouvrir), "
                                              "puis clique sur « Tester la détection »."))

    # --- utilitaires -------------------------------------------------
    def log(self, msg: str):
        self.logbox.appendPlainText(f"[{datetime.now():%H:%M:%S}] {msg}")

    def _open(self, path):
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))

    def _threshold_changed(self, v: float):
        self.cfg.threshold = round(v, 2)

    def _reload_templates(self):
        self.ctx.templates = load_templates(monsters_dir())
        self.ctx.ui_templates = load_ui_templates(ui_dir())
        monsters = {}
        for t in self.ctx.templates:
            monsters.setdefault(t.monster, 0)
            monsters[t.monster] += 1
        txt = ", ".join(f"{m} ({n} variante{'s' if n > 1 else ''})" for m, n in monsters.items())
        self.info.setText(f"Monstres chargés : {txt or 'aucun — ajoute des images dans le dossier des images'}")
        self.log(f"{len(self.ctx.templates)} image(s) de monstre et "
                 f"{len(self.ctx.ui_templates)} image(s) de bouton chargée(s).")

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_test.setEnabled(not running)
        self.btn_testui.setEnabled(not running)
        self.btn_testplace.setEnabled(not running)
        self.btn_testcombat.setEnabled(not running)
        self.btn_snap.setEnabled(not running)
        self.btn_seq.setEnabled(not running)
        self.btn_reload.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _launch(self, mode: str):
        if self.worker and self.worker.isRunning():
            return
        self.ctx.capture.title_contains = self.cfg.window_title_contains
        self.worker = Worker(self.ctx, mode)
        self.worker.log.connect(self.log)
        self.worker.frame.connect(self._show_frame)
        self.worker.finished.connect(self._worker_finished)
        self._set_running(True)
        if mode in HIDE_MODES:
            self._prev_state = self.windowState() & ~Qt.WindowMinimized
            self._hidden = True
            self.showMinimized()
            QTimer.singleShot(HIDE_DELAY_MS, self.worker.start)
        else:
            self.worker.start()

    def _worker_finished(self):
        self._set_running(False)
        if getattr(self, "_hidden", False):
            self._hidden = False
            self.setWindowState(self._prev_state | Qt.WindowActive)
            self.show()
            self.raise_()
            self.activateWindow()

    def _stop(self):
        if self.worker:
            self.worker.request_stop()

    def _show_frame(self, img: np.ndarray):
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def closeEvent(self, event):
        self._stop()
        if self.worker:
            self.worker.wait(3000)
        try:
            save_config(self.cfg, config_path())
        except OSError:
            pass
        event.accept()


def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
