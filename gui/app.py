"""Fenêtre principale (PySide6)."""
import os
import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QPushButton,
                               QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

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

VERSION = "0.8.7"

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
        self.setWindowTitle(f"DofusAuto {VERSION} : combats en boucle")
        self.resize(1000, 780)
        self.cfg = load_config(config_path())
        self.ctx = Context(self.cfg, WindowCapture(self.cfg.window_title_contains))
        self.worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # --- En haut : monstres chargés --------------------------------
        self.info = QLabel()
        self.info.setWordWrap(True)
        root.addWidget(self.info)

        # --- Aperçu ----------------------------------------------------
        self.preview = QLabel("L'aperçu de ce que le programme voit apparaîtra ici.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 300)
        self.preview.setStyleSheet("background:#222;color:#aaa;")
        root.addWidget(self.preview, 1)

        # --- Onglets ---------------------------------------------------
        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        root.addWidget(tabs)

        # Onglet « Combat »
        tab_combat = QWidget()
        lay = QHBoxLayout(tab_combat)
        col = QVBoxLayout()
        self.btn_start = QPushButton("▶  Démarrer")
        self.btn_stop = QPushButton("■  Arrêter (F12)")
        for b in (self.btn_start, self.btn_stop):
            b.setMinimumSize(170, 42)
            f = b.font()
            f.setPointSize(f.pointSize() + 2)
            f.setBold(True)
            b.setFont(f)
            col.addWidget(b)
        col.addStretch(1)
        lay.addLayout(col)

        steps = QGroupBox("Étapes du combat")
        grid = QGridLayout(steps)
        self.chk_combat = QCheckBox("Phase 2 : attendre le combat")
        self.chk_combat.setChecked(self.cfg.detect_combat)
        self.chk_combat.toggled.connect(lambda v: setattr(self.cfg, "detect_combat", bool(v)))
        grid.addWidget(self.chk_combat, 0, 0)
        self.chk_place = QCheckBox("Phase 3 : placement")
        self.chk_place.setChecked(self.cfg.do_placement)
        self.chk_place.toggled.connect(lambda v: setattr(self.cfg, "do_placement", bool(v)))
        grid.addWidget(self.chk_place, 1, 0)
        self.rule = QComboBox()
        for key, label in RULES.items():
            self.rule.addItem(label, key)
        idx = self.rule.findData(self.cfg.placement_rule)
        self.rule.setCurrentIndex(idx if idx >= 0 else 0)
        self.rule.currentIndexChanged.connect(lambda _: setattr(self.cfg, "placement_rule", self.rule.currentData()))
        grid.addWidget(self.rule, 1, 1)
        self.chk_turns = QCheckBox("Phases 4-5 : jouer les tours")
        self.chk_turns.setChecked(self.cfg.do_turns)
        self.chk_turns.toggled.connect(lambda v: setattr(self.cfg, "do_turns", bool(v)))
        grid.addWidget(self.chk_turns, 2, 0)
        casts_row = QHBoxLayout()
        casts_row.addWidget(QLabel("Lancers par tour :"))
        self.casts = QSpinBox()
        self.casts.setRange(1, 3)
        self.casts.setValue(self.cfg.casts_per_turn)
        self.casts.valueChanged.connect(lambda v: setattr(self.cfg, "casts_per_turn", int(v)))
        casts_row.addWidget(self.casts)
        casts_row.addStretch(1)
        grid.addLayout(casts_row, 2, 1)
        self.chk_loop = QCheckBox("Phase 6 : enchaîner les combats")
        self.chk_loop.setChecked(self.cfg.loop_combats)
        self.chk_loop.toggled.connect(lambda v: setattr(self.cfg, "loop_combats", bool(v)))
        grid.addWidget(self.chk_loop, 3, 0)
        self.chk_open = QCheckBox("Sort d'ouverture (1er tour seulement)")
        self.chk_open.setChecked(self.cfg.opening_spell)
        self.chk_open.toggled.connect(lambda v: setattr(self.cfg, "opening_spell", bool(v)))
        grid.addWidget(self.chk_open, 4, 0)
        self.chk_heal = QCheckBox("Contrôler la vie avant chaque combat")
        self.chk_heal.setChecked(self.cfg.heal_enabled)
        self.chk_heal.toggled.connect(lambda v: setattr(self.cfg, "heal_enabled", bool(v)))
        grid.addWidget(self.chk_heal, 5, 0)
        heal_row = QHBoxLayout()
        heal_row.addWidget(QLabel("Vie minimale :"))
        self.min_hp = QSpinBox()
        self.min_hp.setRange(1, 100)
        self.min_hp.setSingleStep(5)
        self.min_hp.setSuffix(" %")
        self.min_hp.setValue(self.cfg.min_hp_percent)
        self.min_hp.valueChanged.connect(lambda v: setattr(self.cfg, "min_hp_percent", int(v)))
        heal_row.addWidget(self.min_hp)
        heal_row.addStretch(1)
        grid.addLayout(heal_row, 5, 1)
        pause_row = QHBoxLayout()
        pause_row.addWidget(QLabel("Pause entre les attaques :"))
        self.pause = QSpinBox()
        self.pause.setRange(0, 5000)
        self.pause.setSingleStep(100)
        self.pause.setSuffix(" ms")
        self.pause.setValue(self.cfg.pause_between_casts_ms)
        self.pause.valueChanged.connect(lambda v: setattr(self.cfg, "pause_between_casts_ms", int(v)))
        pause_row.addWidget(self.pause)
        pause_row.addStretch(1)
        grid.addLayout(pause_row, 3, 1)
        grid.setColumnStretch(2, 1)
        lay.addWidget(steps, 1)
        tabs.addTab(tab_combat, "Combat")

        # Onglet « Tests »
        tab_tests = QWidget()
        lay = QHBoxLayout(tab_tests)
        box = QGroupBox("Tester sans cliquer (DofusAuto se réduit pendant le test)")
        grid = QGridLayout(box)
        self.btn_test = QPushButton("Monstres")
        self.btn_testui = QPushButton("Boutons du jeu")
        self.btn_testplace = QPushButton("Placement")
        self.btn_testcombat = QPushButton("Combat")
        for i, b in enumerate((self.btn_test, self.btn_testui, self.btn_testplace, self.btn_testcombat)):
            b.setMinimumHeight(32)
            grid.addWidget(b, i // 2, i % 2)
        lay.addWidget(box, 2)
        box = QGroupBox("Enregistrer ce que voit le programme")
        v = QVBoxLayout(box)
        self.btn_snap = QPushButton("Une capture")
        self.btn_seq = QPushButton("Une séquence (1 image par seconde)")
        for b in (self.btn_snap, self.btn_seq):
            b.setMinimumHeight(32)
            v.addWidget(b)
        lay.addWidget(box, 1)
        tabs.addTab(tab_tests, "Tests")

        # Onglet « Réglages »
        tab_set = QWidget()
        lay = QHBoxLayout(tab_set)
        box = QGroupBox("Détection et clics")
        form = QFormLayout(box)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.50, 0.99)
        self.spin.setSingleStep(0.01)
        self.spin.setDecimals(2)
        self.spin.setValue(self.cfg.threshold)
        self.spin.valueChanged.connect(self._threshold_changed)
        form.addRow("Seuil de confiance :", self.spin)
        self.clicks = QSpinBox()
        self.clicks.setRange(1, 3)
        self.clicks.setValue(self.cfg.click_count)
        self.clicks.valueChanged.connect(lambda v: setattr(self.cfg, "click_count", int(v)))
        form.addRow("Clics sur le monstre :", self.clicks)
        lay.addWidget(box, 1)
        box = QGroupBox("Images et dossiers")
        grid = QGridLayout(box)
        self.btn_reload = QPushButton("Recharger les images")
        self.btn_imgs = QPushButton("Dossier des images")
        self.btn_logs = QPushButton("Dossier des journaux")
        self.btn_caps = QPushButton("Dossier des captures")
        for i, b in enumerate((self.btn_reload, self.btn_imgs, self.btn_logs, self.btn_caps)):
            b.setMinimumHeight(30)
            grid.addWidget(b, i // 2, i % 2)
        lay.addWidget(box, 1)
        tabs.addTab(tab_set, "Réglages")

        # --- Journal ---------------------------------------------------
        root.addWidget(QLabel("Journal :"))
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        self.logbox.setMaximumHeight(150)
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

        self.log(f"DofusAuto {VERSION} (copie sangliers uniquement : lecture de la portée du sort corrigée)")
        self._reload_templates()
        self._set_running(False)
        QTimer.singleShot(0, lambda: self.log("Prêt. Place BlueStacks bien visible (sans le recouvrir), "
                                              "puis utilise l'onglet « Tests » ou « Démarrer »."))

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
        self.info.setText(f"Monstres chargés : {txt or 'aucun : ajoute des images dans le dossier des images'}")
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
