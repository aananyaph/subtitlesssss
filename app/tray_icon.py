"""System Tray Menu and Settings Controller using PyQt6 / PySide6.

Provides QSystemTrayIcon with Start/Pause capture toggle, ASR Model selector,
Font Size selector, Subtitle Position selector, and Quit option.
"""

import sys
from typing import Callable, Optional

try:
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QColor
    from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
    QT_BINDING = "PyQt6"
except ImportError:
    try:
        from PySide6.QtCore import QObject, Signal as pyqtSignal
        from PySide6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QColor
        from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
        QT_BINDING = "PySide6"
    except ImportError:
        QT_BINDING = None


def create_default_icon(color_hex: str = "#007ACC") -> QIcon:
    """Creates a simple colored square QIcon for system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(color_hex))
    return QIcon(pixmap)


class SystemTrayController(QObject if QT_BINDING else object):
    """System Tray Icon controller for Desktop Live Subtitles."""

    def __init__(
        self,
        on_toggle_capture: Callable[[bool], None],
        on_change_model: Callable[[str], None],
        on_change_font_size: Callable[[str], None],
        on_change_position: Callable[[str], None],
        on_open_settings: Optional[Callable[[], None]] = None,
        on_quit: Callable[[], None] = None,
    ):
        if not QT_BINDING:
            raise RuntimeError("PyQt6 or PySide6 is required for SystemTrayController.")

        super().__init__()

        self.on_toggle_capture = on_toggle_capture
        self.on_change_model = on_change_model
        self.on_change_font_size = on_change_font_size
        self.on_change_position = on_change_position
        self.on_open_settings = on_open_settings
        self.on_quit = on_quit

        self.is_capturing = True

        # Initialize QSystemTrayIcon
        self.tray = QSystemTrayIcon(create_default_icon("#007ACC"), parent=QApplication.instance())
        self.tray.setToolTip("Live Subtitle PC App — Running")

        self._build_menu()
        self.tray.show()

    def _build_menu(self):
        menu = QMenu()

        # 1. Toggle Capture Action
        self.toggle_action = QAction("Pause Capturing", menu)
        self.toggle_action.triggered.connect(self._handle_toggle)
        menu.addAction(self.toggle_action)

        menu.addSeparator()

        # 2. Model Selection Submenu
        model_menu = menu.addMenu("ASR Model")
        model_group = QActionGroup(menu)
        model_group.setExclusive(True)

        m_distil = QAction("distil-small.en (Recommended)", model_menu, checkable=True, checked=True)
        m_distil.triggered.connect(lambda: self.on_change_model("distil-small.en"))
        model_group.addAction(m_distil)
        model_menu.addAction(m_distil)

        m_tiny = QAction("tiny.en (Fastest)", model_menu, checkable=True)
        m_tiny.triggered.connect(lambda: self.on_change_model("tiny.en"))
        model_group.addAction(m_tiny)
        model_menu.addAction(m_tiny)

        # 3. Font Size Submenu
        font_menu = menu.addMenu("Font Size")
        font_group = QActionGroup(menu)
        font_group.setExclusive(True)

        f_small = QAction("Small (20px)", font_menu, checkable=True)
        f_small.triggered.connect(lambda: self.on_change_font_size("small"))
        font_group.addAction(f_small)
        font_menu.addAction(f_small)

        f_med = QAction("Medium (28px)", font_menu, checkable=True, checked=True)
        f_med.triggered.connect(lambda: self.on_change_font_size("medium"))
        font_group.addAction(f_med)
        font_menu.addAction(f_med)

        f_large = QAction("Large (36px)", font_menu, checkable=True)
        f_large.triggered.connect(lambda: self.on_change_font_size("large"))
        font_group.addAction(f_large)
        font_menu.addAction(f_large)

        # 4. Position Submenu
        pos_menu = menu.addMenu("Position")
        pos_group = QActionGroup(menu)
        pos_group.setExclusive(True)

        p_bot = QAction("Bottom", pos_menu, checkable=True, checked=True)
        p_bot.triggered.connect(lambda: self.on_change_position("bottom"))
        pos_group.addAction(p_bot)
        pos_menu.addAction(p_bot)

        p_mid = QAction("Middle", pos_menu, checkable=True)
        p_mid.triggered.connect(lambda: self.on_change_position("middle"))
        pos_group.addAction(p_mid)
        pos_menu.addAction(p_mid)

        p_top = QAction("Top", pos_menu, checkable=True)
        p_top.triggered.connect(lambda: self.on_change_position("top"))
        pos_group.addAction(p_top)
        pos_menu.addAction(p_top)

        menu.addSeparator()

        # 5. Settings Window Action
        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(self._handle_open_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        # 6. Quit Action
        quit_action = QAction("Exit Application", menu)
        quit_action.triggered.connect(self.on_quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)

    def _handle_toggle(self):
        self.is_capturing = not self.is_capturing
        if self.is_capturing:
            self.toggle_action.setText("Pause Capturing")
            self.tray.setIcon(create_default_icon("#007ACC"))
            self.tray.setToolTip("Live Subtitle PC App — Capturing")
        else:
            self.toggle_action.setText("Resume Capturing")
            self.tray.setIcon(create_default_icon("#888888"))
            self.tray.setToolTip("Live Subtitle PC App — Paused")
        self.on_toggle_capture(self.is_capturing)

    def _handle_open_settings(self):
        if self.on_open_settings:
            self.on_open_settings()
