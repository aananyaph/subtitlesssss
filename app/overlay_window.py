"""Desktop Subtitle Overlay Window using PyQt6 / PySide6.

Renders frameless, transparent, click-through, stay-on-top live subtitles
positioned over the bottom/middle/top of the screen.
"""

import sys
import threading
import time
from typing import Optional

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QColor, QFont, QPalette, QScreen
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
    QT_BINDING = "PyQt6"
except ImportError:
    try:
        from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide6.QtGui import QColor, QFont, QPalette, QScreen
        from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
        QT_BINDING = "PySide6"
    except ImportError:
        QT_BINDING = None


class DesktopSubtitleOverlay(QWidget if QT_BINDING else object):
    """Frameless, transparent, click-through, stay-on-top Desktop Subtitle Overlay."""

    # Thread-safe Qt signal for receiving text updates from ASR thread
    update_text_signal = pyqtSignal(str, bool) if QT_BINDING else None

    def __init__(self, font_size_mode: str = "medium", position_mode: str = "bottom"):
        if not QT_BINDING:
            raise RuntimeError("PyQt6 or PySide6 is required for DesktopSubtitleOverlay.")

        super().__init__()

        self.font_size_mode = font_size_mode
        self.position_mode = position_mode
        self.fade_timeout_ms = 8000  # 8 seconds auto-fade (keeps subtitles clearly visible on CPU)

        # Window flags for transparent, click-through, stay-on-top overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # UI Elements
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel("", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)

        self.layout.addWidget(self.label)

        # Connect thread-safe signal
        self.update_text_signal.connect(self._on_update_text)

        # Fade timer
        self.fade_timer = QTimer(self)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(self.hide_text)

        self.apply_style()
        self.reposition()

    def apply_style(self):
        """Applies font size and translucent pill background styling."""
        font_sizes = {"small": 20, "medium": 28, "large": 36}
        padding_sizes = {"small": "6px 16px", "medium": "10px 24px", "large": "14px 32px"}

        px = font_sizes.get(self.font_size_mode, 28)
        padding = padding_sizes.get(self.font_size_mode, "10px 24px")

        style = f"""
            QLabel {{
                background-color: rgba(0, 0, 0, 0.78);
                color: #FFFFFF;
                border-radius: 10px;
                padding: {padding};
                font-family: 'Nirmala UI', 'Segoe UI', Arial, sans-serif;
                font-size: {px}px;
                font-weight: 600;
            }}
        """
        self.label.setStyleSheet(style)

    def reposition(self):
        """Positions the overlay over the active screen based on position_mode (bottom/middle/top)."""
        screen: Optional[QScreen] = QApplication.primaryScreen()
        if not screen:
            return

        geo = screen.availableGeometry()
        width = int(geo.width() * 0.80)
        # Scale height with font size to prevent multi-line text clipping
        font_heights = {"small": 120, "medium": 160, "large": 220}
        height = font_heights.get(self.font_size_mode, 160)

        x = geo.x() + (geo.width() - width) // 2

        if self.position_mode == "top":
            y = geo.y() + int(geo.height() * 0.08)
        elif self.position_mode == "middle":
            y = geo.y() + (geo.height() - height) // 2
        else:  # bottom
            y = geo.y() + int(geo.height() * 0.78)

        self.setGeometry(x, y, width, height)

    def push_live_text(self, text: str, is_final: bool = False):
        """Thread-safe entry point to push live transcript text to the overlay."""
        if QT_BINDING:
            self.update_text_signal.emit(text, is_final)

    @pyqtSlot(str, bool)
    def _on_update_text(self, text: str, is_final: bool):
        """Qt Slot executed on GUI thread to update label text and restart fade timer."""
        clean_text = text.strip()
        if not clean_text:
            self.hide_text()
            return

        self.label.setText(clean_text)
        if not self.label.isVisible():
            self.label.show()
        if not self.isVisible():
            self.show()

        # Restart 4-second auto-fade timer on final or streaming update
        self.fade_timer.start(self.fade_timeout_ms)

    @pyqtSlot()
    def hide_text(self):
        """Hides the subtitle text element."""
        if self.label.isVisible():
            self.label.hide()
        if self.isVisible():
            self.hide()

    def set_font_size(self, mode: str):
        """Updates font size mode ('small', 'medium', 'large')."""
        self.font_size_mode = mode
        self.apply_style()

    def set_position(self, mode: str):
        """Updates screen position mode ('bottom', 'middle', 'top')."""
        self.position_mode = mode
        self.reposition()
