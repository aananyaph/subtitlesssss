"""Desktop Subtitle Overlay Window using PyQt6 / PySide6.

Renders frameless, transparent, click-through, stay-on-top live subtitles
positioned over the bottom/middle/top of the screen. Supports bilingual
display (original source text small on top, translated text large below).
"""
import html
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


def wrap_text(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    """Wraps text into lines with a maximum character count and line ceiling."""
    if not text:
        return ""
    words = text.strip().split()
    if not words:
        return ""
    lines = []
    current_line = []
    current_len = 0
    for w in words:
        w_len = len(w)
        space_len = 1 if current_line else 0
        if current_line and (current_len + space_len + w_len > max_chars):
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = w_len
        else:
            current_line.append(w)
            current_len += space_len + w_len
    if current_line:
        lines.append(" ".join(current_line))
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


class DesktopSubtitleOverlay(QWidget if QT_BINDING else object):
    """Frameless, transparent, click-through, stay-on-top Desktop Subtitle Overlay."""

    # Thread-safe Qt signal supporting 2-arg (legacy) or 3-arg (bilingual) emissions
    update_text_signal = (
        pyqtSignal([str, bool], [str, bool, str]) if QT_BINDING else None
    )

    def __init__(
        self,
        font_size_mode: str = "medium",
        position_mode: str = "bottom",
        show_source: bool = False,
    ):
        if not QT_BINDING:
            raise RuntimeError("PyQt6 or PySide6 is required for DesktopSubtitleOverlay.")

        super().__init__()

        self.font_size_mode = font_size_mode
        self.position_mode = position_mode
        self.show_source = bool(show_source)
        self.fade_timeout_ms = 8000  # 8 seconds auto-fade

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
        self.update_text_signal[str, bool, str].connect(self._on_update_text_with_source)
        self.update_text_signal[str, bool].connect(self._on_update_text)

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
        font_heights = {"small": 130, "medium": 170, "large": 230}
        height = font_heights.get(self.font_size_mode, 170)

        x = geo.x() + (geo.width() - width) // 2

        if self.position_mode == "top":
            y = geo.y() + int(geo.height() * 0.08)
        elif self.position_mode == "middle":
            y = geo.y() + (geo.height() - height) // 2
        else:  # bottom
            y = geo.y() + int(geo.height() * 0.78)

        self.setGeometry(x, y, width, height)

    def push_live_text(self, text: str, is_final: bool = False, source_text: Optional[str] = None):
        """Thread-safe entry point to push live transcript text to the overlay."""
        if QT_BINDING:
            self.update_text_signal[str, bool, str].emit(text or "", is_final, source_text or "")

    @pyqtSlot(str, bool)
    def _on_update_text(self, text: str, is_final: bool):
        self._on_update_text_with_source(text, is_final, "")

    @pyqtSlot(str, bool, str)
    def _on_update_text_with_source(self, text: str, is_final: bool, source_text: str = ""):
        """Qt Slot executed on GUI thread to update label text and restart fade timer."""
        clean_text = (text or "").strip()
        clean_source = (source_text or "").strip()

        if not clean_text and not clean_source:
            self.hide_text()
            return

        font_sizes = {"small": 20, "medium": 28, "large": 36}
        source_font_sizes = {"small": 14, "medium": 18, "large": 22}

        large_px = font_sizes.get(self.font_size_mode, 28)
        small_px = source_font_sizes.get(self.font_size_mode, 18)

        wrapped_text = wrap_text(clean_text, max_chars=42, max_lines=2)

        # Bilingual display: source_text small on top, text large below
        if self.show_source and clean_source and clean_source != clean_text:
            wrapped_source = wrap_text(clean_source, max_chars=42, max_lines=2)
            esc_source = html.escape(wrapped_source).replace("\n", "<br>")
            esc_text = html.escape(wrapped_text).replace("\n", "<br>")
            html_content = (
                f'<div align="center">'
                f'<div style="font-size: {small_px}px; color: #CBD5E1; font-weight: 400; line-height: 1.2; margin-bottom: 4px;">{esc_source}</div>'
                f'<div style="font-size: {large_px}px; color: #FFFFFF; font-weight: 600; line-height: 1.25;">{esc_text}</div>'
                f'</div>'
            )
            self.label.setText(html_content)
        else:
            esc_text = html.escape(wrapped_text).replace("\n", "<br>")
            html_content = (
                f'<div align="center">'
                f'<div style="font-size: {large_px}px; color: #FFFFFF; font-weight: 600; line-height: 1.25;">{esc_text}</div>'
                f'</div>'
            )
            self.label.setText(html_content)

        if not self.label.isVisible():
            self.label.show()
        if not self.isVisible():
            self.show()

        # Restart auto-fade timer on final or partial update
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

    def set_show_source(self, enable: bool):
        """Enables or disables showing the source language text on top."""
        self.show_source = bool(enable)
