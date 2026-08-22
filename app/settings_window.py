"""Settings Window for the Live Subtitle PC App (PyQt6 / PySide6).

A small modal dialog for editing persistent settings. Model, font size and
position apply immediately; other fields are saved and applied on next launch.
"""

from typing import Callable, Dict, Optional

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )
    QT_BINDING = "PyQt6"
except ImportError:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox,
            QDialog,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
        )
        QT_BINDING = "PySide6"
    except ImportError:
        QT_BINDING = None


class SettingsWindow(QDialog if QT_BINDING else object):
    """Modal settings dialog with immediate-apply for display options."""

    def __init__(
        self,
        settings: Dict[str, object],
        on_saved: Optional[Callable[[Dict[str, object]], None]] = None,
        parent=None,
    ):
        if not QT_BINDING:
            raise RuntimeError("PyQt6 or PySide6 is required for SettingsWindow.")

        super().__init__(parent)
        self.setWindowTitle("Live Subtitle Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._settings = dict(settings)
        self._on_saved = on_saved

        form = QFormLayout()

        self.model_combo = QComboBox(self)
        self.model_combo.addItem("distil-small.en (Recommended)", "distil-small.en")
        self.model_combo.addItem("tiny.en (Fastest)", "tiny.en")
        self.model_combo.addItem("small.en (More Accurate)", "small.en")
        idx = self.model_combo.findData(self._settings.get("model"))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.currentIndexChanged.connect(self._apply_display_settings)
        form.addRow("ASR Model", self.model_combo)

        self.font_combo = QComboBox(self)
        self.font_combo.addItem("Small (20px)", "small")
        self.font_combo.addItem("Medium (28px)", "medium")
        self.font_combo.addItem("Large (36px)", "large")
        idx = self.font_combo.findData(self._settings.get("font_size"))
        if idx >= 0:
            self.font_combo.setCurrentIndex(idx)
        self.font_combo.currentIndexChanged.connect(self._apply_display_settings)
        form.addRow("Font Size", self.font_combo)

        self.position_combo = QComboBox(self)
        self.position_combo.addItem("Bottom", "bottom")
        self.position_combo.addItem("Middle", "middle")
        self.position_combo.addItem("Top", "top")
        idx = self.position_combo.findData(self._settings.get("position"))
        if idx >= 0:
            self.position_combo.setCurrentIndex(idx)
        self.position_combo.currentIndexChanged.connect(self._apply_display_settings)
        form.addRow("Subtitle Position", self.position_combo)

        self.chunk_floor_spin = QDoubleSpinBox(self)
        self.chunk_floor_spin.setRange(0.5, 3.0)
        self.chunk_floor_spin.setSingleStep(0.25)
        self.chunk_floor_spin.setValue(float(self._settings.get("min_chunk_floor_sec", 1.0)))
        form.addRow("Min Chunk Floor (s)", self.chunk_floor_spin)

        self.stale_timeout_spin = QDoubleSpinBox(self)
        self.stale_timeout_spin.setRange(0.5, 10.0)
        self.stale_timeout_spin.setSingleStep(0.5)
        self.stale_timeout_spin.setValue(float(self._settings.get("stale_audio_timeout_sec", 2.0)))
        form.addRow("Stale Audio Timeout (s)", self.stale_timeout_spin)

        restart_note = QLabel("Chunk floor & staleness timeout apply on next launch.", self)
        restart_note.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(restart_note)

        self.save_btn = QPushButton("Save Settings", self)
        self.save_btn.clicked.connect(self._on_save)
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)

        self._apply_display_settings()

    def _collect(self) -> Dict[str, object]:
        return {
            "model": self.model_combo.currentData(),
            "font_size": self.font_combo.currentData(),
            "position": self.position_combo.currentData(),
            "min_chunk_floor_sec": self.chunk_floor_spin.value(),
            "stale_audio_timeout_sec": self.stale_timeout_spin.value(),
            "device_index": self._settings.get("device_index"),
        }

    def _apply_display_settings(self):
        """Immediately applies model / font size / position callbacks if provided."""
        if self._on_saved is None:
            return
        try:
            self._on_saved(self._collect())
        except Exception:
            pass

    def _on_save(self):
        settings = self._collect()
        if self._on_saved is not None:
            try:
                self._on_saved(settings)
            except Exception:
                pass
        self.accept()
