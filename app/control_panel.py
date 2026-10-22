"""Control Center Frontend for Live Subtitle PC Application (PyQt6 / PySide6).

Allows users to:
1. Select Spoken Audio Language (Auto-Detect, English, Hindi, Kannada, Spanish, etc.).
2. Select Target Subtitle Language (English Translate, Same as Spoken, Hindi, Kannada).
3. Automatically routes and selects the optimal ASR model (distil-small.en, base, small, IndicConformer).
4. Select audio input/loopback device (Realtek, Headset, etc.).
5. Adjust overlay font size and position with a live preview test button.
6. Monitor live audio levels (VU meter) and see live subtitle transcripts inside the window.
7. Start, stop, and pause the subtitle engine with instant status feedback.
"""

import logging
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

try:
    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPalette, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFrame,
        QGraphicsDropShadowEffect,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpacerItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PyQt6"
except ImportError:
    try:
        from PySide6.QtCore import QObject, Qt, QTimer, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPalette, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFrame,
            QGraphicsDropShadowEffect,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QProgressBar,
            QPushButton,
            QScrollArea,
            QSizePolicy,
            QSpacerItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_BINDING = "PySide6"
    except ImportError:
        QT_BINDING = None


# ── Language Catalog ─────────────────────────────────────────────────────────

AUDIO_LANGUAGES = [
    ("auto", "🌐 Auto-Detect (Any Language)", "Auto"),
    ("en", "🇺🇸 English", "English"),
    ("hi", "🇮🇳 Hindi (हिन्दी)", "Hindi"),
    ("kn", "🇮🇳 Kannada (ಕನ್ನಡ)", "Kannada"),
    ("es", "🇪🇸 Spanish (Español)", "Spanish"),
    ("fr", "🇫🇷 French (Français)", "French"),
    ("de", "🇩🇪 German (Deutsch)", "German"),
    ("ja", "🇯🇵 Japanese (日本語)", "Japanese"),
    ("zh", "🇨🇳 Chinese (中文)", "Chinese"),
    ("it", "🇮🇹 Italian (Italiano)", "Italian"),
    ("ru", "🇷🇺 Russian (Русский)", "Russian"),
    ("ko", "🇰🇷 Korean (한국어)", "Korean"),
    ("pt", "🇵🇹 Portuguese (Português)", "Portuguese"),
    ("ar", "🇸🇦 Arabic (العربية)", "Arabic"),
    ("ta", "🇮🇳 Tamil (தமிழ்)", "Tamil"),
    ("te", "🇮🇳 Telugu (తెలుగు)", "Telugu"),
]

SUBTITLE_LANGUAGES = [
    ("en", "🇺🇸 English Subtitles (Translate / Transcribe)", "English"),
    ("same", "🎯 Original Spoken Language (Direct Transcription)", "Same"),
    ("hi", "🇮🇳 Hindi Subtitles (हिन्दी Devanagari)", "Hindi"),
    ("kn", "🇮🇳 Kannada Subtitles (ಕನ್ನಡ Script)", "Kannada"),
]

MODEL_CHOICES = [
    ("auto", "✨ Auto-Select (Optimal Engine for Intel CPU)"),
    ("distil-small.en", "⚡ distil-small.en (Ultra-Fast English, CPU INT8)"),
    ("base", "🌐 faster-whisper base (Multilingual, Fast CPU INT8)"),
    ("small", "🎯 faster-whisper small (Multilingual, Accurate CPU INT8)"),
    ("tiny", "🚀 faster-whisper tiny (Lightest & Fastest on CPU)"),
    ("indic", "⚠️ IndicConformer (Requires Nvidia GPU — Freeze warning on CPU)"),
]


def resolve_model_strategy(
    audio_lang: str,
    subtitle_lang: str,
    override_model: str = "auto",
) -> Dict[str, object]:
    """Determines the best model, task, language code, and descriptive badges
    based on the user's chosen Audio and Subtitle languages.
    """
    # 1. Native Indic Transcription (Hindi / Kannada)
    if subtitle_lang in ("hi", "kn"):
        target_lang = subtitle_lang
        lang_name = "Hindi" if target_lang == "hi" else "Kannada"

        if override_model == "indic":
            chosen_model = "indic"
            use_indic = True
            badge = f"🇮🇳 AI4Bharat IndicConformer ({lang_name})"
            desc = f"Heavy 600M neural model. Highest Indic accuracy (Warning: CPU intensive)."
        else:
            chosen_model = "small" if override_model == "auto" else override_model
            use_indic = False
            if chosen_model == "small":
                badge = f"🎯 faster-whisper small (Accurate {lang_name} ASR)"
                desc = f"Transcribes {lang_name} speech with high vocabulary accuracy (~5.9s–7.9s on Intel CPU)."
            elif chosen_model == "base":
                badge = f"⚡ faster-whisper base (Fast {lang_name} ASR)"
                desc = f"Fast live updates (~1.8s–2.2s on CPU) with anti-repetition guards."
            else:
                badge = f"🔧 faster-whisper {chosen_model} ({lang_name} ASR)"
                desc = f"Running {chosen_model} for {lang_name} native transcription."

        return {
            "model_name": chosen_model,
            "indic_lang": target_lang if use_indic else None,
            "asr_language": target_lang,
            "asr_task": "transcribe",
            "badge": badge,
            "description": desc,
            "mode_tag": f"Native {lang_name} Transcription",
        }

    # 2. English audio -> English subtitles
    if audio_lang == "en" and subtitle_lang in ("en", "same"):
        chosen_model = "distil-small.en" if override_model == "auto" else override_model
        return {
            "model_name": chosen_model,
            "indic_lang": None,
            "asr_language": "en",
            "asr_task": "transcribe",
            "badge": f"⚡ {chosen_model} (English Specialist)",
            "description": "Ultra-fast English transcription with Intel CPU INT8 optimization. Lowest latency (< 350ms).",
            "mode_tag": "Direct Transcription",
        }

    # 3. Foreign / Auto audio -> English subtitles (Live Translation)
    if subtitle_lang == "en":
        chosen_model = "base" if override_model == "auto" else override_model
        lang_hint = None if audio_lang == "auto" else audio_lang
        audio_name = next((name for code, name, _ in AUDIO_LANGUAGES if code == audio_lang), audio_lang)
        return {
            "model_name": chosen_model,
            "indic_lang": None,
            "asr_language": lang_hint,
            "asr_task": "translate",
            "badge": f"🌐 {chosen_model} (Live Translation ➔ English)",
            "description": f"Listens to {audio_name} and translates speech into English live subtitles in real time.",
            "mode_tag": "Live Translation",
        }

    # 4. Foreign audio -> Same foreign language (Native Transcription)
    if subtitle_lang == "same":
        chosen_model = "base" if override_model == "auto" else override_model
        lang_hint = None if audio_lang == "auto" else audio_lang
        audio_name = next((name for code, name, _ in AUDIO_LANGUAGES if code == audio_lang), audio_lang)
        return {
            "model_name": chosen_model,
            "indic_lang": None,
            "asr_language": lang_hint,
            "asr_task": "transcribe",
            "badge": f"🎯 {chosen_model} (Transcribe in {audio_lang.upper()})",
            "description": f"Transcribes spoken speech directly into native {audio_name} script.",
            "mode_tag": "Direct Transcription",
        }

    # Fallback default
    chosen_model = "base" if override_model == "auto" else override_model
    return {
        "model_name": chosen_model,
        "indic_lang": None,
        "asr_language": None,
        "asr_task": "translate",
        "badge": f"🌐 {chosen_model} (Multilingual)",
        "description": "Balanced multilingual model translating to English.",
        "mode_tag": "Translate",
    }


# ── Modern Dark Stylesheet ───────────────────────────────────────────────────

DARK_STYLESHEET = """
QWidget {
    background-color: #0B0F19;
    color: #F8FAFC;
    font-family: 'Segoe UI', 'Nirmala UI', Arial, sans-serif;
    font-size: 13px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #0B0F19;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Card Container */
QFrame.card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 14px;
}

QFrame.highlight-card {
    background-color: #131E33;
    border: 1px solid #1E3A8A;
    border-radius: 12px;
    padding: 14px;
}

/* Typography */
QLabel.heading {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel.subheading {
    font-size: 13px;
    font-weight: 600;
    color: #94A3B8;
}

QLabel.card-title {
    font-size: 14px;
    font-weight: 700;
    color: #38BDF8;
}

QLabel.badge {
    background-color: #1E293B;
    color: #38BDF8;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
}

/* Inputs & Combos */
QComboBox {
    background-color: #1F2937;
    color: #F8FAFC;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    min-height: 20px;
}
QComboBox:hover {
    border-color: #38BDF8;
    background-color: #273549;
}
QComboBox:focus {
    border-color: #0284C7;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox QAbstractItemView {
    background-color: #1F2937;
    color: #F8FAFC;
    selection-background-color: #0284C7;
    selection-color: #FFFFFF;
    border: 1px solid #374151;
    border-radius: 6px;
    outline: none;
    padding: 4px;
}

/* Buttons */
QPushButton {
    background-color: #1F2937;
    color: #F8FAFC;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #374151;
    border-color: #4B5563;
}
QPushButton:pressed {
    background-color: #111827;
}

QPushButton.primary-start {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #059669);
    color: #FFFFFF;
    border: none;
    font-size: 15px;
    font-weight: 700;
    border-radius: 10px;
    padding: 13px 24px;
}
QPushButton.primary-start:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34D399, stop:1 #10B981);
}

QPushButton.primary-stop {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EF4444, stop:1 #DC2626);
    color: #FFFFFF;
    border: none;
    font-size: 15px;
    font-weight: 700;
    border-radius: 10px;
    padding: 13px 24px;
}
QPushButton.primary-stop:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F87171, stop:1 #EF4444);
}

/* Audio Level Progress Bar */
QProgressBar.vu-meter {
    background-color: #1F2937;
    border: 1px solid #374151;
    border-radius: 5px;
    height: 10px;
    text-align: center;
}
QProgressBar.vu-meter::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:0.7 #F59E0B, stop:1 #EF4444);
    border-radius: 4px;
}

/* Live Text Box */
QTextEdit.live-box {
    background-color: #060911;
    color: #E2E8F0;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 10px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
    line-height: 1.4;
}
"""


# ── Main Control Panel Window ────────────────────────────────────────────────

class ControlPanelWindow(QMainWindow if QT_BINDING else object):
    """Main Control Center UI for live language selection and subtitle orchestration."""

    start_requested = pyqtSignal(dict)  # emits configuration dict
    stop_requested = pyqtSignal()
    pause_toggled = pyqtSignal(bool)
    overlay_preview_requested = pyqtSignal(str, str)  # (size, position)
    meter_signal = pyqtSignal(float, float, str)
    subtitle_signal = pyqtSignal(str, bool)

    def __init__(
        self,
        current_settings: Optional[Dict[str, object]] = None,
        device_list: Optional[List[Dict[str, object]]] = None,
        parent=None,
    ):
        if not QT_BINDING:
            raise RuntimeError("PyQt6 or PySide6 is required for ControlPanelWindow.")

        super().__init__(parent)
        self.setWindowTitle("🎙 Live Subtitle Studio — Control Center")
        self.resize(650, 780)
        self.setMinimumSize(580, 700)

        self.current_settings = current_settings or {}
        self.device_list = device_list or []
        self.is_running = False
        self.is_paused = False

        # Connect internal cross-thread signals
        self.meter_signal.connect(self.update_audio_meter)
        self.subtitle_signal.connect(self.push_live_subtitle)

        self._build_ui()
        self.setStyleSheet(DARK_STYLESHEET)
        self._on_language_changed()  # update model banner initially

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # ── 1. Header Bar ────────────────────────────────────────────────────
        header = QHBoxLayout()
        header_info = QVBoxLayout()
        title_label = QLabel("🎙 Live Subtitle Studio", self)
        title_label.setProperty("class", "heading")
        subtitle_label = QLabel("Real-time Audio Loopback Capture & Multilingual Subtitles", self)
        subtitle_label.setProperty("class", "subheading")
        header_info.addWidget(title_label)
        header_info.addWidget(subtitle_label)

        self.status_pill = QLabel("● IDLE", self)
        self.status_pill.setStyleSheet(
            "background-color: #1E293B; color: #94A3B8; font-weight: 700; "
            "padding: 6px 14px; border-radius: 12px; border: 1px solid #334155; font-size: 12px;"
        )

        header.addLayout(header_info)
        header.addStretch()
        header.addWidget(self.status_pill)
        main_layout.addLayout(header)

        # Scrollable container for cards
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # ── 2. Language Selection Card (STAR FEATURE) ────────────────────────
        lang_card = QFrame()
        lang_card.setProperty("class", "card")
        lang_card_layout = QVBoxLayout(lang_card)
        lang_card_layout.setSpacing(12)

        card_title = QLabel("🌐 Language & Translation Pairing", lang_card)
        card_title.setProperty("class", "card-title")
        lang_card_layout.addWidget(card_title)

        # Two Column Selector: Spoken Language ➔ Subtitle Language
        pair_layout = QHBoxLayout()
        pair_layout.setSpacing(10)

        # Column Left: Spoken Audio
        left_col = QVBoxLayout()
        left_label = QLabel("Spoken Audio Language (Input):", lang_card)
        left_label.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.combo_audio_lang = QComboBox(lang_card)
        for code, label, _ in AUDIO_LANGUAGES:
            self.combo_audio_lang.addItem(label, code)
        self.combo_audio_lang.currentIndexChanged.connect(self._on_language_changed)
        left_col.addWidget(left_label)
        left_col.addWidget(self.combo_audio_lang)

        # Arrow icon
        arrow_lbl = QLabel("➔", lang_card)
        arrow_lbl.setStyleSheet("font-size: 20px; font-weight: 900; color: #38BDF8; padding-top: 18px;")
        arrow_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Column Right: Subtitle Output
        right_col = QVBoxLayout()
        right_label = QLabel("Live Subtitles Language (Output):", lang_card)
        right_label.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.combo_sub_lang = QComboBox(lang_card)
        for code, label, _ in SUBTITLE_LANGUAGES:
            self.combo_sub_lang.addItem(label, code)
        self.combo_sub_lang.currentIndexChanged.connect(self._on_language_changed)
        right_col.addWidget(right_label)
        right_col.addWidget(self.combo_sub_lang)

        pair_layout.addLayout(left_col, 4)
        pair_layout.addWidget(arrow_lbl, 1)
        pair_layout.addLayout(right_col, 4)
        lang_card_layout.addLayout(pair_layout)

        # ── 3. Dynamic Model Recommendation Box ──────────────────────────────
        self.model_card = QFrame(lang_card)
        self.model_card.setProperty("class", "highlight-card")
        model_layout = QVBoxLayout(self.model_card)
        model_layout.setSpacing(6)

        model_hdr = QHBoxLayout()
        self.lbl_model_badge = QLabel("⚡ distil-small.en", self.model_card)
        self.lbl_model_badge.setStyleSheet(
            "background-color: #0369A1; color: #E0F2FE; font-weight: 700; "
            "padding: 4px 10px; border-radius: 6px; font-size: 12px;"
        )
        self.lbl_mode_tag = QLabel("[Direct Transcription]", self.model_card)
        self.lbl_mode_tag.setStyleSheet("color: #38BDF8; font-weight: 600; font-size: 11px;")
        model_hdr.addWidget(self.lbl_model_badge)
        model_hdr.addWidget(self.lbl_mode_tag)
        model_hdr.addStretch()

        self.lbl_model_desc = QLabel("Loading model routing explanation...", self.model_card)
        self.lbl_model_desc.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        self.lbl_model_desc.setWordWrap(True)

        model_layout.addLayout(model_hdr)
        model_layout.addWidget(self.lbl_model_desc)

        # Advanced Model Override
        override_layout = QHBoxLayout()
        override_lbl = QLabel("Model Engine Override:", self.model_card)
        override_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        self.combo_model_override = QComboBox(self.model_card)
        for code, name in MODEL_CHOICES:
            self.combo_model_override.addItem(name, code)
        self.combo_model_override.currentIndexChanged.connect(self._on_language_changed)
        override_layout.addWidget(override_lbl)
        override_layout.addWidget(self.combo_model_override, 1)
        model_layout.addLayout(override_layout)

        lang_card_layout.addWidget(self.model_card)
        scroll_layout.addWidget(lang_card)

        # ── 4. Audio Device & Subtitle Overlay Card ──────────────────────────
        device_card = QFrame()
        device_card.setProperty("class", "card")
        device_layout = QVBoxLayout(device_card)
        device_layout.setSpacing(10)

        dev_title = QLabel("🔊 Audio Source & Overlay Display", device_card)
        dev_title.setProperty("class", "card-title")
        device_layout.addWidget(dev_title)

        # Audio Device Selector with Refresh
        dev_row = QHBoxLayout()
        dev_lbl = QLabel("Playback Device (Loopback):", device_card)
        dev_lbl.setStyleSheet("font-weight: 600; color: #E2E8F0;")
        self.combo_device = QComboBox(device_card)
        self._populate_devices()

        btn_refresh_dev = QPushButton("🔄", device_card)
        btn_refresh_dev.setToolTip("Refresh Audio Devices")
        btn_refresh_dev.setFixedWidth(36)
        btn_refresh_dev.clicked.connect(self.refresh_devices)

        dev_row.addWidget(dev_lbl)
        dev_row.addWidget(self.combo_device, 1)
        dev_row.addWidget(btn_refresh_dev)
        device_layout.addLayout(dev_row)

        # Overlay Controls: Font Size & Position
        overlay_row = QHBoxLayout()
        size_lbl = QLabel("Font Size:", device_card)
        self.combo_font_size = QComboBox(device_card)
        self.combo_font_size.addItem("Medium (28px)", "medium")
        self.combo_font_size.addItem("Small (20px)", "small")
        self.combo_font_size.addItem("Large (36px)", "large")

        pos_lbl = QLabel("Screen Position:", device_card)
        self.combo_pos = QComboBox(device_card)
        self.combo_pos.addItem("Bottom (Default)", "bottom")
        self.combo_pos.addItem("Top", "top")
        self.combo_pos.addItem("Middle", "middle")

        btn_test_overlay = QPushButton("👁 Test Overlay", device_card)
        btn_test_overlay.clicked.connect(self._test_overlay)

        overlay_row.addWidget(size_lbl)
        overlay_row.addWidget(self.combo_font_size)
        overlay_row.addWidget(pos_lbl)
        overlay_row.addWidget(self.combo_pos)
        overlay_row.addWidget(btn_test_overlay)
        device_layout.addLayout(overlay_row)

        scroll_layout.addWidget(device_card)

        # ── 5. Live Audio Monitor & Subtitle Box ─────────────────────────────
        mon_card = QFrame()
        mon_card.setProperty("class", "card")
        mon_layout = QVBoxLayout(mon_card)
        mon_layout.setSpacing(8)

        mon_title_row = QHBoxLayout()
        mon_title = QLabel("📊 Live Audio Monitor & Subtitle Feed", mon_card)
        mon_title.setProperty("class", "card-title")
        self.lbl_vad_state = QLabel("Audio: Silent", mon_card)
        self.lbl_vad_state.setStyleSheet("color: #94A3B8; font-weight: 600; font-size: 11px;")
        mon_title_row.addWidget(mon_title)
        mon_title_row.addStretch()
        mon_title_row.addWidget(self.lbl_vad_state)
        mon_layout.addLayout(mon_title_row)

        # VU Meter
        self.vu_bar = QProgressBar(mon_card)
        self.vu_bar.setProperty("class", "vu-meter")
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setValue(0)
        self.vu_bar.setTextVisible(False)
        mon_layout.addWidget(self.vu_bar)

        # Live Transcript Text Box
        self.txt_live_subtitles = QTextEdit(mon_card)
        self.txt_live_subtitles.setProperty("class", "live-box")
        self.txt_live_subtitles.setReadOnly(True)
        self.txt_live_subtitles.setPlaceholderText(
            "Live transcribed subtitles will appear here as you play YouTube or PC audio..."
        )
        self.txt_live_subtitles.setFixedHeight(95)
        mon_layout.addWidget(self.txt_live_subtitles)

        scroll_layout.addWidget(mon_card)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # ── 6. Bottom Action Bar ─────────────────────────────────────────────
        action_bar = QHBoxLayout()
        action_bar.setSpacing(12)

        self.btn_toggle_start = QPushButton("▶  Start Live Subtitles", self)
        self.btn_toggle_start.setProperty("class", "primary-start")
        self.btn_toggle_start.clicked.connect(self._on_start_toggle_clicked)

        self.btn_pause = QPushButton("⏸ Pause", self)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_clicked)

        self.btn_minimize = QPushButton("🗕 Hide to Tray", self)
        self.btn_minimize.clicked.connect(self.hide)

        action_bar.addWidget(self.btn_toggle_start, 3)
        action_bar.addWidget(self.btn_pause, 1)
        action_bar.addWidget(self.btn_minimize, 1)
        main_layout.addLayout(action_bar)

    def _on_language_changed(self):
        """Re-evaluates model strategy when user changes audio or subtitle language."""
        audio_lang = self.combo_audio_lang.currentData()
        sub_lang = self.combo_sub_lang.currentData()
        override = self.combo_model_override.currentData()

        strategy = resolve_model_strategy(audio_lang, sub_lang, override)

        self.lbl_model_badge.setText(strategy["badge"])
        self.lbl_mode_tag.setText(f"[{strategy['mode_tag']}]")
        self.lbl_model_desc.setText(str(strategy["description"]))

    def _populate_devices(self):
        self.combo_device.clear()
        self.combo_device.addItem("Auto-Detect Default Playback Device", None)
        for dev in self.device_list:
            idx = dev.get("index")
            name = dev.get("name", "")
            is_def = dev.get("is_default", False)
            is_lb = dev.get("is_loopback", False)
            def_tag = " ⭐ (Default)" if is_def else ""
            lb_tag = " [Loopback]" if is_lb else ""
            self.combo_device.addItem(f"#{idx}: {name}{def_tag}{lb_tag}", idx)

    def refresh_devices(self):
        try:
            from app.loopback_capture import get_loopback_devices
            self.device_list = get_loopback_devices()
            self._populate_devices()
            logging.info("[Frontend] Audio devices refreshed.")
        except Exception as e:
            logging.error(f"[Frontend] Failed refreshing devices: {e}")

    def _test_overlay(self):
        size = self.combo_font_size.currentData()
        pos = self.combo_pos.currentData()
        self.overlay_preview_requested.emit(size, pos)

    def _on_start_toggle_clicked(self):
        if not self.is_running:
            # Gather configuration
            audio_lang = self.combo_audio_lang.currentData()
            sub_lang = self.combo_sub_lang.currentData()
            override = self.combo_model_override.currentData()
            strategy = resolve_model_strategy(audio_lang, sub_lang, override)

            config = {
                "model_name": strategy["model_name"],
                "indic_lang": strategy["indic_lang"],
                "asr_language": strategy["asr_language"],
                "asr_task": strategy["asr_task"],
                "device_index": self.combo_device.currentData(),
                "font_size": self.combo_font_size.currentData(),
                "position": self.combo_pos.currentData(),
                "audio_lang_code": audio_lang,
                "subtitle_lang_code": sub_lang,
            }

            self.is_running = True
            self.is_paused = False
            self.btn_toggle_start.setText("⏹  Stop Live Subtitles")
            self.btn_toggle_start.setProperty("class", "primary-stop")
            self.btn_toggle_start.style().unpolish(self.btn_toggle_start)
            self.btn_toggle_start.style().polish(self.btn_toggle_start)

            self.btn_pause.setEnabled(True)
            self.btn_pause.setText("⏸ Pause")

            self.status_pill.setText("● ACTIVE: LISTENING")
            self.status_pill.setStyleSheet(
                "background-color: #064E3B; color: #34D399; font-weight: 700; "
                "padding: 6px 14px; border-radius: 12px; border: 1px solid #059669; font-size: 12px;"
            )

            # Lock inputs while active
            self.combo_audio_lang.setEnabled(False)
            self.combo_sub_lang.setEnabled(False)
            self.combo_model_override.setEnabled(False)
            self.combo_device.setEnabled(False)

            self.start_requested.emit(config)
        else:
            self.is_running = False
            self.is_paused = False
            self.btn_toggle_start.setText("▶  Start Live Subtitles")
            self.btn_toggle_start.setProperty("class", "primary-start")
            self.btn_toggle_start.style().unpolish(self.btn_toggle_start)
            self.btn_toggle_start.style().polish(self.btn_toggle_start)

            self.btn_pause.setEnabled(False)
            self.vu_bar.setValue(0)
            self.lbl_vad_state.setText("Audio: Stopped")

            self.status_pill.setText("● IDLE")
            self.status_pill.setStyleSheet(
                "background-color: #1E293B; color: #94A3B8; font-weight: 700; "
                "padding: 6px 14px; border-radius: 12px; border: 1px solid #334155; font-size: 12px;"
            )

            # Unlock inputs
            self.combo_audio_lang.setEnabled(True)
            self.combo_sub_lang.setEnabled(True)
            self.combo_model_override.setEnabled(True)
            self.combo_device.setEnabled(True)

            self.stop_requested.emit()

    def _on_pause_clicked(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.setText("▶ Resume")
            self.status_pill.setText("● PAUSED")
            self.status_pill.setStyleSheet(
                "background-color: #78350F; color: #FCD34D; font-weight: 700; "
                "padding: 6px 14px; border-radius: 12px; border: 1px solid #D97706; font-size: 12px;"
            )
        else:
            self.btn_pause.setText("⏸ Pause")
            self.status_pill.setText("● ACTIVE: LISTENING")
            self.status_pill.setStyleSheet(
                "background-color: #064E3B; color: #34D399; font-weight: 700; "
                "padding: 6px 14px; border-radius: 12px; border: 1px solid #059669; font-size: 12px;"
            )
        self.pause_toggled.emit(not self.is_paused)

    @pyqtSlot(float, float, str)
    def update_audio_meter(self, peak: float, prob: float, state: str):
        """Thread-safe slot to update visual volume meter and speech indicator."""
        if not self.is_running or self.is_paused:
            return
        level = min(100, int(peak * 180))  # scale for visual clarity
        self.vu_bar.setValue(level)
        if state == "SPEECH":
            self.lbl_vad_state.setText(f"🗣️ Speech Detected ({prob:.0%})")
            self.lbl_vad_state.setStyleSheet("color: #34D399; font-weight: 700;")
        else:
            self.lbl_vad_state.setText(f"Audio Level: {level}%")
            self.lbl_vad_state.setStyleSheet("color: #94A3B8; font-weight: 500;")

    @pyqtSlot(str, bool)
    def push_live_subtitle(self, text: str, is_final: bool):
        """Displays latest recognized subtitle in the GUI text box."""
        clean = text.strip()
        if not clean:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.txt_live_subtitles.append(f"[{timestamp}]  {clean}")
        # Scroll to bottom
        sb = self.txt_live_subtitles.verticalScrollBar()
        sb.setValue(sb.maximum())
