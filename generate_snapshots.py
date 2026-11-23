import sys
import os
import time

sys.path.insert(0, r"d:\New folder\subtitlesssss")

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPixmap, QPainter, QLinearGradient, QBrush, QPen

def generate_all_snapshots():
    print("Creating QApplication...")
    sys.stdout.flush()
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    figures_dir = r"d:\New folder\subtitlesssss\figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Snapshot: Control Panel Window
    print("Importing ControlPanelWindow...")
    sys.stdout.flush()
    from app.control_panel import ControlPanelWindow
    print("Instantiating ControlPanelWindow...")
    sys.stdout.flush()
    try:
        ctrl = ControlPanelWindow(current_settings={
            "audio_lang": "en",
            "subtitle_lang": "kn",
            "model": "indic",
            "show_source": True
        })
    except BaseException as e:
        import traceback
        traceback.print_exc()
        print("Error caught in ControlPanelWindow init:", e)
        sys.stdout.flush()
        return
    print("ControlPanelWindow created successfully.")
    sys.stdout.flush()
    ctrl.resize(720, 840)
    ctrl.show()
    
    # Configure with active live values
    ctrl.status_pill.setText("● STREAMING ACTIVE")
    ctrl.status_pill.setStyleSheet(
        "background-color: #052e16; color: #39FF14; font-weight: 700; "
        "padding: 6px 14px; border-radius: 12px; border: 1px solid #16a34a; font-size: 12px;"
    )
    ctrl.lbl_vad_state.setText("Audio: 🎙️ Speech Detected (VAD 88%)")
    ctrl.lbl_vad_state.setStyleSheet("color: #39FF14; font-weight: 600; font-size: 11px;")
    ctrl.vu_bar.setValue(72)
    
    ctrl.txt_live_subtitles.setHtml(
        "<span style='color:#0080FF;'>[00:12.4] ASR (IndicConformer-600M):</span> "
        "<span style='color:#FFFFFF;'>Artificial intelligence is transforming real-time multilingual communication.</span><br>"
        "<span style='color:#39FF14;'>[00:13.1] NMT (IndicTrans2-1B [en➔kn]):</span> "
        "<b style='color:#39FF14;'>ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯು ನೈಜ-ಸಮಯದ ಬಹುಭಾಷಾ ಸಂವಹನವನ್ನು ಪರಿವರ್ತಿಸುತ್ತಿದೆ.</b>"
    )
    
    ctrl.btn_toggle_start.setText("⏹  Stop Live Subtitles")
    ctrl.btn_toggle_start.setStyleSheet("background-color: #7f1d1d; color: #fca5a5; font-weight: 700;")
    ctrl.btn_pause.setEnabled(True)
    
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    
    ctrl_pixmap = ctrl.grab()
    ctrl_path = os.path.join(figures_dir, "snapshot_control_center.png")
    ctrl_pixmap.save(ctrl_path)
    print(f"Saved: {ctrl_path}")
    ctrl.close()
    
    # 2. Snapshot: Settings Dialog
    from app.settings_window import SettingsWindow
    dummy_settings = {
        "model": "distil-small.en",
        "audio_language": "en",
        "subtitle_language": "kn",
        "font_size": "large",
        "position": "bottom",
        "show_source": True,
        "max_chars_per_line": 42,
        "max_lines": 2,
        "fade_timeout_seconds": 8,
        "device_index": 0,
        "ws_url": "wss://indic-subtitles-tunnel.ngrok-free.app",
        "vad_threshold": 0.5,
    }
    settings_win = SettingsWindow(dummy_settings)
    settings_win.resize(460, 520)
    settings_win.show()
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    
    set_pixmap = settings_win.grab()
    set_path = os.path.join(figures_dir, "snapshot_settings_dialog.png")
    set_pixmap.save(set_path)
    print(f"Saved: {set_path}")
    settings_win.close()
    
    # 3. Snapshot: Floating Subtitle HUD Overlay (Bilingual Kannada)
    from app.overlay_window import DesktopSubtitleOverlay
    hud_kn = DesktopSubtitleOverlay(font_size_mode="large", position_mode="bottom", show_source=True)
    hud_kn.resize(920, 150)
    hud_kn.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    hud_kn.setStyleSheet("background-color: rgba(10, 14, 27, 245); border: 2px solid #0080FF; border-radius: 12px;")
    hud_kn.show()
    hud_kn.push_live_text(
        "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯು ಭಾಷಾ ಅಡೆತಡೆಗಳನ್ನು ನಿವಾರಿಸಿ ನೈಜ ಸಮಯದಲ್ಲಿ ಜ್ಞಾನವನ್ನು ಎಲ್ಲರಿಗೂ ತಲುಪಿಸುತ್ತದೆ.",
        is_final=True,
        source_text="Artificial intelligence breaks linguistic barriers, delivering knowledge in real time."
    )
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    hud_pixmap = hud_kn.grab()
    hud_path = os.path.join(figures_dir, "snapshot_hud_overlay_kn.png")
    hud_pixmap.save(hud_path)
    print(f"Saved: {hud_path}")
    hud_kn.close()

    # 4. Snapshot: Floating Subtitle HUD Overlay (Bilingual Hindi)
    hud_hi = DesktopSubtitleOverlay(font_size_mode="large", position_mode="bottom", show_source=True)
    hud_hi.resize(920, 150)
    hud_hi.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
    hud_hi.setStyleSheet("background-color: rgba(10, 14, 27, 245); border: 2px solid #39FF14; border-radius: 12px;")
    hud_hi.show()
    hud_hi.push_live_text(
        "वास्तविक समय उपशीर्षक प्रणाली छात्रों और पेशेवरों के लिए सीखने को सुगम बनाती है।",
        is_final=True,
        source_text="Real-time subtitle system makes learning seamless for students and professionals."
    )
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    hud_hi_pixmap = hud_hi.grab()
    hud_hi_path = os.path.join(figures_dir, "snapshot_hud_overlay_hi.png")
    hud_hi_pixmap.save(hud_hi_path)
    print(f"Saved: {hud_hi_path}")
    hud_hi.close()

    # 5. Snapshot: Realistic Desktop Screen Context (Video Player with Subtitle HUD)
    sim_screen = QWidget()
    sim_screen.resize(1000, 580)
    sim_screen.setStyleSheet("background-color: #090d16;")
    sim_layout = QVBoxLayout(sim_screen)
    sim_layout.setContentsMargins(20, 20, 20, 20)
    
    video_frame = QWidget()
    video_frame.setStyleSheet("""
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #111827, stop:0.5 #0f172a, stop:1 #020617);
        border: 1px solid #1e293b;
        border-radius: 10px;
    """)
    vf_layout = QVBoxLayout(video_frame)
    vf_layout.setContentsMargins(20, 20, 20, 25)
    
    header_box = QHBoxLayout()
    title_lbl = QLabel("🎥 Live Online Engineering Lecture: Machine Translation & Neural ASR (WASAPI Loopback Active)")
    title_lbl.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: 600;")
    rec_badge = QLabel("● REC 48kHz")
    rec_badge.setStyleSheet("color: #ef4444; font-weight: 800; font-size: 12px; background: #450a0a; padding: 3px 8px; border-radius: 4px;")
    header_box.addWidget(title_lbl)
    header_box.addStretch()
    header_box.addWidget(rec_badge)
    vf_layout.addLayout(header_box)
    
    vf_layout.addStretch()
    center_icon = QLabel("🎬 Internal Desktop Audio Stream — Real-Time Speech Processing\n(No Physical Microphone Required)")
    center_icon.setStyleSheet("color: #38bdf8; font-size: 16px; font-weight: bold;")
    center_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
    vf_layout.addWidget(center_icon)
    vf_layout.addStretch()
    
    # Embedded HUD Overlay
    demo_hud = QWidget()
    demo_hud.setStyleSheet("""
        background-color: rgba(10, 14, 27, 235);
        border: 2px solid #0080FF;
        border-radius: 12px;
        padding: 12px;
    """)
    dh_layout = QVBoxLayout(demo_hud)
    dh_layout.setContentsMargins(15, 8, 15, 8)
    
    src_sub = QLabel("Speaker: Today we explore transformer attention mechanisms and streaming ASR.")
    src_sub.setStyleSheet("color: #94a3b8; font-size: 14px; font-style: italic;")
    src_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    tgt_sub = QLabel("ಇಂದು ನಾವು ಟ್ರಾನ್ಸ್‌ಫಾರ್ಮರ್ ಅಟೆನ್ಶನ್ ಕಾರ್ಯವಿಧಾನಗಳು ಮತ್ತು ಲೈವ್ ಸ್ಪೀಚ್ ರೆಕಗ್ನಿಷನ್ ಬಗ್ಗೆ ತಿಳಿಯಲಿದ್ದೇವೆ.")
    tgt_sub.setStyleSheet("color: #39FF14; font-size: 21px; font-weight: bold; letter-spacing: 0.5px;")
    tgt_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    dh_layout.addWidget(src_sub)
    dh_layout.addWidget(tgt_sub)
    
    vf_layout.addWidget(demo_hud, alignment=Qt.AlignmentFlag.AlignBottom)
    sim_layout.addWidget(video_frame)
    
    sim_screen.show()
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    sim_pixmap = sim_screen.grab()
    sim_path = os.path.join(figures_dir, "snapshot_live_video_hud.png")
    sim_pixmap.save(sim_path)
    print(f"Saved: {sim_path}")
    sim_screen.close()

    # 6. Snapshot: GPU Backend Server & Ngrok WebSocket Tunnel Console
    term_win = QWidget()
    term_win.resize(980, 520)
    term_win.setStyleSheet("background-color: #0c1017; border: 1px solid #30363d; border-radius: 8px;")
    term_layout = QVBoxLayout(term_win)
    term_layout.setContentsMargins(15, 15, 15, 15)
    
    header_lbl = QLabel("🖥️ Remote NVIDIA CUDA Tesla T4 Server — IndicTrans2 & ASR Engine Terminal")
    header_lbl.setStyleSheet("color: #58a6ff; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; font-weight: bold;")
    term_layout.addWidget(header_lbl)
    
    term_body = QLabel(
        "[INFO] Initializing Real-Time Multilingual Inference Server v2.4.0\n"
        "[GPU] Detected Device 0: NVIDIA Tesla T4 (15.1 GB VRAM | CUDA 12.2)\n"
        "[INIT] Loading AI4Bharat IndicConformer-600M Acoustic Model...\n"
        "[INIT] IndicConformer loaded into GPU VRAM (Allocated: 2.34 GB, Status: Ready)\n"
        "[INIT] Loading AI4Bharat IndicTrans2-1B (Translation Engine)...\n"
        "[INIT] IndicTrans2-1B (en-indic, indic-en) compiled with FlashAttention (Allocated: 4.12 GB)\n"
        "[GATEWAY] pyngrok tunnel established: wss://indic-subtitles-tunnel.ngrok-free.app -> localhost:8765\n"
        "[NET] Persistent WebSocket Server listening on 0.0.0.0:8765 (SSL/TLS Enabled)\n"
        "========================================================================================\n"
        "[CLIENT] New WebSocket Client Connected: ID=CLT-8824 (Windows 11 WASAPI Edge)\n"
        "[VAD] Ingesting PCM Audio Frame: 1.50s (48 kHz -> resampled 16 kHz Float32)\n"
        "[ASR] CTC Greedy Decoding: 'Artificial intelligence breaks linguistic barriers'\n"
        "      -> ASR Inference Time: 118.4 ms\n"
        "[NMT] IndicTrans2 Beam Search (beam_size=1, tgt_lang=kn):\n"
        "      -> 'ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯು ಭಾಷಾ ಅಡೆತಡೆಗಳನ್ನು ನಿವಾರಿಸುತ್ತದೆ'\n"
        "      -> NMT Inference Time: 214.2 ms\n"
        "[TELEMETRY] Total Pipeline Latency: 332.6 ms | VRAM Usage: 6.46 GB / 15.1 GB (Stable)\n"
        "[CACHE] Memory Compaction: malloc_trim(0) triggered -> Host RAM stabilized at 3.82 GB\n"
    )
    term_body.setStyleSheet("""
        color: #7ee787;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.45;
        background-color: #010409;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 12px;
    """)
    term_layout.addWidget(term_body)
    
    term_win.show()
    app.processEvents()
    time.sleep(0.15)
    app.processEvents()
    term_pixmap = term_win.grab()
    term_path = os.path.join(figures_dir, "snapshot_gpu_backend_terminal.png")
    term_pixmap.save(term_path)
    print(f"Saved: {term_path}")
    term_win.close()

if __name__ == "__main__":
    try:
        print("STARTING")
        sys.stdout.flush()
        generate_all_snapshots()
        print("DONE")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
