import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

docx_path = r"d:\New folder\subtitlesssss\Realtime_Multilingual_Subtitles_Project_Report.docx"
doc = docx.Document(docx_path)

# Check if Chapter 9 is already added
for p in doc.paragraphs:
    if "CHAPTER 9: SNAPSHOTS" in p.text.upper() or "CHAPTER 9 - SNAPSHOTS" in p.text.upper():
        print("Chapter 9 already in docx, skipping.")
        exit(0)

# Add page break before Chapter 9
doc.add_page_break()

# Heading 1: CHAPTER 9 SNAPSHOTS
h1 = doc.add_paragraph()
r1 = h1.add_run("CHAPTER 9\nSNAPSHOTS")
r1.bold = True
r1.font.size = Pt(20)
r1.font.name = "Arial"
h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
h1.paragraph_format.space_before = Pt(24)
h1.paragraph_format.space_after = Pt(18)

# Intro text
intro = doc.add_paragraph(
    "The presentation and operational layer of the Real-Time Multilingual Subtitle Generation and "
    "Translation System combines a lightweight, native Windows desktop client with a remote GPU cloud inference cluster. "
    "The interface adheres to modern OLED Dark Mode design standards (WCAG AA compliant contrast) and uses consistent semantic colors: "
    "Neon Green (#39FF14) indicates active streaming and valid voice activity; Electric Blue (#0080FF) highlights model badges and English source text; "
    "Amber (#FFD700) represents paused streams and buffer alerts; and Crimson (#EF4444) marks active loopback capture. "
    "Each screen and pipeline diagram is described below alongside its corresponding figure."
)
intro.paragraph_format.space_after = Pt(14)
intro.paragraph_format.line_spacing = 1.15

figures_base = r"d:\New folder\subtitlesssss\figures"

sections = [
    {
        "sec": "9.1 Snapshots of System Architecture and Pipeline",
        "items": [
            {
                "sub": "9.1.1 Master System Architecture & Tiered Edge-Cloud Topology",
                "desc": (
                    "Figure 9.1 illustrates the decoupled hybrid edge–cloud topology of the system. "
                    "The edge layer captures internal audio from the Windows Audio Session API (WASAPI) loopback buffer "
                    "without microphone feedback, executes Silero Voice Activity Detection (VAD) on the client CPU, "
                    "and streams speech chunks over an encrypted TLS reverse tunnel (pyngrok WSS). "
                    "The cloud inference layer deploys AI4Bharat IndicConformer-600M for acoustic speech recognition and "
                    "IndicTrans2-1B for neural machine translation on an NVIDIA Tesla GPU node, returning synchronized translated text frames."
                ),
                "img": os.path.join(figures_base, "System_Design_Architecture_Master.png"),
                "caption": "Fig. 9.1: System architecture of the Real-Time Multilingual Subtitle Generation and Translation System"
            },
            {
                "sub": "9.1.2 End-to-End Audio-ASR-Translation Pipeline Flowchart",
                "desc": (
                    "Figure 9.2 details the master pipeline flowchart executing across the edge client and cloud worker. "
                    "Raw 48 kHz stereo PCM audio captured from internal playback is downmixed and resampled to 16 kHz mono Float32. "
                    "Silero VAD classifies 32 ms acoustic frames into speech and non-speech states. Non-speech frames are discarded, "
                    "while vocal chunks between 1.0 s and 3.0 s are dispatched over persistent WebSockets. On the inference node, "
                    "CTC greedy decoding extracts text transcripts which are immediately routed into the IndicTrans2 encoder-decoder pipeline."
                ),
                "img": os.path.join(figures_base, "pipeline_flowchart.png"),
                "caption": "Fig. 9.2: Master audio acquisition, VAD segmentation, and translation pipeline flowchart"
            }
        ]
    },
    {
        "sec": "9.2 Snapshots of Application Interfaces and Desktop HUD",
        "items": [
            {
                "sub": "9.2.1 Desktop Control Center & Real-Time Monitoring Panel",
                "desc": (
                    "The Live Subtitle Studio Control Center (Fig. 9.3) serves as the primary operator control panel. "
                    "It features two-column interactive dropdowns for spoken and subtitle language pairing, dynamic neural model strategy "
                    "recommendation badges, loopback device selectors with hotplug refresh, real-time VU audio meter with VAD indicators, "
                    "and an embedded live transcript stream displaying timestamps, source speech, and translated vernacular output."
                ),
                "img": os.path.join(figures_base, "snapshot_control_center.png"),
                "caption": "Fig. 9.3: Live Subtitle Studio Control Center with real-time VU meter and live transcript feed"
            },
            {
                "sub": "9.2.2 Application Settings & Model Routing Dialog",
                "desc": (
                    "Figure 9.4 shows the modal Settings Dialog window. It allows users to configure persistent parameters including "
                    "target ASR/NMT engine routing (local quantized INT8 vs. remote GPU), subtitle typography font sizes (20px to 36px), "
                    "screen position anchors (Top, Middle, Bottom), line wrapping limits (42 characters), auto-fade duration (8s), "
                    "and secure remote WebSocket tunnel endpoints (wss://)."
                ),
                "img": os.path.join(figures_base, "snapshot_settings_dialog.png"),
                "caption": "Fig. 9.4: System Configuration Dialog for language pairing, VAD sensitivity, and WebSocket tunnel"
            },
            {
                "sub": "9.2.3 Transparent Hardware-Accelerated Floating Subtitle HUD Overlay (Kannada)",
                "desc": (
                    "The Desktop Subtitle Overlay is a frameless, transparent, click-through, stay-on-top window built on PyQt6 (Fig. 9.5). "
                    "It projects live subtitles over active desktop applications without intercepting mouse clicks or obstructing user workflows. "
                    "When bilingual display is enabled, the recognized English source text is rendered in a smaller muted font above, "
                    "while the high-fidelity Kannada vernacular translation appears in large, bold script below."
                ),
                "img": os.path.join(figures_base, "snapshot_hud_overlay_kn.png"),
                "caption": "Fig. 9.5: Bilingual desktop subtitle overlay rendering Kannada translation with source text"
            },
            {
                "sub": "9.2.4 Live Subtitling Demonstration in Hindi Devanagari Script",
                "desc": (
                    "Figure 9.6 shows the HUD overlay rendering live Hindi translations in native Devanagari script. "
                    "The translucent pill container (rgba(0, 0, 0, 0.78)) provides optimal contrast across arbitrary desktop backgrounds "
                    "while remaining completely click-through for active fullscreen applications."
                ),
                "img": os.path.join(figures_base, "snapshot_hud_overlay_hi.png"),
                "caption": "Fig. 9.6: Subtitle HUD rendering real-time Hindi translation with source text"
            },
            {
                "sub": "9.2.5 Real-Time Desktop Media Playback with Subtitle HUD in Action",
                "desc": (
                    "Figure 9.7 demonstrates the end-to-end system operating in real-world conditions during a live video lecture. "
                    "As the media player outputs 48 kHz digital audio internally, WASAPI loopback captures the PCM frames, "
                    "Silero VAD detects speech segments, IndicTrans2 translates the utterance, and the PyQt6 HUD overlay projects "
                    "synchronized bilingual subtitles directly over the video stream without requiring physical microphone hardware."
                ),
                "img": os.path.join(figures_base, "snapshot_live_video_hud.png"),
                "caption": "Fig. 9.7: Real-time desktop media playback with transparent subtitle HUD overlay in action"
            },
            {
                "sub": "9.2.6 Distributed GPU Inference Server & WebSocket Tunnel Gateway Output",
                "desc": (
                    "Figure 9.8 shows the console log of the remote GPU inference server running on an NVIDIA Tesla T4 instance. "
                    "It displays model memory allocation for IndicConformer-600M and IndicTrans2-1B, pyngrok TLS tunnel initialization, "
                    "per-chunk ASR/NMT latency telemetry (332.6 ms cumulative round-trip), and automated memory compaction via malloc_trim(0)."
                ),
                "img": os.path.join(figures_base, "snapshot_gpu_backend_terminal.png"),
                "caption": "Fig. 9.8: Cloud GPU backend terminal showing model execution, WebSocket tunnel, and telemetry"
            }
        ]
    }
]

for section in sections:
    sec_p = doc.add_paragraph()
    sec_r = sec_p.add_run(section["sec"])
    sec_r.bold = True
    sec_r.font.size = Pt(14)
    sec_p.paragraph_format.space_before = Pt(14)
    sec_p.paragraph_format.space_after = Pt(6)
    
    for item in section["items"]:
        sub_p = doc.add_paragraph()
        sub_r = sub_p.add_run(item["sub"])
        sub_r.bold = True
        sub_r.font.size = Pt(12)
        sub_p.paragraph_format.space_before = Pt(10)
        sub_p.paragraph_format.space_after = Pt(4)
        
        desc_p = doc.add_paragraph(item["desc"])
        desc_p.paragraph_format.line_spacing = 1.15
        desc_p.paragraph_format.space_after = Pt(8)
        
        if os.path.exists(item["img"]):
            img_p = doc.add_paragraph()
            img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            img_r = img_p.add_run()
            img_r.add_picture(item["img"], width=Inches(5.8))
            img_p.paragraph_format.space_after = Pt(4)
            
            cap_p = doc.add_paragraph()
            cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_r = cap_p.add_run(item["caption"])
            cap_r.bold = True
            cap_r.font.size = Pt(9.5)
            cap_r.font.italic = True
            cap_p.paragraph_format.space_after = Pt(14)

doc.save(docx_path)
print("Successfully appended Chapter 9 to Realtime_Multilingual_Subtitles_Project_Report.docx")
