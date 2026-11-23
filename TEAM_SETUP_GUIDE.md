# Team Collaboration & Shared ML Server Guide

This guide describes the setup for teams developing **Live Subtitle Studio** on separate laptops with large ML models (Whisper / IndicConformer / Silero VAD) and CPU inference.

---

## 1. System Architecture

```
                   ┌──────────────────────────────────────────────┐
                   │                 GitHub Repo                  │
                   │        (Code, GitHub Actions CI)              │
                   └──────────────────────┬───────────────────────┘
                                          │ git clone / pull
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
           ┌─────────────────────┐                 ┌─────────────────────┐
           │   Teammate Laptop   │                 │   Teammate Laptop   │
           │   (Laptop A - Win)  │                 │   (Laptop B - Win)  │
           │                     │                 │                     │
           │  WASAPI Loopback    │                 │  WASAPI Loopback    │
           │  Transparent Overlay│                 │  Transparent Overlay│
           │  PyQt6 Studio GUI   │                 │  PyQt6 Studio GUI   │
           └──────────┬──────────┘                 └──────────┬──────────┘
                      │                                       │
      (A) Shared Mode │ (No model downloads)                  │ (No model downloads)
      (Default / Rec) │                                       │
                      ▼                                       ▼
         ┌────────────────────────────────────────────────────────┐
         │                Shared CPU / GPU Server                 │
         │   translation_proxy.py :8765  ->  app_bridge.py :8766  │
         │                                                        │
         │   - WebSocket ASR + translation layer                  │
         │   - Model cache: ./models (gitignored)                 │
         │   - faster-whisper / IndicConformer CPU inference      │
         └────────────────────────────────────────────────────────┘

      (B) Local Mode: Teammates can also run inference directly on their
          own laptop by installing the backend dependencies.
```

### Why this design?
1. **Desktop Audio & Overlay stay native**: Windows WASAPI Loopback Capture (recording PC speaker audio) and transparent frameless desktop overlays need native Windows APIs, so they run on each teammate's laptop.
2. **Model weights live on disk**: Weights (Whisper, VAD, translation models) are stored in `./models/` and are gitignored (`models/`, `*.onnx`, `*.bin`, `*.pt`).
3. **Zero-download setup for teammates**: In shared mode, teammates connect to the inference server over WebSocket and do not download multi-gigabyte models.

---

## 2. Quick Start for Teammates (Laptops)

### Step 1: Clone the repository
```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd subtitlesssss
```

### Step 2: Configure Environment
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in VS Code and set your team's shared server URL:
```ini
ASR_MODE=shared
REMOTE_ASR_URL=ws://<shared-server-ip>:8765
```

### Step 3: Install Desktop Dependencies & Run
```powershell
# Virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install client and backend dependencies
pip install -r requirements.txt
pip install PyQt6 pyaudiowpatch

# Launch Live Subtitles connected to the shared server
python main_app.py --shared
```
You can also toggle between **Shared Server** and **Local Engine** at any time from the **Control Center Settings** dialog.

---

## 3. Dedicated Shared Server Setup (Run once on team server)

On the dedicated machine (e.g. Ubuntu, Windows Server, or an office desktop):

1. Clone the repo and install the backend dependencies:
   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd subtitlesssss
   pip install -r backend/requirements.txt
   ```

2. Start the ASR bridge (terminal 1):
   ```bash
   python backend/app_bridge.py --host 0.0.0.0 --port 8766 --model lazy
   ```
   `--model lazy` avoids opening a local audio device on the server; the model is loaded from each client's handshake.

3. Start the translation proxy (terminal 2, from the repo root):
   ```bash
   python translation_proxy.py
   ```
   It listens on port `8765` and forwards to the bridge on `8766` by default (`PROXY_PORT` / `UPSTREAM_URL` environment variables can override these).

Clients connect to `ws://<shared-server-ip>:8765`. Downloaded models are cached in `./models` (or `HF_HOME`) on the server.

---

## 4. Offline / Local Development (Optional)

If a teammate is traveling or developing new ASR/VAD features locally:

1. In `.env`, set:
   ```ini
   ASR_MODE=local
   MODEL_NAME=base
   ```
2. Install the full set of dependencies:
   ```powershell
   pip install -r requirements.txt
   pip install PyQt6 pyaudiowpatch
   ```
3. Run the desktop app (models are downloaded/cached on first use):
   ```powershell
   python main_app.py --mode local
   ```

---

## 5. Storage Guidelines for Model Weights

| Location | Rules |
| :--- | :--- |
| **Git (`.gitignore`)** | **NEVER commit model files** (`models/`, `*.onnx`, `*.bin`, `*.pt`, `*.safetensors` are strictly ignored). |
| **Shared Server** | Downloaded once into `./models` (or `HF_HOME`) on the server and reused for all clients. |
| **Developer Laptops** | In shared mode, **0 MB** of model files are downloaded. In local mode, cached in `./models`. |

---

## 6. GitHub CI Workflow

Every pull request and push to `main` triggers `.github/workflows/ci.yml`:
- **Code Sanity**: Validates Python syntax across `backend/`, `app/`, and `main_app.py`.
