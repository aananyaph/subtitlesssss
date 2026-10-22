# Team Collaboration & Docker ML Architecture Guide

This guide describes the setup for teams developing **Live Subtitle Studio** on separate laptops with large ML models (Whisper / OpenVINO / Silero VAD) and CPU inference.

---

## 1. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │                 GitHub Repo                  │
                  │  (Code, GitHub Actions CI, docker-compose)   │
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
        │       Shared CPU Server / Central Docker Container     │
        │   `docker compose --profile server up -d`              │
        │                                                        │
        │   - WebSocket ASR Engine (Port 8765)                   │
        │   - Persistent Model Cache: `/srv/models` (mounted)    │
        │   - Faster-Whisper + OpenVINO CPU inference            │
        └────────────────────────────────────────────────────────┘

     (B) Local Mode: Teammates can also run `docker compose --profile local-ml up -d`
         offline on their laptop with `./models` mounted locally.
```

### Why this design?
1. **Desktop Audio & Overlay stay native**: Windows WASAPI Loopback Capture (recording PC speaker audio) and transparent frameless desktop overlays need native Windows APIs and cannot run inside Docker containers.
2. **Model weights stay OUTSIDE Docker images**: Weights (Whisper models, VAD models) are stored in `./models/` on disk and mounted via volumes (`${MODEL_DIR:-./models}:/app/models`). They are never committed to git or baked into Docker image layers.
3. **Zero-download setup for teammates**: Teammates on separate laptops do not need to download multi-gigabyte models or exhaust their laptop CPUs; they connect to the shared inference server over WebSocket.

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

# Install desktop client dependencies
pip install -r backend/requirements.txt PyQt6 pyaudiowpatch

# Launch Live Subtitles connected to the shared server
python main_app.py --shared
```
You can also toggle between **Shared Server** and **Local Engine** at any time from the **Control Center Settings** dialog.

---

## 3. Dedicated Shared Server Setup (Run once on team server)

On the dedicated CPU machine (e.g. Ubuntu, Windows Server, or an office desktop):

1. Clone the repo:
   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd subtitlesssss
   cp .env.example .env
   ```
2. Start the shared subtitle engine:
   ```bash
   docker compose --profile server up -d
   ```
3. Check status and logs:
   ```bash
   docker compose logs -f subtitle-engine-server
   ```
The server will automatically load the model specified in `MODEL_NAME` (default: `base`) and cache it inside `./models`.

---

## 4. Offline / Local Docker Development (Optional)

If a teammate is traveling or developing new ASR/VAD features locally:

1. In `.env`, set:
   ```ini
   ASR_MODE=local
   MODEL_NAME=base
   ```
2. Launch the local ML container:
   ```powershell
   docker compose --profile local-ml up -d
   ```
3. Run the desktop app:
   ```powershell
   python main_app.py --mode local
   ```

To shut down local containers:
```powershell
docker compose --profile local-ml down
```

---

## 5. Storage Guidelines for Model Weights

| Location | Rules |
| :--- | :--- |
| **Git (`.gitignore`)** | **NEVER commit model files** (`models/`, `*.onnx`, `*.bin`, `*.pt` are strictly ignored). |
| **Docker Images (`.dockerignore`)** | **NEVER bake model files into images** (`models/` is excluded from the build context). |
| **Shared Persistent Storage** | Mounted as a read-only volume (`./models:/app/models`). Downloaded once on the server. |
| **Developer Laptops** | In shared mode, **0 MB** of model files are downloaded. In local mode, cached in `./models`. |

---

## 6. GitHub CI Workflow

Every pull request and push to `main` triggers `.github/workflows/docker.yml`:
- **Code Sanity**: Validates Python syntax across `backend/`, `app/`, and `main_app.py`.
- **Compose Validation**: Verifies `.env.example` and tests `docker compose config`.
- **Docker Build Check**: Builds `live-subtitle-studio:latest` to ensure dependencies compile cleanly without bundling large model binaries into image layers.
