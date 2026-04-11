# VOID Model — Troubleshooting Guide

Detailed step-by-step resolution for all known issues when running Netflix VOID on the HP ZGX Nano G1n.

← [Back to README.md](README.md)

---

## Table of Contents

- [Session Resume Checklist](#session-resume-checklist)
- [PyTorch & CUDA 13.0 Installation](#pytorch--cuda-130-installation)
- [Building decord from Source](#decord--manual-build-from-source)
- [Hugging Face Authentication](#hugging-face-authentication)
- [Gemini API & Model Options](#gemini-api--model-options)
- [SAM3 BPE Vocabulary File Missing](#sam3-bpe-vocabulary-file-missing)
- [VLM-MASK-REASONER — GUI Access](#vlm-mask-reasoner--gui-access)
- [VLM-MASK-REASONER — Point Selector GUI How-To](#vlm-mask-reasoner--point-selector-gui-how-to)
- [VLM-MASK-REASONER — Common Pipeline Errors](#vlm-mask-reasoner--common-pipeline-errors)
- [Post-Crash / Post-Reboot — NVIDIA Driver Lost After Kernel Update](#post-crash--post-reboot--nvidia-driver-lost-after-kernel-update)
- [Gated Model Access — 403 Forbidden](#gated-model-access--403-forbidden)
- [General Error Reference](#general-error-reference)

---

## Session Resume Checklist

Run this every time you return to the device after a disconnect or reboot to confirm everything is ready before running inference or the mask pipeline.

```bash
# 1. Confirm GPU and driver are healthy
nvidia-smi

# 2. Confirm CUDA version
nvidia-smi | grep "CUDA Version"

# 3. Activate virtual environment
source ~/void-env/bin/activate

# 4. Set CUDA library path if not in bashrc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 5. Verify PyTorch sees the GPU
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 6. Confirm environment variables are set
echo "HF_TOKEN:      $HF_TOKEN"
echo "GEMINI_API_KEY: $GEMINI_API_KEY"

# 7. Navigate to working directory
cd ~/Desktop/NetflixVOID/void-model
```

Expected state: `nvidia-smi` shows `Driver Version: 580.x / CUDA Version: 13.0`, PyTorch prints `CUDA: True / GPU: NVIDIA GB10`, both tokens are non-empty.

If `nvidia-smi` fails, go to [Post-Crash Driver Recovery](#post-crash--post-reboot--nvidia-driver-lost-after-kernel-update).  
If PyTorch shows `CUDA: False`, go to [PyTorch & CUDA 13.0 Installation](#pytorch--cuda-130-installation).

---

## PyTorch & CUDA 13.0 Installation

### Why this is required

The ZGX Nano GB10 runs CUDA 13.0 (`libcudart.so.13`) with compute capability sm_121. The standard `pip install torch` resolves to PyPI's generic aarch64 wheel which is CPU-only (`2.x.x+cpu`) — no error is shown during install, but at inference time you get:

```
RuntimeError: Cannot get CUDA generator without ATen_cuda library
```

The cu130 aarch64 CUDA wheels exist on `download.pytorch.org/whl/cu130` but require exact version pinning and `uv` for reliable resolution. `pip` and unpinned `uv` both fall back to the CPU build silently.

### Verified working combination (aarch64, CUDA 13.0, Python 3.12)

- `torch==2.11.0+cu130`
- `torchvision==0.26.0+cu130`
- `torchaudio` — **omit entirely**, no aarch64 cu130 wheel exists at any version

### Install steps

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Remove any existing CPU build
pip uninstall torch torchvision torchaudio -y

# Install verified CUDA 13.0 build
uv pip install \
    "torch==2.11.0+cu130" \
    "torchvision==0.26.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match
```

> `--index-strategy unsafe-best-match` is required — without it `uv` rejects cross-version resolution.  
> Do not use `--extra-index-url` — it causes silent fallback to the PyPI CPU wheel.

### Verify

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected:
```
2.11.0+cu130
CUDA: True
GPU: NVIDIA GB10
```

If `CUDA: False` or `+cpu` appears, re-run the uninstall and `uv pip install` steps exactly as written.

### sm_121 compatibility warning

PyTorch may print:
```
UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

This is safe to ignore. sm_120 and sm_121 are binary compatible, confirmed by PyTorch maintainers.

---

## decord — Manual Build from Source

`decord` is a video loading library with no prebuilt aarch64 wheel. If `pip install -r requirements.txt` fails to install it, build from source. The `decord/` directory must sit alongside `void-model/`:

```
NetflixVOID/
├── void-model/
├── decord/          # ← built here
└── void-env/
```

### Step 1 — Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg libavcodec-dev libavfilter-dev \
    libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    cmake build-essential
```

### Step 2 — Clone decord

```bash
cd ~
git clone --recursive https://github.com/dmlc/decord.git
cd decord
```

### Step 3 — Apply FFmpeg 6.x patches

The ZGX Nano ships with FFmpeg 6.x. Two source patches are required before the build will succeed.

```bash
# Patch 1: Add missing bsf.h header (AVBSFContext moved in FFmpeg 4+)
sed -i '/#include <libavcodec\/avcodec.h>/a #include <libavcodec\/bsf.h>' \
    src/video/ffmpeg/ffmpeg_common.h

# Patch 2: Fix const correctness for av_find_best_stream (FFmpeg 5+)
sed -i 's/AVCodec \*dec;/const AVCodec *dec;/' \
    src/video/video_reader.cc

# Verify both patches applied
grep -n "bsf.h" src/video/ffmpeg/ffmpeg_common.h
grep -n "const AVCodec \*dec" src/video/video_reader.cc
```

Both `grep` commands must return a matching line. If either returns nothing, re-run the corresponding `sed` before continuing.

### Step 4 — Build

```bash
mkdir build && cd build

cmake .. \
    -DUSE_CUDA=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"

make -j$(nproc)
cd ..
```

> Use `-DUSE_CUDA=0` only if you encounter CUDA-related build errors.

### Step 5 — Install Python package

```bash
cd python
pip install .
cd ~/void-model
```

### Step 6 — Verify

```bash
python -c "import decord; print(decord.__version__)"
```

A version string (e.g. `0.6.0`) confirms success.

### Step 7 — Re-run requirements.txt

```bash
source ~/void-env/bin/activate
cd ~/void-model
pip install -r requirements.txt
```

---

## Hugging Face Authentication

```bash
pip install huggingface_hub
hf --version        # confirm CLI installed
echo $HF_TOKEN      # check if already set
```

If `$HF_TOKEN` is empty:

```bash
hf auth login
# Paste your token from huggingface.co/settings/tokens
```

Persist across sessions:

```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

---

## Gemini API & Model Options

### The default model is gated

Stage 2 of the VLM-MASK-REASONER pipeline uses `gemini-3.1-pro` by default. This model is behind a paid Google AI Studio plan. The free tier has `limit: 0` and immediately returns:

```
openai.RateLimitError: Error code: 429 — RESOURCE_EXHAUSTED
```

### Fix Option A — Enable billing

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Enable a paid plan
3. Monitor usage at [ai.dev/rate-limit](https://ai.dev/rate-limit)

### Fix Option B — Swap to a free-tier model

Find where the model is set:

```bash
grep -n "gemini-3\|model_name\|MODEL" VLM-MASK-REASONER/stage2_vlm_analysis.py | head -20
```

Free-tier and alternative model options:

| Option | Model string | Notes |
|--------|-------------|-------|
| Gemini free tier | `gemini-1.5-flash` | Free tier, lower capability |
| Gemini free tier | `gemini-2.0-flash` | Free tier, faster |
| Local open source | `ollama/llava` | No API key — runs on the ZGX Nano |
| Any OpenAI-compatible | any | Point `base_url` to your endpoint |

### Fix Option C — Use Ollama locally (no API key)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision-capable model
ollama pull llava
```

Ollama serves an OpenAI-compatible API at `localhost:11434`. Update `base_url` and the model name in `stage2_vlm_analysis.py` accordingly.

---

## SAM3 BPE Vocabulary File Missing

### Error

```
FileNotFoundError: [Errno 2] No such file or directory:
'/home/.../void-env/lib/python3.12/site-packages/assets/bpe_simple_vocab_16e6.txt.gz'
```

### Cause

The `sam3` pip package does not bundle its BPE tokenizer vocabulary file correctly. SAM3's text encoder requires it at startup.

### Fix

```bash
# Create the assets directory
mkdir -p ~/void-env/lib/python3.12/site-packages/assets/

# Download the missing BPE vocab file from OpenAI's CLIP repository
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
    -O ~/void-env/lib/python3.12/site-packages/assets/bpe_simple_vocab_16e6.txt.gz

# Verify
ls -lh ~/void-env/lib/python3.12/site-packages/assets/bpe_simple_vocab_16e6.txt.gz
```

Re-run the pipeline after confirming the file is present.

---

## VLM-MASK-REASONER — GUI Access

### Error

```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

This occurs when running over SSH with no display server. Choose one option:

### Option 1 — Run on the ZGX Nano's local desktop

Disconnect SSH and open a terminal directly on the ZGX Nano's desktop session. No additional configuration needed.

### Option 2 — SSH with X11 forwarding

```bash
# On your local machine:
ssh -X zgx-prod-2@<ip-address>

# Verify DISPLAY is set:
echo $DISPLAY   # should show e.g. localhost:10.0

cd ~/Desktop/NetflixVOID/void-model
source ~/void-env/bin/activate
python VLM-MASK-REASONER/point_selector_gui.py
```

> If `$DISPLAY` is still empty: on macOS install [XQuartz](https://www.xquartz.org) first; on Windows use [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [MobaXterm](https://mobaxterm.mobatek.net).

### Option 3 — Browser-based GUI via noVNC (recommended for SSH users)

Streams the Tkinter window to any browser — no VNC client needed on your local machine.

> **Critical:** Multiple stale `x11vnc` and `websockify` processes conflict silently. A new `x11vnc` will autopick port 5903 instead of 5900 if 5900 is already taken, causing `websockify` to fail with `Connection refused` even though everything appears running. Always kill all instances first and use `-rfbport 5900 -listen localhost` to force consistent binding.

```bash
# 1. Install (one time)
sudo apt install xvfb x11vnc novnc websockify -y

# 2. Kill any stale processes
pkill x11vnc; pkill websockify; pkill Xvfb
sleep 2

# 3. Verify ports are clear — must return nothing before continuing
ss -tlnp | grep -E "5900|5903|6080"

# 4. Start virtual display
Xvfb :1 -screen 0 1920x1080x24 &
sleep 1

# 5. Start x11vnc — force port 5900, bind to localhost only
export DISPLAY=:1
x11vnc -display :1 -nopw -forever -shared -rfbport 5900 -listen localhost &
sleep 1

# 6. Confirm x11vnc is on 5900 — do not proceed if empty
ss -tlnp | grep 5900

# 7. Start noVNC
websockify --web /usr/share/novnc/ 6080 localhost:5900 &
sleep 1

# 8. Launch the GUI
export DISPLAY=:1
cd ~/Desktop/NetflixVOID/void-model
source ~/void-env/bin/activate
python VLM-MASK-REASONER/point_selector_gui.py &
```

Open in browser: `http://192.168.1.47:6080/vnc.html` → click **Connect**.

To stop everything when done:

```bash
pkill x11vnc; pkill websockify; pkill Xvfb
```

### Option 4 — Skip the GUI entirely (headless)

Use the Grounded-SAM-2 open source alternative — it generates masks from a text prompt with no display required. See the [VOID GitHub repo](https://github.com/IDEA-Research/Grounded-SAM-2) for setup instructions.

---

## VLM-MASK-REASONER — Point Selector GUI How-To

The `point_selector_gui.py` Tkinter tool marks the object to remove across video frames and saves a `mask_config_points.json` config file that the pipeline reads for SAM2 segmentation.

### Workflow

1. **Load your video** — use the file picker to open `input_video.mp4` from your video folder inside `sample/`
2. **Navigate frames** — use arrow buttons or the slider to move through the timeline
3. **Click positive points** — left-click on the object to remove across at least 5–10 frames for reliable tracking
4. **Click negative points if needed** — right-click areas the mask should NOT include (background the model might confuse with the object)
5. **Set the instruction** — type a short removal description, e.g. `"remove the sideline referee with the flag"`
6. **Save** — click **Save**. The config is written to `VLM-MASK-REASONER/mask_config_points.json`

### Where the config is saved

```
void-model/
└── VLM-MASK-REASONER/
    └── mask_config_points.json
```

### Run the pipeline

Always run from inside `void-model/`:

```bash
cd ~/Desktop/NetflixVOID/void-model
bash VLM-MASK-REASONER/run_pipeline.sh VLM-MASK-REASONER/mask_config_points.json
```

> Passing just `my_config_points.json` causes `FileNotFoundError` — the script looks for it relative to `void-model/`, not inside `VLM-MASK-REASONER/`.

### To modify an existing config

Re-open the GUI, load the same video, adjust points, and save. The file is overwritten in place.

### Config file format (for manual editing)

```json
{
  "videos": [
    {
      "video_path": "sample/my-video/input_video.mp4",
      "output_dir": "sample/my-video",
      "instruction": "remove the sideline referee with the flag",
      "points": {
        "0":  {"positive": [[208, 284], [259, 423]], "negative": [], "box": [201, 217, 264, 441]},
        "10": {"positive": [[284, 248], [300, 292]], "negative": [], "box": [278, 229, 341, 448]}
      }
    }
  ]
}
```

---

## VLM-MASK-REASONER — Common Pipeline Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: my_config_points.json` | Config path not relative to `void-model/` | Pass `VLM-MASK-REASONER/mask_config_points.json` |
| `ModuleNotFoundError: No module named 'openai'` | `openai` not installed | `pip install openai` |
| `ModuleNotFoundError: No module named 'tkinter'` | System package missing | `sudo apt install python3-tk -y` |
| `❌ Checkpoint not found: ../sam2_hiera_large.pt` | Checkpoint in wrong location | `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O ../sam2_hiera_large.pt` |
| `FileNotFoundError: bpe_simple_vocab_16e6.txt.gz` | SAM3 missing BPE vocab file | See [SAM3 BPE Vocabulary File Missing](#sam3-bpe-vocabulary-file-missing) |
| `GatedRepoError: 403 Client Error — facebook/sam3` | HF access not granted | Visit [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), request access, wait for approval, then re-run. See [Gated Model Access](#gated-model-access--403-forbidden) |
| `UserWarning: cannot import name '_C' from 'sam2'` | SAM2 C extension not compiled | Safe to ignore — segmentation still works |
| `429 RESOURCE_EXHAUSTED` (Gemini) | Free tier quota or paid model | See [Gemini API & Model Options](#gemini-api--model-options) |
| Stage 2 Gemini key error | `GEMINI_API_KEY` not set | `echo $GEMINI_API_KEY` — if blank, re-export and `source ~/.bashrc` |
| `ImportError: SAM3 not available` | `sam3` not installed | `pip install sam3` |

### Pipeline dependency pre-flight check

Run before starting the pipeline to catch any missing packages:

```bash
python -c "import tkinter; print('tkinter      OK')"
python -c "import google.generativeai; print('generativeai OK')"
python -c "import openai; print('openai       OK')"
python -c "import cv2; print('opencv       OK')"
python -c "import PIL; print('Pillow       OK')"
python -c "import sam2; print('SAM2         OK')"
```

---

## Post-Crash / Post-Reboot — NVIDIA Driver Lost After Kernel Update

After a hard crash or unexpected reboot the ZGX Nano may boot into a newer kernel (e.g. `6.14.0` → `6.17.0-1014-nvidia`) for which NVIDIA driver modules have not been compiled. `nvidia-smi` will report `couldn't communicate with the NVIDIA driver` and `modprobe nvidia` will fail with `Module not found`.

### Step 1 — Identify the issue

```bash
uname -r
dpkg -l | grep -E "nvidia-driver|nvidia-dkms"
```

### Step 2 — Rebuild driver modules with DKMS

```bash
sudo apt update
sudo apt install --reinstall nvidia-dkms-580 nvidia-driver-580 -y
sudo dkms autoinstall
sudo reboot
```

> Replace `580` with the version shown in `dpkg -l` if different.

### Step 3 — Verify after reboot

```bash
nvidia-smi
source ~/void-env/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected: `nvidia-smi` shows `Driver Version: 580.x / CUDA Version: 13.0`, PyTorch prints `CUDA: True / GPU: NVIDIA GB10`.

### If SSH is unavailable — use KVM or DGX Dashboard

Connect directly using a KVM switch or open the HP ZGX / DGX Dashboard from a browser on the same network:

```
http://192.168.1.47   # replace with your ZGX Nano's IP
```

The dashboard shows system health, driver version, and supports triggering driver updates and restarts without SSH. Verify the GPU is detected and the driver is active before returning to the SSH workflow above.

---

## Gated Model Access — 403 Forbidden

### Error

```
huggingface_hub.errors.GatedRepoError: 403 Client Error.
Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json.
Access to model facebook/sam3 is restricted and you are not in the authorized list.
```

### Cause

`facebook/sam3` (used in Stage 3 grey mask generation) and `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` (the base inpainting model) are **gated repositories** on Hugging Face. Even with a valid HF token, downloads return `403 Forbidden` until you have manually requested and been granted access on the model's page.

### Step 1 — Request access for each gated model

Visit each URL below while logged in to your Hugging Face account, click **Agree and access repository**, and complete the access form:

- **SAM3:** [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- **CogVideoX-Fun:** [huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP)

Approval is typically granted within minutes to a few hours. You will receive a Hugging Face notification when access is granted.

### Step 2 — Confirm your HF token is set

```bash
echo $HF_TOKEN   # must be non-empty

# If empty, re-authenticate:
hf auth login
```

### Step 3 — Verify access before re-running

```bash
# Test SAM3 access
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam3', 'config.json'); print('SAM3 access OK')"

# Test CogVideoX access
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('alibaba-pai/CogVideoX-Fun-V1.5-5b-InP', 'config.json'); print('CogVideoX access OK')"
```

Both should print `OK`. If either raises `GatedRepoError`, access has not yet been granted — wait and retry.

### Step 4 — Re-run the pipeline

Once both models are accessible, re-run from `void-model/`:

```bash
bash VLM-MASK-REASONER/run_pipeline.sh VLM-MASK-REASONER/mask_config_points.json     --sam2-checkpoint /home/zgx-prod-2/Desktop/NetflixVOID/sam2_hiera_large.pt     --device cuda
```

> Stages 1 and 2 will re-run but complete quickly. Stage 3 will now load SAM3 successfully.

---

## General Error Reference

| Error | Fix |
|-------|-----|
| `git lfs install` fails | `sudo apt install git-lfs -y` then re-run `git lfs install` |
| `RuntimeError: Cannot get CUDA generator` | See [PyTorch & CUDA 13.0 Installation](#pytorch--cuda-130-installation) |
| `libcudart.so.12: cannot open shared object` | Same root cause — wrong CUDA build. See above. |
| `nvidia-smi` not found after reboot | See [Post-Crash Driver Recovery](#post-crash--post-reboot--nvidia-driver-lost-after-kernel-update) |
| ARM wheel not available | `pip install <pkg> --no-binary :all:` |
| `ffmpeg` not found | `ln -sf $(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") ~/.local/bin/ffmpeg` |
| CUDA out of memory | Unlikely on 128 GB — if it occurs, set `--config.system.gpu_memory_mode=model_cpu_offload` |
| SAM2 import error | Reinstall: `cd ~/sam2 && pip install -e .` |
| `torch_dtype` deprecation warning | Safe to ignore — no impact on inference |

---

← [Back to README.md](README.md)
