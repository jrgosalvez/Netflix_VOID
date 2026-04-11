# VOID Model — HP ZGX Nano Setup Guide

**Netflix VOID** (Video Object and Interaction Deletion) on the **HP ZGX Nano G1n**  
Source: [huggingface.co/netflix/void-model](https://huggingface.co/netflix/void-model)

---

## Who this Guide is For

This guide was written to help Media and Entertainment video enditing professionals looking to use Netflix's open source VOID model to remove objects from videos. 

---

## Hardware

| Component | Spec |
|-----------|------|
| SoC | NVIDIA GB10 Grace Blackwell Superchip |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | NVIDIA Blackwell (6144 CUDA Cores, 5th Gen Tensor Cores) |
| Unified Memory | **128 GB** LPDDR5X @ 273 GB/s ✅ (VOID requires 40 GB+) |
| AI Perf | 1,000 TOPS (FP4) |
| Storage | 4 TB NVMe SSD |
| OS | NVIDIA DGX OS (Ubuntu 24.04 base) |

---

## Prerequisites

The ZGX Nano ships with NVIDIA DGX OS. Verify your environment before starting.

```bash
# Confirm OS and kernel
uname -a
cat /etc/os-release

# Confirm NVIDIA GPU is visible
nvidia-smi

# Confirm CUDA is available
nvcc --version

# Confirm Python 3.10+
python3 --version
```

---

## 1 — System Dependencies

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    git git-lfs curl wget ffmpeg \
    python3-pip python3-venv python3-dev \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0

git lfs install
```

---

## 2 — Create Virtual Environment

```bash
python3 -m venv ~/void-env
source ~/void-env/bin/activate

pip install --upgrade pip setuptools wheel
```

> Add `source ~/void-env/bin/activate` to `~/.bashrc` to auto-activate on login.

---

## 3 — Clone VOID Repository

```bash
git clone https://github.com/netflix/void-model.git
cd void-model
```

---

## 4 — Install Python Dependencies

Ensure your virtual environment is active before proceeding:

```bash
source ~/void-env/bin/activate
```

### Step 4a — Install PyTorch with CUDA 13.0 (GB10-specific, required first)

> **Critical:** The ZGX Nano's GB10 Superchip runs CUDA 13.0 (`libcudart.so.13`) with
> compute capability sm_121. `pip install torch` silently resolves to PyPI's generic
> aarch64 wheel which is CPU-only (`2.x.x+cpu`) — no error is shown, but inference fails
> at runtime with `RuntimeError: Cannot get CUDA generator without ATen_cuda library`.
>
> The correct CUDA-enabled wheels for aarch64 are on `download.pytorch.org/whl/cu130`
> but require exact version pinning and `uv` for reliable resolution. `pip` and
> unpinned `uv` both fall back to the CPU build silently.
>
> **Verified working combination for ZGX Nano (aarch64, CUDA 13.0, Python 3.12):**
> - `torch==2.11.0+cu130`
> - `torchvision==0.26.0+cu130`
> - `torchaudio` — **skip entirely**, no aarch64 cu130 wheel exists

**Install `uv` (if not already present):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

**Remove any existing CPU-only torch build first:**

```bash
pip uninstall torch torchvision torchaudio -y
```

**Install the verified CUDA 13.0 aarch64 build using `uv`:**

```bash
uv pip install \
    "torch==2.11.0+cu130" \
    "torchvision==0.26.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match
```

> `--index-strategy unsafe-best-match` is required to allow `uv` to resolve across
> the index without treating version mismatches as dependency confusion attacks.
> Do not use `--extra-index-url` — it causes `uv` to fall back to PyPI for aarch64.
> `torchaudio` is intentionally omitted — no aarch64 cu130 wheel exists for it and
> VOID does not require it.

**Verify CUDA is accessible before continuing:**

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
2.11.0+cu130
CUDA: True
GPU: NVIDIA GB10
```

If `CUDA: False` or `+cpu` appears in the version string, the wrong wheel is installed.
Re-run the uninstall step and the `uv pip install` command above exactly as written.

> **Note on sm_121 compatibility warning:** PyTorch may print a `UserWarning` that the
> GB10 (sm_121) exceeds its maximum supported capability (sm_120). This is safe to
> ignore — sm_120 and sm_121 are binary compatible, confirmed by PyTorch maintainers.

### Step 4b — Install all remaining dependencies

```bash
pip install -r requirements.txt
```

> The ZGX Nano's ARM architecture (aarch64) means some wheels may build from source.
> If any package fails, try: `pip install <package> --no-binary :all:`

### decord — Manual Build (if `requirements.txt` fails to install it)

`decord` is a video loading library with no prebuilt aarch64 wheel, so it must be built
from source. The `decord/` directory should sit **alongside** `void-model/` at the same level:

```
~/
├── void-model/
├── decord/          # ← built here
└── void-env/        # ← virtual environment
```

Make sure the virtual environment is active for all steps below.

#### Step 1 — Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg libavcodec-dev libavfilter-dev \
    libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev \
    cmake build-essential
```

#### Step 2 — Clone the repo (at the same level as `void-model/`)

```bash
cd ~
git clone --recursive https://github.com/dmlc/decord.git
cd decord
```

#### Step 3 — Apply source patches (FFmpeg 6.x compatibility)

The ZGX Nano ships with FFmpeg 6.x, which requires two patches before the build will succeed.

```bash
# Patch 1: Add missing bsf.h header (AVBSFContext moved in FFmpeg 4+)
sed -i '/#include <libavcodec\/avcodec.h>/a #include <libavcodec\/bsf.h>' \
    src/video/ffmpeg/ffmpeg_common.h

# Patch 2: Fix const correctness for av_find_best_stream (FFmpeg 5+)
sed -i 's/AVCodec \*dec;/const AVCodec *dec;/' \
    src/video/video_reader.cc

# Verify both patches applied correctly
grep -n "bsf.h" src/video/ffmpeg/ffmpeg_common.h
grep -n "const AVCodec \*dec" src/video/video_reader.cc
```

Both `grep` commands should return a matching line. If either returns nothing, re-run the corresponding `sed` command before continuing.

#### Step 4 — Build the C++ library

```bash
mkdir build && cd build

cmake .. \
    -DUSE_CUDA=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"

make -j$(nproc)
cd ..
```

> The ZGX Nano has a CUDA-capable Blackwell GPU, so `-DUSE_CUDA=1` is recommended.
> Use `-DUSE_CUDA=0` only if you encounter CUDA-related build errors.

#### Step 5 — Install the Python package

Ensure the virtual environment is still active, then install from the `python/` subdirectory:

```bash
cd python
pip install .
cd ~
```

#### Step 6 — Verify

```bash
python -c "import decord; print(decord.__version__)"
```

A version string (e.g. `0.6.0`) confirms a successful build. If the import fails, confirm
the virtual environment was active during Steps 2–5 and that both patches in Step 3 applied.

#### Step 7 — Re-run requirements.txt

With `decord` now built and installed, return to the `void-model/` directory and re-run
`requirements.txt` to ensure all remaining dependencies are present and consistent.
`decord` will be skipped since it is already installed.

```bash
source ~/void-env/bin/activate
cd ~/void-model
pip install -r requirements.txt
```

Confirm no errors are reported. Any package that previously failed due to a missing `decord`
dependency should now resolve correctly.

---

## 5 — Install SAM2

Required for the mask generation pipeline.

```bash
cd ~
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
cd ~/void-model
```

---

## 6 — Install Hugging Face CLI & Authenticate

```bash
pip install huggingface_hub
```

Verify the CLI installed correctly and check whether an HF token is already set:

```bash
# Check hf CLI version
hf --version

# Check if HF token is already set in the environment
echo $HF_TOKEN
```

If `$HF_TOKEN` is empty, authenticate now:

```bash
hf auth login
# Paste your HF token when prompted (generate at huggingface.co/settings/tokens)
```

To persist the token across sessions:

```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

---

## 7 — Download Base Model

```bash
cd ~/void-model

huggingface-cli download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP \
    --local-dir ./CogVideoX-Fun-V1.5-5b-InP
```

> ~20–40 GB download. The 128 GB unified memory on the ZGX Nano accommodates
> the full model in memory without CPU offloading.

---

## 8 — Download VOID Checkpoints

```bash
huggingface-cli download netflix/void-model \
    --local-dir .
```

Checkpoints downloaded:

| File | Purpose | Required? |
|------|---------|-----------|
| `void_pass1.safetensors` | Base inpainting model | **Yes** |
| `void_pass2.safetensors` | Temporal consistency refinement | Optional |

---

## 9 — Symlink ffmpeg (if needed)

```bash
ln -sf $(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") \
    ~/.local/bin/ffmpeg

export PATH="$HOME/.local/bin:$PATH"
```

---

## 10 — Set Gemini API Key (for mask generation)

Required only if using the VLM-MASK-REASONER pipeline (Stage 1 of mask creation).

**Create a Gemini API key:**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API key** and copy the generated key

**Set the key in your environment:**

```bash
export GEMINI_API_KEY=your_key_here

# To persist across sessions:
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

**How mask `.mp4` files are created:**

The VLM-MASK-REASONER pipeline generates `quadmask_0.mp4` automatically in four stages:

| Stage | Script | What it does | Output |
|-------|--------|-------------|--------|
| 0 | `point_selector_gui.py` | GUI — click the object to remove | `*_points.json` |
| 1 | `stage1_sam2_segmentation.py` | SAM2 segments the selected object per frame | `black_mask.mp4` |
| 2 | `stage2_vlm_analysis.py` | Gemini reasons about interaction-affected regions | `vlm_analysis.json` |
| 3 | `stage3a_generate_grey_masks_v2.py` | Builds grey overlay for affected areas | `grey_mask.mp4` |
| 4 | `stage4_combine_masks.py` | Merges black + grey into final quadmask | `quadmask_0.mp4` |

Run all stages after the GUI step with a single command:

```bash
bash VLM-MASK-REASONER/run_pipeline.sh my_config_points.json --device cuda
```

---

## 11 — Verify Directory Structure

```bash
ls ~/void-model/
```

Expected layout after setup:

```
void-model/
├── assets/
├── config/
├── data_generation/
├── datasets/
├── inference/
├── scripts/
├── videox_fun/
├── VLM-MASK-REASONER/
├── sample/                          # included sample sequences
│   └── my-video/                    # place custom video folders here
│       ├── input_video.mp4
│       ├── quadmask_0.mp4
│       └── prompt.json
├── CogVideoX-Fun-V1.5-5b-InP/      # base model (~20–40 GB)
├── void_pass1.safetensors           # VOID Pass 1 checkpoint
├── void_pass2.safetensors           # VOID Pass 2 checkpoint (optional)
├── gitattributes
├── LICENSE
├── notebook.ipynb
├── README.md
└── requirements.txt
```

> Custom video folders (e.g. `my-video/`) should be placed inside the `sample/` directory so that `--config.data.data_rootdir="./sample"` resolves them correctly during inference.

---

## 12 — Run Inference (Sample Video)

### Pass 1 — Base inference (sufficient for most videos)

```bash
cd ~/void-model

python inference/cogvideox_fun/predict_v2v.py \
    --config config/quadmask_cogvideox.py \
    --config.data.data_rootdir="./sample" \
    --config.experiment.run_seqs="lime" \
    --config.experiment.save_path="./outputs" \
    --config.video_model.model_name="./CogVideoX-Fun-V1.5-5b-InP" \
    --config.video_model.transformer_path="./void_pass1.safetensors"
```

Output saved to: `./outputs/lime.mp4`  
Side-by-side comparison: `./outputs/lime_tuple.mp4`

### Pass 2 — Temporal refinement (optional, longer clips)

```bash
python inference/cogvideox_fun/inference_with_pass1_warped_noise.py \
    --video_name lime \
    --data_rootdir ./sample \
    --pass1_dir ./outputs \
    --output_dir ./outputs_pass2 \
    --model_checkpoint ./void_pass2.safetensors \
    --model_name ./CogVideoX-Fun-V1.5-5b-InP
```

---

## 13 — Run on a Custom Video

### Step 13a — Install GUI and pipeline dependencies

The mask generation pipeline requires `tkinter` for the point selector GUI and several
additional system and Python packages. These are not installed by `requirements.txt` and
must be set up before running any VLM-MASK-REASONER scripts.

**Install `tkinter` (system package — cannot be installed via pip):**

```bash
sudo apt install python3-tk -y
```

Verify it is available in the virtual environment:

```bash
python -c "import tkinter; print('tkinter OK')"
```

**Install remaining pipeline dependencies:**

```bash
pip install google-generativeai openai opencv-python-headless Pillow requests
```

> `opencv-python-headless` is used instead of `opencv-python` because the ZGX Nano
> runs headless for most pipeline stages and avoids display server conflicts.
> `openai` is required by `stage2_vlm_analysis.py` even when using the Gemini path —
> omitting it causes `ModuleNotFoundError: No module named 'openai'` at Stage 2.

**Pre-flight checks — verify all pipeline dependencies before running:**

```bash
python -c "import tkinter; print('tkinter      OK')"
python -c "import google.generativeai; print('generativeai OK')"
python -c "import openai; print('openai       OK')"
python -c "import cv2; print('opencv       OK')"
python -c "import PIL; print('Pillow       OK')"
python -c "import sam2; print('SAM2         OK')"
```

All six should print `OK`. Fix any that fail before running the pipeline.

**Verify SAM2 is installed** (required for Stage 1 segmentation):

```bash
python -c "import sam2; print('SAM2 OK')"
```

If this fails, reinstall SAM2 from Step 5:

```bash
cd ~/sam2 && pip install -e . && cd ~/void-model
```

**Verify the SAM2 checkpoint exists at the correct location:**

```bash
ls ~/Desktop/NetflixVOID/sam2_hiera_large.pt
```

If missing, download it from `void-model/`:

```bash
cd ~/Desktop/NetflixVOID/void-model
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -O ../sam2_hiera_large.pt
```

> The pipeline script (`run_pipeline.sh`) looks for the checkpoint at `../sam2_hiera_large.pt`
> relative to `void-model/` — i.e. at `NetflixVOID/sam2_hiera_large.pt`. Placing it
> anywhere else will cause `❌ Checkpoint not found`.

---

### Step 13b — Set up your video folder

Each video needs its own folder inside `sample/` with three files:

```
sample/
└── my-video/
    ├── input_video.mp4     # source video
    ├── quadmask_0.mp4      # 4-value segmentation mask (0=remove, 63=overlap, 127=affected, 255=keep)
    └── prompt.json         # {"bg": "scene description after removal"}
```

Create the folder and add your video:

```bash
mkdir -p ~/void-model/sample/my-video
cp /path/to/your/video.mp4 ~/void-model/sample/my-video/input_video.mp4
```

Create the prompt file (edit the description to match your scene after removal):

```bash
echo '{"bg": "description of scene after object is removed"}' \
    > ~/void-model/sample/my-video/prompt.json
```

---

### Step 13c — Generate masks automatically (SAM2 + Gemini)

The default VLM-MASK-REASONER pipeline uses the Gemini API to reason about interaction-affected
regions (Stage 2). A Gemini API key is required for this path.

**Get a Gemini API key:**

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **Create API key** and copy the generated key
4. Set and persist the key:

```bash
export GEMINI_API_KEY=your_key_here
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

Verify the key is set before running the pipeline:

```bash
echo $GEMINI_API_KEY   # should print your key, not blank
```

**Step 0 — Select object points via GUI:**

```bash
python VLM-MASK-REASONER/point_selector_gui.py
```

> This opens a Tkinter window. If running over SSH, ensure X11 forwarding is enabled
> (`ssh -X user@host`) or run directly on the ZGX Nano's local display.

**Steps 1–4 — Run the full mask pipeline:**

```bash
bash VLM-MASK-REASONER/run_pipeline.sh VLM-MASK-REASONER/mask_config_points.json
```

> The config file path must be relative to `void-model/` where the command is run.
> The GUI saves the config as `mask_config_points.json` inside `VLM-MASK-REASONER/`.
> Always pass the full relative path `VLM-MASK-REASONER/mask_config_points.json` —
> passing just `my_config_points.json` causes `FileNotFoundError` because the script
> looks for it in the current working directory (`void-model/`), not in `VLM-MASK-REASONER/`.

---

#### Open Source Alternative — Grounded-SAM-2 (no API key required)

If you prefer not to use the Gemini API, **Grounded-SAM-2** is a fully open source
alternative for generating the primary object segmentation mask (`black_mask.mp4`).
It combines **Grounding DINO** (text-prompted object detection) with **SAM2** (mask
generation and video tracking) — no external API key needed.

> **Important:** Grounded-SAM-2 produces a binary black/white segmentation mask for the
> primary object only (equivalent to Stage 1 output). It does not reason about
> interaction-affected regions (the grey `127` layer). You will need to draw the affected
> region manually using the GUI editor (`VLM-MASK-REASONER/edit_quadmask.py`) or set the
> affected region to `255` (keep) if interaction effects are not needed for your video.

**Install Grounded-SAM-2:**

```bash
cd ~
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
pip install -e .
pip install supervision transformers
```

**Download SAM2 checkpoint:**

```bash
cd ~/Grounded-SAM-2/checkpoints
bash download_ckpts.sh
```

**Run on your video with a text prompt:**

```python
# grounded_sam2_mask.py — save this script and run from ~/Grounded-SAM-2/
import torch
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

TEXT_PROMPT = "person"          # describe the object to remove
VIDEO_PATH  = "../void-model/sample/my-video/input_video.mp4"
OUTPUT_PATH = "../void-model/sample/my-video/black_mask.mp4"

# Load Grounding DINO
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
detector  = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny").cuda()

# Load SAM2 video predictor
predictor = build_sam2_video_predictor(
    "sam2_hiera_large.yaml",
    "checkpoints/sam2_hiera_large.pt"
)
# ... (run detection + propagate masks across frames, save as black_mask.mp4)
```

```bash
python grounded_sam2_mask.py
```

Once `black_mask.mp4` is generated, manually create a simple `quadmask_0.mp4` where
the black region (value `0`) marks the object to remove and everything else is white
(`255`), then skip the Gemini stages and run inference directly:

```bash
# Use edit_quadmask.py to refine or add the affected region (grey=127) if needed
python VLM-MASK-REASONER/edit_quadmask.py
```

See [github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
for full documentation and advanced usage.

---

### Step 13d — Run inference on your video

```bash
python inference/cogvideox_fun/predict_v2v.py \
    --config config/quadmask_cogvideox.py \
    --config.data.data_rootdir="./sample" \
    --config.experiment.run_seqs="my-video" \
    --config.experiment.save_path="./outputs" \
    --config.video_model.model_name="./CogVideoX-Fun-V1.5-5b-InP" \
    --config.video_model.transformer_path="./void_pass1.safetensors"
```

---

## Key Inference Options

| Flag | Default | Notes |
|------|---------|-------|
| `--config.data.sample_size` | `384x672` | Output resolution (HxW) |
| `--config.data.max_video_length` | `197` | Max frames |
| `--config.video_model.num_inference_steps` | `50` | Denoising steps |
| `--config.video_model.guidance_scale` | `1.0` | CFG scale |
| `--config.system.gpu_memory_mode` | `model_cpu_offload_and_qfloat8` | Use `model_full_load` on ZGX Nano (128 GB fits full model) |

> On the ZGX Nano, set `--config.system.gpu_memory_mode=model_full_load` for best
> performance since the full 128 GB unified memory is available to both CPU and GPU.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `git lfs install` fails | Install manually: `sudo apt install git-lfs -y`, then re-run `git lfs install` |
| `RuntimeError: Cannot get CUDA generator` | CPU-only torch wheel installed. Run: `pip uninstall torch torchvision torchaudio -y` then `uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" --index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`. Do not include torchaudio — no aarch64 cu130 wheel exists. |
| `libcudart.so.12: cannot open shared object` | Same root cause as above — wrong CUDA build. Reinstall PyTorch from the `cu130` index (see Step 4a) |
| `nvidia-smi` not found | DGX OS drivers may need reinstall: `sudo apt install nvidia-driver-*` |
| ARM wheel not available | Build from source: `pip install <pkg> --no-binary :all:` |
| `ffmpeg` not found | Follow Step 9 to symlink the imageio-ffmpeg binary |
| CUDA out of memory | Unlikely on 128 GB — if it occurs, set `gpu_memory_mode=model_cpu_offload` |
| SAM2 import error | Ensure SAM2 was installed with `pip install -e .` inside the `sam2/` directory |

### Post-Crash / Post-Reboot — NVIDIA Driver Lost After Kernel Update

After a hard crash or unexpected reboot the ZGX Nano may boot into a newer kernel
(e.g. `6.14.0` → `6.17.0-1014-nvidia`) for which the NVIDIA driver modules have not
yet been compiled. `nvidia-smi` will report `couldn't communicate with the NVIDIA driver`
and `modprobe nvidia` will fail with `Module not found`.

**Step 1 — Confirm the kernel changed and identify the installed driver:**

```bash
uname -r
dpkg -l | grep -E "nvidia-driver|nvidia-dkms"
```

**Step 2 — Rebuild driver modules for the new kernel using DKMS:**

```bash
sudo apt update
sudo apt install --reinstall nvidia-dkms-580 nvidia-driver-580 -y
sudo dkms autoinstall
sudo reboot
```

> Replace `580` with the version shown in `dpkg -l` if different.

**Step 3 — After reboot, verify the driver and CUDA are restored:**

```bash
nvidia-smi
source ~/void-env/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected: `nvidia-smi` shows `Driver Version: 580.x` and `CUDA Version: 13.0`, and
PyTorch prints `CUDA: True` and `GPU: NVIDIA GB10`.

**Alternative — Use KVM or the DGX Dashboard if SSH is unavailable**

If the crash left the ZGX Nano in a state where SSH is unresponsive, connect directly
using a KVM switch (keyboard, video, mouse) or open the **HP ZGX / DGX Dashboard** from
a browser on the same network:

```
http://192.168.1.47   # replace with your ZGX Nano's IP
```

The dashboard provides system health status, driver version information, and the ability
to trigger driver updates and system restarts without requiring SSH access. Use it to
verify the GPU is detected and the correct driver version is active before returning to
the SSH workflow above.

### VLM-MASK-REASONER — `_tkinter.TclError: no display name and no $DISPLAY environment variable`

This error means the GUI has no display server to render into. This occurs when running
over a plain SSH session with no X11 forwarding. Two options:

**Option 1 — Run directly on the ZGX Nano's desktop (simplest)**

If you're connected via SSH, disconnect and run the script directly from a terminal
on the ZGX Nano's local desktop session instead. No additional configuration needed.

**Option 2 — Enable X11 forwarding over SSH**

Reconnect to the ZGX Nano from your local machine with X11 forwarding enabled:

```bash
# On your local machine, reconnect with:
ssh -X zgx-prod-2@<ip-address>

# Verify DISPLAY is set after connecting:
echo $DISPLAY   # should show something like localhost:10.0

# Then run:
cd ~/Desktop/NetflixVOID/void-model
source ~/void-env/bin/activate
python VLM-MASK-REASONER/point_selector_gui.py
```

> If `echo $DISPLAY` is still empty after reconnecting with `-X`, your local machine
> may not have an X server running. On macOS install [XQuartz](https://www.xquartz.org)
> first. On Windows use [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or
> [MobaXterm](https://mobaxterm.mobatek.net).

**Option 3 — Browser-based GUI via noVNC (recommended for SSH users)**

This streams the Tkinter window to any browser over the local network — no VNC client
or X server needed on your local machine.

> **Critical lessons from ZGX Nano testing:** Multiple stale `x11vnc` and `websockify`
> processes conflict silently. The new `x11vnc` will autopick port 5903 instead of 5900
> if 5900 is already taken, causing `websockify` to fail with `Connection refused` even
> though everything appears to be running. Always kill all instances and use `-rfbport 5900
> -listen localhost` to force consistent binding. Run the full block below as a clean start.

```bash
# 1. Install dependencies (one time)
sudo apt install xvfb x11vnc novnc websockify -y

# 2. Kill any stale processes from previous attempts
pkill x11vnc; pkill websockify; pkill Xvfb
sleep 2

# 3. Verify all ports are clear before continuing
ss -tlnp | grep -E "5900|5903|6080"   # must return nothing

# 4. Start virtual display
Xvfb :1 -screen 0 1920x1080x24 &
sleep 1

# 5. Start x11vnc — force port 5900, bind to localhost only
export DISPLAY=:1
x11vnc -display :1 -nopw -forever -shared -rfbport 5900 -listen localhost &
sleep 1

# 6. Confirm x11vnc is on 5900 before starting websockify
ss -tlnp | grep 5900   # must show x11vnc here — do not proceed if empty

# 7. Start noVNC websocket proxy
websockify --web /usr/share/novnc/ 6080 localhost:5900 &
sleep 1

# 8. Launch the GUI
export DISPLAY=:1
cd ~/Desktop/NetflixVOID/void-model
source ~/void-env/bin/activate
python VLM-MASK-REASONER/point_selector_gui.py &
```

Open a browser on your local machine and go to:

```
http://192.168.1.47:6080/vnc.html
```

Click **Connect** — the Tkinter GUI will appear and is fully interactive.

To stop everything when done:

```bash
pkill x11vnc; pkill websockify; pkill Xvfb
```

**Option 4 — Skip the GUI entirely (headless)**

Use the Grounded-SAM-2 open source alternative documented in Step 13c, which runs
fully headless with no display required.

---

### VLM-MASK-REASONER — Point Selector GUI: How to Use

The `point_selector_gui.py` Tkinter tool is used to mark the object you want to remove
across multiple video frames. It saves a `mask_config_points.json` file that the pipeline
reads for SAM2 segmentation.

**Workflow inside the GUI:**

1. **Load your video** — use the file picker to open `input_video.mp4` from your video folder inside `sample/`
2. **Navigate frames** — use the arrow buttons or slider to move through the video timeline
3. **Click points on the object** — left-click to add positive points (object to remove) on the target object across several frames. Spread points across at least 5–10 frames for better tracking
4. **Add negative points if needed** — right-click to mark areas the mask should NOT include (background regions the model might confuse with the object)
5. **Set the instruction** — type a short description in the instruction field, e.g. `"remove the sideline referee with the flag"`
6. **Save the config** — click **Save** or use the save button. The GUI writes `mask_config_points.json` into the `VLM-MASK-REASONER/` directory

**Where the config is saved:**

```
void-model/
└── VLM-MASK-REASONER/
    └── mask_config_points.json    # ← saved here by the GUI
```

**Run the pipeline using the saved config (from `void-model/`):**

```bash
cd ~/Desktop/NetflixVOID/void-model
bash VLM-MASK-REASONER/run_pipeline.sh VLM-MASK-REASONER/mask_config_points.json
```

**To modify an existing config** — re-open the GUI, load the same video, adjust points, and save again. The file is overwritten in place.

**Config file format** (for manual editing if needed):

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

### VLM-MASK-REASONER — Common Pipeline Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: my_config_points.json` | Config path not found in `void-model/` | Pass the full relative path: `VLM-MASK-REASONER/mask_config_points.json` |
| `ModuleNotFoundError: No module named 'openai'` | `openai` not installed | `pip install openai` |
| `❌ Checkpoint not found: ../sam2_hiera_large.pt` | SAM2 checkpoint in wrong location | Download to `NetflixVOID/`: `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O ../sam2_hiera_large.pt` |
| `UserWarning: cannot import name '_C' from 'sam2'` | SAM2 C extension not compiled | Safe to ignore — segmentation still works correctly |
| Stage 2 Gemini error | `GEMINI_API_KEY` not set | Run `echo $GEMINI_API_KEY` — if blank, re-export and source `~/.bashrc` |
| Stage 1 re-runs on every pipeline call | Expected behaviour — frames are re-extracted | Output from a completed stage is not cached between runs |

---

## References

- Model: [huggingface.co/netflix/void-model](https://huggingface.co/netflix/void-model)
- Project Page: [void-model.github.io](https://void-model.github.io/)
- GitHub: [github.com/netflix/void-model](https://github.com/netflix/void-model)
- Paper: [arxiv.org/abs/2604.02296](https://arxiv.org/abs/2604.02296)
- HP ZGX Nano: [hp.com/us-en/workstations/zgx-nano-ai-station](https://www.hp.com/us-en/workstations/zgx-nano-ai-station.html)
- NVIDIA DGX Spark Build: [build.nvidia.com/spark](https://build.nvidia.com/spark)
- License: Apache 2.0
