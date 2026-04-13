# Netflix VOID Model - HP ZGX Nano Setup Guide

**Netflix VOID** (Video Object and Interaction Deletion) on the **HP ZGX Nano G1n** with **HP ZGX Toolkit VS Code Extension**

[![Model](https://img.shields.io/badge/HuggingFace-netflix%2Fvoid--model-orange)](https://huggingface.co/netflix/void-model)
[![Paper](https://img.shields.io/badge/arXiv-2604.02296-red)](https://arxiv.org/abs/2604.02296)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/netflix/void-model/blob/main/LICENSE)

<small>**Credit:** This guide is based on the official Netflix VOID model repository [github.com/Netflix/void-model](https://github.com/Netflix/void-model) for use with ZGX Nano companion GPU. </small>

---

## Preview

| Before | After |
|:------:|:-----:|
| ![Before. input_video](input_video.gif) | ![After. VOID output](output_video.gif) |
| `input_video.mp4` | `output.mp4` |

---

## Who This Guide is For

This guide was written for Media and Entertainment video content creators looking to use the HP ZGX Nano as an SSH remote GPU device with HP ZGX Toolkit to edit objects in videos and produce production quality outputs.

---

## Hardware

| Component | Spec |
|-----------|------|
| SoC | NVIDIA GB10 Grace Blackwell Superchip |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| GPU | NVIDIA Blackwell (6144 CUDA Cores, 5th Gen Tensor Cores) |
| Unified Memory | **128 GB** LPDDR5X @ 273 GB/s ✅ (VOID requires 40 GB+) |
| AI Performance | 1,000 TOPS (FP4) |
| Storage | 4 TB NVMe SSD |
| OS | NVIDIA DGX OS (Ubuntu 24.04 base) |

---

## What VOID Does

VOID removes objects from videos along with all physical interactions they induce. not just shadows and reflections, but effects like objects falling when a person is removed. It is built on CogVideoX and fine-tuned for video inpainting with interaction-aware quadmask conditioning.

---

## Prerequisites

The ZGX Nano ships with NVIDIA DGX OS. Verify your environment before starting.

```bash
uname -a && cat /etc/os-release
nvidia-smi
nvcc --version
python3 --version
```

> If `nvidia-smi` fails after a reboot, see [TROUBLESHOOTING.md → Post-Crash Driver Recovery](TROUBLESHOOTING.md#post-crash--post-reboot--nvidia-driver-lost-after-kernel-update).

---

# Setup

## 1. ZGX Nano System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    git git-lfs curl wget ffmpeg python3-tk \
    python3-pip python3-venv python3-dev \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0
git lfs install
```

## 2. Create and Activate Virtual Environment on ZGX Nano

```bash
python3 -m venv ~/void-env
source ~/void-env/bin/activate
pip install --upgrade pip setuptools wheel
```

> Add `source ~/void-env/bin/activate` to `~/.bashrc` to auto-activate on login.

## 3. Clone VOID Repository to Nano

```bash
git clone https://github.com/netflix/void-model.git
cd void-model
```

## 4. Install PyTorch (CUDA 13.0, GB10-specific)

> The standard `pip install torch` installs a CPU-only build on aarch64. Use `uv` with exact version pinning to get the verified CUDA 13.0 build. See [TROUBLESHOOTING.md → PyTorch CUDA](TROUBLESHOOTING.md#pytorch--cuda-130-installation) for full details.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

pip uninstall torch torchvision torchaudio -y

uv pip install \
    "torch==2.11.0+cu130" \
    "torchvision==0.26.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match
```

Verify CUDA before continuing:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected: `2.11.0+cu130 / CUDA: True / GPU: NVIDIA GB10`

## 5. Install Python Dependencies for VOID

```bash
pip install -r requirements.txt
pip install google-generativeai openai opencv-python-headless Pillow requests
```

> If `decord` fails to install, it must be built from source. See [TROUBLESHOOTING.md → Building decord](TROUBLESHOOTING.md#decord--manual-build-from-source).

## 6. Install SAM2

```bash
cd ~
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
cd ~/void-model
```

**NOTE**: Download the SAM2 checkpoint one level above `void-model/`:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
    -O ../sam2_hiera_large.pt
```

## 7. Authenticate with Hugging Face

```bash
pip install huggingface_hub
hf auth login
```

> Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). See [TROUBLESHOOTING.md → HF Token](TROUBLESHOOTING.md#hugging-face-authentication) for persistence.

### 7a. Request Access to Gated Models

Two models used in this pipeline require manual access approval on Hugging Face before they can be downloaded. Request access **before** running the pipeline. approval can take minutes to hours.

| Model | URL | Required for |
|-------|-----|-------------|
| `facebook/sam3` | [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) | Stage 3 grey mask generation |
| `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` | [huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) | Base inpainting model |

For each: visit the link → click **Agree and access repository** → submit the form. Once approved, your authenticated HF token will unlock the download automatically.

> See [TROUBLESHOOTING.md → Gated Model Access](TROUBLESHOOTING.md#gated-model-access--403-forbidden) for details on verifying access and re-running after approval.

## 8. Download Models

```bash
cd ~/void-model

# Base inpainting model (~20–40 GB)
huggingface-cli download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP \
    --local-dir ./CogVideoX-Fun-V1.5-5b-InP

# VOID checkpoints
huggingface-cli download netflix/void-model --local-dir .
```

## 9. Set Gemini API Key

Required for the VLM-MASK-REASONER pipeline (Stage 2). Stage 2 uses `gemini-3.1-pro` by default. **this model requires a paid Google AI Studio plan**. The free tier returns a `429` error immediately.

- Enable billing at [aistudio.google.com](https://aistudio.google.com), or
- Swap the model for a free alternative. see [TROUBLESHOOTING.md → Gemini Model Options](TROUBLESHOOTING.md#gemini-api--model-options)

```bash
export GEMINI_API_KEY=your_key_here
echo 'export GEMINI_API_KEY=your_key_here' >> ~/.bashrc
source ~/.bashrc
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

---

## Directory Structure

After setup, `void-model/` should look like this:

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
├── sample/
│   └── my-video/          # place custom video folders here
│       ├── input_video.mp4
│       ├── quadmask_0.mp4
│       └── prompt.json
├── CogVideoX-Fun-V1.5-5b-InP/
├── void_pass1.safetensors
├── void_pass2.safetensors
├── LICENSE
├── notebook.ipynb
└── requirements.txt
```

And one level above:

```
NetflixVOID/
├── sam2_hiera_large.pt    # ← required here
├── void-model/
└── void-env/
```

---

## Run Inference. Sample Video

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

Output: `./outputs/lime.mp4`. Side-by-side comparison: `./outputs/lime_tuple.mp4`

> **ZGX Nano tip:** Add `--config.system.gpu_memory_mode=model_full_load` to use the full 128 GB unified memory for best performance.

> **Before starting inference. clear GPU memory buffers:** If you have run other processes or the pipeline since your last session, clear cached GPU memory first to avoid fragmentation or out-of-memory errors during the ~6 minute inference run:
> ```bash
> python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('GPU memory cleared')"
> ```
> Check current GPU memory usage with `nvidia-smi` before and after to confirm.

---

# Run on a Custom Video

## 1. Set up your video folder

### 1a. Create a new folder inside `sample/` named after your project and add the required files:

```bash
mkdir -p ~/void-model/sample/my-video
cp /path/to/your/video.mp4 ~/void-model/sample/my-video/input_video.mp4
```

Your folder must contain these three files before running the pipeline, be sure to rename files to match:

```
sample/
└── my-video/
    ├── input_video.mp4          # your source video
    ├── mask_config.json         # config you will use to load yoru video into the GUI (see GUI in step 2 below)
    └── prompt.json              # background description after removal
```

### 1b. Create `prompt.json`. edit the description to match **what the scene should look like** after the object is removed:

```bash
echo '{"bg": "description of scene after object is removed"}' \
    > ~/void-model/sample/my-video/prompt.json
```

Example `prompt.json`:
```json
{"bg": "An empty football sideline with grass and crowd in background."}
```

> Update both `mask_config.json` and `prompt.json` to match your folder name and file paths before running the pipeline. Path references inside `mask_config.json` should point to `sample/my-video/input_video.mp4` and `output_dir` must be `sample/my-video`.

---

## 2. Select points (GUI) and generate mask config

Before running the pipeline, create a `mask_config.json` directly in your project folder (`sample/my-video/`). Use it to load into the GUI to create points and save points. 

**Saving points from the GUI to your project folder will look similar to:**

```
{
  "videos": [
    {
      "video_path": "sample/my-video/input_video.mp4",
      "output_dir": "sample/my-video",
      "instruction": "remove the object from the scene",
      "points": {}
    }
  ]
}
```

> Set `instruction` to describe what to remove (e.g. `"remove the sideline referee with the flag"`). The `points` field will be populated by the GUI automatically after you select points by frame(s) and save as `mask_config_points.json`. Both `video_path` and `output_dir` will point to your project folder as well as all pipeline outputs (`black_mask.mp4`, `grey_mask.mp4`, `quadmask_0.mp4`, `vlm_analysis.json`). 

### 2a. Launch the point selector GUI:

```bash
export DISPLAY=:1   # if running over SSH via noVNC
python VLM-MASK-REASONER/point_selector_gui.py
```

> For GUI access over SSH, see [TROUBLESHOOTING.md → VNC Browser GUI](TROUBLESHOOTING.md#browser-based-gui-via-novnc-recommended-for-ssh-users).

### 2b. In the GUI: 

load `sample/my-video/mask_config.json` with will load your `input_video.mp4`, click points on the object across at least 10 frames, set the removal instruction, then **Save**. The GUI will write the updated config with points back to `sample/my-video/mask_config_points.json`.

After saving, verify the config looks correct:

```bash
cat sample/my-video/mask_config_points.json
```

Confirm `video_path`, `output_dir`, and `instruction` are set correctly before continuing.

### 2c. Run the full mask pipeline:

```bash
bash VLM-MASK-REASONER/run_pipeline.sh /void-model/sample/my-video/mask_config_points.json
```

> **Note:** Update the `mask_config_points.json` file path to match your current working directory. If you are running from inside `void-model/`, use `sample/my-video/mask_config_points.json`. If running from the parent directory (`NetflixVOID/`), use `void-model/sample/my-video/mask_config_points.json`. Run `pwd` to confirm your current location before executing.

All pipeline outputs will be written to `sample/my-video/`:

```
sample/my-video/
├── input_video.mp4
├── mask_config_points.json
├── prompt.json
├── black_mask.mp4         # Stage 1. SAM2 segmentation
├── vlm_analysis.json      # Stage 2. Gemini VLM analysis
├── grey_mask.mp4          # Stage 3. affected region mask
└── quadmask_0.mp4         # Stage 4. final combined mask (used by inference)
```

## 3. Run inference

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
| `--config.system.gpu_memory_mode` | `model_cpu_offload_and_qfloat8` | Use `model_full_load` on ZGX Nano |

---

## Troubleshooting

For details and step-by-resolutions of known issues, including CUDA errors, decord build, PyTorch wheel selection, Gemini quota, VNC GUI setup, driver recovery after crash, SAM3 BPE vocabulary errors. see:

**[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

---

## References

- Model: [huggingface.co/netflix/void-model](https://huggingface.co/netflix/void-model)
- Project Page: [void-model.github.io](https://void-model.github.io/)
- GitHub: [github.com/netflix/void-model](https://github.com/netflix/void-model)
- Paper: [arxiv.org/abs/2604.02296](https://arxiv.org/abs/2604.02296)
- HP ZGX Nano: [hp.com/us-en/workstations/zgx-nano-ai-station](https://www.hp.com/us-en/workstations/zgx-nano-ai-station.html)
- NVIDIA DGX Spark Build: [build.nvidia.com/spark](https://build.nvidia.com/spark)
