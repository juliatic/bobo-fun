#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "bobo"; then
    conda activate bobo
  else
    conda create -y -n bobo python=3.11
    conda activate bobo
  fi
else
  PY_BIN=""
  if command -v python3.11 >/dev/null 2>&1; then
    PY_BIN="python3.11"
  elif command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    PY_BIN="$(brew --prefix)/bin/python3.11"
  fi
  if [ -z "$PY_BIN" ]; then
    echo "Python 3.11 is required. Please install Conda or python3.11."
    exit 1
  fi
  if [ ! -d ".venv" ]; then
    "$PY_BIN" -m venv .venv
  fi
  source .venv/bin/activate
  PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [ "$PY_VERSION" != "3.11" ]; then
    echo "Existing virtualenv uses Python $PY_VERSION; recreating with Python 3.11."
    deactivate || true
    rm -rf .venv
    "$PY_BIN" -m venv .venv
    source .venv/bin/activate
  fi
fi

PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_VERSION" != "3.11" ]; then
  echo "Detected Python $PY_VERSION. This project requires Python 3.11 for dependency compatibility."
  echo "Please ensure Python 3.11 is active and re-run start.sh."
  exit 1
fi

python -m pip install -U pip setuptools wheel
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install -r requirements.txt
python -m pip install kivy

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  else
    echo "ffmpeg not found. Please install ffmpeg."
  fi
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export ACCELERATE_USE_MPS_DEVICE=1
export PYTORCH_MPS_TENSOR_CORE_ENABLED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="$(pwd)/models"
export HF_HOME="$(pwd)/models"
mkdir -p "$HF_HUB_CACHE"

python ui_app.py