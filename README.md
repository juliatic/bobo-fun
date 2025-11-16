# bobo-fun (Fork)

This repository is a fork of bobo-fun.

Upstream README: https://github.com/BoBo0037/bobo-fun/blob/main/README.md

## start.sh in this fork

The `start.sh` script provides a one-command launcher that:

- Ensures Python 3.11 via Conda or a local virtualenv.
- Installs project dependencies (`requirements.txt`) and `kivy`.
- Sets macOS Apple Silicon optimizations for PyTorch MPS.
- Redirects Hugging Face caches to the local `models` directory.
- Launches the Kivy UI (`ui_app.py`).

Environment variables configured by `start.sh`:

- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `ACCELERATE_USE_MPS_DEVICE=1`
- `PYTORCH_MPS_TENSOR_CORE_ENABLED=1`
- `TOKENIZERS_PARALLELISM=false`
- `HF_HUB_CACHE=$(pwd)/models`
- `HF_HOME=$(pwd)/models`

Run:

```sh
./start.sh
```
