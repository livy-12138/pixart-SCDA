# Linux CUDA setup

This project is configured for Python 3.10, PyTorch 2.1.1, and NVIDIA CUDA
12.1 runtime wheels. The host NVIDIA driver must support CUDA 12.1 (driver
version 530 or newer). A locally installed CUDA toolkit is not required for
the normal application or training paths.

Create the environment from the repository root:

```bash
conda env create -f environment-linux.yml
conda activate pixart-linux
```

The first command installs the CUDA PyTorch wheels and the Python packages in
`requirements-linux.txt`. `mmcv` is pinned to 1.7.2 because the native training
scripts import `mmcv.runner`; mmcv 2.x removed that module.

Verify an NVIDIA GPU instance after it is started:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

The 512 model is already present at `models/PixArt-XL-2-512x512`. Start the
local Gradio application with:

```bash
DEMO_PORT=12345 python app/app_512.py
```

Set `PIXART_MODEL_PATH` only when the model lives elsewhere:

```bash
PIXART_MODEL_PATH=/data/PixArt-XL-2-512x512 DEMO_PORT=12345 python app/app_512.py
```

The application needs an NVIDIA GPU to load the pipeline. The CUDA-enabled
environment can be installed on a machine without a GPU, but generation cannot
be tested until an NVIDIA GPU is attached.
