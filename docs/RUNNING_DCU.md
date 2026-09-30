# Linux DCU setup

This repository includes a separate environment for Hygon DCU nodes using
DTK 24.04 and the HIP/ROCm-compatible PyTorch 2.1.1 wheels.

```bash
conda env create -f environment-dcu.yml
conda activate pixart-dcu
source /opt/dtk/env.sh
./setup_dcu.sh
```

The setup script installs the DCU-compatible PyTorch packages and verifies the
HIP runtime. The host driver must expose `/dev/kfd` and a DCU device; if
`torch.cuda.device_count()` is zero, the driver/kernel module is unavailable
and must be fixed by the system administrator.

Run training with the standard CUDA-compatible device variable (the Hygon
runtime maps it to DCU devices):

```bash
source /opt/dtk/env.sh
CUDA_VISIBLE_DEVICES=0 ./train.sh
```

`xformers` is intentionally not installed because the upstream wheel is
CUDA-specific. The project falls back to PyTorch attention operators.
