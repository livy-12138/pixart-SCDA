#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /opt/dtk/env.sh ]]; then
  echo "DTK environment file /opt/dtk/env.sh was not found" >&2
  exit 1
fi

source /opt/dtk/env.sh
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/rocm5.6"

if [[ "${CONDA_DEFAULT_ENV:-}" != "pixart-dcu" ]]; then
  echo "Create/activate the environment first:"
  echo "  conda env create -f ${repo_root}/environment-dcu.yml"
  echo "  conda activate pixart-dcu"
  exit 2
fi

python -m pip install --upgrade pip
python -m pip install --force-reinstall setuptools==69.5.1
python -m pip install --force-reinstall \
  torch==2.1.1+rocm5.6 torchvision==0.16.1+rocm5.6 torchaudio==2.1.1+rocm5.6
python -m pip install -r "${repo_root}/requirements-dcu.txt"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP runtime:", torch.version.hip)
print("DCU visible:", torch.cuda.device_count())
print("DCU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
