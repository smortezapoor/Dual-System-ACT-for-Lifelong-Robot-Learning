# A container that can train and evaluate this project on a CUDA GPU.
#
#   docker build -t dual-system-vla .
#   docker run --gpus all -v $(pwd)/outputs:/app/outputs dual-system-vla \
#       python scripts/00_check_setup.py
#
# We start from plain Ubuntu rather than an NVIDIA CUDA image, because the torch
# wheels bring their own CUDA libraries. What we DO need from the host is the
# graphics driver, which is why NVIDIA_DRIVER_CAPABILITIES below asks for
# "graphics" and not just "compute": without it, torch.cuda works and rendering
# silently does not.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# libglvnd and libegl provide the EGL loader that MuJoCo needs to render without
# a screen. libgl1 and libosmesa6 cover the software fallback paths.
# build-essential, cmake and pkg-config are needed to compile LeRobot's
# source-only dependencies (egl_probe / hf-egl-probe) at pip-install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv \
        git curl ca-certificates \
        build-essential cmake pkg-config \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libosmesa6 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# The virtual environment lives outside /app so that mounting your working copy
# over /app at run time does not hide it.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3.12 -m venv /opt/venv && pip install --upgrade pip

# torch on its own layer, so rebuilding after a code change does not re-download
# two gigabytes.
RUN pip install "torch>=2.7,<2.12" torchvision \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install "lerobot[training,libero]>=0.6.0" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
 && pip install matplotlib pandas pyarrow opencv-python nvidia-npp-cu12

# torchcodec (which reads the dataset videos) needs NVIDIA's NPP libraries, and
# the torch wheels do not include them. This file loads them at Python start-up.
# Without it, torchcodec fails to import and LeRobot falls back to a much slower
# decoder, with the reason buried in a long traceback.
RUN printf '%s\n' \
    "import ctypes, glob, os, site" \
    "folders = [os.path.join(s, 'nvidia', 'npp', 'lib') for s in site.getsitepackages()]" \
    "files = [f for d in folders for f in sorted(glob.glob(os.path.join(d, '*.so.12')), key=lambda p: (0 if 'libnppc.so' in p else 1))]" \
    "[ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL) for f in files if os.path.exists(f)]" \
    > /opt/venv/lib/python3.12/site-packages/zz_npp_preload.pth

# Importing libero with no config file asks a question on the terminal, which
# hangs any unattended run. Write the config now so it never asks.
ENV LIBERO_CONFIG_PATH=/opt/libero_config
RUN mkdir -p /opt/libero_config && python - <<'PY' && chmod -R a+rX /opt/libero_config
import importlib.util, os, yaml
root = os.path.dirname(importlib.util.find_spec("libero.libero").origin)
config = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "datasets": "/app/outputs/libero_datasets",
    "assets": os.path.join(root, "assets"),
}
os.makedirs("/opt/libero_config", exist_ok=True)
with open("/opt/libero_config/config.yaml", "w") as f:
    yaml.safe_dump(config, f)
PY

# NVIDIA EGL vendor ICD. The Container Toolkit injects libEGL_nvidia.so.0 when
# the "graphics" capability is requested, but on some toolkit versions it does
# NOT inject /usr/share/glvnd/egl_vendor.d/10_nvidia.json. Without it the GLVND
# loader falls back to Mesa llvmpipe (a CPU rasteriser): rendering still
# "works", 10-30x slower, and nvidia-smi keeps reporting a healthy GPU. The
# 10_ prefix sorts ahead of 50_mesa.json so NVIDIA wins when present.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '%s\n' \
      '{' \
      '    "file_format_version" : "1.0.0",' \
      '    "ICD" : {' \
      '        "library_path" : "libEGL_nvidia.so.0"' \
      '    }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
 && chmod 0644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /app
COPY . /app
RUN pip install --no-deps -e /app/policy

CMD ["python", "scripts/00_check_setup.py"]
