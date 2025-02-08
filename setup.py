from setuptools import setup
from platform import platform
from os import environ

mlx_deps = []
if "macOS" in platform() and "arm64" in platform():
    mlx_deps += ["mlx-whisper~=0.4.1", "mlx>=0.12.0"]

cuda_deps = []
try:
    # we expect torch with cuda to be installed before the fact since we cannot depend on it
    import torch

    if torch.cuda.is_available():
        cuda_deps += ["insanely-fast-whisper~=0.0.15", "transformers~=4.39.3"]
        if "CUDA_HOME" in environ:
            cuda_deps += ["flash-attn~=2.5.8"]
except ImportError:
    pass


setup(
    name="decent_whisper",
    version="0.0.1",
    install_requires=[
        # faster whisper:
        "pywhispercpp~=1.3.0",
        # cuda:
        *cuda_deps,
        # mlx:
        *mlx_deps,
    ],
)
