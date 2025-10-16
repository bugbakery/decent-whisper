from setuptools import setup
from platform import platform

mlx_deps = []
if "macOS" in platform() and "arm64" in platform():
    mlx_deps += ["mlx-whisper==0.4.1", "mlx>=0.12.0"]

setup(
    name="decent_whisper",
    version="0.0.1",
    install_requires=[
        "huggingface-hub~=0.34.1",
        "tqdm~=4.0",
        # faster whisper:
        "faster-whisper~=1.1.1",
        # mlx:
        *mlx_deps,
    ],
)
