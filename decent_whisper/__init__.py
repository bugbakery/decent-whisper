import os
from pathlib import Path
from turtle import back
from typing import Iterable, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from decent_whisper.model import ModelInfo, is_model_downloaded


class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class TranscriptionInfo(NamedTuple):
    language: str


class Settings:
    models_dir: Path = Path(".models")
    cpu_threads: int = os.cpu_count() or 4


settings = Settings

def available_backends():
    from .backends import mlx_whisper, faster_whisper, transformers_whisper

    # these are ranked best-to-worst
    backends = [
        transformers_whisper,
        mlx_whisper,
        faster_whisper
    ]

    return [backend for backend in backends if backend.is_available()]

def get_backend(name=None):
    backends = available_backends()

    if name:
        if name not in backends:
            raise ValueError(f"Backend {name} is not available")
        else:
            return next(backend for backend in backends if backend.name() == name)
    else:
        return backends[0]

def transcribe(
    audio: NDArray[np.float64] | str, *, model: ModelInfo, language: str | None = None
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    backend = get_backend()

    if not is_model_downloaded(model):
        raise ValueError(f"Model {model.name} is not downloaded")

    return backend.transcribe(audio, model, language)


def available_models():
    return get_backend().available_models()


def downloaded_models():
    [model for model in available_models() if is_model_downloaded(model)]
