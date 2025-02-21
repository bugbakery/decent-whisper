import os
from pathlib import Path
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


def get_backend():
    from .backends import insanely_fast_whisper, mlx_whisper, faster_whisper

    # if insanely_fast_whisper.should_use():
    #     return insanely_fast_whisper
    if mlx_whisper.should_use():
        return mlx_whisper
    else:
        return faster_whisper


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
