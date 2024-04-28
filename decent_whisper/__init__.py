from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple
import tempfile


class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float


class TranscriptionInfo(NamedTuple):
    language: str


class Settings(NamedTuple):
    model_dir: Path = Path(tempfile.mkdtemp())

settings = Settings

def get_backend() -> str:
    from .backends import insanely_fast_whisper, mlx_whisper, faster_whisper

    if insanely_fast_whisper.should_use():
        return insanely_fast_whisper
    elif mlx_whisper.should_use():
        return mlx_whisper
    else:
        return faster_whisper


def transcribe(
    audio: List[float], model_name: str, lang_code: Optional[str] = "en"
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    return get_backend().transcribe(audio, model_name, lang_code)
