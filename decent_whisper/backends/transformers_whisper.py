from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from numpy.typing import NDArray
import numpy as np
from transformers.generation.streamers import BaseStreamer

from .. import TranscriptionInfo, Word
from ..model import ALL_LANGUAGES, TURBO_LANGUAGES, ModelInfo, path_for_model


def name():
    return "transformers"

def is_available():
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False

def should_use():
    try:
        import torch
        return torch.cuda.is_available()
        # TODO: check for flash attn, better heuristics
    except ImportError:
        return False


def is_flash_attn_2_available():
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def transcribe(
    audio: NDArray[np.float64] | str, model: ModelInfo, language: Optional[str] = None
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    import torch
    from transformers import pipeline, AutoProcessor, WhisperForConditionalGeneration
    from transformers.pipelines.audio_classification import ffmpeg_read

    if isinstance(audio, str):
        audio = ffmpeg_read(Path(audio).read_bytes(), 16000)

    """
    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.ndarray, "partial" bool}` With optionally a `"stride" (int, int)` key if
        `stride_length_s` is defined.
    """
    def chunk_audio(audio: NDArray[np.float64], chunk_length_s: float, stride_s: float):
        chunk_size = int(chunk_length_s * 16000)
        stride = int(stride_s * 16000)
        i = 0
        while True:
            regular_start = i * chunk_size
            start = max(0, regular_start - stride)
            end = min(len(audio), start + chunk_size)

            stride_start =
            stride_end =

            yield {
                "sampling_rate": 16000,
                "raw": audio[i:i+chunk_size],
                "partial": False,
                "stride": (None, None)
            }
            chunks.append(audio[i:i+chunk_size])

    return outputs, TranscriptionInfo(language=None)


def available_models():
    return [
        ModelInfo(
            name="large-v3-turbo",
            hf_repo="openai/whisper-large-v3-turbo",
            size="large",
            languages=TURBO_LANGUAGES,
        ),
        ModelInfo(
            name="large-v3",
            hf_repo="openai/whisper-large-v3",
            size="large",
            languages=TURBO_LANGUAGES,
        ),


        ModelInfo(
            name="medium.en",
            hf_repo="openai/whisper-medium.en",
            size="medium",
            languages=["en"],
        ),
        ModelInfo(
            name="medium",
            hf_repo="openai/whisper-medium",
            size="medium",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="small.en",
            hf_repo="openai/whisper-small.en",
            size="small",
            languages=["en"],
        ),
        ModelInfo(
            name="small",
            hf_repo="openai/whisper-small",
            size="small",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="base.en",
            hf_repo="openai/whisper-base.en",
            size="base",
            languages=["en"],
        ),
        ModelInfo(
            name="base",
            hf_repo="openai/whisper-base",
            size="base",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="tiny.en",
            hf_repo="openai/whisper-tiny.en",
            size="tiny",
            languages=["en"],
        ),
        ModelInfo(
            name="tiny",
            hf_repo="openai/whisper-tiny",
            size="tiny",
            languages=ALL_LANGUAGES,
        ),
    ]
