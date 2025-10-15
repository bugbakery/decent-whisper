from os import path
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from numpy.typing import NDArray
import numpy as np
from regex.regex import Regex
from transformers.generation.streamers import BaseStreamer
from transformers.integrations.flex_attention import Offset
from transformers.models.auto import AutoModel
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)
from transformers.models.whisper.modeling_whisper import WhisperPreTrainedModel
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer

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

    # whisper_model = WhisperPreTrainedModel.from_pretrained(path_for_model(model))
    # tokenizer = WhisperTokenizer.from_pretrained(path_for_model(model))
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(path_for_model(model))

    pipe = pipeline(
        "automatic-speech-recognition",
        # model=whisper_model,
        model="openai/whisper-small",
        # tokenizer=tokenizer,
        # feature_extractor=feature_extractor,
        return_timestamps="word",
        device=0,
    )

    sampling_rate = 16000

    def chunk_audio(audio: NDArray[np.float64], chunk_length_s: float, stride_s: float):
        chunk_size = int(chunk_length_s * sampling_rate)
        stride = int(stride_s * sampling_rate)

        if stride * 2 >= chunk_size:
            raise ValueError(
                f"Stride needs to be strictly smaller than chunk_size: ({stride} * 2) vs {chunk_size}"
            )

        offset = 0

        while True:
            end = min(len(audio), offset + chunk_size)
            stride_left = min(offset, stride)
            stride_right = min(end - offset, stride)

            # if (offset + stride_left + stride_right) > len(audio):
            #     break

            print(
                {
                    "sampling_rate": sampling_rate,
                    "partial": False,
                    "start": offset,
                    "end": end,
                    "stride": (
                        stride_left,
                        stride_right,
                    ),
                    "len": len(audio[offset:end]),
                }
            )

            yield {
                "sampling_rate": sampling_rate,
                "raw": audio[offset:end],
                "partial": False,
                "stride": (stride_left, stride_right),
            }

            offset += chunk_size - stride_right - stride_left

    chunks = chunk_audio(audio, chunk_length_s=5, stride_s=1)

    for item in pipe(chunks):
        print(item)

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
