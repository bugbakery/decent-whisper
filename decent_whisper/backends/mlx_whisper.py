import importlib.util
from queue import SimpleQueue
import threading
from time import sleep
from typing import Iterable, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from threading import Thread

from decent_whisper import Word, TranscriptionInfo
from decent_whisper.model import ALL_LANGUAGES, TURBO_LANGUAGES, ModelInfo, path_for_model
from .mlx_whisper_transcribe import transcribe as transcribe_mlx


def name():
    return "mlx_whisper"


def should_use():
    return importlib.util.find_spec("mlx") is not None


def transcribe(
    audio: NDArray[np.float64] | str, model: ModelInfo, language: Optional[str] = None
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    queue = SimpleQueue()
    error_event = threading.Event()
    exception = None

    def work():
        def on_new_segment(segment: dict):
            queue.put(segment)

        def on_language_detected(lang: str):
            nonlocal language
            language = lang

        try:
            transcribe_mlx(
                audio,
                path_or_hf_repo=str(path_for_model(model)),
                language=language,
                word_timestamps=True,
                on_new_segment=on_new_segment,
                on_language_detected=on_language_detected,
                silent=True,
                hallucination_silence_threshold=0.5,
            )
            queue.put(None)
        except Exception as e:
            nonlocal exception
            exception = e
            error_event.set()
            queue.put(None)

    thread = Thread(target=work)
    thread.start()

    def map_segments():
        while True:
            segment = queue.get()
            if segment is None:
                if error_event.is_set():
                    raise exception or RuntimeError("Transcription failed")
                break

            yield [
                Word(
                    start=word["start"],
                    end=word["end"],
                    probability=word["probability"],
                    word=word["word"]
                )
                for word in segment["words"]
            ]

        thread.join()

    while language is None:
        if error_event.is_set():
            raise exception or RuntimeError("Transcription failed")

        sleep(0.1)

    return map_segments(), TranscriptionInfo(language=language)


def available_models():
    return [
        ModelInfo(
            name="large-v3-turbo",
            hf_repo="mlx-community/whisper-large-v3-turbo",
            size="large",
            languages=TURBO_LANGUAGES,
        ),
        ModelInfo(
            name="large-v3",
            hf_repo="mlx-community/whisper-large-v3-mlx",
            size="large",
            languages=TURBO_LANGUAGES,
        ),


        ModelInfo(
            name="medium.en",
            hf_repo="mlx-community/whisper-medium.en-mlx",
            size="medium",
            languages=["en"],
        ),
        ModelInfo(
            name="medium",
            hf_repo="mlx-community/whisper-medium-mlx",
            size="medium",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="small.en",
            hf_repo="mlx-community/whisper-small.en-mlx",
            size="small",
            languages=["en"],
        ),
        ModelInfo(
            name="small",
            hf_repo="mlx-community/whisper-small-mlx",
            size="small",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="base.en",
            hf_repo="mlx-community/whisper-base.en-mlx",
            size="base",
            languages=["en"],
        ),
        ModelInfo(
            name="base",
            hf_repo="mlx-community/whisper-base-mlx",
            size="base",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="tiny.en",
            hf_repo="mlx-community/whisper-tiny.en-mlx",
            size="tiny",
            languages=["en"],
        ),
        ModelInfo(
            name="tiny",
            hf_repo="mlx-community/whisper-tiny",
            size="tiny",
            languages=ALL_LANGUAGES,
        ),
    ]
