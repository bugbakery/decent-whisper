import importlib.util
from queue import SimpleQueue
from time import sleep
from typing import Iterable, List, Optional, Tuple
from numpy.typing import NDArray
from threading import Thread

from decent_whisper import Word, TranscriptionInfo
from .mlx_whisper_transcribe import transcribe as transcribe_mlx


def should_use():
    return importlib.util.find_spec("mlx") is not None


def transcribe(
    audio: NDArray, model_name: str, lang_code: Optional[str] = "en"
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    if lang_code == "auto":
        language = None
    else:
        language = lang_code

    queue = SimpleQueue()

    def work():
        def on_new_segment(segment: dict):
            queue.put(segment)

        def on_language_detected(lang: str):
            nonlocal language
            language = lang

        transcribe_mlx(
            audio,
            # path_or_hf_repo=model_name,
            language=language,
            word_timestamps=True,
            on_new_segment=on_new_segment,
            on_language_detected=on_language_detected,
            silent=True,
        )
        queue.put(None)

    thread = Thread(target=work)
    thread.start()

    def map_segments():
        while True:
            segment = queue.get()
            if segment is None:
                break

            words = []
            for word in segment["words"]:
                words.append(Word(start=word["start"], end=word["end"], probability=word["probability"], word=word["word"]))
            yield words

        thread.join()

    while language is None:
        sleep(0.1)

    return map_segments(), TranscriptionInfo(language=language)
