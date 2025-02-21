from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from decent_whisper import Word, TranscriptionInfo
from decent_whisper.model import ALL_LANGUAGES, TURBO_LANGUAGES, ModelInfo, path_for_model


def name():
    return "faster_whisper"


def transcribe(
    audio: NDArray[np.float64] | str, model: ModelInfo, language: Optional[str] = None,
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    import faster_whisper.transcribe
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel(
        model_size_or_path=str(path_for_model(model)),
    )

    def map_iter(
        iter: Iterable[faster_whisper.transcribe.Segment],
    ) -> Iterable[List[Word]]:
        for segment in iter:
            words = segment.words
            if words is not None:
                yield [
                    Word(
                        start=word.start,
                        end=word.end,
                        word=word.word,
                        probability=word.probability,
                    )
                    for word in words
                ]

    seg_iter, info = whisper_model.transcribe(
        audio=audio, word_timestamps=True, language=language, vad_filter=True
    )

    return map_iter(seg_iter), TranscriptionInfo(language=info.language)


def available_models():
    return [
        ModelInfo(
            name="large-v3-turbo",
            hf_repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            size="large",
            languages=TURBO_LANGUAGES,
        ),
        ModelInfo(
            name="large-v3",
            hf_repo="Systran/faster-whisper-large-v3",
            size="large",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="medium.en",
            hf_repo="Systran/faster-whisper-medium.en",
            size="medium",
            languages=["en"],
        ),
        ModelInfo(
            name="medium",
            hf_repo="Systran/faster-whisper-medium",
            size="medium",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="small.en",
            hf_repo="Systran/faster-whisper-small.en",
            size="small",
            languages=["en"],
        ),
        ModelInfo(
            name="small",
            hf_repo="Systran/faster-whisper-small",
            size="small",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="base.en",
            hf_repo="Systran/faster-whisper-base.en",
            size="base",
            languages=["en"],
        ),
        ModelInfo(
            name="base",
            hf_repo="Systran/faster-whisper-base",
            size="base",
            languages=ALL_LANGUAGES,
        ),


        ModelInfo(
            name="tiny.en",
            hf_repo="Systran/faster-whisper-tiny.en",
            size="tiny",
            languages=["en"],
        ),
        ModelInfo(
            name="tiny",
            hf_repo="Systran/faster-whisper-tiny",
            size="tiny",
            languages=ALL_LANGUAGES,
        ),
    ]
