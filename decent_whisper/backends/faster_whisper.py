from typing import Iterable, List, Optional, Tuple
from decent_whisper import TranscriptionInfo, Word, TranscriptionInfo, settings


def transcribe(
    audio: List[float], model_name: str, lang_code: Optional[str] = "en"
) -> Tuple[Iterable[List[Word]], TranscriptionInfo]:
    import faster_whisper
    import faster_whisper.transcribe
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size_or_path=model_name,
        download_root=str(settings.model_dir.absolute()),
    )

    def map_iter(
        iter: Iterable[faster_whisper.transcribe.Segment],
    ) -> Iterable[List[Word]]:
        for segment in iter:
            yield segment.words

    seg_iter, info = model.transcribe(
        audio=audio, word_timestamps=True, language=lang_code
    )

    return map_iter(seg_iter), TranscriptionInfo(language=info.language)
