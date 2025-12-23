from typing import Tuple

import numpy as np
from faster_whisper.transcribe import Word
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)


def apply_vad_to_audio(
    audio, vad_parameters=None
) -> Tuple[np.ndarray, SpeechTimestampsMap]:
    if vad_parameters is None:
        vad_parameters = VadOptions()
    elif isinstance(vad_parameters, dict):
        vad_parameters = VadOptions(**vad_parameters)
    speech_chunks = get_speech_timestamps(audio, vad_parameters)
    audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
    audio = np.concatenate(audio_chunks, axis=0)
    speech_timestamps_map = SpeechTimestampsMap(speech_chunks, sampling_rate=16000)
    return audio, speech_timestamps_map


def remap_timestamps_iter(speech_timestamps_map: SpeechTimestampsMap, iter):
    def iterator():
        for segment in iter:
            yield [
                Word(
                    start=speech_timestamps_map.get_original_time(word["start"]),
                    end=speech_timestamps_map.get_original_time(word["end"]),
                    probability=word["probability"],
                    word=word["word"],
                )
                for word in segment["words"]
            ]

    return iterator()
