import os
from decent_whisper.backends import mlx_whisper, faster_whisper
from decent_whisper.model import choose_model

def run_test_for_backend(backend):
    model_info = choose_model(
        backend.available_models(),
        model_size="tiny"
    )
    assert model_info is not None

    iter, info = backend.transcribe(os.path.dirname(os.path.realpath(__file__)) + "/data/audio.mp3", model=model_info)

    for segment in iter:
        assert segment is not None

    assert info.language == "en" # how flaky is this?


def test_mlx_whisper():
    run_test_for_backend(mlx_whisper)


def test_faster_whisper():
    run_test_for_backend(faster_whisper)
