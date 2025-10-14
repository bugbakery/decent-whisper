import os
from decent_whisper.backends import mlx_whisper, faster_whisper
from decent_whisper.model import choose_model, download_model, is_model_downloaded

def run_test_for_backend(backend):
    model_info = choose_model(
        backend.available_models(),
        model_size="tiny"
    )
    assert model_info is not None

    if not is_model_downloaded(model_info):
        download_model(model_info, lambda loaded, total: None)


    iter, info = backend.transcribe(os.path.dirname(os.path.realpath(__file__)) + "/data/audio.mp3", model=model_info)

    for segment in iter:
        assert segment is not None

    assert info.language == "en" # how flaky is this?
    print(f"sucessfully tested {backend}")


def test_mlx_whisper():
    run_test_for_backend(mlx_whisper)


def test_faster_whisper():
    run_test_for_backend(faster_whisper)
