from .backends import insanely_fast_whisper, lightning_whisper_mlx, faster_whisper

def get_backend() -> str:
    if insanely_fast_whisper.should_use():
        return "insanely-fast-whisper"
    elif lightning_whisper_mlx.should_use():
        return "lightning-whisper-mlx"
    else:
        return "faster-whisper"
