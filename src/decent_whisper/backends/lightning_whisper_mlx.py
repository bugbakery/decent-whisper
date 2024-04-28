#from lightning_whisper_mlx import LightningWhisperMLX

def should_use():
    try:
        import mlx
        return True
    except ImportError:
        return False

def transcribe():
    whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

    text = whisper.transcribe(audio_path="/audio.mp3")['text']

    print(text)
