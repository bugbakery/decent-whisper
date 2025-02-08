import argparse
from . import get_backend, transcribe
from time import time

parser = argparse.ArgumentParser(
    prog="decent_whisper",
    description="transcribe audio using the fastest whisper implementation on every hardware",
)

parser.add_argument('audio_file')
parser.add_argument('--model', help='which model to use', choices=['tiny', 'base', 'small', 'medium', 'large'], default='large', required=False)
parser.add_argument('--lang', help='language code', default=None, required=False)

args = parser.parse_args()

print(f"-> transcribing with backend {get_backend().__name__}")

start = time()

iter, info = transcribe(args.audio_file, args.model, args.lang)
for segment in iter:
    pass

end = time()
duration = end - start
print(f"-> transcribed in {duration}s")

from mlx_whisper.audio import load_audio, SAMPLE_RATE
audio_duration = len(load_audio(args.audio_file)) / SAMPLE_RATE
print(f"-> realtime factor of {audio_duration / duration}")
