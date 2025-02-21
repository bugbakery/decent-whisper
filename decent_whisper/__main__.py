import argparse
import signal
import sys
import threading

from decent_whisper.model import choose_model, download_model, is_model_downloaded
from . import available_models, get_backend, transcribe
from time import time
from faster_whisper.audio import decode_audio
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="decent_whisper",
    description="transcribe audio using the fastest whisper implementation on every hardware",
)

parser.add_argument('audio_file')
parser.add_argument('--model-size', help='which model size to use', choices=['tiny', 'base', 'small', 'medium', 'large'], default='large', required=False)
parser.add_argument('--lang', help='language code', default=None, required=False)
parser.add_argument('--use-single-language-models', help='allow to choose single language models', default=None, required=False)
parser.add_argument('--model', help='which model to use', default=None, required=False)

args = parser.parse_args()

def log(*args):
    print(*args, file=sys.stderr)

log(f"-> transcribing with backend {get_backend().name()}")

if args.model is not None:
    models = [model for model in available_models() if model.name == args.model]
    if len(models) > 0:
        model_info = models[0]
else:
    model_info = choose_model(
        available_models(),
        language=args.lang,
        use_single_language_models=args.use_single_language_models,
        model_size=args.model_size
    )

if not model_info:
    log("No matching model found")
    log("Available models:")
    for model in available_models():
        log("\t" + model.name)

    raise ValueError("No matching model found")

log("-> and model", model_info.name)

if not is_model_downloaded(model_info):
    log("-> downloading model...")
    cancel_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *args: cancel_event.set())

    progress_bar = None
    last_loaded = 0

    try:
        def on_progress(loaded, total):
            global last_loaded, progress_bar

            if progress_bar is None:
                progress_bar = tqdm(total=total, unit="B", unit_scale=True)

            global last_loaded
            progress_bar.update(loaded - last_loaded)
            last_loaded=loaded

        download_model(model_info, on_progress, cancel_event=cancel_event)
    except Exception as e:
        raise e
    finally:
        if progress_bar:
            progress_bar.close()

    log("-> model downloaded")

start = time()

iter, info = transcribe(
    args.audio_file,
    model=model_info,
    language=args.lang,
)

for segment in iter:
    print("".join([segment.word for segment in segment]), flush=True)

end = time()
duration = end - start
log(f"-> transcribed in {duration}s")

audio_duration = len(decode_audio(args.audio_file)) / 16000
log(f"-> realtime factor of {audio_duration / duration}")
