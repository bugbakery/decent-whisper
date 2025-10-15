# decent-whisper

use the fastest whisper implementation on every hardware

```diff
- this package is highly work-in-progress and not ready for usage yet
```

## backends

Currently, this package can dispatch to (in order of preference):

1. [transformers whisper with flash attn](https://huggingface.co/docs/transformers/model_doc/whisper)
   (On NVIDIA systems)
2. [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
   (On Apple Silicon)
3. [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (Everything else)

## installation

If you want to use insanely-fast-whisper (on an nvidia system), you have to install pytorch as
recommended [in the pytorch docs](https://pytorch.org/get-started/locally/) before. Also it is
recommended to install the CUDA-SDK and set the `$CUDA_HOME` environment variable to install
flash-attn.

## usage example

```py
from decent_whiser import available_models, transcribe
from decent_whisper.model import choose_model, download_model, is_model_downloaded

model_info = choose_model(
   available_models(),
   model_size="small",
)

if not model_info:
   raise ValueError("No matching model found")

if not is_model_downloaded(model_info):
   download_model(model_info)

iter, info = transcribe(
    "audio.mp3",
    model=model_info,
)

for segment in iter:
    print("".join([segment.word for segment in segment]))
```
