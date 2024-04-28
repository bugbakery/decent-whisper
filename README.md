# decent-whisper

use the fastest whisper implementation on every hardware

```diff
- this package is highly work-in-progress and not ready for usage yet
```

## backends

Currently, this package can dispatch to (in order of preference):

1. [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
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
