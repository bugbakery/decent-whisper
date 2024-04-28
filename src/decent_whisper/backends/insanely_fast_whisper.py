import torch
#from transformers import pipeline
#from transformers.utils import is_flash_attn_2_available

def should_use():
    return torch.cuda.is_available()

def load_model():
    pass

def transcribe():


    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0", # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    outputs = pipe(
        "<FILE_NAME>",
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    outputs