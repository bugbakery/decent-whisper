import asyncio
from dataclasses import dataclass
import shutil
import threading
from typing import Callable, List

import huggingface_hub
import requests


ALL_LANGUAGES = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
]

# Turbo performs poorly on some languages:
# https://github.com/openai/whisper/discussions/2363
TURBO_LANGUAGES = [lang for lang in ALL_LANGUAGES if lang not in ["yue", "ta", "th"]]


@dataclass
class ModelInfo:
    name: str
    hf_repo: str
    size: str
    languages: List[str]
    depcrecated: bool = False


def path_for_model(model: ModelInfo):
    from decent_whisper import settings
    return settings.models_dir / model.hf_repo


def download_model(
    model: ModelInfo,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: asyncio.Event | threading.Event | None = None,
):
    model_dir = path_for_model(model)
    model_dir.mkdir(exist_ok=True, parents=True)

    api = huggingface_hub.HfApi()
    repo_info = api.repo_info(model.hf_repo, files_metadata=True)

    if repo_info.siblings is None or len(repo_info.siblings) == 0:
        raise ValueError("No files found in the model repository")

    total = sum(f.size if f.size is not None else 0 for f in repo_info.siblings)
    loaded = 0

    try:
        for f in repo_info.siblings:
            url = huggingface_hub.hf_hub_url(model.hf_repo, f.rfilename)
            with open(model_dir / f.rfilename, "wb") as file:
                response = requests.get(url, stream=True)
                for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)
                ):
                    loaded += len(data)
                    file.write(data)

                    if progress_callback:
                        progress_callback(loaded, total)

                    if cancel_event and cancel_event.is_set():
                        raise ValueError("Model download cancelled")
    except Exception as e:
        shutil.rmtree(model_dir)
        raise e


def is_model_downloaded(model: ModelInfo):
    config_path = path_for_model(model) / "config.json"
    return config_path.exists()


def choose_model(
    models,
    *,
    model_size: str | None = None,
    language: str | None = None,
    use_single_language_models: bool = False,
) -> ModelInfo | None:
    if model_size:
        models = [model for model in models if model.size == model_size]

    if language is None or not use_single_language_models:
        models = [model for model in models if len(model.languages) > 1]

    if language:
        models = [model for model in models if language in model.languages]

    if len(models) == 0:
        return None

    return models[0]
