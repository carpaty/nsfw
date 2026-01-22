"""Simple image generation server using Diffusers and Flask."""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import gc
import ngrok
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from flask import Flask, request, render_template_string, send_file, jsonify
from transformers import Pipeline, pipeline

DEFAULT_PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
DEFAULT_NEGATIVE = ""

LOGGER = logging.getLogger(__name__)

# Cache for loaded models
_PIPELINES = {}

# Model configuration (set during initialization)
MODELS_PATH: str = ""
MODEL_OPTIONS: list[str] = []
DEFAULT_MODEL: str = ""

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Generator</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        input, select { width: 100%; padding: 10px; font-size: 16px; margin-bottom: 10px; }
        button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        img { max-width: 100%; margin-top: 20px; border: 1px solid #ddd; }
        .loading { display: none; color: #666; margin-top: 10px; }
        label { font-weight: bold; display: block; margin-top: 10px; margin-bottom: 5px; }
        .error { color: #a94442; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>AI Image Generator</h1>
    <form method="POST">
        <label>Model:</label>
        <select name="model">
            {% for option in models %}
            <option value="{{ option }}" {% if option == model %}selected{% endif %}>{{ option }}</option>
            {% endfor %}
        </select>

        <label>Prompt:</label>
        <input type="text" name="prompt" placeholder="Enter your prompt..." value="{{ prompt }}" required>

        <label>Negative Prompt:</label>
        <input type="text" name="negative" placeholder="Enter your negative prompt..." value="{{ negative }}">

        <button type="submit">Generate Image</button>
    </form>
    <div class="loading" id="loading">Generating image...</div>
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
    {% if image %}
    <h3>Result:</h3>
    <img src="data:image/png;base64,{{ image }}" alt="Generated image">
    <p><strong>Model:</strong> {{ model }}</p>
    <p><strong>Prompt:</strong> {{ prompt }}</p>
    <p><strong>Negative:</strong> {{ negative }}</p>
    {% endif %}
    <script>
        document.querySelector('form').onsubmit = function() {
            document.getElementById('loading').style.display = 'block';
        }
    </script>
</body>
</html>
"""

app = Flask(__name__)

GENERATOR = torch.Generator(device="cuda").manual_seed(42)


@dataclass
class ImageSettings:
    """Sanitized prompt settings for image generation."""
    model: str = ""
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE


@dataclass
class ChatSettings:
    """Sanitized prompt settings for text generation."""

    model: str = ""
    prompt: str = DEFAULT_PROMPT


def free_gpu_cache(model_name: str) -> None:
    """Release CUDA memory held by the current pipeline.

    :param model_name: Model key that may still exist in the cache.
    :type model_name: str
    """
    if not torch.cuda.is_available():
        return

    LOGGER.debug("torch.cuda.memory_allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    LOGGER.debug("torch.cuda.memory_reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    _PIPELINES.pop(model_name, None)
    _PIPELINES.clear()
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.debug("torch.cuda.memory_allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    LOGGER.debug("torch.cuda.memory_reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)


def sanitize_model_name(candidate: Any | None) -> str:
    """Return an allowed model name or the default.

    :param candidate: Value provided by the user.
    :type candidate: Optional[Any]
    :returns: A valid model name that is in :data:`MODEL_OPTIONS`.
    :rtype: str
    """
    if isinstance(candidate, str) and candidate in MODEL_OPTIONS:
        return candidate
    return DEFAULT_MODEL


def normalize_prompt_text(value: Any | None, default: str) -> str:
    """Strip whitespace and fall back to ``default`` for empty prompt values.

    :param value: Raw prompt text supplied by the client.
    :type value: Optional[Any]
    :param default: Default fallback prompt.
    :type default: str
    :returns: Trimmed prompt text or the fallback default.
    :rtype: str
    """
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return default


def normalize_optional_text(value: Any | None, default: str) -> str:
    """Return trimmed optional text (negative prompt) or keep default.

    :param value: Raw text supplied by the client.
    :type value: Optional[Any]
    :param default: Default fallback.
    :type default: str
    :returns: Trimmed value or fallback.
    :rtype: str
    """
    if isinstance(value, str):
        return value.strip()
    return default


def build_img_settings(source: Mapping[str, Any] | None = None) -> ImageSettings:
    """Create normalized prompt settings from request data.

    :param source: Mapping of request fields (form or JSON).
    :type source: Optional[Mapping[str, Any]]
    :returns: Sanitized prompt settings.
    :rtype: ImageSettings
    """
    if not source:
        return ImageSettings(
            model=DEFAULT_MODEL,
            prompt=DEFAULT_PROMPT,
            negative=DEFAULT_NEGATIVE,
        )
    return ImageSettings(
        model=sanitize_model_name(source.get("model")),
        prompt=normalize_prompt_text(source.get("prompt"), DEFAULT_PROMPT),
        negative=normalize_optional_text(source.get("negative"), DEFAULT_NEGATIVE),
    )


def build_chat_settings(source: Mapping[str, Any] | None = None) -> ChatSettings:
    """Create normalized chat settings from a JSON payload.

    :param source: Mapping of request fields (JSON).
    :type source: Optional[Mapping[str, Any]]
    :returns: Sanitized chat settings.
    :rtype: ChatSettings
    """

    if not source:
        return ChatSettings(
            model=DEFAULT_MODEL,
            prompt=DEFAULT_PROMPT,
        )
    return ChatSettings(
        model=sanitize_model_name(source.get("model")),
        prompt=normalize_prompt_text(source.get("prompt"), DEFAULT_PROMPT),
    )


def load_img_pipeline(model_name: str) -> DiffusionPipeline:
    """Return a cached image pipeline for the requested model.

    :param model_name: Model directory name.
    :type model_name: str
    :returns: Cached pipeline instance.
    :rtype: DiffusionPipeline
    """
    if model_name not in _PIPELINES:
        free_gpu_cache(model_name)
        model_path = os.path.join(MODELS_PATH, model_name)
        LOGGER.info("Loading model from: %s", model_path)
        _PIPELINES[model_name] = DiffusionPipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    return _PIPELINES[model_name]


def load_text_pipeline(model_name: str) -> Pipeline:
    """Load or return a cached text-generation pipeline.

    :param model_name: Name of the model directory inside ``models/``.
    :type model_name: str
    :returns: A Hugging Face text-generation pipeline ready to run on CPU/GPU.
    :rtype: Pipeline
    """
    if model_name not in _PIPELINES:
        free_gpu_cache(model_name)
        model_path = os.path.join(MODELS_PATH, model_name)
        LOGGER.info("Loading text pipeline from: %s", model_path)
        device = 0 if torch.cuda.is_available() else -1
        _PIPELINES[model_name] = pipeline(
            "text-generation",
            model=model_path,
            device=device,
        )
    return _PIPELINES[model_name]


def generate_chat_message(settings: ChatSettings) -> str:
    """Generate a chat response for the provided settings.

    :param settings: Sanitized chat settings including model and prompt.
    :type settings: ChatSettings
    :returns: Generated text from the pipeline.
    :rtype: str
    """

    pipe = load_text_pipeline(settings.model)
    result = pipe(
        settings.prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        return_full_text=False,
    )
    del pipe
    return result[0]["generated_text"]


def generate_image(settings: ImageSettings) -> str:
    """Generate an image for the supplied prompt settings.

    :param settings: Sanitized prompt settings including model, prompt, and negative text.
    :type settings: ImageSettings
    :returns: Base64-encoded PNG of the generated image.
    :rtype: str
    """
    image = generate_pil_image(settings)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_pil_image(settings: ImageSettings):
    """Generate a PIL image from the supplied prompt settings.

    :param settings: Sanitized prompt settings including model, prompt, and negative text.
    :type settings: ImageSettings
    :returns: Generated PIL image.
    :rtype: PIL.Image.Image
    """
    pipe = load_img_pipeline(settings.model)
    result = pipe(
        settings.prompt,
        negative_prompt=settings.negative,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=GENERATOR,
    ).images[0]
    del pipe
    return result


def render_template(
    settings: ImageSettings,
    image: str | None = None,
    error: str | None = None,
) -> str:
    """Render the HTML UI with the provided prompt details.

    :param settings: Current prompt settings for the user session.
    :type settings: ImageSettings
    :param image: Base64-encoded image data if available.
    :type image: Optional[str]
    :param error: Optional error message to display to the user.
    :type error: Optional[str]
    :returns: Rendered HTML page.
    :rtype: str
    """
    return render_template_string(
        HTML,
        models=MODEL_OPTIONS,
        image=image,
        model=settings.model,
        prompt=settings.prompt,
        negative=settings.negative,
        error=error,
    )


def discover_models(models_path: str) -> list[str]:
    """Discover available models by scanning directory names in ``models_path``.

    :param models_path: Path to directory containing model subdirectories.
    :type models_path: str
    :returns: List of model directory names.
    :rtype: list[str]
    """
    path = Path(models_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Models path does not exist or is not a directory: {models_path}")

    models = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not models:
        raise ValueError(f"No model directories found in: {models_path}")

    return sorted(models)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for server configuration."""
    parser = argparse.ArgumentParser(description="Run the NSFW Diffusers server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind the Flask server to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on.")
    parser.add_argument("--models-path", default="models",
                        help="Path to directory containing model subdirectories.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging mode.")
    return parser.parse_args()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main HTML UI and handle image generation requests."""
    settings = ImageSettings()
    image_base64 = None
    error_message = None

    if request.method == 'POST':
        settings = build_img_settings(request.form)
        try:
            image_base64 = generate_image(settings)
        except (RuntimeError, ValueError) as exc:
            LOGGER.exception("Image generation failed")
            error_message = f"Unable to generate image: {exc}"
    return render_template(settings, image=image_base64, error=error_message)


@app.route('/image', methods=['POST'])
def img():
    """API endpoint for programmatic image generation."""
    if not request.is_json:
        return jsonify({"error": "POST /image requires a JSON payload."}), 415

    settings = build_img_settings(request.get_json())
    LOGGER.info(
        "/image generating with model=%s prompt=%s negative=%s",
        settings.model,
        settings.prompt,
        settings.negative,
    )
    image = generate_pil_image(settings)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint for text generation."""

    if not request.is_json:
        return jsonify({"error": "POST /chat requires a JSON payload."}), 415

    settings = build_chat_settings(request.get_json())
    LOGGER.info(
        "/chat generating with model=%s prompt=%s",
        settings.model,
        settings.prompt,
    )
    try:
        message = generate_chat_message(settings)
    except Exception as exc:
        LOGGER.exception("Chat generation failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"model": settings.model, "message": message})


if __name__ == '__main__':
    args = parse_args()
    tunnel_nsfw = ngrok.connect(5000)
    tunnel_ollama = ngrok.connect(11434)


    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Initialize model configuration
    MODELS_PATH = args.models_path
    MODEL_OPTIONS = discover_models(MODELS_PATH)
    DEFAULT_MODEL = MODEL_OPTIONS[0]

    LOGGER.info("Discovered models: %s", MODEL_OPTIONS)
    LOGGER.info("Default model: %s", DEFAULT_MODEL)
    LOGGER.info("Server starting at http://%s:%s", args.host, args.port)
    LOGGER.info("Public URL NSWF: %s", tunnel_nsfw.url())
    LOGGER.info("Public URL OLLAMA: %s", tunnel_ollama.url())
    app.run(host=args.host, port=args.port, debug=False)
