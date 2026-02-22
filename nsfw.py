# SPDX-License-Identifier: MIT
# Copyright (c) 2026 carpaty
"""Simple image generation server using Diffusers and Flask."""
from __future__ import annotations

import argparse
import base64
import json
import io
import logging
import os
import gc
import threading
import time
import pinggy
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from flask import Flask, request, render_template_string, send_file, jsonify, current_app
from transformers import Pipeline, pipeline

DEFAULT_PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
DEFAULT_NEGATIVE = ""

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for available models."""

    models_path: str
    model_options: list[str]
    default_model: str


class PipelineCache:
    """Cache for loaded image and text pipelines."""

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}

    def clear(self) -> None:
        if not torch.cuda.is_available():
            self._pipelines.clear()
            return

        LOGGER.debug("torch.cuda.memory_allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        LOGGER.debug("torch.cuda.memory_reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
        self._pipelines.clear()
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.debug("torch.cuda.memory_allocated: %fGB", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        LOGGER.debug("torch.cuda.memory_reserved: %fGB", torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)

    def load_img_pipeline(self, model_name: str, models_path: str) -> DiffusionPipeline:
        if model_name not in self._pipelines:
            self.clear()
            model_path = os.path.join(models_path, model_name)
            LOGGER.info("Loading model from: %s", model_path)
            self._pipelines[model_name] = DiffusionPipeline.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="cuda",
            )
        return self._pipelines[model_name]

    def load_text_pipeline(self, model_name: str, models_path: str) -> Pipeline:
        if model_name not in self._pipelines:
            self.clear()
            model_path = os.path.join(models_path, model_name)
            LOGGER.info("Loading text pipeline from: %s", model_path)
            device = 0 if torch.cuda.is_available() else -1
            self._pipelines[model_name] = pipeline(
                "text-generation",
                model=model_path,
                device=device,
            )
        return self._pipelines[model_name]

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


def get_app_config() -> AppConfig:
    """Return the active application configuration."""
    return current_app.config["APP_CONFIG"]


def get_pipeline_cache() -> PipelineCache:
    """Return the shared pipeline cache."""
    return current_app.config["PIPELINE_CACHE"]


def get_generator() -> torch.Generator:
    """Return the shared RNG for reproducible generations."""
    return current_app.config["GENERATOR"]


def sanitize_model_name(candidate: Any | None, config: AppConfig) -> str:
    """Return an allowed model name or the default.

    :param candidate: Value provided by the user.
    :type candidate: Optional[Any]
    :returns: A valid model name that is in ``config.model_options``.
    :rtype: str
    """
    if isinstance(candidate, str) and candidate in config.model_options:
        return candidate
    return config.default_model


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


def build_img_settings(source: Mapping[str, Any] | None, config: AppConfig) -> ImageSettings:
    """Create normalized prompt settings from request data.

    :param source: Mapping of request fields (form or JSON).
    :type source: Optional[Mapping[str, Any]]
    :returns: Sanitized prompt settings.
    :rtype: ImageSettings
    """
    if not source:
        return ImageSettings(
            model=config.default_model,
            prompt=DEFAULT_PROMPT,
            negative=DEFAULT_NEGATIVE,
        )
    return ImageSettings(
        model=sanitize_model_name(source.get("model"), config),
        prompt=normalize_prompt_text(source.get("prompt"), DEFAULT_PROMPT),
        negative=normalize_optional_text(source.get("negative"), DEFAULT_NEGATIVE),
    )


def build_chat_settings(source: Mapping[str, Any] | None, config: AppConfig) -> ChatSettings:
    """Create normalized chat settings from a JSON payload.

    :param source: Mapping of request fields (JSON).
    :type source: Optional[Mapping[str, Any]]
    :returns: Sanitized chat settings.
    :rtype: ChatSettings
    """

    if not source:
        return ChatSettings(
            model=config.default_model,
            prompt=DEFAULT_PROMPT,
        )
    return ChatSettings(
        model=sanitize_model_name(source.get("model"), config),
        prompt=normalize_prompt_text(source.get("prompt"), DEFAULT_PROMPT),
    )


def generate_chat_message(settings: ChatSettings) -> str:
    """Generate a chat response for the provided settings.

    :param settings: Sanitized chat settings including model and prompt.
    :type settings: ChatSettings
    :returns: Generated text from the pipeline.
    :rtype: str
    """

    config = get_app_config()
    cache = get_pipeline_cache()
    pipe = cache.load_text_pipeline(settings.model, config.models_path)
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
    config = get_app_config()
    cache = get_pipeline_cache()
    pipe = cache.load_img_pipeline(settings.model, config.models_path)
    result = pipe(
        settings.prompt,
        negative_prompt=settings.negative,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=get_generator(),
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
        models=get_app_config().model_options,
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
    parser.add_argument(
        "--enable-tunnels",
        action="store_true",
        help="Enable public pinggy tunnels.",
    )
    parser.add_argument(
        "--tunnel-register-url",
        default=None,
        help="Optional URL to register the NSFW HTTPS tunnel as JSON {'URL': '<tunnel_url>'}.",
    )
    parser.add_argument(
        "--tunnel-check-interval",
        type=int,
        default=1200,
        help="Seconds between pinggy tunnel health checks when tunnels are enabled.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging mode.")
    return parser.parse_args()


def configure_app(flask_app: Flask, config: AppConfig) -> None:
    """Attach runtime configuration and shared objects to the Flask app."""
    flask_app.config["APP_CONFIG"] = config
    flask_app.config["PIPELINE_CACHE"] = PipelineCache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flask_app.config["GENERATOR"] = torch.Generator(device=device).manual_seed(42)


def start_tunnels() -> dict[str, Any]:
    """Start public tunnels for the app and return tunnel metadata."""
    tunnel_nsfw = pinggy.start_tunnel(forwardto="localhost:5000")
    tunnel_ollama = pinggy.start_tunnel(forwardto="localhost:11434")
    return {
        "nsfw": tunnel_nsfw,
        "ollama": tunnel_ollama,
    }


def select_https_tunnel_url(tunnel: Any) -> str | None:
    """Select the first HTTPS URL from pinggy tunnel metadata."""
    candidates: list[str] = []
    single_url = getattr(tunnel, "url", None)
    if isinstance(single_url, str):
        candidates.append(single_url)
    urls = getattr(tunnel, "urls", None)
    if isinstance(urls, list | tuple):
        candidates.extend(url for url in urls if isinstance(url, str))
    for url in candidates:
        if url.startswith("https://"):
            return url
    return None


def register_pinggy_url(tunnel_url: str, endpoint_url: str) -> bool:
    """Register an NSFW tunnel URL with a coordination service endpoint."""
    payload = json.dumps({"url": tunnel_url, "provider": "nsfw"}).encode("utf-8")
    request = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = response.getcode()
            if 200 <= status_code < 300:
                LOGGER.info("Registered NSFW tunnel URL with %s", endpoint_url)
                return True
            LOGGER.warning(
                "Failed to register NSFW tunnel URL. status=%s endpoint=%s",
                status_code,
                endpoint_url,
            )
            return False
    except urllib.error.URLError as exc:
        LOGGER.warning("Failed to register NSFW tunnel URL: %s", exc)
        return False


def is_tunnel_url_reachable(tunnel_url: str, timeout: float = 5.0) -> bool:
    """Check whether a tunnel URL responds to an HTTP request."""
    request = urllib.request.Request(tunnel_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.getcode() < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def log_tunnel_urls(tunnels: dict[str, Any]) -> None:
    """Log tunnel URL metadata for NSFW and OLLAMA endpoints."""
    LOGGER.info("Public URL NSWF: %s", getattr(tunnels["nsfw"], "urls", []))
    LOGGER.info("Public URL OLLAMA: %s", getattr(tunnels["ollama"], "urls", []))


def maybe_register_nsfw_tunnel(tunnels: dict[str, Any], register_endpoint_url: str | None) -> None:
    """Register NSFW HTTPS tunnel URL when a registration endpoint is configured."""
    if not register_endpoint_url:
        LOGGER.info("Tunnel registration skipped. Set --tunnel-register-url to enable it.")
        return
    nsfw_https_url = select_https_tunnel_url(tunnels["nsfw"])
    if nsfw_https_url is None:
        LOGGER.warning("Could not find HTTPS NSFW tunnel URL in pinggy response")
        return
    register_pinggy_url(nsfw_https_url, register_endpoint_url)


def refresh_tunnels_if_needed(
    tunnels: dict[str, Any],
    register_endpoint_url: str | None,
) -> dict[str, Any]:
    """Recreate pinggy tunnels when the NSFW tunnel appears disconnected."""
    nsfw_https_url = select_https_tunnel_url(tunnels["nsfw"])
    if nsfw_https_url and is_tunnel_url_reachable(nsfw_https_url):
        return tunnels

    LOGGER.warning("NSFW tunnel disconnected or unreachable. Reconnecting pinggy tunnels.")
    new_tunnels = start_tunnels()
    log_tunnel_urls(new_tunnels)
    maybe_register_nsfw_tunnel(new_tunnels, register_endpoint_url)
    return new_tunnels


def tunnel_maintenance_worker(
    initial_tunnels: dict[str, Any],
    register_endpoint_url: str | None,
    check_interval_seconds: int,
) -> None:
    """Background worker that keeps pinggy tunnels alive and re-registers NSFW URL."""
    tunnels = initial_tunnels
    interval = max(5, check_interval_seconds)
    while True:
        time.sleep(interval)
        tunnels = refresh_tunnels_if_needed(tunnels, register_endpoint_url)


def prompt_len(value: str) -> int:
    """Return prompt length for safe logging without prompt content."""
    return len(value)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main HTML UI and handle image generation requests."""
    settings = ImageSettings()
    image_base64 = None
    error_message = None

    if request.method == 'POST':
        settings = build_img_settings(request.form, get_app_config())
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

    settings = build_img_settings(request.get_json(), get_app_config())
    LOGGER.info(
        "/image generating with model=%s prompt_len=%d negative_len=%d",
        settings.model,
        prompt_len(settings.prompt),
        prompt_len(settings.negative),
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

    settings = build_chat_settings(request.get_json(), get_app_config())
    LOGGER.info(
        "/chat generating with model=%s prompt_len=%d",
        settings.model,
        prompt_len(settings.prompt),
    )
    try:
        message = generate_chat_message(settings)
    except Exception as exc:
        LOGGER.exception("Chat generation failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"model": settings.model, "message": message})


if __name__ == '__main__':
    args = parse_args()
    tunnels = None
    if args.enable_tunnels:
        tunnels = start_tunnels()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Initialize model configuration
    model_options = discover_models(args.models_path)
    config = AppConfig(
        models_path=args.models_path,
        model_options=model_options,
        default_model=model_options[0],
    )
    configure_app(app, config)

    LOGGER.info("Discovered models: %s", config.model_options)
    LOGGER.info("Default model: %s", config.default_model)
    LOGGER.info("Server starting at http://%s:%s", args.host, args.port)
    if tunnels is not None:
        log_tunnel_urls(tunnels)
        maybe_register_nsfw_tunnel(tunnels, args.tunnel_register_url)
        worker = threading.Thread(
            target=tunnel_maintenance_worker,
            args=(tunnels, args.tunnel_register_url, args.tunnel_check_interval),
            daemon=True,
            name="pinggy-tunnel-maintenance",
        )
        worker.start()
        LOGGER.info(
            "Started pinggy tunnel maintenance thread with check interval=%ss",
            max(5, args.tunnel_check_interval),
        )
    else:
        LOGGER.info("Public tunnels disabled. Use --enable-tunnels to turn them on.")
    app.run(host=args.host, port=args.port, debug=False)
