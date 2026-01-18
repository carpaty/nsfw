"""Simple NSFW image generation server using Diffusers and Flask."""
import argparse
import base64
import io
import os
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from flask import Flask, request, render_template_string, send_file


DEFAULT_PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
DEFAULT_NEGATIVE = ""

# Cache for loaded models
loaded_models: dict[str, DiffusionPipeline] = {}
current_model_state: dict[str, str | None] = {"name": None}

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

generator = torch.Generator(device="cuda").manual_seed(42)


@dataclass
class PromptSettings:
    """Sanitized prompt settings for image generation."""
    model: str = ""
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE


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


def build_prompt_settings(source: Mapping[str, Any] | None = None) -> PromptSettings:
    """Create normalized prompt settings from request data.

    :param source: Mapping of request fields (form or JSON).
    :type source: Optional[Mapping[str, Any]]
    :returns: Sanitized prompt settings.
    :rtype: PromptSettings
    """
    if not source:
        return PromptSettings(
            model=DEFAULT_MODEL,
            prompt=DEFAULT_PROMPT,
            negative=DEFAULT_NEGATIVE,
        )
    return PromptSettings(
        model=sanitize_model_name(source.get("model")),
        prompt=normalize_prompt_text(source.get("prompt"), DEFAULT_PROMPT),
        negative=normalize_optional_text(source.get("negative"), DEFAULT_NEGATIVE),
    )


def load_model(model_name: str) -> DiffusionPipeline:
    """Load the requested pipeline into cache, reusing if already loaded.

    :param model_name: Model directory name.
    :type model_name: str
    :returns: Cached pipeline instance.
    :rtype: DiffusionPipeline
    """
    if model_name not in loaded_models:
        model_path = os.path.join(MODELS_PATH, model_name)
        print(f"Loading model from: {model_path}")
        loaded_models[model_name] = DiffusionPipeline.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda"
        )
    current_model_state["name"] = model_name
    return loaded_models[model_name]


def render_template(settings: PromptSettings, image: str | None = None, error: str | None = None):
    """Render the HTML UI with the provided prompt details.

    :param settings: Current prompt settings for the user session.
    :type settings: PromptSettings
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


def generate_image(settings: PromptSettings) -> str:
    """Generate an image for the supplied prompt settings.

    :param settings: Sanitized prompt settings including model, prompt, and negative text.
    :type settings: PromptSettings
    :returns: Base64-encoded PNG of the generated image.
    :rtype: str
    """
    pipe = load_model(settings.model)
    image = pipe(
        settings.prompt,
        negative_prompt=settings.negative,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=generator,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def discover_models(models_path: str) -> list[str]:
    """Discover available models by scanning directory names in models_path.

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
    return parser.parse_args()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main HTML UI and handle image generation requests."""
    settings = PromptSettings()
    image_base64 = None
    error_message = None

    if request.method == 'POST':
        settings = build_prompt_settings(request.form)
        try:
            image_base64 = generate_image(settings)
        except (RuntimeError, ValueError) as exc:
            error_message = f"Unable to generate image: {exc}"
    return render_template(settings, image=image_base64, error=error_message)


@app.route('/v1', methods=['POST'])
def v1():
    """API endpoint for programmatic image generation."""
    settings = build_prompt_settings(request.get_json() if request.is_json else request.form)
    print(f"/v1 generating with model: {settings.model}, "
          f"prompt: {settings.prompt}, negative: {settings.negative}")
    pipe = load_model(settings.model)
    image = pipe(
        settings.prompt,
        negative_prompt=settings.negative,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=generator,
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


if __name__ == '__main__':
    args = parse_args()

    # Initialize model configuration
    MODELS_PATH = args.models_path
    MODEL_OPTIONS = discover_models(MODELS_PATH)
    DEFAULT_MODEL = MODEL_OPTIONS[0]

    print(f"Discovered models: {MODEL_OPTIONS}")
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Server starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
