# NSFW Diffusers Server

Lightweight Flask app that exposes both image generation (`/`, `/image`) and text chat (`/chat`) via local Diffusers checkpoints.

> **Note:** The server can produce explicit content depending on the model. Only host it in controlled environments, honor model licenses, and comply with applicable laws.

## Features

- Web UI that drives Diffusers pipelines through a simple form
- `/image` REST endpoint for scripted image generation
- `/chat` endpoint that reuses the same local checkpoints for short text responses
- Automatic discovery of checkpoint directories under `models/`
- Optional `--debug` flag that surfaces CUDA memory diagnostics

## Requirements

- Python 3.10+
- CUDA-capable GPU (necessary for reasonable performance)
- See `requirements.txt` for dependency pinning

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Populate the `models/` directory with one or more Diffusers checkpoints (each in its own folder).

  Example using Hugging Face repository:

  ```bash
  git lfs install
  git clone https://huggingface.co/votepurchase/waiREALCN_v150 models/waiREALCN_v150
  ```
4. Run the server:

    ```bash
    python nsfw.py --models-path models
    ```

5. Visit `http://localhost:5000` to use the UI.

## Configuration and CLI Flags

| Flag | Description | Default |
| --- | --- | --- |
| `--host` | Host/IP to bind the Flask server | `0.0.0.0` |
| `--port` | Port to listen on | `5000` |
| `--models-path` | Directory containing model subfolders | `models` |
| `--debug` | Enable debug logging, including CUDA metrics | `false` |

## API Usage

### `POST /image`

Accepts JSON payload:

- `model` (string) — directory name under `models/`
- `prompt` (string)
- `negative` (string, optional)

CURL example (outputs PNG):

```bash
curl -X POST http://localhost:5000/image \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "prompt": "Astronaut in a jungle", "negative": ""}' \
  --output result.png
```

### `POST /chat`

Runs a Hugging Face text-generation pipeline and returns JSON:

Payload:
- `model` (string)
- `prompt` (string)

Example response:

```json
{ "model": "chat-mpt", "message": "<generated text>" }
```

Example CLI call:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "chat-mpt", "prompt": "Write a short poem about space."}'
```

## Troubleshooting

- **No models found**: Ensure `models/` contains at least one checkpoint directory and `--models-path` points to it.
- **CUDA errors**: Verify your PyTorch installation supports CUDA and drivers are installed.
- **GPU memory spikes**: Run with `--debug` to see `LOGGER.debug` metrics emitted by `free_gpu_cache` and confirm the pipeline cache is emptied.

## Security Guidance

- Do not expose the server publicly without authentication, rate limiting, and HTTPS/TLS termination.
- Validate user prompts if deploying in a multi-user setting.

## License

See `LICENSE`.
