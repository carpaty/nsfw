# NSFW Diffusers Image Generator

Simple Flask server for running local Diffusers models with a minimal web UI and a JSON API.

> **Note**
> This project generates images from user prompts. Make sure you comply with local laws, model licenses, and your organization’s policies.

## Features

- Flask web UI for prompt-based image generation
- JSON API endpoint for programmatic use
- Local model discovery from a models folder
- Model caching for faster repeated generations

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Diffusers + PyTorch with CUDA support

## Quick Start

1) Create and activate a virtual environment.

2) Install dependencies:

```
pip install torch diffusers flask
```

3) Place one or more Diffusers model folders inside a `models/` directory (or provide `--models-path`).

	Example using `waiREALCN_v150` from Hugging Face:

	```bash
	git lfs install
	git clone https://huggingface.co/votepurchase/waiREALCN_v150 models/waiREALCN_v150
	```

4) Run the server:

```
python nsfw.py --models-path models
```

5) Open the UI in your browser:

```
http://localhost:5000
```

## Configuration

Command-line options:

```
--host         Host/IP to bind (default: 0.0.0.0)
--port         Port to listen on (default: 5000)
--models-path  Path to model directories (default: models)
```

## API Usage

### `POST /v1`

Accepts JSON or form data with the following fields:

- `model` (string) — directory name from `models/`
- `prompt` (string)
- `negative` (string, optional)

Example using `curl`:

```
curl -X POST http://localhost:5000/v1 \
	-H "Content-Type: application/json" \
	-d '{"model": "my-model", "prompt": "Astronaut in a jungle", "negative": ""}' \
	--output result.png
```

## Project Structure

```
.
├── nsfw.py
└── README.md
```

## Security Notes

- Do not expose this server to the public Internet without authentication and rate limiting.
- Consider running behind a reverse proxy with HTTPS.

## Troubleshooting

- **No models found**: Verify `--models-path` points to a folder containing model subdirectories.
- **CUDA errors**: Ensure your PyTorch build has CUDA support and the correct drivers are installed.

## License

See `LICENSE`.
