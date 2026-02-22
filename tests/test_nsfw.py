import importlib
import sys
import types
import base64
import json

import pytest


def _install_stub_modules() -> None:
    if "pinggy" not in sys.modules:
        pinggy = types.ModuleType("pinggy")
        pinggy.start_tunnel = lambda **_kwargs: types.SimpleNamespace(urls=[])
        sys.modules["pinggy"] = pinggy

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()

        class DummyGenerator:
            def __init__(self, device: str):
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int):
                self.seed = seed
                return self

        cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            memory_allocated=lambda _idx=0: 0,
            memory_reserved=lambda _idx=0: 0,
        )
        torch.Generator = DummyGenerator
        torch.cuda = cuda
        sys.modules["torch"] = torch

    if "diffusers" not in sys.modules:
        diffusers = types.ModuleType("diffusers")

        class DummyDiffusionPipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        diffusers.DiffusionPipeline = DummyDiffusionPipeline
        sys.modules["diffusers"] = diffusers

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.Pipeline = object
        transformers.pipeline = lambda *_args, **_kwargs: None
        sys.modules["transformers"] = transformers


def _import_nsfw_module():
    try:
        return importlib.import_module("nsfw")
    except ModuleNotFoundError as exc:
        if exc.name not in {"pinggy", "torch", "diffusers", "transformers"}:
            raise
        _install_stub_modules()
        sys.modules.pop("nsfw", None)
        return importlib.import_module("nsfw")


nsfw = _import_nsfw_module()


@pytest.fixture()
def app_config():
    return nsfw.AppConfig(
        models_path="models",
        model_options=["m1", "m2"],
        default_model="m1",
    )


@pytest.fixture()
def flask_app(app_config):
    app = nsfw.app
    app.testing = True
    with app.app_context():
        nsfw.configure_app(app, app_config)
    return app


def test_sanitize_model_name_accepts_known_model(app_config):
    assert nsfw.sanitize_model_name("m2", app_config) == "m2"


def test_sanitize_model_name_falls_back_to_default(app_config):
    assert nsfw.sanitize_model_name("unknown", app_config) == "m1"
    assert nsfw.sanitize_model_name(None, app_config) == "m1"


def test_normalize_prompt_text_trims_and_falls_back():
    assert nsfw.normalize_prompt_text("  hello  ", "default") == "hello"
    assert nsfw.normalize_prompt_text("   ", "default") == "default"
    assert nsfw.normalize_prompt_text(None, "default") == "default"


def test_normalize_optional_text_trims_and_defaults():
    assert nsfw.normalize_optional_text("  x  ", "default") == "x"
    assert nsfw.normalize_optional_text(None, "default") == "default"


def test_build_img_settings_from_mapping(app_config):
    settings = nsfw.build_img_settings(
        {"model": "m2", "prompt": " hi ", "negative": " lowres "},
        app_config,
    )
    assert settings.model == "m2"
    assert settings.prompt == "hi"
    assert settings.negative == "lowres"


def test_build_img_settings_with_missing_payload_uses_defaults(app_config):
    settings = nsfw.build_img_settings(None, app_config)
    assert settings.model == "m1"
    assert settings.prompt == nsfw.DEFAULT_PROMPT
    assert settings.negative == nsfw.DEFAULT_NEGATIVE


def test_build_chat_settings_from_mapping(app_config):
    settings = nsfw.build_chat_settings(
        {"model": "m2", "prompt": " hi "},
        app_config,
    )
    assert settings.model == "m2"
    assert settings.prompt == "hi"


def test_discover_models_success(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / ".hidden").mkdir()
    assert nsfw.discover_models(str(tmp_path)) == ["a", "b"]


def test_discover_models_missing_path_raises():
    with pytest.raises(ValueError):
        nsfw.discover_models("/definitely/missing/path")


def test_discover_models_empty_path_raises(tmp_path):
    with pytest.raises(ValueError):
        nsfw.discover_models(str(tmp_path))


def test_configure_app_uses_cpu_generator_when_cuda_unavailable(monkeypatch, app_config):
    class DummyGenerator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    monkeypatch.setattr(nsfw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nsfw.torch, "Generator", DummyGenerator)

    app = nsfw.app
    with app.app_context():
        nsfw.configure_app(app, app_config)
        generator = app.config["GENERATOR"]
        assert generator.device == "cpu"
        assert generator.seed == 42


def test_image_requires_json(flask_app):
    client = flask_app.test_client()
    response = client.post("/image", data="plain-text", content_type="text/plain")
    assert response.status_code == 415
    assert response.get_json()["error"] == "POST /image requires a JSON payload."


def test_chat_requires_json(flask_app):
    client = flask_app.test_client()
    response = client.post("/chat", data="plain-text", content_type="text/plain")
    assert response.status_code == 415
    assert response.get_json()["error"] == "POST /chat requires a JSON payload."


def test_chat_returns_500_on_generation_failure(flask_app, monkeypatch):
    client = flask_app.test_client()

    def boom(_settings):
        raise RuntimeError("model failed")

    monkeypatch.setattr(nsfw, "generate_chat_message", boom)
    response = client.post("/chat", json={"model": "m1", "prompt": "hello"})
    assert response.status_code == 500
    assert response.get_json()["error"] == "model failed"


def test_chat_success_response(flask_app, monkeypatch):
    client = flask_app.test_client()
    monkeypatch.setattr(nsfw, "generate_chat_message", lambda _settings: "ok")
    response = client.post("/chat", json={"model": "m1", "prompt": "hello"})
    assert response.status_code == 200
    assert response.get_json() == {"model": "m1", "message": "ok"}


def test_image_success_returns_png(flask_app, monkeypatch):
    client = flask_app.test_client()

    class FakeImage:
        def save(self, buf, format):
            assert format == "PNG"
            buf.write(b"\x89PNG\r\n\x1a\nfake")

    monkeypatch.setattr(nsfw, "generate_pil_image", lambda _settings: FakeImage())
    response = client.post("/image", json={"model": "m1", "prompt": "hello"})
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data.startswith(b"\x89PNG\r\n\x1a\n")


def test_chat_logs_prompt_length_not_content(flask_app, monkeypatch, caplog):
    client = flask_app.test_client()
    monkeypatch.setattr(nsfw, "generate_chat_message", lambda _settings: "ok")

    payload = {"model": "m1", "prompt": "super-secret-prompt"}
    with caplog.at_level("INFO"):
        response = client.post("/chat", json=payload)

    assert response.status_code == 200
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "prompt_len=" in log_text
    assert "super-secret-prompt" not in log_text


def test_getters_return_values_from_current_app(flask_app):
    with flask_app.app_context():
        assert nsfw.get_app_config() == flask_app.config["APP_CONFIG"]
        assert nsfw.get_pipeline_cache() == flask_app.config["PIPELINE_CACHE"]
        assert nsfw.get_generator() == flask_app.config["GENERATOR"]


def test_prompt_len():
    assert nsfw.prompt_len("abc") == 3
    assert nsfw.prompt_len("") == 0


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["nsfw.py"])
    args = nsfw.parse_args()
    assert args.host == "0.0.0.0"
    assert args.port == 5000
    assert args.models_path == "models"
    assert args.debug is False
    assert args.enable_tunnels is False
    assert args.tunnel_register_url is None
    assert args.tunnel_check_interval == 1200


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nsfw.py",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--models-path",
            "/x/models",
            "--debug",
            "--enable-tunnels",
            "--tunnel-register-url",
            "https://example.com/pinggy",
            "--tunnel-check-interval",
            "45",
        ],
    )
    args = nsfw.parse_args()
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.models_path == "/x/models"
    assert args.debug is True
    assert args.enable_tunnels is True
    assert args.tunnel_register_url == "https://example.com/pinggy"
    assert args.tunnel_check_interval == 45


def test_start_tunnels_calls_pinggy_twice(monkeypatch):
    calls = []

    def fake_start_tunnel(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(urls=[f"url-{len(calls)}"])

    monkeypatch.setattr(nsfw.pinggy, "start_tunnel", fake_start_tunnel)
    result = nsfw.start_tunnels()
    assert "nsfw" in result
    assert "ollama" in result
    assert calls == [
        {"forwardto": "localhost:5000"},
        {"forwardto": "localhost:11434"},
    ]


def test_select_https_tunnel_url_prefers_https_from_urls():
    tunnel = types.SimpleNamespace(
        url="http://plain.example",
        urls=["http://plain.example", "https://secure.example"],
    )
    assert nsfw.select_https_tunnel_url(tunnel) == "https://secure.example"


def test_select_https_tunnel_url_uses_single_https_url():
    tunnel = types.SimpleNamespace(url="https://secure.example")
    assert nsfw.select_https_tunnel_url(tunnel) == "https://secure.example"


def test_register_pinggy_url_posts_json_payload(monkeypatch):
    endpoint = "https://example.com/pinggy"
    calls = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout):
        calls["request"] = request
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(nsfw.urllib.request, "urlopen", fake_urlopen)
    ok = nsfw.register_pinggy_url("https://secure.example", endpoint)
    assert ok is True
    assert calls["timeout"] == 10
    assert calls["request"].full_url == endpoint
    assert calls["request"].get_method() == "POST"
    assert calls["request"].get_header("Content-type") == "application/json"
    assert json.loads(calls["request"].data.decode("utf-8")) == {
        "url": "https://secure.example",
        "provider": "nsfw",
    }


def test_register_pinggy_url_returns_false_on_url_error(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise nsfw.urllib.error.URLError("offline")

    monkeypatch.setattr(nsfw.urllib.request, "urlopen", fake_urlopen)
    assert nsfw.register_pinggy_url("https://secure.example", "https://example.com/pinggy") is False


def test_refresh_tunnels_if_needed_keeps_healthy_tunnel(monkeypatch):
    tunnels = {"nsfw": types.SimpleNamespace(urls=["https://secure.example"]), "ollama": object()}
    monkeypatch.setattr(nsfw, "select_https_tunnel_url", lambda _tunnel: "https://secure.example")
    monkeypatch.setattr(nsfw, "is_tunnel_url_reachable", lambda _url: True)
    monkeypatch.setattr(nsfw, "start_tunnels", lambda: pytest.fail("start_tunnels should not be called"))
    out = nsfw.refresh_tunnels_if_needed(tunnels, "https://example.com/pinggy")
    assert out is tunnels


def test_refresh_tunnels_if_needed_reconnects_and_registers(monkeypatch):
    old_tunnels = {"nsfw": types.SimpleNamespace(urls=["https://old.example"]), "ollama": object()}
    new_tunnels = {
        "nsfw": types.SimpleNamespace(urls=["https://new.example"], url="https://new.example"),
        "ollama": types.SimpleNamespace(urls=["https://ollama.example"]),
    }
    state = {"registered": None}
    monkeypatch.setattr(nsfw, "select_https_tunnel_url", lambda tunnel: getattr(tunnel, "url", None))
    monkeypatch.setattr(nsfw, "start_tunnels", lambda: new_tunnels)
    monkeypatch.setattr(nsfw, "log_tunnel_urls", lambda _tunnels: None)

    def fake_register(tunnel_url, endpoint_url):
        state["registered"] = (tunnel_url, endpoint_url)

    monkeypatch.setattr(nsfw, "register_pinggy_url", fake_register)
    out = nsfw.refresh_tunnels_if_needed(old_tunnels, "https://example.com/pinggy")
    assert out is new_tunnels
    assert state["registered"] == ("https://new.example", "https://example.com/pinggy")


def test_pipeline_cache_clear_cpu_only(monkeypatch):
    cache = nsfw.PipelineCache()
    cache._pipelines["x"] = object()
    monkeypatch.setattr(nsfw.torch.cuda, "is_available", lambda: False)
    cache.clear()
    assert cache._pipelines == {}


def test_pipeline_cache_clear_gpu_runs_cleanup(monkeypatch):
    cache = nsfw.PipelineCache()
    cache._pipelines["x"] = object()
    state = {"gc": 0, "empty_cache": 0}
    monkeypatch.setattr(nsfw.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nsfw.torch.cuda, "memory_allocated", lambda _idx=0: 0)
    monkeypatch.setattr(nsfw.torch.cuda, "memory_reserved", lambda _idx=0: 0)
    monkeypatch.setattr(nsfw.gc, "collect", lambda: state.__setitem__("gc", state["gc"] + 1))
    monkeypatch.setattr(
        nsfw.torch.cuda, "empty_cache", lambda: state.__setitem__("empty_cache", state["empty_cache"] + 1)
    )
    cache.clear()
    assert cache._pipelines == {}
    assert state["gc"] == 1
    assert state["empty_cache"] == 1


def test_pipeline_cache_load_img_pipeline_uses_cache(monkeypatch):
    cache = nsfw.PipelineCache()
    monkeypatch.setattr(cache, "clear", lambda: None)

    calls = []

    class DummyPipe:
        pass

    def fake_from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return DummyPipe()

    monkeypatch.setattr(nsfw.DiffusionPipeline, "from_pretrained", fake_from_pretrained)
    p1 = cache.load_img_pipeline("m1", "root")
    p2 = cache.load_img_pipeline("m1", "root")
    assert p1 is p2
    assert len(calls) == 1
    assert calls[0][0].endswith("root/m1")


def test_pipeline_cache_load_text_pipeline_uses_cache(monkeypatch):
    cache = nsfw.PipelineCache()
    monkeypatch.setattr(cache, "clear", lambda: None)
    monkeypatch.setattr(nsfw.torch.cuda, "is_available", lambda: False)
    calls = []

    def fake_pipeline(task, model, device):
        calls.append((task, model, device))
        return object()

    monkeypatch.setattr(nsfw, "pipeline", fake_pipeline)
    p1 = cache.load_text_pipeline("m2", "root")
    p2 = cache.load_text_pipeline("m2", "root")
    assert p1 is p2
    assert calls == [("text-generation", "root/m2", -1)]


def test_generate_chat_message_uses_expected_generation_args(monkeypatch):
    settings = nsfw.ChatSettings(model="m1", prompt="hello")

    class DummyPipe:
        def __call__(self, *args, **kwargs):
            assert args == ("hello",)
            assert kwargs["max_new_tokens"] == 80
            assert kwargs["do_sample"] is True
            assert kwargs["temperature"] == 0.8
            assert kwargs["return_full_text"] is False
            return [{"generated_text": "ok"}]

    monkeypatch.setattr(nsfw, "get_app_config", lambda: nsfw.AppConfig("models", ["m1"], "m1"))
    monkeypatch.setattr(
        nsfw,
        "get_pipeline_cache",
        lambda: types.SimpleNamespace(load_text_pipeline=lambda _m, _p: DummyPipe()),
    )
    assert nsfw.generate_chat_message(settings) == "ok"


def test_generate_pil_image_calls_pipeline_with_expected_args(monkeypatch):
    settings = nsfw.ImageSettings(model="m1", prompt="hello", negative="bad")
    sentinel_image = object()

    class DummyPipe:
        def __call__(self, *args, **kwargs):
            assert args == ("hello",)
            assert kwargs["negative_prompt"] == "bad"
            assert kwargs["num_inference_steps"] == 50
            assert kwargs["true_cfg_scale"] == 4.0
            assert kwargs["generator"] == "GEN"
            return types.SimpleNamespace(images=[sentinel_image])

    monkeypatch.setattr(nsfw, "get_app_config", lambda: nsfw.AppConfig("models", ["m1"], "m1"))
    monkeypatch.setattr(nsfw, "get_generator", lambda: "GEN")
    monkeypatch.setattr(
        nsfw,
        "get_pipeline_cache",
        lambda: types.SimpleNamespace(load_img_pipeline=lambda _m, _p: DummyPipe()),
    )
    assert nsfw.generate_pil_image(settings) is sentinel_image


def test_generate_image_returns_base64_png(monkeypatch):
    class FakeImage:
        def save(self, buf, format):
            assert format == "PNG"
            buf.write(b"abc")

    monkeypatch.setattr(nsfw, "generate_pil_image", lambda _settings: FakeImage())
    out = nsfw.generate_image(nsfw.ImageSettings())
    assert out == base64.b64encode(b"abc").decode()


def test_render_template_includes_values(monkeypatch, flask_app):
    settings = nsfw.ImageSettings(model="m1", prompt="p", negative="n")

    with flask_app.app_context():
        html = nsfw.render_template(settings, image="imgb64", error="oops")

    assert "m1" in html
    assert "p" in html
    assert "n" in html
    assert "oops" in html
