"""Tests for config env var overrides."""

import importlib
import os
import sys


def _reload_config(env: dict[str, str]):
    """Reload ares.config with a given environment."""
    patched = {**os.environ, **env}
    # Remove overrides that aren't in env so we test defaults too
    original = os.environ.copy()
    os.environ.update(env)
    # Remove any existing keys that should be absent
    for k in ["ARES_OLLAMA_HOST", "ARES_ORCHESTRATOR_MODEL",
              "ARES_CODER_MODEL", "ARES_THINKER_MODEL", "ARES_RUNNER_MODEL"]:
        if k not in env:
            os.environ.pop(k, None)
    if "ares.config" in sys.modules:
        del sys.modules["ares.config"]
    import ares.config as cfg
    # Restore
    os.environ.clear()
    os.environ.update(original)
    if "ares.config" in sys.modules:
        del sys.modules["ares.config"]
    return cfg


def test_defaults():
    cfg = _reload_config({})
    assert cfg.OLLAMA_HOST == "http://localhost:11434"
    assert cfg.FLUX == "qwen3.5:4b"
    assert cfg.FORGE == "qwen2.5-coder:7b"
    assert cfg.RUNE == "deepseek-r1:7b"
    assert cfg.SWIFT == "qwen2.5-coder:3b"


def test_env_overrides():
    cfg = _reload_config({
        "ARES_OLLAMA_HOST": "http://gpu-box:11434",
        "ARES_ORCHESTRATOR_MODEL": "llama3:8b",
        "ARES_CODER_MODEL": "codestral:latest",
    })
    assert cfg.OLLAMA_HOST == "http://gpu-box:11434"
    assert cfg.FLUX == "llama3:8b"
    assert cfg.FORGE == "codestral:latest"
    # Unset keys fall back to defaults
    assert cfg.RUNE == "deepseek-r1:7b"


def test_max_tokens_orchestrator_not_512():
    cfg = _reload_config({})
    assert cfg.AGENT_DEFAULTS["orchestrator"]["max_tokens"] >= 2048, (
        "Orchestrator max_tokens must be >= 2048 to relay sub-agent responses"
    )


def test_required_models_listed():
    cfg = _reload_config({})
    assert cfg.FLUX in cfg.REQUIRED_MODELS
    assert cfg.FORGE in cfg.REQUIRED_MODELS
    assert cfg.RUNE in cfg.REQUIRED_MODELS
    assert cfg.SWIFT in cfg.REQUIRED_MODELS


def test_env_files_fill_gaps_without_overriding(tmp_path, monkeypatch):
    from ares.cli import _load_env

    first = tmp_path / "first.env"
    first.write_text("# comment\nARES_T_A=from_first\nARES_T_B='quoted'\n")
    second = tmp_path / "second.env"
    second.write_text("ARES_T_A=from_second\nARES_T_C=from_second\n")
    monkeypatch.setenv("ARES_T_C", "from_shell")
    for key in ("ARES_T_A", "ARES_T_B"):
        monkeypatch.delenv(key, raising=False)

    _load_env([first, tmp_path / "missing.env", second])

    import os
    assert os.environ["ARES_T_A"] == "from_first"
    assert os.environ["ARES_T_B"] == "quoted"
    assert os.environ["ARES_T_C"] == "from_shell"
    for key in ("ARES_T_A", "ARES_T_B"):
        monkeypatch.delenv(key)
