from pathlib import Path

from ngn6_bot.config import RuntimeConfig, load_config


def test_load_default_config():
    config = load_config(Path("config/ngn6.yaml"))
    assert config.dry_run is True
    assert config.get("instrument", "ticker") == "NGN6"


def test_configured_token_env_falls_back_to_t_invest_api_token(monkeypatch, tmp_path):
    monkeypatch.delenv("T_INVEST_TOKEN", raising=False)
    monkeypatch.setenv("T_INVEST_API_TOKEN", "abc.def_12345678901234567890")

    config = RuntimeConfig({"auth": {"token_env": "T_INVEST_TOKEN"}}, tmp_path / "config.yaml")

    assert config.token == "abc.def_12345678901234567890"
