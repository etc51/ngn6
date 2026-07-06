from pathlib import Path

import pytest
import yaml

from ngn6_bot.config import load_config


def _mutated_config_path(tmp_path, mutate):
    data = yaml.safe_load(Path("config/ngn6.yaml").read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "ngn6.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_runtime_config_rejects_negative_book_pressure(tmp_path):
    path = _mutated_config_path(
        tmp_path,
        lambda data: data["microstructure"].update({"min_book_pressure": -0.35}),
    )

    with pytest.raises(ValueError, match="min_book_pressure"):
        load_config(path)


def test_runtime_config_rejects_ema_adx_macd_always_trade(tmp_path):
    path = _mutated_config_path(
        tmp_path,
        lambda data: data["signals"].update({"ema_adx_macd_always_trade": True}),
    )

    with pytest.raises(ValueError, match="always_trade"):
        load_config(path)


def test_runtime_config_rejects_control_exploration(tmp_path):
    path = _mutated_config_path(
        tmp_path,
        lambda data: data["learning"].update({"mode": "control", "exploration_enabled": True}),
    )

    with pytest.raises(ValueError, match="exploration_enabled"):
        load_config(path)
