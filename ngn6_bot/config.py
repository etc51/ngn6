from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TOKEN_ENV_NAMES = ("T_INVEST_TOKEN", "T_INVEST_API_TOKEN", "INVEST_TOKEN")


def _token_env_names(configured_env: str | None) -> tuple[str, ...]:
    if not configured_env:
        return DEFAULT_TOKEN_ENV_NAMES
    return (configured_env, *(name for name in DEFAULT_TOKEN_ENV_NAMES if name != configured_env))


@dataclass(frozen=True)
class RuntimeConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def dry_run(self) -> bool:
        return bool(self.raw["bot"].get("dry_run", True)) and bool(
            self.raw.get("trading", {}).get("dry_run", True)
        )

    @property
    def live_enabled(self) -> bool:
        return any(
            bool(section.get("live_enabled", False) or section.get("real_trading", False))
            for section in (
                self.raw.get("bot", {}),
                self.raw.get("trading", {}),
                self.raw.get("execution", {}),
            )
        )

    @property
    def timezone(self) -> str:
        return str(self.raw["bot"].get("timezone", "Europe/Moscow"))

    @property
    def token(self) -> str:
        configured_env = self.raw.get("auth", {}).get("token_env")
        env_names = _token_env_names(configured_env)
        token = next((os.getenv(env_name) for env_name in env_names if os.getenv(env_name)), None)
        if token:
            return token.strip()

        token_file = self.raw.get("auth", {}).get("token_file")
        if token_file:
            path = Path(os.path.expandvars(os.path.expanduser(token_file)))
            if path.exists():
                return _normalize_token(path.read_text(encoding="utf-8-sig"))

        raise RuntimeError(
            "T-Invest token is missing. Set one of "
            f"{', '.join(env_names)} or auth.token_file in {self.path}."
        )

    @property
    def account_id(self) -> str | None:
        account = self.raw.get("account", {})
        env_name = account.get("account_id_env", "T_INVEST_ACCOUNT_ID")
        return os.getenv(env_name) or account.get("account_id") or None

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.raw
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must contain a YAML object.")
    _validate_config(data)
    return RuntimeConfig(raw=data, path=config_path)


def _normalize_token(raw: str) -> str:
    text = raw.strip().strip("\"'")
    if not text:
        raise RuntimeError("Token file is empty.")

    bearer_match = re.search(r"Bearer\s+([A-Za-z0-9._=-]{20,})", text, flags=re.IGNORECASE)
    if bearer_match:
        return bearer_match.group(1)

    for line in text.splitlines():
        line = line.strip().strip("\"'")
        if not line:
            continue
        if "=" in line:
            line = line.split("=", 1)[1].strip().strip("\"'")
        token_match = re.search(r"([A-Za-z0-9._=-]{20,})", line)
        if token_match:
            return token_match.group(1)

    if re.fullmatch(r"[A-Za-z0-9._=-]{20,}", text):
        return text
    raise RuntimeError("Token file does not contain a valid-looking T-Invest token.")


def _validate_config(data: dict[str, Any]) -> None:
    required_sections = [
        "bot",
        "auth",
        "instrument",
        "account",
        "indicators",
        "orderbook",
        "signals",
        "risk",
        "execution",
        "session",
    ]
    missing = [section for section in required_sections if section not in data]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    risk_pct = float(data["risk"]["risk_per_trade_pct"])
    max_risk_pct = float(data["risk"]["max_risk_per_trade_pct"])
    if risk_pct <= 0 or max_risk_pct <= 0 or risk_pct > max_risk_pct:
        raise ValueError("risk_per_trade_pct must be > 0 and <= max_risk_per_trade_pct.")

    if not bool(data["bot"].get("dry_run", True)) or not bool(
        data.get("trading", {}).get("dry_run", True)
    ):
        raise ValueError("Live execution is prohibited: dry_run must stay true.")

    for section_name in ("bot", "trading", "execution"):
        section = data.get(section_name, {})
        if bool(section.get("live_enabled", False)) or bool(section.get("real_trading", False)):
            raise ValueError("Live execution is prohibited: live_enabled/real_trading must be false.")

    min_book_pressure = float(data.get("microstructure", {}).get("min_book_pressure", 0.0))
    if min_book_pressure < 0:
        raise ValueError("microstructure.min_book_pressure must be >= 0.")

    signals = data.get("signals", {})
    if (
        str(signals.get("engine", "")).lower() == "ema_adx_macd"
        and bool(signals.get("ema_adx_macd_always_trade", False))
    ):
        raise ValueError("signals.ema_adx_macd_always_trade must be false for runtime config.")

    learning = data.get("learning", {})
    if (
        str(learning.get("mode", "")).lower() in {"control", "shadow_then_control"}
        and bool(learning.get("exploration_enabled", False))
    ):
        raise ValueError("learning.exploration_enabled must be false in control mode.")

    execution = data.get("execution", {})
    if bool(execution.get("allow_fallback_entries", False)) or bool(
        signals.get("allow_fallback_entries", False)
    ):
        raise ValueError("fallback entries must stay disabled.")
    if bool(execution.get("allow_trade_without_promoted_model", False)):
        raise ValueError("entries without promoted ML model must stay disabled.")
