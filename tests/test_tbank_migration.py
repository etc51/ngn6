import logging
import os
from types import SimpleNamespace

import pytest

from ngn6_bot import tbank


def test_gateway_uses_tbank_target_and_sdk_cert_mode(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, token, *, target):
            captured["token"] = token
            captured["target"] = target

        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class FakeInvest:
        Client = FakeClient

    monkeypatch.setenv(tbank.T_INVEST_SSL_VERIFY_ENV, "false")
    monkeypatch.setattr(tbank, "_invest_sdk", lambda: FakeInvest)
    monkeypatch.setattr(tbank, "_sdk_default_target", lambda: tbank.T_INVEST_API_TARGET)

    with tbank.TInvestGateway("test-token", {}, logging.getLogger("test")):
        pass

    assert captured == {"token": "test-token", "target": "invest-public-api.tbank.ru"}
    assert os.environ[tbank.T_INVEST_SSL_VERIFY_ENV] == "true"


@pytest.mark.parametrize(
    "target",
    [
        "invest-public-api.tinkoff.ru",
        "https://invest-public-api.tbank.ru",
        "localhost:443",
    ],
)
def test_api_target_rejects_obsolete_or_non_tbank_hosts(monkeypatch, target):
    monkeypatch.setenv("T_INVEST_API_TARGET", target)

    with pytest.raises(RuntimeError):
        tbank._api_target({})


def test_api_target_rejects_empty_target():
    with pytest.raises(RuntimeError):
        tbank._validate_api_target("")


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (SimpleNamespace(name="TRADE_DIRECTION_BUY"), "buy"),
        (SimpleNamespace(name="TRADE_DIRECTION_SELL"), "sell"),
        (1, "buy"),
        (2, "sell"),
        (0, "unknown"),
    ],
)
def test_trade_direction_supports_sdk_enum_and_numeric_values(direction, expected):
    assert tbank._trade_direction(SimpleNamespace(direction=direction)) == expected
