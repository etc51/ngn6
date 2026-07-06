from ngn6_bot.config import _normalize_token


def test_normalize_bearer_token():
    assert _normalize_token("Authorization: Bearer abc.def_12345678901234567890") == (
        "abc.def_12345678901234567890"
    )


def test_normalize_env_style_token():
    assert _normalize_token('T_INVEST_TOKEN="abc.def_12345678901234567890"') == (
        "abc.def_12345678901234567890"
    )
