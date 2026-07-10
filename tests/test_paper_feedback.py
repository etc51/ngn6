import json
from datetime import datetime, timedelta, timezone

from ngn6_bot.config import RuntimeConfig, load_config
from ngn6_bot.learning.paper_feedback import sync_paper_trade_feedback


def test_closed_paper_loss_becomes_matured_flat_label(tmp_path):
    base = load_config("config/ngn6.yaml")
    raw = json.loads(json.dumps(base.raw))
    events = tmp_path / "paper_events.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    labels = tmp_path / "feedback_labels.jsonl"
    raw["paper"]["events_file"] = str(events)
    raw["data_collection"]["decisions_file"] = str(decisions)
    raw["learning"]["labels_file"] = str(labels)
    raw["learning"]["paper_trade_feedback_started_at"] = "2026-01-01T00:00:00+00:00"
    config = RuntimeConfig(raw=raw, path=base.path)
    opened_at = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    closed_at = opened_at + timedelta(minutes=5)
    features = {"return_1": 0.1, "adx": 0.4, "position_side": 0.0}
    decisions.write_text("", encoding="utf-8")
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": opened_at.isoformat(),
                        "event": "paper_open",
                        "details": {
                            "side": "long",
                            "lots": 1,
                            "price": 3.0,
                            "reason": "ml_entry:long",
                            "commission": 1.0,
                            "feedback_context": {
                                "features": features,
                                "feature_complete": True,
                                "market_data_trusted": True,
                                "candidate_execution": True,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": closed_at.isoformat(),
                        "event": "paper_close",
                        "details": {
                            "realized_pnl": -99.0,
                            "remaining_lots": 0,
                            "reason": "hard_stop_hit",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = sync_paper_trade_feedback(config)
    second = sync_paper_trade_feedback(config)
    label = json.loads(labels.read_text(encoding="utf-8").splitlines()[0])

    assert first.labels_added == 1
    assert second.labels_added == 0
    assert label["target"] == "flat"
    assert label["label_matured"] is True
    assert label["market_data_trusted"] is True
    assert label["outcomes"]["long"] < 0
    assert label["features"] == features
