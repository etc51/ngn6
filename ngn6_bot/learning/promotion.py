from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.learning.feedback_model import FeedbackModel
from ngn6_bot.runtime_metadata import with_commit_hash


@dataclass(frozen=True)
class PromotionCheckReport:
    model_path: str
    ready: bool
    reason: str
    details: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(with_commit_hash(asdict(self)), ensure_ascii=False, indent=2)


def check_model_eligibility(
    config: RuntimeConfig,
    *,
    model_path: str | Path | None = None,
) -> PromotionCheckReport:
    active_path = Path(model_path) if model_path is not None else Path(
        config.get("learning", "ensemble_model_path")
    )
    raw = copy.deepcopy(config.raw)
    raw.setdefault("learning", {})["ensemble_model_path"] = str(active_path)
    raw["learning"]["enabled"] = True
    raw["learning"]["mode"] = "shadow_then_control"
    raw["learning"]["ensemble_enabled"] = True
    raw["learning"]["control_require_ensemble_model"] = True
    raw["learning"]["control_require_schema_v2"] = True
    raw["learning"]["control_require_promoted_model"] = True
    raw["learning"]["active_can_trade_only_if_promoted"] = True
    model = FeedbackModel.from_runtime_config(RuntimeConfig(raw=raw, path=config.path))
    ready, reason, details = model.control_model_validation()
    return PromotionCheckReport(
        model_path=str(active_path),
        ready=ready,
        reason=reason,
        details=details,
    )


def save_promotion_check(report: PromotionCheckReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.to_json(), encoding="utf-8")
    return target
