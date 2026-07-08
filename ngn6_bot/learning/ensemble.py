from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ngn6_bot.learning.feedback_model import (
    ClassBalanceConfig,
    DIRECTION_TARGETS,
    ENTRY_TARGETS,
    EXIT_CONTROL_TARGETS,
    FEATURE_SCHEMA_VERSION,
    FEATURE_KEYS,
    OPPORTUNITY_TARGETS,
    TRAINABLE_TARGETS,
    FeedbackExample,
    label_to_target,
)


@dataclass(frozen=True)
class FeedbackEnsembleReport:
    path: Path
    examples: int
    classes: list[str]
    models: list[str]
    holdout_accuracy: float | None
    trained_at: str
    task_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    promotion_score: float | None = None
    promoted: bool | None = None
    model_status: str | None = None
    promotion_status: str | None = None
    promotion_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackEnsemblePrediction:
    target: str
    score: float
    examples: int
    model_scores: dict[str, float]
    reason: str = "ml_feedback_ensemble"
    probabilities: dict[str, float] = field(default_factory=dict)


class FeedbackEnsemble:
    def __init__(
        self,
        *,
        feature_keys: list[str],
        classes: list[str],
        models: list[dict[str, Any]],
        examples: int,
        trained_at: str,
        holdout_accuracy: float | None = None,
        heads: dict[str, dict[str, Any]] | None = None,
        task_reports: dict[str, dict[str, Any]] | None = None,
        promotion_score: float | None = None,
        schema_version: int = 2,
        model_status: str = "candidate",
        promotion_status: str = "candidate",
        promotion_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        feature_schema_version: int = 1,
        training_started_at: str | None = None,
        training_finished_at: str | None = None,
        dataset_hash: str | None = None,
    ):
        self.schema_version = schema_version
        self.feature_keys = feature_keys
        self.heads = heads or {
            "entry": {
                "classes": classes,
                "models": models,
                "examples": examples,
                "holdout_accuracy": holdout_accuracy,
            }
        }
        entry_head = self.heads.get("entry") or next(iter(self.heads.values()), {})
        self.classes = list(entry_head.get("classes") or classes)
        self.models = list(entry_head.get("models") or models)
        self.examples = int(sum(int(head.get("examples", 0)) for head in self.heads.values()))
        self.trained_at = trained_at
        self.holdout_accuracy = holdout_accuracy
        self.task_reports = task_reports or {}
        self.promotion_score = promotion_score
        self.metadata = dict(metadata or {})
        self.model_status = str(
            self.metadata.get("model_status") or model_status or "candidate"
        )
        self.promotion_status = str(
            self.metadata.get("promotion_status") or promotion_status or "candidate"
        )
        self.promotion_metrics = dict(
            promotion_metrics or self.metadata.get("promotion_metrics") or {}
        )
        self.feature_schema_version = int(
            self.metadata.get("feature_schema_version", feature_schema_version)
        )
        self.training_started_at = str(
            self.metadata.get("training_started_at") or training_started_at or trained_at
        )
        self.training_finished_at = str(
            self.metadata.get("training_finished_at") or training_finished_at or trained_at
        )
        self.dataset_hash = str(self.metadata.get("dataset_hash") or dataset_hash or "")
        self.metadata.update(
            {
                "schema_version": self.schema_version,
                "model_status": self.model_status,
                "promotion_status": self.promotion_status,
                "promotion_metrics": self.promotion_metrics,
                "feature_schema_version": self.feature_schema_version,
                "training_started_at": self.training_started_at,
                "training_finished_at": self.training_finished_at,
                "dataset_hash": self.dataset_hash,
            }
        )

    @classmethod
    def load(cls, path: str | Path) -> FeedbackEnsemble:
        payload = joblib.load(path)
        if "heads" in payload:
            heads = dict(payload["heads"])
            first_head = next(iter(heads.values()), {})
            return cls(
                feature_keys=list(payload["feature_keys"]),
                classes=list(first_head.get("classes", [])),
                models=list(first_head.get("models", [])),
                examples=int(sum(int(head.get("examples", 0)) for head in heads.values())),
                trained_at=str(payload["trained_at"]),
                holdout_accuracy=payload.get("holdout_accuracy"),
                heads=heads,
                task_reports=dict(payload.get("task_reports") or {}),
                promotion_score=payload.get("promotion_score"),
                schema_version=int(payload.get("schema_version", 1)),
                model_status=str(payload.get("model_status") or "unknown"),
                promotion_status=str(payload.get("promotion_status") or ""),
                promotion_metrics=dict(payload.get("promotion_metrics") or {}),
                metadata=dict(payload.get("metadata") or {}),
                feature_schema_version=int(payload.get("feature_schema_version", 1)),
                training_started_at=payload.get("training_started_at"),
                training_finished_at=payload.get("training_finished_at"),
                dataset_hash=payload.get("dataset_hash"),
            )
        return cls(
            feature_keys=list(payload["feature_keys"]),
            classes=list(payload["classes"]),
            models=list(payload["models"]),
            examples=int(payload["examples"]),
            trained_at=str(payload["trained_at"]),
            holdout_accuracy=payload.get("holdout_accuracy"),
            schema_version=int(payload.get("schema_version", 1)),
            model_status=str(payload.get("model_status") or "unknown"),
            promotion_status=str(payload.get("promotion_status") or ""),
            promotion_metrics=dict(payload.get("promotion_metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
            feature_schema_version=int(payload.get("feature_schema_version", 1)),
            training_started_at=payload.get("training_started_at"),
            training_finished_at=payload.get("training_finished_at"),
            dataset_hash=payload.get("dataset_hash"),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": 2,
                "feature_keys": self.feature_keys,
                "classes": self.classes,
                "models": self.models,
                "examples": self.examples,
                "trained_at": self.trained_at,
                "holdout_accuracy": self.holdout_accuracy,
                "heads": self.heads,
                "task_reports": self.task_reports,
                "promotion_score": self.promotion_score,
                "model_status": self.model_status,
                "promotion_status": self.promotion_status,
                "promotion_metrics": self.promotion_metrics,
                "feature_schema_version": self.feature_schema_version,
                "training_started_at": self.training_started_at,
                "training_finished_at": self.training_finished_at,
                "dataset_hash": self.dataset_hash,
                "metadata": self.metadata,
            },
            target,
        )
        return target

    def predict(self, features: dict[str, float], *, task: str = "entry") -> FeedbackEnsemblePrediction:
        if task == "entry" and {"opportunity", "direction"}.issubset(self.heads):
            return self._predict_two_stage_entry(features)
        head = self.heads.get(task)
        return self._predict_head(features, task=task, head=head)

    def _predict_head(
        self,
        features: dict[str, float],
        *,
        task: str,
        head: dict[str, Any] | None,
    ) -> FeedbackEnsemblePrediction:
        if head is None:
            return FeedbackEnsemblePrediction("unknown", 0.0, self.examples, {}, f"no_{task}_head")
        models = list(head.get("models") or [])
        classes = list(head.get("classes") or [])
        examples = int(head.get("examples", 0))
        if not models:
            return FeedbackEnsemblePrediction("unknown", 0.0, examples, {}, "no_ml_models")

        row = _features_to_frame([features], self.feature_keys)
        totals = {target: 0.0 for target in classes}
        model_scores: dict[str, float] = {}
        active = 0
        for item in models:
            probabilities = _predict_model_probabilities(item, row, classes)
            if not probabilities:
                continue
            active += 1
            best_target, best_score = max(probabilities.items(), key=lambda pair: pair[1])
            model_scores[item["name"]] = round(float(best_score), 4)
            for target, value in probabilities.items():
                totals[target] = totals.get(target, 0.0) + float(value)

        if active == 0:
            return FeedbackEnsemblePrediction("unknown", 0.0, examples, {}, "no_ml_votes")

        averaged = {target: value / active for target, value in totals.items()}
        target, score = max(averaged.items(), key=lambda pair: pair[1])
        return FeedbackEnsemblePrediction(
            target=target,
            score=float(score),
            examples=examples,
            model_scores=model_scores,
            probabilities={key: float(value) for key, value in averaged.items()},
        )

    def _predict_two_stage_entry(self, features: dict[str, float]) -> FeedbackEnsemblePrediction:
        opportunity = self._predict_head(
            features,
            task="opportunity",
            head=self.heads.get("opportunity"),
        )
        direction = self._predict_head(
            features,
            task="direction",
            head=self.heads.get("direction"),
        )
        if opportunity.target == "unknown":
            return opportunity
        trade_score = float(opportunity.probabilities.get("trade", 0.0))
        no_trade_score = float(opportunity.probabilities.get("no_trade", 0.0))
        long_score = trade_score * float(direction.probabilities.get("long", 0.0))
        short_score = trade_score * float(direction.probabilities.get("short", 0.0))
        probabilities = {
            "flat": no_trade_score,
            "long": long_score,
            "short": short_score,
        }
        target, score = max(probabilities.items(), key=lambda pair: pair[1])
        return FeedbackEnsemblePrediction(
            target=target,
            score=float(score),
            examples=int(self.heads.get("entry", {}).get("examples", opportunity.examples)),
            model_scores={
                "opportunity": round(trade_score, 4),
                "direction": round(float(direction.score), 4),
            },
            reason="ml_feedback_two_stage",
            probabilities=probabilities,
        )


def train_feedback_ensemble(
    examples: list[FeedbackExample],
    *,
    output_path: str | Path,
    min_examples: int = 20,
    class_balance: ClassBalanceConfig | None = None,
) -> FeedbackEnsembleReport:
    training_started_at = datetime.now(timezone.utc).isoformat()
    balance_config = class_balance or ClassBalanceConfig()
    usable = _quality_filtered_examples(_normalized_examples(examples))
    if len(usable) < min_examples:
        raise ValueError(f"Need at least {min_examples} feedback examples, got {len(usable)}.")

    heads: dict[str, dict[str, Any]] = {}
    task_reports: dict[str, dict[str, Any]] = {}
    all_models: list[str] = []
    all_classes: set[str] = set()
    accuracy_values: list[float] = []
    entry_examples = [
        item for item in usable if item.task == "entry" and item.target in ENTRY_TARGETS
    ]
    balanced_entry, balance_summary = _balance_entry_examples(entry_examples, balance_config)
    if balance_summary:
        task_reports["class_balance"] = balance_summary

    head_specs: list[tuple[str, list[FeedbackExample], set[str], bool]] = [
        ("entry", balanced_entry, ENTRY_TARGETS, True),
    ]
    if balance_config.enabled and balance_config.train_two_stage:
        head_specs.extend(
            [
                ("opportunity", _opportunity_examples(balanced_entry), OPPORTUNITY_TARGETS, False),
                (
                    "direction",
                    [item for item in balanced_entry if item.target in DIRECTION_TARGETS],
                    DIRECTION_TARGETS,
                    True,
                ),
            ]
        )
    head_specs.append(
        (
            "exit",
            [
                item
                for item in usable
                if item.task == "exit" and item.target in EXIT_CONTROL_TARGETS
            ],
            EXIT_CONTROL_TARGETS,
            True,
        )
    )

    for task, task_usable, allowed_targets, include_money in head_specs:
        head, report = _train_head(
            task,
            task_usable,
            allowed_targets,
            min_examples=min_examples,
            include_money=include_money,
            use_class_weights=balance_config.use_class_weights,
        )
        if head is None or report is None:
            continue
        heads[task] = head
        model_names = [item["name"] for item in head["models"]]
        all_models.extend(f"{task}:{name}" for name in model_names)
        all_classes.update(head["classes"])
        if head["holdout_accuracy"] is not None:
            accuracy_values.append(head["holdout_accuracy"])
        task_reports[task] = report

    if not heads:
        raise ValueError("No feedback ensemble heads could be trained.")

    holdout_accuracy = float(np.mean(accuracy_values)) if accuracy_values else None
    promotion_score = _promotion_score(task_reports)
    trained_at = datetime.now(timezone.utc).isoformat()
    dataset_hash = _dataset_hash(usable)
    metadata = {
        "schema_version": 2,
        "model_status": "candidate",
        "promotion_status": "candidate",
        "promotion_metrics": {},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_started_at": training_started_at,
        "training_finished_at": trained_at,
        "dataset_hash": dataset_hash,
        "class_balance": balance_summary,
    }
    ensemble = FeedbackEnsemble(
        feature_keys=list(FEATURE_KEYS),
        classes=sorted(all_classes),
        models=[],
        examples=sum(int(head["examples"]) for head in heads.values()),
        trained_at=trained_at,
        holdout_accuracy=holdout_accuracy,
        heads=heads,
        task_reports=task_reports,
        promotion_score=promotion_score,
        model_status="candidate",
        promotion_status="candidate",
        promotion_metrics={},
        metadata=metadata,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_started_at=training_started_at,
        training_finished_at=trained_at,
        dataset_hash=dataset_hash,
    )
    path = ensemble.save(output_path)
    return FeedbackEnsembleReport(
        path=path,
        examples=sum(int(head["examples"]) for head in heads.values()),
        classes=sorted(all_classes),
        models=all_models,
        holdout_accuracy=holdout_accuracy,
        trained_at=trained_at,
        task_reports=task_reports,
        promotion_score=promotion_score,
        model_status="candidate",
        promotion_status="candidate",
        promotion_metrics={},
    )


def _normalized_examples(examples: list[FeedbackExample]) -> list[FeedbackExample]:
    normalized: list[FeedbackExample] = []
    for item in examples:
        target = item.target if item.target in TRAINABLE_TARGETS else label_to_target(item.label)
        if target not in TRAINABLE_TARGETS:
            target = label_to_target(item.target.upper())
        if target not in TRAINABLE_TARGETS:
            continue
        task = item.task
        if target in EXIT_CONTROL_TARGETS:
            task = "exit"
        elif target in ENTRY_TARGETS:
            task = "entry"
        normalized.append(
            FeedbackExample(
                label=item.label,
                target=target,
                features=item.features,
                source=item.source,
                timestamp=item.timestamp,
                task=task,
                pnl_pct=item.pnl_pct,
                outcomes=item.outcomes,
                feature_complete=item.feature_complete,
                label_matured=item.label_matured,
                market_data_trusted=item.market_data_trusted,
                reject_reason=item.reject_reason,
            )
        )
    return normalized


def _quality_filtered_examples(examples: list[FeedbackExample]) -> list[FeedbackExample]:
    return [
        item
        for item in examples
        if item.feature_complete and item.label_matured and item.market_data_trusted
    ]


def _train_head(
    task: str,
    examples: list[FeedbackExample],
    allowed_targets: set[str],
    *,
    min_examples: int,
    include_money: bool,
    use_class_weights: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    task_usable = [item for item in examples if item.target in allowed_targets]
    if len(task_usable) < min_examples:
        return None, None
    classes = sorted({item.target for item in task_usable})
    if len(classes) < 2:
        return None, None
    x = _features_to_frame([item.features for item in task_usable], FEATURE_KEYS)
    y = pd.Series([item.target for item in task_usable])
    models = _train_models(x, y, use_class_weights=use_class_weights)
    if not models:
        return None, None
    holdout_accuracy = _holdout_accuracy(
        x,
        y,
        models,
        classes,
        use_class_weights=use_class_weights,
    )
    money = (
        _holdout_money_metrics(
            x,
            task_usable,
            classes,
            use_class_weights=use_class_weights,
        )
        if include_money
        else {
            "holdout_expected_value_pct": None,
            "holdout_profit_factor": None,
            "holdout_trades": 0,
        }
    )
    class_counts = {str(key): int(value) for key, value in y.value_counts().items()}
    head = {
        "classes": classes,
        "class_counts": class_counts,
        "models": models,
        "examples": len(task_usable),
        "holdout_accuracy": holdout_accuracy,
        "holdout_money": money,
    }
    report = {
        "examples": len(task_usable),
        "classes": classes,
        "class_counts": class_counts,
        "models": [item["name"] for item in models],
        "holdout_accuracy": holdout_accuracy,
        **money,
    }
    return head, report


def _balance_entry_examples(
    examples: list[FeedbackExample],
    config: ClassBalanceConfig,
) -> tuple[list[FeedbackExample], dict[str, Any]]:
    original_counts = _target_counts(examples)
    summary = {
        "enabled": bool(config.enabled),
        "train_two_stage": bool(config.train_two_stage),
        "original_class_counts": original_counts,
        "balanced_class_counts": original_counts,
        "flat_downsample_ratio": config.flat_downsample_ratio,
        "max_flat_share_after_balance": config.max_flat_share_after_balance,
        "min_directional_examples": config.min_directional_examples,
    }
    if not config.enabled:
        return examples, summary
    directional = [item for item in examples if item.target in DIRECTION_TARGETS]
    flat = [item for item in examples if item.target == "flat"]
    if len(directional) < config.min_directional_examples:
        summary["status"] = "directional_examples_below_min"
        return examples, summary
    ratio_cap = int(math.ceil(len(directional) * max(config.flat_downsample_ratio, 0.0)))
    if 0.0 < config.max_flat_share_after_balance < 1.0:
        share_cap = int(
            math.floor(
                config.max_flat_share_after_balance
                * len(directional)
                / (1.0 - config.max_flat_share_after_balance)
            )
        )
        flat_limit = min(len(flat), ratio_cap, share_cap)
    else:
        flat_limit = min(len(flat), ratio_cap)
    selected_flat = _sort_examples(flat)[-max(0, flat_limit):] if flat_limit > 0 else []
    balanced = _sort_examples(directional + selected_flat)
    summary["status"] = "balanced"
    summary["balanced_class_counts"] = _target_counts(balanced)
    summary["dropped_flat_examples"] = max(0, len(flat) - len(selected_flat))
    return balanced, summary


def _opportunity_examples(examples: list[FeedbackExample]) -> list[FeedbackExample]:
    result: list[FeedbackExample] = []
    for item in examples:
        if item.target not in ENTRY_TARGETS:
            continue
        target = "trade" if item.target in DIRECTION_TARGETS else "no_trade"
        result.append(
            FeedbackExample(
                label=item.label,
                target=target,
                features=item.features,
                source=item.source,
                timestamp=item.timestamp,
                task="opportunity",
                pnl_pct=item.pnl_pct,
                outcomes=item.outcomes,
                feature_complete=item.feature_complete,
                label_matured=item.label_matured,
                market_data_trusted=item.market_data_trusted,
                reject_reason=item.reject_reason,
            )
        )
    return result


def _target_counts(examples: list[FeedbackExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in examples:
        counts[item.target] = counts.get(item.target, 0) + 1
    return counts


def _sort_examples(examples: list[FeedbackExample]) -> list[FeedbackExample]:
    return sorted(
        examples,
        key=lambda item: (item.timestamp or "", item.source, item.label, item.target),
    )


def _dataset_hash(examples: list[FeedbackExample]) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        examples,
        key=lambda example: (
            example.task,
            example.target,
            example.timestamp or "",
            example.source,
        ),
    ):
        payload = {
            "task": item.task,
            "target": item.target,
            "timestamp": item.timestamp,
            "features": {
                key: round(float(value), 8)
                for key, value in sorted(item.features.items())
                if key in FEATURE_KEYS
            },
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _features_to_frame(items: list[dict[str, float]], feature_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{key: float(item.get(key, 0.0)) for key in feature_keys} for item in items],
        columns=feature_keys,
    ).fillna(0.0)


def _train_models(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    include_heavy: bool | None = None,
    use_class_weights: bool = True,
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    counts = y.value_counts()
    min_class_count = int(counts.min())
    heavy_allowed = len(y) <= 1200 if include_heavy is None else include_heavy

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if use_class_weights else None,
            random_state=17,
        ),
    )
    if min_class_count >= 3 and len(y) >= 18:
        cv = min(3, min_class_count)
        logistic_model = CalibratedClassifierCV(logistic, cv=cv, method="sigmoid")
        model_name = "calibrated_logistic"
    else:
        logistic_model = logistic
        model_name = "logistic"
    logistic_model.fit(x, y)
    models.append({"name": model_name, "kind": "sklearn", "model": logistic_model, "classes": list(logistic_model.classes_)})

    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=3,
            num_leaves=12,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced" if use_class_weights else None,
            random_state=17,
            verbosity=-1,
        )
        model.fit(x, y)
        models.append({"name": "lightgbm", "kind": "sklearn", "model": model, "classes": list(model.classes_)})
    except Exception:
        pass

    if heavy_allowed:
        try:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=80,
                learning_rate=0.05,
                depth=4,
                loss_function="MultiClass",
                random_seed=17,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(x, y)
            models.append({"name": "catboost", "kind": "sklearn", "model": model, "classes": list(model.classes_)})
        except Exception:
            pass

        try:
            from xgboost import XGBClassifier

            encoder = LabelEncoder()
            encoded_y = encoder.fit_transform(y)
            objective = "multi:softprob" if len(encoder.classes_) > 2 else "binary:logistic"
            model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective=objective,
                eval_metric="mlogloss" if len(encoder.classes_) > 2 else "logloss",
                random_state=17,
                n_jobs=1,
            )
            model.fit(x, encoded_y)
            models.append(
                {
                    "name": "xgboost",
                    "kind": "xgboost_encoded",
                    "model": model,
                    "encoder": encoder,
                    "classes": list(encoder.classes_),
                }
            )
        except Exception:
            pass

    return models


def _predict_model_probabilities(
    item: dict[str, Any],
    row: pd.DataFrame,
    expected_classes: list[str],
) -> dict[str, float]:
    model = item["model"]
    probabilities = model.predict_proba(row)[0]
    if item["kind"] == "xgboost_encoded":
        classes = list(item["encoder"].classes_)
    else:
        classes = list(item.get("classes") or getattr(model, "classes_", []))

    values = {target: 0.0 for target in expected_classes}
    for target, value in zip(classes, probabilities, strict=False):
        if str(target) in values:
            values[str(target)] = float(value)
    return values


def _holdout_accuracy(
    x: pd.DataFrame,
    y: pd.Series,
    models: list[dict[str, Any]],
    classes: list[str],
    *,
    use_class_weights: bool = True,
) -> float | None:
    if len(y) < 24 or y.nunique() < 2:
        return None
    split = TimeSeriesSplit(n_splits=3)
    scores: list[float] = []
    for train_idx, test_idx in split.split(x):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].empty:
            continue
        local_models = _train_models(
            x.iloc[train_idx],
            y.iloc[train_idx],
            include_heavy=False,
            use_class_weights=use_class_weights,
        )
        if not local_models:
            continue
        correct = 0
        for row_idx in test_idx:
            prediction = _predict_with_models(
                local_models,
                x.iloc[[row_idx]],
                classes,
            )
            if prediction == y.iloc[row_idx]:
                correct += 1
        scores.append(correct / len(test_idx))
    if not scores:
        return None
    return float(np.mean(scores))


def _holdout_money_metrics(
    x: pd.DataFrame,
    examples: list[FeedbackExample],
    classes: list[str],
    *,
    use_class_weights: bool = True,
) -> dict[str, float | int | None]:
    if len(examples) < 24 or len(classes) < 2:
        return {
            "holdout_expected_value_pct": None,
            "holdout_profit_factor": None,
            "holdout_trades": 0,
        }
    y = pd.Series([item.target for item in examples])
    split = TimeSeriesSplit(n_splits=3)
    values: list[float] = []
    trade_values: list[float] = []
    for train_idx, test_idx in split.split(x):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].empty:
            continue
        local_models = _train_models(
            x.iloc[train_idx],
            y.iloc[train_idx],
            include_heavy=False,
            use_class_weights=use_class_weights,
        )
        if not local_models:
            continue
        for row_idx in test_idx:
            prediction = _predict_with_models(local_models, x.iloc[[row_idx]], classes)
            example = examples[int(row_idx)]
            value = _outcome_value(example, prediction)
            values.append(value)
            if prediction in {"long", "short", "hold"}:
                trade_values.append(value)

    if not values:
        return {
            "holdout_expected_value_pct": None,
            "holdout_profit_factor": None,
            "holdout_trades": 0,
        }
    wins = [value for value in trade_values if value > 0]
    losses = [value for value in trade_values if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else None
    return {
        "holdout_expected_value_pct": float(np.mean(values)),
        "holdout_profit_factor": profit_factor,
        "holdout_trades": len(trade_values),
    }


def _outcome_value(example: FeedbackExample, prediction: str) -> float:
    if example.outcomes and prediction in example.outcomes:
        return float(example.outcomes[prediction])
    if prediction == example.target:
        return float(example.pnl_pct)
    if prediction in {"flat", "exit"}:
        return 0.0
    return -abs(float(example.pnl_pct))


def _promotion_score(task_reports: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(report["holdout_expected_value_pct"])
        for report in task_reports.values()
        if report.get("holdout_expected_value_pct") is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def _predict_with_models(
    models: list[dict[str, Any]],
    row: pd.DataFrame,
    classes: list[str],
) -> str:
    totals = {target: 0.0 for target in classes}
    active = 0
    for item in models:
        probabilities = _predict_model_probabilities(item, row, classes)
        if not probabilities:
            continue
        active += 1
        for target, value in probabilities.items():
            totals[target] = totals.get(target, 0.0) + value
    if active == 0:
        return "unknown"
    return max(totals.items(), key=lambda pair: pair[1])[0]
