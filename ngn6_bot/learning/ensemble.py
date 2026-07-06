from __future__ import annotations

import hashlib
import json
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
    ENTRY_TARGETS,
    EXIT_CONTROL_TARGETS,
    FEATURE_KEYS,
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
        head = self.heads.get(task)
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


def train_feedback_ensemble(
    examples: list[FeedbackExample],
    *,
    output_path: str | Path,
    min_examples: int = 20,
) -> FeedbackEnsembleReport:
    training_started_at = datetime.now(timezone.utc).isoformat()
    usable = _normalized_examples(examples)
    if len(usable) < min_examples:
        raise ValueError(f"Need at least {min_examples} feedback examples, got {len(usable)}.")

    heads: dict[str, dict[str, Any]] = {}
    task_reports: dict[str, dict[str, Any]] = {}
    all_models: list[str] = []
    all_classes: set[str] = set()
    accuracy_values: list[float] = []
    for task, allowed_targets in [
        ("entry", ENTRY_TARGETS),
        ("exit", EXIT_CONTROL_TARGETS),
    ]:
        task_usable = [
            item for item in usable if item.task == task and item.target in allowed_targets
        ]
        if len(task_usable) < min_examples:
            continue
        classes = sorted({item.target for item in task_usable})
        if len(classes) < 2:
            continue
        x = _features_to_frame([item.features for item in task_usable], FEATURE_KEYS)
        y = pd.Series([item.target for item in task_usable])
        models = _train_models(x, y)
        if not models:
            continue
        holdout_accuracy = _holdout_accuracy(x, y, models, classes)
        money = _holdout_money_metrics(x, task_usable, classes)
        heads[task] = {
            "classes": classes,
            "class_counts": {str(key): int(value) for key, value in y.value_counts().items()},
            "models": models,
            "examples": len(task_usable),
            "holdout_accuracy": holdout_accuracy,
            "holdout_money": money,
        }
        model_names = [item["name"] for item in models]
        all_models.extend(f"{task}:{name}" for name in model_names)
        all_classes.update(classes)
        if holdout_accuracy is not None:
            accuracy_values.append(holdout_accuracy)
        task_reports[task] = {
            "examples": len(task_usable),
            "classes": classes,
            "class_counts": {str(key): int(value) for key, value in y.value_counts().items()},
            "models": model_names,
            "holdout_accuracy": holdout_accuracy,
            **money,
        }

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
        "feature_schema_version": 1,
        "training_started_at": training_started_at,
        "training_finished_at": trained_at,
        "dataset_hash": dataset_hash,
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
        feature_schema_version=1,
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
            )
        )
    return normalized


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
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    counts = y.value_counts()
    min_class_count = int(counts.min())
    heavy_allowed = len(y) <= 1200 if include_heavy is None else include_heavy

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=17),
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
            class_weight="balanced",
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
) -> float | None:
    if len(y) < 24 or y.nunique() < 2:
        return None
    split = TimeSeriesSplit(n_splits=3)
    scores: list[float] = []
    for train_idx, test_idx in split.split(x):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].empty:
            continue
        local_models = _train_models(x.iloc[train_idx], y.iloc[train_idx], include_heavy=False)
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
        local_models = _train_models(x.iloc[train_idx], y.iloc[train_idx], include_heavy=False)
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
