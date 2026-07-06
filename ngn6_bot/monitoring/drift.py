from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DriftConfig:
    psi_warn: float = 0.15
    psi_block: float = 0.25
    min_trades: int = 20


@dataclass(frozen=True)
class DriftDecision:
    state: str
    reason: str
    metrics: dict[str, float]
    disable_ml_entries: bool = False
    raise_thresholds: bool = False
    require_repromotion: bool = False


def population_stability_index(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    buckets: int = 10,
) -> float:
    ref = np.asarray([value for value in reference if math.isfinite(float(value))], dtype=float)
    cur = np.asarray([value for value in current if math.isfinite(float(value))], dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, max(2, buckets) + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = _shares(ref_counts)
    cur_share = _shares(cur_counts)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def rolling_trade_metrics(pnls: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in pnls if math.isfinite(float(value))]
    if not values:
        return {
            "trades": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "hard_stop_share": 0.0,
        }
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    hard_stops = [value for value in values if value <= -1.0]
    return {
        "trades": float(len(values)),
        "profit_factor": float(profit_factor),
        "expectancy": float(np.mean(values)),
        "hard_stop_share": len(hard_stops) / len(values),
    }


def evaluate_drift(
    *,
    psi_values: dict[str, float],
    trade_metrics: dict[str, float] | None = None,
    config: DriftConfig | None = None,
) -> DriftDecision:
    active_config = config or DriftConfig()
    max_psi = max((float(value) for value in psi_values.values()), default=0.0)
    metrics = {"max_psi": max_psi, **{f"psi_{key}": value for key, value in psi_values.items()}}
    if trade_metrics:
        metrics.update({key: float(value) for key, value in trade_metrics.items()})

    if max_psi >= active_config.psi_block:
        return DriftDecision(
            state="block",
            reason="psi_block",
            metrics=metrics,
            disable_ml_entries=True,
            require_repromotion=True,
        )
    if max_psi >= active_config.psi_warn:
        return DriftDecision(
            state="warn",
            reason="psi_warn",
            metrics=metrics,
            raise_thresholds=True,
        )
    return DriftDecision(state="ok", reason="within_limits", metrics=metrics)


def _shares(counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(counts))
    if total <= 0:
        return np.full_like(counts, 1.0 / len(counts), dtype=float)
    return np.maximum(counts.astype(float) / total, 1e-6)
