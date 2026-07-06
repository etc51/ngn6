from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardFold:
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def purged_walk_forward_splits(
    events: pd.DataFrame,
    *,
    folds: int,
    train_window_days: int,
    validation_window_days: int,
    test_window_days: int,
    embargo_bars: int = 0,
    purge_by_event_end: bool = True,
    timestamp_col: str = "timestamp",
    event_end_col: str = "event_end",
) -> list[PurgedWalkForwardFold]:
    if events.empty:
        return []
    frame = events.reset_index(drop=True).copy()
    original_index_col = "_original_position"
    frame[original_index_col] = range(len(frame))
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    if event_end_col in frame:
        frame[event_end_col] = pd.to_datetime(frame[event_end_col], utc=True)
    else:
        frame[event_end_col] = frame[timestamp_col]
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)

    start = frame[timestamp_col].min()
    last = frame[timestamp_col].max()
    train_delta = pd.Timedelta(days=train_window_days)
    validation_delta = pd.Timedelta(days=validation_window_days)
    test_delta = pd.Timedelta(days=test_window_days)
    step = test_delta
    result: list[PurgedWalkForwardFold] = []

    cursor = start
    while len(result) < folds:
        train_start = cursor
        train_end = train_start + train_delta
        validation_start = train_end
        validation_end = validation_start + validation_delta
        test_start = validation_end
        test_end = test_start + test_delta
        if test_start > last:
            break

        train_mask = (frame[timestamp_col] >= train_start) & (frame[timestamp_col] < train_end)
        validation_mask = (frame[timestamp_col] >= validation_start) & (
            frame[timestamp_col] < validation_end
        )
        test_mask = (frame[timestamp_col] >= test_start) & (frame[timestamp_col] < test_end)

        if purge_by_event_end:
            overlaps_test = (frame[timestamp_col] < test_end) & (frame[event_end_col] >= test_start)
            train_mask &= ~overlaps_test

        if embargo_bars > 0 and test_mask.any():
            test_positions = list(frame.index[test_mask])
            start_pos = max(0, min(test_positions) - embargo_bars)
            end_pos = min(len(frame) - 1, max(test_positions) + embargo_bars)
            embargo_positions = set(range(start_pos, end_pos + 1))
            train_mask &= ~frame.index.isin(embargo_positions)

        train_indices = _original_indices(frame, original_index_col, train_mask)
        validation_indices = _original_indices(frame, original_index_col, validation_mask)
        test_indices = _original_indices(frame, original_index_col, test_mask)
        if train_indices and test_indices:
            result.append(
                PurgedWalkForwardFold(
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    test_indices=test_indices,
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

        cursor += step
        if train_start > last:
            break

    return result


def _original_indices(frame: pd.DataFrame, column: str, mask: pd.Series) -> list[int]:
    return [int(value) for value in frame.loc[mask, column].tolist()]
