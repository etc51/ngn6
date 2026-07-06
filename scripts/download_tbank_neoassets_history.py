from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import t_tech.invest as invest


TOKEN_ENV_NAMES = ("T_INVEST_TOKEN", "T_INVEST_API_TOKEN", "INVEST_TOKEN")
DEFAULT_ASSETS = {
    "bitcoin": {"query": "Neo Bitcoin", "ticker": "BTCUSDperpA"},
    "ethereum": {"query": "Neo Ethereum", "ticker": "ETHUSDperpA"},
}


@dataclass(frozen=True)
class NeoAsset:
    key: str
    name: str
    ticker: str
    figi: str
    uid: str
    position_uid: str
    class_code: str
    instrument_type: str


def _intervals() -> dict[str, Any]:
    return {
        "1min": invest.CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5min": invest.CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15min": invest.CandleInterval.CANDLE_INTERVAL_15_MIN,
        "hour": invest.CandleInterval.CANDLE_INTERVAL_HOUR,
        "day": invest.CandleInterval.CANDLE_INTERVAL_DAY,
    }


def _chunk_sizes() -> dict[str, timedelta]:
    return {
        "1min": timedelta(days=1),
        "5min": timedelta(days=7),
        "15min": timedelta(days=14),
        "hour": timedelta(days=60),
        "day": timedelta(days=365),
    }


def _token() -> str:
    for name in TOKEN_ENV_NAMES:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    raise RuntimeError(f"T-Invest token is missing. Set one of: {', '.join(TOKEN_ENV_NAMES)}")


def _parse_utc(value: str | None, *, default: datetime | None = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("datetime value is required")
        return default
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        parsed = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _quotation_to_float(value: Any) -> float:
    return float(value.units) + float(value.nano) / 1_000_000_000


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def _candle_row(candle: Any) -> dict[str, Any]:
    timestamp = _iso_utc(candle.time)
    return {
        "timestamp": timestamp,
        "open": _quotation_to_float(candle.open),
        "high": _quotation_to_float(candle.high),
        "low": _quotation_to_float(candle.low),
        "close": _quotation_to_float(candle.close),
        "volume": int(candle.volume),
        "is_complete": bool(getattr(candle, "is_complete", True)),
    }


def discover_neo_assets(client: Any, requested: list[str]) -> list[NeoAsset]:
    assets: list[NeoAsset] = []
    for key in requested:
        spec = DEFAULT_ASSETS[key]
        response = client.instruments.find_instrument(query=spec["query"])
        candidates = [
            item
            for item in response.instruments
            if str(getattr(item, "name", "")).casefold() == spec["query"].casefold()
            and str(getattr(item, "ticker", "")).casefold() == spec["ticker"].casefold()
        ]
        if not candidates:
            candidates = [
                item
                for item in response.instruments
                if spec["query"].casefold() in str(getattr(item, "name", "")).casefold()
            ]
        if not candidates:
            raise RuntimeError(f"{spec['query']} was not found in T-Bank instruments.")
        item = candidates[0]
        assets.append(
            NeoAsset(
                key=key,
                name=str(getattr(item, "name", "")),
                ticker=str(getattr(item, "ticker", "")),
                figi=str(getattr(item, "figi", "")),
                uid=str(getattr(item, "uid", "")),
                position_uid=str(getattr(item, "position_uid", "")),
                class_code=str(getattr(item, "class_code", "")),
                instrument_type=str(getattr(item, "instrument_type", "")),
            )
        )
    return assets


def _chunk_ranges(start: datetime, end: datetime, chunk: timedelta) -> list[tuple[int, datetime, datetime]]:
    ranges: list[tuple[int, datetime, datetime]] = []
    cursor = start
    index = 0
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        ranges.append((index, cursor, chunk_end))
        index += 1
        cursor = chunk_end
    return ranges


def _fetch_chunk(
    client: Any,
    asset: NeoAsset,
    interval_name: str,
    cursor: datetime,
    chunk_end: datetime,
    *,
    retries: int,
    pause_seconds: float,
    include_incomplete: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    interval = _intervals()[interval_name]
    for attempt in range(1, retries + 1):
        try:
            response = client.market_data.get_candles(
                instrument_id=asset.uid,
                from_=cursor,
                to=chunk_end,
                interval=interval,
            )
            rows = [
                _candle_row(candle)
                for candle in response.candles
                if include_incomplete or bool(getattr(candle, "is_complete", True))
            ]
            rows.sort(key=lambda row: row["timestamp"])
            return rows, None
        except Exception as error:
            message = str(error)
            if attempt >= retries:
                return [], message[:500]
            sleep_for = max(pause_seconds, 2.0 if "EXHAUSTED" in message else 0.5)
            time.sleep(sleep_for)
    return [], "unknown fetch error"


def download_candles_to_csv(
    client: Any,
    asset: NeoAsset,
    interval_name: str,
    start: datetime,
    end: datetime,
    path: Path,
    *,
    retries: int,
    pause_seconds: float,
    include_incomplete: bool,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    chunks = _chunk_ranges(start, end, _chunk_sizes()[interval_name])
    failed_chunks: list[dict[str, str]] = []
    rows_count = 0
    non_empty_chunks = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    last_written_timestamp: str | None = None
    completed_chunks = 0

    batch_size = max(1, batch_size)
    workers = max(1, workers)
    with temp_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume", "is_complete"],
        )
        writer.writeheader()

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            results: dict[int, list[dict[str, Any]]] = {}
            errors: dict[int, str] = {}

            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                futures = {
                    executor.submit(
                        _fetch_chunk,
                        client,
                        asset,
                        interval_name,
                        cursor,
                        chunk_end,
                        retries=retries,
                        pause_seconds=pause_seconds,
                        include_incomplete=include_incomplete,
                    ): (index, cursor, chunk_end)
                    for index, cursor, chunk_end in batch
                }
                for future in as_completed(futures):
                    index, cursor, chunk_end = futures[future]
                    rows, error = future.result()
                    results[index] = rows
                    if error:
                        errors[index] = error
                        failed_chunks.append(
                            {
                                "from": _iso_utc(cursor),
                                "to": _iso_utc(chunk_end),
                                "error": error,
                            }
                        )

            for index, _, _ in batch:
                rows = results.get(index, [])
                if rows:
                    non_empty_chunks += 1
                for row in rows:
                    timestamp = row["timestamp"]
                    if timestamp == last_written_timestamp:
                        continue
                    writer.writerow(row)
                    rows_count += 1
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                    last_written_timestamp = timestamp

            completed_chunks += len(batch)
            output.flush()
            print(
                f"  {asset.ticker} {interval_name}: "
                f"{completed_chunks}/{len(chunks)} chunks, {rows_count} rows",
                flush=True,
            )
            if pause_seconds > 0:
                time.sleep(pause_seconds)

    temp_path.replace(path)
    stats = {
        "requests": len(chunks),
        "non_empty_chunks": non_empty_chunks,
        "failed_chunks": failed_chunks,
        "rows": rows_count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download T-Bank Neo Bitcoin/Ethereum candles.")
    parser.add_argument("--output-dir", default="data/tbank_neoassets")
    parser.add_argument("--from", dest="from_", default="2010-01-01")
    parser.add_argument("--to", default=None)
    parser.add_argument("--assets", default="bitcoin,ethereum")
    parser.add_argument("--intervals", default="1min,5min,15min,hour,day")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--pause-seconds", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-incomplete", action="store_true")
    args = parser.parse_args(argv)

    requested_assets = [item.strip().lower() for item in args.assets.split(",") if item.strip()]
    unknown_assets = [item for item in requested_assets if item not in DEFAULT_ASSETS]
    if unknown_assets:
        raise RuntimeError(f"Unknown assets: {', '.join(unknown_assets)}")

    intervals = [item.strip().lower() for item in args.intervals.split(",") if item.strip()]
    unknown_intervals = [item for item in intervals if item not in _intervals()]
    if unknown_intervals:
        raise RuntimeError(f"Unknown intervals: {', '.join(unknown_intervals)}")

    start = _parse_utc(args.from_)
    end = _parse_utc(args.to, default=datetime.now(timezone.utc))
    output_dir = Path(args.output_dir)
    manifest: dict[str, Any] = {
        "generated_at": _iso_utc(datetime.now(timezone.utc)),
        "requested_from": _iso_utc(start),
        "requested_to": _iso_utc(end),
        "assets": {},
    }

    os.environ["SSL_TBANK_VERIFY"] = "true"
    with invest.Client(_token()) as client:
        assets = discover_neo_assets(client, requested_assets)
        for asset in assets:
            print(f"Downloading {asset.name} ({asset.ticker})", flush=True)
            asset_dir = output_dir / f"{asset.key}_{_slug(asset.ticker)}"
            manifest["assets"][asset.key] = {
                "instrument": asdict(asset),
                "intervals": {},
            }

            day_path = asset_dir / "day.csv"
            day_stats = download_candles_to_csv(
                client,
                asset,
                "day",
                start,
                end,
                day_path,
                retries=args.retries,
                pause_seconds=args.pause_seconds,
                include_incomplete=args.include_incomplete,
                workers=args.workers,
                batch_size=args.batch_size,
            )
            if "day" in intervals:
                day_stats["path"] = str(day_path)
                manifest["assets"][asset.key]["intervals"]["day"] = day_stats

            if not day_stats["first_timestamp"]:
                print(f"  no daily candles found for {asset.ticker}; skipping intraday intervals")
                continue

            available_start = _parse_utc(day_stats["first_timestamp"])
            for interval_name in intervals:
                if interval_name == "day":
                    continue
                path = asset_dir / f"{interval_name}.csv"
                stats = download_candles_to_csv(
                    client,
                    asset,
                    interval_name,
                    available_start,
                    end,
                    path,
                    retries=args.retries,
                    pause_seconds=args.pause_seconds,
                    include_incomplete=args.include_incomplete,
                    workers=args.workers,
                    batch_size=args.batch_size,
                )
                stats["path"] = str(path)
                stats["requested_from"] = _iso_utc(available_start)
                stats["requested_to"] = _iso_utc(end)
                manifest["assets"][asset.key]["intervals"][interval_name] = stats
                print(
                    f"  {interval_name}: {stats['rows']} rows, "
                    f"{stats['first_timestamp']} .. {stats['last_timestamp']}"
                )

            if "day" in intervals:
                print(
                    f"  day: {day_stats['rows']} rows, "
                    f"{day_stats['first_timestamp']} .. {day_stats['last_timestamp']}"
                    ,
                    flush=True,
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
