from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from ngn6_bot.config import RuntimeConfig
from ngn6_bot.recorder import read_jsonl_tail


def run_dashboard(config: RuntimeConfig, host: str, port: int) -> None:
    handler = _handler_factory(config)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()


def _handler_factory(config: RuntimeConfig):
    paper_state_file = Path(config.get("paper", "state_file", default="data/paper_state.json"))
    market_file = Path(
        config.get("data_collection", "market_structure_file", default="data/market_structure.jsonl")
    )
    decisions_file = Path(config.get("data_collection", "decisions_file", default="data/decisions.jsonl"))
    review_dir = Path(config.get("review", "output_dir", default="reports/review"))

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self._send_json(_status_payload(config, paper_state_file, market_file, decisions_file, review_dir))
                return
            if parsed.path == "/api/market":
                self._send_json({"rows": read_jsonl_tail(market_file, 300)})
                return
            if parsed.path == "/api/decisions":
                self._send_json({"rows": read_jsonl_tail(decisions_file, 300)})
                return
            if parsed.path == "/api/reviews":
                self._send_json({"rows": _review_files(review_dir)})
                return
            if parsed.path.startswith("/review-images/"):
                self._send_review_image(parsed.path.removeprefix("/review-images/"), review_dir)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_review_image(self, raw_name: str, base_dir: Path) -> None:
            name = Path(unquote(raw_name)).name
            target = (base_dir / name).resolve()
            try:
                base = base_dir.resolve()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if base not in target.parents or not target.exists() or target.suffix.lower() != ".png":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def _status_payload(
    config: RuntimeConfig,
    paper_state_file: Path,
    market_file: Path,
    decisions_file: Path,
    review_dir: Path,
) -> dict[str, Any]:
    market_tail = read_jsonl_tail(market_file, 1)
    decision_tail = read_jsonl_tail(decisions_file, 40)
    return {
        "bot": {
            "ticker": config.get("instrument", "ticker"),
            "dry_run": config.dry_run,
            "timezone": config.timezone,
        },
        "paper": _read_json(paper_state_file),
        "market": market_tail[-1] if market_tail else {},
        "decisions": decision_tail,
        "reviews": _review_files(review_dir),
    }


def _review_files(review_dir: Path) -> list[dict[str, str]]:
    if not review_dir.exists():
        return []
    files = sorted(review_dir.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [
        {
            "name": path.name,
            "url": f"/review-images/{quote(path.name)}",
            "updated_at": str(path.stat().st_mtime),
        }
        for path in files[:20]
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NGN6 AI Trader</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --good: #0f766e;
      --bad: #b91c1c;
      --warn: #a16207;
      --info: #1d4ed8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    main { max-width: 1220px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 18px; }
    h1 { margin: 0 0 6px; font-size: 22px; line-height: 1.2; font-weight: 700; }
    h2 { margin: 0 0 12px; font-size: 15px; line-height: 1.25; font-weight: 700; }
    .sub { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .wide { grid-column: span 2; }
    .full { grid-column: 1 / -1; }
    section, .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 22px; line-height: 1.2; font-weight: 700; margin-top: 6px; overflow-wrap: anywhere; }
    .metric .note { color: var(--muted); margin-top: 4px; font-size: 12px; min-height: 16px; }
    .bad { color: var(--bad); }
    .good { color: var(--good); }
    .warn { color: var(--warn); }
    .info { color: var(--info); }
    .muted { color: var(--muted); }
    .mono { font-family: Consolas, ui-monospace, monospace; }
    .pillbar { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .pill { border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; background: #fff; color: var(--muted); font-size: 12px; line-height: 1; white-space: nowrap; }
    .pill.good { border-color: #99f6e4; background: #f0fdfa; color: var(--good); }
    .pill.bad { border-color: #fecaca; background: #fef2f2; color: var(--bad); }
    .pill.warn { border-color: #fde68a; background: #fffbeb; color: var(--warn); }
    .pill.info { border-color: #bfdbfe; background: #eff6ff; color: var(--info); }
    .table-wrap { width: 100%; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    tr:last-child td, tr:last-child th { border-bottom: 0; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    pre { margin: 0; max-height: 260px; overflow: auto; font: 12px/1.45 Consolas, ui-monospace, monospace; color: #374151; white-space: pre-wrap; }
    .empty { color: var(--muted); margin: 0; }
    .reviews { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
    .reviews a { display: block; color: inherit; text-decoration: none; }
    .reviews img { width: 100%; display: block; border: 1px solid var(--line); border-radius: 8px; background: #111827; }
    .review-name { margin-top: 6px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .depth-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 980px) {
      main { padding: 18px; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .wide { grid-column: 1 / -1; }
    }
    @media (max-width: 640px) {
      main { padding: 14px; }
      header { display: grid; }
      .grid { grid-template-columns: 1fr; }
      .wide { grid-column: auto; }
      .pillbar { justify-content: flex-start; }
      .depth-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1 id="title">NGN6 AI Trader</h1>
      <div class="sub" id="generated">Generated - | auto refresh 5s</div>
    </div>
    <div class="pillbar" id="pillbar">
      <span class="pill warn">loading</span>
    </div>
  </header>

  <div class="grid">
    <div class="metric"><div class="label">Equity</div><div class="value" id="equity">-</div><div class="note" id="equity-note"></div></div>
    <div class="metric"><div class="label">Open / Decisions</div><div class="value" id="counts">-</div><div class="note">position state / latest rows</div></div>
    <div class="metric"><div class="label">Open PnL</div><div class="value" id="open-pnl">-</div><div class="note">unrealized</div></div>
    <div class="metric"><div class="label">Realized PnL</div><div class="value" id="realized-pnl">-</div><div class="note" id="realized-note"></div></div>

    <section class="wide">
      <h2>Open Positions</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>Dir</th><th class="num">Lots</th><th class="num">Entry</th><th class="num">Now</th><th>Src</th><th class="num">PnL</th><th class="num">Stop</th><th class="num">Take</th><th class="num">Signal</th></tr></thead>
          <tbody id="positions"></tbody>
        </table>
      </div>
    </section>

    <section class="wide">
      <h2>Latest Cycle</h2>
      <div class="table-wrap">
        <table><tbody id="latest-cycle"></tbody></table>
      </div>
    </section>

    <section class="wide">
      <h2>Generated Decisions</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Action</th><th>Side</th><th class="num">Price</th><th class="num">Conf</th><th>Reason</th></tr></thead>
          <tbody id="decisions"></tbody>
        </table>
      </div>
    </section>

    <section class="wide">
      <h2>Market Depth</h2>
      <div class="depth-grid">
        <div class="table-wrap">
          <table>
            <thead><tr><th>Bid</th><th class="num">Qty</th></tr></thead>
            <tbody id="bids"></tbody>
          </table>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Ask</th><th class="num">Qty</th></tr></thead>
            <tbody id="asks"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="wide">
      <h2>Recent Trades</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Side</th><th class="num">Price</th><th class="num">Qty</th></tr></thead>
          <tbody id="trades"></tbody>
        </table>
      </div>
    </section>

    <section class="wide">
      <h2>Review Charts</h2>
      <div id="reviews" class="reviews"></div>
    </section>

    <section class="full">
      <h2>Runtime Log</h2>
      <pre id="runtime-log"></pre>
    </section>
  </div>
</main>
<script>
  const rub = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
  const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 });
  const intNumber = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[char]));
  }
  function finite(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  function fmtMoney(value) {
    const parsed = finite(value);
    return parsed === null ? '-' : `${rub.format(parsed)} RUB`;
  }
  function fmtNum(value) {
    const parsed = finite(value);
    return parsed === null ? '-' : number.format(parsed);
  }
  function fmtInt(value) {
    const parsed = finite(value);
    return parsed === null ? '-' : intNumber.format(parsed);
  }
  function fmtPct(value) {
    const parsed = finite(value);
    return parsed === null ? '-' : `${number.format(parsed)}%`;
  }
  function fmtTime(value) {
    if (!value) {
      return '-';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value).replace('T', ' ').slice(0, 19);
    }
    return date.toISOString().replace('T', ' ').slice(0, 19);
  }
  function signedClass(value) {
    const parsed = finite(value);
    if (parsed === null || parsed === 0) {
      return '';
    }
    return parsed > 0 ? 'good' : 'bad';
  }
  function sideClass(side) {
    return side === 'long' ? 'good' : side === 'short' ? 'bad' : 'muted';
  }
  function statusAgeClass(timestamp) {
    const date = new Date(timestamp || 0);
    if (Number.isNaN(date.getTime())) {
      return 'warn';
    }
    const ageSeconds = (Date.now() - date.getTime()) / 1000;
    if (ageSeconds < 20) {
      return 'good';
    }
    if (ageSeconds < 120) {
      return 'warn';
    }
    return 'bad';
  }
  function row(label, value, className = '') {
    return `<tr><th>${esc(label)}</th><td class="${className}">${value || '-'}</td></tr>`;
  }
  function emptyRow(message, colspan) {
    return `<tr><td class="muted" colspan="${colspan}">${esc(message)}</td></tr>`;
  }
  function latestConfidence(decisions) {
    const latest = [...decisions].reverse().find((item) => item?.details?.confidence != null || item?.details?.metadata?.signal_strength != null);
    return latest?.details?.confidence ?? latest?.details?.metadata?.signal_strength ?? null;
  }
  function decisionSide(item) {
    return item?.side || item?.signal?.side || item?.position?.side || '';
  }
  function decisionPrice(item) {
    return item?.price ?? item?.details?.price ?? item?.signal?.price ?? item?.market_context?.last_price ?? null;
  }
  function decisionConfidence(item) {
    return item?.details?.confidence ?? item?.details?.metadata?.signal_strength ?? item?.confidence ?? null;
  }
  function marketSource(market) {
    return market?.orderbook?.source || market?.trade_flow?.source || 'unknown';
  }
  function makePill(text, cls = '') {
    return `<span class="pill ${cls}">${esc(text)}</span>`;
  }
  function updateHeader(data, paper, market) {
    const ticker = data.bot?.ticker || market.ticker || 'NGN6';
    document.getElementById('title').textContent = `${ticker} AI Trader`;
    document.title = `${ticker} AI Trader`;
    document.getElementById('generated').textContent = `Generated ${fmtTime(new Date().toISOString())} | auto refresh 5s`;

    const stateTime = paper.updated_at || market.timestamp;
    const stateClass = statusAgeClass(stateTime);
    const position = paper.position || {};
    const isOpen = position.side && position.side !== 'flat' && Number(position.lots || 0) > 0;
    const mode = data.bot?.dry_run ? 'paper' : 'live';
    const source = marketSource(market);
    document.getElementById('pillbar').innerHTML = [
      makePill(stateClass === 'good' ? 'loop live' : stateClass === 'warn' ? 'loop stale' : 'loop stopped', stateClass),
      makePill(`prices ${source}`, source === 'live' ? 'good' : 'warn'),
      makePill(`${mode} / open=${isOpen ? 'true' : 'false'}`, mode === 'paper' ? 'good' : 'warn'),
      makePill(`ticker ${ticker}`, 'info'),
      makePill(data.bot?.timezone || 'timezone unknown')
    ].join('');
  }
  function updateMetrics(data, paper) {
    const position = paper.position || {};
    const openCount = position.side && position.side !== 'flat' && Number(position.lots || 0) > 0 ? 1 : 0;
    const decisionsCount = (data.decisions || []).length;
    const realizedPct = finite(paper.initial_cash) ? (Number(paper.realized_pnl || 0) / Number(paper.initial_cash)) * 100 : null;

    document.getElementById('equity').textContent = fmtMoney(paper.equity);
    document.getElementById('equity-note').textContent = fmtTime(paper.updated_at);
    document.getElementById('counts').textContent = `${openCount} / ${decisionsCount}`;
    document.getElementById('open-pnl').textContent = fmtMoney(paper.unrealized_pnl);
    document.getElementById('open-pnl').className = `value ${signedClass(paper.unrealized_pnl)}`;
    document.getElementById('realized-pnl').textContent = fmtMoney(paper.realized_pnl);
    document.getElementById('realized-pnl').className = `value ${signedClass(paper.realized_pnl)}`;
    document.getElementById('realized-note').textContent = realizedPct === null ? 'initial cash -' : `initial cash ${fmtPct(realizedPct)}`;
  }
  function updatePositions(data, paper, market) {
    const position = paper.position || {};
    const isOpen = position.side && position.side !== 'flat' && Number(position.lots || 0) > 0;
    if (!isOpen) {
      document.getElementById('positions').innerHTML = emptyRow('No open position.', 10);
      return;
    }
    const confidence = latestConfidence(data.decisions || []);
    const take = [position.take_profit1, position.take_profit2].filter((value) => value != null).map(fmtNum).join(' / ');
    document.getElementById('positions').innerHTML = `
      <tr>
        <td>${esc(data.bot?.ticker || market.ticker || 'NGN6')}</td>
        <td class="${sideClass(position.side)}">${esc(position.side)}</td>
        <td class="num">${fmtInt(position.lots)}</td>
        <td class="num">${fmtNum(position.avg_price)}</td>
        <td class="num">${fmtNum(paper.mark_price ?? market.last_price)}</td>
        <td>${esc(marketSource(market))}</td>
        <td class="num ${signedClass(paper.unrealized_pnl)}">${fmtMoney(paper.unrealized_pnl)}</td>
        <td class="num">${fmtNum(position.stop_price ?? position.trailing_stop)}</td>
        <td class="num">${take || '-'}</td>
        <td class="num">${fmtNum(confidence)}</td>
      </tr>
    `;
  }
  function updateLatestCycle(data, paper, market) {
    const ob = market.orderbook || {};
    const flow = market.trade_flow || {};
    const counts = market.candle_counts || {};
    document.getElementById('latest-cycle').innerHTML = [
      row('equity_rub', fmtMoney(paper.equity)),
      row('cash_rub', fmtMoney(paper.cash)),
      row('margin_used_rub', fmtMoney(paper.margin_used)),
      row('margin_available_rub', fmtMoney(paper.margin_available)),
      row('contract_value_rub', fmtMoney(paper.contract_value)),
      row('price_step_rub', fmtMoney(paper.money_value_per_price_step)),
      row('go_buy_sell_rub', `${fmtMoney(paper.initial_margin_on_buy)} / ${fmtMoney(paper.initial_margin_on_sell)}`),
      row('last_price', fmtNum(market.last_price ?? paper.mark_price)),
      row('best_bid_ask', `${fmtNum(ob.best_bid)} / ${fmtNum(ob.best_ask)}`),
      row('spread_bps', fmtNum(ob.spread_bps)),
      row('bid_ask_imbalance', fmtNum(ob.bid_ask_imbalance)),
      row('trade_flow_volume', fmtInt(flow.total_volume)),
      row('signals_total', fmtInt((data.decisions || []).length)),
      row('candles_1min', fmtInt(counts['1min'])),
      row('trading_halted', 'False')
    ].join('');
  }
  function updateDecisions(decisions) {
    const rows = (decisions || []).slice().reverse().slice(0, 10).map((item) => {
      const side = decisionSide(item);
      const conf = decisionConfidence(item);
      return `
        <tr>
          <td class="muted">${esc(fmtTime(item.timestamp))}</td>
          <td>${esc(item.action || '')}</td>
          <td class="${sideClass(side)}">${esc(side || '-')}</td>
          <td class="num">${fmtNum(decisionPrice(item))}</td>
          <td class="num">${fmtNum(conf)}</td>
          <td>${esc(item.reason || item.signal_reason || '')}</td>
        </tr>
      `;
    }).join('');
    document.getElementById('decisions').innerHTML = rows || emptyRow('No decisions yet.', 6);
  }
  function updateDepth(market) {
    const snapshot = market.orderbook_snapshot || {};
    document.getElementById('bids').innerHTML = (snapshot.bids || []).slice(0, 8).map((level) => `
      <tr><td class="good">${fmtNum(level.price)}</td><td class="num">${fmtInt(level.quantity)}</td></tr>
    `).join('') || emptyRow('No bids.', 2);
    document.getElementById('asks').innerHTML = (snapshot.asks || []).slice(0, 8).map((level) => `
      <tr><td class="bad">${fmtNum(level.price)}</td><td class="num">${fmtInt(level.quantity)}</td></tr>
    `).join('') || emptyRow('No asks.', 2);
  }
  function updateTrades(market) {
    const rows = (market.recent_trades || []).slice().reverse().slice(0, 10).map((trade) => `
      <tr>
        <td class="muted">${esc(fmtTime(trade.timestamp))}</td>
        <td class="${sideClass(trade.side)}">${esc(trade.side || '-')}</td>
        <td class="num">${fmtNum(trade.price)}</td>
        <td class="num">${fmtInt(trade.quantity)}</td>
      </tr>
    `).join('');
    document.getElementById('trades').innerHTML = rows || emptyRow('No recent trades.', 4);
  }
  function updateReviews(reviews) {
    document.getElementById('reviews').innerHTML = (reviews || []).slice(0, 6).map((item) => `
      <a href="${esc(item.url)}" target="_blank" rel="noopener" title="${esc(item.name)}">
        <img src="${esc(item.url)}" alt="${esc(item.name)}">
        <div class="review-name">${esc(item.name)}</div>
      </a>
    `).join('') || '<p class="empty">No review charts yet.</p>';
  }
  function updateRuntimeLog(data, paper, market) {
    const latestDecision = (data.decisions || []).slice(-1)[0] || null;
    const summary = {
      bot: data.bot || {},
      paper_updated_at: paper.updated_at || null,
      market_timestamp: market.timestamp || null,
      latest_decision: latestDecision ? {
        timestamp: latestDecision.timestamp,
        action: latestDecision.action,
        reason: latestDecision.reason,
        confidence: decisionConfidence(latestDecision)
      } : null
    };
    document.getElementById('runtime-log').textContent = JSON.stringify(summary, null, 2);
  }
  async function load() {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }
    const data = await response.json();
    const paper = data.paper || {};
    const market = data.market || {};
    updateHeader(data, paper, market);
    updateMetrics(data, paper);
    updatePositions(data, paper, market);
    updateLatestCycle(data, paper, market);
    updateDecisions(data.decisions || []);
    updateDepth(market);
    updateTrades(market);
    updateReviews(data.reviews || []);
    updateRuntimeLog(data, paper, market);
  }
  load().catch((error) => {
    document.getElementById('pillbar').innerHTML = makePill(`load failed: ${error.message}`, 'bad');
    console.error(error);
  });
  setInterval(() => load().catch(console.error), 5000);
</script>
</body>
</html>
"""
