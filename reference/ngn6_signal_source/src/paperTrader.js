const fs = require("fs");
const path = require("path");
const {
  adjustStopAfterTakeProfit1,
  buildExecutablePlan,
  buildSignalPayload,
  getMoscowClock,
  sendNtfy,
} = require("./signalWatcher");
const { defaultHeartbeatFile, runWithTimeout, writeHeartbeat } = require("./lib/heartbeat");
const { fetchTBankGasOrderBook } = require("./lib/tbankClient");

const DEFAULT_STATE_FILE = path.join(process.cwd(), "data", "paper-trader-state.json");
const DEFAULT_EVENT_DIR = path.join(process.cwd(), "data", "paper", "events");
const DEFAULT_ORDERBOOK_DIR = path.join(process.cwd(), "data", "paper", "orderbook");
const DEFAULT_HEARTBEAT_FILE = defaultHeartbeatFile("paper-trader");

function loadEnvFile() {
  const envPath = path.join(process.cwd(), ".env");
  if (!fs.existsSync(envPath)) {
    return;
  }

  for (const rawLine of fs.readFileSync(envPath, "utf8").split(/\r?\n/u)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const match = line.match(/^([^#=]+?)\s*=\s*(.*)$/u);
    if (!match) {
      continue;
    }

    const key = match[1].trim();
    if (process.env[key]) {
      continue;
    }

    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

function round(value, digits = 3) {
  if (!Number.isFinite(Number(value))) {
    return null;
  }
  return Number(Number(value).toFixed(digits));
}

function loadState(filePath) {
  try {
    return {
      startedAt: new Date().toISOString(),
      realizedPnl: 0,
      tradeCount: 0,
      openPosition: null,
      lastTickAt: null,
      ...JSON.parse(fs.readFileSync(filePath, "utf8")),
    };
  } catch (_error) {
    return {
      startedAt: new Date().toISOString(),
      realizedPnl: 0,
      tradeCount: 0,
      openPosition: null,
      lastTickAt: null,
    };
  }
}

function saveState(filePath, state) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function appendJsonl(filePath, record) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.appendFileSync(filePath, `${JSON.stringify(record)}\n`, "utf8");
}

function dateKey(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

function sumQuantity(levels, depth) {
  return levels.slice(0, depth).reduce((sum, level) => sum + Number(level.quantity || 0), 0);
}

function weightedAverage(levels, depth) {
  const selected = levels.slice(0, depth);
  const qty = selected.reduce((sum, level) => sum + Number(level.quantity || 0), 0);
  if (!qty) {
    return null;
  }

  return (
    selected.reduce((sum, level) => sum + Number(level.price || 0) * Number(level.quantity || 0), 0) /
    qty
  );
}

function findWall(levels, depth) {
  return levels.slice(0, depth).reduce(
    (best, level) => (Number(level.quantity || 0) > Number(best.quantity || 0) ? level : best),
    { price: null, quantity: 0 },
  );
}

function summarizeOrderBook(orderBook, depth = 20) {
  const bids = Array.isArray(orderBook?.bids) ? orderBook.bids : [];
  const asks = Array.isArray(orderBook?.asks) ? orderBook.asks : [];
  const bestBid = bids[0]?.price ?? null;
  const bestAsk = asks[0]?.price ?? null;
  const spread = Number.isFinite(bestBid) && Number.isFinite(bestAsk) ? bestAsk - bestBid : null;
  const mid = spread == null ? orderBook?.lastPrice ?? null : (bestBid + bestAsk) / 2;
  const bidQtyTop = sumQuantity(bids, 1);
  const askQtyTop = sumQuantity(asks, 1);
  const bidQty5 = sumQuantity(bids, Math.min(depth, 5));
  const askQty5 = sumQuantity(asks, Math.min(depth, 5));
  const bidQty = sumQuantity(bids, depth);
  const askQty = sumQuantity(asks, depth);
  const topTotal = bidQtyTop + askQtyTop;
  const depthTotal = bidQty + askQty;

  return {
    bestBid,
    bestAsk,
    mid: round(mid),
    spread: round(spread),
    spreadPct: mid ? round((spread / mid) * 100, 4) : null,
    bidQtyTop,
    askQtyTop,
    bidQty5,
    askQty5,
    bidQty,
    askQty,
    topImbalance: topTotal ? round((bidQtyTop - askQtyTop) / topTotal, 4) : null,
    depthImbalance: depthTotal ? round((bidQty - askQty) / depthTotal, 4) : null,
    weightedBid: round(weightedAverage(bids, depth)),
    weightedAsk: round(weightedAverage(asks, depth)),
    microprice:
      topTotal && Number.isFinite(bestBid) && Number.isFinite(bestAsk)
        ? round((bestAsk * bidQtyTop + bestBid * askQtyTop) / topTotal)
        : null,
    bidWall: findWall(bids, depth),
    askWall: findWall(asks, depth),
    orderbookTs: orderBook?.orderbookTs || null,
    fetchedAt: orderBook?.fetchedAt || new Date().toISOString(),
  };
}

function getOptionsFromEnv() {
  return {
    enabled: process.env.PAPER_TRADER_ENABLED !== "false",
    intervalMs: Number(process.env.PAPER_TRADER_INTERVAL_MS || 60_000),
    quantity: Number(process.env.PAPER_POSITION_CONTRACTS || 1),
    pointValue: Number(process.env.PAPER_POINT_VALUE || 1),
    depth: Number(process.env.PAPER_ORDERBOOK_DEPTH || 20),
    takeProfit1Fraction: Number(process.env.PAPER_TP1_FRACTION || 0.5),
    closeOnSignalFlip: process.env.PAPER_CLOSE_ON_SIGNAL_FLIP !== "false",
    closeOnNeutral: process.env.PAPER_CLOSE_ON_NEUTRAL === "true",
    notifyTrades: process.env.PAPER_NOTIFY_TRADES !== "false",
    stateFile: process.env.PAPER_STATE_FILE || DEFAULT_STATE_FILE,
    eventDir: process.env.PAPER_EVENT_DIR || DEFAULT_EVENT_DIR,
    orderBookDir: process.env.PAPER_ORDERBOOK_DIR || DEFAULT_ORDERBOOK_DIR,
    heartbeatFile: process.env.PAPER_HEARTBEAT_FILE || DEFAULT_HEARTBEAT_FILE,
    cycleTimeoutMs: Number(process.env.PAPER_CYCLE_TIMEOUT_MS || 180_000),
  };
}

function getMarkPrice(side, bookSummary, fallback) {
  if (side === "long") {
    return Number(bookSummary.bestBid ?? fallback);
  }
  return Number(bookSummary.bestAsk ?? fallback);
}

function getEntryFillPrice(side, bookSummary, fallback) {
  if (side === "long") {
    return Number(bookSummary.bestAsk ?? fallback);
  }
  return Number(bookSummary.bestBid ?? fallback);
}

function getExitFillPrice(side, bookSummary, fallback) {
  if (side === "long") {
    return Number(bookSummary.bestBid ?? fallback);
  }
  return Number(bookSummary.bestAsk ?? fallback);
}

function isEntryTriggered(plan, payload, bookSummary) {
  const price = Number(payload.close);
  const entry = Number(plan?.entry);
  if (!plan || !Number.isFinite(entry)) {
    return false;
  }

  if (plan.side === "long") {
    return Math.max(Number(bookSummary.bestAsk ?? price), price) >= entry;
  }

  return Math.min(Number(bookSummary.bestBid ?? price), price) <= entry;
}

function isStopTriggered(position, bookSummary, fallback) {
  if (!position?.stop) {
    return false;
  }

  const price = getMarkPrice(position.side, bookSummary, fallback);
  return position.side === "long" ? price <= position.stop : price >= position.stop;
}

function isTargetTriggered(position, bookSummary, fallback, targetKey) {
  const target = Number(position[targetKey]);
  if (!Number.isFinite(target)) {
    return false;
  }

  const price = getMarkPrice(position.side, bookSummary, fallback);
  return position.side === "long" ? price >= target : price <= target;
}

function calculatePnl(side, entryPrice, exitPrice, quantity, pointValue) {
  const raw = side === "long" ? exitPrice - entryPrice : entryPrice - exitPrice;
  return raw * quantity * pointValue;
}

function markToMarket(position, bookSummary, fallback, pointValue) {
  if (!position) {
    return {
      unrealizedPnl: 0,
      markPrice: null,
      pnlPct: 0,
    };
  }

  const markPrice = getMarkPrice(position.side, bookSummary, fallback);
  const unrealizedPnl = calculatePnl(
    position.side,
    position.entryPrice,
    markPrice,
    position.remainingQuantity,
    pointValue,
  );
  const pnlPct =
    position.side === "long"
      ? ((markPrice - position.entryPrice) / position.entryPrice) * 100
      : ((position.entryPrice - markPrice) / position.entryPrice) * 100;

  return {
    markPrice: round(markPrice),
    unrealizedPnl: round(unrealizedPnl, 4),
    pnlPct: round(pnlPct, 4),
  };
}

function createOpenPosition({ payload, plan, bookSummary, options, reason }) {
  const fallback = Number(payload.close);
  const entryPrice = round(getEntryFillPrice(plan.side, bookSummary, fallback));

  return {
    id: `${Date.now()}-${plan.side}`,
    side: plan.side,
    signal: payload.signal,
    reason,
    quantity: options.quantity,
    remainingQuantity: options.quantity,
    entryPrice,
    plannedEntry: plan.entry,
    stop: plan.stop,
    takeProfit1: plan.takeProfit1,
    takeProfit2: plan.takeProfit2,
    takeProfit1Filled: false,
    stopMovedAfterTakeProfit1: false,
    openedAt: new Date().toISOString(),
    openedSignal: {
      headline: payload.headline,
      probability: payload.probability,
      origin: payload.signalOrigin || null,
      close: payload.close,
      date: payload.date,
    },
  };
}

function closeQuantity(state, position, quantity, exitPrice, reason, options) {
  const closedQuantity = Math.min(quantity, position.remainingQuantity);
  const pnl = calculatePnl(position.side, position.entryPrice, exitPrice, closedQuantity, options.pointValue);
  position.remainingQuantity = round(position.remainingQuantity - closedQuantity, 6);
  state.realizedPnl = round(Number(state.realizedPnl || 0) + pnl, 4);
  state.tradeCount = Number(state.tradeCount || 0) + 1;

  const trade = {
    type: "paper-fill",
    reason,
    side: position.side,
    quantity: closedQuantity,
    entryPrice: position.entryPrice,
    exitPrice,
    pnl: round(pnl, 4),
    realizedPnl: state.realizedPnl,
    remainingQuantity: position.remainingQuantity,
    at: new Date().toISOString(),
  };

  if (position.remainingQuantity <= 0) {
    trade.closedPositionId = position.id;
    state.openPosition = null;
  }

  return trade;
}

function isOppositeSignal(position, payload) {
  return (
    (position.side === "long" && payload.signal === "south") ||
    (position.side === "short" && payload.signal === "north")
  );
}

function moveStopAfterTp1(position, payload) {
  const adjustedStop = adjustStopAfterTakeProfit1(
    {
      side: position.side,
      entry: position.entryPrice,
      stop: position.stop,
      takeProfit1: position.takeProfit1,
      takeProfit2: position.takeProfit2,
    },
    payload,
  );

  if (adjustedStop != null && adjustedStop !== position.stop) {
    position.stop = adjustedStop;
    position.stopMovedAfterTakeProfit1 = true;
    return adjustedStop;
  }

  return null;
}

function applyPaperDecision({ state, payload, plan, bookSummary, options }) {
  const events = [];
  const fallback = Number(payload.close);

  if (!state.openPosition) {
    if (!plan || payload.signal === "neutral" || plan.allowed === false) {
      return {
        action: "wait-no-plan",
        events,
      };
    }

    if (!isEntryTriggered(plan, payload, bookSummary)) {
      return {
        action: "wait-entry",
        events,
      };
    }

    const position = createOpenPosition({
      payload,
      plan,
      bookSummary,
      options,
      reason: "signal-entry-triggered",
    });
    state.openPosition = position;
    events.push({
      type: "paper-open",
      position,
      at: new Date().toISOString(),
    });
    return {
      action: "open-position",
      events,
    };
  }

  const position = state.openPosition;

  if (isStopTriggered(position, bookSummary, fallback)) {
    const exitPrice = round(getExitFillPrice(position.side, bookSummary, fallback));
    events.push(closeQuantity(state, position, position.remainingQuantity, exitPrice, "stop", options));
    return {
      action: "stop",
      events,
    };
  }

  if (isTargetTriggered(position, bookSummary, fallback, "takeProfit2")) {
    const exitPrice = round(getExitFillPrice(position.side, bookSummary, fallback));
    events.push(closeQuantity(state, position, position.remainingQuantity, exitPrice, "take-profit-2", options));
    return {
      action: "take-profit-2",
      events,
    };
  }

  if (!position.takeProfit1Filled && isTargetTriggered(position, bookSummary, fallback, "takeProfit1")) {
    const exitPrice = round(getExitFillPrice(position.side, bookSummary, fallback));
    const quantityToClose = Math.max(1, Math.floor(position.remainingQuantity * options.takeProfit1Fraction));
    events.push(closeQuantity(state, position, quantityToClose, exitPrice, "take-profit-1", options));
    if (state.openPosition) {
      state.openPosition.takeProfit1Filled = true;
      const adjustedStop = moveStopAfterTp1(state.openPosition, payload);
      if (adjustedStop != null) {
        events.push({
          type: "paper-stop-move",
          reason: "after-take-profit-1",
          stop: adjustedStop,
          at: new Date().toISOString(),
        });
      }
    }
    return {
      action: "take-profit-1",
      events,
    };
  }

  if (
    options.closeOnSignalFlip &&
    (isOppositeSignal(position, payload) || (options.closeOnNeutral && payload.signal === "neutral"))
  ) {
    const exitPrice = round(getExitFillPrice(position.side, bookSummary, fallback));
    events.push(closeQuantity(state, position, position.remainingQuantity, exitPrice, "signal-exit", options));
    return {
      action: "signal-exit",
      events,
    };
  }

  return {
    action: "hold",
    events,
  };
}

function eventFile(options) {
  return path.join(options.eventDir, `${dateKey()}.jsonl`);
}

function orderBookFile(options) {
  return path.join(options.orderBookDir, `${dateKey()}.jsonl`);
}

async function notifyTradeEvents(events, options) {
  if (!options.notifyTrades) {
    return;
  }

  for (const event of events) {
    if (!["paper-open", "paper-fill", "paper-stop-move"].includes(event.type)) {
      continue;
    }

    try {
      await sendNtfy(`NGN6 PAPER ${event.type}\n${JSON.stringify(event, null, 2)}`);
    } catch (_error) {
      // Paper trading must never stop because a notification channel failed.
    }
  }
}

async function runPaperCycle(state, options) {
  const [payload, orderBook] = await Promise.all([
    buildSignalPayload("intraday"),
    fetchTBankGasOrderBook(options.depth),
  ]);
  const plan = buildExecutablePlan(payload);
  const bookSummary = summarizeOrderBook(orderBook, options.depth);
  const decision = applyPaperDecision({ state, payload, plan, bookSummary, options });
  const mtm = markToMarket(state.openPosition, bookSummary, payload.close, options.pointValue);
  const clock = getMoscowClock();

  state.lastTickAt = new Date().toISOString();
  state.lastSignal = {
    signal: payload.signal,
    headline: payload.headline,
    probability: payload.probability,
    close: payload.close,
    date: payload.date,
  };
  state.lastBook = bookSummary;
  state.lastMarkToMarket = mtm;
  saveState(options.stateFile, state);

  appendJsonl(orderBookFile(options), {
    type: "orderbook",
    clock,
    summary: bookSummary,
    orderBook,
  });

  appendJsonl(eventFile(options), {
    type: "paper-tick",
    clock,
    action: decision.action,
    signal: payload,
    plan,
    orderBookSummary: bookSummary,
    markToMarket: mtm,
    state,
    events: decision.events,
  });

  for (const event of decision.events) {
    appendJsonl(eventFile(options), {
      ...event,
      clock,
      orderBookSummary: bookSummary,
      signal: state.lastSignal,
    });
  }

  await notifyTradeEvents(decision.events, options);
  console.log(
    JSON.stringify({
      at: state.lastTickAt,
      action: decision.action,
      signal: payload.signal,
      probability: payload.probability,
      close: payload.close,
      bestBid: bookSummary.bestBid,
      bestAsk: bookSummary.bestAsk,
      position: state.openPosition,
      realizedPnl: state.realizedPnl,
      markToMarket: mtm,
    }),
  );
}

async function runSupervisedPaperCycle(state, options) {
  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  writeHeartbeat(options.heartbeatFile, {
    service: "paper-trader",
    status: "cycle-start",
    startedAt,
  });

  try {
    await runWithTimeout(() => runPaperCycle(state, options), options.cycleTimeoutMs, "paper trader cycle");
    writeHeartbeat(options.heartbeatFile, {
      service: "paper-trader",
      status: "ok",
      startedAt,
      durationMs: Date.now() - startedAtMs,
      intervalMs: options.intervalMs,
      cycleTimeoutMs: options.cycleTimeoutMs,
    });
  } catch (error) {
    const record = {
      type: "paper-error",
      at: new Date().toISOString(),
      message: error.message,
      code: error.code || null,
      stack: error.stack,
    };
    appendJsonl(eventFile(options), record);
    writeHeartbeat(options.heartbeatFile, {
      service: "paper-trader",
      status: "error",
      startedAt,
      durationMs: Date.now() - startedAtMs,
      message: error.message,
      code: error.code || null,
    });
    console.error(JSON.stringify(record));
    if (error.code === "CYCLE_TIMEOUT") {
      process.exit(1);
    }
  }
}

function scheduleSupervisedPaperLoop(state, options) {
  setTimeout(() => {
    runSupervisedPaperCycle(state, options).finally(() => {
      scheduleSupervisedPaperLoop(state, options);
    });
  }, options.intervalMs);
}

async function main() {
  loadEnvFile();
  const options = getOptionsFromEnv();
  const once = process.argv.includes("--once") || process.env.PAPER_TRADER_ONCE === "true";
  if (!options.enabled) {
    console.log("Paper trader disabled by PAPER_TRADER_ENABLED=false");
    return;
  }

  const state = loadState(options.stateFile);
  console.log(`Paper trader started: every ${options.intervalMs} ms, depth=${options.depth}`);
  writeHeartbeat(options.heartbeatFile, {
    service: "paper-trader",
    status: "starting",
    intervalMs: options.intervalMs,
    cycleTimeoutMs: options.cycleTimeoutMs,
  });
  await runSupervisedPaperCycle(state, options);
  if (once) {
    return;
  }

  scheduleSupervisedPaperLoop(state, options);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  applyPaperDecision,
  calculatePnl,
  isEntryTriggered,
  markToMarket,
  summarizeOrderBook,
};
