const test = require("node:test");
const assert = require("node:assert/strict");
const {
  applyPaperDecision,
  calculatePnl,
  isEntryTriggered,
  markToMarket,
  summarizeOrderBook,
} = require("../src/paperTrader");

const longPayload = {
  signal: "north",
  headline: "Buy",
  probability: 61,
  close: 3.326,
  date: "2026-06-30 12:00 UTC",
  tradeLevels: {},
};

const shortPayload = {
  ...longPayload,
  signal: "south",
  headline: "Sell",
};

test("summarizeOrderBook computes spread, imbalance and walls", () => {
  const summary = summarizeOrderBook({
    bids: [
      { price: 3.326, quantity: 325 },
      { price: 3.325, quantity: 100 },
    ],
    asks: [
      { price: 3.327, quantity: 25 },
      { price: 3.328, quantity: 75 },
    ],
    fetchedAt: "2026-06-30T09:00:00.000Z",
  });

  assert.equal(summary.bestBid, 3.326);
  assert.equal(summary.bestAsk, 3.327);
  assert.equal(summary.spread, 0.001);
  assert.ok(summary.topImbalance > 0);
  assert.deepEqual(summary.bidWall, { price: 3.326, quantity: 325 });
});

test("isEntryTriggered uses executable live price, not candle extremes", () => {
  assert.equal(
    isEntryTriggered(
      { side: "long", entry: 3.33 },
      { ...longPayload, marketLevels: { currentHigh: 3.35 } },
      { bestAsk: 3.327 },
    ),
    false,
  );
  assert.equal(isEntryTriggered({ side: "long", entry: 3.33 }, longPayload, { bestAsk: 3.331 }), true);
  assert.equal(isEntryTriggered({ side: "short", entry: 3.32 }, shortPayload, { bestBid: 3.319 }), true);
});

test("calculatePnl handles long and short paper fills", () => {
  assert.ok(Math.abs(calculatePnl("long", 3.3, 3.35, 10, 100) - 50) < 0.000001);
  assert.ok(Math.abs(calculatePnl("short", 3.3, 3.25, 10, 100) - 50) < 0.000001);
});

test("applyPaperDecision opens, partially closes at TP1 and moves stop", () => {
  const state = { realizedPnl: 0, tradeCount: 0, openPosition: null };
  const options = {
    quantity: 4,
    pointValue: 100,
    takeProfit1Fraction: 0.5,
    closeOnSignalFlip: true,
    closeOnNeutral: false,
  };
  const plan = {
    side: "long",
    entry: 3.3,
    stop: 3.25,
    takeProfit1: 3.35,
    takeProfit2: 3.4,
    allowed: true,
  };

  const open = applyPaperDecision({
    state,
    payload: { ...longPayload, close: 3.301, marketLevels: { atr: 0.02 } },
    plan,
    bookSummary: { bestBid: 3.3, bestAsk: 3.301 },
    options,
  });

  assert.equal(open.action, "open-position");
  assert.equal(state.openPosition.remainingQuantity, 4);
  assert.equal(state.openPosition.entryPrice, 3.301);

  const tp1 = applyPaperDecision({
    state,
    payload: { ...longPayload, close: 3.356, marketLevels: { atr: 0.02 } },
    plan,
    bookSummary: { bestBid: 3.355, bestAsk: 3.356 },
    options,
  });

  assert.equal(tp1.action, "take-profit-1");
  assert.equal(state.openPosition.remainingQuantity, 2);
  assert.equal(state.openPosition.takeProfit1Filled, true);
  assert.ok(state.openPosition.stop > 3.25);
  assert.ok(tp1.events.some((event) => event.type === "paper-stop-move"));
});

test("markToMarket marks longs on bid and shorts on ask", () => {
  assert.deepEqual(
    markToMarket({ side: "long", entryPrice: 3.3, remainingQuantity: 10 }, { bestBid: 3.31 }, 3.3, 100),
    { markPrice: 3.31, unrealizedPnl: 10, pnlPct: 0.303 },
  );
  assert.deepEqual(
    markToMarket({ side: "short", entryPrice: 3.3, remainingQuantity: 10 }, { bestAsk: 3.29 }, 3.3, 100),
    { markPrice: 3.29, unrealizedPnl: 10, pnlPct: 0.303 },
  );
});
