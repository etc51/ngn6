const test = require("node:test");
const assert = require("node:assert/strict");
const { buildFallbackSnapshot } = require("../src/lib/fallbackMarket");
const { computeSignal, backtest } = require("../src/lib/signalEngine");

function extractLevelNumbers(text) {
  return (String(text).match(/\d+(?:\.\d+)?/g) || []).map(Number);
}

test("computeSignal returns a bounded probability and known headline", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const result = computeSignal(snapshot, {
    newsBias: "positive",
    retest: "bullish",
    structure: "uptrend",
    eventRisk: "none",
  });

  assert.ok(["Покупка", "Ожидание", "Продажа"].includes(result.headline));
  assert.ok(result.probability >= 50 && result.probability <= 69);
  assert.ok(Array.isArray(result.factors));
  assert.equal(typeof result.decisionPlan.summary, "string");
  assert.equal(result.decisionPlan.steps.length, 3);
  assert.ok(result.tradePlan);
  assert.ok(result.tradePlan.permissions);
  assert.ok(Array.isArray(result.tradePlan.entries));
  assert.ok(Array.isArray(result.tradePlan.exits));
});

test("intraday signal uses a tighter probability cap", () => {
  const snapshot = buildFallbackSnapshot("intraday");
  const result = computeSignal(snapshot, {
    newsBias: "negative",
    retest: "bearish",
    structure: "downtrend",
    eventRisk: "none",
  });

  assert.equal(result.timeframe, "intraday");
  assert.ok(result.probability >= 50 && result.probability <= 64.5);
});

test("news bias materially changes the gas score", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const bullish = computeSignal(snapshot, {
    newsBias: "positive",
    retest: "bullish",
    structure: "uptrend",
    eventRisk: "none",
  });
  const bearish = computeSignal(snapshot, {
    newsBias: "negative",
    retest: "bearish",
    structure: "downtrend",
    eventRisk: "none",
  });

  assert.ok(bullish.score > bearish.score);
});

test("decision plan stays in a tactical distance from current price", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const result = computeSignal(snapshot, {
    newsBias: "negative",
    retest: "bearish",
    structure: "downtrend",
    eventRisk: "none",
  });

  const atr = result.marketLevels.atr;
  const levelNumbers = result.decisionPlan.steps.flatMap((step) => extractLevelNumbers(step.levelText));
  const maxDistance = Math.max(...levelNumbers.map((level) => Math.abs(level - result.close)));

  assert.ok(Number.isFinite(atr) && atr > 0);
  assert.ok(maxDistance <= atr * 3 + 0.8);
});

test("directional plans stay on the correct side of current price", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const shortResult = computeSignal(snapshot, {
    newsBias: "negative",
    retest: "bearish",
    structure: "downtrend",
    eventRisk: "none",
  });
  const longResult = computeSignal(snapshot, {
    newsBias: "positive",
    retest: "bullish",
    structure: "uptrend",
    eventRisk: "none",
  });

  const shortNumbers = shortResult.decisionPlan.steps.map((step) => extractLevelNumbers(step.levelText));
  const longNumbers = longResult.decisionPlan.steps.map((step) => extractLevelNumbers(step.levelText));

  if (shortResult.signal === "south") {
    assert.ok(shortNumbers[0][0] >= shortResult.close || shortNumbers[0][1] >= shortResult.close);
    assert.ok(shortNumbers[2][0] > shortResult.close);
  }

  if (longResult.signal === "north") {
    assert.ok(longNumbers[0][0] <= longResult.close || longNumbers[0][1] <= longResult.close);
    assert.ok(longNumbers[2][0] < longResult.close);
  }
});

test("displayed support and resistance levels are ordered around current price", () => {
  for (const timeframe of ["intraday", "daily"]) {
    const snapshot = buildFallbackSnapshot(timeframe);
    const result = computeSignal(snapshot, {
      newsBias: "negative",
      retest: "bearish",
      structure: "downtrend",
      eventRisk: "none",
    });

    const resistanceValues = result.tradePlan.resistance.map((item) => Number(item.value));
    const supportValues = result.tradePlan.support.map((item) => Number(item.value));
    const atr = result.marketLevels.atr;

    assert.ok(resistanceValues.every((value) => value >= result.close));
    assert.ok(supportValues.every((value) => value <= result.close));
    assert.ok(resistanceValues[1] >= resistanceValues[0]);
    assert.ok(supportValues[1] <= supportValues[0]);
    assert.ok(resistanceValues[1] - result.close <= atr * 4 + 0.5);
    assert.ok(result.close - supportValues[1] <= atr * 4 + 0.5);
  }
});

test("backtest produces a non-negative sample set", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const result = backtest(snapshot, 90);

  assert.ok(result.trades >= 0);
  assert.ok(result.wins >= 0);
  assert.ok(result.accuracy >= 0 && result.accuracy <= 100);
  assert.ok(Array.isArray(result.samples));
});
