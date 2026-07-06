const test = require("node:test");
const assert = require("node:assert/strict");
const { inferRetest, inferStructure, resolveNewsBias } = require("../src/lib/autoContext");
const { buildFallbackSnapshot } = require("../src/lib/fallbackMarket");
const { classifyCalendarRisk, getFallbackUpcomingEvent } = require("../src/lib/officialCalendar");
const { buildFeatureSet, computeSignal } = require("../src/lib/signalEngine");

test("inferStructure recognizes an aligned bullish structure", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const features = buildFeatureSet(snapshot, {});

  features.close = 4.2;
  features.levels.smaFast = 4.0;
  features.levels.smaSlow = 3.8;
  features.breakdown.trendBias = 0.35;

  assert.equal(inferStructure(features), "uptrend");
});

test("inferRetest can detect a bullish retest near local support", () => {
  const retest = inferRetest(
    {
      close: 4,
      levels: {
        smaFast: 4.03,
        highLocal: 4.22,
        lowLocal: 3.97,
      },
    },
    "uptrend",
  );

  assert.equal(retest, "bullish");
});

test("neutral RSS news bias remains neutral", () => {
  assert.equal(resolveNewsBias({ bias: "neutral" }), "neutral");
  assert.equal(resolveNewsBias({ bias: "negative" }), "negative");
  assert.equal(resolveNewsBias({ bias: "unexpected" }), "neutral");
});

test("calendar marks same-day EIA gas storage release as high risk", () => {
  assert.equal(classifyCalendarRisk({ daysUntil: 0, severity: "high" }), "high");
  assert.equal(classifyCalendarRisk({ daysUntil: 1, severity: "scheduled" }), "scheduled");
  assert.equal(classifyCalendarRisk({ daysUntil: 3, severity: "scheduled" }), "none");
});

test("recurring calendar provides the next weekly gas event without network fetch", () => {
  const event = getFallbackUpcomingEvent(
    {
      code: "EIA_GAS",
      label: "EIA weekly natural gas storage report",
      severity: "high",
      weekday: 4,
    },
    new Date("2026-06-27T00:00:00Z"),
  );

  assert.equal(event.code, "EIA_GAS");
  assert.equal(event.date, "2026-07-02");
  assert.equal(event.daysUntil, 5);
});

test("high event risk lowers the final probability", () => {
  const snapshot = buildFallbackSnapshot("daily");
  const baseManual = {
    newsBias: "positive",
    retest: "bullish",
    structure: "uptrend",
  };

  const withoutRisk = computeSignal(snapshot, { ...baseManual, eventRisk: "none" });
  const withHighRisk = computeSignal(snapshot, { ...baseManual, eventRisk: "high" });

  assert.ok(withHighRisk.probability < withoutRisk.probability);
  assert.match(withHighRisk.explanation.join(" "), /событийный риск/i);
});
