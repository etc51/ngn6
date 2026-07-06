const { buildFeatureSet } = require("./signalEngine");
const { getMacroCalendar } = require("./officialCalendar");
const { getNewsPulse } = require("./newsFeed");

const RISK_ORDER = ["none", "scheduled", "unknown", "high"];

function isFiniteNumber(value) {
  return Number.isFinite(value);
}

function inferStructure(features) {
  const { close } = features;
  const { smaFast, smaSlow } = features.levels;

  if (!isFiniteNumber(close) || !isFiniteNumber(smaFast) || !isFiniteNumber(smaSlow)) {
    return "neutral";
  }

  if (close > smaFast && smaFast > smaSlow && features.breakdown.trendBias > 0.16) {
    return "uptrend";
  }

  if (close < smaFast && smaFast < smaSlow && features.breakdown.trendBias < -0.16) {
    return "downtrend";
  }

  return "neutral";
}

function inferRetest(features, structure) {
  const { close } = features;
  const { smaFast, highLocal, lowLocal } = features.levels;

  if (!isFiniteNumber(close) || close === 0) {
    return "none";
  }

  const nearSmaFast = isFiniteNumber(smaFast) ? Math.abs(close - smaFast) / close <= 0.012 : false;
  const nearHigh = isFiniteNumber(highLocal)
    ? Math.abs(close - highLocal) / close <= 0.01
    : false;
  const nearLow = isFiniteNumber(lowLocal)
    ? Math.abs(close - lowLocal) / close <= 0.01
    : false;

  if (structure === "uptrend" && (nearSmaFast || nearLow)) {
    return "bullish";
  }

  if (structure === "downtrend" && (nearSmaFast || nearHigh)) {
    return "bearish";
  }

  return "none";
}

function mergeRisk(left, right) {
  return RISK_ORDER[Math.max(RISK_ORDER.indexOf(left), RISK_ORDER.indexOf(right))] || "none";
}

function resolveNewsBias(newsPulse = {}) {
  return ["positive", "negative", "neutral"].includes(newsPulse.bias) ? newsPulse.bias : "neutral";
}

function buildContextSummary(context, calendar, newsPulse) {
  const parts = [];

  if (context.structure === "uptrend") {
    parts.push("старший тренд газа направлен вверх");
  } else if (context.structure === "downtrend") {
    parts.push("старший тренд газа направлен вниз");
  } else {
    parts.push("старший тренд газа без явного перекоса");
  }

  if (context.retest === "bullish") {
    parts.push("цена стоит у поддержки");
  } else if (context.retest === "bearish") {
    parts.push("цена стоит у сопротивления");
  }

  if (newsPulse.summary) {
    parts.push(newsPulse.summary);
  }

  if (calendar.events[0]) {
    parts.push(`ближайшее событие: ${calendar.events[0].label} ${calendar.events[0].date}`);
  }

  return parts.join(" • ");
}

function mergeContextWithOverrides(autoContext, overrides = {}) {
  const merged = { ...autoContext };

  for (const key of ["newsBias", "retest", "structure", "eventRisk"]) {
    const value = overrides[key];
    if (value && value !== "auto") {
      merged[key] = value;
    }
  }

  return merged;
}

async function getAutomaticContext(snapshot) {
  const features = buildFeatureSet(snapshot, {});
  const [calendar, newsPulse] = await Promise.all([getMacroCalendar(), getNewsPulse()]);
  const structure = inferStructure(features);
  const retest = inferRetest(features, structure);
  const newsBias = resolveNewsBias(newsPulse);
  const eventRisk = mergeRisk(calendar.eventRisk, newsPulse.eventRisk);

  const context = {
    newsBias,
    retest,
    structure,
    eventRisk,
  };

  return {
    context,
    summary: buildContextSummary(context, calendar, newsPulse),
    upcomingEvents: calendar.events,
    newsPulse,
    sources: {
      calendar: calendar.source,
      news: newsPulse.source,
      technicals: "rule-based",
    },
    warning: [calendar.warning, newsPulse.warning].filter(Boolean).join("; ") || null,
  };
}

module.exports = {
  getAutomaticContext,
  inferRetest,
  inferStructure,
  mergeContextWithOverrides,
  resolveNewsBias,
};
