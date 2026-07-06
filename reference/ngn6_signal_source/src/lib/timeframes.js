const TIMEFRAME_CONFIGS = {
  daily: {
    key: "daily",
    label: "День",
    forecastLabel: "следующий день",
    yahooRange: process.env.MARKET_RANGE || "12mo",
    yahooInterval: "1d",
    minCandles: 80,
    backtestLookbackDefault: 120,
    tbank: {
      interval: "CANDLE_INTERVAL_DAY",
      days: 280,
    },
    periods: {
      trendFast: 20,
      trendSlow: 50,
      momentumFast: 5,
      momentumSlow: 20,
      range: 20,
      local: 5,
      atr: 14,
    },
    scales: {
      trendFastPct: 4.2,
      trendSlowPct: 9.5,
      momentumFastPct: 6.5,
      momentumSlowPct: 18,
      dxyFastPct: 0.45,
      dxySlowPct: 1.4,
      brentFastPct: 1.7,
      brentSlowPct: 5,
      rangeFloorPct: 0.012,
      volatilityTargetPct: 4.8,
    },
    scoring: {
      probabilitySlope: 12.8,
      alignmentSlope: 4.2,
      probabilityCap: 69,
      minProbabilityForSignal: 53,
      scoreThreshold: 0.16,
    },
  },
  intraday: {
    key: "intraday",
    label: "Интрадей 15м",
    forecastLabel: "следующие 15 минут",
    yahooRange: process.env.MARKET_RANGE_INTRADAY || "60d",
    yahooInterval: process.env.MARKET_INTERVAL_INTRADAY || "15m",
    minCandles: 220,
    backtestLookbackDefault: 320,
    demoStepMs: 15 * 60 * 1000,
    tbank: {
      interval: "CANDLE_INTERVAL_15_MIN",
      days: 20,
    },
    periods: {
      trendFast: 20,
      trendSlow: 56,
      momentumFast: 4,
      momentumSlow: 16,
      range: 20,
      local: 6,
      atr: 14,
    },
    scales: {
      trendFastPct: 1.3,
      trendSlowPct: 3.2,
      momentumFastPct: 1.6,
      momentumSlowPct: 4.8,
      dxyFastPct: 0.2,
      dxySlowPct: 0.7,
      brentFastPct: 0.9,
      brentSlowPct: 2.6,
      rangeFloorPct: 0.0035,
      volatilityTargetPct: 1.15,
    },
    scoring: {
      probabilitySlope: 12.4,
      alignmentSlope: 3.6,
      probabilityCap: 64.5,
      minProbabilityForSignal: 52,
      scoreThreshold: 0.12,
    },
  },
};

const TIMEFRAME_ALIASES = new Map(
  [
    ["daily", "daily"],
    ["day", "daily"],
    ["1d", "daily"],
    ["d1", "daily"],
    ["intraday", "intraday"],
    ["intra", "intraday"],
    ["hour", "intraday"],
    ["1h", "intraday"],
    ["60m", "intraday"],
    ["15m", "intraday"],
    ["m15", "intraday"],
  ],
);

function normalizeTimeframe(value) {
  return TIMEFRAME_ALIASES.get(String(value || "").trim().toLowerCase()) || "daily";
}

function getTimeframeConfig(value) {
  return TIMEFRAME_CONFIGS[normalizeTimeframe(value)];
}

module.exports = {
  TIMEFRAME_CONFIGS,
  getTimeframeConfig,
  normalizeTimeframe,
};
