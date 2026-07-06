const { getTimeframeConfig } = require("./timeframes");

const DAY_MS = 24 * 60 * 60 * 1000;
const HOUR_MS = 60 * 60 * 1000;
const FIFTEEN_MIN_MS = 15 * 60 * 1000;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatCandleDate(date, timeframeKey) {
  if (timeframeKey === "intraday") {
    return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
  }

  return date.toISOString().slice(0, 10);
}

function makeSeries({
  symbol,
  shortName,
  startPrice,
  drift,
  volatility,
  points,
  stepMs,
  timeframeKey,
  phase = 0,
}) {
  const candles = [];
  let price = startPrice;
  const end = new Date();
  const start = new Date(end.getTime() - points * stepMs);

  for (let i = 0; i < points; i += 1) {
    const date = new Date(start.getTime() + i * stepMs);
    const cycle = Math.sin((i + phase) / 4.1) * volatility * 0.58;
    const wave = Math.cos((i + phase) / 6.9) * volatility * 0.42;
    const step = drift + cycle + wave;
    const nextClose = clamp(price * (1 + step / 100), startPrice * 0.35, startPrice * 2.4);
    const high = Math.max(price, nextClose) * (1 + volatility / 180);
    const low = Math.min(price, nextClose) * (1 - volatility / 180);

    candles.push({
      date: formatCandleDate(date, timeframeKey),
      timestamp: date.toISOString(),
      open: Number(price.toFixed(3)),
      high: Number(high.toFixed(3)),
      low: Number(low.toFixed(3)),
      close: Number(nextClose.toFixed(3)),
      volume: 1000 + i * 11,
    });

    price = nextClose;
  }

  return {
    symbol,
    shortName,
    candles,
    latest: candles[candles.length - 1],
  };
}

function buildFallbackSnapshot(timeframe = "daily") {
  const config = getTimeframeConfig(timeframe);
  const isIntraday = config.key === "intraday";
  const points = isIntraday ? 900 : 220;
  const stepMs = isIntraday ? config.demoStepMs || FIFTEEN_MIN_MS || HOUR_MS : DAY_MS;

  const gas = makeSeries({
    symbol: "NG=F",
    shortName: isIntraday ? "Henry Hub Natural Gas (demo intraday)" : "Henry Hub Natural Gas (demo)",
    startPrice: 3.45,
    drift: isIntraday ? 0.02 : 0.11,
    volatility: isIntraday ? 0.95 : 2.5,
    points,
    stepMs,
    timeframeKey: config.key,
    phase: 3,
  });

  const dxy = makeSeries({
    symbol: "DX-Y.NYB",
    shortName: isIntraday ? "US Dollar Index (demo intraday)" : "US Dollar Index (demo)",
    startPrice: 104.2,
    drift: isIntraday ? -0.003 : -0.018,
    volatility: isIntraday ? 0.08 : 0.22,
    points,
    stepMs,
    timeframeKey: config.key,
    phase: 11,
  });

  const brent = makeSeries({
    symbol: "BZ=F",
    shortName: isIntraday ? "Brent Crude Oil (demo intraday)" : "Brent Crude Oil (demo)",
    startPrice: 78.9,
    drift: isIntraday ? 0.011 : 0.074,
    volatility: isIntraday ? 0.39 : 1.1,
    points,
    stepMs,
    timeframeKey: config.key,
    phase: 17,
  });

  return {
    timeframe: config.key,
    timeframeLabel: config.label,
    forecastLabel: config.forecastLabel,
    source: "demo",
    generatedAt: new Date().toISOString(),
    gas,
    dxy,
    brent,
  };
}

module.exports = {
  buildFallbackSnapshot,
};
