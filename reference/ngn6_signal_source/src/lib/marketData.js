const { buildFallbackSnapshot } = require("./fallbackMarket");
const { getRuntimeConfig } = require("./config");
const { fetchTBankGasSeries, validateTBankToken } = require("./tbankClient");
const { getTimeframeConfig, normalizeTimeframe } = require("./timeframes");

const YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/";
const SNAPSHOT_CACHE_MS = Number(process.env.SNAPSHOT_CACHE_MS || 60_000);
const MARKET_REQUEST_TIMEOUT_MS = Number(process.env.MARKET_REQUEST_TIMEOUT_MS || 8_000);

const snapshotCache = new Map();

function getCacheEntry(timeframeKey) {
  return (
    snapshotCache.get(timeframeKey) || {
      value: null,
      expiresAt: 0,
      promise: null,
    }
  );
}

function setCacheEntry(timeframeKey, entry) {
  snapshotCache.set(timeframeKey, entry);
}

function formatCandleDate(timestamp, interval) {
  const date = new Date(timestamp * 1000);
  if (interval === "1d") {
    return date.toISOString().slice(0, 10);
  }

  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

function toCandles(result, timeframeConfig) {
  const timestamps = result.timestamp || [];
  const quote = result.indicators?.quote?.[0] || {};
  const opens = quote.open || [];
  const highs = quote.high || [];
  const lows = quote.low || [];
  const closes = quote.close || [];
  const volumes = quote.volume || [];
  const candles = [];

  for (let i = 0; i < timestamps.length; i += 1) {
    const open = opens[i];
    const high = highs[i];
    const low = lows[i];
    const close = closes[i];

    if ([open, high, low, close].some((value) => value == null || Number.isNaN(value))) {
      continue;
    }

    candles.push({
      date: formatCandleDate(timestamps[i], timeframeConfig.yahooInterval),
      timestamp: new Date(timestamps[i] * 1000).toISOString(),
      open: Number(open.toFixed(3)),
      high: Number(high.toFixed(3)),
      low: Number(low.toFixed(3)),
      close: Number(close.toFixed(3)),
      volume: volumes[i] || 0,
    });
  }

  return candles;
}

async function fetchYahooSeries(symbol, timeframe = "daily") {
  const config = getTimeframeConfig(timeframe);
  const url = `${YAHOO_CHART_URL}${encodeURIComponent(
    symbol,
  )}?range=${encodeURIComponent(config.yahooRange)}&interval=${encodeURIComponent(
    config.yahooInterval,
  )}&includePrePost=false&events=div%2Csplits`;

  const response = await fetch(url, {
    headers: {
      "User-Agent": "ngn6-gas-bot/1.0",
      Accept: "application/json",
    },
    signal: AbortSignal.timeout(MARKET_REQUEST_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(`Yahoo request failed for ${symbol}: ${response.status}`);
  }

  const json = await response.json();
  const result = json.chart?.result?.[0];
  const error = json.chart?.error;

  if (error) {
    throw new Error(`Yahoo error for ${symbol}: ${error.description || error.code}`);
  }

  if (!result) {
    throw new Error(`No market data for ${symbol}`);
  }

  const candles = toCandles(result, config);

  if (candles.length < config.minCandles) {
    throw new Error(`Not enough ${config.yahooInterval} history for ${symbol}`);
  }

  return {
    symbol,
    shortName: result.meta?.shortName || symbol,
    candles,
    latest: candles[candles.length - 1],
  };
}

async function getMarketSnapshot(timeframe = "daily") {
  const timeframeKey = normalizeTimeframe(timeframe);
  const timeframeConfig = getTimeframeConfig(timeframeKey);
  const cacheEntry = getCacheEntry(timeframeKey);
  const now = Date.now();

  if (cacheEntry.value && cacheEntry.expiresAt > now) {
    return cacheEntry.value;
  }

  if (cacheEntry.promise) {
    return cacheEntry.promise;
  }

  const nextEntry = { ...cacheEntry };
  nextEntry.promise = (async () => {
    const config = getRuntimeConfig();
    const fallback = buildFallbackSnapshot(timeframeKey);
    const warnings = [];
    const providers = {
      gas: "demo",
      intermarket: "demo",
      tbankTokenConfigured: Boolean(config.tbank.tokenCandidates?.length),
      tbankTokenSource: config.tbank.tokenSource,
    };

    let gas = null;
    let dxy = null;
    let brent = null;

    if (config.tbank.token) {
      try {
        await validateTBankToken();
        gas = await fetchTBankGasSeries(timeframeKey);
        providers.gas = "tbank";
      } catch (error) {
        warnings.push(`T-Bank NGN6 fallback: ${error.message}`);
      }
    } else {
      warnings.push("T-Bank token is not configured, using Yahoo for natural gas.");
    }

    if (!gas) {
      try {
        gas = await fetchYahooSeries("NG=F", timeframeKey);
        providers.gas = "yahoo";
      } catch (error) {
        warnings.push(`Yahoo natural gas fallback failed: ${error.message}`);
        gas = fallback.gas;
        providers.gas = "demo";
      }
    }

    try {
      [dxy, brent] = await Promise.all([
        fetchYahooSeries("DX-Y.NYB", timeframeKey),
        fetchYahooSeries("BZ=F", timeframeKey),
      ]);
      providers.intermarket = "yahoo";
    } catch (error) {
      warnings.push(`Yahoo intermarket fallback failed: ${error.message}`);
      dxy = fallback.dxy;
      brent = fallback.brent;
      providers.intermarket = "demo";
    }

    return {
      timeframe: timeframeConfig.key,
      timeframeLabel: timeframeConfig.label,
      forecastLabel: timeframeConfig.forecastLabel,
      source: providers.gas === "demo" && providers.intermarket === "demo" ? "demo" : "live",
      generatedAt: new Date().toISOString(),
      gas,
      dxy,
      brent,
      providers,
      fallbackReason: warnings.length ? warnings.join(" | ") : null,
    };
  })();

  setCacheEntry(timeframeKey, nextEntry);

  try {
    const snapshot = await nextEntry.promise;
    setCacheEntry(timeframeKey, {
      value: snapshot,
      expiresAt: Date.now() + SNAPSHOT_CACHE_MS,
      promise: null,
    });
    return snapshot;
  } catch (error) {
    setCacheEntry(timeframeKey, {
      value: null,
      expiresAt: 0,
      promise: null,
    });
    throw error;
  }
}

module.exports = {
  fetchYahooSeries,
  getMarketSnapshot,
  normalizeTimeframe,
};
