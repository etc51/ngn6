const { getRuntimeConfig } = require("./config");
const { getTimeframeConfig } = require("./timeframes");

// This is the official protobuf package namespace, not the network endpoint domain.
const CONTRACT_PREFIX = "tinkoff.public.invest.api.contract.v1";
const TBANK_REQUEST_TIMEOUT_MS = Number(process.env.TBANK_REQUEST_TIMEOUT_MS || 8_000);

let workingToken = null;

function quotationToNumber(value) {
  if (!value) {
    return null;
  }

  if (typeof value === "number") {
    return value;
  }

  const units = Number(value.units || 0);
  const nano = Number(value.nano || 0);
  return units + nano / 1_000_000_000;
}

function instrumentIdOf(instrument) {
  return (
    instrument?.uid ||
    instrument?.instrumentUid ||
    instrument?.positionUid ||
    instrument?.figi ||
    instrument?.instrumentId ||
    null
  );
}

function scoreInstrument(instrument) {
  const text =
    `${instrument?.ticker || ""} ${instrument?.name || ""} ${instrument?.classCode || ""} ${
      instrument?.basicAsset || ""
    }`.toLowerCase();
  let score = 0;

  if (text.includes("ngn6") || text.includes("ng-7.26")) {
    score += 24;
  }

  if (text.includes("природ")) {
    score += 14;
  }

  if (text.includes("natural gas") || text.includes("henry hub")) {
    score += 12;
  }

  if (text.includes("газ (сша)") || text.includes("gas")) {
    score += 6;
  }

  if (text.includes("calendar spread") || text.includes("spread")) {
    score -= 12;
  }

  if (text.includes("mini")) {
    score -= 8;
  }

  if (text.includes("fut") || text.includes("future") || text.includes("фьючер")) {
    score += 6;
  }

  if (instrument?.apiTradeAvailableFlag) {
    score += 3;
  }

  if (text.includes("spbfut")) {
    score += 2;
  }

  return score;
}

async function describeGasInstrument(instrumentId) {
  const { tbank } = getRuntimeConfig();

  try {
    const payload = await callTBank("InstrumentsService/FutureBy", {
      idType: "INSTRUMENT_ID_TYPE_UID",
      id: instrumentId,
    });
    const instrument = payload.instrument || payload.future || payload;

    return {
      instrumentId,
      name: instrument.name || tbank.gasName,
      ticker: instrument.ticker || tbank.gasTicker,
      expirationDate: instrument.expirationDate || null,
      lastTradeDate: instrument.lastTradeDate || null,
    };
  } catch (_error) {
    return {
      instrumentId,
      name: tbank.gasName,
      ticker: tbank.gasTicker,
      expirationDate: null,
      lastTradeDate: null,
    };
  }
}

function formatTBankDate(value, timeframeKey) {
  const date = new Date(value);
  if (timeframeKey === "intraday") {
    return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
  }

  return date.toISOString().slice(0, 10);
}

async function callTBank(contractPath, body = {}) {
  const { tbank } = getRuntimeConfig();
  const tokenCandidates = [
    ...new Set([workingToken, ...(tbank.tokenCandidates || []), tbank.token].filter(Boolean)),
  ];

  if (!tokenCandidates.length) {
    throw new Error("T-Bank token is not configured.");
  }

  let lastError = null;
  let invalidTokenErrors = 0;
  let attempts = 0;

  for (const token of tokenCandidates) {
    for (const baseUrl of tbank.restBaseUrls) {
      try {
        attempts += 1;
        const response = await fetch(`${baseUrl}/${CONTRACT_PREFIX}.${contractPath}`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
            Accept: "application/json",
            "User-Agent": "ngn6-gas-bot/1.0",
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(TBANK_REQUEST_TIMEOUT_MS),
        });

        const text = await response.text();
        let payload = {};
        try {
          payload = text ? JSON.parse(text) : {};
        } catch {
          payload = { message: text };
        }

        if (!response.ok) {
          const message =
            payload?.message || payload?.description || payload?.code || `HTTP ${response.status}`;
          if (/Authentication token is missing or invalid/i.test(message)) {
            invalidTokenErrors += 1;
          }
          throw new Error(`${message} (${baseUrl})`);
        }

        workingToken = token;
        return payload;
      } catch (error) {
        lastError = error;
      }
    }
  }

  if (attempts && invalidTokenErrors === attempts) {
    throw new Error(
      `No valid T-Bank Invest token found (${tokenCandidates.length} candidate(s) checked).`,
    );
  }

  throw lastError || new Error("T-Bank API request failed.");
}

async function resolveGasInstrument() {
  const { tbank } = getRuntimeConfig();

  if (tbank.gasInstrumentId) {
    return describeGasInstrument(tbank.gasInstrumentId);
  }

  for (const query of tbank.gasSearchQueries) {
    const payload = await callTBank("InstrumentsService/FindInstrument", {
      query,
      apiTradeAvailableFlag: true,
    });

    const instruments = Array.isArray(payload?.instruments) ? payload.instruments : [];
    const ranked = instruments
      .map((instrument) => ({
        instrument,
        score: scoreInstrument(instrument),
      }))
      .filter((entry) => entry.score > 0 && instrumentIdOf(entry.instrument))
      .sort((left, right) => right.score - left.score);

    if (ranked.length) {
      const winner = ranked[0].instrument;
      return {
        instrumentId: instrumentIdOf(winner),
        name: winner.name || query,
        ticker: winner.ticker || winner.classCode || query,
        expirationDate: winner.expirationDate || null,
        lastTradeDate: winner.lastTradeDate || null,
      };
    }
  }

  throw new Error("Unable to resolve an NGN6 natural gas instrument in T-Bank.");
}

function toCandle(candle, timeframeKey) {
  return {
    date: formatTBankDate(candle.time, timeframeKey),
    timestamp: new Date(candle.time).toISOString(),
    open: Number(quotationToNumber(candle.open)?.toFixed(3)),
    high: Number(quotationToNumber(candle.high)?.toFixed(3)),
    low: Number(quotationToNumber(candle.low)?.toFixed(3)),
    close: Number(quotationToNumber(candle.close)?.toFixed(3)),
    volume: Number(candle.volume || 0),
  };
}

function toOrderBookLevel(level) {
  return {
    price: Number(quotationToNumber(level.price)?.toFixed(3)),
    quantity: Number(level.quantity || 0),
  };
}

async function fetchTBankGasSeries(timeframe = "daily") {
  const timeframeConfig = getTimeframeConfig(timeframe);
  const instrument = await resolveGasInstrument();
  const to = new Date();
  const from = new Date(to);
  from.setUTCDate(from.getUTCDate() - timeframeConfig.tbank.days);

  const payload = await callTBank("MarketDataService/GetCandles", {
    instrumentId: instrument.instrumentId,
    from: from.toISOString(),
    to: to.toISOString(),
    interval: timeframeConfig.tbank.interval,
  });

  const candles = Array.isArray(payload?.candles)
    ? payload.candles
        .map((candle) => toCandle(candle, timeframeConfig.key))
        .filter((candle) => Number.isFinite(candle.close))
    : [];

  if (candles.length < timeframeConfig.minCandles) {
    throw new Error(`T-Bank returned too little ${timeframeConfig.label} history for NGN6.`);
  }

  return {
    symbol: instrument.ticker,
    shortName: instrument.name,
    instrumentId: instrument.instrumentId,
    expirationDate: instrument.expirationDate,
    lastTradeDate: instrument.lastTradeDate,
    candles,
    latest: candles[candles.length - 1],
  };
}

async function fetchTBankGasOrderBook(depth = 20) {
  const instrument = await resolveGasInstrument();
  const payload = await callTBank("MarketDataService/GetOrderBook", {
    instrumentId: instrument.instrumentId,
    depth,
  });

  return {
    instrument,
    depth: Number(payload.depth || depth),
    bids: Array.isArray(payload.bids) ? payload.bids.map(toOrderBookLevel) : [],
    asks: Array.isArray(payload.asks) ? payload.asks.map(toOrderBookLevel) : [],
    lastPrice: quotationToNumber(payload.lastPrice),
    closePrice: quotationToNumber(payload.closePrice),
    limitUp: quotationToNumber(payload.limitUp),
    limitDown: quotationToNumber(payload.limitDown),
    ticker: payload.ticker || instrument.ticker,
    classCode: payload.classCode || null,
    instrumentUid: payload.instrumentUid || instrument.instrumentId,
    orderbookTs: payload.orderbookTs || null,
    lastPriceTs: payload.lastPriceTs || null,
    closePriceTs: payload.closePriceTs || null,
    fetchedAt: new Date().toISOString(),
  };
}

async function validateTBankToken() {
  const payload = await callTBank("UsersService/GetUserTariff", {});
  return Boolean(payload);
}

module.exports = {
  callTBank,
  fetchTBankGasOrderBook,
  fetchTBankGasSeries,
  quotationToNumber,
  resolveGasInstrument,
  validateTBankToken,
};
