const fs = require("fs");
const path = require("path");
const dns = require("dns");
const { getAutomaticContext } = require("./lib/autoContext");
const { toConsoleText } = require("./lib/consoleText");
const { defaultHeartbeatFile, runWithTimeout, writeHeartbeat } = require("./lib/heartbeat");
const { getMarketSnapshot } = require("./lib/marketData");
const { computeSignal } = require("./lib/signalEngine");

dns.setDefaultResultOrder?.("ipv4first");

const DEFAULT_INTERVAL_MS = 60_000;
const DEFAULT_REPEAT_MS = 30 * 60_000;
const MOSCOW_OFFSET_MS = 3 * 60 * 60 * 1000;
const DEFAULT_STATE_FILE = path.join(process.cwd(), "data", "signal-watcher-state.json");
const DEFAULT_HEARTBEAT_FILE = defaultHeartbeatFile("signal-watcher");

function parseCsvList(value, fallback) {
  if (!value) {
    return fallback;
  }

  const parsed = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  return parsed.length ? parsed : fallback;
}

function formatTimestamp(value) {
  if (!value) {
    return "-";
  }

  return new Date(value).toLocaleString("ru-RU", {
    dateStyle: "short",
    timeStyle: "medium",
  });
}

function formatItems(items = []) {
  return items
    .filter(Boolean)
    .map((item) => `${item.label}: ${item.levelText || item.value || "-"}`)
    .join("\n");
}

function getInstrumentSymbol(payload) {
  return process.env.TBANK_GAS_TICKER || "NGN6";
}

function getFactorValue(payload, id) {
  const factor = payload?.factors?.find((item) => item.id === id);
  return Number(factor?.value || 0);
}

function buildPlanPermissions(side) {
  if (side === "long") {
    return {
      longAllowed: true,
      shortAllowed: false,
    };
  }

  if (side === "short") {
    return {
      longAllowed: false,
      shortAllowed: true,
    };
  }

  return {
    longAllowed: false,
    shortAllowed: false,
  };
}

function roundPrice(value) {
  if (!Number.isFinite(value)) {
    return null;
  }

  return Number(value.toFixed(2));
}

function formatPrice(value) {
  const rounded = roundPrice(Number(value));
  return rounded == null ? "-" : rounded.toFixed(2);
}

function getMoscowClock(date = new Date()) {
  const shifted = new Date(date.getTime() + MOSCOW_OFFSET_MS);
  return {
    date: shifted.toISOString().slice(0, 10),
    weekday: shifted.getUTCDay(),
    minutes: shifted.getUTCHours() * 60 + shifted.getUTCMinutes(),
    label: `${String(shifted.getUTCHours()).padStart(2, "0")}:${String(
      shifted.getUTCMinutes(),
    ).padStart(2, "0")} МСК`,
  };
}

function getMoscowParts(date = new Date()) {
  const shifted = new Date(date.getTime() + MOSCOW_OFFSET_MS);
  return {
    date: shifted.toISOString().slice(0, 10),
    minutes: shifted.getUTCHours() * 60 + shifted.getUTCMinutes(),
  };
}

function parseScheduleMinutes(value) {
  const match = String(value || "10:20").match(/^(\d{1,2}):(\d{2})$/);

  if (!match) {
    return 10 * 60 + 20;
  }

  const hours = Math.min(Math.max(Number(match[1]), 0), 23);
  const minutes = Math.min(Math.max(Number(match[2]), 0), 59);
  return hours * 60 + minutes;
}

function parseScheduleEntries(value, fallback) {
  return parseCsvList(value, fallback).map((label) => ({
    label,
    minutes: parseScheduleMinutes(label),
  }));
}

function buildSignalFingerprint(payload) {
  const entries = payload.tradePlan?.entries || [];
  const exits = payload.tradePlan?.exits || [];
  const news = payload.newsPulse || {};

  return [
    payload.timeframe,
    payload.signal,
    payload.headline,
    payload.probability,
    payload.close,
    news.bias,
    entries.map((item) => item.levelText).join("|"),
    exits.map((item) => item.levelText).join("|"),
  ].join("::");
}

function buildSignalMessage(payload) {
  const entries = formatItems(payload.tradePlan?.entries);
  const exits = formatItems(payload.tradePlan?.exits);
  const newsPulse = payload.newsPulse || {};
  const market = payload.market?.gas || {};
  const symbol = getInstrumentSymbol(payload);

  return [
    `${symbol} ${payload.timeframeLabel} | ${payload.headline} ${payload.probability}%`,
    `Цена: ${payload.close} | свеча: ${payload.date}`,
    `Источник: ${market.symbol || symbol} / ${payload.providers?.gas || "unknown"}`,
    "",
    "Входы:",
    entries || "-",
    "",
    "Выходы:",
    exits || "-",
    "",
    `Новости: ${newsPulse.bias || "neutral"} | ${newsPulse.summary || "нет свежего резюме"}`,
    `Обновлено: ${formatTimestamp(payload.generatedAt)}`,
  ].join("\n");
}

function getExecutionPlanConfig() {
  return {
    stopBufferAtr: Number(process.env.SIGNAL_EXECUTION_STOP_BUFFER_ATR || 0.1),
    stopBufferPct: Number(process.env.SIGNAL_EXECUTION_STOP_BUFFER_PCT || 0.0015),
    minRiskAtr: Number(process.env.SIGNAL_EXECUTION_MIN_RISK_ATR || 0.25),
    minRiskPct: Number(process.env.SIGNAL_EXECUTION_MIN_RISK_PCT || 0.004),
    maxRiskAtr: Number(process.env.SIGNAL_EXECUTION_MAX_RISK_ATR || 0.7),
    maxRiskPct: Number(process.env.SIGNAL_EXECUTION_MAX_RISK_PCT || 0.018),
    takeProfit1R: Number(process.env.SIGNAL_EXECUTION_TP1_R || 0.9),
    takeProfit2R: Number(process.env.SIGNAL_EXECUTION_TP2_R || 1.55),
  };
}

function getImpulseConfig() {
  return {
    enabled: process.env.SIGNAL_IMPULSE_ENABLED !== "false",
    sessionStartMinutes: parseScheduleMinutes(process.env.SIGNAL_IMPULSE_SESSION_START || "09:00"),
    movePct: Number(process.env.SIGNAL_IMPULSE_MOVE_PCT || 1.2),
    breakoutBufferPct: Number(process.env.SIGNAL_IMPULSE_BREAKOUT_BUFFER_PCT || 0.03),
    minTrend: Number(process.env.SIGNAL_IMPULSE_MIN_TREND || 0.28),
    minMomentum: Number(process.env.SIGNAL_IMPULSE_MIN_MOMENTUM || 0.24),
    minCandles: Number(process.env.SIGNAL_IMPULSE_MIN_CANDLES || 2),
    minProbability: Number(process.env.SIGNAL_IMPULSE_MIN_PROBABILITY || 55.5),
    maxProbability: Number(process.env.SIGNAL_IMPULSE_MAX_PROBABILITY || 63.5),
  };
}

function getNumber(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) {
      return number;
    }
  }

  return null;
}

function buildExecutionLevels(payload, side, entry, fallbackStop, fallbackTakeProfit1, fallbackTakeProfit2) {
  const config = getExecutionPlanConfig();
  const marketLevels = payload.marketLevels || {};
  const close = getNumber(payload.close, entry);
  const atr = getNumber(marketLevels.atr, close ? close * 0.02 : null) || 1;
  const buffer = Math.max(atr * config.stopBufferAtr, entry * config.stopBufferPct);
  const minRisk = Math.max(atr * config.minRiskAtr, entry * config.minRiskPct);
  const maxRisk = Math.max(atr * config.maxRiskAtr, entry * config.maxRiskPct);

  if (side === "long") {
    const nearestInvalidation = getNumber(marketLevels.currentLow, payload.tradeLevels?.localSupport, close);
    const fallbackCapped = Number.isFinite(fallbackStop)
      ? Math.max(fallbackStop, entry - maxRisk)
      : entry - maxRisk;
    const rawStop = Math.max(fallbackCapped, nearestInvalidation - buffer);
    const stop = clampExecutionStop(rawStop, entry - maxRisk, entry - minRisk);
    const risk = entry - stop;
    const takeProfit1 = Math.max(
      Number.isFinite(fallbackTakeProfit1) ? fallbackTakeProfit1 : entry + risk,
      entry + risk * config.takeProfit1R,
    );
    const takeProfit2 = Math.max(
      Number.isFinite(fallbackTakeProfit2) ? fallbackTakeProfit2 : entry + risk * 1.5,
      entry + risk * config.takeProfit2R,
    );

    return {
      stop: roundPrice(stop),
      takeProfit1: roundPrice(takeProfit1),
      takeProfit2: roundPrice(takeProfit2),
    };
  }

  const nearestInvalidation = getNumber(marketLevels.currentHigh, payload.tradeLevels?.localResistance, close);
  const fallbackCapped = Number.isFinite(fallbackStop)
    ? Math.min(fallbackStop, entry + maxRisk)
    : entry + maxRisk;
  const rawStop = Math.min(fallbackCapped, nearestInvalidation + buffer);
  const stop = clampExecutionStop(rawStop, entry + minRisk, entry + maxRisk);
  const risk = stop - entry;
  const takeProfit1 = Math.min(
    Number.isFinite(fallbackTakeProfit1) ? fallbackTakeProfit1 : entry - risk,
    entry - risk * config.takeProfit1R,
  );
  const takeProfit2 = Math.min(
    Number.isFinite(fallbackTakeProfit2) ? fallbackTakeProfit2 : entry - risk * 1.5,
    entry - risk * config.takeProfit2R,
  );

  return {
    stop: roundPrice(stop),
    takeProfit1: roundPrice(takeProfit1),
    takeProfit2: roundPrice(takeProfit2),
  };
}

function clampExecutionStop(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function buildExecutablePlan(payload) {
  if (payload?.impulsePlan) {
    return payload.impulsePlan;
  }

  const levels = payload.tradeLevels || {};
  const permissions = payload.tradePlan?.permissions || {};

  if (payload.signal === "north" && levels.breakoutLevel && levels.stopLevel) {
    const entry = roundPrice(levels.breakoutLevel);
    const execution = buildExecutionLevels(
      payload,
      "long",
      entry,
      levels.stopLevel,
      levels.takeProfit1,
      levels.takeProfit2,
    );

    return {
      side: "long",
      sideLabel: "ЛОНГ",
      actionLabel: "Вход в лонг",
      entryCondition: "если цена пробивает уровень вверх",
      entry,
      stop: execution.stop,
      takeProfit1: execution.takeProfit1,
      takeProfit2: execution.takeProfit2,
      scenarioStop: roundPrice(levels.stopLevel),
      allowed: permissions.longAllowed !== false,
    };
  }

  if (payload.signal === "south" && levels.breakdownLevel && levels.stopLevel) {
    const entry = roundPrice(levels.breakdownLevel);
    const execution = buildExecutionLevels(
      payload,
      "short",
      entry,
      levels.stopLevel,
      levels.takeProfit1,
      levels.takeProfit2,
    );

    return {
      side: "short",
      sideLabel: "ШОРТ",
      actionLabel: "Вход в шорт",
      entryCondition: "если цена пробивает уровень вниз",
      entry,
      stop: execution.stop,
      takeProfit1: execution.takeProfit1,
      takeProfit2: execution.takeProfit2,
      scenarioStop: roundPrice(levels.stopLevel),
      allowed: permissions.shortAllowed !== false,
    };
  }

  return null;
}

function buildSessionMetrics(snapshot, sessionStartMinutes = getImpulseConfig().sessionStartMinutes) {
  const candles = snapshot?.gas?.candles || [];
  if (!candles.length) {
    return null;
  }

  const latest = candles[candles.length - 1];
  const latestParts = getMoscowParts(new Date(latest.timestamp));
  const sessionCandles = candles.filter((candle) => {
    const parts = getMoscowParts(new Date(candle.timestamp));
    return parts.date === latestParts.date && parts.minutes >= sessionStartMinutes;
  });

  if (!sessionCandles.length) {
    return null;
  }

  const latestCandle = sessionCandles[sessionCandles.length - 1];
  const priorCandles = sessionCandles.slice(0, -1);
  const sessionOpen = Number(sessionCandles[0].open);
  const sessionClose = Number(latestCandle.close);
  const sessionHigh = Math.max(...sessionCandles.map((candle) => Number(candle.high)));
  const sessionLow = Math.min(...sessionCandles.map((candle) => Number(candle.low)));
  const priorSessionHigh = priorCandles.length
    ? Math.max(...priorCandles.map((candle) => Number(candle.high)))
    : null;
  const priorSessionLow = priorCandles.length
    ? Math.min(...priorCandles.map((candle) => Number(candle.low)))
    : null;

  return {
    date: latestParts.date,
    candlesSinceOpen: sessionCandles.length,
    sessionOpen: roundPrice(sessionOpen),
    sessionClose: roundPrice(sessionClose),
    sessionHigh: roundPrice(sessionHigh),
    sessionLow: roundPrice(sessionLow),
    priorSessionHigh: roundPrice(priorSessionHigh),
    priorSessionLow: roundPrice(priorSessionLow),
    movePct: sessionOpen ? Number((((sessionClose - sessionOpen) / sessionOpen) * 100).toFixed(2)) : 0,
  };
}

function buildImpulsePlan(payload, side) {
  const session = payload.sessionMetrics || {};
  const marketLevels = payload.marketLevels || {};
  const entry =
    side === "long"
      ? roundPrice(
          Math.max(
            Number(payload.close || 0),
            Number(session.priorSessionHigh || marketLevels.currentHigh || payload.close || 0),
          ),
        )
      : roundPrice(
          Math.min(
            Number(payload.close || 0),
            Number(session.priorSessionLow || marketLevels.currentLow || payload.close || 0),
          ),
        );

  if (!Number.isFinite(entry)) {
    return null;
  }

  const fallbackStop =
    side === "long"
      ? Number(session.sessionLow || marketLevels.currentLow || entry)
      : Number(session.sessionHigh || marketLevels.currentHigh || entry);
  const execution = buildExecutionLevels(payload, side, entry, fallbackStop);

  if (side === "long") {
    return {
      side: "long",
      sideLabel: "ЛОНГ",
      actionLabel: "Вход в лонг",
      entryCondition: "импульс выше утреннего максимума",
      entry,
      stop: execution.stop,
      takeProfit1: execution.takeProfit1,
      takeProfit2: execution.takeProfit2,
      scenarioStop: roundPrice(fallbackStop),
      allowed: true,
    };
  }

  return {
    side: "short",
    sideLabel: "ШОРТ",
    actionLabel: "Вход в шорт",
    entryCondition: "импульс ниже утреннего минимума",
    entry,
    stop: execution.stop,
    takeProfit1: execution.takeProfit1,
    takeProfit2: execution.takeProfit2,
    scenarioStop: roundPrice(fallbackStop),
    allowed: true,
  };
}

function applyImpulseOverlay(payload, options = getImpulseConfig()) {
  if (!options.enabled || payload.timeframe !== "intraday") {
    return payload;
  }

  const session = payload.sessionMetrics;
  if (!session || session.candlesSinceOpen < options.minCandles) {
    return payload;
  }

  const trend = getFactorValue(payload, "trend");
  const momentum = getFactorValue(payload, "momentum");
  const priorHigh = Number(session.priorSessionHigh);
  const priorLow = Number(session.priorSessionLow);
  const close = Number(payload.close);
  const bufferUp = priorHigh * (options.breakoutBufferPct / 100);
  const bufferDown = priorLow * (options.breakoutBufferPct / 100);
  const breakoutUp = Number.isFinite(priorHigh) && close >= priorHigh + bufferUp;
  const breakoutDown = Number.isFinite(priorLow) && close <= priorLow - bufferDown;
  const newsBias = payload.newsPulse?.bias || "neutral";

  let side = null;
  if (
    session.movePct >= options.movePct &&
    breakoutUp &&
    trend >= options.minTrend &&
    momentum >= options.minMomentum &&
    newsBias !== "negative"
  ) {
    side = "long";
  } else if (
    session.movePct <= -options.movePct &&
    breakoutDown &&
    trend <= -options.minTrend &&
    momentum <= -options.minMomentum &&
    newsBias !== "positive"
  ) {
    side = "short";
  }

  if (!side) {
    return payload;
  }

  const impulsePlan = buildImpulsePlan(payload, side);
  if (!impulsePlan) {
    return payload;
  }

  const signal = side === "long" ? "north" : "south";
  const headline = side === "long" ? "Покупка" : "Продажа";
  const sideLabel = side === "long" ? "лонг" : "шорт";
  const boostedProbability = Math.min(
    options.maxProbability,
    Math.max(
      options.minProbability,
      Number(payload.probability || 50) +
        Math.abs(session.movePct) * 1.35 +
        Math.abs(trend) * 3.2 +
        Math.abs(momentum) * 2.6,
    ),
  );

  return {
    ...payload,
    signal,
    headline,
    probability: Number(boostedProbability.toFixed(1)),
    signalOrigin: "impulse-breakout",
    hybridMode: "impulse-breakout",
    impulsePlan,
    explanation: [
      ...(payload.explanation || []),
      `утренний импульс подтверждает ${sideLabel}: ${session.movePct}% от открытия с пробоем экстремума`,
    ],
    tradePlan: {
      ...(payload.tradePlan || {}),
      permissions: buildPlanPermissions(side),
      summary: `Импульсный ${sideLabel} по NGN6: утреннее движение ${session.movePct}% уже пробило стартовый диапазон.`,
    },
  };
}

function selectMorningPayload(dailyPayload, intradayPayload, options) {
  if (!intradayPayload) {
    return dailyPayload;
  }

  if (intradayPayload.signalOrigin === "impulse-breakout") {
    return {
      ...intradayPayload,
      hybridMode: "hybrid-impulse",
    };
  }

  if (
    dailyPayload.signal === "neutral" &&
    intradayPayload.signal !== "neutral" &&
    Number(intradayPayload.probability || 0) >= options.hybridMorningMinProbability
  ) {
    return {
      ...intradayPayload,
      hybridMode: "hybrid-intraday",
    };
  }

  return dailyPayload;
}

function isEntryReached(plan, price) {
  if (!plan || !Number.isFinite(Number(price))) {
    return false;
  }

  return plan.side === "long" ? Number(price) >= plan.entry : Number(price) <= plan.entry;
}

function getEntryWarningDistancePct() {
  return Number(process.env.SIGNAL_ENTRY_WARNING_DISTANCE_PCT || 0.3);
}

function getTakeProfitManagementConfig() {
  return {
    enabled: process.env.SIGNAL_TP_MANAGEMENT_ENABLED !== "false",
    stopLockR: Number(process.env.SIGNAL_TP1_STOP_LOCK_R || 0.25),
    stopBufferAtr: Number(process.env.SIGNAL_TP1_STOP_BUFFER_ATR || 0.1),
    stopBufferPct: Number(process.env.SIGNAL_TP1_STOP_BUFFER_PCT || 0.0015),
  };
}

function getEntryDistancePct(plan, price) {
  if (!plan || !Number.isFinite(Number(price)) || !Number.isFinite(Number(plan.entry))) {
    return null;
  }

  if (plan.entry === 0) {
    return null;
  }

  const distance = plan.side === "long" ? plan.entry - Number(price) : Number(price) - plan.entry;
  return (distance / Math.abs(plan.entry)) * 100;
}

function shouldWarnBeforeEntry(plan, price, thresholdPct = getEntryWarningDistancePct()) {
  const distancePct = getEntryDistancePct(plan, price);

  return distancePct != null && distancePct > 0 && distancePct <= thresholdPct;
}

function getCurrentExtreme(payload, side, fallbackPrice) {
  const levels = payload.marketLevels || {};
  const value = side === "long" ? levels.currentHigh : levels.currentLow;
  const number = Number(value);
  return Number.isFinite(number) ? number : Number(fallbackPrice);
}

function isTakeProfitReached(plan, payload, target = "takeProfit1") {
  if (!plan || !Number.isFinite(Number(plan[target]))) {
    return false;
  }

  const price = Number(payload.close);
  const extreme = getCurrentExtreme(payload, plan.side, price);
  const trigger = Number(plan[target]);

  return plan.side === "long"
    ? Math.max(price, extreme) >= trigger
    : Math.min(price, extreme) <= trigger;
}

function adjustStopAfterTakeProfit1(plan, payload, config = getTakeProfitManagementConfig()) {
  if (!plan || !config.enabled) {
    return null;
  }

  const entry = Number(plan.entry);
  const stop = Number(plan.stop);
  const price = Number(payload.close);
  const atr = Number(payload.marketLevels?.atr || 0);

  if (![entry, stop, price].every(Number.isFinite)) {
    return null;
  }

  const buffer = Math.max(
    Number.isFinite(atr) && atr > 0 ? atr * config.stopBufferAtr : 0,
    Math.abs(price) * config.stopBufferPct,
  );

  if (plan.side === "long") {
    const risk = Math.max(entry - stop, 0);
    const lockStop = entry + risk * config.stopLockR;
    const maxPracticalStop = price - buffer;
    return roundPrice(Math.max(stop, Math.min(lockStop, maxPracticalStop)));
  }

  const risk = Math.max(stop - entry, 0);
  const lockStop = entry - risk * config.stopLockR;
  const minPracticalStop = price + buffer;
  return roundPrice(Math.min(stop, Math.max(lockStop, minPracticalStop)));
}

function buildDailyPlanMessage(payload, plan, clock = getMoscowClock()) {
  const newsPulse = payload.newsPulse || {};
  const permissions = payload.tradePlan?.permissions || {};
  const symbol = getInstrumentSymbol(payload);
  const permissionText = `Лонг: ${permissions.longAllowed ? "да" : "нет"} | Шорт: ${
    permissions.shortAllowed ? "да" : "нет"
  }`;
  const modeText = payload.hybridMode
    ? `Режим: ${
        payload.hybridMode === "hybrid-impulse"
          ? "гибрид daily + impulse breakout"
          : payload.hybridMode === "hybrid-intraday"
            ? "гибрид daily + intraday"
            : "intraday impulse"
      }`
    : null;

  if (!plan) {
    return [
      `${symbol} дневной план | ${clock.date} ${clock.label}`,
      "Сделки сегодня нет: чистого преимущества нет.",
      `Цена сейчас: ${formatPrice(payload.close)} | свеча: ${payload.date}`,
      permissionText,
      `Новости: ${newsPulse.bias || "neutral"} | ${newsPulse.summary || "нет свежего резюме"}`,
      "Следующее уведомление будет только при экстренном событии.",
    ].join("\n");
  }

  const entryState = isEntryReached(plan, payload.close)
    ? "Вход уже достигнут по текущей цене."
    : "Ждём касание входа; повторов не будет до достижения точки или экстренного события.";

  return [
    `${symbol} дневной план | ${clock.date} ${clock.label}`,
    `Приоритет: ${plan.sideLabel} | вероятность ${payload.probability}%`,
    `Цена сейчас: ${formatPrice(payload.close)} | свеча: ${payload.date}`,
    permissionText,
    ...(modeText ? [modeText] : []),
    "",
    `${plan.actionLabel}: ${formatPrice(plan.entry)}`,
    `Условие: ${plan.entryCondition}`,
    `Стоп: ${formatPrice(plan.stop)}`,
    `Тейк 1: ${formatPrice(plan.takeProfit1)}`,
    `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
    "",
    entryState,
    `Новости: ${newsPulse.bias || "neutral"} | ${newsPulse.summary || "нет свежего резюме"}`,
  ].join("\n");
}

function buildEntryWarningMessage(payload, plan, clock = getMoscowClock()) {
  const symbol = getInstrumentSymbol(payload);
  const distancePct = getEntryDistancePct(plan, payload.close);

  return [
    `${symbol} скоро возможен вход | ${clock.date} ${clock.label}`,
    `${plan.actionLabel}: ${formatPrice(plan.entry)}`,
    `Условие: ${plan.entryCondition}`,
    `Цена сейчас: ${formatPrice(payload.close)}`,
    `До входа: ${distancePct == null ? "-" : distancePct.toFixed(2)}%`,
    `Стоп: ${formatPrice(plan.stop)}`,
    `Тейк 1: ${formatPrice(plan.takeProfit1)}`,
    `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
  ].join("\n");
}

function buildEntryReachedMessage(payload, plan, clock = getMoscowClock()) {
  const symbol = getInstrumentSymbol(payload);

  return [
    `${symbol} цена достигла входа | ${clock.date} ${clock.label}`,
    `${plan.actionLabel}: ${formatPrice(plan.entry)}`,
    `Условие: ${plan.entryCondition}`,
    `Цена сейчас: ${formatPrice(payload.close)}`,
    `Стоп: ${formatPrice(plan.stop)}`,
    `Тейк 1: ${formatPrice(plan.takeProfit1)}`,
    `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
    "Входи в позицию только если уровень подтверждается по твоим правилам исполнения.",
  ].join("\n");
}

function buildTakeProfit1ReachedMessage(payload, plan, adjustedStop, clock = getMoscowClock()) {
  const symbol = getInstrumentSymbol(payload);

  return [
    `${symbol} тейк 1 достигнут | ${clock.date} ${clock.label}`,
    `${plan.sideLabel}: цель 1 ${formatPrice(plan.takeProfit1)} взята`,
    `Цена сейчас: ${formatPrice(payload.close)}`,
    `Старый стоп: ${formatPrice(plan.stop)}`,
    `Новый стоп: ${formatPrice(adjustedStop)}`,
    `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
    "Действие: зафиксировать часть позиции и подтянуть стоп по остатку.",
  ].join("\n");
}

async function buildSignalPayload(timeframe) {
  const snapshot = await getMarketSnapshot(timeframe);
  const autoContext = await getAutomaticContext(snapshot);
  const signal = computeSignal(snapshot, autoContext.context);
  const sessionMetrics = timeframe === "intraday" ? buildSessionMetrics(snapshot) : null;

  return applyImpulseOverlay({
    ...signal,
    generatedAt: snapshot.generatedAt,
    providers: snapshot.providers,
    automaticContext: autoContext.context,
    newsPulse: autoContext.newsPulse,
    sessionMetrics,
    market: {
      gas: {
        symbol: snapshot.gas.symbol,
        name: snapshot.gas.shortName,
        instrumentId: snapshot.gas.instrumentId,
        latest: snapshot.gas.latest,
        expirationDate: snapshot.gas.expirationDate,
        lastTradeDate: snapshot.gas.lastTradeDate,
      },
    },
  });
}

async function sendTelegram(message) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;

  if (!token || !chatId) {
    return false;
  }

  const response = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: chatId,
      text: message,
      disable_web_page_preview: true,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Telegram send failed: ${response.status} ${text}`);
  }

  return true;
}

async function sendMax(message) {
  const token = process.env.MAX_BOT_TOKEN;
  const userId = process.env.MAX_USER_ID;
  const chatId = process.env.MAX_CHAT_ID;
  const baseUrl = process.env.MAX_API_BASE_URL || "https://platform-api2.max.ru";

  if (!token || (!userId && !chatId)) {
    return false;
  }

  const query = new URLSearchParams(userId ? { user_id: userId } : { chat_id: chatId });
  const response = await fetch(`${baseUrl}/messages?${query}`, {
    method: "POST",
    headers: {
      Authorization: token,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: message,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`MAX send failed: ${response.status} ${text}`);
  }

  return true;
}

function buildNtfyUrl() {
  const explicitUrl = process.env.NTFY_URL;
  const topic = process.env.NTFY_TOPIC;

  if (explicitUrl) {
    return explicitUrl;
  }

  if (!topic) {
    return "";
  }

  const server = process.env.NTFY_SERVER || "https://ntfy.sh";
  return `${server.replace(/\/+$/, "")}/${encodeURIComponent(topic)}`;
}

function addNtfyAuth(headers) {
  const token = process.env.NTFY_AUTH_TOKEN;
  const username = process.env.NTFY_USERNAME;
  const password = process.env.NTFY_PASSWORD;

  if (token) {
    headers.Authorization = `Bearer ${token}`;
    return;
  }

  if (username && password) {
    headers.Authorization = `Basic ${Buffer.from(`${username}:${password}`).toString("base64")}`;
  }
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function formatErrorDetails(error) {
  const parts = [];

  if (error?.message) {
    parts.push(error.message);
  }

  if (error?.cause?.code) {
    parts.push(`code=${error.cause.code}`);
  }

  if (error?.cause?.errno) {
    parts.push(`errno=${error.cause.errno}`);
  }

  if (error?.cause?.address) {
    parts.push(`address=${error.cause.address}`);
  }

  if (error?.cause?.port) {
    parts.push(`port=${error.cause.port}`);
  }

  return parts.join(" | ") || String(error);
}

async function fetchWithRetry(url, options, attempts = 3, timeoutMs = 15_000) {
  let lastError = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(`timeout after ${timeoutMs}ms`), timeoutMs);

    try {
      return await fetch(url, {
        ...options,
        signal: controller.signal,
      });
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await delay(500 * attempt);
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }

  throw lastError;
}

async function sendNtfy(message) {
  const url = buildNtfyUrl();

  if (!url) {
    return false;
  }

  const headers = {
    "Content-Type": "text/plain; charset=utf-8",
    Title: process.env.NTFY_TITLE || `${process.env.TBANK_GAS_TICKER || "NGN6"} gas signal`,
    Priority: process.env.NTFY_PRIORITY || "high",
    Tags: process.env.NTFY_TAGS || "chart_with_upwards_trend,gas",
  };

  if (process.env.NTFY_CLICK_URL) {
    headers.Click = process.env.NTFY_CLICK_URL;
  }

  addNtfyAuth(headers);

  const response = await fetchWithRetry(
    url,
    {
      method: "POST",
      headers,
      body: message,
    },
    Number(process.env.NTFY_RETRY_ATTEMPTS || 5),
    Number(process.env.NTFY_TIMEOUT_MS || 15_000),
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`ntfy send failed: ${response.status} ${text}`);
  }

  return true;
}

async function sendWebhook(payload, message) {
  const url = process.env.SIGNAL_WEBHOOK_URL;

  if (!url) {
    return false;
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      signal: payload,
    }),
  });

  if (!response.ok) {
    throw new Error(`Webhook send failed: ${response.status}`);
  }

  return true;
}

async function trySend(channel, send) {
  try {
    return await send();
  } catch (error) {
    console.error(`[${new Date().toISOString()}] ${channel}: ${formatErrorDetails(error)}`);
    return false;
  }
}

async function sendMessageToChannels(message, payload = null) {
  const sentTelegram = await trySend("telegram", () => sendTelegram(message));
  const sentMax = await trySend("max", () => sendMax(message));
  const sentNtfy = await trySend("ntfy", () => sendNtfy(message));
  const sentWebhook = await trySend("webhook", () => sendWebhook(payload, message));

  if (!sentTelegram && !sentMax && !sentNtfy && !sentWebhook) {
    console.log(toConsoleText(message));
    console.log("");
  }

  return sentTelegram || sentMax || sentNtfy || sentWebhook;
}

async function notify(payload) {
  return sendMessageToChannels(buildSignalMessage(payload), payload);
}

function shouldNotify({ mode, previous, current, lastSentAt, repeatMs, now }) {
  if (mode === "always") {
    return true;
  }

  if (!previous || previous !== current) {
    return true;
  }

  return repeatMs > 0 && now - lastSentAt >= repeatMs;
}

function loadDailyPlanState(filePath) {
  try {
    const state = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return {
      lastPlanDate: null,
      activePlan: null,
      entryWarningDate: null,
      entryAlertDate: null,
      takeProfit1Date: null,
      emergencyFingerprints: [],
      intradayRechecks: {},
      tradeSignalCountDate: null,
      tradeSignalCount: 0,
      ...state,
    };
  } catch (_error) {
    return {
      lastPlanDate: null,
      activePlan: null,
      entryWarningDate: null,
      entryAlertDate: null,
      takeProfit1Date: null,
      emergencyFingerprints: [],
      intradayRechecks: {},
      tradeSignalCountDate: null,
      tradeSignalCount: 0,
    };
  }
}

function saveDailyPlanState(filePath, state) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function shouldRunDailyPlan({ clock, state, scheduleMinutes, windowMinutes, weekdaysOnly = true }) {
  if (weekdaysOnly && (clock.weekday === 0 || clock.weekday === 6)) {
    return false;
  }

  if (state.lastPlanDate === clock.date) {
    return false;
  }

  return clock.minutes >= scheduleMinutes && clock.minutes < scheduleMinutes + windowMinutes;
}

function getDueIntradayRecheck({ clock, state, scheduleEntries, windowMinutes, weekdaysOnly = true }) {
  if (weekdaysOnly && (clock.weekday === 0 || clock.weekday === 6)) {
    return null;
  }

  const sentForDate = state.intradayRechecks?.[clock.date] || {};
  return scheduleEntries.find(
    (entry) =>
      !sentForDate[entry.label] &&
      clock.minutes >= entry.minutes &&
      clock.minutes < entry.minutes + windowMinutes,
  ) || null;
}

function markIntradayRecheck(state, clock, scheduleEntry, result) {
  state.intradayRechecks = state.intradayRechecks || {};
  state.intradayRechecks[clock.date] = {
    ...(state.intradayRechecks[clock.date] || {}),
    [scheduleEntry.label]: {
      checkedAt: new Date().toISOString(),
      ...result,
    },
  };

  for (const date of Object.keys(state.intradayRechecks)) {
    if (date !== clock.date) {
      delete state.intradayRechecks[date];
    }
  }
}

function getTradeSignalCount(state, clock) {
  return state.tradeSignalCountDate === clock.date ? Number(state.tradeSignalCount || 0) : 0;
}

function incrementTradeSignalCount(state, clock) {
  state.tradeSignalCountDate = clock.date;
  state.tradeSignalCount = getTradeSignalCount(state, clock) + 1;
}

function normalizeEmergencyFingerprints(state, clock) {
  state.emergencyFingerprints = (state.emergencyFingerprints || []).filter((fingerprint) =>
    fingerprint.startsWith(`${clock.date}:`),
  );
}

function isNewsAgainstActivePlan(activeSignal, newsPulse = {}) {
  if (!activeSignal || activeSignal === "neutral") {
    return false;
  }

  if (activeSignal === "north") {
    return newsPulse.bias === "negative";
  }

  if (activeSignal === "south") {
    return newsPulse.bias === "positive";
  }

  return false;
}

function detectEmergency(payload, state, options, clock = getMoscowClock()) {
  const reasons = [];
  const newsPulse = payload.newsPulse || {};
  const topNews = newsPulse.items?.[0];
  let closePosition = false;

  if (
    newsPulse.eventRisk === "high" &&
    topNews &&
    topNews.ageHours <= options.emergencyNewsMaxAgeHours &&
    Math.abs(Number(topNews.score || 0)) >= options.emergencyNewsScore
  ) {
    reasons.push({
      key: `news:${topNews.title}`,
      text: `сильная новость: ${topNews.title}`,
    });
  }

  const activePlan = state.activePlan;
  if (activePlan?.priceAtPlan && payload.close) {
    const movePct = ((payload.close - activePlan.priceAtPlan) / activePlan.priceAtPlan) * 100;
    if (Math.abs(movePct) >= options.emergencyPriceMovePct) {
      reasons.push({
        key: `price:${Math.sign(movePct)}:${Math.floor(Math.abs(movePct))}`,
        text: `резкое движение от утреннего плана: ${movePct.toFixed(2)}%`,
      });
    }
  }

  if (
    activePlan?.plan &&
    activePlan.signal &&
    activePlan.signal !== "neutral" &&
    payload.signal === "neutral"
  ) {
    closePosition = true;
    reasons.push({
      key: `close:${activePlan.signal}:neutral`,
      text: "модель потеряла активное направление",
    });
  }

  if (
    activePlan?.plan &&
    activePlan.signal &&
    activePlan.signal !== "neutral" &&
    payload.signal !== "neutral" &&
    activePlan.signal !== payload.signal
  ) {
    closePosition = true;
    reasons.push({
      key: `flip:${activePlan.signal}:${payload.signal}`,
      text: `модель сменила направление ${activePlan.signal} -> ${payload.signal}`,
    });
  }

  if (
    activePlan?.plan &&
    isNewsAgainstActivePlan(activePlan.signal, newsPulse) &&
    newsPulse.eventRisk === "high" &&
    Number(newsPulse.confidence || 0) >= options.closeNewsConfidence
  ) {
    closePosition = true;
    reasons.push({
      key: `close-news:${activePlan.signal}:${newsPulse.bias}`,
      text: "сильные новости пошли против активной позиции",
    });
  }

  if (!reasons.length) {
    return null;
  }

  const fingerprint = `${clock.date}:${reasons.map((reason) => reason.key).join("|")}`;
  if ((state.emergencyFingerprints || []).includes(fingerprint)) {
    return null;
  }

  return {
    fingerprint,
    closePosition,
    reasons: reasons.map((reason) => reason.text),
  };
}

function buildEmergencyMessage(payload, plan, emergency, clock = getMoscowClock()) {
  const symbol = getInstrumentSymbol(payload);
  const lines = [
    emergency.closePosition
      ? `${symbol} СРОЧНО ЗАКРЫТЬ ПОЗИЦИЮ | ${clock.date} ${clock.label}`
      : `${symbol} ЭКСТРЕННО | ${clock.date} ${clock.label}`,
    ...emergency.reasons.map((reason) => `Причина: ${reason}`),
    `Цена сейчас: ${formatPrice(payload.close)} | сигнал: ${payload.headline} ${payload.probability}%`,
  ];

  if (emergency.closePosition) {
    lines.push(
      "",
      "Действие: закрыть активную позицию или не исполнять старый план до нового подтверждения.",
    );
  }

  if (plan) {
    lines.push(
      "",
      emergency.closePosition ? "Новый план только после подтверждения:" : "Обновленный точный план:",
      `${plan.actionLabel}: ${formatPrice(plan.entry)}`,
      `Условие: ${plan.entryCondition}`,
      `Стоп: ${formatPrice(plan.stop)}`,
      `Тейк 1: ${formatPrice(plan.takeProfit1)}`,
      `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
    );
  } else {
    lines.push("", "Обновленный план: сделки нет, старый план лучше не исполнять.");
  }

  return lines.join("\n");
}

function levelsChangedEnough(previousPlan, nextPlan, thresholdPct) {
  if (!previousPlan || !nextPlan || previousPlan.side !== nextPlan.side) {
    return true;
  }

  for (const key of ["entry", "stop", "takeProfit1", "takeProfit2"]) {
    const previous = Number(previousPlan[key]);
    const next = Number(nextPlan[key]);
    if (!Number.isFinite(previous) || !Number.isFinite(next) || previous === 0) {
      continue;
    }

    const changePct = (Math.abs(next - previous) / Math.abs(previous)) * 100;
    if (changePct >= thresholdPct) {
      return true;
    }
  }

  return false;
}

function shouldSendIntradayRecheck({ payload, plan, state, options, clock }) {
  if (!plan || payload.signal === "neutral" || plan.allowed === false) {
    return {
      send: false,
      reason: "no-actionable-plan",
    };
  }

  if (getTradeSignalCount(state, clock) >= options.maxTradeSignalsPerDay) {
    return {
      send: false,
      reason: "daily-signal-limit",
    };
  }

  const activePlan = state.activePlan;
  const directionChanged =
    activePlan?.signal && activePlan.signal !== "neutral" && activePlan.signal !== payload.signal;
  const impulseBreakout = payload.signalOrigin === "impulse-breakout";
  const strongSignal = Number(payload.probability || 0) >= options.intradayMinProbability;
  const levelsChanged = levelsChangedEnough(
    activePlan?.plan,
    plan,
    options.intradayMinLevelChangePct,
  );

  if (impulseBreakout && (!activePlan?.signal || activePlan.signal !== payload.signal || levelsChanged)) {
    return {
      send: true,
      reason: "impulse-breakout",
    };
  }

  if (activePlan?.signal === payload.signal && !levelsChanged) {
    return {
      send: false,
      reason: "same-direction-same-levels",
    };
  }

  if (!directionChanged && !strongSignal) {
    return {
      send: false,
      reason: "weak-intraday-signal",
    };
  }

  return {
    send: true,
    reason: directionChanged ? "direction-change" : "strong-intraday-signal",
  };
}

function buildIntradayRecheckMessage(payload, plan, reason, clock = getMoscowClock()) {
  const symbol = getInstrumentSymbol(payload);
  const reasonText =
    reason === "direction-change"
      ? "смена направления относительно утреннего плана"
      : reason === "impulse-breakout"
        ? "утренний импульсный breakout по газу"
      : "сильный внутридневной сигнал";

  return [
    `${symbol} контрольный пересчет | ${clock.date} ${clock.label}`,
    `Причина: ${reasonText}`,
    `Сигнал: ${payload.headline} ${payload.probability}% | цена: ${formatPrice(payload.close)}`,
    "",
    `${plan.actionLabel}: ${formatPrice(plan.entry)}`,
    `Условие: ${plan.entryCondition}`,
    `Стоп: ${formatPrice(plan.stop)}`,
    `Тейк 1: ${formatPrice(plan.takeProfit1)}`,
    `Тейк 2: ${formatPrice(plan.takeProfit2)}`,
    "",
    `Новости: ${payload.newsPulse?.bias || "neutral"} | ${payload.newsPulse?.summary || "нет свежего резюме"}`,
  ].join("\n");
}

async function sendDailyPlan(state, options, payload, clock) {
  const plan = buildExecutablePlan(payload);
  const delivered = await sendMessageToChannels(buildDailyPlanMessage(payload, plan, clock), payload);
  if (!delivered) {
    return false;
  }

  state.lastPlanDate = clock.date;
  state.entryWarningDate = null;
  state.entryAlertDate = plan && isEntryReached(plan, payload.close) ? clock.date : null;
  state.takeProfit1Date = null;
  state.activePlan = plan
    ? {
        date: clock.date,
        signal: payload.signal,
        priceAtPlan: roundPrice(payload.close),
        plan,
      }
    : {
        date: clock.date,
        signal: payload.signal,
        priceAtPlan: roundPrice(payload.close),
        plan: null,
      };

  normalizeEmergencyFingerprints(state, clock);
  if (plan && payload.signal !== "neutral") {
    incrementTradeSignalCount(state, clock);
  }
  saveDailyPlanState(options.stateFile, state);
  return true;
}

async function monitorDailyPlan(state, options, payload, clock) {
  const activePlan = state.activePlan;
  const plan = activePlan?.plan;

  normalizeEmergencyFingerprints(state, clock);

  if (activePlan?.date === clock.date && plan && state.entryAlertDate !== clock.date) {
    if (state.entryWarningDate !== clock.date && shouldWarnBeforeEntry(plan, payload.close, options.entryWarningDistancePct)) {
      const delivered = await sendMessageToChannels(buildEntryWarningMessage(payload, plan, clock), payload);
      if (delivered) {
        state.entryWarningDate = clock.date;
        saveDailyPlanState(options.stateFile, state);
      }
    }

    if (isEntryReached(plan, payload.close)) {
      const delivered = await sendMessageToChannels(buildEntryReachedMessage(payload, plan, clock), payload);
      if (delivered) {
        state.entryAlertDate = clock.date;
        saveDailyPlanState(options.stateFile, state);
      }
    }
  }

  if (
    activePlan?.date === clock.date &&
    plan &&
    state.entryAlertDate === clock.date &&
    state.takeProfit1Date !== clock.date &&
    isTakeProfitReached(plan, payload, "takeProfit1")
  ) {
    const adjustedStop = adjustStopAfterTakeProfit1(plan, payload, options.takeProfitManagement);
    if (adjustedStop != null && adjustedStop !== plan.stop) {
      const delivered = await sendMessageToChannels(
        buildTakeProfit1ReachedMessage(payload, plan, adjustedStop, clock),
        payload,
      );
      if (delivered) {
        state.takeProfit1Date = clock.date;
        state.activePlan.plan = {
          ...plan,
          stop: adjustedStop,
          stopMovedAfterTakeProfit1: true,
        };
        saveDailyPlanState(options.stateFile, state);
      }
    }
  }

  const emergency = detectEmergency(payload, state, options, clock);
  if (!emergency) {
    return;
  }

  const nextPlan = buildExecutablePlan(payload);
  const delivered = await sendMessageToChannels(
    buildEmergencyMessage(payload, nextPlan, emergency, clock),
    payload,
  );
  if (!delivered) {
    return;
  }

  state.emergencyFingerprints = [...(state.emergencyFingerprints || []), emergency.fingerprint].slice(-12);
  if (nextPlan && payload.signal !== "neutral") {
    incrementTradeSignalCount(state, clock);
  }
  state.activePlan = {
    date: clock.date,
    signal: payload.signal,
    priceAtPlan: roundPrice(payload.close),
    plan: nextPlan,
  };
  state.entryWarningDate = null;
  state.entryAlertDate = null;
  state.takeProfit1Date = null;
  saveDailyPlanState(options.stateFile, state);
}

async function runIntradayRecheck(state, options, scheduleEntry, clock, payload = null) {
  payload = payload || (await buildSignalPayload(options.intradayRecheckTimeframe));
  const plan = buildExecutablePlan(payload);
  const decision = shouldSendIntradayRecheck({ payload, plan, state, options, clock });

  if (!decision.send) {
    markIntradayRecheck(state, clock, scheduleEntry, {
      sent: false,
      reason: decision.reason,
      signal: payload.signal,
      probability: payload.probability,
    });
    saveDailyPlanState(options.stateFile, state);
    return false;
  }

  const delivered = await sendMessageToChannels(
    buildIntradayRecheckMessage(payload, plan, decision.reason, clock),
    payload,
  );
  if (!delivered) {
    return false;
  }

  incrementTradeSignalCount(state, clock);
  state.activePlan = {
    date: clock.date,
    signal: payload.signal,
    priceAtPlan: roundPrice(payload.close),
    plan,
  };
  state.entryWarningDate = null;
  state.entryAlertDate = isEntryReached(plan, payload.close) ? clock.date : null;
  state.takeProfit1Date = null;
  markIntradayRecheck(state, clock, scheduleEntry, {
    sent: true,
    reason: decision.reason,
    signal: payload.signal,
    probability: payload.probability,
  });
  saveDailyPlanState(options.stateFile, state);
  return true;
}

async function runDailyPlanOnce(state, options) {
  const clock = getMoscowClock();

  try {
    const shouldSendPlan = shouldRunDailyPlan({
      clock,
      state,
      scheduleMinutes: options.dailyPlanMinutes,
      windowMinutes: options.dailyPlanWindowMinutes,
      weekdaysOnly: options.dailyPlanWeekdaysOnly,
    });
    const dueIntradayRecheck = getDueIntradayRecheck({
      clock,
      state,
      scheduleEntries: options.intradayRecheckSchedule,
      windowMinutes: options.intradayRecheckWindowMinutes,
      weekdaysOnly: options.dailyPlanWeekdaysOnly,
    });

    if (!shouldSendPlan && !dueIntradayRecheck && state.activePlan?.date !== clock.date) {
      normalizeEmergencyFingerprints(state, clock);
      saveDailyPlanState(options.stateFile, state);
      return;
    }

    const dailyPayload = await buildSignalPayload("daily");
    const shouldBuildIntraday =
      shouldSendPlan || dueIntradayRecheck || state.activePlan?.date === clock.date;
    const intradayPayload = shouldBuildIntraday
      ? await buildSignalPayload(options.intradayRecheckTimeframe)
      : null;

    if (shouldSendPlan) {
      await sendDailyPlan(state, options, selectMorningPayload(dailyPayload, intradayPayload, options), clock);
      return;
    }

    await monitorDailyPlan(state, options, intradayPayload || dailyPayload, clock);

    if (dueIntradayRecheck) {
      await runIntradayRecheck(state, options, dueIntradayRecheck, clock, intradayPayload);
    }
  } catch (error) {
    console.error(`[${new Date().toISOString()}] daily-plan: ${error.message}`);
  }
}

async function runOnce(state, options) {
  for (const timeframe of options.timeframes) {
    try {
      const payload = await buildSignalPayload(timeframe);
      const fingerprint = buildSignalFingerprint(payload);
      const now = Date.now();
      const previous = state.fingerprints.get(timeframe);
      const lastSentAt = state.lastSentAt.get(timeframe) || 0;

      if (
        shouldNotify({
          mode: options.mode,
          previous,
          current: fingerprint,
          lastSentAt,
          repeatMs: options.repeatMs,
          now,
        })
      ) {
        await notify(payload);
        state.lastSentAt.set(timeframe, now);
      }

      state.fingerprints.set(timeframe, fingerprint);
    } catch (error) {
      console.error(`[${new Date().toISOString()}] ${timeframe}: ${error.message}`);
    }
  }
}

async function runSupervisedSignalCycle(state, options, cycle) {
  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  writeHeartbeat(options.heartbeatFile, {
    service: "signal-watcher",
    status: "cycle-start",
    mode: options.deliveryMode,
    startedAt,
  });

  try {
    await runWithTimeout(cycle, options.cycleTimeoutMs, "signal watcher cycle");
    writeHeartbeat(options.heartbeatFile, {
      service: "signal-watcher",
      status: "ok",
      mode: options.deliveryMode,
      startedAt,
      durationMs: Date.now() - startedAtMs,
    });
  } catch (error) {
    writeHeartbeat(options.heartbeatFile, {
      service: "signal-watcher",
      status: "error",
      mode: options.deliveryMode,
      startedAt,
      durationMs: Date.now() - startedAtMs,
      message: error.message,
      code: error.code || null,
    });
    console.error(`[${new Date().toISOString()}] signal-watcher-supervisor: ${error.message}`);
    if (error.code === "CYCLE_TIMEOUT") {
      process.exit(1);
    }
  }
}

function scheduleSupervisedLoop(state, options, cycle) {
  setTimeout(() => {
    runSupervisedSignalCycle(state, options, cycle).finally(() => {
      scheduleSupervisedLoop(state, options, cycle);
    });
  }, options.intervalMs);
}

function getOptionsFromEnv() {
  return {
    deliveryMode: process.env.SIGNAL_DELIVERY_MODE || "daily-plan",
    intervalMs: Number(process.env.SIGNAL_WATCH_INTERVAL_MS || DEFAULT_INTERVAL_MS),
    repeatMs: Number(process.env.SIGNAL_REPEAT_MS || DEFAULT_REPEAT_MS),
    mode: process.env.SIGNAL_NOTIFY_MODE || "change",
    timeframes: parseCsvList(process.env.SIGNAL_TIMEFRAMES, ["daily"]),
    dailyPlanMinutes: parseScheduleMinutes(process.env.SIGNAL_DAILY_PLAN_TIME || "10:20"),
    dailyPlanWindowMinutes: Number(process.env.SIGNAL_DAILY_PLAN_WINDOW_MINUTES || 20),
    dailyPlanWeekdaysOnly: process.env.SIGNAL_DAILY_PLAN_WEEKDAYS_ONLY !== "false",
    intradayRecheckSchedule: parseScheduleEntries(process.env.SIGNAL_INTRADAY_RECHECK_TIMES, [
      "09:05",
      "09:30",
      "10:00",
      "10:30",
      "15:00",
      "17:45",
      "20:30",
    ]),
    intradayRecheckWindowMinutes: Number(process.env.SIGNAL_INTRADAY_RECHECK_WINDOW_MINUTES || 15),
    intradayRecheckTimeframe: process.env.SIGNAL_INTRADAY_RECHECK_TIMEFRAME || "intraday",
    intradayMinProbability: Number(process.env.SIGNAL_INTRADAY_MIN_PROBABILITY || 56),
    intradayMinLevelChangePct: Number(process.env.SIGNAL_INTRADAY_MIN_LEVEL_CHANGE_PCT || 0.45),
    hybridMorningMinProbability: Number(process.env.SIGNAL_HYBRID_MORNING_MIN_PROBABILITY || 53),
    maxTradeSignalsPerDay: Number(process.env.SIGNAL_MAX_TRADE_SIGNALS_PER_DAY || 2),
    stateFile: process.env.SIGNAL_STATE_FILE || DEFAULT_STATE_FILE,
    emergencyNewsScore: Number(process.env.SIGNAL_EMERGENCY_NEWS_SCORE || 0.85),
    emergencyNewsMaxAgeHours: Number(process.env.SIGNAL_EMERGENCY_NEWS_MAX_AGE_HOURS || 6),
    emergencyPriceMovePct: Number(process.env.SIGNAL_EMERGENCY_PRICE_MOVE_PCT || 1.8),
    closeNewsConfidence: Number(process.env.SIGNAL_CLOSE_NEWS_CONFIDENCE || 0.65),
    entryWarningDistancePct: Number(process.env.SIGNAL_ENTRY_WARNING_DISTANCE_PCT || 0.3),
    takeProfitManagement: getTakeProfitManagementConfig(),
    heartbeatFile: process.env.SIGNAL_HEARTBEAT_FILE || DEFAULT_HEARTBEAT_FILE,
    cycleTimeoutMs: Number(process.env.SIGNAL_CYCLE_TIMEOUT_MS || 180_000),
  };
}

async function main() {
  const options = getOptionsFromEnv();

  if (options.deliveryMode === "daily-plan") {
    const state = loadDailyPlanState(options.stateFile);
    console.log(
      `Signal watcher started: daily plan at ${process.env.SIGNAL_DAILY_PLAN_TIME || "10:20"} MSK, every ${options.intervalMs} ms`,
    );

    writeHeartbeat(options.heartbeatFile, {
      service: "signal-watcher",
      status: "starting",
      mode: options.deliveryMode,
      intervalMs: options.intervalMs,
      cycleTimeoutMs: options.cycleTimeoutMs,
    });
    await runSupervisedSignalCycle(state, options, () => runDailyPlanOnce(state, options));
    scheduleSupervisedLoop(state, options, () => runDailyPlanOnce(state, options));
    return;
  }

  const state = {
    fingerprints: new Map(),
    lastSentAt: new Map(),
  };

  console.log(
    `Signal watcher started: ${options.timeframes.join(", ")} every ${options.intervalMs} ms, mode=${options.mode}`,
  );

  writeHeartbeat(options.heartbeatFile, {
    service: "signal-watcher",
    status: "starting",
    mode: options.deliveryMode,
    intervalMs: options.intervalMs,
    cycleTimeoutMs: options.cycleTimeoutMs,
  });
  await runSupervisedSignalCycle(state, options, () => runOnce(state, options));
  scheduleSupervisedLoop(state, options, () => runOnce(state, options));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  buildDailyPlanMessage,
  buildEmergencyMessage,
  buildEntryReachedMessage,
  buildEntryWarningMessage,
  buildExecutablePlan,
  buildIntradayRecheckMessage,
  buildTakeProfit1ReachedMessage,
  applyImpulseOverlay,
  adjustStopAfterTakeProfit1,
  detectEmergency,
  buildSessionMetrics,
  formatErrorDetails,
  selectMorningPayload,
  buildSignalFingerprint,
  buildSignalMessage,
  buildSignalPayload,
  buildNtfyUrl,
  getDueIntradayRecheck,
  getMoscowClock,
  levelsChangedEnough,
  shouldWarnBeforeEntry,
  isTakeProfitReached,
  isEntryReached,
  sendMax,
  sendNtfy,
  sendDailyPlan,
  shouldSendIntradayRecheck,
  shouldNotify,
  shouldRunDailyPlan,
};
