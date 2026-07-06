const test = require("node:test");
const assert = require("node:assert/strict");
const {
  applyImpulseOverlay,
  adjustStopAfterTakeProfit1,
  buildDailyPlanMessage,
  buildEmergencyMessage,
  buildExecutablePlan,
  buildEntryWarningMessage,
  buildIntradayRecheckMessage,
  buildTakeProfit1ReachedMessage,
  buildNtfyUrl,
  detectEmergency,
  buildSignalFingerprint,
  buildSignalMessage,
  buildSessionMetrics,
  getDueIntradayRecheck,
  levelsChangedEnough,
  isTakeProfitReached,
  isEntryReached,
  sendNtfy,
  sendDailyPlan,
  selectMorningPayload,
  shouldSendIntradayRecheck,
  shouldWarnBeforeEntry,
  shouldNotify,
  shouldRunDailyPlan,
} = require("../src/signalWatcher");

const samplePayload = {
  timeframe: "daily",
  timeframeLabel: "День",
  headline: "Продажа",
  signal: "south",
  probability: 56.4,
  close: 3.42,
  date: "2026-06-26 20:00 UTC",
  generatedAt: "2026-06-27T05:00:00.000Z",
  providers: { gas: "tbank" },
  market: { gas: { symbol: "NGN6", name: "NG-7.26 Природный газ" } },
  tradePlan: {
    permissions: {
      longAllowed: false,
      shortAllowed: true,
    },
    entries: [
      { label: "Вход от отката", levelText: "3.38 - 3.45" },
      { label: "Вход по пробою", levelText: "Ниже 3.31" },
    ],
    exits: [
      { label: "Стоп / отмена", levelText: "Выше 3.56" },
      { label: "Фиксация 1", levelText: "Ниже 3.18" },
    ],
  },
  tradeLevels: {
    breakdownLevel: 3.31,
    stopLevel: 3.56,
    takeProfit1: 3.18,
    takeProfit2: 3.02,
  },
  newsPulse: {
    bias: "negative",
    summary: "Новостной поток давит на природный газ.",
  },
};

test("buildSignalMessage includes the key trading numbers", () => {
  const message = buildSignalMessage(samplePayload);

  assert.match(message, /NGN6/);
  assert.match(message, /Продажа 56\.4%/);
  assert.match(message, /3\.38 - 3\.45/);
  assert.match(message, /Выше 3\.56/);
});

test("notification messages stay branded as NGN6 even on market fallback symbols", () => {
  const message = buildSignalMessage({
    ...samplePayload,
    market: {
      gas: {
        symbol: "NG=F",
        name: "Natural Gas",
      },
    },
  });

  assert.match(message, /^NGN6 /);
  assert.match(message, /Источник: NG=F \/ tbank/);
});

test("fingerprint changes when signal levels change", () => {
  const left = buildSignalFingerprint(samplePayload);
  const right = buildSignalFingerprint({
    ...samplePayload,
    tradePlan: {
      ...samplePayload.tradePlan,
      exits: [{ label: "Стоп / отмена", levelText: "Выше 3.70" }],
    },
  });

  assert.notEqual(left, right);
});

test("change notification mode avoids repeated unchanged signals until repeat timeout", () => {
  const now = 100_000;

  assert.equal(
    shouldNotify({
      mode: "change",
      previous: "same",
      current: "same",
      lastSentAt: now - 1_000,
      repeatMs: 60_000,
      now,
    }),
    false,
  );

  assert.equal(
    shouldNotify({
      mode: "change",
      previous: "same",
      current: "same",
      lastSentAt: now - 70_000,
      repeatMs: 60_000,
      now,
    }),
    true,
  );

  assert.equal(
    shouldNotify({
      mode: "change",
      previous: "old",
      current: "new",
      lastSentAt: now - 1_000,
      repeatMs: 60_000,
      now,
    }),
    true,
  );
});

test("buildNtfyUrl builds a public ntfy topic URL", () => {
  const previousUrl = process.env.NTFY_URL;
  const previousTopic = process.env.NTFY_TOPIC;
  const previousServer = process.env.NTFY_SERVER;

  delete process.env.NTFY_URL;
  process.env.NTFY_TOPIC = "ngn6 test/topic";
  process.env.NTFY_SERVER = "https://ntfy.sh/";

  assert.equal(buildNtfyUrl(), "https://ntfy.sh/ngn6%20test%2Ftopic");

  if (previousUrl === undefined) {
    delete process.env.NTFY_URL;
  } else {
    process.env.NTFY_URL = previousUrl;
  }

  if (previousTopic === undefined) {
    delete process.env.NTFY_TOPIC;
  } else {
    process.env.NTFY_TOPIC = previousTopic;
  }

  if (previousServer === undefined) {
    delete process.env.NTFY_SERVER;
  } else {
    process.env.NTFY_SERVER = previousServer;
  }
});

test("sendNtfy publishes signal text with ntfy headers", async () => {
  const previousFetch = global.fetch;
  const previousTopic = process.env.NTFY_TOPIC;
  const previousTitle = process.env.NTFY_TITLE;
  const previousPriority = process.env.NTFY_PRIORITY;
  const calls = [];

  process.env.NTFY_TOPIC = "ngn6-secret";
  process.env.NTFY_TITLE = "NGN6 test";
  process.env.NTFY_PRIORITY = "urgent";

  global.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true };
  };

  try {
    assert.equal(await sendNtfy("test signal"), true);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://ntfy.sh/ngn6-secret");
    assert.equal(calls[0].options.method, "POST");
    assert.equal(calls[0].options.body, "test signal");
    assert.equal(calls[0].options.headers.Title, "NGN6 test");
    assert.equal(calls[0].options.headers.Priority, "urgent");
    assert.match(calls[0].options.headers["Content-Type"], /text\/plain/);
  } finally {
    global.fetch = previousFetch;

    if (previousTopic === undefined) {
      delete process.env.NTFY_TOPIC;
    } else {
      process.env.NTFY_TOPIC = previousTopic;
    }

    if (previousTitle === undefined) {
      delete process.env.NTFY_TITLE;
    } else {
      process.env.NTFY_TITLE = previousTitle;
    }

    if (previousPriority === undefined) {
      delete process.env.NTFY_PRIORITY;
    } else {
      process.env.NTFY_PRIORITY = previousPriority;
    }
  }
});

test("daily plan state is not marked sent when delivery failed", async () => {
  const previousTopic = process.env.NTFY_TOPIC;
  const previousFetch = global.fetch;
  const previousConsoleLog = console.log;
  const state = {
    lastPlanDate: null,
    activePlan: null,
    entryWarningDate: null,
    entryAlertDate: null,
    emergencyFingerprints: [],
    intradayRechecks: {},
    tradeSignalCountDate: null,
    tradeSignalCount: 0,
  };

  delete process.env.NTFY_TOPIC;
  global.fetch = async () => {
    throw new Error("fetch should not be called without channels");
  };
  console.log = () => {};

  try {
    const delivered = await sendDailyPlan(
      state,
      { stateFile: "__unused__.json" },
      samplePayload,
      { date: "2026-06-30", label: "10:20 МСК" },
    );

    assert.equal(delivered, false);
    assert.equal(state.lastPlanDate, null);
    assert.equal(state.activePlan, null);
    assert.equal(state.tradeSignalCount, 0);
  } finally {
    global.fetch = previousFetch;
    console.log = previousConsoleLog;

    if (previousTopic === undefined) {
      delete process.env.NTFY_TOPIC;
    } else {
      process.env.NTFY_TOPIC = previousTopic;
    }
  }
});

test("daily plan message uses exact prices instead of entry ranges", () => {
  const plan = buildExecutablePlan(samplePayload);
  const message = buildDailyPlanMessage(samplePayload, plan, {
    date: "2026-06-27",
    label: "10:20 МСК",
  });

  assert.equal(plan.side, "short");
  assert.equal(plan.actionLabel, "Вход в шорт");
  assert.equal(plan.entry, 3.31);
  assert.equal(plan.stop, 3.37);
  assert.equal(plan.scenarioStop, 3.56);
  assert.match(message, /Вход в шорт: 3\.31/);
  assert.match(message, /Условие: если цена пробивает уровень вниз/);
  assert.doesNotMatch(message, /SELL STOP|BUY STOP/);
  assert.match(message, /Стоп: 3\.37/);
  assert.match(message, /Тейк 1: 3\.18/);
  assert.doesNotMatch(message, /3\.38 - 3\.45/);
});

test("daily execution stop stays near the nearest invalidation, not the far scenario cancel", () => {
  const plan = buildExecutablePlan({
    ...samplePayload,
    close: 3.52,
    tradeLevels: {
      breakdownLevel: 3.31,
      stopLevel: 4.05,
      takeProfit1: 3.1,
      takeProfit2: 2.95,
      localResistance: 3.84,
    },
    marketLevels: {
      currentHigh: 3.62,
      currentLow: 3.37,
      atr: 0.18,
    },
  });

  assert.equal(plan.entry, 3.31);
  assert.ok(plan.stop < plan.scenarioStop);
  assert.ok(plan.stop - plan.entry < 0.2);
});

test("session metrics use the Moscow 09:00 start for intraday impulse checks", () => {
  const metrics = buildSessionMetrics({
    gas: {
      candles: [
        { timestamp: "2026-06-30T05:45:00.000Z", open: 3.18, high: 3.19, low: 3.17, close: 3.18 },
        { timestamp: "2026-06-30T06:00:00.000Z", open: 3.2, high: 3.22, low: 3.19, close: 3.21 },
        { timestamp: "2026-06-30T06:15:00.000Z", open: 3.21, high: 3.25, low: 3.2, close: 3.24 },
        { timestamp: "2026-06-30T06:30:00.000Z", open: 3.24, high: 3.31, low: 3.23, close: 3.3 },
      ],
    },
  });

  assert.equal(metrics.date, "2026-06-30");
  assert.equal(metrics.candlesSinceOpen, 3);
  assert.equal(metrics.sessionOpen, 3.2);
  assert.equal(metrics.priorSessionHigh, 3.25);
  assert.ok(Math.abs(metrics.movePct - 3.13) <= 0.01);
});

test("impulse overlay upgrades a neutral intraday payload into a long breakout plan", () => {
  const payload = applyImpulseOverlay(
    {
      ...samplePayload,
      timeframe: "intraday",
      timeframeLabel: "Интрадей 15м",
      signal: "neutral",
      headline: "Ожидание",
      probability: 51,
      close: 3.3,
      tradePlan: {
        permissions: {
          longAllowed: false,
          shortAllowed: false,
        },
      },
      marketLevels: {
        currentHigh: 3.31,
        currentLow: 3.23,
        atr: 0.02,
      },
      sessionMetrics: {
        candlesSinceOpen: 3,
        sessionOpen: 3.2,
        sessionHigh: 3.31,
        sessionLow: 3.19,
        priorSessionHigh: 3.25,
        priorSessionLow: 3.19,
        movePct: 3.13,
      },
      factors: [
        { id: "trend", value: 0.36 },
        { id: "momentum", value: 0.48 },
      ],
      newsPulse: {
        bias: "neutral",
        summary: "нет свежего резюме",
      },
    },
    {
      enabled: true,
      movePct: 1.2,
      breakoutBufferPct: 0.03,
      minTrend: 0.28,
      minMomentum: 0.24,
      minCandles: 2,
      minProbability: 55.5,
      maxProbability: 63.5,
    },
  );
  const plan = buildExecutablePlan(payload);

  assert.equal(payload.signal, "north");
  assert.equal(payload.signalOrigin, "impulse-breakout");
  assert.equal(plan.side, "long");
  assert.equal(plan.entryCondition, "импульс выше утреннего максимума");
  assert.ok(payload.probability >= 55.5);
});

test("morning selector prefers an impulse intraday payload over neutral daily", () => {
  const dailyPayload = {
    ...samplePayload,
    signal: "neutral",
    headline: "Ожидание",
    probability: 50.4,
  };
  const intradayPayload = {
    ...samplePayload,
    timeframe: "intraday",
    signal: "north",
    headline: "Покупка",
    probability: 58,
    signalOrigin: "impulse-breakout",
    impulsePlan: {
      side: "long",
      sideLabel: "ЛОНГ",
      actionLabel: "Вход в лонг",
      entryCondition: "импульс выше утреннего максимума",
      entry: 3.3,
      stop: 3.24,
      takeProfit1: 3.35,
      takeProfit2: 3.4,
      allowed: true,
    },
  };

  const selected = selectMorningPayload(dailyPayload, intradayPayload, {
    hybridMorningMinProbability: 53,
  });
  const message = buildDailyPlanMessage(selected, buildExecutablePlan(selected), {
    date: "2026-06-30",
    label: "10:20 МСК",
  });

  assert.equal(selected.signal, "north");
  assert.equal(selected.hybridMode, "hybrid-impulse");
  assert.match(message, /гибрид daily \+ impulse breakout/);
});

test("entry reached follows direction-specific crossing", () => {
  assert.equal(isEntryReached({ side: "short", entry: 3.31 }, 3.32), false);
  assert.equal(isEntryReached({ side: "short", entry: 3.31 }, 3.31), true);
  assert.equal(isEntryReached({ side: "long", entry: 3.55 }, 3.54), false);
  assert.equal(isEntryReached({ side: "long", entry: 3.55 }, 3.55), true);
});

test("entry warning fires within 0.3 percent before the entry level", () => {
  const shortPlan = { side: "short", actionLabel: "Вход в шорт", entryCondition: "если цена пробивает уровень вниз", entry: 3.31, stop: 3.37, takeProfit1: 3.18, takeProfit2: 3.02 };
  const longPlan = { side: "long", actionLabel: "Вход в лонг", entryCondition: "если цена пробивает уровень вверх", entry: 3.52, stop: 3.41, takeProfit1: 3.66, takeProfit2: 3.81 };

  assert.equal(shouldWarnBeforeEntry(shortPlan, 3.325, 0.3), false);
  assert.equal(shouldWarnBeforeEntry(shortPlan, 3.319, 0.3), true);
  assert.equal(shouldWarnBeforeEntry(shortPlan, 3.31, 0.3), false);
  assert.equal(shouldWarnBeforeEntry(longPlan, 3.508, 0.3), false);
  assert.equal(shouldWarnBeforeEntry(longPlan, 3.511, 0.3), true);
  assert.equal(shouldWarnBeforeEntry(longPlan, 3.52, 0.3), false);

  const message = buildEntryWarningMessage(
    { ...samplePayload, close: 3.319 },
    shortPlan,
    { date: "2026-06-29", label: "12:00 МСК" },
  );
  assert.match(message, /скоро возможен вход/);
  assert.match(message, /До входа: 0\.27%/);
});

test("take profit 1 detection uses the current candle extreme", () => {
  const longPlan = { side: "long", takeProfit1: 3.33 };
  const shortPlan = { side: "short", takeProfit1: 3.18 };

  assert.equal(
    isTakeProfitReached(longPlan, { close: 3.32, marketLevels: { currentHigh: 3.33 } }),
    true,
  );
  assert.equal(
    isTakeProfitReached(shortPlan, { close: 3.19, marketLevels: { currentLow: 3.18 } }),
    true,
  );
  assert.equal(
    isTakeProfitReached(longPlan, { close: 3.32, marketLevels: { currentHigh: 3.329 } }),
    false,
  );
});

test("take profit 1 stop adjustment moves risk closer after partial profit", () => {
  const longPlan = { side: "long", entry: 3.31, stop: 3.29, takeProfit1: 3.33, takeProfit2: 3.35, sideLabel: "ЛОНГ" };
  const shortPlan = { side: "short", entry: 3.31, stop: 3.37, takeProfit1: 3.18, takeProfit2: 3.02, sideLabel: "ШОРТ" };
  const config = {
    enabled: true,
    stopLockR: 0.25,
    stopBufferAtr: 0.1,
    stopBufferPct: 0.0015,
  };

  assert.ok(
    adjustStopAfterTakeProfit1(longPlan, { close: 3.33, marketLevels: { atr: 0.02 } }, config) > longPlan.stop,
  );
  assert.ok(
    adjustStopAfterTakeProfit1(shortPlan, { close: 3.18, marketLevels: { atr: 0.02 } }, config) < shortPlan.stop,
  );

  const message = buildTakeProfit1ReachedMessage(
    { close: 3.33 },
    longPlan,
    3.31,
    { date: "2026-06-30", label: "18:00 МСК" },
  );
  assert.match(message, /тейк 1 достигнут/);
  assert.match(message, /Новый стоп: 3\.31/);
});

test("daily plan schedule fires once inside the Moscow weekday window", () => {
  const clock = { date: "2026-06-29", weekday: 1, minutes: 10 * 60 + 20 };
  const scheduleMinutes = 10 * 60 + 20;

  assert.equal(
    shouldRunDailyPlan({
      clock,
      state: { lastPlanDate: null },
      scheduleMinutes,
      windowMinutes: 20,
    }),
    true,
  );

  assert.equal(
    shouldRunDailyPlan({
      clock,
      state: { lastPlanDate: "2026-06-29" },
      scheduleMinutes,
      windowMinutes: 20,
    }),
    false,
  );

  assert.equal(
    shouldRunDailyPlan({
      clock: { date: "2026-06-29", weekday: 1, minutes: 11 * 60 },
      state: { lastPlanDate: null },
      scheduleMinutes,
      windowMinutes: 20,
    }),
    false,
  );

  assert.equal(
    shouldRunDailyPlan({
      clock: { date: "2026-06-27", weekday: 6, minutes: 10 * 60 + 20 },
      state: { lastPlanDate: null },
      scheduleMinutes,
      windowMinutes: 20,
    }),
    false,
  );
});

test("intraday recheck schedule fires once per configured slot", () => {
  const state = { intradayRechecks: {} };
  const clock = { date: "2026-06-29", weekday: 1, minutes: 15 * 60 + 3 };
  const scheduleEntries = [
    { label: "15:00", minutes: 15 * 60 },
    { label: "17:45", minutes: 17 * 60 + 45 },
  ];

  assert.equal(
    getDueIntradayRecheck({
      clock,
      state,
      scheduleEntries,
      windowMinutes: 15,
    })?.label,
    "15:00",
  );

  state.intradayRechecks = {
    "2026-06-29": {
      "15:00": { sent: false, reason: "weak-intraday-signal" },
    },
  };

  assert.equal(
    getDueIntradayRecheck({
      clock,
      state,
      scheduleEntries,
      windowMinutes: 15,
    }),
    null,
  );
});

test("intraday recheck sends only on direction change or strong signal", () => {
  const plan = buildExecutablePlan(samplePayload);
  const options = {
    intradayMinProbability: 56,
    intradayMinLevelChangePct: 0.45,
    maxTradeSignalsPerDay: 2,
  };
  const clock = { date: "2026-06-29" };

  assert.deepEqual(
    shouldSendIntradayRecheck({
      payload: { ...samplePayload, probability: 55.9 },
      plan,
      state: {
        tradeSignalCountDate: "2026-06-29",
        tradeSignalCount: 1,
        activePlan: { date: "2026-06-29", signal: "north", plan: { ...plan, side: "long" } },
      },
      options,
      clock,
    }),
    { send: true, reason: "direction-change" },
  );

  assert.deepEqual(
    shouldSendIntradayRecheck({
      payload: { ...samplePayload, probability: 56 },
      plan,
      state: {
        tradeSignalCountDate: "2026-06-29",
        tradeSignalCount: 1,
        activePlan: { date: "2026-06-29", signal: "south", plan },
      },
      options,
      clock,
    }),
    { send: false, reason: "same-direction-same-levels" },
  );

  assert.deepEqual(
    shouldSendIntradayRecheck({
      payload: { ...samplePayload, probability: 60 },
      plan,
      state: {
        tradeSignalCountDate: "2026-06-29",
        tradeSignalCount: 2,
        activePlan: { date: "2026-06-29", signal: "north", plan: { ...plan, side: "long" } },
      },
      options,
      clock,
    }),
    { send: false, reason: "daily-signal-limit" },
  );

  assert.deepEqual(
    shouldSendIntradayRecheck({
      payload: { ...samplePayload, signal: "north", signalOrigin: "impulse-breakout", probability: 54 },
      plan: { ...plan, side: "long", entry: 3.5, stop: 3.42, takeProfit1: 3.6, takeProfit2: 3.7 },
      state: {
        tradeSignalCountDate: "2026-06-29",
        tradeSignalCount: 1,
        activePlan: { date: "2026-06-29", signal: "neutral", plan: null },
      },
      options,
      clock,
    }),
    { send: true, reason: "impulse-breakout" },
  );
});

test("level change filter ignores tiny repeated same-direction updates", () => {
  const previous = { side: "short", entry: 3.31, stop: 3.37, takeProfit1: 3.18, takeProfit2: 3.02 };
  const tiny = { side: "short", entry: 3.32, stop: 3.38, takeProfit1: 3.18, takeProfit2: 3.02 };
  const changed = { side: "short", entry: 3.35, stop: 3.43, takeProfit1: 3.12, takeProfit2: 2.95 };

  assert.equal(levelsChangedEnough(previous, tiny, 0.45), false);
  assert.equal(levelsChangedEnough(previous, changed, 0.45), true);
});

test("intraday recheck message contains exact execution levels", () => {
  const plan = buildExecutablePlan(samplePayload);
  const message = buildIntradayRecheckMessage(
    samplePayload,
    plan,
    "direction-change",
    { date: "2026-06-29", label: "17:45 МСК" },
  );

  assert.match(message, /контрольный пересчет/);
  assert.match(message, /смена направления/);
  assert.match(message, /Вход в шорт: 3\.31/);
  assert.match(message, /Стоп: 3\.37/);
});

test("emergency marks active position for urgent close when signal goes neutral", () => {
  const plan = buildExecutablePlan(samplePayload);
  const emergency = detectEmergency(
    {
      ...samplePayload,
      signal: "neutral",
      headline: "Ожидание",
      probability: 50,
      newsPulse: { bias: "neutral", eventRisk: "none", items: [] },
    },
    {
      activePlan: {
        date: "2026-06-29",
        signal: "south",
        priceAtPlan: 3.42,
        plan,
      },
      emergencyFingerprints: [],
    },
    {
      emergencyNewsMaxAgeHours: 6,
      emergencyNewsScore: 0.85,
      emergencyPriceMovePct: 1.8,
      closeNewsConfidence: 0.65,
    },
    { date: "2026-06-29" },
  );

  assert.equal(emergency.closePosition, true);
  assert.match(emergency.reasons.join(" "), /потеряла активное направление/);

  const message = buildEmergencyMessage(
    { ...samplePayload, signal: "neutral", headline: "Ожидание", probability: 50 },
    null,
    emergency,
    { date: "2026-06-29", label: "15:00 МСК" },
  );
  assert.match(message, /СРОЧНО ЗАКРЫТЬ ПОЗИЦИЮ/);
  assert.match(message, /закрыть активную позицию/);
});

test("emergency marks active position for urgent close on strong opposite news", () => {
  const plan = buildExecutablePlan(samplePayload);
  const emergency = detectEmergency(
    {
      ...samplePayload,
      signal: "south",
      newsPulse: {
        bias: "positive",
        confidence: 0.8,
        eventRisk: "high",
        items: [{ title: "Natural gas jumps on surprise storage draw", ageHours: 1, score: 0.9 }],
      },
    },
    {
      activePlan: {
        date: "2026-06-29",
        signal: "south",
        priceAtPlan: 3.42,
        plan,
      },
      emergencyFingerprints: [],
    },
    {
      emergencyNewsMaxAgeHours: 6,
      emergencyNewsScore: 0.85,
      emergencyPriceMovePct: 1.8,
      closeNewsConfidence: 0.65,
    },
    { date: "2026-06-29" },
  );

  assert.equal(emergency.closePosition, true);
  assert.match(emergency.reasons.join(" "), /новости пошли против/);
});
