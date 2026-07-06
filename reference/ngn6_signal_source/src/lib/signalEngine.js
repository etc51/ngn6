const { getTimeframeConfig } = require("./timeframes");

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sma(values, period, endIndex = values.length - 1) {
  if (endIndex + 1 < period) {
    return null;
  }

  let sum = 0;
  for (let index = endIndex - period + 1; index <= endIndex; index += 1) {
    sum += values[index];
  }

  return sum / period;
}

function pctChange(values, period, endIndex = values.length - 1) {
  if (endIndex - period < 0) {
    return null;
  }

  const previous = values[endIndex - period];
  const current = values[endIndex];

  if (!previous) {
    return null;
  }

  return ((current - previous) / previous) * 100;
}

function highest(values, period, endIndex = values.length - 1) {
  if (endIndex + 1 < period) {
    return null;
  }

  return Math.max(...values.slice(endIndex - period + 1, endIndex + 1));
}

function lowest(values, period, endIndex = values.length - 1) {
  if (endIndex + 1 < period) {
    return null;
  }

  return Math.min(...values.slice(endIndex - period + 1, endIndex + 1));
}

function atr(candles, period, endIndex = candles.length - 1) {
  if (endIndex + 1 < period + 1) {
    return null;
  }

  const trueRanges = [];

  for (let index = endIndex - period + 1; index <= endIndex; index += 1) {
    const candle = candles[index];
    const prevClose = candles[index - 1].close;
    const tr = Math.max(
      candle.high - candle.low,
      Math.abs(candle.high - prevClose),
      Math.abs(candle.low - prevClose),
    );
    trueRanges.push(tr);
  }

  return average(trueRanges);
}

function normalize(value, scale) {
  if (value == null || !Number.isFinite(value)) {
    return 0;
  }

  return clamp(value / scale, -1, 1);
}

function roundPrice(value) {
  if (!Number.isFinite(value)) {
    return null;
  }

  return Number(value.toFixed(2));
}

function formatAboveLevel(value) {
  return `Выше ${roundPrice(value)}`;
}

function formatBelowLevel(value) {
  return `Ниже ${roundPrice(value)}`;
}

function formatZone(low, high) {
  return `${roundPrice(low)} - ${roundPrice(high)}`;
}

function mapNewsBias(value) {
  const map = {
    positive: 1,
    neutral: 0,
    negative: -1,
  };

  return map[value] ?? 0;
}

function mapRetest(value) {
  const map = {
    bullish: 0.55,
    none: 0,
    bearish: -0.55,
  };

  return map[value] ?? 0;
}

function mapStructure(value) {
  const map = {
    uptrend: 0.3,
    neutral: 0,
    downtrend: -0.3,
  };

  return map[value] ?? 0;
}

function eventPenalty(value) {
  const map = {
    none: 0,
    scheduled: 0.7,
    unknown: 1.1,
    high: 1.8,
  };

  return map[value] ?? 0;
}

function buildFeatureSet(snapshot, manual = {}, endIndex = null) {
  const timeframe = getTimeframeConfig(snapshot.timeframe);
  const { periods, scales } = timeframe;
  const gasCandles = snapshot.gas.candles;
  const dxyCandles = snapshot.dxy.candles;
  const brentCandles = snapshot.brent.candles;

  const gasIndex = endIndex ?? gasCandles.length - 1;
  const dxyIndex = Math.min(gasIndex, dxyCandles.length - 1);
  const brentIndex = Math.min(gasIndex, brentCandles.length - 1);

  const gasCloses = gasCandles.map((candle) => candle.close);
  const gasHighs = gasCandles.map((candle) => candle.high);
  const gasLows = gasCandles.map((candle) => candle.low);
  const dxyCloses = dxyCandles.map((candle) => candle.close);
  const brentCloses = brentCandles.map((candle) => candle.close);

  const close = gasCloses[gasIndex];
  const currentCandle = gasCandles[gasIndex];
  const previousCandle = gasCandles[Math.max(0, gasIndex - 1)] || currentCandle;

  const smaFast = sma(gasCloses, periods.trendFast, gasIndex);
  const smaSlow = sma(gasCloses, periods.trendSlow, gasIndex);
  const gasFast = pctChange(gasCloses, periods.momentumFast, gasIndex);
  const gasSlow = pctChange(gasCloses, periods.momentumSlow, gasIndex);
  const dxyFast = pctChange(dxyCloses, periods.momentumFast, dxyIndex);
  const dxySlow = pctChange(dxyCloses, periods.momentumSlow, dxyIndex);
  const brentFast = pctChange(brentCloses, periods.momentumFast, brentIndex);
  const brentSlow = pctChange(brentCloses, periods.momentumSlow, brentIndex);
  const highLocal = highest(gasHighs, periods.local, gasIndex);
  const lowLocal = lowest(gasLows, periods.local, gasIndex);
  const highRange = highest(gasHighs, periods.range, gasIndex);
  const lowRange = lowest(gasLows, periods.range, gasIndex);
  const atrValue = atr(gasCandles, periods.atr, gasIndex);

  const currentHigh = currentCandle.high;
  const currentLow = currentCandle.low;
  const previousHigh = previousCandle.high;
  const previousLow = previousCandle.low;
  const previousClose = previousCandle.close;
  const pivot = (previousHigh + previousLow + previousClose) / 3;

  const trendBias =
    normalize(((close - smaFast) / smaFast) * 100, scales.trendFastPct) * 0.52 +
    normalize(((close - smaSlow) / smaSlow) * 100, scales.trendSlowPct) * 0.48;

  const momentumBias =
    normalize(gasFast, scales.momentumFastPct) * 0.6 +
    normalize(gasSlow, scales.momentumSlowPct) * 0.4;

  const dollarBias =
    normalize(-dxyFast, scales.dxyFastPct) * 0.6 +
    normalize(-dxySlow, scales.dxySlowPct) * 0.4;

  const brentBias =
    normalize(brentFast, scales.brentFastPct) * 0.58 +
    normalize(brentSlow, scales.brentSlowPct) * 0.42;

  const macroBias = dollarBias * 0.4 + brentBias * 0.6;
  const rangeMid = (highRange + lowRange) / 2;
  const halfRange = Math.max((highRange - lowRange) / 2, close * scales.rangeFloorPct);
  const rangeBias = normalize((close - rangeMid) / halfRange, 1);
  const breakoutBias =
    close >= highRange * 0.997 ? 0.55 : close <= lowRange * 1.003 ? -0.55 : 0;
  const atrPercent = atrValue ? (atrValue / close) * 100 : 0;
  const volatilityComfort = clamp(
    1.2 - Math.abs(atrPercent - scales.volatilityTargetPct) / 1.5,
    -0.85,
    1.2,
  );

  const manualNews = mapNewsBias(manual.newsBias);
  const manualRetest = mapRetest(manual.retest);
  const manualStructure = mapStructure(manual.structure);
  const manualEventPenalty = eventPenalty(manual.eventRisk);

  const directionalScore =
    trendBias * 0.17 +
    momentumBias * 0.15 +
    macroBias * 0.12 +
    rangeBias * 0.04 +
    breakoutBias * 0.04 +
    manualNews * 0.34 +
    manualRetest * 0.08 +
    manualStructure * 0.06;

  const alignmentScore =
    Math.abs(trendBias + momentumBias + macroBias * 0.6 + manualNews * 1.35) / 4 +
    Math.max(0, volatilityComfort * 0.15);

  return {
    timeframe: timeframe.key,
    timeframeLabel: timeframe.label,
    forecastLabel: timeframe.forecastLabel,
    date: gasCandles[gasIndex].date,
    close,
    source: snapshot.source,
    breakdown: {
      trendBias,
      momentumBias,
      dollarBias,
      brentBias,
      macroBias,
      rangeBias,
      breakoutBias,
      manualNews,
      manualRetest,
      manualStructure,
      volatilityComfort,
      eventPenalty: manualEventPenalty,
    },
    levels: {
      smaFast,
      smaSlow,
      currentHigh,
      currentLow,
      previousHigh,
      previousLow,
      previousClose,
      pivot,
      highLocal,
      lowLocal,
      highRange,
      lowRange,
      atrValue,
      atrPercent,
      rangeMid,
    },
    score: directionalScore,
    alignmentScore,
  };
}

function deriveTradeLevels(features, signal) {
  const isIntraday = features.timeframe === "intraday";
  const safeAtr = features.levels.atrValue || features.close * 0.02;
  const currentHigh = features.levels.currentHigh || features.close + safeAtr * 0.15;
  const currentLow = features.levels.currentLow || features.close - safeAtr * 0.15;
  const previousHigh = features.levels.previousHigh || features.close + safeAtr * 0.12;
  const previousLow = features.levels.previousLow || features.close - safeAtr * 0.12;
  const pivot = features.levels.pivot || features.close;
  const highLocal = features.levels.highLocal || Math.max(previousHigh, currentHigh);
  const lowLocal = features.levels.lowLocal || Math.min(previousLow, currentLow);
  const highRange = features.levels.highRange || highLocal;
  const lowRange = features.levels.lowRange || lowLocal;

  const triggerBuffer = Math.max(
    safeAtr * (isIntraday ? 0.24 : 0.2),
    features.close * (isIntraday ? 0.0022 : 0.0032),
  );
  const zoneHalf = Math.max(
    safeAtr * (isIntraday ? 0.2 : 0.28),
    features.close * (isIntraday ? 0.0026 : 0.004),
  );
  const cancelBuffer = Math.max(
    safeAtr * (isIntraday ? 0.46 : 0.62),
    features.close * (isIntraday ? 0.005 : 0.0085),
  );
  const tacticalWindow = Math.max(
    safeAtr * (isIntraday ? 1.2 : 1.55),
    features.close * (isIntraday ? 0.012 : 0.022),
  );
  const displayWindow = tacticalWindow * (isIntraday ? 2 : 1.8);
  const localResistance = Math.max(currentHigh, previousHigh);
  const localSupport = Math.min(currentLow, previousLow);
  const rangeResistance = clamp(
    Math.max(highRange, localResistance),
    localResistance,
    features.close + displayWindow,
  );
  const rangeSupport = clamp(
    Math.min(lowRange, localSupport),
    features.close - displayWindow,
    localSupport,
  );

  const base = {
    localResistance: roundPrice(localResistance),
    rangeResistance: roundPrice(rangeResistance),
    localSupport: roundPrice(localSupport),
    rangeSupport: roundPrice(rangeSupport),
  };

  if (signal === "north") {
    const anchor = clamp(
      Math.min(pivot, currentLow + safeAtr * 0.08, features.close - safeAtr * 0.08),
      features.close - tacticalWindow,
      features.close - triggerBuffer * 0.8,
    );
    const zoneLow = clamp(anchor - zoneHalf, features.close - tacticalWindow, features.close);
    const zoneHigh = clamp(anchor + zoneHalf, zoneLow, features.close - triggerBuffer * 0.15);
    const breakoutLevel = clamp(
      Math.max(currentHigh, previousHigh, highLocal),
      features.close + triggerBuffer,
      features.close + tacticalWindow,
    );
    const stopLevel = clamp(
      Math.min(currentLow, previousLow, zoneLow - cancelBuffer),
      features.close - tacticalWindow * 1.1,
      features.close - triggerBuffer,
    );

    return {
      ...base,
      entryZone: { low: zoneLow, high: zoneHigh },
      reclaimLevel: roundPrice(zoneHigh + triggerBuffer * 0.35),
      breakoutLevel: roundPrice(breakoutLevel),
      stopLevel: roundPrice(stopLevel),
      takeProfit1: roundPrice(breakoutLevel + safeAtr * (isIntraday ? 0.82 : 0.6)),
      takeProfit2: roundPrice(breakoutLevel + safeAtr * (isIntraday ? 1.4 : 1.05)),
    };
  }

  if (signal === "south") {
    const anchor = clamp(
      Math.max(pivot, currentHigh - safeAtr * 0.08, features.close + safeAtr * 0.08),
      features.close + triggerBuffer * 0.8,
      features.close + tacticalWindow,
    );
    const zoneLow = clamp(anchor - zoneHalf, features.close + triggerBuffer * 0.15, anchor);
    const zoneHigh = clamp(anchor + zoneHalf, zoneLow, features.close + tacticalWindow);
    const breakdownLevel = clamp(
      Math.min(currentLow, previousLow, lowLocal),
      features.close - tacticalWindow,
      features.close - triggerBuffer,
    );
    const stopLevel = clamp(
      Math.max(currentHigh, previousHigh, zoneHigh + cancelBuffer),
      features.close + triggerBuffer,
      features.close + tacticalWindow * 1.1,
    );

    return {
      ...base,
      entryZone: { low: zoneLow, high: zoneHigh },
      reclaimLevel: roundPrice(zoneLow - triggerBuffer * 0.35),
      breakdownLevel: roundPrice(breakdownLevel),
      stopLevel: roundPrice(stopLevel),
      takeProfit1: roundPrice(breakdownLevel - safeAtr * (isIntraday ? 0.82 : 0.6)),
      takeProfit2: roundPrice(breakdownLevel - safeAtr * (isIntraday ? 1.4 : 1.05)),
    };
  }

  return {
    ...base,
    buyTrigger: roundPrice(Math.max(previousHigh, highLocal, features.close + triggerBuffer)),
    sellTrigger: roundPrice(Math.min(previousLow, lowLocal, features.close - triggerBuffer)),
    waitZone: {
      low: roundPrice(features.close - zoneHalf * 0.9),
      high: roundPrice(features.close + zoneHalf * 0.9),
    },
  };
}

function buildPositionAccess(signal) {
  if (signal === "north") {
    return {
      longAllowed: true,
      shortAllowed: false,
      summary: "Приоритет только в лонг по газу. Шорт не нужен, пока дневная модель поддерживает рост.",
    };
  }

  if (signal === "south") {
    return {
      longAllowed: false,
      shortAllowed: true,
      summary: "Приоритет только в шорт по газу. Лонг не нужен, пока дневная модель поддерживает снижение.",
    };
  }

  return {
    longAllowed: false,
    shortAllowed: false,
    summary: "Явного преимущества нет. Лучше дождаться выхода из локального диапазона.",
  };
}

function buildDecisionPlan(features, signal, headline, levels) {
  const intradayText =
    features.timeframe === "intraday" ? "на ближайший час" : "на следующий день";

  if (signal === "north") {
    return {
      summary: `${headline} ${intradayText}: газ лучше рассматривать от отката в поддержку или по импульсному пробою.`,
      steps: [
        {
          id: "buy-zone",
          tone: "north",
          badge: "Покупка",
          title: "Откат к поддержке",
          levelText: formatZone(levels.entryZone.low, levels.entryZone.high),
          action: `Лонг после возврата выше ${levels.reclaimLevel}.`,
        },
        {
          id: "buy-breakout",
          tone: "north",
          badge: "Покупка",
          title: "Пробой локального хая",
          levelText: formatAboveLevel(levels.breakoutLevel),
          action: "Если цена закрепляется выше уровня, допустим вход по импульсу.",
        },
        {
          id: "buy-cancel",
          tone: "south",
          badge: "Выход",
          title: "Отмена сценария",
          levelText: formatBelowLevel(levels.stopLevel),
          action: "Если уровень потерян, лонг лучше закрыть.",
        },
      ],
    };
  }

  if (signal === "south") {
    return {
      summary: `${headline} ${intradayText}: газ лучше рассматривать от возврата в сопротивление или по пробою вниз.`,
      steps: [
        {
          id: "sell-zone",
          tone: "south",
          badge: "Продажа",
          title: "Возврат к сопротивлению",
          levelText: formatZone(levels.entryZone.low, levels.entryZone.high),
          action: `Шорт после возврата ниже ${levels.reclaimLevel}.`,
        },
        {
          id: "sell-breakdown",
          tone: "south",
          badge: "Продажа",
          title: "Пробой локального лоя",
          levelText: formatBelowLevel(levels.breakdownLevel),
          action: "Если цена закрепляется ниже уровня, допустим вход по импульсу.",
        },
        {
          id: "sell-cancel",
          tone: "north",
          badge: "Выход",
          title: "Отмена сценария",
          levelText: formatAboveLevel(levels.stopLevel),
          action: "Если уровень потерян, шорт лучше закрыть.",
        },
      ],
    };
  }

  return {
    summary: `По газу пока нет чистого преимущества ${intradayText}. Ждём выход из зоны шума.`,
    steps: [
      {
        id: "wait-buy",
        tone: "north",
        badge: "Покупка",
        title: "Пробой вверх",
        levelText: formatAboveLevel(levels.buyTrigger),
        action: "Лонг только после подтверждения пробоя.",
      },
      {
        id: "wait-sell",
        tone: "south",
        badge: "Продажа",
        title: "Пробой вниз",
        levelText: formatBelowLevel(levels.sellTrigger),
        action: "Шорт только после подтверждения пробоя.",
      },
      {
        id: "wait-zone",
        tone: "neutral",
        badge: "Ожидание",
        title: "Зона шума",
        levelText: formatZone(levels.waitZone.low, levels.waitZone.high),
        action: "Внутри диапазона лучше не форсировать вход.",
      },
    ],
  };
}

function describeNewsState(features) {
  const newsScore = features.breakdown.manualNews;

  if (newsScore > 0.35) {
    return "новости поддерживают газ";
  }

  if (newsScore < -0.35) {
    return "новости давят на газ";
  }

  return "новости нейтральны";
}

function buildTradePlan(features, signal, levels) {
  const access = buildPositionAccess(signal);
  const newsState = describeNewsState(features);

  const base = {
    bias: signal === "north" ? "buy" : signal === "south" ? "sell" : "wait",
    summary:
      signal === "north"
        ? `Дневная модель поддерживает покупки газа; ${newsState}`
        : signal === "south"
          ? `Дневная модель поддерживает продажи газа; ${newsState}`
          : `По газу пока лучше ждать; ${newsState}`,
    permissions: access,
    resistance: [
      { label: "Ближнее сопротивление", value: levels.localResistance },
      { label: "Широкое сопротивление", value: levels.rangeResistance },
    ],
    support: [
      { label: "Ближняя поддержка", value: levels.localSupport },
      { label: "Широкая поддержка", value: levels.rangeSupport },
    ],
  };

  if (signal === "north") {
    return {
      ...base,
      entries: [
        {
          label: "Вход от отката",
          levelText: formatZone(levels.entryZone.low, levels.entryZone.high),
          note: `После возврата выше ${levels.reclaimLevel}`,
        },
        {
          label: "Вход по пробою",
          levelText: formatAboveLevel(levels.breakoutLevel),
          note: "Пробой локального импульса",
        },
      ],
      exits: [
        {
          label: "Стоп / отмена",
          levelText: formatBelowLevel(levels.stopLevel),
          note: "Сценарий сломан, лонг закрыть",
        },
        {
          label: "Фиксация 1",
          levelText: formatAboveLevel(levels.takeProfit1),
          note: "Частичная фиксация прибыли",
        },
        {
          label: "Фиксация 2",
          levelText: formatAboveLevel(levels.takeProfit2),
          note: "Основная фиксация прибыли",
        },
      ],
    };
  }

  if (signal === "south") {
    return {
      ...base,
      entries: [
        {
          label: "Вход от отката",
          levelText: formatZone(levels.entryZone.low, levels.entryZone.high),
          note: `После возврата ниже ${levels.reclaimLevel}`,
        },
        {
          label: "Вход по пробою",
          levelText: formatBelowLevel(levels.breakdownLevel),
          note: "Пробой локального импульса вниз",
        },
      ],
      exits: [
        {
          label: "Стоп / отмена",
          levelText: formatAboveLevel(levels.stopLevel),
          note: "Сценарий сломан, шорт закрыть",
        },
        {
          label: "Фиксация 1",
          levelText: formatBelowLevel(levels.takeProfit1),
          note: "Частичная фиксация прибыли",
        },
        {
          label: "Фиксация 2",
          levelText: formatBelowLevel(levels.takeProfit2),
          note: "Основная фиксация прибыли",
        },
      ],
    };
  }

  return {
    ...base,
    entries: [
      {
        label: "Лонг после пробоя",
        levelText: formatAboveLevel(levels.buyTrigger),
        note: "Пока без позиции до выхода вверх",
      },
      {
        label: "Шорт после пробоя",
        levelText: formatBelowLevel(levels.sellTrigger),
        note: "Пока без позиции до выхода вниз",
      },
    ],
    exits: [
      {
        label: "Зона ожидания",
        levelText: formatZone(levels.waitZone.low, levels.waitZone.high),
        note: "Пока цена здесь, рынок шумный",
      },
    ],
  };
}

function scoreToSignal(features) {
  const timeframe = getTimeframeConfig(features.timeframe);
  const { scoring } = timeframe;
  const absoluteScore = Math.abs(features.score);
  const rawProbability =
    50 +
    absoluteScore * scoring.probabilitySlope +
    features.alignmentScore * scoring.alignmentSlope;
  const riskAdjusted = rawProbability - features.breakdown.eventPenalty;
  const probability = clamp(riskAdjusted, 50, scoring.probabilityCap);

  let signal = "neutral";
  let headline = "Ожидание";

  if (features.score >= scoring.scoreThreshold && probability >= scoring.minProbabilityForSignal) {
    signal = "north";
    headline = "Покупка";
  } else if (
    features.score <= -scoring.scoreThreshold &&
    probability >= scoring.minProbabilityForSignal
  ) {
    signal = "south";
    headline = "Продажа";
  }

  const explanation = [];

  if (features.breakdown.manualNews > 0.25) {
    explanation.push("свежие новости поддерживают газ");
  } else if (features.breakdown.manualNews < -0.25) {
    explanation.push("свежие новости давят на газ");
  }

  if (features.breakdown.brentBias > 0.18) {
    explanation.push("энергокомплекс через Brent помогает росту");
  } else if (features.breakdown.brentBias < -0.18) {
    explanation.push("Brent не подтверждает силу газа");
  }

  if (features.breakdown.dollarBias > 0.18) {
    explanation.push("слабость доллара помогает сырьевому сценарию");
  } else if (features.breakdown.dollarBias < -0.18) {
    explanation.push("сильный доллар мешает росту газа");
  }

  if (features.breakdown.trendBias > 0.18) {
    explanation.push("локальный тренд газа смотрит вверх");
  } else if (features.breakdown.trendBias < -0.18) {
    explanation.push("локальный тренд газа смотрит вниз");
  }

  if (features.breakdown.eventPenalty >= 1.8) {
    explanation.push("сильный событийный риск снижает надёжность сигнала");
  } else if (features.breakdown.eventPenalty >= 1.1) {
    explanation.push("рядом событие, поэтому сигнал нужно трактовать осторожнее");
  }

  const tradeLevels = deriveTradeLevels(features, signal);
  const decisionPlan = buildDecisionPlan(features, signal, headline, tradeLevels);
  const tradePlan = buildTradePlan(features, signal, tradeLevels);

  return {
    timeframe: timeframe.key,
    timeframeLabel: timeframe.label,
    forecastLabel: timeframe.forecastLabel,
    signal,
    headline,
    probability: Number(probability.toFixed(1)),
    score: Number(features.score.toFixed(3)),
    date: features.date,
    close: features.close,
    source: features.source,
    explanation,
    decisionPlan,
    tradePlan,
    tradeLevels,
    factors: [
      { id: "news", label: "Новостной поток", value: Number(features.breakdown.manualNews.toFixed(3)) },
      { id: "trend", label: "Тренд газа", value: Number(features.breakdown.trendBias.toFixed(3)) },
      { id: "momentum", label: "Импульс газа", value: Number(features.breakdown.momentumBias.toFixed(3)) },
      { id: "macro", label: "DXY + Brent", value: Number(features.breakdown.macroBias.toFixed(3)) },
      { id: "range", label: "Позиция в диапазоне", value: Number(features.breakdown.rangeBias.toFixed(3)) },
      { id: "retest", label: "Ретест", value: Number(features.breakdown.manualRetest.toFixed(3)) },
      { id: "structure", label: "Старший ТФ", value: Number(features.breakdown.manualStructure.toFixed(3)) },
    ],
    diagnostics: {
      dollarBias: Number(features.breakdown.dollarBias.toFixed(3)),
      brentBias: Number(features.breakdown.brentBias.toFixed(3)),
      breakoutBias: Number(features.breakdown.breakoutBias.toFixed(3)),
      volatilityComfort: Number(features.breakdown.volatilityComfort.toFixed(3)),
      eventPenalty: Number(features.breakdown.eventPenalty.toFixed(3)),
      alignmentScore: Number(features.alignmentScore.toFixed(3)),
    },
    marketLevels: {
      currentHigh: roundPrice(features.levels.currentHigh),
      currentLow: roundPrice(features.levels.currentLow),
      previousHigh: roundPrice(features.levels.previousHigh),
      previousLow: roundPrice(features.levels.previousLow),
      pivot: roundPrice(features.levels.pivot),
      supportRange: tradeLevels.rangeSupport,
      resistanceRange: tradeLevels.rangeResistance,
      smaFast: roundPrice(features.levels.smaFast),
      smaSlow: roundPrice(features.levels.smaSlow),
      atr: roundPrice(features.levels.atrValue),
    },
  };
}

function computeSignal(snapshot, manual = {}) {
  const features = buildFeatureSet(snapshot, manual);
  return scoreToSignal(features);
}

function backtest(snapshot, lookback = 120) {
  const timeframe = getTimeframeConfig(snapshot.timeframe);
  const candles = snapshot.gas.candles;
  const startIndex = Math.max(timeframe.periods.trendSlow + 6, candles.length - lookback - 1);
  let total = 0;
  let wins = 0;
  const samples = [];

  for (let index = startIndex; index < candles.length - 1; index += 1) {
    const features = buildFeatureSet(snapshot, {}, index);
    const result = scoreToSignal(features);

    if (result.signal === "neutral") {
      continue;
    }

    const nextClose = candles[index + 1].close;
    const currentClose = candles[index].close;
    const actualDirection =
      nextClose > currentClose ? "north" : nextClose < currentClose ? "south" : "flat";

    total += 1;
    if (result.signal === actualDirection) {
      wins += 1;
    }

    samples.push({
      date: candles[index].date,
      signal: result.headline,
      probability: result.probability,
      actual: actualDirection === "north" ? "Рост" : actualDirection === "south" ? "Падение" : "Флэт",
    });
  }

  return {
    timeframe: timeframe.key,
    timeframeLabel: timeframe.label,
    forecastLabel: timeframe.forecastLabel,
    source: snapshot.source,
    lookback,
    trades: total,
    wins,
    accuracy: total ? Number(((wins / total) * 100).toFixed(1)) : 0,
    samples: samples.slice(-8),
  };
}

module.exports = {
  backtest,
  buildFeatureSet,
  computeSignal,
  scoreToSignal,
};
