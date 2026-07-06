const express = require("express");
const path = require("path");
const { getMarketSnapshot } = require("./lib/marketData");
const { computeSignal, backtest } = require("./lib/signalEngine");
const { getAutomaticContext, mergeContextWithOverrides } = require("./lib/autoContext");
const { getTimeframeConfig } = require("./lib/timeframes");
const { getNewsPulse } = require("./lib/newsFeed");
const { buildExecutablePlan } = require("./signalWatcher");

const app = express();
const port = Number(process.env.PORT || 3002);
const PRODUCT_TIMEFRAME = "daily";

app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "public")));

function buildMarketView(snapshot) {
  return {
    gas: {
      symbol: snapshot.gas.symbol,
      name: snapshot.gas.shortName,
      instrumentId: snapshot.gas.instrumentId,
      latest: snapshot.gas.latest,
      expirationDate: snapshot.gas.expirationDate,
      lastTradeDate: snapshot.gas.lastTradeDate,
    },
    dxy: {
      symbol: snapshot.dxy.symbol,
      name: snapshot.dxy.shortName,
      latest: snapshot.dxy.latest,
    },
    brent: {
      symbol: snapshot.brent.symbol,
      name: snapshot.brent.shortName,
      latest: snapshot.brent.latest,
    },
  };
}

function sanitizeOverrides(body = {}) {
  return Object.fromEntries(
    Object.entries({
      newsBias: body.newsBias,
      retest: body.retest,
      structure: body.structure,
      eventRisk: body.eventRisk,
    }).filter(([, value]) => value && value !== "auto"),
  );
}

function resolveTimeframe() {
  return PRODUCT_TIMEFRAME;
}

app.get("/api/health", (_request, response) => {
  response.json({
    ok: true,
    service: "ngn6-gas-bot",
    timestamp: new Date().toISOString(),
  });
});

app.get("/api/news", async (_request, response) => {
  try {
    response.json(await getNewsPulse());
  } catch (error) {
    response.status(500).json({ error: error.message });
  }
});

app.get("/api/snapshot", async (request, response) => {
  try {
    const timeframe = resolveTimeframe(request.query.timeframe);
    const snapshot = await getMarketSnapshot(timeframe);
    const autoContext = await getAutomaticContext(snapshot);
    response.json({
      timeframe: snapshot.timeframe,
      timeframeLabel: snapshot.timeframeLabel,
      forecastLabel: snapshot.forecastLabel,
      source: snapshot.source,
      generatedAt: snapshot.generatedAt,
      fallbackReason: snapshot.fallbackReason || null,
      providers: snapshot.providers,
      automaticContext: autoContext.context,
      automaticContextSummary: autoContext.summary,
      upcomingEvents: autoContext.upcomingEvents,
      newsPulse: autoContext.newsPulse,
      market: buildMarketView(snapshot),
    });
  } catch (error) {
    response.status(500).json({ error: error.message });
  }
});

app.post("/api/signal", async (request, response) => {
  try {
    const timeframe = resolveTimeframe(request.body?.timeframe);
    const snapshot = await getMarketSnapshot(timeframe);
    const autoContext = await getAutomaticContext(snapshot);
    const usedContext = mergeContextWithOverrides(autoContext.context, sanitizeOverrides(request.body));
    const signal = computeSignal(snapshot, usedContext);
    const executablePlan = buildExecutablePlan(signal);
    response.json({
      ...signal,
      executablePlan,
      timeframe: snapshot.timeframe,
      timeframeLabel: snapshot.timeframeLabel,
      forecastLabel: snapshot.forecastLabel,
      generatedAt: snapshot.generatedAt,
      fallbackReason: snapshot.fallbackReason || null,
      providers: snapshot.providers,
      automaticContext: autoContext.context,
      usedContext,
      automaticContextSummary: autoContext.summary,
      upcomingEvents: autoContext.upcomingEvents,
      contextSources: autoContext.sources,
      contextWarning: autoContext.warning,
      newsPulse: autoContext.newsPulse,
      market: buildMarketView(snapshot),
    });
  } catch (error) {
    response.status(500).json({ error: error.message });
  }
});

app.get("/api/backtest", async (request, response) => {
  try {
    const timeframe = resolveTimeframe(request.query.timeframe);
    const timeframeConfig = getTimeframeConfig(timeframe);
    const lookback = Number(request.query.lookback || timeframeConfig.backtestLookbackDefault);
    const snapshot = await getMarketSnapshot(timeframe);
    response.json({
      ...backtest(snapshot, lookback),
      providers: snapshot.providers,
      timeframe: snapshot.timeframe,
      timeframeLabel: snapshot.timeframeLabel,
      forecastLabel: snapshot.forecastLabel,
    });
  } catch (error) {
    response.status(500).json({ error: error.message });
  }
});

app.get("*", (_request, response) => {
  response.sendFile(path.join(__dirname, "..", "public", "index.html"));
});

app.listen(port, () => {
  console.log(`NGN6 Gas Bot listening on http://localhost:${port}`);
});
