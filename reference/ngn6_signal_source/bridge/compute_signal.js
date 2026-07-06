const fs = require("fs");
const { buildFeatureSet, computeSignal } = require("../src/lib/signalEngine");
const { inferRetest, inferStructure, mergeContextWithOverrides } = require("../src/lib/autoContext");
const {
  applyImpulseOverlay,
  buildExecutablePlan,
  buildSessionMetrics,
  isEntryReached,
} = require("../src/signalWatcher");

function readStdin() {
  return fs.readFileSync(0, "utf8");
}

function jsonError(message, details = {}) {
  process.stdout.write(JSON.stringify({ ok: false, error: message, details }));
}

function cleanManual(value) {
  if (!value || typeof value !== "object") {
    return {};
  }
  return value;
}

function main() {
  let request;
  try {
    request = JSON.parse(readStdin());
  } catch (error) {
    jsonError("invalid_json", { message: error.message });
    return;
  }

  try {
    const snapshot = request.snapshot;
    if (!snapshot?.gas?.candles?.length) {
      jsonError("missing_snapshot_candles");
      return;
    }

    const baseFeatures = buildFeatureSet(snapshot, {});
    const structure = inferStructure(baseFeatures);
    const autoContext = {
      newsBias: "neutral",
      retest: inferRetest(baseFeatures, structure),
      structure,
      eventRisk: "none",
    };
    const manual = mergeContextWithOverrides(autoContext, cleanManual(request.manual));
    let payload = computeSignal(snapshot, manual);

    payload = {
      ...payload,
      autoContext: { context: manual },
      newsPulse: {
        bias: manual.newsBias || "neutral",
        summary: "python bridge",
        source: "python-bridge",
      },
      sessionMetrics: buildSessionMetrics(snapshot),
    };

    if (request.impulseOptions) {
      payload = applyImpulseOverlay(payload, request.impulseOptions);
    }

    const plan = buildExecutablePlan(payload);
    const currentPrice = Number(request.currentPrice ?? payload.close);
    const entryReached = plan ? isEntryReached(plan, currentPrice) : false;

    process.stdout.write(
      JSON.stringify({
        ok: true,
        manual,
        payload,
        plan,
        currentPrice,
        entryReached,
      }),
    );
  } catch (error) {
    jsonError("bridge_failed", { message: error.message, stack: error.stack });
  }
}

main();
