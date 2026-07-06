const test = require("node:test");
const assert = require("node:assert/strict");
const { isRelevantItem, scoreHeadline } = require("../src/lib/newsFeed");

test("news relevance requires natural-gas context", () => {
  assert.equal(
    isRelevantItem({
      title: "Bitcoin falls as ETF outflows and Fed outlook weigh on sentiment",
      description: "The dollar and Fed outlook pressured crypto assets.",
      link: "https://example.com/crypto",
    }),
    false,
  );

  assert.equal(
    isRelevantItem({
      title: "Natural gas rises as heatwave lifts cooling demand and storage fears grow",
      description: "Henry Hub futures advance before the EIA report.",
      link: "https://example.com/gas",
    }),
    true,
  );
});

test("gas headline scoring detects bullish gas drivers", () => {
  const scored = scoreHeadline({
    title: "Natural gas rallies as heatwave boosts demand and storage draw deepens",
    description: "Henry Hub prices climbed while LNG exports stayed strong.",
    link: "https://example.com/gas",
    publishedAt: new Date().toUTCString(),
  });

  assert.equal(scored.tone, "positive");
  assert.ok(scored.score > 1);
});
