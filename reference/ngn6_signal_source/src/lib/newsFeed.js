const { getRuntimeConfig } = require("./config");

const PRIMARY_GAS_KEYWORDS = [
  "natural gas",
  "nat gas",
  "natgas",
  "henry hub",
  "lng",
  "liquefied natural gas",
  "gas storage",
  "european gas",
  "ttf gas",
  "pipeline gas",
];

const SECONDARY_GAS_KEYWORDS = [
  "storage",
  "weather",
  "heatwave",
  "cooling demand",
  "heating demand",
  "pipeline",
  "export",
  "imports",
  "freeport",
  "supply",
  "production",
  "inventory",
  "eia",
  "futures",
  "draw",
  "build",
];

const EXCLUDED_KEYWORDS = ["gasoline", "gas station", "ethereum gas", "eth gas"];

const POSITIVE_PATTERNS = [
  { pattern: /\b(price[s]? (surge|jump|rise|rally)|higher natural gas|bullish natural gas)\b/i, weight: 0.95 },
  { pattern: /\b(heatwave|hot weather|cooling demand|strong demand)\b/i, weight: 0.95 },
  { pattern: /\b(storage draw|inventory draw|drawdown)\b/i, weight: 1.2 },
  { pattern: /\b(lng export[s]? rise|export demand|freeport restart|terminal restart)\b/i, weight: 1.1 },
  { pattern: /\b(freeze-off|supply disruption|outage|pipeline outage|production falls)\b/i, weight: 1.25 },
];

const NEGATIVE_PATTERNS = [
  { pattern: /\b(price[s]? (fall|drop|slump|sink)|lower natural gas|bearish natural gas)\b/i, weight: -0.95 },
  { pattern: /\b(mild weather|warmer winter|weak demand)\b/i, weight: -0.95 },
  { pattern: /\b(storage build|inventory build|oversupply)\b/i, weight: -1.15 },
  { pattern: /\b(record production|output increase|supply increase)\b/i, weight: -1.2 },
  { pattern: /\b(lng outage|export outage|terminal outage|export demand falls)\b/i, weight: -1.1 },
];

let newsCache = {
  value: null,
  expiresAt: 0,
  promise: null,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function decodeHtmlEntities(value) {
  return String(value || "")
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&#039;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ");
}

function stripTags(value) {
  return decodeHtmlEntities(value).replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

function getTag(block, tagName) {
  const match = block.match(new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)<\\/${tagName}>`, "i"));
  return match ? stripTags(match[1]) : "";
}

function hoursAgo(value) {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) {
    return 9_999;
  }

  return Math.max(0, (Date.now() - timestamp) / 3_600_000);
}

function freshnessWeight(ageHours) {
  if (ageHours <= 2) {
    return 1.65;
  }

  if (ageHours <= 6) {
    return 1.35;
  }

  if (ageHours <= 24) {
    return 1;
  }

  if (ageHours <= 48) {
    return 0.7;
  }

  return 0.45;
}

function isRelevantItem(item) {
  const text = `${item.title} ${item.description} ${item.link}`.toLowerCase();
  if (EXCLUDED_KEYWORDS.some((keyword) => text.includes(keyword))) {
    return false;
  }

  const hasPrimaryKeyword = PRIMARY_GAS_KEYWORDS.some((keyword) => text.includes(keyword));
  const hasContextualGas = /\bgas\b/.test(text) && SECONDARY_GAS_KEYWORDS.some((keyword) => text.includes(keyword));

  return hasPrimaryKeyword || hasContextualGas;
}

function scoreHeadline(item) {
  const text = `${item.title}. ${item.description}`;
  const normalizedText = text.toLowerCase();
  let score = 0;

  for (const { pattern, weight } of POSITIVE_PATTERNS) {
    if (pattern.test(normalizedText)) {
      score += weight;
    }
  }

  for (const { pattern, weight } of NEGATIVE_PATTERNS) {
    if (pattern.test(normalizedText)) {
      score += weight;
    }
  }

  if (/natural gas|henry hub|lng|gas storage/i.test(normalizedText)) {
    score *= 1.18;
  }

  const ageHours = hoursAgo(item.publishedAt);
  const weightedScore = score * freshnessWeight(ageHours);

  return {
    ...item,
    ageHours,
    score: Number(weightedScore.toFixed(3)),
    tone:
      weightedScore >= 0.3 ? "positive" : weightedScore <= -0.3 ? "negative" : "neutral",
  };
}

function summarizeBias(score) {
  if (score >= 0.18) {
    return "positive";
  }

  if (score <= -0.18) {
    return "negative";
  }

  return "neutral";
}

function summarizeItems(items) {
  if (!items.length) {
    return "В релевантной газовой ленте сейчас нет свежих сильных драйверов.";
  }

  const positive = items.filter((item) => item.score > 0.2).length;
  const negative = items.filter((item) => item.score < -0.2).length;
  const leader = items[0];

  if (positive > negative) {
    return `Новостной поток поддерживает природный газ. Главный драйвер: ${leader.title}`;
  }

  if (negative > positive) {
    return `Новостной поток давит на природный газ. Главный драйвер: ${leader.title}`;
  }

  return `Новостной поток смешанный. Самый заметный заголовок: ${leader.title}`;
}

async function fetchRssSource(url, timeoutMs) {
  const response = await fetch(url, {
    headers: {
      "User-Agent": "ngn6-gas-bot/1.0",
      Accept: "application/rss+xml, application/xml, text/xml",
    },
    signal: AbortSignal.timeout(timeoutMs),
  });

  if (!response.ok) {
    throw new Error(`RSS request failed with ${response.status} for ${url}`);
  }

  const xml = await response.text();
  const channelTitle = getTag(xml, "title") || url;
  const itemBlocks = xml.match(/<item\b[\s\S]*?<\/item>/gi) || [];

  return itemBlocks.map((block) => ({
    source: channelTitle,
    title: getTag(block, "title"),
    link: getTag(block, "link"),
    description: getTag(block, "description"),
    publishedAt: getTag(block, "pubDate"),
  }));
}

async function getNewsPulse() {
  const now = Date.now();
  if (newsCache.value && newsCache.expiresAt > now) {
    return newsCache.value;
  }

  if (newsCache.promise) {
    return newsCache.promise;
  }

  newsCache.promise = (async () => {
    const config = getRuntimeConfig().news;
    const warnings = [];
    const collected = [];

    await Promise.all(
      config.rssUrls.map(async (url) => {
        try {
          const items = await fetchRssSource(url, config.requestTimeoutMs);
          collected.push(...items);
        } catch (error) {
          warnings.push(error.message);
        }
      }),
    );

    const deduped = [];
    const seen = new Set();
    for (const item of collected) {
      const key = `${item.link}::${item.title}`;
      if (!item.title || seen.has(key) || !isRelevantItem(item)) {
        continue;
      }

      seen.add(key);
      deduped.push(item);
    }

    const freshItems = deduped
      .map(scoreHeadline)
      .filter((item) => item.ageHours <= config.lookbackHours)
      .sort((left, right) => Math.abs(right.score) - Math.abs(left.score) || left.ageHours - right.ageHours)
      .slice(0, config.maxItems);

    const totalScore = freshItems.reduce((sum, item) => sum + item.score, 0);
    const normalizedScore = freshItems.length ? clamp(totalScore / (freshItems.length * 1.45), -1, 1) : 0;
    const confidence = clamp(
      freshItems.reduce((sum, item) => sum + Math.abs(item.score), 0) / Math.max(freshItems.length, 1),
      0,
      1,
    );
    const bias = summarizeBias(normalizedScore);
    const strongestFreshItem = freshItems.find((item) => item.ageHours <= 6 && Math.abs(item.score) >= 0.85);
    const eventRisk = strongestFreshItem ? "high" : freshItems.length ? "scheduled" : "none";

    return {
      bias,
      score: Number(normalizedScore.toFixed(3)),
      confidence: Number(confidence.toFixed(3)),
      eventRisk,
      summary: summarizeItems(freshItems),
      generatedAt: new Date().toISOString(),
      source: config.rssUrls.join(", "),
      warning: warnings.length ? warnings.join("; ") : null,
      items: freshItems.map((item) => ({
        title: item.title,
        link: item.link,
        source: item.source,
        publishedAt: item.publishedAt,
        ageHours: Number(item.ageHours.toFixed(1)),
        tone: item.tone,
        score: item.score,
      })),
    };
  })();

  try {
    const news = await newsCache.promise;
    newsCache = {
      value: news,
      expiresAt: Date.now() + getRuntimeConfig().news.refreshMs,
      promise: null,
    };
    return news;
  } catch (error) {
    newsCache = {
      value: null,
      expiresAt: 0,
      promise: null,
    };
    throw error;
  }
}

module.exports = {
  getNewsPulse,
  isRelevantItem,
  scoreHeadline,
};
