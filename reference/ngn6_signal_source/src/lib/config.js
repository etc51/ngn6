const fs = require("fs");
const os = require("os");
const path = require("path");

const DEFAULT_TBANK_GAS_INSTRUMENT_ID = "ddd7405e-f3df-4c29-a876-e865013c4e54";
const DEFAULT_TBANK_GAS_TICKER = "NGN6";
const DEFAULT_TBANK_GAS_NAME = "NG-7.26 Природный газ";
const DEFAULT_TBANK_REST_BASE_URLS = ["https://invest-public-api.tbank.ru/rest"];
const ALLOWED_TBANK_REST_HOSTS = new Set([
  "invest-public-api.tbank.ru",
  "sandbox-invest-public-api.tbank.ru",
]);

function readTextFile(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
}

function cleanToken(value) {
  return value.replace(/^['"]|['"]$/g, "").trim();
}

function scoreTokenCandidate(token, line = "") {
  let score = 0;

  if (/^t\.[A-Za-z0-9._:-]{40,}$/i.test(token)) {
    score += 10;
  }

  if (token.length >= 70) {
    score += 4;
  } else if (token.length >= 40) {
    score += 2;
  }

  if (/\./.test(token)) {
    score += 2;
  }

  if (/invest|tbank|tinkoff|token/i.test(line)) {
    score += 3;
  }

  if (/^sk-/i.test(token) || /^https?:\/\//i.test(token)) {
    score -= 10;
  }

  return score;
}

function normalizeToken(rawValue) {
  if (!rawValue) {
    return [];
  }

  const lines = rawValue
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const candidates = [];
  const seen = new Set();

  function pushCandidate(token, line, priority = 0) {
    const cleaned = cleanToken(token);
    if (!cleaned || seen.has(cleaned)) {
      return;
    }

    seen.add(cleaned);
    candidates.push({
      token: cleaned,
      score: scoreTokenCandidate(cleaned, line) + priority,
    });
  }

  for (const line of lines) {
    const match = line.match(/(?:TBANK_API_TOKEN|TINKOFF_TOKEN|TOKEN)\s*[:=]\s*([A-Za-z0-9._:-]{20,})/i);
    if (match) {
      pushCandidate(match[1], line, 20);
      continue;
    }

    const rightSide = line.split(/=(.+)/)[1]?.trim();
    if (
      rightSide &&
      /token|tbank|tinkoff/i.test(line) &&
      /^[A-Za-z0-9._:-]{20,}$/.test(rightSide)
    ) {
      pushCandidate(rightSide, line, 15);
      continue;
    }

    const embeddedMatches = line.match(/[A-Za-z0-9._:-]{20,}/g) || [];
    for (const token of embeddedMatches) {
      if (!/^https?:\/\//i.test(token)) {
        pushCandidate(token, line, /^[A-Za-z0-9._:-]{20,}$/.test(line) ? 5 : 0);
      }
    }
  }

  return candidates
    .sort((left, right) => right.score - left.score)
    .map((entry) => entry.token);
}

function candidateTokenFiles() {
  const desktopDir = path.join(os.homedir(), "Desktop");
  return [
    process.env.TBANK_TOKEN_FILE,
    path.join(desktopDir, "жрт новый про.txt"),
    path.join(desktopDir, "tbank_token.txt"),
    path.join(desktopDir, "tinkoff_token.txt"),
    path.join(desktopDir, "token.txt"),
  ].filter(Boolean);
}

function resolveTBankToken() {
  if (process.env.TBANK_API_TOKEN) {
    return {
      token: process.env.TBANK_API_TOKEN.trim(),
      tokenCandidates: [process.env.TBANK_API_TOKEN.trim()],
      source: "env:TBANK_API_TOKEN",
    };
  }

  for (const filePath of candidateTokenFiles()) {
    const raw = readTextFile(filePath);
    const tokenCandidates = normalizeToken(raw);
    if (tokenCandidates.length) {
      return {
        token: tokenCandidates[0],
        tokenCandidates,
        source: `file:${path.basename(filePath)}`,
      };
    }
  }

  return {
    token: null,
    tokenCandidates: [],
    source: "missing",
  };
}

function parseCsvList(rawValue, fallback) {
  if (!rawValue) {
    return fallback;
  }

  const values = rawValue
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);

  return values.length ? values : fallback;
}

function normalizeTBankRestBaseUrl(rawValue) {
  try {
    const url = new URL(rawValue);
    if (url.protocol !== "https:" || !ALLOWED_TBANK_REST_HOSTS.has(url.hostname)) {
      return null;
    }

    const pathName = url.pathname.replace(/\/+$/u, "") || "/rest";
    return `${url.origin}${pathName}`;
  } catch {
    return null;
  }
}

function parseTBankRestBaseUrls(rawValue) {
  const candidates = parseCsvList(rawValue, DEFAULT_TBANK_REST_BASE_URLS);
  const urls = [
    ...new Set(candidates.map((value) => normalizeTBankRestBaseUrl(value)).filter(Boolean)),
  ];

  return urls.length ? urls : DEFAULT_TBANK_REST_BASE_URLS;
}

function getRuntimeConfig() {
  const tokenState = resolveTBankToken();

  return {
    marketRange: process.env.MARKET_RANGE || "12mo",
    news: {
      refreshMs: Number(process.env.NEWS_CACHE_MS || 60_000),
      requestTimeoutMs: Number(process.env.NEWS_REQUEST_TIMEOUT_MS || 8_000),
      lookbackHours: Number(process.env.NEWS_LOOKBACK_HOURS || 72),
      maxItems: Number(process.env.NEWS_MAX_ITEMS || 10),
      rssUrls: parseCsvList(process.env.NEWS_RSS_URLS, [
        "https://www.naturalgasintel.com/feed/",
        "https://www.eia.gov/rss/todayinenergy.xml",
        "https://www.investing.com/rss/commodities.rss",
        "https://oilprice.com/rss/main",
      ]),
    },
    tbank: {
      token: tokenState.token,
      tokenCandidates: tokenState.tokenCandidates,
      tokenSource: tokenState.source,
      gasInstrumentId: process.env.TBANK_GAS_INSTRUMENT_ID || DEFAULT_TBANK_GAS_INSTRUMENT_ID,
      gasTicker: process.env.TBANK_GAS_TICKER || DEFAULT_TBANK_GAS_TICKER,
      gasName: process.env.TBANK_GAS_NAME || DEFAULT_TBANK_GAS_NAME,
      gasSearchQueries: parseCsvList(process.env.TBANK_GAS_SEARCH, [
        "NGN6",
        "NG-7.26",
        "Природный газ",
        "Natural Gas",
        "Henry Hub",
      ]),
      restBaseUrls: parseTBankRestBaseUrls(process.env.TBANK_REST_URLS),
    },
  };
}

module.exports = {
  getRuntimeConfig,
  normalizeTBankRestBaseUrl,
  parseTBankRestBaseUrls,
};
