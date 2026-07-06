const test = require("node:test");
const assert = require("node:assert/strict");
const { getRuntimeConfig, normalizeTBankRestBaseUrl, parseTBankRestBaseUrls } = require("../src/lib/config");

function withEnv(name, value, callback) {
  const previous = process.env[name];
  if (value == null) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }

  try {
    callback();
  } finally {
    if (previous == null) {
      delete process.env[name];
    } else {
      process.env[name] = previous;
    }
  }
}

test("T-Bank REST default uses only the tbank.ru domain", () => {
  withEnv("TBANK_REST_URLS", null, () => {
    assert.deepEqual(getRuntimeConfig().tbank.restBaseUrls, ["https://invest-public-api.tbank.ru/rest"]);
  });
});

test("legacy tinkoff.ru REST endpoints are filtered out", () => {
  assert.deepEqual(
    parseTBankRestBaseUrls(
      "https://invest-public-api.tinkoff.ru/rest,https://invest-public-api.tbank.ru/rest/",
    ),
    ["https://invest-public-api.tbank.ru/rest"],
  );
});

test("T-Bank REST URL normalizer rejects non-HTTPS and non-T-Bank hosts", () => {
  assert.equal(normalizeTBankRestBaseUrl("http://invest-public-api.tbank.ru/rest"), null);
  assert.equal(normalizeTBankRestBaseUrl("https://invest-public-api.tinkoff.ru/rest"), null);
  assert.equal(
    normalizeTBankRestBaseUrl("https://invest-public-api.tbank.ru/rest/"),
    "https://invest-public-api.tbank.ru/rest",
  );
});
