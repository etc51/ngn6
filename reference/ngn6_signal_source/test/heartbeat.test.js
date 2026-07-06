const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");
const { runWithTimeout, writeHeartbeat } = require("../src/lib/heartbeat");

test("writeHeartbeat writes a readable status file", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ngn6-heartbeat-"));
  const file = path.join(dir, "heartbeat.json");

  writeHeartbeat(file, { service: "test-worker", status: "ok" });

  const heartbeat = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(heartbeat.service, "test-worker");
  assert.equal(heartbeat.status, "ok");
  assert.equal(heartbeat.pid, process.pid);
  assert.match(heartbeat.updatedAt, /^\d{4}-\d{2}-\d{2}T/u);
});

test("runWithTimeout rejects stuck cycles with a timeout code", async () => {
  await assert.rejects(
    runWithTimeout(
      () =>
        new Promise((resolve) => {
          setTimeout(resolve, 50);
        }),
      5,
      "test cycle",
    ),
    (error) => error.code === "CYCLE_TIMEOUT" && /test cycle timed out/.test(error.message),
  );
});
