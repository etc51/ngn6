const fs = require("fs");
const path = require("path");

function defaultHeartbeatFile(name) {
  return path.join(process.cwd(), "data", `${name}-heartbeat.json`);
}

function writeHeartbeat(filePath, record = {}) {
  if (!filePath) {
    return;
  }

  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    `${JSON.stringify(
      {
        updatedAt: new Date().toISOString(),
        pid: process.pid,
        ...record,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
}

function runWithTimeout(task, timeoutMs, label = "operation") {
  const timeout = Number(timeoutMs || 0);
  if (!Number.isFinite(timeout) || timeout <= 0) {
    return Promise.resolve().then(task);
  }

  let timeoutHandle = null;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutHandle = setTimeout(() => {
      const error = new Error(`${label} timed out after ${timeout} ms`);
      error.code = "CYCLE_TIMEOUT";
      reject(error);
    }, timeout);
  });

  return Promise.race([Promise.resolve().then(task), timeoutPromise]).finally(() => {
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }
  });
}

module.exports = {
  defaultHeartbeatFile,
  runWithTimeout,
  writeHeartbeat,
};
