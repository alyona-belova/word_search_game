#!/usr/bin/env node
// usage: node fetch-logs.js
// columns: visitID, date, clientID, isNewUser, visitDuration, goalsID, params
// params: { ab_group, is_returning, level, level_status, hints_used, drop_off_pct }

import https from "https";
import fs from "fs";

const TOKEN = "your_token";
const COUNTER = "your_counter";

const today = new Date();
const thirtyAgo = new Date(+today - 30 * 24 * 60 * 60 * 1000);
const fmt = (d) => d.toISOString().slice(0, 10);
const DATE1 = fmt(thirtyAgo);
const DATE2 = fmt(today);

const FIELDS = [
  "ym:s:visitID",
  "ym:s:date",
  "ym:s:clientID",
  "ym:s:isNewUser",
  "ym:s:visitDuration",
  "ym:s:goalsID",
  "ym:s:params",
].join(",");

function apiRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const headers = { Authorization: `OAuth ${TOKEN}` };
    if (body) {
      headers["Content-Type"] = "application/x-www-form-urlencoded";
      headers["Content-Length"] = Buffer.byteLength(body);
    }
    const req = https.request(
      { hostname: "api-metrika.yandex.net", path, method, headers },
      (res) => {
        let data = "";
        res.on("data", (c) => (data += c));
        res.on("end", () => {
          try {
            resolve({ status: res.statusCode, body: JSON.parse(data) });
          } catch {
            resolve({ status: res.statusCode, body: data });
          }
        });
      },
    );
    req.on("error", reject);
    if (body) req.write(body);
    req.end();
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  console.log(`Logs API · counter ${COUNTER} · ${DATE1} → ${DATE2}\n`);

  const createRes = await apiRequest(
    "POST",
    `/logs/v1/counter/${COUNTER}/logrequest?` +
      new URLSearchParams({
        date1: DATE1,
        date2: DATE2,
        source: "visits",
        fields: FIELDS,
      }),
  );
  if (createRes.status !== 200) {
    console.error("Create failed:", JSON.stringify(createRes.body, null, 2));
    process.exit(1);
  }
  const requestId = createRes.body.log_request?.request_id;
  console.log("Request ID:", requestId);

  for (;;) {
    await sleep(10_000);
    const info = await apiRequest(
      "GET",
      `/logs/v1/counter/${COUNTER}/logrequest/${requestId}`,
    );
    const status = info.body.log_request?.status ?? "";
    console.log("Status:", status);
    if (status === "processed") break;
    if (status === "cleaned_by_user" || status === "processing_failed") {
      console.error("Request failed:", info.body);
      process.exit(1);
    }
  }

  const infoRes = await apiRequest(
    "GET",
    `/logs/v1/counter/${COUNTER}/logrequest/${requestId}`,
  );
  const parts = infoRes.body.log_request?.parts ?? [];
  console.log(`Downloading ${parts.length} part(s)…`);

  const rows = [];
  for (const part of parts) {
    const res = await apiRequest(
      "GET",
      `/logs/v1/counter/${COUNTER}/logrequest/${requestId}/part/${part.part_number}/download`,
    );
    const text =
      typeof res.body === "string" ? res.body : JSON.stringify(res.body);
    const lines = text.split("\n").filter(Boolean);
    if (rows.length === 0) rows.push(lines[0]);
    rows.push(...lines.slice(1));
  }

  const outFile = `reports/metrica-sessions-${DATE1}-${DATE2}.tsv`;
  fs.writeFileSync(outFile, rows.join("\n") + "\n");
  console.log(`\nSaved ${rows.length - 1} sessions → ${outFile}`);

  await apiRequest(
    "POST",
    `/logs/v1/counter/${COUNTER}/logrequest/${requestId}/clean`,
  );
  console.log("Log request cleaned.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
