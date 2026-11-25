// recipes/SOURCES.sha256: the hash of every recipe file must be the hash of the committed file, and every recipe file must be listed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";

const dir = new URL("../recipes/", import.meta.url);
const rows = readFileSync(new URL("SOURCES.sha256", dir), "utf8").split("\n").map((l) => /^([0-9a-f]{64})  (marimba-audio\/\S+)$/.exec(l)).filter(Boolean);

test("every listed recipe file has the listed sha256", () => {
  assert.ok(rows.length >= 11, `only ${rows.length} rows`);
  for (const [, hash, file] of rows) assert.equal(createHash("sha256").update(readFileSync(new URL(file, dir))).digest("hex"), hash, file);
});

test("every recipe file (but the README) is listed", () => {
  const listed = new Set(rows.map((r) => r[2].replace("marimba-audio/", "")));
  for (const f of readdirSync(new URL("marimba-audio/", dir))) if (f !== "README.md") assert.ok(listed.has(f), `${f} is not in SOURCES.sha256`);
});
