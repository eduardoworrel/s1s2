// Shipped files describe what was measured and how, in plain words: no internal study / notice / arm identifiers.
// Scans every git-tracked text file (README, docs, spec, src, converter, tests, examples, manifests).
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// Built from pieces so this file does not match itself.
const FORBIDDEN = [
  ["lab", "-s12"],
  ["study", " 0"],
  ["notice", " 0"],
  ["estudo", " 0"],
].map(([a, b]) => new RegExp(`${a}${b}`, "i"));
const ARM = new RegExp(`\\b[Aa]rm [A-Z]\\b|\\b[Bb]raço [A-Z]\\b`);  // "arm K", "braço C"

test("no internal study names in shipped files", () => {
  const files = execFileSync("git", ["ls-files", "-z"], { cwd: root, encoding: "utf8" }).split("\0").filter(Boolean);
  assert.ok(files.length > 50, "git ls-files returned the repository");
  const hits = [];
  for (const f of files) {
    let text;
    try { text = readFileSync(join(root, f), "utf8"); } catch { continue; }  // deleted in the working tree
    if (text.includes("\0")) continue;  // binary
    text.split("\n").forEach((line, i) => {
      for (const re of [...FORBIDDEN, ARM]) if (re.test(line)) hits.push(`${f}:${i + 1}: ${line.trim().slice(0, 120)}`);
    });
  }
  assert.deepEqual(hits, [], "internal names found:\n" + hits.join("\n"));
});
