// Manifest schema id and public exports.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as lib from "../dist/index.js";
import { validateManifestFile } from "../dist/package.js";
import { clone, toyManifest } from "./helpers.mjs";

test("S1S2 is the only model export", () => {
  assert.equal(typeof lib.S1S2.load, "function");
  assert.ok(!("S12" in lib));
});

test("manifest schema id is s1s2.manifest/1", async () => {
  assert.equal(lib.MANIFEST_SCHEMA, "s1s2.manifest/1");
  const m = clone(toyManifest());
  assert.equal(lib.validateManifest({ ...m, schema: "s1s2.manifest/1" }).schema, "s1s2.manifest/1");
  for (const schema of ["s12.manifest/1", "s13.manifest/1"]) {
    assert.throws(() => lib.validateManifest({ ...m, schema }), (e) => e instanceof lib.ManifestError && /unsupported schema id/.test(e.message));
  }

  const dir = mkdtempSync(join(tmpdir(), "s1s2-schema-"));
  const write = (schema) => { const f = join(dir, `${schema.replace("/", "_")}.json`); writeFileSync(f, JSON.stringify({ ...m, schema })); return f; };
  const bad = await validateManifestFile(write("s12.manifest/1"), { manifestOnly: true });
  assert.equal(bad.ok, false);
  assert.ok(bad.errors.some((e) => /unsupported schema id/.test(e)), bad.errors.join("\n"));
  const cur = await validateManifestFile(write("s1s2.manifest/1"), { manifestOnly: true });
  assert.deepEqual(cur.errors, []);
  assert.deepEqual(cur.warnings, []);
});
