// The rename to s1s2: S12 and "s12.manifest/1" keep working as deprecated aliases.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as lib from "../dist/index.js";
import { validateManifestFile } from "../dist/package.js";
import { clone, toyManifest } from "./helpers.mjs";

test("S12 is a deprecated alias of S1S2", () => {
  assert.equal(lib.S12, lib.S1S2);
  assert.equal(typeof lib.S1S2.load, "function");
});

test("manifest schema: s1s2.manifest/1, and the deprecated s12.manifest/1 is still accepted", async () => {
  assert.equal(lib.MANIFEST_SCHEMA, "s1s2.manifest/1");
  const m = clone(toyManifest());
  assert.equal(lib.validateManifest({ ...m, schema: "s1s2.manifest/1" }).schema, "s1s2.manifest/1");
  assert.equal(lib.validateManifest({ ...m, schema: "s12.manifest/1" }).schema, "s12.manifest/1");
  assert.ok(lib.isLegacyManifestSchema("s12.manifest/1") && !lib.isLegacyManifestSchema("s1s2.manifest/1"));
  assert.throws(() => lib.validateManifest({ ...m, schema: "s13.manifest/1" }), lib.ManifestError);

  const dir = mkdtempSync(join(tmpdir(), "s1s2-schema-"));
  const write = (schema) => { const f = join(dir, `${schema.replace("/", "_")}.json`); writeFileSync(f, JSON.stringify({ ...m, schema })); return f; };
  const old = await validateManifestFile(write("s12.manifest/1"), { manifestOnly: true });
  assert.deepEqual(old.errors, []);
  assert.ok(old.warnings.some((w) => /deprecated/.test(w)), old.warnings.join("\n"));
  const cur = await validateManifestFile(write("s1s2.manifest/1"), { manifestOnly: true });
  assert.deepEqual(cur.errors, []);
  assert.ok(!cur.warnings.some((w) => /deprecated/.test(w)));
});
