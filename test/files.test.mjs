import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, utimesSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { Files, IntegrityError, sha256Hex, verifyFileOnce } from "../dist/index.js";

const sha = (b) => createHash("sha256").update(b).digest("hex");
const bytes = new TextEncoder().encode("hello head");
const pin = { path: "heads/h.bin", sha256: sha(bytes), size: bytes.length };

test("sha256Hex (WebCrypto)", async () => { assert.equal(await sha256Hex(bytes), sha(bytes)); });

test("browser path: small files are hashed; a wrong sha or size fails loudly", async () => {
  const f = new Files({ platform: "browser", fetch: async () => new Response(bytes) });
  assert.deepEqual(await f.bytes("o/r", "a".repeat(40), pin), bytes);
  await assert.rejects(new Files({ platform: "browser", fetch: async () => new Response("tampered!!") }).bytes("o/r", "a".repeat(40), pin), IntegrityError);
  await assert.rejects(new Files({ platform: "browser", fetch: async () => new Response(bytes) }).bytes("o/r", "a".repeat(40), { ...pin, sha256: "0".repeat(64) }), /sha256/);
});

test("browser path: large Hub files are checked with paths-info at the pinned revision", async () => {
  const rev = "b".repeat(40), big = { path: "onnx/model.onnx_data", sha256: "c".repeat(64), size: 3e9 };
  const calls = [];
  const ok = new Files({ platform: "browser", fetch: async (url, init) => { calls.push([url, init?.method, init?.body]); return Response.json([{ path: big.path, size: 3e9, lfs: { oid: big.sha256, size: 3e9 } }]); } });
  assert.equal(await ok.large("o/r", rev, big), `https://huggingface.co/o/r/resolve/${rev}/onnx/model.onnx_data`);
  assert.equal(calls[0][0], `https://huggingface.co/api/models/o/r/paths-info/${rev}`); assert.equal(calls[0][1], "POST");
  const wrong = new Files({ platform: "browser", fetch: async () => Response.json([{ path: big.path, size: 3e9, lfs: { oid: "d".repeat(64) } }]) });
  await assert.rejects(wrong.large("o/r", rev, big), /sha256 d+ != manifest c+/);
  const size = new Files({ platform: "browser", fetch: async () => Response.json([{ path: big.path, size: 5, lfs: { oid: big.sha256 } }]) });
  await assert.rejects(size.large("o/r", rev, big), /size 5 != manifest/);
  const gone = new Files({ platform: "browser", fetch: async () => new Response("", { status: 404 }) });
  await assert.rejects(gone.large("o/r", rev, big), /HTTP 404/);
});

test("browser path: off-Hub files are hashed when size is pinned and small, refused when large", async () => {
  const lib = { path: "https://cdn.example/lib.wasm", sha256: sha(bytes), size: bytes.length };
  const f = new Files({ platform: "browser", fetch: async () => new Response(bytes) });
  assert.equal(await f.large("o/r", "a".repeat(40), lib), lib.path);
  await assert.rejects(f.large("o/r", "a".repeat(40), { ...lib, size: undefined }), /allowUnverified/);
  const self = new Files({ platform: "browser", baseUrls: { "o/r": "https://h.example/m/" }, fetch: async () => new Response(bytes) });
  await assert.rejects(self.large("o/r", "a".repeat(40), { path: "w.bin", sha256: "e".repeat(64), size: 4e9 }), IntegrityError);
  const allow = new Files({ platform: "browser", baseUrls: { "o/r": "https://h.example/m/" }, allowUnverified: true });
  assert.equal(await allow.large("o/r", "a".repeat(40), { path: "w.bin", sha256: "e".repeat(64), size: 4e9 }), "https://h.example/m/w.bin");
});

test("node: download into <cache>/<revision>/<repo>/<path>, hashed while streaming; tampered download rejected", async () => {
  const cache = mkdtempSync(join(tmpdir(), "s1s2-"));
  let n = 0;
  const f = new Files({ platform: "node", cacheDir: cache, fetch: async () => { n++; return new Response(bytes); } });
  const p = await f.local("o/r", "f".repeat(40), pin);
  assert.equal(p, join(cache, "f".repeat(40), "o", "r", "heads", "h.bin"));
  await f.local("o/r", "f".repeat(40), pin);
  assert.equal(n, 1, "second load served from the cache");
  const bad = new Files({ platform: "node", cacheDir: cache, fetch: async () => new Response("tampered!!") });
  await assert.rejects(bad.local("o/r", "e".repeat(40), pin), IntegrityError);
  assert.deepEqual(readdirSync(join(cache, "e".repeat(40), "o", "r", "heads")), [], "no partial file left");
});

test("node: localDirs are read-only, hashed once, re-hashed when the file changes", async () => {
  const dir = mkdtempSync(join(tmpdir(), "s1s2-local-")), cache = mkdtempSync(join(tmpdir(), "s1s2-c-"));
  const file = join(dir, "x.bin"); writeFileSync(file, bytes);
  const p = { path: "x.bin", sha256: sha(bytes), size: bytes.length };
  const f = new Files({ platform: "node", cacheDir: cache, localDirs: { "o/r": dir } });
  assert.equal(await f.local("o/r", "a".repeat(40), p), file);
  assert.deepEqual(readdirSync(dir), ["x.bin"], "nothing written next to the local file");
  writeFileSync(file, "hello hea!"); utimesSync(file, new Date(), new Date(Date.now() + 5000));
  await assert.rejects(verifyFileOnce(file, p, cache), /sha256/);
});
