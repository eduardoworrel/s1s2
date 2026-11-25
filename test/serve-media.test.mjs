// examples/serve-media.mjs: path containment and malformed requests (the server must answer, never throw).
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createMediaServer } from "../examples/serve-media.mjs";

let tmp, server, port;
before(async () => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), "serve-media-"));
  fs.mkdirSync(path.join(tmp, "pub")); fs.mkdirSync(path.join(tmp, "pub-secret"));
  fs.writeFileSync(path.join(tmp, "pub", "s.txt"), "public!");
  fs.writeFileSync(path.join(tmp, "pub-secret", "s.txt"), "secret!");
  server = createMediaServer({ extra: { "/t/": path.join(tmp, "pub") } });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  port = server.address().port;
});
after(() => { server.close(); fs.rmSync(tmp, { recursive: true, force: true }); });

const get = (p, headers = {}) => new Promise((resolve, reject) => {
  http.get({ host: "127.0.0.1", port, path: p, headers }, (res) => {
    const c = []; res.on("data", (d) => c.push(d)); res.on("end", () => resolve({ status: res.statusCode, body: Buffer.concat(c).toString(), headers: res.headers }));
  }).on("error", reject);
});

test("serves a file and a byte range", async () => {
  assert.deepEqual([(await get("/t/s.txt")).status, (await get("/t/s.txt")).body], [200, "public!"]);
  const r = await get("/t/s.txt", { Range: "bytes=1-3" });
  assert.equal(r.status, 206); assert.equal(r.body, "ubl"); assert.equal(r.headers["content-range"], "bytes 1-3/7");
  assert.equal((await get("/t/s.txt", { Range: "bytes=4-" })).body, "ic!");
  assert.equal((await get("/t/s.txt", { Range: "bytes=-2" })).body, "c!");
  assert.equal((await get("/t/s.txt", { Range: "bytes=2-99" })).body, "blic!");
});

test("a sibling directory that shares the root's name prefix is not reachable", async () => {
  for (const p of ["/t/..%2fpub-secret/s.txt", "/t/../pub-secret/s.txt", "/t/..%2f..%2f..%2fetc/passwd"]) {
    const r = await get(p);
    assert.ok(r.status === 403 || r.status === 404, `${p}: ${r.status}`);
    assert.ok(!r.body.includes("secret!"), p);
  }
});

test("malformed requests get an error status and the server keeps answering", async () => {
  assert.equal((await get("/%")).status, 400);
  assert.equal((await get("/t/a%00")).status, 400);
  for (const range of ["bytes=5-2", "bytes=99-", "bytes=abc", "bytes=-", "bytes=0-1,3-4", "bytes=-0", "items=1-2"]) {
    const r = await get("/t/s.txt", { Range: range });
    assert.equal(r.status, 416, range);
  }
  assert.equal((await get("/t/missing.txt")).status, 404);
  const ok = await get("/t/s.txt");
  assert.equal(ok.status, 200); assert.equal(ok.body, "public!");
});
