#!/usr/bin/env node
// Serve the forecast example and an optional package whose local files are verified at startup.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createMediaServer } from "../serve-media.mjs";
import { validateManifestFile } from "../../dist/package.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const lib = path.resolve(here, "../..");
const transformers = path.join(lib, "node_modules/@huggingface/transformers");
const args = process.argv.slice(2);
const option = (key) => args.includes(`--${key}`) ? args[args.indexOf(`--${key}`) + 1] : undefined;
const port = Number(option("port") ?? 8788);
const routes = {
  "/vendor/ort/": path.join(lib, "node_modules/onnxruntime-web/dist"),
  "/vendor/ortc/": path.join(lib, "node_modules/onnxruntime-common/dist/esm"),
  "/vendor/tjs/": path.join(transformers, "dist"),
  "/": here,
};
if (option("model")) {
  const root = fs.realpathSync(path.resolve(option("model")));
  const file = path.join(root, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(file, "utf8"));
  const repos = [manifest.tokenizer.repo, ...Object.values(manifest.builds).map((b) => b.repo)];
  if (manifest.forecast?.head.repo) repos.push(manifest.forecast.head.repo);
  const roots = Object.fromEntries(repos.map((repo) => [repo, root]));
  const result = await validateManifestFile(file, { strict: true, roots });
  const problems = [...result.errors, ...result.warnings.filter((warning) => warning.endsWith("is a URL (not checked locally)"))];
  if (problems.length) throw new Error(problems.join("\n"));
  routes["/model/"] = root;
}
createMediaServer({ extra: routes, canonical: true }).listen(port, "127.0.0.1", () => {
  console.log(`http://127.0.0.1:${port}/`);
});
