// Privacy Policy Checker: on-device, with s1s2 (System 1 decision + optional System 2 quote on the same weights).
// The requirement questions are the exact prompts the heads were trained with (requirements.json), so
// a head trained by the lab plugs in through the manifest (builds.<name>.heads, defaults.method.choice = "head").
import * as ort from "onnxruntime-web/webgpu";
import * as tjs from "@huggingface/transformers";
import { S1S2 } from "./lib/index.js";

const $ = (id) => document.getElementById(id);
const status = (m) => { $("status").textContent = m; };
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const params = new URLSearchParams(location.search);

let REQ = null, LAW = "GDPR";
const models = {}; // manifest file -> S1S2
let mirrors = {};

async function loadModel(file) {
  if (models[file]) return models[file];
  const url = params.get(file.includes(".pt.") ? "pt" : "en") ?? (/^https?:\/\//.test(file) ? file : `./manifests/${file}`);
  const manifest = await (await fetch(url)).json();
  const baseUrls = {};
  for (const repo of [manifest.tokenizer.repo, ...Object.values(manifest.builds).map((b) => b.repo)])
    if (mirrors[repo]) baseUrls[repo] = new URL(mirrors[repo], location.href).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
  const lib = await S1S2.load({ manifest, build: "genai-int4-webgpu", modules: { ort, transformers: tjs }, baseUrls,
    genai: { verifyEp: true }, onStatus: (m) => status(`${manifest.id}: ${m}`) });
  models[file] = lib;
  return lib;
}

/** Split a policy into passages: paragraphs, short lines merged up to ~40 words, long ones cut at sentences (~220 words). */
export function splitPassages(text) {
  const paras = text.replace(/\r/g, "").split(/\n\s*\n|\n(?=\s*(?:[-•*]|\d+[.)]|[A-Z][A-Z \d&,'-]{3,}$))/m)
    .map((p) => p.replace(/\s*\n\s*/g, " ").trim()).filter(Boolean);
  const out = [];
  let buf = "";
  const words = (s) => (s.match(/\S+/g) || []).length;
  const push = (p) => {
    if (words(p) <= 220) { out.push(p); return; }
    let cur = "";
    for (const s of p.match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [p]) {
      if (cur && words(cur) + words(s) > 220) { out.push(cur.trim()); cur = ""; }
      cur += s;
    }
    if (cur.trim()) out.push(cur.trim());
  };
  for (const p of paras) {
    buf = buf ? buf + " " + p : p;
    if (words(buf) >= 40) { push(buf); buf = ""; }
  }
  if (buf) { if (out.length && words(buf) < 15) out[out.length - 1] += " " + buf; else push(buf); }
  return out;
}

function question(law, r) {
  return law.question.replace("{name}", r.name_of).replace("{definition}", r.definition);
}

async function run() {
  const law = REQ.laws[LAW];
  const text = $("text").value.trim();
  if (!text) { status("Paste a policy first."); return; }
  const reqs = law.requirements.filter((r) => !$("primaryOnly").checked || r.primary);
  const passages = splitPassages(text);
  $("run").disabled = true;
  try {
    const lib = await loadModel(law.manifest);
    const yes = lib.manifest.lab.labels.yes, no = lib.manifest.lab.labels.no;
    const best = reqs.map(() => ({ p: -1, i: -1 }));
    const t0 = performance.now();
    for (const [i, passage] of passages.entries()) {
      status(`Reading passage ${i + 1} of ${passages.length}…`);
      if (lib.countTokens(passage) > 900) continue;
      // Both option orders, averaged: the decision does not depend on where "Yes" is listed.
      const pYes = [];
      for (const order of [[yes, no], [no, yes]]) {
        const s = await lib.decide({ input: passage, questions: reqs.map((r) => ({ name: r.code, type: "choice", instructions: question(law, r), choices: order })) }, { keepState: false });
        s.answers.forEach((a, k) => { pYes[k] = (pYes[k] ?? 0) + a.probabilities.find((x) => x.value === yes).probability / 2; });
      }
      pYes.forEach((p, k) => { if (p > best[k].p) best[k] = { p, i }; });
    }
    const rows = reqs.map((r, k) => ({ r, ...best[k], addressed: best[k].p >= 0.5 }));
    if ($("quote").checked) {
      for (const row of rows.filter((x) => x.addressed)) {
        status(`Quoting for “${row.r.name}”…`);
        const prompt = law.quote.replace("{passage}", passages[row.i]).replace("{name}", row.r.name_of).replace("{definition}", row.r.definition);
        row.quote = (await lib.generate(prompt, { maxNewTokens: 96, temperature: 0 })).text.trim();
      }
    }
    render(law, rows, passages, lib, (performance.now() - t0) / 1000);
    status("");
  } catch (e) {
    status(`Error: ${e?.message ?? e}`);
    console.error(e);
  } finally { $("run").disabled = false; }
}

function render(law, rows, passages, lib, secs) {
  const n = rows.filter((x) => x.addressed).length;
  const method = lib.heads.includes("choice") ? "pointer head" : "next-token logits";
  const adapted = lib.manifest.lab?.variant && lib.manifest.lab.variant !== "base";
  const el = $("results");
  el.hidden = false;
  el.innerHTML = `
    <div class="summary"><div><span class="big">${n}</span><span class="big" style="color:var(--muted)"> / ${rows.length}</span></div>
      <div><b>${LAW}</b> requirements addressed by at least one passage<br><small>${passages.length} passages · ${secs.toFixed(1)} s · ${esc(lib.manifest.base.repo)} (${esc(lib.build)}, ${method}${adapted ? "" : ", base model: not adapted, results are illustrative"})</small></div></div>
    <table><thead><tr><th>Requirement</th><th>Result</th><th class="hide-sm">Best passage</th></tr></thead><tbody>
    ${rows.map((x) => `<tr>
      <td><b>${esc(x.r.name)}</b>${x.r.primary ? "" : '<span class="tag">secondary</span>'}<div class="def">${esc(x.r.cites.slice(0, 2).join("; "))}</div></td>
      <td><span class="pill ${x.addressed ? "yes" : "no"}">${x.addressed ? "Addressed" : "Not found"}</span>
          <div class="bar" title="P(yes) = ${x.p.toFixed(2)}"><i style="width:${Math.round(100 * Math.max(0, x.p))}%"></i></div></td>
      <td class="hide-sm">${x.i >= 0 ? `<details><summary>Passage ${x.i + 1} · p = ${x.p.toFixed(2)}</summary><blockquote>${esc(passages[x.i].slice(0, 900))}</blockquote></details>` : "—"}
          ${x.quote ? `<blockquote class="quote">${esc(x.quote)}</blockquote>` : ""}</td></tr>`).join("")}
    </tbody></table>`;
}

async function fetchUrl() {
  const u = $("url").value.trim();
  if (!u) return;
  status("Fetching…");
  try {
    const r = await fetch(u);
    const html = await r.text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    doc.querySelectorAll("script,style,noscript,nav,header,footer,aside,svg,form,iframe").forEach((n) => n.remove());
    doc.querySelectorAll("p,li,h1,h2,h3,h4,h5,h6,div,section,br,tr").forEach((n) => n.append("\n\n"));
    $("text").value = (doc.body?.textContent ?? "").replace(/[ \t]+/g, " ").replace(/\n\s*\n\s*(\n\s*)+/g, "\n\n").trim();
    status("");
  } catch {
    status("This site does not allow in-browser fetching (CORS). Open the policy, copy the text and paste it here.");
  }
}

async function main() {
  REQ = await (await fetch("./requirements.json")).json();
  try { mirrors = await (await fetch("./models.json")).json(); } catch { mirrors = {}; }
  document.querySelectorAll(".laws button").forEach((b) => b.addEventListener("click", () => {
    LAW = b.dataset.law;
    document.querySelectorAll(".laws button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  $("run").addEventListener("click", run);
  $("fetch").addEventListener("click", fetchUrl);
  if (params.get("law") && REQ.laws[params.get("law")]) document.querySelector(`.laws button[data-law=${params.get("law")}]`).click();
  if (!navigator.gpu) { status("This demo needs WebGPU (recent Chrome or Edge)."); $("run").textContent = "WebGPU unavailable"; return; }
  try {
    await loadModel(REQ.laws.GDPR.manifest);
    $("run").textContent = "Check policy";
    $("run").disabled = false;
    status("Model ready. Everything runs locally.");
  } catch (e) { status(`Could not load the model: ${e?.message ?? e}`); }
  $("foot").textContent = "Requirements and question wording: GDPR arts. 13/14; LGPD; CCPA §1798.130 and 11 CCR §7011). Not legal advice.";
}
main();
