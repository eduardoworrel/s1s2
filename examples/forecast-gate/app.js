// Forecast gate: decide() and forecast() read one prefill; generate() continues from it only when the gate allows.
import * as ort from "onnxruntime-web/webgpu";
import * as tjs from "@huggingface/transformers";
import { S1S2 } from "./lib/index.js";

const $ = (id) => document.getElementById(id);
const status = (m) => { $("status").textContent = m; };
const QUESTION = { name: "q", type: "predicate", instructions: "Does the request ask for a list?" };
let model = null, loaded = "";

async function load(url) {
  if (model && loaded === url) return model;
  await model?.dispose();
  const manifest = await (await fetch(url)).json();
  const base = new URL(".", new URL(url, location.href)).href;
  const baseUrls = Object.fromEntries([manifest.tokenizer.repo, ...Object.values(manifest.builds).map((b) => b.repo)].map((r) => [r, base]));
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
  model = await S1S2.load({ manifest, modules: { ort, transformers: tjs }, baseUrls, onStatus: status });
  loaded = url;
  if (model.forecastUnavailable) throw new Error(model.forecastUnavailable);
  $("beh").innerHTML = model.forecastBehaviors.map((b) => `<option>${b}</option>`).join("");
  return model;
}

$("run").onclick = async () => {
  $("run").disabled = true;
  try {
    const m = await load($("manifest").value);
    status("prefill...");
    const { forecast, session } = await m.forecast({ input: $("prompt").value, questions: [QUESTION] });
    const a = session.answers[0];
    $("decide").textContent = `${a.name}: p(yes) = ${a.probability.toFixed(3)} (${a.method})`;
    $("fc").innerHTML = Object.entries(forecast.behaviors).map(([k, v]) =>
      `<div class="b"><span>${k}</span><div class="bar"><i style="width:${(v.p * 100).toFixed(1)}%"></i></div><span>${v.p.toFixed(2)}</span></div>`).join("");
    $("cost").textContent = `layer ${forecast.layer}, pooling ${forecast.pooling}; ${forecast.cost.prefillTokens} prompt tokens, ${forecast.cost.extraPrefillTokens} extra, head ${forecast.cost.headMs.toFixed(2)} ms`;
    const p = forecast.behaviors[$("beh").value].p, thr = Number($("thr").value);
    $("out").hidden = false;
    if (p > thr) { $("gate").textContent = `p(${$("beh").value}) = ${p.toFixed(2)} > ${thr}: generation skipped`; }
    else {
      status("generating...");
      const g = await session.generate($("prompt").value, { maxNewTokens: 64, temperature: 0 });
      $("gate").textContent = `p(${$("beh").value}) = ${p.toFixed(2)} <= ${thr}: generated (${g.sharedPrefill ? `continued from the same prefill, ${g.reusedTokens} tokens reused` : "this backend re-prefilled"})\n\n${g.text}`;
    }
    await session.dispose();
    status("");
  } catch (e) { status(String(e?.message ?? e)); }
  finally { $("run").disabled = false; }
};
