// transformers.js backend (ported from ct0-edge backends/transformersjs.ts). One forward per
// prefill; no state is kept, so a session's System-2 continuation re-prefills (sharedPrefill: false).

import { createGate, resolveGenerateOptions, checkPositions, type Backend, type GenerateOptions } from "../backend.js";
import { tensorToF32, resolveHidden } from "./common.js";

export interface TransformersJsOptions {
  eos: Set<number>;
  hiddenOutput: string | null;
}

export function transformersJsFromModel(model: any, Tensor: any, opts: TransformersJsOptions): Backend {
  if (!(opts?.eos instanceof Set) || !opts.eos.size) throw new Error("transformers.js: opts.eos (Set of EOS ids from the manifest) is required");
  const eos = opts.eos;
  const sess = model.sessions?.model ?? Object.values(model.sessions ?? {})[0] as any;
  const hiddenName = resolveHidden("transformers.js", sess?.outputNames ?? [], opts.hiddenOutput);
  const toTensors = (ids: number[]) => ({
    input_ids: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
  });
  const gate = createGate("transformers.js");
  return {
    kind: "transformers.js",
    hiddenName,
    async prefill(ids, want) {
      gate.assertLive(); checkPositions("transformers.js", ids, want.positions);
      if (want.positions.length && !hiddenName) throw new Error("transformers.js: this build has no hidden output; use method 'logits'");
      const rel = await gate.acquire();
      try {
        const out = await model.forward(toTensors(ids));
        const L = out.logits, V = L.dims.at(-1);
        const lastLogits = tensorToF32(L.data, L.type, (L.dims.at(-2) - 1) * V, V);
        const hiddens: Float32Array[] = [];
        if (want.positions.length) {
          const H = out[hiddenName!], D = H.dims.at(-1);
          if (H.dims.at(-2) !== ids.length) throw new Error(`transformers.js: hidden seq ${H.dims.at(-2)} != ${ids.length}`);
          for (const p of want.positions) hiddens.push(tensorToF32(H.data, H.type, p * D, D));
        }
        return { hiddens, lastLogits };
      } finally { rel(); }
    },
    dispose() { return gate.dispose(() => model.dispose?.()); },
    async *generate(ids, o: GenerateOptions) {
      gate.assertLive();
      const g = resolveGenerateOptions(o);
      const max = g.maxNewTokens, temp = g.temperature;
      if (!ids.length) throw new Error("generate: empty ids");
      if (max === 0) return;
      const rel = await gate.acquire();
      const queue: number[] = []; let done = false, wake: (() => void) | null = null, err: any = null, first = true, stop = false;
      const ping = () => { const w = wake; wake = null; w?.(); };
      const streamer = {
        put(v: any) { if (first) { first = false; return; } // the first put is the prompt
          if (stop) return;
          for (const t of v.flat ? v.flat() : v) queue.push(Number(t)); ping(); },
        end() { done = true; ping(); },
      };
      const criterion = (batch: any[]) => batch.map(() => stop || !!o.signal?.aborted);
      const args: any = { ...toTensors(ids), max_new_tokens: max, streamer, stopping_criteria: [criterion], do_sample: temp > 0, repetition_penalty: g.repetitionPenalty };
      if (temp > 0) { args.temperature = temp; args.top_p = g.topP; args.top_k = g.topK; }
      const run = Promise.resolve().then(() => model.generate(args)).then(() => { done = true; ping(); }, (e: any) => { err = e; done = true; ping(); });
      let emitted = 0;
      try {
        while (true) {
          if (o.signal?.aborted) break;
          if (queue.length) {
            const t = queue.shift()!;
            if (eos.has(t)) break;
            yield t;
            if (++emitted >= max) break;
          } else if (done) break;
          else await new Promise<void>((r) => { wake = r; if (queue.length || done) ping(); });
        }
      } finally {
        stop = true;
        try { await run; } finally { rel(); }
      }
      if (err && !o.signal?.aborted) throw err;
    },
  };
}
