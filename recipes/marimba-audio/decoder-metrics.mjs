// Comparison of decoder answers with the torch references (spec/MULTIMODAL.md section 9.4); shared by decoder-parity.mjs (Node CPU)
// and test/browser/media-chrome-decoder.mjs (Chrome WebGPU).
// A result row: { id, probabilities: {code: p}, scores: {code: logp sum}, hyps: [5 texts], terms: {term: fraction}, ms: {open, language, term, total}, featDiff? }
const softmax = (o) => { const ks = Object.keys(o), m = Math.max(...ks.map((k) => o[k])), e = ks.map((k) => Math.exp(o[k] - m)), s = e.reduce((a, b) => a + b, 0); return Object.fromEntries(ks.map((k, i) => [k, e[i] / s])); };
const top = (p) => Object.keys(p).reduce((a, b) => (p[a] >= p[b] ? a : b));
const med = (a) => { const s = a.filter((x) => Number.isFinite(x)).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const ok = (b) => (b ? "ok" : "FAIL");

/** Reference rows (dec_ref.jsonl lines) against result rows of the same ids. */
export function compare(refRows, results, { mode = "features", baseline = "torch" } = {}) {
  const byId = new Map(results.map((r) => [r.id, r]));
  const S = { windows: 0, argmax: 0, topDp: 0, maxDp: 0, maxDScore: 0, pairs: 0, eq: 0, within: 0, dec: 0, greedy: 0, beam: 0, beams: 0, feat: 0 };
  const perWindow = [];
  for (const ref of refRows) {
    const r = byId.get(ref.id); if (!r) continue;
    const pRef = softmax(ref.language), t = top(pRef);
    S.windows++; S.argmax += top(r.probabilities) === t ? 1 : 0;
    const dp = Math.max(...Object.keys(pRef).map((k) => Math.abs(pRef[k] - r.probabilities[k])));
    S.maxDp = Math.max(S.maxDp, dp); S.topDp = Math.max(S.topDp, Math.abs(pRef[t] - r.probabilities[t]));
    S.maxDScore = Math.max(S.maxDScore, ...Object.keys(ref.language).map((k) => Math.abs(ref.language[k] - r.scores[k])));
    S.greedy += r.hyps[0] === ref.hyps[0] ? 1 : 0;
    for (let k = 1; k < 5; k++) { S.beams++; S.beam += r.hyps[k] === ref.hyps[k] ? 1 : 0; }
    if (r.featDiff != null) S.feat = Math.max(S.feat, r.featDiff);
    let pe = 0;
    for (const [term, q] of Object.entries(ref.terms)) {
      const p = r.terms[term]; if (p === undefined) continue;
      S.pairs++; S.eq += Math.abs(p - q) < 1e-9 ? 1 : 0; S.within += Math.abs(p - q) <= 0.2 + 1e-9 ? 1 : 0; S.dec += (p >= 0.5) === (q >= 0.5) ? 1 : 0; pe += Math.abs(p - q) < 1e-9 ? 1 : 0;
    }
    perWindow.push({ id: ref.id, lang: [t, top(r.probabilities)], dp, greedyEqual: r.hyps[0] === ref.hyps[0], beamsEqual: [1, 2, 3, 4].map((k) => r.hyps[k] === ref.hyps[k]), termEqual: pe, terms: Object.keys(ref.terms).length });
  }
  const sum = {
    baseline, mode, windows: S.windows, featuresMaxAbsDiff: mode === "e2e" ? S.feat : null,
    languageArgmaxEqual: `${S.argmax}/${S.windows}`, languageMaxAbsDiffTopProbability: S.topDp, languageMaxAbsDiffProbability: S.maxDp, languageMaxAbsDiffScore: S.maxDScore,
    termPairs: S.pairs, termFractionIdentical: S.eq / S.pairs, termFractionWithin02: S.within / S.pairs, termDecisionEqual: S.dec / S.pairs,
    greedyTextEqual: S.greedy / S.windows, beamTextEqual: S.beam / S.beams,
    medianMs: { open: med(results.map((r) => r.ms.open)), language: med(results.map((r) => r.ms.language)), term: med(results.map((r) => r.ms.term)), total: med(results.map((r) => r.ms.total)) },
  };
  sum.gates = {
    "language argmax identical on every window": ok(S.argmax === S.windows),
    "language top probability abs diff <= 0.02": ok(S.topDp <= 0.02),
    "term fraction within 0.2 on >= 95% of pairs": ok(S.within / S.pairs >= 0.95),
    "term fraction identical on >= 85% of pairs": ok(S.eq / S.pairs >= 0.85),
    "term decision at 0.5 identical on >= 95% of pairs": ok(S.dec / S.pairs >= 0.95),
    ...(mode === "e2e" ? { "audio_features max abs diff <= 2e-3": ok(S.feat <= 2e-3) } : {}),
  };
  return { summary: sum, perWindow };
}

/** Result row from the answers of [language, ...terms] (what AudioModel returns) and the cost parts. */
export function rowFromAnswers(id, answers, terms, timing, ms, featDiff) {
  const lang = answers[0];
  const probabilities = Object.fromEntries(lang.probabilities.map((x) => [x.value, x.probability]));
  const scores = Object.fromEntries(lang.probabilities.map((x, i) => [x.value, lang.raw[i]]));
  const t = {}; terms.forEach((w, i) => { t[w] = answers[1 + i].probability; });
  return { id, probabilities, scores, hyps: answers[1].hypotheses, terms: t, ms: { open: timing.decoderOpen ?? 0, language: timing.language ?? 0, term: timing.term ?? 0, total: ms }, featDiff };
}
