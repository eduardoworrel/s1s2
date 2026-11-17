// Margin policy ("margin gate"): escalate low-margin System-1 answers.
//
// Every answer carries `margin` (probability gap, study 002's definition) and `rawMargin` (readout.ts). A policy
// says which one it reads and, per question type, below which value an answer is
// escalated, and to what: another build of the same manifest (e.g. int4 -> int8), or System 2 on the same weights
// continuing from the decision prefill. Named MarginPolicy to stay clear of backend.ts's concurrency `Gate`.
//
// Status of the numbers. The library ships NO default threshold. lab-s12's dev probe (exploratory) ranked items by rawMargin and escalated a fixed share (5-20%), not an absolute margin;
// study 002 (H4/G4a) used the 10%/15% quantiles of dev-B int4 margins, per model and task. Absolute
// thresholds differ per model, build and readout. A threshold is either measured by the user for their build
// (`thresholdFromQuantile` over their own margins) or comes from a manifest that says where it came from. Every
// default below is provisional: study 002 fixed quantiles per model and task, not a threshold that transfers.
// S1 -> S2 escalation ("generate") is NOT measured anywhere; the generated text is attached and the typed decision
// is left unchanged.

import type { EscalationSpec, Manifest, QType } from "./manifest.js";
import type { Answer } from "./readout.js";

export const PROVISIONAL_NOTE = "provisional - set by study 002";

/** Library defaults (no thresholds). Provisional - set by study 002. */
export const MARGIN_POLICY_DEFAULTS: Readonly<Required<Pick<EscalationSpec, "prompt" | "maxNewTokens" | "provisional" | "source">>> = Object.freeze({
  prompt: "Reconsider the question above step by step, then state your final answer as one of: {options}.",
  maxNewTokens: 256,
  provisional: true,
  source: PROVISIONAL_NOTE,
});

/** A policy after merging every level. */
export interface MarginPolicy {
  metric: "margin" | "rawMargin";
  below: Partial<Record<QType, number>>;
  to: "generate" | { build: string } | null;
  prompt: string;
  maxNewTokens: number;
  provisional: boolean;
  source: string;
}

/** Merge spec levels (later wins; `below` merges per type). Provisional stays true if any level that sets a
 *  threshold is provisional or silent about it. */
export function mergePolicy(...levels: (EscalationSpec | undefined | null)[]): MarginPolicy {
  const out: MarginPolicy = { metric: "margin", below: {}, to: null, prompt: MARGIN_POLICY_DEFAULTS.prompt, maxNewTokens: MARGIN_POLICY_DEFAULTS.maxNewTokens, provisional: false, source: "" };
  let anyThreshold = false, provisional = false;
  const sources: string[] = [];
  for (const l of levels) {
    if (!l) continue;
    if (l.below && Object.keys(l.below).length) {
      Object.assign(out.below, l.below);
      anyThreshold = true;
      if (l.provisional !== false) provisional = true;
      sources.push(l.source ?? "unspecified");
    }
    if (l.metric !== undefined) out.metric = l.metric;
    if (l.to !== undefined) out.to = l.to;
    if (l.prompt !== undefined) out.prompt = l.prompt;
    if (l.maxNewTokens !== undefined) out.maxNewTokens = l.maxNewTokens;
  }
  out.provisional = anyThreshold ? provisional : MARGIN_POLICY_DEFAULTS.provisional;
  out.source = sources.length ? sources.join(" < ") : MARGIN_POLICY_DEFAULTS.source;
  return out;
}

/** Policy for a manifest/build pair (task and call levels are added per question). */
export function manifestPolicy(m: Manifest, build?: string): EscalationSpec[] {
  return [m.escalation, build ? m.builds[build]?.escalation : undefined].filter(Boolean) as EscalationSpec[];
}

/** Threshold for one answer, or null when the policy does not gate this type / the answer has no margin. */
export function shouldEscalate(a: Pick<Answer, "type" | "margin" | "rawMargin">, p: MarginPolicy): { threshold: number; value: number } | null {
  const t = p.below[a.type], v = a[p.metric];
  if (t === undefined || v === null || v === undefined || !Number.isFinite(v)) return null;
  return v < t ? { threshold: t, value: v } : null;
}

/**
 * Threshold that escalates the lowest-margin share `q` of a calibration set (what lab-s12's dev probe did: rank by
 * margin, escalate a fixed budget). Returns the smallest margin NOT escalated, so `margin < threshold` selects
 * floor(q * n) items when margins are distinct. Calibrate on your own build and readout; never reuse across models.
 */
export function thresholdFromQuantile(margins: number[], q: number): number {
  if (!(q > 0 && q < 1)) throw new Error("thresholdFromQuantile: q must be in (0, 1)");
  const xs = margins.filter(Number.isFinite).sort((a, b) => a - b);
  if (xs.length < 2) throw new Error("thresholdFromQuantile: need at least 2 finite margins");
  const k = Math.floor(q * xs.length);
  return xs[Math.min(k, xs.length - 1)];
}
