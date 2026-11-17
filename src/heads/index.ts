// One entry point for every head type of spec 0.2 (section 6): pointer (head.bin | safetensors), linear, joint.
import type { HeadSpec } from "../manifest.js";
import { loadHead, type HeadWeights } from "../readout.js";
import { loadPointerSafetensors } from "./pointer.js";
import { loadLinearHead, type LinearHead } from "./linear.js";
import { loadJointHead, type JointHead, type JointConfig } from "./joint.js";

export type AnyHead = HeadWeights | LinearHead | JointHead;
export const isLinear = (h: AnyHead | undefined): h is LinearHead => (h as any)?.kind === "linear";
export const isJoint = (h: AnyHead | undefined): h is JointHead => (h as any)?.kind === "joint";

type Spec = Pick<HeadSpec, "kind" | "format" | "scale" | "temperature" | "scoreOffset" | "tensors" | "rows" | "softcap" | "layer" | "config" | "read">;

export function loadAnyHead(buf: ArrayBuffer, spec: Spec = {}): AnyHead {
  const kind = spec.kind ?? "pointer";
  if (spec.layer !== undefined && spec.layer !== "final")
    throw new Error(`head reads layer ${spec.layer}: s1s2 0.1 builds expose the final hidden state only (spec 6.6)`);
  const T = spec.temperature;
  if (kind === "pointer") {
    if ((spec.format ?? "head.bin") === "head.bin") return loadHead(buf, spec.scale);
    return loadPointerSafetensors(buf, { scale: spec.scale, temperature: typeof T === "number" ? T : undefined, scoreOffset: spec.scoreOffset });
  }
  if (kind === "linear") {
    const temperature = typeof T === "number" ? { predicate: T, choice: T, score: T, evidence: T } : T;
    return loadLinearHead(buf, { tensors: spec.tensors, rows: spec.rows, softcap: spec.softcap, temperature, read: spec.read });
  }
  return loadJointHead(buf, spec.config as unknown as JointConfig);
}
