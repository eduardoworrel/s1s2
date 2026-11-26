// Image reader, pure parts (spec/MULTIMODAL.md section 10): pre-processing, prompt, positions and the option readout. No model, no I/O: everything here is
// checked against the reference implementation (Hugging Face transformers Qwen3-VL processor and get_rope_index, Visual Jev prompts.py) by fixtures.

/** Python's round(): half to even (JavaScript's Math.round is half up). */
export function roundHalfEven(x: number): number {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export interface SmartResizeOptions { factor: number; minPixels: number; maxPixels: number }

/** The `smart_resize` of the Qwen2-VL / Qwen3-VL image processors: both sides multiples of `factor`, area within [minPixels, maxPixels], aspect ratio kept. */
export function smartResize(height: number, width: number, o: SmartResizeOptions): [number, number] {
  if (!(Number.isInteger(height) && Number.isInteger(width) && height > 0 && width > 0)) throw new Error("smartResize: height and width are positive integers");
  if (Math.max(height, width) / Math.min(height, width) > 200) throw new Error(`smartResize: absolute aspect ratio must be smaller than 200, got ${Math.max(height, width) / Math.min(height, width)}`);
  const { factor: f } = o;
  let h = roundHalfEven(height / f) * f, w = roundHalfEven(width / f) * f;
  if (h * w > o.maxPixels) {
    const beta = Math.sqrt((height * width) / o.maxPixels);
    h = Math.max(f, Math.floor(height / beta / f) * f);
    w = Math.max(f, Math.floor(width / beta / f) * f);
  } else if (h * w < o.minPixels) {
    const beta = Math.sqrt(o.minPixels / (height * width));
    h = Math.ceil(height * beta / f) * f;
    w = Math.ceil(width * beta / f) * f;
  }
  return [h, w];
}

// ---- Pillow-compatible bicubic resize (uint8, 3 channels) ------------------------------------------------------------------------------------
// The Hugging Face Pillow backend resizes the uint8 image with PIL BICUBIC (support scaled by the reduction factor, 22-bit fixed-point coefficients,
// each pass rounded to uint8, horizontal pass first unless the height shrinks by more than twice the width). Ported from Pillow's Resample.c (fixtures: Pillow 13.0.0).

const PRECISION_BITS = 32 - 8 - 2;

function bicubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

interface Coeffs { bounds: Int32Array; kk: Int32Array; ksize: number }

function precomputeCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize, filterscale = Math.max(scale, 1);
  const support = 2 * filterscale, ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2), kk = new Int32Array(outSize * ksize), k = new Float64Array(ksize);
  const ss = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5), xmax = Math.trunc(center + support + 0.5);
    if (xmin < 0) xmin = 0;
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
    for (let x = 0; x < xmax; x++) if (ww !== 0) k[x] /= ww;
    for (let x = xmax; x < ksize; x++) k[x] = 0;
    for (let x = 0; x < ksize; x++) kk[xx * ksize + x] = Math.trunc(k[x] < 0 ? -0.5 + k[x] * (1 << PRECISION_BITS) : 0.5 + k[x] * (1 << PRECISION_BITS));
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

const clip8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : v);

/** PIL `Image.resize((dw, dh), Image.BICUBIC)` of an interleaved uint8 RGB image. */
export function resizeBicubicRGB(src: Uint8Array | Uint8ClampedArray, sh: number, sw: number, dh: number, dw: number): Uint8Array {
  if (src.length !== sh * sw * 3) throw new Error(`resizeBicubicRGB: ${src.length} bytes for a ${sw}x${sh} RGB image`);
  if (sh === dh && sw === dw) return Uint8Array.from(src);
  let cur: Uint8Array = Uint8Array.from(src), ch = sh, cw = sw;
  const horizontal = () => {
    const { bounds, kk, ksize } = precomputeCoeffs(cw, dw);
    const out = new Uint8Array(ch * dw * 3);
    for (let y = 0; y < ch; y++) for (let xx = 0; xx < dw; xx++) {
      const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
      for (let c = 0; c < 3; c++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) ss += cur[(y * cw + x + xmin) * 3 + c] * kk[xx * ksize + x];
        out[(y * dw + xx) * 3 + c] = clip8(ss >> PRECISION_BITS);
      }
    }
    cur = out; cw = dw;
  };
  const vertical = () => {
    const { bounds, kk, ksize } = precomputeCoeffs(ch, dh);
    const out = new Uint8Array(dh * cw * 3);
    for (let yy = 0; yy < dh; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
      for (let x = 0; x < cw; x++) for (let c = 0; c < 3; c++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < ymax; y++) ss += cur[((y + ymin) * cw + x) * 3 + c] * kk[yy * ksize + y];
        out[(yy * cw + x) * 3 + c] = clip8(ss >> PRECISION_BITS);
      }
    }
    cur = out; ch = dh;
  };
  // Pillow (current main): the vertical pass goes first when the height shrinks by more than twice the width's reduction; each pass rounds to uint8.
  const horizontalFirst = !(sh - dh > 0 && sh - dh > (sw - dw) * 2);
  if (horizontalFirst) { if (dw !== sw) horizontal(); if (dh !== sh) vertical(); }
  else { if (dh !== sh) vertical(); if (dw !== sw) horizontal(); }
  return cur;
}

/** RGBA (canvas ImageData) to RGB the way PIL's convert("RGB") does: the alpha channel is dropped, not composited. */
export function rgbaToRgb(rgba: Uint8Array | Uint8ClampedArray): Uint8Array {
  const n = rgba.length / 4, out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) { out[i * 3] = rgba[i * 4]; out[i * 3 + 1] = rgba[i * 4 + 1]; out[i * 3 + 2] = rgba[i * 4 + 2]; }
  return out;
}

export interface PatchSpec { patchSize: number; mergeSize: number; temporalPatchSize: number; mean: number[]; std: number[] }

/**
 * rescale (1/255) + normalise + the flat patch layout of the Qwen processors: rows ordered (grid row block, grid column block, row in the 2x2 merge group,
 * column in the group), each row [channel][temporal copy][patchSize][patchSize]; a single image is repeated over the temporal patch. -> [gh * gw, 3 * T * p * p].
 */
export function patchify(rgb: Uint8Array, h: number, w: number, s: PatchSpec): { pixelValues: Float32Array; rows: number; dim: number; grid: [number, number, number] } {
  const { patchSize: p, mergeSize: m, temporalPatchSize: T } = s;
  if (h % (p * m) || w % (p * m)) throw new Error(`patchify: ${w}x${h} is not a multiple of ${p * m}`);
  if (rgb.length !== h * w * 3) throw new Error("patchify: size mismatch");
  const gh = h / p, gw = w / p, dim = 3 * T * p * p, out = new Float32Array(gh * gw * dim);
  // the reference normalises in float32: (x * (1/255) - mean) / std
  const lut: Float32Array[] = [0, 1, 2].map((c) => { const t = new Float32Array(256); for (let v = 0; v < 256; v++) t[v] = (Math.fround(Math.fround(v * Math.fround(1 / 255)) - s.mean[c])) / s.std[c]; return t; });
  let row = 0;
  for (let by = 0; by < gh / m; by++) for (let bx = 0; bx < gw / m; bx++) for (let my = 0; my < m; my++) for (let mx = 0; mx < m; mx++) {
    const base = row * dim;
    for (let c = 0; c < 3; c++) for (let t = 0; t < T; t++) for (let py = 0; py < p; py++) for (let px = 0; px < p; px++) {
      const y = (by * m + my) * p + py, x = (bx * m + mx) * p + px;
      out[base + ((c * T + t) * p + py) * p + px] = lut[c][rgb[(y * w + x) * 3 + c]];
    }
    row++;
  }
  return { pixelValues: out, rows: gh * gw, dim, grid: [1, gh, gw] };
}

export interface PrepareSpec extends PatchSpec { minPixels: number; maxPixels: number }

/** RGB uint8 image of any size -> what the vision graph takes: resized (smart_resize + PIL bicubic), normalised, patched. */
export function prepareImage(rgb: Uint8Array, h: number, w: number, s: PrepareSpec) {
  const [rh, rw] = smartResize(h, w, { factor: s.patchSize * s.mergeSize, minPixels: s.minPixels, maxPixels: s.maxPixels });
  const resized = resizeBicubicRGB(rgb, h, w, rh, rw);
  return { ...patchify(resized, rh, rw, s), resized: [rh, rw] as [number, number], mergedTokens: (rh / s.patchSize / s.mergeSize) * (rw / s.patchSize / s.mergeSize) };
}

// ---- prompt (Visual Jev, code/vdm/models/prompts.py) ----------------------------------------------------------------------------------------------

export const LETTERS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
export const SYSTEM = "You are a visual decision model. You inspect the image and answer each question by choosing exactly one of the given options. You never explain.";
export const MIN_OPTIONS = 2, MAX_OPTIONS = 16;

/** Text of the shared prefix: system turn, user turn opener, the image (`nImageTokens` pads) and the public context. */
export function prefixText(context: string, nImageTokens: number): string {
  const c = context.trim();
  return `<|im_start|>system\n${SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>${"<|image_pad|>".repeat(nImageTokens)}<|vision_end|>${c ? `\n${c}` : ""}`;
}

/** Text of one question's branch, ending at the readout position (the colon of "Answer:"). 2 to 16 options. */
export function suffixText(instruction: string, options: string[]): string {
  if (options.length < MIN_OPTIONS || options.length > MAX_OPTIONS) throw new Error(`a question takes ${MIN_OPTIONS} to ${MAX_OPTIONS} options, got ${options.length} (reject, do not truncate)`);
  const lines = [`\n\nQuestion: ${instruction.trim()}`, "Options:", ...options.map((o, i) => `${LETTERS[i]}. ${o}`), "Reply with the single letter of the correct option."];
  return `${lines.join("\n")}<|im_end|>\n<|im_start|>assistant\nAnswer:`;
}

// ---- M-RoPE positions ------------------------------------------------------------------------------------------------------------------------------

/**
 * Position ids [3][n] of a sequence with one image whose pad tokens are `isImage`: text tokens take one counter on all three axes; the image takes
 * (t = start, h = start + row, w = start + column) over the merged grid, and the counter then advances by max(rows, columns). Port of Qwen3VLModel.get_rope_index.
 */
export function mropePositions(isImage: ArrayLike<boolean | number>, grid: [number, number, number], spatialMerge = 2): Int32Array[] {
  const n = isImage.length, pos = [new Int32Array(n), new Int32Array(n), new Int32Array(n)];
  const gh = grid[1] / spatialMerge, gw = grid[2] / spatialMerge;
  let cur = 0, i = 0, seen = 0;
  while (i < n) {
    let j = i;
    const img = !!isImage[i];
    while (j < n && !!isImage[j] === img) j++;
    if (!img) { for (let k = i; k < j; k++, cur++) { pos[0][k] = pos[1][k] = pos[2][k] = cur; } }
    else {
      if (seen++) throw new Error("mropePositions: one image per sequence");
      if (j - i !== grid[0] * gh * gw) throw new Error(`mropePositions: ${j - i} image tokens for a grid of ${grid[0] * gh * gw}`);
      let k = i;
      for (let t = 0; t < grid[0]; t++) for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++, k++) { pos[0][k] = cur + t; pos[1][k] = cur + y; pos[2][k] = cur + x; }
      cur += Math.max(gh, gw);
    }
    i = j;
  }
  return pos;
}

/** First position id of the suffix of a branch: the prefix's largest position + 1 (equal on the three axes after the image). */
export function nextPosition(prefix: Int32Array[]): number {
  let m = -1;
  for (const a of prefix) for (const v of a) if (v > m) m = v;
  return m + 1;
}

// ---- option readout --------------------------------------------------------------------------------------------------------------------------------

export interface OptionScores { logits: number[]; probabilities: number[]; choice: number; margin: number; rawMargin: number }

/**
 * Scores of a choice question: the LM-head logit of the token of each option letter at the readout position, softmax over the options supplied.
 * `hidden`: final-norm output at the readout position [hidden]; `rows`: embedding rows of " A", " B", ... [K][hidden] (the LM head is tied to the embedding).
 * Accumulates in float64. `temperature` 1 = the raw probabilities of the reference.
 */
export function optionScores(hidden: ArrayLike<number>, rows: ArrayLike<number>, k: number, temperature = 1): OptionScores {
  const d = hidden.length;
  if (rows.length < k * d) throw new Error(`optionScores: ${rows.length} row values for ${k} options of ${d}`);
  if (!(k >= MIN_OPTIONS)) throw new Error("optionScores: at least two options");
  const logits: number[] = [];
  for (let o = 0; o < k; o++) { let s = 0; for (let i = 0; i < d; i++) s += hidden[i] * rows[o * d + i]; logits.push(s); }
  const mx = Math.max(...logits), e = logits.map((v) => Math.exp((v - mx) / temperature)), z = e.reduce((a, b) => a + b, 0), probabilities = e.map((v) => v / z);
  const order = probabilities.map((p, i) => i).sort((a, b) => probabilities[b] - probabilities[a]);
  const sortedLogits = [...logits].sort((a, b) => b - a);
  return { logits, probabilities, choice: order[0], margin: probabilities[order[0]] - probabilities[order[1]], rawMargin: sortedLogits[0] - sortedLogits[1] };
}
