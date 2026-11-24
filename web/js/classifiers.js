// Classifiers: the three trained CNNs (ONNX, run with onnxruntime-web) and a
// nearest-neighbour classifier for custom gestures recorded in the browser.
/* global ort */

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);

export class OnnxModel {
  constructor(entry) {
    this.entry = entry; // from models/manifest.json
    this.session = null;
    this.loading = null;
    this.lastMs = 0;
  }

  get id() { return this.entry.id; }
  get name() { return this.entry.name; }
  get classes() { return this.entry.classes; }
  get spec() { return this.entry.input; }

  load() {
    if (!this.loading) {
      this.loading = ort.InferenceSession.create(this.entry.file, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }).then((s) => { this.session = s; return this; });
    }
    return this.loading;
  }

  /** input: {data, dims} from vision.makeInput. Returns Float32Array of class probabilities. */
  async predict(input) {
    await this.load();
    const t = performance.now();
    const tensor = new ort.Tensor("float32", input.data, input.dims);
    const out = await this.session.run({ [this.entry.input_name]: tensor });
    this.lastMs = performance.now() - t;
    return Float32Array.from(out[this.session.outputNames[0]].data);
  }
}

/**
 * k-nearest-neighbours over landmark features. Each custom gesture is a set of
 * recorded samples; a new hand is labelled by distance-weighted vote of its k
 * closest samples. No training step, so adding/removing gestures is instant.
 */
export function knnPredict(feature, gestures, k = 5, maxDistance = 0.25) {
  const usable = gestures.filter((g) => g.samples.length > 0);
  const probs = new Float32Array(gestures.length);
  if (!usable.length) return { probs, index: -1, distance: Infinity, unknown: true };

  const neighbours = [];
  gestures.forEach((g, gi) => {
    for (const s of g.samples) {
      let d = 0;
      for (let i = 0; i < 63; i++) { const t = s[i] - feature[i]; d += t * t; }
      neighbours.push([Math.sqrt(d / 21), gi]); // RMS distance per landmark
    }
  });
  neighbours.sort((a, b) => a[0] - b[0]);
  const top = neighbours.slice(0, Math.max(1, Math.min(k, neighbours.length)));
  let total = 0;
  for (const [d, gi] of top) {
    const wgt = 1 / (d + 1e-3);
    probs[gi] += wgt;
    total += wgt;
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= total;
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;
  const distance = top[0][0];
  const unknown = distance > maxDistance;
  if (unknown) probs.fill(0);
  return { probs, index: unknown ? -1 : index, distance, unknown };
}

/**
 * Turns noisy per-frame predictions into a stable command: majority vote over
 * the last `window` frames, mean confidence of the winner, and a hold timer.
 */
export class Smoother {
  constructor() { this.reset(); }

  reset() {
    this.history = [];
    this.stable = null;
    this.since = 0;
    this.fired = false;
  }

  push(probs, now, settings) {
    if (!probs || !probs.length) { this.reset(); return null; }
    this.history.push(probs);
    while (this.history.length > settings.window) this.history.shift();
    const votes = new Map();
    for (const p of this.history) {
      let best = -1, bi = -1;
      p.forEach((v, i) => { if (v > best) { best = v; bi = i; } });
      if (best > 0) votes.set(bi, (votes.get(bi) || 0) + 1);
    }
    if (!votes.size) { this.stable = null; return null; }
    const index = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const confidence = this.history.reduce((s, p) => s + p[index], 0) / this.history.length;
    if (index !== this.stable) {
      this.stable = index;
      this.since = now;
      this.fired = false;
    }
    const active = confidence >= settings.confidence && now - this.since >= settings.holdMs;
    return { index, confidence, active };
  }
}
