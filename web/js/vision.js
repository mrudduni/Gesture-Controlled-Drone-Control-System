// Hand tracking (MediaPipe Tasks, in the browser) and the crop/preprocessing
// that turns a hand into model input. Mirrors gesture_common.py so the browser
// feeds the models exactly what they were trained on.

import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
export const PAD_RATIO = 0.25; // same as gesture_common.PAD_RATIO

let landmarker = null;
let mode = "VIDEO";

export async function initHands() {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
  mode = "VIDEO";
}

async function setMode(m) {
  if (mode !== m) {
    await landmarker.setOptions({ runningMode: m });
    mode = m;
  }
}

/** Landmarks (21 × {x,y,z}, normalised) of the first hand in a video frame, or null. */
export async function detectVideo(source, timestampMs) {
  await setMode("VIDEO");
  const r = landmarker.detectForVideo(source, timestampMs);
  return r.landmarks && r.landmarks.length ? r.landmarks[0] : null;
}

/** Same for a still image. */
export async function detectImage(source) {
  await setMode("IMAGE");
  const r = landmarker.detect(source);
  return r.landmarks && r.landmarks.length ? r.landmarks[0] : null;
}

// ---------------------------------------------------------------- boxes

function tightBox(lms, w, h) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of lms) {
    x0 = Math.min(x0, p.x * w); x1 = Math.max(x1, p.x * w);
    y0 = Math.min(y0, p.y * h); y1 = Math.max(y1, p.y * h);
  }
  return [x0, y0, x1, y1];
}

/** Square box around the hand with PAD_RATIO padding (may extend past the frame). */
export function paddedBox(lms, w, h) {
  const [x0, y0, x1, y1] = tightBox(lms, w, h);
  const side = Math.max(x1 - x0, y1 - y0) * (1 + 2 * PAD_RATIO);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2].map(Math.round);
}

/**
 * The original model's crop (old Data_Collection.py): 40 px margin at 640×480,
 * squared, then shifted/clamped to stay inside the frame.
 */
export function legacyBox(lms, w, h) {
  const margin = 40 * (w / 640);
  let [x0, y0, x1, y1] = tightBox(lms, w, h);
  x0 = Math.max(0, Math.floor(x0 - margin)); y0 = Math.max(0, Math.floor(y0 - margin));
  x1 = Math.min(w, Math.floor(x1 + margin)); y1 = Math.min(h, Math.floor(y1 + margin));
  const bw = x1 - x0, bh = y1 - y0, size = Math.max(bw, bh);
  const cx = x0 + Math.floor(bw / 2), cy = y0 + Math.floor(bh / 2);
  let sx0 = cx - Math.floor(size / 2), sy0 = cy - Math.floor(size / 2);
  let sx1 = cx + Math.floor(size / 2), sy1 = cy + Math.floor(size / 2);
  if (sx0 < 0) { sx1 -= sx0; sx0 = 0; }
  if (sy0 < 0) { sy1 -= sy0; sy0 = 0; }
  if (sx1 > w) { sx0 -= sx1 - w; sx1 = w; }
  if (sy1 > h) { sy0 -= sy1 - h; sy1 = h; }
  return [Math.max(0, sx0), Math.max(0, sy0), Math.min(w, sx1), Math.min(h, sy1)];
}

// ---------------------------------------------------------------- drawing

export function drawHand(ctx, lms, w, h, { color = "#4ade80", lineWidth = 2, radius = 3 } = {}) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (const { start, end } of HAND_CONNECTIONS) {
    ctx.moveTo(lms[start].x * w, lms[start].y * h);
    ctx.lineTo(lms[end].x * w, lms[end].y * h);
  }
  ctx.stroke();
  ctx.fillStyle = "#fff";
  for (const p of lms) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/** The skeleton exactly as the old collection script burned it into its training images. */
function drawLegacySkeleton(ctx, lms, w, h) {
  const s = w / 640;
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgb(0,255,255)"; // BGR (255,255,0)
  ctx.lineWidth = 2 * s;
  ctx.beginPath();
  for (const { start, end } of HAND_CONNECTIONS) {
    ctx.moveTo(lms[start].x * w, lms[start].y * h);
    ctx.lineTo(lms[end].x * w, lms[end].y * h);
  }
  ctx.stroke();
  for (const p of lms) {
    const x = p.x * w, y = p.y * h;
    ctx.lineWidth = 2 * s;
    ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, 3 * s, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = "rgb(0,255,0)";
    ctx.beginPath(); ctx.arc(x, y, 2 * s, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------- crops

const scratch = document.createElement("canvas");

/** Draw box from src into a size×size canvas; anything outside src stays black. */
function cropInto(out, src, srcW, srcH, box, size) {
  out.width = size; out.height = size;
  const ctx = out.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  const [x0, y0, x1, y1] = box;
  const bw = x1 - x0, bh = y1 - y0;
  if (bw <= 0 || bh <= 0) return null;
  // Clip the source rect to the image and move the destination rect to match.
  const cx0 = Math.max(0, x0), cy0 = Math.max(0, y0);
  const cx1 = Math.min(srcW, x1), cy1 = Math.min(srcH, y1);
  if (cx1 <= cx0 || cy1 <= cy0) return null;
  const sx = size / bw, sy = size / bh;
  ctx.drawImage(src, cx0, cy0, cx1 - cx0, cy1 - cy0,
    (cx0 - x0) * sx, (cy0 - y0) * sy, (cx1 - cx0) * sx, (cy1 - cy0) * sy);
  return ctx;
}

/**
 * Model input for a hand in frame (a canvas holding the clean, mirrored frame).
 * Returns { data: Float32Array (NHWC), dims, preview: canvas } or null.
 */
export function makeInput(frame, lms, spec, preview = document.createElement("canvas")) {
  const w = frame.width, h = frame.height;
  let src = frame, box;
  if (spec.crop === "legacy") {
    // Re-create the old pipeline: skeleton drawn on the frame, then cropped.
    scratch.width = w; scratch.height = h;
    const sctx = scratch.getContext("2d");
    sctx.drawImage(frame, 0, 0);
    drawLegacySkeleton(sctx, lms, w, h);
    src = scratch;
    box = legacyBox(lms, w, h);
  } else {
    box = paddedBox(lms, w, h);
  }
  const size = spec.size;
  const ctx = cropInto(preview, src, w, h, box, size);
  if (!ctx) return null;
  const px = ctx.getImageData(0, 0, size, size).data;
  const ch = spec.channels;
  const data = new Float32Array(size * size * ch);
  const k = spec.scale === "0-1" ? 1 / 255 : 1;
  for (let i = 0, j = 0; i < px.length; i += 4) {
    if (ch === 1) {
      data[j++] = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) * k;
    } else {
      data[j++] = px[i] * k; data[j++] = px[i + 1] * k; data[j++] = px[i + 2] * k;
    }
  }
  return { data, dims: [1, size, size, ch], preview, box };
}

/**
 * 63-number hand-shape feature for custom gestures: landmarks relative to the
 * wrist, in pixels, scaled by hand size. Keeps direction (so "point left" and
 * "point right" differ) but ignores where the hand is and how big it looks.
 */
export function landmarkFeature(lms, w, h) {
  const ox = lms[0].x * w, oy = lms[0].y * h, oz = lms[0].z * w;
  let scale = 1e-6;
  for (const p of lms) scale = Math.max(scale, Math.hypot(p.x * w - ox, p.y * h - oy));
  const f = new Array(63);
  lms.forEach((p, i) => {
    f[i * 3] = (p.x * w - ox) / scale;
    f[i * 3 + 1] = (p.y * h - oy) / scale;
    f[i * 3 + 2] = (p.z * w - oz) / scale;
  });
  return f;
}
