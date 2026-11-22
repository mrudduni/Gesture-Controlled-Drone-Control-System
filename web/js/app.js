import { initHands, detectVideo, detectImage, makeInput, drawHand, landmarkFeature } from "./vision.js";
import { OnnxModel, knnPredict, Smoother } from "./classifiers.js";
import { Drone, renderDrone } from "./drone.js";
import * as store from "./store.js";

const $ = (id) => document.getElementById(id);
const BUILTIN_CLASSES = ["up", "down", "left", "right", "hover", "land"];
const POSES = {
  up: "Thumbs up",
  down: "Thumbs down",
  left: "Index finger pointing left",
  right: "Index finger pointing right",
  hover: "Open palm, fingers spread",
  land: "Fist",
};
const RECORD_COUNT = 30;
const RECORD_INTERVAL_MS = 90;

const state = store.load();
const drone = new Drone();
const smoother = new Smoother();
const models = new Map(); // id -> OnnxModel, in manifest order
let manifest = { models: [] };

const video = $("video");
const view = $("view");
const frame = document.createElement("canvas"); // clean mirrored frame (what the models see)
const cropCanvas = $("cropCanvas");

let handsReady = false;
let cameraOn = false;
let stream = null;
let lastTick = performance.now();
let lastCompare = 0;
let busy = false;
let recording = null; // {gesture, until, startAt, lastAt}
let lastResult = null; // for UI
let stillHand = null; // landmarks from the last still photo (camera off)
const compareResults = new Map();

// ------------------------------------------------------------------ helpers

function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), ms);
}

function setStatus(text, kind = "") {
  $("statusText").textContent = text;
  $("status").className = `status ${kind}`;
}

function persist() {
  if (!store.save(state)) toast("Couldn't save to this browser's storage — changes last until you close the tab.");
}

const actionOptions = (selected) =>
  store.ACTIONS.map((a) => `<option value="${a.id}" ${a.id === selected ? "selected" : ""}>${a.label}</option>`).join("");

/** Two-step button: first click arms it ("Sure?"), second click within 3 s runs fn. */
function confirmClick(btn, fn) {
  btn.addEventListener("click", () => {
    if (btn.dataset.armed) {
      clearTimeout(+btn.dataset.timer);
      delete btn.dataset.armed;
      fn();
      return;
    }
    const label = btn.textContent;
    btn.dataset.armed = "1";
    btn.textContent = "Sure?";
    btn.dataset.timer = setTimeout(() => {
      delete btn.dataset.armed;
      btn.textContent = label;
    }, 3000);
  });
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

let theme = {};
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  theme = {
    surface2: v("--surface-2"), grid: v("--grid"), border: v("--border"), text: v("--text"),
    muted: v("--muted"), accent: v("--accent"), drone: v("--drone"), danger: v("--danger"),
  };
}
readTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTheme);

// ------------------------------------------------------------------ current classifier

const isCustom = () => state.modelId === "custom";
const currentModel = () => models.get(state.modelId) || null;

function currentClasses() {
  return isCustom() ? state.custom.map((g) => g.name) : currentModel()?.classes || [];
}

function actionFor(index) {
  if (index < 0) return "none";
  if (isCustom()) return state.custom[index]?.action || "none";
  const cls = currentModel()?.classes[index];
  return state.builtinMap[cls] || "none";
}

async function classify(lms) {
  const w = frame.width, h = frame.height;
  if (isCustom()) {
    const r = knnPredict(landmarkFeature(lms, w, h), state.custom, state.settings.knnK, state.settings.knnMaxDistance);
    // still show what the model crop would be, for orientation
    makeInput(frame, lms, { size: 128, channels: 3, scale: "0-255", crop: "padded" }, cropCanvas);
    return { probs: r.probs, unknown: r.unknown, distance: r.distance };
  }
  const model = currentModel();
  if (!model) return null;
  const input = makeInput(frame, lms, model.spec, cropCanvas);
  if (!input) return null;
  return { probs: await model.predict(input), unknown: false, box: input.box };
}

async function runCompare(lms) {
  for (const model of models.values()) {
    const input = makeInput(frame, lms, model.spec);
    if (!input) continue;
    try {
      const probs = await model.predict(input);
      let bi = 0;
      probs.forEach((p, i) => { if (p > probs[bi]) bi = i; });
      compareResults.set(model.id, { cls: model.classes[bi], conf: probs[bi], ms: model.lastMs });
    } catch (e) {
      compareResults.set(model.id, { error: e.message });
    }
  }
  renderCompare();
}

// ------------------------------------------------------------------ per-frame processing

async function processHand(lms, now, live) {
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;

  if (lms && recording && live) captureSample(lms, now);

  let result = null;
  if (lms) {
    const out = await classify(lms);
    if (out) {
      const s = live ? smoother.push(out.probs, now, state.settings) : stillResult(out.probs);
      result = { ...out, ...(s || {}), classes: currentClasses(), live };
    }
  } else {
    smoother.reset();
  }
  lastResult = result;

  if (live && result && result.active && result.index >= 0) {
    const action = actionFor(result.index);
    const def = store.actionById[action];
    if (def?.oneShot) {
      if (!smoother.fired) {
        smoother.fired = true;
        const msg = action === "takeoff" ? drone.takeoff() : drone.land();
        toast(msg, 1200);
      }
    } else if (action !== "none") {
      drone.apply(action, dt, state.settings.speed);
    }
  }

  const compareVisible = !document.querySelector('[data-body="compare"]').hidden;
  if (lms && compareVisible && (!live || now - lastCompare > 300)) {
    lastCompare = now;
    await runCompare(lms);
  }
  renderGesture();
}

function stillResult(probs) {
  let index = -1, best = 0;
  probs.forEach((p, i) => { if (p > best) { best = p; index = i; } });
  return { index, confidence: best, active: false };
}

function drawView(lms) {
  view.width = frame.width;
  view.height = frame.height;
  const ctx = view.getContext("2d");
  ctx.drawImage(frame, 0, 0);
  if (lms) {
    const w = frame.width, h = frame.height;
    const lw = Math.max(2, w / 400);
    drawHand(ctx, lms, w, h, { lineWidth: lw, radius: lw * 1.4 });
    const box = lastResult?.box;
    if (box) {
      const active = lastResult?.active;
      ctx.strokeStyle = active ? theme.accent : "#fbbf24";
      ctx.lineWidth = lw * 1.5;
      ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    }
  }
  $("cropFigure").hidden = !lms;
}

// ------------------------------------------------------------------ camera

async function startCamera() {
  if (!handsReady) return toast("Hand tracker is still loading…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
  } catch (e) {
    toast(e.name === "NotAllowedError" ? "Camera permission was denied." : `Camera error: ${e.message}`, 4000);
    return;
  }
  video.srcObject = stream;
  await video.play();
  cameraOn = true;
  stillHand = null;
  clearSampleSelection();
  $("cameraBtn").textContent = "Stop camera";
  $("stageEmpty").hidden = true;
  smoother.reset();
  requestAnimationFrame(cameraTick);
}

function stopCamera() {
  cameraOn = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  stopRecording();
  $("cameraBtn").textContent = "Start camera";
}

async function cameraTick(now) {
  if (!cameraOn) return;
  if (!busy && video.readyState >= 2 && video.videoWidth) {
    busy = true;
    try {
      frame.width = video.videoWidth;
      frame.height = video.videoHeight;
      const fctx = frame.getContext("2d", { willReadFrequently: true });
      fctx.save();
      if (state.settings.mirror) { fctx.translate(frame.width, 0); fctx.scale(-1, 1); }
      fctx.drawImage(video, 0, 0);
      fctx.restore();
      const lms = await detectVideo(frame, now);
      await processHand(lms, now, true);
      drawView(lms);
    } catch (e) {
      console.error(e);
    } finally {
      busy = false;
    }
  }
  requestAnimationFrame(cameraTick);
}

// ------------------------------------------------------------------ still images

async function runOnImage(img) {
  if (!handsReady) return toast("Hand tracker is still loading…");
  if (cameraOn) stopCamera();
  const scale = Math.min(1, 1280 / Math.max(img.naturalWidth, img.naturalHeight));
  frame.width = Math.round(img.naturalWidth * scale);
  frame.height = Math.round(img.naturalHeight * scale);
  frame.getContext("2d").drawImage(img, 0, 0, frame.width, frame.height);
  $("stageEmpty").hidden = true;
  const lms = await detectImage(frame);
  stillHand = lms;
  smoother.reset();
  await processHand(lms, performance.now(), false);
  drawView(lms);
  renderCustom(); // Record buttons become "+ Add photo"
  if (!lms) toast("No hand found in that image.");
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = src;
  });
}

function clearSampleSelection() {
  document.querySelectorAll(".sample-row button").forEach((b) => b.classList.remove("active"));
}

async function setupSamples() {
  try {
    const list = await (await fetch("samples/samples.json")).json();
    const row = $("sampleRow");
    for (const s of list) {
      const b = document.createElement("button");
      b.title = `${s.label} (${s.pose})`;
      b.innerHTML = `<img src="samples/${s.file}" alt="${escapeHtml(s.pose)}" loading="lazy" />`;
      b.onclick = async () => {
        clearSampleSelection();
        b.classList.add("active");
        await runOnImage(await loadImage(`samples/${s.file}`));
      };
      row.appendChild(b);
    }
  } catch {
    $("sampleRow").closest(".samples").hidden = true;
  }
}

// ------------------------------------------------------------------ recording custom samples

function startRecording(gesture) {
  if (!cameraOn) {
    // From a still photo: add that one hand as a sample.
    if (stillHand) {
      gesture.samples.push(landmarkFeature(stillHand, frame.width, frame.height).map((v) => +v.toFixed(4)));
      persist();
      renderCustom();
      return toast(`Added 1 sample from the photo to “${gesture.name}”.`);
    }
    return toast("Start the camera (or open a photo with a hand) to record samples.");
  }
  if (recording) stopRecording();
  const startAt = performance.now() + 2000; // 2 s to get the pose ready
  recording = { gesture, startAt, lastAt: 0, got: 0 };
  renderCustom();
  updateRecordBanner(performance.now());
}

function stopRecording() {
  if (!recording) return;
  recording = null;
  $("recordBanner").hidden = true;
  persist();
  renderCustom();
}

function captureSample(lms, now) {
  updateRecordBanner(now);
  if (now < recording.startAt || now - recording.lastAt < RECORD_INTERVAL_MS) return;
  recording.lastAt = now;
  recording.gesture.samples.push(landmarkFeature(lms, frame.width, frame.height).map((v) => +v.toFixed(4)));
  recording.got += 1;
  if (recording.got >= RECORD_COUNT) {
    const name = recording.gesture.name;
    stopRecording();
    toast(`Recorded ${RECORD_COUNT} samples for “${name}”.`);
  } else {
    renderCustom();
  }
}

function updateRecordBanner(now) {
  const b = $("recordBanner");
  if (!recording) { b.hidden = true; return; }
  b.hidden = false;
  const wait = Math.ceil((recording.startAt - now) / 1000);
  b.textContent = wait > 0
    ? `Get ready: “${recording.gesture.name}” in ${wait}…`
    : `Recording “${recording.gesture.name}” ${recording.got}/${RECORD_COUNT}`;
}

// ------------------------------------------------------------------ rendering: gesture card

function renderGesture() {
  const r = lastResult;
  const card = $("gestureCard");
  const probsEl = $("probs");
  if (!r) {
    card.classList.remove("active");
    $("gestureName").textContent = "—";
    $("gestureConf").textContent = "";
    $("gestureAction").textContent = isCustom() && !state.custom.some((g) => g.samples.length)
      ? "No custom gestures yet — add some in the Gestures tab."
      : "No hand";
    probsEl.innerHTML = "";
    return;
  }
  if (isCustom() && !state.custom.some((g) => g.samples.length)) {
    card.classList.remove("active");
    $("gestureName").textContent = "—";
    $("gestureConf").textContent = "";
    $("gestureAction").textContent = "No custom gestures yet — add some in the Gestures tab.";
    probsEl.innerHTML = "";
    return;
  }
  const name = r.index >= 0 ? r.classes[r.index] : r.unknown ? "unknown" : "—";
  $("gestureName").textContent = name;
  $("gestureConf").textContent = r.index >= 0 ? `${Math.round(r.confidence * 100)}%` : "";
  const action = actionFor(r.index);
  const label = store.actionById[action]?.label || "Do nothing";
  let text = r.unknown ? "Doesn't match any custom gesture" : `→ ${label}`;
  if (!r.live) {
    if (!r.unknown) text += " (photo: not sent to the drone)";
  } else if (r.index >= 0 && !r.active && action !== "none") {
    text += r.confidence < state.settings.confidence ? " (low confidence)" : " (hold…)";
  }
  if (r.index >= 0 && r.active && !drone.flying && action !== "takeoff" && action !== "none") {
    text += " — take off first";
  }
  $("gestureAction").textContent = text;
  card.classList.toggle("active", !!r.active && action !== "none");

  const order = [...r.probs.keys()].sort((a, b) => r.probs[b] - r.probs[a]).slice(0, 4);
  probsEl.innerHTML = order
    .map((i, k) => `<div class="prob ${k === 0 && r.index >= 0 ? "top" : ""}"><span>${escapeHtml(r.classes[i] ?? "?")}</span>`
      + `<div class="bar"><i style="width:${(r.probs[i] * 100).toFixed(1)}%"></i></div>`
      + `<span>${Math.round(r.probs[i] * 100)}%</span></div>`)
    .join("");
}

// ------------------------------------------------------------------ rendering: drone

function renderTelemetry() {
  $("telemetry").innerHTML = [
    ["Status", drone.flying ? "Flying" : "Landed"],
    ["Altitude", Math.round(drone.y)],
    ["Position", `${Math.round(drone.x)}, ${Math.round(-drone.z)}`],
    ["Heading", `${Math.round(drone.heading)}°`],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
}

function droneLoop() {
  renderDrone($("droneCanvas"), drone, theme);
  renderTelemetry();
  requestAnimationFrame(droneLoop);
}

// ------------------------------------------------------------------ rendering: model switch

function renderModelSwitch() {
  const nav = $("modelSwitch");
  const entries = [...models.values()].map((m, i) => ({ id: m.id, label: m.name, num: i + 1 }));
  entries.push({ id: "custom", label: "Custom", num: "★" });
  nav.innerHTML = entries
    .map((e) => `<button data-model="${e.id}" class="${e.id === state.modelId ? "active" : ""}" title="${
      e.id === "custom" ? "Your own gestures (nearest-neighbour on hand landmarks)" : escapeHtml(models.get(e.id).entry.description)
    }"><span class="num">${e.num}</span>${escapeHtml(e.label)}</button>`)
    .join("");
  nav.querySelectorAll("button").forEach((b) => (b.onclick = () => switchModel(b.dataset.model)));
}

async function switchModel(id) {
  state.modelId = id;
  persist();
  smoother.reset();
  lastResult = null;
  renderModelSwitch();
  renderGesturesTab();
  renderCompare();
  if (id !== "custom") {
    const m = models.get(id);
    setStatus(`Loading ${m.name}…`);
    try {
      await m.load();
      setStatus(`${m.name} ready`, "ok");
    } catch (e) {
      setStatus(`Couldn't load ${m.name}`, "err");
      console.error(e);
    }
  } else {
    const n = state.custom.filter((g) => g.samples.length).length;
    setStatus(n ? `Custom: ${n} gesture${n > 1 ? "s" : ""}` : "Custom: no gestures yet", n ? "ok" : "");
  }
  renderGesture();
}

// ------------------------------------------------------------------ rendering: gestures tab

function renderGesturesTab() {
  $("builtinSection").classList.toggle("dim", isCustom());
  $("builtinMap").innerHTML = BUILTIN_CLASSES.map((c) => `
    <div class="map-item">
      <div class="pose"><b>${c}</b><span>${POSES[c]}</span></div>
      <select data-class="${c}" aria-label="Action for ${c}">${actionOptions(state.builtinMap[c])}</select>
    </div>`).join("");
  $("builtinMap").querySelectorAll("select").forEach((s) => (s.onchange = () => {
    state.builtinMap[s.dataset.class] = s.value;
    persist();
  }));
  renderCustom();
}

function renderCustom() {
  const list = $("customList");
  if (!state.custom.length) {
    list.innerHTML = `<div class="empty">No custom gestures yet. Click <b>+ New gesture</b>, give it a name and an action, then record it.</div>`;
    return;
  }
  list.innerHTML = state.custom.map((g, i) => {
    const rec = recording?.gesture === g;
    const n = g.samples.length;
    return `
    <div class="custom-item" data-i="${i}">
      <span class="swatch" style="background:${g.color}"></span>
      <input type="text" value="${escapeHtml(g.name)}" maxlength="40" aria-label="Gesture name" data-role="name" />
      <select data-role="action" aria-label="Action">${actionOptions(g.action)}</select>
      <span class="count ${n < 10 ? "low" : ""}">${n} sample${n === 1 ? "" : "s"}</span>
      <div class="actions">
        <button class="btn small ${rec ? "rec" : ""}" data-role="record" title="Camera on: record ${RECORD_COUNT} samples. Photo open: add the photo's hand.">${rec ? "Stop" : !cameraOn && stillHand ? "+ Add photo" : n ? `+ Record ${RECORD_COUNT}` : `Record ${RECORD_COUNT}`}</button>
        <button class="btn small ghost" data-role="clear" ${n ? "" : "disabled"}>Clear</button>
        <button class="btn small ghost danger" data-role="delete">Delete</button>
      </div>
    </div>`;
  }).join("");

  list.querySelectorAll(".custom-item").forEach((row) => {
    const g = state.custom[+row.dataset.i];
    row.querySelector('[data-role="name"]').onchange = (e) => {
      g.name = e.target.value.trim().slice(0, 40) || "Untitled";
      persist();
    };
    row.querySelector('[data-role="action"]').onchange = (e) => { g.action = e.target.value; persist(); };
    row.querySelector('[data-role="record"]').onclick = () => (recording?.gesture === g ? stopRecording() : startRecording(g));
    confirmClick(row.querySelector('[data-role="clear"]'), () => {
      g.samples = [];
      persist(); renderCustom(); smoother.reset();
    });
    confirmClick(row.querySelector('[data-role="delete"]'), () => {
      if (recording?.gesture === g) stopRecording();
      state.custom.splice(state.custom.indexOf(g), 1);
      persist(); renderCustom(); smoother.reset();
    });
  });
}

// ------------------------------------------------------------------ rendering: compare tab

function renderCompare() {
  const grid = $("compareGrid");
  if (!models.size) {
    grid.innerHTML = `<div class="empty">No trained models found in <code>models/manifest.json</code>.</div>`;
    return;
  }
  const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
  grid.innerHTML = [...models.values()].map((m, i) => {
    const r = compareResults.get(m.id);
    const e = m.entry;
    return `
    <div class="compare-card ${m.id === state.modelId ? "current" : ""}">
      <header><b>${i + 1}. ${escapeHtml(m.name)}</b><span class="muted small">${e.input.channels === 1 ? "gray" : "RGB"} ${e.input.size}px</span></header>
      <div class="pred">${r ? (r.error ? "error" : escapeHtml(r.cls)) : "—"}</div>
      <div class="muted small">${r && !r.error ? `${Math.round(r.conf * 100)}% · ${r.ms.toFixed(0)} ms` : "Show a hand to compare"}</div>
      <div class="meta">
        <span>Val accuracy</span><b>${pct(e.val_accuracy)}</b>
        <span>Your captures</span><b>${pct(e.val_accuracy_own)}${e.id === "legacy" ? "*" : ""}</b>
        <span>HaGRID (unseen)</span><b>${pct(e.val_accuracy_hagrid)}</b>
        <span>Parameters</span><b>${(e.params / 1e6).toFixed(2)}M</b>
        <span>Download</span><b>${e.size_mb} MB</b>
      </div>
    </div>`;
  }).join("");
}

// ------------------------------------------------------------------ settings tab

const SETTINGS_UI = [
  { key: "confidence", label: "Confidence threshold", min: 0.3, max: 0.95, step: 0.05, fmt: (v) => `${Math.round(v * 100)}%`,
    help: "Mean probability over the smoothing window needed before a gesture acts." },
  { key: "holdMs", label: "Hold time", min: 0, max: 1500, step: 50, fmt: (v) => `${v} ms`,
    help: "How long a gesture must stay the same before it acts." },
  { key: "window", label: "Smoothing window", min: 1, max: 15, step: 1, fmt: (v) => `${v} frames`,
    help: "Frames in the majority vote. Higher = steadier but slower to react." },
  { key: "speed", label: "Drone speed", min: 10, max: 150, step: 5, fmt: (v) => `${v} u/s` },
  { key: "knnK", label: "Custom: neighbours (k)", min: 1, max: 15, step: 1, fmt: (v) => `${v}`,
    help: "How many recorded samples vote on each frame." },
  { key: "knnMaxDistance", label: "Custom: match distance", min: 0.05, max: 0.8, step: 0.01, fmt: (v) => v.toFixed(2),
    help: "Hands farther than this from every recorded sample count as “unknown”. Raise it if your gestures often show as unknown." },
];

function renderSettings() {
  const el = $("settings");
  el.innerHTML = SETTINGS_UI.map((s) => `
    <div class="setting">
      <label for="set-${s.key}">${s.label}<span id="val-${s.key}">${s.fmt(state.settings[s.key])}</span></label>
      <input id="set-${s.key}" type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${state.settings[s.key]}" />
      ${s.help ? `<p>${s.help}</p>` : ""}
    </div>`).join("") + `
    <div class="setting check">
      <label><input id="set-mirror" type="checkbox" ${state.settings.mirror ? "checked" : ""} /> Mirror camera</label>
      <p>On = like a mirror (the models were trained this way: point to your left → “left”).</p>
    </div>`;
  for (const s of SETTINGS_UI) {
    $(`set-${s.key}`).oninput = (e) => {
      state.settings[s.key] = +e.target.value;
      $(`val-${s.key}`).textContent = s.fmt(state.settings[s.key]);
      persist();
    };
  }
  $("set-mirror").onchange = (e) => { state.settings.mirror = e.target.checked; persist(); };
}

// ------------------------------------------------------------------ wiring

function wire() {
  $("cameraBtn").onclick = () => (cameraOn ? stopCamera() : startCamera());
  $("imageInput").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    try { clearSampleSelection(); await runOnImage(await loadImage(url)); } finally { URL.revokeObjectURL(url); }
    e.target.value = "";
  };
  $("takeoffBtn").onclick = () => toast(drone.takeoff(), 1200);
  $("landBtn").onclick = () => toast(drone.land(), 1200);
  $("resetBtn").onclick = () => drone.reset();
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === "t") toast(drone.takeoff(), 1200);
    else if (k === "l") toast(drone.land(), 1200);
    else if (k === "r") drone.reset();
    else if (["1", "2", "3"].includes(k)) {
      const m = [...models.keys()][+k - 1];
      if (m) switchModel(m);
    } else if (k === "4" || k === "c") switchModel("custom");
  });

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".tab-body").forEach((b) => (b.hidden = b.dataset.body !== t.dataset.tab));
    if (t.dataset.tab === "compare") renderCompare();
  }));

  $("addGestureBtn").onclick = () => {
    store.newGesture(state, `Gesture ${state.custom.length + 1}`);
    persist();
    renderCustom();
    const inputs = $("customList").querySelectorAll('[data-role="name"]');
    const last = inputs[inputs.length - 1];
    last?.focus();
    last?.select();
  };
  $("starterBtn").onclick = async () => {
    const n = await loadStarter();
    toast(n ? `Added ${n} starter gesture${n > 1 ? "s" : ""}.` : "All starter gestures are already in your list.");
  };
  $("exportBtn").onclick = () => {
    const blob = new Blob([store.exportJSON(state)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "gesture-drone-gestures.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("importInput").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      store.importJSON(state, await file.text());
      persist();
      renderGesturesTab(); renderSettings();
      toast(`Imported ${state.custom.length} custom gesture(s).`);
    } catch (err) {
      toast(`Import failed: ${err.message}`, 4000);
    }
    e.target.value = "";
  };
  $("resetSettingsBtn").onclick = () => {
    state.settings = { ...store.DEFAULT_SETTINGS };
    persist(); renderSettings();
  };
  confirmClick($("clearAllBtn"), () => {
    stopRecording();
    state.custom = [];
    persist(); renderCustom(); smoother.reset();
    toast("All custom gestures deleted.");
  });
}

async function loadStarter() {
  try {
    const starter = await (await fetch("starter-gestures.json")).json();
    const n = store.addStarter(state, starter);
    persist();
    renderCustom();
    return n;
  } catch {
    toast("Couldn't load the starter gestures.");
    return 0;
  }
}

async function main() {
  wire();
  if (!state.starterLoaded && !state.custom.length) await loadStarter();
  renderSettings();
  renderGesturesTab();
  requestAnimationFrame(droneLoop);
  setupSamples();

  try {
    manifest = await (await fetch("models/manifest.json")).json();
  } catch {
    manifest = { models: [] };
  }
  for (const entry of manifest.models) models.set(entry.id, new OnnxModel(entry));
  if (state.modelId !== "custom" && !models.has(state.modelId)) state.modelId = models.has("cnn") ? "cnn" : "custom";
  renderModelSwitch();
  renderCompare();

  setStatus("Loading hand tracker…");
  try {
    await initHands();
    handsReady = true;
  } catch (e) {
    console.error(e);
    setStatus("Hand tracker failed to load", "err");
    toast(`Couldn't load MediaPipe: ${e.message}`, 5000);
    return;
  }
  await switchModel(state.modelId);
}

main();
