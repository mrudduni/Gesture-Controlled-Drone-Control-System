// Records a demo of the web app: detection + the drone flying in 3D.
//
// Serves web/, opens it in headless Edge/Chrome, replaces the webcam with a
// stream that plays the HaGRID sample photos in sequence (thumbs up -> take off
// and climb, point right/left, thumbs down, open palm, fist -> land), and saves
// screencast frames + timestamps. scripts/make_demo_video.py turns them into an MP4.
//
//   node scripts/record_demo.mjs [outDir]          (Node 22+, no npm packages)
//
// Env: BROWSER=<path to msedge.exe/chrome.exe>, HEADLESS=0 to watch it run.

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { join, extname, resolve, dirname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const WEB = join(ROOT, "web");
const OUT = resolve(process.argv[2] || join(ROOT, "demo", "frames"));
const W = 1280, H = 720;

// [photo or null, seconds]. Photos are web/samples/<name>.jpg.
const SEQUENCE = [
  [null, 1.5],
  ["up", 2.7],     // hold thumbs up: takes off, then climbs
  ["right", 2.2],
  ["left", 3.2],
  ["right", 1.4],
  ["down", 1.3],
  ["hover", 2.2],
  ["land", 3.2],   // fist: lands
  [null, 1.5],
];
const LABELS = { up: "Thumbs up", down: "Thumbs down", left: "Point left", right: "Point right", hover: "Open palm", land: "Fist" };

// ------------------------------------------------------------------ static server
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".jpg": "image/jpeg", ".png": "image/png", ".onnx": "application/octet-stream", ".svg": "image/svg+xml" };
const server = createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  const file = join(WEB, p);
  if (!file.startsWith(WEB) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] || "application/octet-stream", "Cache-Control": "no-store" });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const PORT = server.address().port;

// ------------------------------------------------------------------ browser
const candidates = [process.env.BROWSER,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome", "/usr/bin/chromium"].filter(Boolean);
const browserPath = candidates.find((p) => existsSync(p));
if (!browserPath) throw new Error("No Chrome/Edge found; set BROWSER=/path/to/browser");

const profile = join(tmpdir(), `gesture-demo-${Date.now()}`);
const args = [
  `--remote-debugging-port=0`, `--user-data-dir=${profile}`, `--window-size=${W},${H}`,
  "--no-first-run", "--no-default-browser-check", "--autoplay-policy=no-user-gesture-required",
  "--use-fake-ui-for-media-stream", "--ignore-gpu-blocklist", "--enable-gpu-rasterization",
  "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
  "about:blank",
];
if (process.env.HEADLESS !== "0") args.unshift("--headless=new");
const browser = spawn(browserPath, args, { stdio: "ignore" });
// The browser writes "<port>\n<path>" here once DevTools is listening (more reliable
// than parsing stderr, which Windows builds don't always print).
const portFile = join(profile, "DevToolsActivePort");
let wsUrl = null;
for (let i = 0; i < 100 && !wsUrl; i++) {
  await sleep(200);
  if (existsSync(portFile)) {
    const [port, path] = readFileSync(portFile, "utf8").trim().split(/\r?\n/);
    if (port && path) wsUrl = `ws://127.0.0.1:${port}${path}`;
  }
}
if (!wsUrl) throw new Error("browser did not start (no DevToolsActivePort)");

// ------------------------------------------------------------------ CDP
const ws = new WebSocket(wsUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 1;
const pending = new Map();
const handlers = [];
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve: ok, reject: bad } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? bad(new Error(JSON.stringify(msg.error))) : ok(msg.result);
  } else if (msg.method) handlers.forEach((h) => h(msg));
});
const send = (method, params = {}, sessionId) => new Promise((ok, bad) => {
  const id = nextId++;
  pending.set(id, { resolve: ok, reject: bad });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
const page = (m, p) => send(m, p, sessionId);
const evaluate = async (expression) => {
  const r = await page("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
};

await page("Page.enable");
await page("Runtime.enable");
await page("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await page("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });

// Fake webcam: a canvas stream showing whichever photo window.__demoShow() picked.
// Photos are drawn mirrored because the app mirrors the camera like a selfie view.
await page("Page.addScriptToEvaluateOnNewDocument", { source: `
  (() => {
    const c = document.createElement('canvas'); c.width = 640; c.height = 480;
    const g = c.getContext('2d');
    const imgs = {}; let cur = null;
    window.__demoShow = async (name) => {
      if (name && !imgs[name]) { const im = new Image(); im.src = '/samples/' + name + '.jpg'; await im.decode(); imgs[name] = im; }
      cur = name ? imgs[name] : null;
    };
    (function draw() {
      g.fillStyle = '#101418'; g.fillRect(0, 0, 640, 480);
      if (cur) {
        const s = Math.min(640 / cur.width, 480 / cur.height), w = cur.width * s, h = cur.height * s;
        g.save(); g.translate(640, 0); g.scale(-1, 1); g.drawImage(cur, (640 - w) / 2, (480 - h) / 2, w, h); g.restore();
      }
      setTimeout(draw, 33);
    })();
    const stream = c.captureStream(30);
    navigator.mediaDevices.getUserMedia = async () => stream;
  })();` });

console.log(`Browser: ${browserPath}`);
await page("Page.navigate", { url: `http://127.0.0.1:${PORT}/` });

// Wait for the model and the 3D view
const t0 = Date.now();
for (;;) {
  const ready = await evaluate(`(() => {
    const s = document.querySelector('#status, .status')?.textContent || '';
    const c = document.getElementById('drone3d');
    return /ready/i.test(s) && c && !c.hidden && c.width > 300;
  })()`).catch(() => false);
  if (ready) break;
  if (Date.now() - t0 > 90000) throw new Error("app did not become ready");
  await sleep(500);
}
console.log("WebGL:", await evaluate(`(() => { const gl = document.createElement('canvas').getContext('webgl2');
  const d = gl && gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'none'; })()`));

// Caption showing which photo is being fed in, so the video is honest about its input.
await evaluate(`(() => {
  const el = document.createElement('div'); el.id = 'demoCaption';
  el.style.cssText = 'position:fixed;left:16px;bottom:14px;z-index:99;padding:7px 12px;border-radius:8px;' +
    'background:rgba(0,0,0,.72);color:#fff;font:600 14px system-ui,sans-serif;pointer-events:none';
  document.body.appendChild(el);
  window.__demoCaption = (t) => { el.textContent = t; };
  window.__demoCaption('Demo input: HaGRID photos played as the webcam');
  scrollTo(0, 0);
})()`);
await evaluate(`document.getElementById('cameraBtn').click()`);
await sleep(1500);

// ------------------------------------------------------------------ record
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
const frames = [];
handlers.push(async (msg) => {
  if (msg.method !== "Page.screencastFrame" || msg.sessionId !== sessionId) return;
  const { data, metadata, sessionId: frameSession } = msg.params;
  const name = `f${String(frames.length).padStart(5, "0")}.jpg`;
  writeFileSync(join(OUT, name), Buffer.from(data, "base64"));
  frames.push({ name, t: metadata.timestamp });
  page("Page.screencastFrameAck", { sessionId: frameSession }).catch(() => {});
});
await page("Page.startScreencast", { format: "jpeg", quality: 88, maxWidth: W, maxHeight: H, everyNthFrame: 1 });

const events = [];
const start = Date.now() / 1000;
for (const [photo, secs] of SEQUENCE) {
  await evaluate(`window.__demoShow(${JSON.stringify(photo)})`);
  await evaluate(`window.__demoCaption(${JSON.stringify(photo ? `Input: ${LABELS[photo]} (HaGRID photo)` : "Input: no hand")})`);
  events.push({ t: Date.now() / 1000 - start, photo });
  const until = Date.now() + secs * 1000;
  while (Date.now() < until) {
    const s = await evaluate(`[document.getElementById('telemetry').innerText.replace(/\\n/g,' '), document.getElementById('gestureAction').textContent]`);
    process.stdout.write(`\r${(Date.now() / 1000 - start).toFixed(1).padStart(5)}s  ${String(photo).padEnd(6)} ${s[0]} | ${s[1]}`.padEnd(150).slice(0, 150));
    await sleep(250);
  }
}
process.stdout.write("\n");
await page("Page.stopScreencast");
await sleep(500);

const fps = await evaluate(`new Promise(r => { let n = 0; const t = performance.now(); (function f() { n++; performance.now() - t < 1000 ? requestAnimationFrame(f) : r(n); })(); })`);
writeFileSync(join(OUT, "frames.json"), JSON.stringify({ width: W, height: H, frames, events, pageFps: fps }, null, 1));
console.log(`Saved ${frames.length} frames to ${OUT} (page ran at ~${fps} fps)`);

ws.close();
browser.kill();
server.close();
setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch {} process.exit(0); }, 1000);

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
