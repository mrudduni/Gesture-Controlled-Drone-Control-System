// Simulated drone and its top-down view.
// x: east/west, z: north/south (both -150..150), y: altitude 0..200, heading in degrees.

// phase: "landed" -> "takeoff" (climbing to cruiseAlt) -> "flying" -> "landing" (descending) -> "landed"
export class Drone {
  constructor() {
    this.bound = 150;
    this.maxAlt = 200;
    this.minFlyAlt = 20;
    this.cruiseAlt = 50;
    this.climbRate = 60; // units/s during takeoff
    this.sinkRate = 70; // units/s during landing
    this.reset(true);
  }

  /** True from the moment of takeoff until touchdown. */
  get flying() { return this.phase !== "landed"; }

  /** True only once airborne and accepting movement commands. */
  get controllable() { return this.phase === "flying"; }

  get statusText() {
    return { landed: "Landed", takeoff: "Taking off", flying: "Flying", landing: "Landing" }[this.phase];
  }

  reset(full = false) {
    this.x = 0; this.z = 0; this.heading = 0;
    if (full || this.phase === "landed" || this.phase === undefined) { this.y = 0; this.phase = "landed"; }
    else { this.y = this.cruiseAlt; this.phase = "flying"; }
    this.trail = [];
    this.lastAction = "none";
  }

  takeoff() {
    if (this.phase === "takeoff") return "Taking off…";
    if (this.phase === "flying") return "Already flying";
    this.phase = "takeoff"; // also aborts a landing in progress
    return "Taking off";
  }

  land() {
    if (this.phase === "landed") return "Already landed";
    if (this.phase === "landing") return "Landing…";
    this.phase = "landing";
    return "Landing";
  }

  /** Advance takeoff/landing animations. Call every frame. */
  update(dt) {
    if (this.phase === "takeoff") {
      this.y = Math.min(this.cruiseAlt, this.y + this.climbRate * dt);
      if (this.y >= this.cruiseAlt) this.phase = "flying";
    } else if (this.phase === "landing") {
      this.y = Math.max(0, this.y - this.sinkRate * dt);
      if (this.y <= 0) this.phase = "landed";
    }
  }

  /** Apply a continuous action for dt seconds at `speed` units/s. */
  apply(action, dt, speed) {
    this.lastAction = action;
    if (!this.controllable) return;
    const rad = (this.heading * Math.PI) / 180;
    const fwd = [Math.sin(rad), -Math.cos(rad)]; // heading 0 = up on the map
    const right = [Math.cos(rad), Math.sin(rad)];
    const d = speed * dt;
    let dx = 0, dz = 0;
    switch (action) {
      case "up": this.y += d; break;
      case "down": this.y -= d; break;
      case "left": dx -= right[0] * d; dz -= right[1] * d; break;
      case "right": dx += right[0] * d; dz += right[1] * d; break;
      case "forward": dx += fwd[0] * d; dz += fwd[1] * d; break;
      case "backward": dx -= fwd[0] * d; dz -= fwd[1] * d; break;
      case "rotate_left": this.heading -= 90 * dt; break;
      case "rotate_right": this.heading += 90 * dt; break;
      default: break;
    }
    this.x = clamp(this.x + dx, -this.bound, this.bound);
    this.z = clamp(this.z + dz, -this.bound, this.bound);
    this.y = clamp(this.y, this.minFlyAlt, this.maxAlt);
    this.heading = ((this.heading % 360) + 360) % 360;
  }
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function renderDrone(canvas, drone, theme) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;
  ctx.clearRect(0, 0, W, H);

  const gaugeW = 46;
  const mapSize = Math.min(W - gaugeW - 24, H - 24);
  const mx = 12, my = (H - mapSize) / 2;
  const toPx = (v) => ((v + drone.bound) / (2 * drone.bound)) * mapSize;

  // Map + grid
  ctx.fillStyle = theme.surface2;
  roundRect(ctx, mx, my, mapSize, mapSize, 10); ctx.fill();
  ctx.strokeStyle = theme.grid; ctx.lineWidth = 1;
  for (let i = 1; i < 6; i++) {
    const p = (mapSize / 6) * i;
    line(ctx, mx + p, my, mx + p, my + mapSize);
    line(ctx, mx, my + p, mx + mapSize, my + p);
  }
  ctx.strokeStyle = theme.border;
  roundRect(ctx, mx, my, mapSize, mapSize, 10); ctx.stroke();
  ctx.fillStyle = theme.muted; ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("N", mx + mapSize / 2 - 3, my + 14);

  // Trail
  const px = mx + toPx(drone.x), pz = my + toPx(drone.z);
  if (drone.controllable) {
    const last = drone.trail[drone.trail.length - 1];
    if (!last || Math.hypot(last[0] - drone.x, last[1] - drone.z) > 2) drone.trail.push([drone.x, drone.z]);
    if (drone.trail.length > 80) drone.trail.shift();
  }
  drone.trail.forEach(([tx, tz], i) => {
    ctx.globalAlpha = (i + 1) / drone.trail.length * 0.6;
    ctx.fillStyle = theme.accent;
    ctx.beginPath(); ctx.arc(mx + toPx(tx), my + toPx(tz), 2, 0, Math.PI * 2); ctx.fill();
  });
  ctx.globalAlpha = 1;

  // Shadow grows apart from the drone with altitude
  const alt = drone.y / drone.maxAlt;
  const r = 13 + alt * 7;
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.beginPath(); ctx.ellipse(px + alt * 14, pz + alt * 18, r, r * 0.6, 0, 0, Math.PI * 2); ctx.fill();

  // Drone body
  ctx.save();
  ctx.translate(px, pz);
  ctx.rotate((drone.heading * Math.PI) / 180);
  const body = drone.flying ? theme.drone : theme.muted;
  ctx.strokeStyle = body; ctx.lineWidth = 3;
  line(ctx, -r, -r, r, r); line(ctx, r, -r, -r, r);
  for (const [dx, dy] of [[-r, -r], [r, -r], [-r, r], [r, r]]) {
    ctx.fillStyle = drone.flying ? theme.accent : theme.border;
    ctx.beginPath(); ctx.arc(dx, dy, 6, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = body;
  ctx.beginPath(); ctx.arc(0, 0, 7, 0, Math.PI * 2); ctx.fill();
  // Nose
  ctx.fillStyle = theme.danger;
  ctx.beginPath(); ctx.moveTo(0, -r - 6); ctx.lineTo(-5, -r + 3); ctx.lineTo(5, -r + 3); ctx.closePath(); ctx.fill();
  ctx.restore();

  // Altitude gauge
  const gx = mx + mapSize + 16, gy = my + 18, gh = mapSize - 36;
  ctx.fillStyle = theme.surface2;
  roundRect(ctx, gx, gy, 14, gh, 7); ctx.fill();
  const fill = gh * alt;
  if (fill > 0) {
    ctx.fillStyle = theme.accent;
    roundRect(ctx, gx, gy + gh - fill, 14, fill, 7); ctx.fill();
  }
  ctx.fillStyle = theme.muted; ctx.font = "11px system-ui, sans-serif";
  ctx.fillText("ALT", gx - 3, gy - 6);
  ctx.fillStyle = theme.text;
  ctx.fillText(`${Math.round(drone.y)}`, gx - 2, gy + gh + 16);
}

function line(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
