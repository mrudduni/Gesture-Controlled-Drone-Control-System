// 3D view of the simulated drone (three.js, loaded on first use).
// World units: 1 three.js unit = 10 drone units, so the 300x300 flying area is 30x30
// and the 200-unit altitude ceiling is 20 units up. Heading 0 points north (-z).

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
const S = 0.1; // drone units -> world units

export async function createDrone3D(canvas) {
  const THREE = await import(THREE_URL);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.3);
  sun.position.set(8, 20, 6);
  scene.add(sun);

  // ---------------------------------------------------------------- ground
  const half = 15;
  const groundMat = new THREE.MeshLambertMaterial({ color: 0x8fa58a });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), groundMat);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const grid = new THREE.GridHelper(2 * half, 12, 0x000000, 0x000000);
  grid.material.transparent = true; grid.material.opacity = 0.18;
  grid.position.y = 0.01;
  scene.add(grid);

  // Boundary of the flying area
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * half, 0.02, 2 * half)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
  edge.position.y = 0.02;
  scene.add(edge);

  // Landing pad with an H
  const pad = new THREE.Mesh(new THREE.CircleGeometry(1.6, 40), new THREE.MeshLambertMaterial({ color: 0x3b4450 }));
  pad.rotation.x = -Math.PI / 2; pad.position.y = 0.02;
  scene.add(pad);
  const padRing = new THREE.Mesh(new THREE.RingGeometry(1.35, 1.5, 40), new THREE.MeshBasicMaterial({ color: 0xfacc15 }));
  padRing.rotation.x = -Math.PI / 2; padRing.position.y = 0.03;
  scene.add(padRing);
  const hMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  for (const [w, d, x] of [[0.18, 1.2, -0.4], [0.18, 1.2, 0.4], [0.8, 0.18, 0]]) {
    const bar = new THREE.Mesh(new THREE.PlaneGeometry(w, d), hMat);
    bar.rotation.x = -Math.PI / 2; bar.position.set(x, 0.035, 0);
    scene.add(bar);
  }

  // Trees and a few blocks around the area, so movement and height read clearly.
  const trunkMat = new THREE.MeshLambertMaterial({ color: 0x7c5a3a });
  const leafMat = new THREE.MeshLambertMaterial({ color: 0x2f7d46 });
  const rand = mulberry32(7);
  for (let i = 0; i < 26; i++) {
    const a = rand() * Math.PI * 2, r = half + 1.5 + rand() * 9;
    const x = Math.cos(a) * r, z = Math.sin(a) * r, h = 1.5 + rand() * 2.5;
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.2, h * 0.4, 6), trunkMat);
    trunk.position.set(x, h * 0.2, z);
    const leaves = new THREE.Mesh(new THREE.ConeGeometry(h * 0.35, h * 0.8, 8), leafMat);
    leaves.position.set(x, h * 0.4 + h * 0.4, z);
    scene.add(trunk, leaves);
  }
  const blockMat = new THREE.MeshLambertMaterial({ color: 0xc9ced6 });
  for (const [x, z, w, h] of [[-10, -10, 2, 3], [10, -9, 1.5, 5], [-9, 10, 2.5, 2], [11, 11, 1.2, 4]]) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, w), blockMat);
    b.position.set(x, h / 2, z);
    scene.add(b);
  }

  // ---------------------------------------------------------------- drone
  const drone = new THREE.Group();      // position + heading
  const body = new THREE.Group();       // tilt
  drone.add(body);
  scene.add(drone);

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2b3440, roughness: 0.5, metalness: 0.3 });
  const accentMat = new THREE.MeshStandardMaterial({ color: 0x60a5fa, roughness: 0.4 });
  const hull = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.28, 1.1), bodyMat);
  hull.position.y = 0.3;
  const top = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.12, 0.7), accentMat);
  top.position.y = 0.5;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.35, 12), new THREE.MeshStandardMaterial({ color: 0xef4444 }));
  nose.rotation.x = -Math.PI / 2; nose.position.set(0, 0.3, -0.7);
  const camPod = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), new THREE.MeshStandardMaterial({ color: 0x111111 }));
  camPod.position.set(0, 0.12, -0.45);
  body.add(hull, top, nose, camPod);

  const rotors = [];
  const armMat = new THREE.MeshStandardMaterial({ color: 0x1f2630, roughness: 0.6 });
  const motorMat = new THREE.MeshStandardMaterial({ color: 0x9aa4b1, metalness: 0.6, roughness: 0.3 });
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0xdde3ea, transparent: true, opacity: 0.85 });
  const discMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.0, depthWrite: false });
  for (const [sx, sz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 1.15), armMat);
    arm.position.set(sx * 0.42, 0.32, sz * 0.42);
    arm.rotation.y = Math.atan2(sx, sz);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.18, 12), motorMat);
    motor.position.set(sx * 0.85, 0.38, sz * 0.85);
    const rotor = new THREE.Group();
    rotor.position.set(sx * 0.85, 0.5, sz * 0.85);
    for (const r of [0, Math.PI / 2]) {
      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.015, 0.08), bladeMat);
      blade.rotation.y = r;
      rotor.add(blade);
    }
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.46, 24), discMat);
    disc.rotation.x = -Math.PI / 2;
    rotor.add(disc);
    rotor.userData.dir = sx * sz;
    rotors.push(rotor);
    // Leg
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.3, 6), armMat);
    leg.position.set(sx * 0.35, 0.12, sz * 0.4);
    body.add(arm, motor, rotor, leg);
  }
  // Status LED under the nose
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
  led.position.set(0, 0.3, -0.56);
  body.add(led);

  // Blob shadow (cheaper and steadier than shadow maps)
  const shadowTex = radialTexture(THREE);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 2.6),
    new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.04;
  scene.add(shadow);

  // Altitude line from the ground to the drone
  const altGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const altLine = new THREE.Line(altGeom, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.25, gapSize: 0.2, transparent: true, opacity: 0.7 }));
  scene.add(altLine);

  // Trail
  const TRAIL = 120;
  const trailPos = new Float32Array(TRAIL * 3);
  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  trailGeom.setDrawRange(0, 0);
  const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.8 }));
  scene.add(trailLine);
  let trailCount = 0;
  const lastTrail = new THREE.Vector3(1e9, 0, 0);

  // ---------------------------------------------------------------- state
  const camPos = new THREE.Vector3(0, 6, 10);
  const camTarget = new THREE.Vector3();
  const prev = new THREE.Vector3();
  const vel = new THREE.Vector3();
  let rotorAngle = 0, rotorSpeed = 0, hoverT = 0, first = true;
  let themeKey = "";

  function applyTheme(theme) {
    const key = JSON.stringify(theme);
    if (key === themeKey) return;
    themeKey = key;
    const bg = new THREE.Color(theme.dark ? 0x0f1a24 : 0xbfd9ee);
    scene.background = bg;
    scene.fog = new THREE.Fog(bg, 28, 70);
    groundMat.color.set(theme.dark ? 0x26372b : 0x9cb894);
    accentMat.color.set(theme.drone || "#60a5fa");
    trailLine.material.color.set(theme.drone || "#60a5fa");
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return false;
    const pr = renderer.getPixelRatio();
    if (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr)) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    return true;
  }

  /** Draw one frame. `sim` is the Drone from drone.js. */
  function render(sim, theme, dt) {
    if (!resize()) return;
    applyTheme(theme);
    dt = Math.min(0.5, Math.max(0, dt));

    // Position: gentle bob while airborne
    hoverT += dt;
    const airborne = sim.y > 0.5;
    const bob = airborne ? Math.sin(hoverT * 2.2) * 0.06 : 0;
    const target = new THREE.Vector3(sim.x * S, sim.y * S + bob, sim.z * S);
    if (first) { prev.copy(target); first = false; }
    drone.position.copy(target);
    drone.rotation.y = -(sim.heading * Math.PI) / 180;

    // Tilt toward the direction of travel (in the drone's own frame)
    const v = target.clone().sub(prev).divideScalar(Math.max(dt, 1e-3));
    vel.lerp(v, 0.15);
    prev.copy(target);
    const local = vel.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), -drone.rotation.y);
    body.rotation.x = clamp(local.z * 0.12, -0.35, 0.35);   // forward (-z) pitches nose down
    body.rotation.z = clamp(-local.x * 0.12, -0.35, 0.35);  // right rolls right

    // Rotors: stopped on the ground, spin up for takeoff, idle fast in flight
    const want = sim.phase === "landed" ? 0 : sim.phase === "takeoff" ? 55 : 40;
    rotorSpeed += (want - rotorSpeed) * Math.min(1, dt * 2.5);
    rotorAngle += rotorSpeed * dt;
    for (const r of rotors) r.rotation.y = rotorAngle * r.userData.dir;
    discMat.opacity = Math.min(0.35, rotorSpeed / 120);
    led.material.color.set(sim.phase === "flying" ? 0x22c55e : sim.phase === "landed" ? 0x64748b : 0xf59e0b);

    // Shadow fades and spreads with altitude
    const alt = sim.y / sim.maxAlt;
    shadow.position.set(target.x, 0.04, target.z);
    shadow.scale.setScalar(1 + alt * 1.2);
    shadow.material.opacity = 0.75 - alt * 0.5;

    // Altitude line
    const p = altGeom.attributes.position;
    p.setXYZ(0, target.x, 0.05, target.z);
    p.setXYZ(1, target.x, Math.max(0.05, target.y + 0.1), target.z);
    p.needsUpdate = true;
    altLine.computeLineDistances();
    altLine.visible = airborne;

    // Trail
    if (airborne && target.distanceTo(lastTrail) > 0.15) {
      if (trailCount === TRAIL) { trailPos.copyWithin(0, 3); trailCount--; }
      trailPos.set([target.x, target.y + 0.3, target.z], trailCount * 3);
      trailCount++;
      lastTrail.copy(target);
      trailGeom.attributes.position.needsUpdate = true;
      trailGeom.setDrawRange(0, trailCount);
    }

    // Chase camera: behind and above, following heading smoothly
    const back = new THREE.Vector3(0, 0, 1).applyAxisAngle(new THREE.Vector3(0, 1, 0), drone.rotation.y);
    const want3 = target.clone().add(back.multiplyScalar(7)).add(new THREE.Vector3(0, 2.6 + target.y * 0.08, 0));
    camPos.lerp(want3, Math.min(1, dt * 2.5));
    camTarget.lerp(target.clone().add(new THREE.Vector3(0, 0.4, 0)), Math.min(1, dt * 4));
    camera.position.copy(camPos);
    camera.lookAt(camTarget);

    renderer.render(scene, camera);
  }

  function resetTrail() { trailCount = 0; trailGeom.setDrawRange(0, 0); lastTrail.set(1e9, 0, 0); }

  return { render, resetTrail, dispose: () => renderer.dispose() };
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function radialTexture(THREE) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(0,0,0,0.65)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  return t;
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
