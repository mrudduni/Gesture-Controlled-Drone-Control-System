// Persistent app state: gesture → action mappings, custom gestures, settings.
// Everything lives in this browser's localStorage (nothing is uploaded).

const KEY = "gesture-drone:v1";

export const ACTIONS = [
  { id: "none", label: "Do nothing" },
  { id: "takeoff", label: "Take off", oneShot: true },
  { id: "land", label: "Land", oneShot: true },
  { id: "up", label: "Climb" },
  { id: "down", label: "Descend" },
  { id: "left", label: "Strafe left" },
  { id: "right", label: "Strafe right" },
  { id: "forward", label: "Forward" },
  { id: "backward", label: "Backward" },
  { id: "rotate_left", label: "Rotate left" },
  { id: "rotate_right", label: "Rotate right" },
  { id: "hover", label: "Hover" },
];
export const actionById = Object.fromEntries(ACTIONS.map((a) => [a.id, a]));

// What each class of the trained models does by default.
export const DEFAULT_BUILTIN_MAP = {
  up: "up",
  down: "down",
  left: "left",
  right: "right",
  hover: "hover",
  land: "land",
};

export const DEFAULT_SETTINGS = {
  confidence: 0.7, // mean probability over the smoothing window
  holdMs: 400, // a gesture must be stable this long before it acts
  window: 7, // frames in the majority vote
  speed: 60, // drone units per second
  knnK: 5,
  // Custom gestures: farther than this from every sample = "unknown". Calibrated on
  // HaGRID: same-gesture nearest distance p90 = 0.17, other-gesture p10 = 0.25.
  knnMaxDistance: 0.25,
  mirror: true,
};

const COLORS = ["#4ade80", "#60a5fa", "#f472b6", "#facc15", "#a78bfa", "#fb923c", "#2dd4bf", "#f87171"];

function defaults() {
  return {
    modelId: "cnn",
    builtinMap: { ...DEFAULT_BUILTIN_MAP },
    settings: { ...DEFAULT_SETTINGS },
    custom: [], // [{id, name, action, color, samples: number[][]}]
    starterLoaded: false, // starter gestures are added once, on the first visit
  };
}

/** Add the starter gestures that aren't already present (matched by id). Returns how many were added. */
export function addStarter(state, starter) {
  const have = new Set(state.custom.map((g) => g.id));
  let added = 0;
  for (const g of starter.custom || []) {
    if (have.has(g.id) || !Array.isArray(g.samples)) continue;
    state.custom.push({ id: g.id, name: g.name, action: actionById[g.action] ? g.action : "none",
      color: g.color || COLORS[state.custom.length % COLORS.length], samples: g.samples.filter((s) => s.length === 63) });
    added++;
  }
  state.starterLoaded = true;
  return added;
}

export function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return defaults();
    const saved = JSON.parse(raw);
    const d = defaults();
    return {
      ...d,
      ...saved,
      builtinMap: { ...d.builtinMap, ...(saved.builtinMap || {}) },
      settings: { ...d.settings, ...(saved.settings || {}) },
      custom: Array.isArray(saved.custom) ? saved.custom : [],
    };
  } catch {
    return defaults();
  }
}

export function save(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
    return true;
  } catch {
    return false; // private mode / storage full: the app still works, it just won't remember
  }
}

export function newGesture(state, name) {
  const id = `g${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const color = COLORS[state.custom.length % COLORS.length];
  const g = { id, name, action: "none", color, samples: [] };
  state.custom.push(g);
  return g;
}

export function exportJSON(state) {
  return JSON.stringify({ format: "gesture-drone", version: 1, builtinMap: state.builtinMap,
    settings: state.settings, custom: state.custom }, null, 1);
}

export function importJSON(state, text) {
  const data = JSON.parse(text);
  if (data.format !== "gesture-drone") throw new Error("Not a gesture-drone export file.");
  if (data.builtinMap) state.builtinMap = { ...DEFAULT_BUILTIN_MAP, ...data.builtinMap };
  if (data.settings) state.settings = { ...DEFAULT_SETTINGS, ...data.settings };
  if (Array.isArray(data.custom)) {
    state.custom = data.custom
      .filter((g) => g && typeof g.name === "string" && Array.isArray(g.samples))
      .map((g, i) => ({
        id: g.id || `imp${i}${Date.now().toString(36)}`,
        name: g.name.slice(0, 40),
        action: actionById[g.action] ? g.action : "none",
        color: typeof g.color === "string" ? g.color : COLORS[i % COLORS.length],
        samples: g.samples.filter((s) => Array.isArray(s) && s.length === 63),
      }));
  }
}
