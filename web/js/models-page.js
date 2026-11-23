// Renders the layer-by-layer architecture tables on models.html and fills in
// live numbers (parameters, size, accuracy) from models/manifest.json.

// [type, layer, output shape, params, note]
const ARCH = {
  legacy: [
    ["input", "Input: grayscale, 0–1", "224×224×1", 0],
    ["conv", "Conv 3×3, 32 filters, ReLU", "222×222×32", 320],
    ["pool", "MaxPool 2×2", "111×111×32", 0],
    ["conv", "Conv 3×3, 64, ReLU", "109×109×64", 18496],
    ["pool", "MaxPool 2×2", "54×54×64", 0],
    ["conv", "Conv 3×3, 128, ReLU", "52×52×128", 73856],
    ["pool", "MaxPool 2×2", "26×26×128", 0],
    ["other", "Flatten", "86,528", 0],
    ["dense", "Dense 128, ReLU", "128", 11075712, "99% of all weights"],
    ["other", "Dropout 0.5", "128", 0],
    ["dense", "Dense 6, softmax", "6", 774],
  ],
  cnn: [
    ["input", "Input: RGB, 0–255", "128×128×3", 0],
    ["aug", "Random rotate ±15° · zoom · shift (training only)", "128×128×3", 0],
    ["other", "Rescale ÷255", "128×128×3", 0],
    ["conv", "Stem: Conv 3×3 stride 2, 32 → BN → ReLU", "64×64×32", 992],
    ["conv", "Block 1: 2× [Conv 3×3, 32 → BN → ReLU]", "64×64×32", 18688],
    ["pool", "MaxPool 2×2", "32×32×32", 0],
    ["conv", "Block 2: 2× [Conv 3×3, 64 → BN → ReLU]", "32×32×64", 55808],
    ["pool", "MaxPool 2×2", "16×16×64", 0],
    ["conv", "Block 3: 2× [Conv 3×3, 128 → BN → ReLU]", "16×16×128", 222208],
    ["pool", "MaxPool 2×2", "8×8×128", 0],
    ["conv", "Block 4: 2× [Conv 3×3, 192 → BN → ReLU]", "8×8×192", 554496],
    ["pool", "MaxPool 2×2", "4×4×192", 0],
    ["conv", "Conv 3×3, 256, ReLU", "4×4×256", 442624],
    ["pool", "Global average pooling", "256", 0],
    ["other", "Dropout 0.4", "256", 0],
    ["dense", "Dense 6, softmax", "6", 1542],
  ],
  mobilenetv2: [
    ["input", "Input: RGB, 0–255", "128×128×3", 0],
    ["aug", "Random rotate ±15° · zoom · shift (training only)", "128×128×3", 0],
    ["other", "Rescale → [−1, 1]", "128×128×3", 0],
    ["conv", "Conv 3×3 stride 2, 32 → BN → ReLU6", "64×64×32", 992, "frozen"],
    ["conv", "Stage 1: 1 inverted residual, 16 ch", "64×64×16", 992, "frozen"],
    ["conv", "Stage 2: 2 inverted residuals, 24 ch", "32×32×24", 15024, "frozen"],
    ["conv", "Stage 3: 3 inverted residuals, 32 ch", "16×16×32", 42000, "frozen"],
    ["conv", "Stage 4: 4 inverted residuals, 64 ch", "8×8×64", 189760, "frozen"],
    ["conv", "Stage 5: 3 inverted residuals, 96 ch", "8×8×96", 309888, "frozen"],
    ["conv", "Stage 6: 3 inverted residuals, 160 ch", "4×4×160", 806208, "fine-tuned"],
    ["conv", "Stage 7: 1 inverted residual, 320 ch", "4×4×320", 478400, "fine-tuned"],
    ["conv", "Conv 1×1, 1280 → BN → ReLU6", "4×4×1280", 414720, "fine-tuned"],
    ["pool", "Global average pooling", "1280", 0],
    ["other", "Dropout 0.3", "1280", 0],
    ["dense", "Dense 6, softmax (new head)", "6", 7686, "trained"],
  ],
  custom: [
    ["input", "Webcam frame", "H×W×3", null],
    ["other", "MediaPipe palm detector", "hand box", null, "Google, pretrained"],
    ["other", "MediaPipe hand-landmark model", "21 × (x, y, z)", null, "Google, pretrained"],
    ["other", "Subtract wrist, divide by hand size", "63 numbers", 0],
    ["dense", "Compare with every sample you recorded", "distance per sample", null, "your samples"],
    ["dense", "Vote of the 5 nearest (closer = stronger)", "score per gesture", 0],
    ["other", "Nearest > 0.25 → “unknown”", "gesture or unknown", 0],
  ],
};

const fmt = (n) => (n == null ? "—" : n === 0 ? "" : n.toLocaleString("en-US"));

function renderLayers(el, rows) {
  const total = rows.reduce((s, r) => s + (r[3] || 0), 0);
  const max = Math.max(...rows.map((r) => r[3] || 0), 1);
  el.innerHTML = `
    <div class="layer head"><span>Layer</span><span>Output</span><span>Weights</span></div>
    ${rows.map(([type, name, shape, params, note]) => `
      <div class="layer t-${type}">
        <span class="name">${name}${note ? ` <em class="note">${note}</em>` : ""}</span>
        <span class="shape">${shape}</span>
        <span class="params">${fmt(params)}${params ? `<i style="width:${Math.max(2, Math.sqrt(params / max) * 100)}%"></i>` : ""}</span>
      </div>`).join("")}
    ${total ? `<div class="layer total"><span>Total</span><span></span><span>${fmt(total)}</span></div>` : ""}`;
}

for (const el of document.querySelectorAll(".layers[data-arch]")) renderLayers(el, ARCH[el.dataset.arch]);

// Live numbers from the exported models.
try {
  const { models } = await (await fetch("models/manifest.json")).json();
  const byId = Object.fromEntries(models.map((m) => [m.id, m]));
  for (const td of document.querySelectorAll("td[data-m]")) {
    const m = byId[td.dataset.m];
    if (!m) continue;
    const k = td.dataset.k;
    if (k === "params") td.textContent = `${(m.params / 1e6).toFixed(2)}M`;
    else if (k === "size") td.textContent = `${m.size_mb} MB`;
    else if (k === "hagrid" && m.val_accuracy_hagrid != null) td.textContent = `${(m.val_accuracy_hagrid * 100).toFixed(1)}%`;
    else if (k === "own" && m.val_accuracy_own != null) td.textContent = `${(m.val_accuracy_own * 100).toFixed(1)}%${m.id === "legacy" ? "*" : ""}`;
  }
} catch {
  /* manifest missing: keep the static values */
}
