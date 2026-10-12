const $ = (id) => document.getElementById(id);

const els = {
  dropzone: $('dropzone'), pick: $('pick'), file: $('file'),
  samples: $('samples'), samplesRow: $('samples-row'),
  viewer: $('viewer'), filename: $('filename'), replace: $('replace'),
  frame: $('frame'), base: $('base'), overlay: $('overlay'),
  tagLeft: $('tag-left'), tagRight: $('tag-right'), divider: $('divider'), status: $('status'),
  models: $('models'), modes: $('modes'), modeHelp: $('mode-help'),
  results: $('results'), legend: $('legend'), opacity: $('opacity'), meta: $('meta'),
  dlMask: $('dl-mask'), dlOverlay: $('dl-overlay'),
};

const MODE_HELP = {
  auto: 'Jusqu’à 1536 px de côté, l’image est traitée comme un seul champ. Au-delà, elle est découpée en tuiles de 1024 px.',
  whole: 'Toute l’image est réduite à 256 × 256 px, comme un patch d’entraînement.',
  tiles: 'Découpage en tuiles de 1024 × 1024 px, l’échelle des patchs d’entraînement.',
};

const BLANK = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800"/>');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const fr = (n, digits) => n.toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits });

const state = {
  classes: [], models: [], model: null, mode: 'auto',
  file: null, fileKey: null, objectUrl: null,
  result: null, ids: null, hidden: new Set(),
  request: 0, controller: null,
};
const cache = new Map();

/* --- Spring: animates from the live value, so it can be grabbed at any moment --- */
class Spring {
  constructor(onUpdate, { response = 0.4, damping = 1 } = {}) {
    this.onUpdate = onUpdate;
    this.stiffness = (2 * Math.PI / response) ** 2;
    this.friction = 4 * Math.PI * damping / response;
    this.value = 0; this.velocity = 0; this.target = 0; this.frame = 0;
  }
  set(value) { this.stop(); this.value = value; this.velocity = 0; this.onUpdate(value); }
  stop() { cancelAnimationFrame(this.frame); this.frame = 0; }
  to(target) {
    this.target = target;
    if (reducedMotion.matches) return this.set(target);
    if (this.frame) return;
    let last = performance.now();
    const tick = (now) => {
      // Real elapsed time, so the motion lasts the same however often frames arrive.
      let dt = Math.min((now - last) / 1000, 0.5);
      last = now;
      while (dt > 0) {
        const step = Math.min(dt, 1 / 240);
        this.velocity += (this.stiffness * (this.target - this.value) - this.friction * this.velocity) * step;
        this.value += this.velocity * step;
        dt -= step;
      }
      if (Math.abs(this.target - this.value) < 0.0005 && Math.abs(this.velocity) < 0.0005) {
        this.frame = 0;
        this.value = this.target;
        this.onUpdate(this.value);
        return;
      }
      this.onUpdate(this.value);
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }
}

/* --- Compare divider: tracks the pointer 1:1 from where it was grabbed --- */
const clamp01 = (v) => Math.min(1, Math.max(0, v));

const split = new Spring((value) => {
  const v = clamp01(value);
  els.frame.style.setProperty('--split', v);
  els.divider.setAttribute('aria-valuenow', Math.round(v * 100));
  els.tagLeft.style.opacity = v < 0.14 ? 0 : 1;
  els.tagRight.style.opacity = v > 0.86 ? 0 : 1;
});
split.set(0.5);

let grabOffset = 0;
els.divider.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  split.stop();
  els.divider.setPointerCapture(e.pointerId);
  els.divider.classList.add('is-dragging');
  const rect = els.frame.getBoundingClientRect();
  grabOffset = e.clientX - (rect.left + split.value * rect.width);
  e.preventDefault();
});
els.divider.addEventListener('pointermove', (e) => {
  if (!els.divider.hasPointerCapture(e.pointerId)) return;
  const rect = els.frame.getBoundingClientRect();
  split.set(clamp01((e.clientX - grabOffset - rect.left) / rect.width));
});
const endDrag = () => els.divider.classList.remove('is-dragging');
els.divider.addEventListener('pointerup', endDrag);
els.divider.addEventListener('pointercancel', endDrag);
els.divider.addEventListener('dblclick', () => split.to(0.5));
els.divider.addEventListener('keydown', (e) => {
  const moves = { ArrowLeft: -0.02, ArrowRight: 0.02, PageDown: -0.1, PageUp: 0.1 };
  if (e.key in moves) split.set(clamp01(split.value + moves[e.key]));
  else if (e.key === 'Home') split.set(0);
  else if (e.key === 'End') split.set(1);
  else return;
  e.preventDefault();
});

/* --- Status pill --- */
function setStatus(text, kind) {
  els.status.hidden = !text;
  els.status.textContent = text || '';
  els.status.className = 'status' + (kind ? ` is-${kind}` : '');
  els.frame.classList.toggle('is-busy', kind === 'busy');
}

/* --- Mask rendering --- */
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [n >> 16 & 255, n >> 8 & 255, n & 255];
}

async function decodeMask(dataUrl) {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const ids = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < ids.length; i++) ids[i] = Math.round(data[i * 4] / 50);
  return { ids, width: canvas.width, height: canvas.height };
}

function paintMask(canvas, { background = null, respectHidden = true } = {}) {
  const { ids, width, height } = state.ids;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width, height);
  const palette = state.classes.map((c) => (c.color ? hexToRgb(c.color) : null));
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    const rgb = palette[id] && !(respectHidden && state.hidden.has(id)) ? palette[id] : background;
    if (!rgb) continue;
    image.data[i * 4] = rgb[0];
    image.data[i * 4 + 1] = rgb[1];
    image.data[i * 4 + 2] = rgb[2];
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function renderLegend(result) {
  const rows = result.classes.filter((c) => state.classes[c.id].color);
  const max = Math.max(...rows.map((c) => c.percent), 0.0001);
  els.legend.replaceChildren(...rows.map((c) => {
    const def = state.classes[c.id];
    const li = document.createElement('li');
    li.innerHTML = `
      <label class="legend-row" style="--c:${def.color}">
        <input type="checkbox" ${state.hidden.has(c.id) ? '' : 'checked'}>
        <span class="legend-head">
          <span class="swatch"></span>
          <span class="legend-label"></span>
          <span class="legend-value"></span>
        </span>
        <span class="bar"><i></i></span>
      </label>`;
    li.querySelector('.legend-label').textContent = def.label;
    li.querySelector('.legend-value').textContent = `${fr(c.percent, 2)} %`;
    li.querySelector('input').addEventListener('change', (e) => {
      if (e.target.checked) state.hidden.delete(c.id); else state.hidden.add(c.id);
      paintMask(els.overlay);
    });
    const bar = li.querySelector('i');
    requestAnimationFrame(() => requestAnimationFrame(() => bar.style.setProperty('--v', c.percent / max)));
    return li;
  }));
}

async function showResult(result, isNewImage) {
  state.ids = await decodeMask(result.mask);
  state.result = result;
  els.base.src = result.image;
  paintMask(els.overlay);
  els.frame.classList.add('has-result');
  setStatus('');

  const model = state.models.find((m) => m.id === result.model);
  const tiles = result.tiles > 1 ? `${result.tiles} tuiles` : '1 champ';
  els.meta.textContent = `Part de la surface de l’image. ${model.name} (${model.run}) · ${tiles} · ${result.width} × ${result.height} px · ${result.elapsed_ms} ms`;
  renderLegend(result);
  els.results.hidden = false;

  if (isNewImage) { split.set(1); split.to(0.5); }
}

/* --- Inference --- */
async function analyse() {
  if (!state.file || !state.model) return;
  const key = `${state.fileKey}|${state.model}|${state.mode}`;
  const isNewImage = !state.result || state.result.fileKey !== state.fileKey;
  const request = ++state.request;
  state.controller?.abort();

  if (cache.has(key)) return showResult(cache.get(key), isNewImage);

  state.controller = new AbortController();
  setStatus('Analyse en cours…', 'busy');
  const body = new FormData();
  body.append('file', state.file);
  body.append('model', state.model);
  body.append('mode', state.mode);
  try {
    const response = await fetch('api/predict', { method: 'POST', body, signal: state.controller.signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `Erreur du serveur (${response.status}).`);
    payload.fileKey = state.fileKey;
    cache.set(key, payload);
    if (request === state.request) await showResult(payload, isNewImage);
  } catch (error) {
    if (error.name === 'AbortError' || request !== state.request) return;
    const offline = error instanceof TypeError;
    setStatus(offline ? 'Serveur injoignable. Vérifiez qu’il est démarré, puis réessayez.' : error.message, 'error');
  }
}

function setFile(file) {
  if (!file) return;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.fileKey = `${file.name}|${file.size}|${file.lastModified}`;
  state.result = null;
  state.ids = null;
  els.frame.classList.remove('has-result');
  els.results.hidden = true;
  els.filename.textContent = file.name;

  // Browsers cannot display TIFF: show a neutral placeholder until the server returns a preview.
  const displayable = /^image\/(png|jpeg|webp|bmp|gif)$/.test(file.type);
  state.objectUrl = displayable ? URL.createObjectURL(file) : null;
  els.base.src = state.objectUrl || BLANK;

  els.dropzone.hidden = true;
  els.viewer.hidden = false;
  analyse();
}

/* --- File input, drag and drop --- */
els.pick.addEventListener('click', () => els.file.click());
els.replace.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', () => { setFile(els.file.files[0]); els.file.value = ''; });

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { dragDepth++; els.dropzone.classList.add('is-over'); } });
window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; els.dropzone.classList.remove('is-over'); } });
window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('drop', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  els.dropzone.classList.remove('is-over');
  setFile(e.dataTransfer.files[0]);
});

/* --- Settings --- */
function setMode(mode) {
  state.mode = mode;
  els.modeHelp.textContent = MODE_HELP[mode];
}
els.modes.addEventListener('change', (e) => { setMode(e.target.value); analyse(); });
setMode('auto');

els.opacity.addEventListener('input', () => { els.overlay.style.opacity = els.opacity.value / 100; });
els.overlay.style.opacity = els.opacity.value / 100;

function renderModels() {
  els.models.replaceChildren(...state.models.map((m) => {
    const label = document.createElement('label');
    label.className = 'model';
    label.innerHTML = `
      <input type="radio" name="model">
      <span class="model-body">
        <span class="model-name"></span>
        <span class="model-score" title="Dice moyen (validation)"></span>
        <span class="model-detail"></span>
      </span>`;
    const input = label.querySelector('input');
    input.value = m.id;
    input.checked = m.id === state.model;
    input.disabled = !m.available;
    label.querySelector('.model-name').textContent = m.name;
    label.querySelector('.model-score').textContent = fr(m.dice.overall, 3);
    label.querySelector('.model-detail').textContent = m.available ? `${m.run} · ${m.detail}` : `${m.run} · non installé`;
    return label;
  }));
}
els.models.addEventListener('change', (e) => { state.model = e.target.value; analyse(); });

/* --- Downloads --- */
function save(canvas, suffix) {
  canvas.toBlob((blob) => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${state.file.name.replace(/\.[^.]+$/, '')}_${state.result.model}_${suffix}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, 'image/png');
}

els.dlMask.addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  paintMask(canvas, { background: [0, 0, 0], respectHidden: false });
  save(canvas, 'masque');
});

els.dlOverlay.addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  canvas.width = state.ids.width;
  canvas.height = state.ids.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(els.base, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = els.opacity.value / 100;
  ctx.drawImage(els.overlay, 0, 0);
  save(canvas, 'superposition');
});

/* --- Start-up --- */
async function loadModels() {
  try {
    const response = await fetch('api/models');
    if (!response.ok) throw new Error();
    const data = await response.json();
    state.classes = data.classes;
    state.models = data.models;
    state.model = (data.models.find((m) => m.available) || {}).id || null;
    renderModels();
  } catch {
    els.models.textContent = 'Impossible de charger la liste des modèles. Vérifiez que le serveur est démarré.';
  }
}

async function loadSamples() {
  try {
    const response = await fetch('samples/index.json');
    if (!response.ok) return;
    const samples = await response.json();
    if (!samples.length) return;
    els.samplesRow.replaceChildren(...samples.map((s) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'sample';
      button.title = s.name;
      const img = document.createElement('img');
      img.src = `samples/${s.thumb}`;
      img.alt = s.name;
      button.append(img);
      button.addEventListener('click', async () => {
        const blob = await (await fetch(`samples/${s.file}`)).blob();
        setFile(new File([blob], s.file, { type: blob.type }));
      });
      return button;
    }));
    els.samples.hidden = false;
  } catch { /* samples are optional */ }
}

loadModels();
loadSamples();
