import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SparkXr, SplatLoader, SplatMesh } from '@sparkjsdev/spark';
import './style.css';

const $ = (selector) => document.querySelector(selector);
const stage = $('#viewer-canvas');
const fileInput = $('#file-input');
const sceneList = $('#scene-list');
const sceneMenu = $('#scene-menu');
const menuToggle = $('#menu-toggle');
const menuBackdrop = $('#menu-backdrop');
const compareButton = $('#compare-button');
const compareOverlay = $('#compare-overlay');
const compareOriginal = $('#compare-original');
const compareDivider = $('#compare-divider');
const uploadButton = $('#upload-button');
const settingsButton = $('#settings-button');
const settingsDialog = $('#settings-dialog');
const depthEstimatorSelect = $('#depth-estimator-select');
const fillToolSelect = $('#fill-tool-select');
const fillCoverageSelect = $('#fill-coverage-select');
const sharpResolutionSelect = $('#sharp-resolution-select');
const upscaleSelect = $('#upscale-select');
const depthAlignmentSelect = $('#depth-alignment-select');
const processingDialog = $('#processing-dialog');
const processingPreview = $('#processing-preview');
const panoramaButton = $('#process-panorama');
const imageButton = $('#process-image');
const loadingPanel = $('#viewer-loading');
const loadingDetail = $('#viewer-loading-detail');
const dropOverlay = $('#drop-overlay');
const jobBanner = $('#job-banner');
const jobCancelButton = $('#job-cancel');
const vrButton = $('#vr-button');
const viewerRoot = $('#drop-zone');
const pipelineToggle = $('#pipeline-toggle');
const pipelinePanel = $('#pipeline-panel');
const pipelineSteps = $('#pipeline-steps');
const artifactViewer = $('#artifact-viewer');
const artifactImage = $('#artifact-image');
const splatBadge = $('#splat-badge');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(53, 1, 0.015, 2000);
const canvas = document.createElement('canvas');
// An XR-compatible context lets VR start without making the context compatible first.
const context = canvas.getContext('webgl2', {
  alpha: true, antialias: false, depth: true, stencil: false, powerPreference: 'high-performance', xrCompatible: true,
});
const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x151515, 0);
stage.appendChild(renderer.domElement);
renderer.domElement.tabIndex = 0;
renderer.domElement.setAttribute('aria-label', '3D-Ansicht');
scene.add(new SparkRenderer({ renderer }));
// WebXR poses the camera relative to this rig; the controllers move the rig.
const xrRig = new THREE.Group();
xrRig.add(camera);
scene.add(xrRig);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.62;
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.6;
controls.minDistance = 0.25;
controls.maxDistance = 250;
controls.screenSpacePanning = true;

let scenes = [];
let activeSceneId = null;
let currentMesh = null;
// What the viewer shows: a scene id, or the URL of an intermediate splat.
let currentMeshSource = null;
let pipelineOpen = false;
let pipelineTimer = null;
let pipelineSignature = '';
let pipelineData = null;
let selectedArtifactUrl = null;
let pipelineRefreshing = false;
let pipelineRefreshQueued = false;
let comparePosition = 50;
let compareDragging = false;
let compareEnabled = false;
let loadVersion = 0;
let activeJob = null;
let pollingJobId = null;
let uploadAbort = null;
let toastTimer = null;
let dragDepth = 0;
let viewerMode = 'image';
let panoramaReady = false;
let panoramaMessage = '360° wird verfügbar, sobald die Panorama-Erweiterung eingerichtet ist.';
const fillToolStorageKey = 'sharp-studio-fill-tool';
const fillToolDefinitions = {
  'flux-fill': {
    mode: 'hemisphere', title: '180°',
    description: 'Vordere Halbkugel mit FLUX Fill ergänzen',
    settingsDescription: 'Erweitert dein Foto zu einer 180°-Halbkugel vor dir. Der Originalausschnitt bleibt erhalten.',
    setupMessage: 'FLUX Fill fehlt. Zuerst die Modellbedingungen bei Hugging Face annehmen und anmelden; dann setup-panorama.ps1 -FillTool flux-fill ausführen. Anleitung: STUDIO.md.',
  },
  'klein-lora': {
    mode: 'panorama', title: '360°',
    description: 'Rundum-Panorama mit FLUX.2 Klein ergänzen',
    settingsDescription: 'Erweitert dein Foto zu einer vollständigen 360°-Umgebung. Das Modell kann auch den Originalausschnitt verändern.',
    setupMessage: 'Einrichten: setup-panorama.ps1 -FillTool klein-lora.',
  },
};
let selectedFillTool = loadFillTool();
let fillTools = {};
const fillCoverageStorageKey = 'sharp-studio-fill-coverage';
let selectedFillCoverage = loadStoredChoice(fillCoverageStorageKey, ['180', '360'], '180');
let viewerCoverage = 360;
let da3Ready = false;
let da3Message = 'Depth Anything 3 ist noch nicht eingerichtet.';
const depthEstimatorStorageKey = 'sharp-studio-depth-estimator';
let selectedDepthEstimator = loadDepthEstimator();
const sharpResolutionStorageKey = 'sharp-studio-sharp-resolution';
const sharpResolutionDescriptions = {
  1024: 'Etwa 0,5 Mio. Gaussians pro Ansicht. Schnell, aber gröber als der Standard.',
  1536: 'Etwa 1,2 Mio. Gaussians pro Ansicht. Die Auflösung, mit der SHARP trainiert wurde.',
  2048: 'Etwa 2,1 Mio. Gaussians pro Ansicht und rund 11 GB Grafikspeicher. Lohnt sich bei Fotos ab etwa 2000 px oder mit KI-Upscaling.',
  2560: 'Etwa 3,3 Mio. Gaussians pro Ansicht. Braucht mehr als 16 GB Grafikspeicher und wird sonst deutlich langsamer.',
};
let selectedSharpResolution = loadStoredChoice(sharpResolutionStorageKey, ['1024', '1536', '2048', '2560'], '1536');
const upscaleStorageKey = 'sharp-studio-upscale';
let selectedUpscale = loadStoredChoice(upscaleStorageKey, ['off', 'auto'], 'off');
let upscalerReady = false;
let upscalerMessage = 'KI-Upscaler fehlt. Einrichten: setup-upscale.ps1 ausführen.';
const depthAlignmentStorageKey = 'sharp-studio-depth-alignment';
let selectedDepthAlignment = loadStoredChoice(depthAlignmentStorageKey, ['off', 'da2'], 'off');
let depthAlignmentReady = false;
let depthAlignmentMessage = 'DA² fehlt. Einrichten: setup-da2.ps1 ausführen.';
let pendingImage = null;
let pendingImageUrl = null;
const lookDirection = new THREE.Vector3();
const movement = new THREE.Vector3();
const pressedKeys = new Set();
const movementKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight']);
let lastFrameTime = null;
let movementFast = false;
let xrSavedView = null;
let xrButtonsHeld = new Set();
// Temporary VR start diagnostics: where does the time go after clicking VR?
let xrTiming = null;

function stopMovement() {
  pressedKeys.clear();
  movementFast = false;
  lastFrameTime = null;
}

function movePanoramaCamera(deltaSeconds) {
  if (viewerMode !== 'panorama' || !currentMesh || document.activeElement !== renderer.domElement) return;
  const forward = Number(pressedKeys.has('KeyW')) - Number(pressedKeys.has('KeyS'));
  const sideways = Number(pressedKeys.has('KeyD')) - Number(pressedKeys.has('KeyA'));
  const vertical = Number(pressedKeys.has('KeyE')) - Number(pressedKeys.has('KeyQ'));
  camera.getWorldDirection(lookDirection);
  const yaw = Math.atan2(lookDirection.x, -lookDirection.z);
  movement.set(
    Math.sin(yaw) * forward + Math.cos(yaw) * sideways,
    vertical,
    -Math.cos(yaw) * forward + Math.sin(yaw) * sideways,
  );
  if (movement.lengthSq() === 0) return;
  const speed = movementFast ? 6 : 2;
  movement.normalize().multiplyScalar(speed * deltaSeconds);
  camera.position.add(movement);
  controls.target.add(movement);
}

function rotatePanoramaView(yawDelta, pitchDelta) {
  const position = camera.position.clone();
  const target = controls.target.clone();
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.update();
  camera.position.copy(position);
  controls.target.copy(target);
  const distance = controls.getDistance();
  lookDirection.subVectors(controls.target, camera.position).normalize();
  let yaw = Math.atan2(lookDirection.x, -lookDirection.z) + yawDelta;
  const pitch = THREE.MathUtils.clamp(
    Math.asin(THREE.MathUtils.clamp(lookDirection.y, -1, 1)) + pitchDelta,
    -Math.PI / 2 + 0.01,
    Math.PI / 2 - 0.01,
  );
  if (viewerCoverage === 180) yaw = THREE.MathUtils.clamp(yaw, -Math.PI / 2, Math.PI / 2);
  const horizontal = Math.cos(pitch);
  lookDirection.set(
    Math.sin(yaw) * horizontal,
    Math.sin(pitch),
    -Math.cos(yaw) * horizontal,
  );
  // Turn at the current camera position, keeping the mouse's orbit distance.
  controls.target.copy(camera.position).addScaledVector(lookDirection, distance);
  controls.update();
  controls.enableDamping = damping;
}

function resetCamera() {
  stopMovement();
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.update();
  camera.position.set(0, 0, 0);
  camera.up.set(0, 1, 0);
  camera.fov = viewerMode === 'panorama' ? 75 : 53;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, -3);
  controls.update();
  controls.enableDamping = damping;
}

function setViewerMode(mode) {
  viewerCoverage = mode === 'hemisphere' ? 180 : 360;
  viewerMode = ['panorama', 'hemisphere'].includes(mode) ? 'panorama' : 'image';
  controls.minAzimuthAngle = viewerCoverage === 180 ? -Math.PI / 2 : -Infinity;
  controls.maxAzimuthAngle = viewerCoverage === 180 ? Math.PI / 2 : Infinity;
  stage.classList.toggle('panorama', viewerMode === 'panorama');
  stage.classList.remove('dragging');
  $('#viewer-hint').textContent = viewerMode === 'panorama'
    ? 'Ziehen zum Kippen · Pfeiltasten zum Drehen · Scrollen zum Zoomen'
    : 'Ziehen zum Drehen · Scrollen zum Zoomen';
  const help = viewerMode === 'panorama'
    ? 'Maus ziehen kippt die Szene horizontal und vertikal wie bei normalen Bildern. Pfeiltasten drehen die Blickrichtung; Shift dreht schneller. W/S vor und zurück, A/D seitlich, Q/E hoch und runter. Scrollen zum Zoomen.'
    : 'Maus ziehen zum Drehen, Scrollen zum Zoomen.';
  $('#viewer-hint').title = help;
  renderer.domElement.setAttribute('aria-label', `3D-Ansicht. ${help}`);
  resetCamera();
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  renderer.domElement.focus({ preventScroll: true });
  movementFast = event.shiftKey;
});
controls.addEventListener('start', () => {
  stage.classList.add('dragging');
});
controls.addEventListener('end', () => {
  stage.classList.remove('dragging');
});

renderer.domElement.addEventListener('keydown', (event) => {
  if (viewerMode !== 'panorama' || !currentMesh || renderer.xr.isPresenting) return;
  if (event.ctrlKey || event.altKey || event.metaKey) {
    stopMovement();
    return;
  }
  if (event.code.startsWith('Arrow')) {
    event.preventDefault();
    if (event.repeat) return;
    const angle = THREE.MathUtils.degToRad(event.shiftKey ? 12 : 4);
    if (event.code === 'ArrowLeft') rotatePanoramaView(-angle, 0);
    else if (event.code === 'ArrowRight') rotatePanoramaView(angle, 0);
    else if (event.code === 'ArrowUp') rotatePanoramaView(0, angle);
    else if (event.code === 'ArrowDown') rotatePanoramaView(0, -angle);
    else return;
    return;
  }
  if (!movementKeys.has(event.code)) return;
  event.preventDefault();
  movementFast = event.shiftKey;
  pressedKeys.add(event.code);
});
renderer.domElement.addEventListener('keyup', (event) => {
  movementFast = event.shiftKey;
  pressedKeys.delete(event.code);
});
renderer.domElement.addEventListener('blur', stopMovement);
window.addEventListener('blur', stopMovement);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopMovement();
  else if (xrTiming?.sessionStarted && xrTiming.visibleAt === null) xrTiming.visibleAt = performance.now();
});

function resize() {
  if (renderer.xr.isPresenting) return;
  const width = Math.max(1, stage.clientWidth);
  const height = Math.max(1, stage.clientHeight);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function resetXrRig() {
  xrRig.position.set(0, 0, 0);
  xrRig.rotation.set(0, 0, 0);
}

function placeXrRig() {
  // Enter VR at the desktop camera position, facing the same horizontal direction.
  camera.getWorldDirection(lookDirection);
  xrRig.position.copy(camera.position);
  xrRig.rotation.set(0, Math.atan2(-lookDirection.x, -lookDirection.z), 0);
}

function handleXrButtons() {
  const held = new Set();
  for (const source of renderer.xr.getSession()?.inputSources ?? []) {
    const buttons = source.gamepad?.buttons;
    if (!buttons || source.hand) continue;
    // xr-standard mapping: 4 is A/X, 5 is B/Y.
    if (buttons[4]?.pressed) held.add(`${source.handedness}-a`);
    if (buttons[5]?.pressed) held.add(`${source.handedness}-b`);
  }
  const pressed = (key) => held.has(key) && !xrButtonsHeld.has(key);
  if (pressed('right-a')) stepScene(1);
  if (pressed('right-b')) stepScene(-1);
  if (pressed('left-a') || pressed('left-b')) resetXrRig();
  xrButtonsHeld = held;
}

function setVrButtonState(presenting) {
  const label = presenting ? 'VR beenden' : 'In VR ansehen';
  vrButton.setAttribute('aria-pressed', String(presenting));
  vrButton.setAttribute('aria-label', label);
  vrButton.title = label;
  vrButton.querySelector('span').textContent = presenting ? 'VR beenden' : 'VR';
}

function recordXrFrame() {
  if (!xrTiming) return;
  const now = performance.now();
  if (xrTiming.firstFrame === null) xrTiming.firstFrame = now;
  else if (now - xrTiming.firstFrame < 5000) {
    const frameTime = now - xrTiming.lastFrame;
    xrTiming.frames += 1;
    xrTiming.frameTimeSum += frameTime;
    xrTiming.frameTimeMax = Math.max(xrTiming.frameTimeMax, frameTime);
  }
  xrTiming.lastFrame = now;
}

function reportXrTiming() {
  if (!xrTiming?.sessionStarted) return;
  const seconds = (from, to) => (to === null ? '–' : `${((to - from) / 1000).toFixed(1)} s`);
  const average = xrTiming.frames ? `${(xrTiming.frameTimeSum / xrTiming.frames).toFixed(0)} ms` : '–';
  const hidden = xrTiming.hiddenAtStart ? ` · Seite verdeckt bis ${seconds(xrTiming.sessionStarted, xrTiming.visibleAt)}` : ' · Seite sichtbar';
  const message = `VR-Start: Sitzung ${seconds(xrTiming.clicked, xrTiming.sessionStarted)} · erstes Bild ${seconds(xrTiming.sessionStarted, xrTiming.firstFrame)}${hidden} · Bildzeit Ø ${average}, max. ${xrTiming.frameTimeMax.toFixed(0)} ms`;
  console.info(message, xrTiming);
  showToast(message);
  xrTiming = null;
}

vrButton.addEventListener('click', () => {
  if (!renderer.xr.isPresenting) {
    xrTiming = { clicked: performance.now(), sessionStarted: null, firstFrame: null, lastFrame: 0, frames: 0, frameTimeSum: 0, frameTimeMax: 0 };
  }
});

const sparkXr = new SparkXr({
  renderer,
  element: vrButton,
  mode: 'vr',
  frameBufferScaleFactor: 1,
  controllers: { moveSpeed: 1.5, moveDirection: true },
  onReady: (supported) => { vrButton.hidden = !supported; },
});

renderer.xr.addEventListener('sessionstart', () => {
  if (xrTiming) {
    xrTiming.sessionStarted = performance.now();
    xrTiming.hiddenAtStart = document.hidden;
    xrTiming.visibleAt = document.hidden ? null : xrTiming.sessionStarted;
  }
  stopMovement();
  setSceneMenuOpen(false);
  xrSavedView = { sceneId: activeSceneId, position: camera.position.clone(), target: controls.target.clone() };
  placeXrRig();
  controls.enabled = false;
  sparkXr.lastControllersUpdate = 0;
  xrButtonsHeld = new Set();
  setVrButtonState(true);
});

renderer.xr.addEventListener('sessionend', () => {
  resetXrRig();
  controls.enabled = true;
  if (xrSavedView && xrSavedView.sceneId === activeSceneId) {
    const damping = controls.enableDamping;
    controls.enableDamping = false;
    camera.position.copy(xrSavedView.position);
    camera.up.set(0, 1, 0);
    controls.target.copy(xrSavedView.target);
    controls.update();
    controls.enableDamping = damping;
  } else {
    resetCamera();
  }
  xrSavedView = null;
  setVrButtonState(false);
  resize();
  reportXrTiming();
});

resetCamera();
new ResizeObserver(resize).observe(stage);
resize();
renderer.setAnimationLoop((time) => {
  const deltaSeconds = lastFrameTime === null ? 0 : Math.min(0.1, Math.max(0, (time - lastFrameTime) / 1000));
  lastFrameTime = time;
  if (renderer.xr.isPresenting) {
    recordXrFrame();
    sparkXr.updateControllers(camera);
    handleXrButtons();
  } else {
    movePanoramaCamera(deltaSeconds);
    if (controls.enabled) controls.update();
  }
  renderer.render(scene, camera);
});

function sceneLabel(item) {
  return item.name.replace(/\.(jpe?g|png|webp)$/i, '');
}

function showToast(message, error = false) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.toggle('error', error);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
}

function updateSceneActions(item) {
  const download = $('#download-link');
  download.hidden = !item.plyUrl;
  download.href = item.plyUrl;
  download.download = `${sceneLabel(item)}.ply`;
  compareButton.hidden = !item.imageUrl;
  if (item.imageUrl) compareOriginal.src = item.imageUrl;
  else compareOriginal.removeAttribute('src');
  if (!item.imageUrl) setCompareEnabled(false);
}

function updateSceneListSelection() {
  for (const button of sceneList.querySelectorAll('.scene-row-select')) {
    const selected = button.closest('.scene-row')?.dataset.sceneId === activeSceneId;
    button.classList.toggle('active', selected);
    if (selected) button.setAttribute('aria-current', 'true');
    else button.removeAttribute('aria-current');
  }
  const currentIndex = scenes.findIndex((item) => item.id === activeSceneId);
  $('#scene-position').textContent = currentIndex < 0 ? '' : `${currentIndex + 1} / ${scenes.length}`;
  $('#scene-stepper').hidden = scenes.length < 2;
}

function renderSceneList() {
  sceneList.replaceChildren();
  $('#scene-count').textContent = `${scenes.length} ${scenes.length === 1 ? 'Bild' : 'Bilder'}`;
  if (!scenes.length) {
    const empty = document.createElement('p');
    empty.className = 'scene-list-empty';
    empty.textContent = 'Noch keine Bilder vorhanden.';
    sceneList.append(empty);
  }
  for (const item of scenes) {
    const row = document.createElement('div');
    row.className = 'scene-row';
    row.dataset.sceneId = item.id;
    const button = document.createElement('button');
    button.className = 'scene-row-select';
    button.type = 'button';
    button.setAttribute('aria-label', `Bild öffnen: ${sceneLabel(item)}`);
    button.title = sceneLabel(item);
    if (item.imageUrl) {
      const image = document.createElement('img');
      image.src = item.imageUrl;
      image.alt = '';
      image.loading = 'lazy';
      button.append(image);
    } else {
      button.classList.add('no-preview');
    }
    const copy = document.createElement('span');
    copy.className = 'scene-row-copy';
    const label = document.createElement('span');
    label.className = 'scene-row-label';
    label.textContent = sceneLabel(item);
    const kind = document.createElement('span');
    kind.className = 'scene-row-kind';
    kind.textContent = item.kind === 'upload'
      ? (item.mode === 'panorama' ? 'Eigener Upload · 360°' : item.mode === 'hemisphere' ? 'Eigener Upload · 180°' : 'Eigener Upload')
        + (item.depthAlignment === 'da2' ? ' · DA²' : '')
      : 'Beispielbild';
    copy.append(label, kind);
    button.append(copy);
    button.addEventListener('click', () => {
      setSceneMenuOpen(false);
      void selectScene(item.id);
    });
    row.append(button);
    if (item.kind === 'upload') {
      const remove = document.createElement('button');
      remove.className = 'scene-delete';
      remove.type = 'button';
      remove.title = `„${sceneLabel(item)}“ löschen`;
      remove.setAttribute('aria-label', `„${sceneLabel(item)}“ löschen`);
      remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3" /></svg>';
      remove.addEventListener('click', () => { void deleteScene(item); });
      row.append(remove);
    }
    sceneList.append(row);
  }
  updateSceneListSelection();
}

function setCompareEnabled(enabled) {
  compareEnabled = enabled && !!activeSceneId && !!compareOriginal.getAttribute('src');
  compareButton.setAttribute('aria-pressed', String(compareEnabled));
  compareOverlay.hidden = !compareEnabled;
  compareDivider.hidden = !compareEnabled;
  compareOverlay.setAttribute('aria-hidden', String(!compareEnabled));
  updateComparePosition(comparePosition);
}

function updateComparePosition(value) {
  comparePosition = THREE.MathUtils.clamp(value, 0, 100);
  compareOverlay.style.setProperty('--compare-position', `${comparePosition}%`);
  compareDivider.style.left = `${comparePosition}%`;
  compareDivider.setAttribute('aria-valuenow', String(Math.round(comparePosition)));
}

function moveCompareDivider(clientX) {
  const bounds = stage.getBoundingClientRect();
  if (bounds.width) updateComparePosition(((clientX - bounds.left) / bounds.width) * 100);
}

function setSceneMenuOpen(open) {
  sceneMenu.hidden = !open;
  menuBackdrop.hidden = !open;
  menuToggle.setAttribute('aria-expanded', String(open));
  if (open) (sceneMenu.querySelector('.scene-row-select.active') || sceneMenu.querySelector('.scene-row-select'))?.focus();
  else if (sceneMenu.contains(document.activeElement)) menuToggle.focus();
}

function stepScene(direction) {
  if (!scenes.length) return;
  const currentIndex = scenes.findIndex((item) => item.id === activeSceneId);
  const nextIndex = (currentIndex + direction + scenes.length) % scenes.length;
  void selectScene(scenes[nextIndex].id);
}

async function fetchScenes() {
  const response = await fetch('/api/scenes', { cache: 'no-store' });
  if (!response.ok) throw new Error('Szenen konnten nicht geladen werden.');
  scenes = (await response.json()).scenes;
  renderSceneList();
}

async function deleteScene(item) {
  if (item.kind !== 'upload') return;
  if (!window.confirm(`„${sceneLabel(item)}“ und die zugehörigen Dateien wirklich löschen?`)) return;
  const row = [...sceneList.querySelectorAll('.scene-row')].find((entry) => entry.dataset.sceneId === item.id);
  const removeButton = row?.querySelector('.scene-delete');
  if (removeButton) removeButton.disabled = true;
  try {
    const response = await fetch(`/api/scenes/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Bild konnte nicht gelöscht werden.');
    const wasActive = activeSceneId === item.id;
    scenes = scenes.filter((sceneItem) => sceneItem.id !== item.id);
    if (wasActive) {
      if (currentMesh) {
        scene.remove(currentMesh);
        currentMesh.dispose?.();
        currentMesh = null;
      }
      activeSceneId = null;
      setCompareEnabled(false);
      $('#download-link').hidden = true;
      compareButton.hidden = true;
    }
    renderSceneList();
    if (wasActive && scenes.length) {
      void selectScene(scenes[0].id);
    } else if (wasActive) {
      $('#viewer-empty').hidden = false;
    }
    showToast(`„${sceneLabel(item)}“ wurde gelöscht.`);
  } catch (error) {
    showToast(error.message, true);
    renderSceneList();
  }
}

async function fetchCapabilities() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    if (response.ok) {
      const health = await response.json();
      panoramaReady = health.panorama?.ready === true;
      panoramaMessage = health.panorama?.message || panoramaMessage;
      fillTools = health.fillTools || { 'klein-lora': health.panorama };
      da3Ready = health.depthEstimators?.da3?.ready === true;
      da3Message = health.depthEstimators?.da3?.message || 'Depth Anything 3 ist noch nicht eingerichtet.';
      upscalerReady = health.upscaler?.ready === true;
      upscalerMessage = health.upscaler?.message || upscalerMessage;
      depthAlignmentReady = health.depthAlignment?.ready === true;
      depthAlignmentMessage = health.depthAlignment?.message || depthAlignmentMessage;
    }
  } catch {
    // Existing servers may not expose panorama capabilities yet.
  }
  updateDepthSettings();
  updateFillSettings();
  updateSharpSettings();
}

function loadStoredChoice(key, allowed, fallback) {
  try {
    const saved = localStorage.getItem(key);
    return allowed.includes(saved) ? saved : fallback;
  } catch {
    return fallback;
  }
}

function storeChoice(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // The current selection still works when browser storage is unavailable.
  }
}

function updateSharpSettings() {
  const sharpActive = selectedDepthEstimator === 'sharp';
  sharpResolutionSelect.value = selectedSharpResolution;
  sharpResolutionSelect.disabled = !sharpActive;
  $('#sharp-resolution-description').textContent = sharpResolutionDescriptions[selectedSharpResolution];
  upscaleSelect.querySelector('[value="auto"]').disabled = !upscalerReady;
  upscaleSelect.value = effectiveUpscale();
  upscaleSelect.disabled = !sharpActive;
  $('#upscale-description').textContent = effectiveUpscale() === 'auto'
    ? `Fotos, deren lange Seite kleiner als ${selectedSharpResolution} px ist, werden vorher per KI vergrößert. Feine Details werden dabei erfunden, nicht wiederhergestellt.`
    : 'Kleinere Fotos werden nur rechnerisch auf die SHARP-Auflösung vergrößert.';
  const note = $('#upscale-note');
  note.hidden = upscalerReady || selectedUpscale !== 'auto';
  note.textContent = upscalerMessage;
  depthAlignmentSelect.querySelector('[value="da2"]').disabled = !depthAlignmentReady;
  depthAlignmentSelect.value = effectiveDepthAlignment();
  depthAlignmentSelect.disabled = !sharpActive;
  $('#depth-alignment-description').textContent = effectiveDepthAlignment() === 'da2'
    ? 'DA² schätzt die Tiefe des ganzen Panoramas in einem Durchlauf; SHARP gleicht jede Ansicht daran an. Das beseitigt Tiefensprünge zwischen den Ansichten; die 3D-Berechnung dauert etwa doppelt so lange. Einzelne Fotos werden nicht verändert.'
    : 'SHARP schätzt jede Ansicht einer 180°- oder 360°-Szene einzeln. An den Übergängen sind Tiefensprünge möglich.';
  const alignmentNote = $('#depth-alignment-note');
  alignmentNote.hidden = depthAlignmentReady || selectedDepthAlignment !== 'da2';
  alignmentNote.textContent = depthAlignmentMessage;
}

function effectiveDepthAlignment() {
  return selectedDepthAlignment === 'da2' && depthAlignmentReady ? 'da2' : 'off';
}

function effectiveUpscale() {
  return selectedUpscale === 'auto' && upscalerReady ? 'auto' : 'off';
}

function loadFillTool() {
  try {
    const saved = localStorage.getItem(fillToolStorageKey);
    return saved ? saved : 'flux-fill';
  } catch {
    return 'flux-fill';
  }
}

// Built-in tools are fixed; additional models (setup.ps1) come from /api/health.
function fillDefinition(key) {
  if (Object.hasOwn(fillToolDefinitions, key)) return fillToolDefinitions[key];
  const tool = fillTools[key];
  const coverages = tool?.coverages || [180];
  const coverage = coverages.includes(Number(selectedFillCoverage)) ? Number(selectedFillCoverage) : coverages[0];
  return {
    mode: coverage === 360 ? 'panorama' : 'hemisphere', title: `${coverage}°`, coverages, extra: true,
    description: `${coverage === 360 ? 'Rundum-Panorama' : 'Vordere Halbkugel'} mit ${tool?.label || 'diesem Modell'} ergänzen`,
    settingsDescription: tool ? tool.description : 'Dieses Modell ist nicht installiert.',
    setupMessage: 'Dieses Modell ist nicht installiert. Mit setup.ps1 nachinstallieren.',
  };
}

function renderFillOptions() {
  const extra = Object.entries(fillTools).filter(([key, tool]) => tool.extra);
  const signature = JSON.stringify([extra.map(([key, tool]) => [key, tool.label]), selectedFillTool]);
  if (fillToolSelect.dataset.signature === signature) return;
  fillToolSelect.dataset.signature = signature;
  fillToolSelect.replaceChildren(
    new Option('FLUX Fill · 180° vor dir', 'flux-fill'),
    new Option('FLUX.2 Klein + LoRA · 360° rundum', 'klein-lora'),
  );
  if (!Object.hasOwn(fillToolDefinitions, selectedFillTool) && !fillTools[selectedFillTool]) {
    extra.push([selectedFillTool, { label: 'Modell (nicht installiert)' }]);
  }
  if (extra.length) {
    const group = document.createElement('optgroup');
    group.label = 'Weitere Modelle';
    for (const [key, tool] of extra) group.append(new Option(tool.label, key));
    fillToolSelect.append(group);
  }
}

function updateFillSettings() {
  renderFillOptions();
  const definition = fillDefinition(selectedFillTool);
  const capability = fillTools[selectedFillTool];
  const ready = capability?.ready === true;
  const modelReady = selectedDepthEstimator !== 'da3' || da3Ready;
  const fillMessage = ready ? '' : capability?.message || definition.setupMessage;
  panoramaReady = ready && modelReady;
  panoramaMessage = modelReady ? fillMessage : da3Message;
  fillToolSelect.value = selectedFillTool;
  fillCoverageSelect.hidden = !definition.extra || definition.coverages.length < 2;
  for (const option of fillCoverageSelect.options) option.disabled = !definition.coverages?.includes(Number(option.value));
  fillCoverageSelect.value = String(definition.mode === 'panorama' ? 360 : 180);
  $('#fill-tool-description').textContent = definition.settingsDescription;
  const note = $('#fill-tool-note');
  note.hidden = ready;
  note.textContent = fillMessage;
  panoramaButton.querySelector('.processing-choice-title').textContent = `${definition.title} + ${selectedDepthEstimator === 'da3' ? 'DA3' : 'SHARP'}`;
  panoramaButton.querySelector('.processing-choice-description').textContent = definition.description;
  panoramaButton.disabled = !panoramaReady;
  const processingNote = $('#processing-note');
  processingNote.hidden = panoramaReady;
  processingNote.textContent = panoramaMessage;
}

function loadDepthEstimator() {
  try {
    return localStorage.getItem(depthEstimatorStorageKey) === 'da3' ? 'da3' : 'sharp';
  } catch {
    return 'sharp';
  }
}

function saveDepthEstimator() {
  try {
    localStorage.setItem(depthEstimatorStorageKey, selectedDepthEstimator);
  } catch {
    // The current selection still works when browser storage is unavailable.
  }
}

function updateDepthSettings() {
  depthEstimatorSelect.querySelector('[value="da3"]').disabled = !da3Ready;
  depthEstimatorSelect.value = selectedDepthEstimator;
  $('#depth-estimator-description').textContent = selectedDepthEstimator === 'da3'
    ? 'Erzeugt Tiefe und Gaussians vollständig mit DA3. Ohne Apple Depth Pro.'
    : 'SHARP erzeugt die Gaussian-Szene mit seiner Depth-Pro-basierten Vorhersage.';
  imageButton.querySelector('.processing-choice-title').textContent = selectedDepthEstimator === 'da3' ? 'Nur DA3' : 'Nur SHARP';
  imageButton.disabled = selectedDepthEstimator === 'da3' && !da3Ready;
  const note = $('#depth-estimator-note');
  note.hidden = da3Ready;
  note.textContent = da3Ready ? '' : /setup-depth\.ps1/i.test(da3Message)
    ? da3Message
    : `${da3Message} Einrichten: setup-depth.ps1.`;
}

async function loadSplatMesh(url, version, label) {
  const packedSplats = await new SplatLoader().loadAsync(url, (event) => {
    if (version !== loadVersion || event.type !== 'progress') return;
    if (event.lengthComputable && event.total) {
      const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
      loadingDetail.textContent = `${label} · ${percent} %`;
    }
  });
  if (version !== loadVersion) {
    packedSplats.dispose?.();
    return null;
  }
  const mesh = new SplatMesh({ packedSplats });
  // Apple SHARP writes camera coordinates as x-right, y-down, z-forward.
  // A half-turn around X puts the scene in Three.js's camera coordinates.
  mesh.quaternion.set(1, 0, 0, 0);
  scene.add(mesh);
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.dispose?.();
  }
  currentMesh = mesh;
  return mesh;
}

async function selectScene(id) {
  const item = scenes.find((entry) => entry.id === id);
  if (!item) return false;
  const version = ++loadVersion;
  if (activeSceneId === id && currentMesh && currentMeshSource === id) {
    loadingPanel.hidden = true;
    return true;
  }
  $('#viewer-empty').hidden = true;
  loadingPanel.hidden = false;
  loadingDetail.textContent = 'Szene wird geladen …';
  try {
    if (!await loadSplatMesh(item.plyUrl, version, 'Szene wird geladen')) return false;
    currentMeshSource = id;
    splatBadge.hidden = true;
    selectedArtifactUrl = null;
    activeSceneId = id;
    updateSceneListSelection();
    updateSceneActions(item);
    setViewerMode(item.mode);
    if (renderer.xr.isPresenting) resetXrRig();
    loadingPanel.hidden = true;
    if (pipelineOpen) void refreshPipeline();
    return true;
  } catch (error) {
    if (version !== loadVersion) return false;
    loadingPanel.hidden = true;
    $('#viewer-empty').hidden = !!currentMesh;
    showToast(`Splat konnte nicht geladen werden: ${error.message}`, true);
    console.error(error);
    return false;
  }
}

function jobStartedAt(job, fallback = Date.now()) {
  const timestamp = Date.parse(job.startedAt || '');
  return Number.isFinite(timestamp) ? timestamp : fallback;
}

function showJobStatus(job) {
  const percent = Number(job.progressPercent);
  const stageText = job.stage || 'Bild wird verarbeitet …';
  $('#job-stage').textContent = Number.isFinite(percent) && job.progressPercent != null
    ? `${stageText} · ${Math.round(THREE.MathUtils.clamp(percent, 0, 100))} %`
    : stageText;
  $('#job-elapsed').textContent = `${Math.max(0, Math.round((Date.now() - activeJob.started) / 1000))} s`;
  jobBanner.hidden = false;
}

async function resumeActiveJob() {
  if (activeJob) return;
  try {
    const response = await fetch('/api/jobs', { cache: 'no-store' });
    if (!response.ok) return;
    const jobs = (await response.json()).jobs;
    if (!Array.isArray(jobs)) return;
    const candidates = jobs
      .map((job, index) => ({ job, index }))
      .filter(({ job }) => job.id && !['ready', 'error'].includes(job.status))
      .sort((a, b) => {
        const processing = Number(b.job.status === 'processing') - Number(a.job.status === 'processing');
        if (processing) return processing;
        const aTime = Date.parse(a.job.createdAt || a.job.created || '') || 0;
        const bTime = Date.parse(b.job.createdAt || b.job.created || '') || 0;
        return aTime && bTime ? bTime - aTime || b.index - a.index : b.index - a.index;
      });
    const job = candidates[0]?.job;
    if (!job || activeJob) return;
    activeJob = { id: job.id, name: job.name, mode: job.mode, depthEstimator: job.depthEstimator || 'sharp', started: jobStartedAt(job) };
    uploadButton.disabled = true;
    jobCancelButton.disabled = false;
    $('#viewer-empty').hidden = true;
    showJobStatus(job);
    void pollJob(job.id);
  } catch {
    // Older servers do not expose the list of running jobs.
  }
}

function canUploadImage(file) {
  if (!file) return false;
  if (uploadButton.disabled) {
    showToast('Bitte warte, bis das aktuelle Bild fertig verarbeitet ist.', true);
    return false;
  }
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) {
    showToast('Bitte ein JPG-, PNG- oder WebP-Bild auswählen.', true);
    return false;
  }
  if (file.size > 30 * 1024 * 1024) {
    showToast('Das Bild darf höchstens 30 MB groß sein.', true);
    return false;
  }
  return true;
}

function clearPendingImage() {
  pendingImage = null;
  if (pendingImageUrl) URL.revokeObjectURL(pendingImageUrl);
  pendingImageUrl = null;
  processingPreview.removeAttribute('src');
}

function dismissProcessingChoice() {
  clearPendingImage();
  if (processingDialog.open) processingDialog.close();
}

function offerProcessingChoice(file) {
  if (!canUploadImage(file)) return;
  if (settingsDialog.open) settingsDialog.close();
  clearPendingImage();
  pendingImage = file;
  pendingImageUrl = URL.createObjectURL(file);
  processingPreview.hidden = false;
  processingPreview.src = pendingImageUrl;
  $('#processing-filename').textContent = file.name;
  $('#processing-filename').title = file.name;
  stopMovement();
  if (!processingDialog.open) processingDialog.showModal();
  (panoramaReady ? panoramaButton : imageButton).focus();
}

function chooseProcessingMode(mode) {
  const file = pendingImage;
  if (!file || (mode !== 'image' && !panoramaReady)) return;
  dismissProcessingChoice();
  void uploadImage(file, mode);
}

async function uploadImage(file, mode) {
  if (!canUploadImage(file) || !['panorama', 'hemisphere', 'image'].includes(mode)) return;
  if (mode !== 'image' && !panoramaReady) {
    showToast(panoramaMessage, true);
    return;
  }
  const depthEstimator = selectedDepthEstimator;
  if (depthEstimator === 'da3' && !da3Ready) {
    showToast('Depth Anything 3 ist noch nicht bereit. Bitte die Einrichtung in den Einstellungen prüfen.', true);
    return;
  }
  uploadButton.disabled = true;
  jobCancelButton.disabled = false;
  $('#viewer-empty').hidden = true;
  $('#job-stage').textContent = 'Bild wird hochgeladen …';
  $('#job-elapsed').textContent = '0 s';
  jobBanner.hidden = false;
  uploadAbort = new AbortController();
  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      signal: uploadAbort.signal,
      headers: {
        'X-Filename': encodeURIComponent(file.name),
        'X-Mode': mode,
        'X-Depth-Estimator': depthEstimator,
        'X-Fill-Tool': selectedFillTool,
        'X-Sharp-Resolution': selectedSharpResolution,
        'X-Upscale': effectiveUpscale(),
        'X-Depth-Alignment': effectiveDepthAlignment(),
        'Content-Type': file.type || 'application/octet-stream',
      },
      body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload fehlgeschlagen.');
    activeJob = { id: data.jobId, name: file.name, mode, depthEstimator, started: jobStartedAt(data) };
    $('#job-stage').textContent = 'Bild wird verarbeitet …';
    if (pipelineOpen) void refreshPipeline();
    void pollJob(data.jobId);
  } catch (error) {
    endJobStatus();
    if (error.name === 'AbortError') showToast('Upload abgebrochen.');
    else showToast(error.message, true);
  } finally {
    uploadAbort = null;
  }
}

function endJobStatus() {
  activeJob = null;
  uploadButton.disabled = false;
  jobBanner.hidden = true;
  if (!currentMesh) $('#viewer-empty').hidden = false;
  if (pipelineOpen) void refreshPipeline();
}

// Pipeline view: a timeline of the job's intermediate results, refreshed while it runs.
const pipelineStatusText = {
  done: 'fertig', running: 'läuft', pending: 'wartet', skipped: 'nicht gespeichert', error: 'Fehler', cancelled: 'abgebrochen',
};
const jobStatusText = {
  queued: 'In Warteschlange', processing: 'Wird verarbeitet', ready: 'Fertig', error: 'Fehlgeschlagen', cancelled: 'Abgebrochen', unknown: 'Unbekannt',
};

function pipelineTargetId() {
  if (activeJob?.id) return activeJob.id;
  return /^[a-f0-9]{12}$/.test(activeSceneId || '') ? activeSceneId : null;
}

function setPipelineOpen(open) {
  pipelineOpen = open;
  pipelinePanel.hidden = !open;
  viewerRoot.classList.toggle('pipeline-open', open);
  pipelineToggle.setAttribute('aria-expanded', String(open));
  clearTimeout(pipelineTimer);
  if (open) {
    pipelineSignature = '';
    void refreshPipeline();
  } else {
    closeArtifactImage();
  }
}

async function refreshPipeline() {
  clearTimeout(pipelineTimer);
  if (!pipelineOpen) return;
  if (pipelineRefreshing) {
    pipelineRefreshQueued = true;
    return;
  }
  const id = pipelineTargetId();
  if (!id) {
    pipelineData = null;
    pipelineSignature = '';
    $('#pipeline-subtitle').textContent = '';
    const empty = document.createElement('li');
    empty.className = 'pipeline-empty';
    empty.textContent = activeSceneId
      ? 'Für Beispielbilder gibt es keine gespeicherten Zwischenschritte. Wähle einen eigenen Upload oder starte eine neue Verarbeitung.'
      : 'Noch keine Szene ausgewählt.';
    pipelineSteps.replaceChildren(empty);
    return;
  }
  pipelineRefreshing = true;
  try {
    const response = await fetch(`/api/pipeline/${encodeURIComponent(id)}`, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Pipeline konnte nicht geladen werden.');
    const signature = JSON.stringify(data);
    if (signature !== pipelineSignature && pipelineOpen && pipelineTargetId() === id) {
      pipelineSignature = signature;
      pipelineData = data;
      renderPipeline(data);
    }
    if (['queued', 'processing'].includes(data.status)) pipelineTimer = setTimeout(() => void refreshPipeline(), 1500);
  } catch (error) {
    const message = document.createElement('li');
    message.className = 'pipeline-empty';
    message.textContent = error.message;
    pipelineSteps.replaceChildren(message);
    pipelineSignature = '';
  } finally {
    pipelineRefreshing = false;
    if (pipelineRefreshQueued) {
      pipelineRefreshQueued = false;
      void refreshPipeline();
    }
  }
}

function formatClock(iso) {
  return new Date(iso).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDuration(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  return seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`;
}

function formatBytes(bytes) {
  return bytes >= 1e6 ? `${(bytes / 1e6).toLocaleString('de-DE', { maximumFractionDigits: 0 })} MB` : `${Math.ceil(bytes / 1e3)} kB`;
}

function metaList(meta) {
  const entries = Object.entries(meta || {});
  if (!entries.length) return null;
  const list = document.createElement('p');
  list.className = 'pipeline-meta';
  for (const [key, value] of entries) {
    const item = document.createElement('span');
    const strong = document.createElement('b');
    strong.textContent = value;
    item.append(`${key}: `, strong);
    list.append(item);
  }
  return list;
}

function artifactGrid(artifacts, context) {
  const grid = document.createElement('div');
  grid.className = 'artifact-grid';
  for (const artifact of artifacts) {
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'artifact-tile';
    tile.dataset.url = artifact.url;
    tile.setAttribute('aria-pressed', String(artifact.url === selectedArtifactUrl));
    const label = context.endsWith(artifact.label) ? context : `${context} · ${artifact.label}`;
    tile.title = artifact.kind === 'splat' ? `${label} im 3D-Viewer laden (${formatBytes(artifact.sizeBytes)})` : `${label} groß anzeigen`;
    if (artifact.kind === 'image') {
      const image = document.createElement('img');
      image.src = artifact.thumbnailUrl;
      image.alt = '';
      image.loading = 'lazy';
      tile.append(image);
    } else {
      const icon = document.createElement('span');
      icon.className = 'artifact-splat';
      icon.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9" /></svg>';
      tile.append(icon);
    }
    const caption = document.createElement('span');
    caption.textContent = artifact.kind === 'splat' ? `${artifact.label} · ${formatBytes(artifact.sizeBytes)}` : artifact.label;
    tile.append(caption);
    tile.addEventListener('click', () => void openArtifact(artifact, label));
    grid.append(tile);
  }
  return grid;
}

function seamTable(seams) {
  const table = document.createElement('table');
  table.className = 'seam-table';
  table.innerHTML = '<thead><tr><th>Naht</th><th>Median</th><th>90 %</th></tr></thead>';
  const body = document.createElement('tbody');
  const rows = [['alle', seams], ...Object.entries(seams.pairs || {})];
  for (const [name, values] of rows) {
    const row = document.createElement('tr');
    for (const value of [name.replace('/', ' / '), values.median, values.p90]) {
      const cell = document.createElement('td');
      cell.textContent = typeof value === 'number' ? value.toLocaleString('de-DE', { minimumFractionDigits: 3 }) : value;
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body);
  return table;
}

function renderPipeline(data) {
  const scrollTop = pipelineSteps.scrollTop;
  const coverage = data.mode === 'panorama' ? ' · 360°' : data.mode === 'hemisphere' ? ' · 180°' : '';
  $('#pipeline-subtitle').textContent = `${sceneLabel(data)}${coverage} · ${jobStatusText[data.status] || data.status}`;
  const items = [];
  let previous = Date.parse(data.startedAt || data.created || '') || null;
  for (const step of data.steps) {
    const item = document.createElement('li');
    item.className = 'pipeline-step';
    item.dataset.status = step.status;
    const head = document.createElement('div');
    head.className = 'pipeline-step-head';
    const title = document.createElement('h3');
    title.className = 'pipeline-step-title';
    title.textContent = step.title;
    const time = document.createElement('span');
    time.className = 'pipeline-step-time';
    if (step.finishedAt) {
      const finished = Date.parse(step.finishedAt);
      time.textContent = formatClock(step.finishedAt) + (previous && step.key !== 'input' && finished >= previous ? ` · ${formatDuration(finished - previous)}` : '');
      if (step.key !== 'input') previous = finished;
    } else {
      time.textContent = pipelineStatusText[step.status] || '';
    }
    head.append(title, time);
    item.append(head);
    const text = document.createElement('p');
    text.className = 'pipeline-step-text';
    text.textContent = step.description;
    item.append(text);
    if (step.status === 'running' && step.stage) {
      const stage = document.createElement('p');
      stage.className = 'pipeline-step-stage';
      stage.textContent = `${step.stage}${data.progressPercent != null ? ` · ${data.progressPercent} %` : ''}`;
      item.append(stage);
    }
    if (step.error) {
      const error = document.createElement('p');
      error.className = 'pipeline-error';
      error.textContent = step.error;
      item.append(error);
      if (data.logTail) {
        const details = document.createElement('details');
        details.className = 'pipeline-log';
        const summary = document.createElement('summary');
        summary.textContent = 'Protokoll anzeigen';
        const log = document.createElement('pre');
        log.textContent = data.logTail;
        details.append(summary, log);
        item.append(details);
      }
    }
    const meta = metaList(step.meta);
    if (meta) item.append(meta);
    if (step.artifacts.length) item.append(artifactGrid(step.artifacts, step.title));
    for (const group of step.groups) {
      const block = document.createElement('div');
      block.className = 'pipeline-group';
      const groupTitle = document.createElement('h4');
      groupTitle.className = 'pipeline-group-title';
      groupTitle.textContent = group.title;
      block.append(groupTitle);
      const groupMeta = metaList(group.meta);
      if (groupMeta) block.append(groupMeta);
      block.append(artifactGrid(group.artifacts, `${step.title} · ${group.title}`));
      item.append(block);
    }
    if (step.key === 'scene' && data.seams) {
      item.append(seamTable(data.seams));
    } else if (step.key === 'scene' && data.seamsAvailable) {
      const measure = document.createElement('button');
      measure.type = 'button';
      measure.className = 'quiet-button pipeline-action';
      measure.textContent = 'Nähte messen';
      measure.title = 'Misst, wie weit benachbarte Ansichten dieselbe Oberfläche auseinander platzieren (|log r₁/r₂|).';
      measure.addEventListener('click', () => void measureSeams(data.id, measure));
      item.append(measure);
    }
    if (['done', 'skipped'].includes(step.status) && !step.artifacts.length && !step.groups.length) {
      text.textContent += ' Für diesen Auftrag wurden keine Zwischenergebnisse gespeichert.';
    }
    items.push(item);
  }
  pipelineSteps.replaceChildren(...items);
  pipelineSteps.scrollTop = scrollTop;
}

async function measureSeams(id, button) {
  button.disabled = true;
  button.textContent = 'Nähte werden gemessen …';
  try {
    const response = await fetch(`/api/pipeline/${encodeURIComponent(id)}/seams`, { cache: 'no-store' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Messung fehlgeschlagen.');
    pipelineSignature = '';
    await refreshPipeline();
  } catch (error) {
    button.disabled = false;
    button.textContent = 'Nähte messen';
    showToast(error.message, true);
  }
}

// Viewing directions of the panorama views in Three.js axes (the mesh is flipped about X).
const viewDirections = {
  front: [0, 0, -1], right: [1, 0, 0], back: [0, 0, 1], left: [-1, 0, 0], up: [0, 0.999, -0.04], down: [0, -0.999, -0.04],
};

function lookToward(view) {
  const direction = viewDirections[view];
  if (!direction) return;
  const damping = controls.enableDamping;
  controls.enableDamping = false;
  controls.target.copy(camera.position).add(new THREE.Vector3(...direction).normalize().multiplyScalar(3));
  controls.update();
  controls.enableDamping = damping;
}

function markSelectedArtifact() {
  for (const tile of pipelineSteps.querySelectorAll('.artifact-tile')) {
    tile.setAttribute('aria-pressed', String(tile.dataset.url === selectedArtifactUrl));
  }
}

function closeArtifactImage() {
  if (artifactViewer.hidden) return;
  artifactViewer.hidden = true;
  artifactImage.removeAttribute('src');
  if (selectedArtifactUrl && currentMeshSource !== selectedArtifactUrl) selectedArtifactUrl = null;
  markSelectedArtifact();
}

async function openArtifact(artifact, label) {
  selectedArtifactUrl = artifact.url;
  markSelectedArtifact();
  if (artifact.kind === 'image') {
    artifactImage.src = artifact.url;
    artifactImage.alt = label;
    $('#artifact-label').textContent = label;
    $('#artifact-open').href = artifact.url;
    artifactViewer.hidden = false;
    return;
  }
  closeArtifactImage();
  selectedArtifactUrl = artifact.url;
  markSelectedArtifact();
  const version = ++loadVersion;
  $('#viewer-empty').hidden = true;
  loadingPanel.hidden = false;
  loadingDetail.textContent = 'Zwischenergebnis wird geladen …';
  try {
    if (!await loadSplatMesh(artifact.url, version, 'Zwischenergebnis wird geladen')) return;
    currentMeshSource = artifact.url;
    setCompareEnabled(false);
    setViewerMode(pipelineData?.mode || 'image');
    // A single view's Gaussians lie in its own direction; face it.
    lookToward(artifact.view);
    $('#splat-badge-label').textContent = label;
    $('#splat-badge-back').hidden = !scenes.some((item) => item.id === activeSceneId);
    splatBadge.hidden = false;
  } catch (error) {
    if (version === loadVersion) showToast(`Zwischenergebnis konnte nicht geladen werden: ${error.message}`, true);
    console.error(error);
  } finally {
    if (version === loadVersion) loadingPanel.hidden = true;
  }
}

async function cancelActiveJob() {
  if (jobCancelButton.disabled) return;
  jobCancelButton.disabled = true;
  if (!activeJob) {
    uploadAbort?.abort();
    return;
  }
  $('#job-stage').textContent = 'Wird abgebrochen …';
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(activeJob.id)}/cancel`, { method: 'POST' });
    // 409: the job finished meanwhile; polling reports its final state.
    if (!response.ok && response.status !== 409) {
      throw new Error((await response.json()).error || 'Abbrechen fehlgeschlagen.');
    }
  } catch (error) {
    jobCancelButton.disabled = false;
    showToast(error.message, true);
  }
}

async function pollJob(id) {
  if (pollingJobId === id) return;
  pollingJobId = id;
  try {
    while (activeJob?.id === id) {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(id)}`, { cache: 'no-store' });
        const job = await response.json();
        if (activeJob?.id !== id) return;
        if (!response.ok) throw new Error(job.error || 'Status konnte nicht geladen werden.');
        activeJob.started = jobStartedAt(job, activeJob.started);
        showJobStatus(job);
        if (job.status === 'ready') {
          await fetchScenes();
          const loaded = await selectScene(id);
          activeJob = null;
          uploadButton.disabled = false;
          jobBanner.hidden = true;
          if (loaded) showToast('Deine 3D-Szene ist fertig.');
          return;
        }
        if (job.status === 'cancelled') {
          endJobStatus();
          showToast('Verarbeitung abgebrochen.');
          return;
        }
        if (job.status === 'error') throw new Error(job.error || 'Die Szene konnte nicht berechnet werden.');
      } catch (error) {
        endJobStatus();
        showToast(error.message, true);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 850));
    }
  } finally {
    if (pollingJobId === id) pollingJobId = null;
  }
}

uploadButton.addEventListener('click', () => fileInput.click());
jobCancelButton.addEventListener('click', () => void cancelActiveJob());
settingsButton.addEventListener('click', () => {
  stopMovement();
  setSceneMenuOpen(false);
  updateDepthSettings();
  updateFillSettings();
  updateSharpSettings();
  if (!settingsDialog.open) settingsDialog.showModal();
  fillToolSelect.focus();
  void fetchCapabilities();
});
$('#settings-close').addEventListener('click', () => settingsDialog.close());
depthEstimatorSelect.addEventListener('change', () => {
  const estimator = depthEstimatorSelect.value;
  if (estimator === 'sharp' || (estimator === 'da3' && da3Ready)) {
    selectedDepthEstimator = estimator;
    saveDepthEstimator();
  }
  updateDepthSettings();
  updateFillSettings();
  updateSharpSettings();
});
sharpResolutionSelect.addEventListener('change', () => {
  if (!Object.hasOwn(sharpResolutionDescriptions, sharpResolutionSelect.value)) return;
  selectedSharpResolution = sharpResolutionSelect.value;
  storeChoice(sharpResolutionStorageKey, selectedSharpResolution);
  updateSharpSettings();
});
depthAlignmentSelect.addEventListener('change', () => {
  if (!['off', 'da2'].includes(depthAlignmentSelect.value)) return;
  selectedDepthAlignment = depthAlignmentSelect.value;
  storeChoice(depthAlignmentStorageKey, selectedDepthAlignment);
  updateSharpSettings();
});
upscaleSelect.addEventListener('change', () => {
  if (!['off', 'auto'].includes(upscaleSelect.value)) return;
  selectedUpscale = upscaleSelect.value;
  storeChoice(upscaleStorageKey, selectedUpscale);
  updateSharpSettings();
});
fillToolSelect.addEventListener('change', () => {
  if (!Object.hasOwn(fillToolDefinitions, fillToolSelect.value) && !Object.hasOwn(fillTools, fillToolSelect.value)) return;
  selectedFillTool = fillToolSelect.value;
  try {
    localStorage.setItem(fillToolStorageKey, selectedFillTool);
  } catch {
    // Selection remains usable if browser storage is unavailable.
  }
  updateFillSettings();
});
menuToggle.addEventListener('click', () => setSceneMenuOpen(sceneMenu.hidden));
$('#menu-close').addEventListener('click', () => { setSceneMenuOpen(false); menuToggle.focus(); });
menuBackdrop.addEventListener('click', () => setSceneMenuOpen(false));
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !sceneMenu.hidden) setSceneMenuOpen(false);
});
compareButton.addEventListener('click', () => setCompareEnabled(!compareEnabled));
compareDivider.addEventListener('pointerdown', (event) => {
  compareDragging = true;
  compareDivider.setPointerCapture(event.pointerId);
  moveCompareDivider(event.clientX);
});
compareDivider.addEventListener('pointermove', (event) => {
  if (compareDragging && compareDivider.hasPointerCapture(event.pointerId)) moveCompareDivider(event.clientX);
});
const stopCompareDrag = () => { compareDragging = false; };
compareDivider.addEventListener('pointerup', stopCompareDrag);
compareDivider.addEventListener('pointercancel', stopCompareDrag);
compareDivider.addEventListener('lostpointercapture', stopCompareDrag);
compareDivider.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    event.preventDefault();
    updateComparePosition(comparePosition + (event.key === 'ArrowRight' ? 2 : -2));
  } else if (event.key === 'Home') {
    updateComparePosition(0);
  } else if (event.key === 'End') {
    updateComparePosition(100);
  }
});
panoramaButton.addEventListener('click', () => chooseProcessingMode(fillDefinition(selectedFillTool).mode));
fillCoverageSelect.addEventListener('change', () => {
  selectedFillCoverage = fillCoverageSelect.value;
  storeChoice(fillCoverageStorageKey, selectedFillCoverage);
  updateFillSettings();
});

imageButton.addEventListener('click', () => chooseProcessingMode('image'));
$('#processing-cancel').addEventListener('click', dismissProcessingChoice);
processingDialog.addEventListener('cancel', (event) => {
  event.preventDefault();
  dismissProcessingChoice();
});
processingPreview.addEventListener('error', () => { processingPreview.hidden = true; });
processingPreview.addEventListener('load', () => { if (pendingImage) processingPreview.hidden = false; });
$('#scene-previous').addEventListener('click', () => stepScene(-1));
$('#scene-next').addEventListener('click', () => stepScene(1));
fileInput.addEventListener('change', () => {
  offerProcessingChoice(fileInput.files?.[0]);
  fileInput.value = '';
});

window.addEventListener('dragenter', (event) => {
  if (!event.dataTransfer?.types?.includes('Files')) return;
  event.preventDefault();
  dragDepth += 1;
  dropOverlay.hidden = false;
});
window.addEventListener('dragover', (event) => {
  if (!event.dataTransfer?.types?.includes('Files')) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) dropOverlay.hidden = true;
});
window.addEventListener('drop', (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropOverlay.hidden = true;
  offerProcessingChoice(event.dataTransfer?.files?.[0]);
});

pipelineToggle.addEventListener('click', () => setPipelineOpen(!pipelineOpen));
$('#pipeline-close').addEventListener('click', () => { setPipelineOpen(false); pipelineToggle.focus(); });
$('#artifact-close').addEventListener('click', closeArtifactImage);
$('#splat-badge-back').addEventListener('click', () => {
  if (!activeSceneId) return;
  currentMeshSource = null;
  void selectScene(activeSceneId);
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape' || !sceneMenu.hidden || settingsDialog.open || processingDialog.open) return;
  if (!artifactViewer.hidden) closeArtifactImage();
  else if (pipelineOpen) setPipelineOpen(false);
});

$('#reset-view').addEventListener('click', () => (renderer.xr.isPresenting ? resetXrRig() : resetCamera()));

uploadButton.disabled = true;
settingsButton.disabled = true;
try {
  await Promise.all([fetchScenes(), fetchCapabilities()]);
  if (scenes.length) await selectScene(scenes[0].id);
  else $('#viewer-empty').hidden = false;
} catch (error) {
  $('#viewer-empty').hidden = false;
  showToast(error.message, true);
}
await resumeActiveJob();
settingsButton.disabled = false;
if (!activeJob) {
  uploadButton.disabled = false;
}
