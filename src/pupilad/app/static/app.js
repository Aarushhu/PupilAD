import { DEFAULT_SETTINGS, SETTING_LABELS, TASKS, requestAbort, runBattery, taskIcon } from "./battery.js";
import { renderTraceChart } from "./chart.js";
import {
  Recorder,
  grabFrame,
  levelMeter,
  listCameras,
  mediaSupport,
  openCamera,
  openMicrophone,
  regionBrightness,
  stopStream,
} from "./media.js";

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "pupilad.settings.v2";
const TASKS_KEY = "pupilad.tasks";
const CAMERA_KEY = "pupilad.camera";
const CHECK_INTERVAL_MS = 700;
const HOLD_TO_STOP_MS = 1500;
const ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" aria-hidden="true"><path d="M5 12l5 5L20 7"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" aria-hidden="true"><path d="M12 7v6M12 17h.01"/></svg>';
const ICON_WAIT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" aria-hidden="true"><path d="M8 12h.01M12 12h.01M16 12h.01"/></svg>';

const escapeHtml = (text) =>
  String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, { cache: "no-store", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function loadJson(key, fallback) {
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(key) || "{}") };
  } catch {
    return { ...fallback };
  }
}

let settings = loadJson(SETTINGS_KEY, DEFAULT_SETTINGS);
let enabledTasks = loadJson(TASKS_KEY, Object.fromEntries(TASKS.map((t) => [t.id, true])));
const selectedTasks = () => TASKS.filter((t) => enabledTasks[t.id] !== false).map((t) => t.id);

function showView(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === (name === "camera" ? "setup" : name));
  }
  $("view-setup").hidden = name !== "setup";
  $("view-camera").hidden = name !== "camera";
  $("view-results").hidden = name !== "results";
  if (name !== "camera") closeCameraCheck(true);
  if (name === "results") loadSessions();
  window.scrollTo(0, 0);
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showView(tab.dataset.view));
}

function renderTaskList() {
  $("task-list").replaceChildren(
    ...TASKS.map((task) => {
      const row = document.createElement("label");
      row.className = "taskrow";
      row.innerHTML = `
        <div class="ticon ${task.tone}">${taskIcon(task.id)}</div>
        <div><b>${task.name}</b><span>${task.science} · ${task.blurb.toLowerCase()}</span></div>`;
      const box = document.createElement("input");
      box.type = "checkbox";
      box.className = "switch";
      box.setAttribute("aria-label", `Include ${task.name}`);
      box.checked = enabledTasks[task.id] !== false;
      row.classList.toggle("off", !box.checked);
      box.addEventListener("change", () => {
        enabledTasks[task.id] = box.checked;
        row.classList.toggle("off", !box.checked);
        localStorage.setItem(TASKS_KEY, JSON.stringify(enabledTasks));
      });
      row.append(box);
      return row;
    })
  );
}

function renderSettings() {
  $("settings-grid").replaceChildren(
    ...Object.entries(SETTING_LABELS).map(([key, text]) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.min = "0";
      input.inputMode = "decimal";
      input.value = settings[key];
      input.addEventListener("change", () => {
        const value = Number(input.value);
        if (Number.isFinite(value) && value >= 0) settings[key] = value;
        else input.value = settings[key];
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
      });
      label.append(text, input);
      return label;
    })
  );
}

$("reset-settings").addEventListener("click", () => {
  settings = { ...DEFAULT_SETTINGS };
  localStorage.removeItem(SETTINGS_KEY);
  renderSettings();
});

function chip(text, tone = "") {
  const icon = tone === "" ? ICON_OK : ICON_WARN;
  return `<span class="pill ${tone}">${icon}${escapeHtml(text)}</span>`;
}

async function renderDeviceChips() {
  const support = mediaSupport();
  const chips = [];
  if (!support.secure) chips.push(chip("Camera needs https or localhost", "bad"));
  else if (!support.camera || !support.recorder) chips.push(chip("This browser can't record video", "bad"));
  else chips.push(chip("Camera recording supported"));
  try {
    const battery = await navigator.getBattery?.();
    if (battery && !battery.charging && battery.level < 0.4) {
      chips.push(chip(`Battery ${Math.round(battery.level * 100)}%: plug in`, "warn"));
    }
  } catch {
  }
  $("device-chips").innerHTML = chips.join("");
}

let existingFiles = [];
let checkTimer = null;

function showSetupError(message) {
  $("setup-error").hidden = !message;
  $("setup-error").textContent = message || "";
}

$("subject-id").addEventListener("input", () => {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(checkSubject, 300);
});

async function checkSubject() {
  const subjectId = $("subject-id").value.trim();
  const warning = $("subject-warning");
  existingFiles = [];
  warning.hidden = true;
  showSetupError("");
  if (!subjectId) return;
  try {
    existingFiles = (await api("/api/session/check", { subject_id: subjectId })).existing;
  } catch (error) {
    showSetupError(error.message);
    return;
  }
  if (existingFiles.length) {
    warning.innerHTML = `
      <span><b>${escapeHtml(subjectId)}</b> already has recorded data (${existingFiles.length} files).</span>
      <label class="check-line"><input type="checkbox" id="overwrite"> Replace the existing files</label>`;
    warning.hidden = false;
  }
}

const overwriteChecked = () => Boolean($("overwrite")?.checked);

async function loadInfo() {
  try {
    const info = await api("/api/info");
    if (info.lan_url) {
      $("lan-url").textContent = info.lan_url;
      $("lan-card").hidden = false;
    }
  } catch {
  }
}

const camera = {
  stream: null,
  microphone: null,
  meter: null,
  timer: null,
  busy: false,
  history: [],
  canvas: document.createElement("canvas"),
  usable: false,
};

function closeCameraCheck(releaseStreams) {
  clearInterval(camera.timer);
  camera.timer = null;
  cancelAnimationFrame(camera.micFrame);
  if (releaseStreams) {
    camera.meter?.close();
    camera.meter = null;
    stopStream(camera.stream);
    stopStream(camera.microphone);
    camera.stream = null;
    camera.microphone = null;
    $("cam-video").srcObject = null;
  }
}

function showCameraError(message) {
  $("camera-error").hidden = !message;
  $("camera-error").textContent = message || "";
}

function validateSetup() {
  const subjectId = $("subject-id").value.trim();
  if (!subjectId) return "Enter a participant ID first.";
  if (!/^[A-Za-z0-9_-]{1,40}$/.test(subjectId)) return "Participant ID may only contain letters, digits, '-' and '_' (max 40).";
  if (!selectedTasks().length) return "Switch on at least one task.";
  if (existingFiles.length && !overwriteChecked()) return `${subjectId} already has recorded data. Tick "Replace the existing files" or use a different ID.`;
  return null;
}

$("to-camera").addEventListener("click", async () => {
  clearTimeout(checkTimer);
  await checkSubject();
  const problem = validateSetup();
  if (problem) return showSetupError(problem);
  showSetupError("");
  showView("camera");
  await startCameraCheck();
});

$("cam-back").addEventListener("click", () => showView("setup"));

async function populateCameraSelect() {
  const cameras = await listCameras().catch(() => []);
  const select = $("camera-select");
  $("camera-select-wrap").hidden = cameras.length < 2;
  const current = camera.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  select.replaceChildren(
    ...cameras.map((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      option.selected = device.deviceId === current;
      return option;
    })
  );
}

$("camera-select").addEventListener("change", async (event) => {
  localStorage.setItem(CAMERA_KEY, event.target.value);
  closeCameraCheck(false);
  stopStream(camera.stream);
  camera.stream = null;
  await startCameraCheck();
});

function renderChecks(items) {
  $("cam-checks").innerHTML = items
    .map(({ state, title, detail }) => {
      const icon = state === "ok" ? ICON_OK : state === "warn" ? ICON_WARN : ICON_WAIT;
      return `<div class="check ${state}"><span class="dot">${icon}</span><div>${title}<small>${detail}</small></div></div>`;
    })
    .join("");
}

function evaluateFrame(result, brightness) {
  const face = result.face;
  const checks = [];
  let tip = null;
  if (!face) {
    checks.push({ state: "warn", title: "Face centred", detail: "No face found yet" });
    tip = "Move so your whole face is inside the oval, looking at the screen.";
  } else {
    const cx = face.x + face.w / 2;
    const cy = face.y + face.h / 2;
    const centred = cx > 0.32 && cx < 0.68 && cy > 0.25 && cy < 0.7;
    checks.push({ state: centred ? "ok" : "warn", title: "Face centred", detail: centred ? "Inside the oval" : "Move toward the middle" });
    if (!centred) tip ??= "Move your face toward the middle of the oval.";
  }
  const eyesOk = result.eyes >= 2 || result.pupil_px !== null;
  checks.push({
    state: !face ? "wait" : eyesOk ? "ok" : "warn",
    title: "Eyes found",
    detail: !face ? "Waiting for a face" : eyesOk ? "No glare on glasses" : "Look straight at the screen",
  });
  if (face && !eyesOk) tip ??= "Look straight at the screen. If you wear glasses, tilt them slightly to remove glare.";
  let lightState = "wait";
  let lightDetail = "Waiting for a face";
  if (face && brightness) {
    const unevenness = Math.abs(brightness.left - brightness.right) / Math.max(1, brightness.mean);
    if (brightness.mean < 70) {
      lightState = "warn";
      lightDetail = "Too dark";
      tip ??= "It's a little dark. Turn on a light or face a window.";
    } else if (brightness.mean > 215) {
      lightState = "warn";
      lightDetail = "Too bright";
      tip ??= "Too much light on your face. Move away from a bright lamp or window.";
    } else if (unevenness > 0.35) {
      lightState = "warn";
      lightDetail = "One side is in shadow";
      tip ??= "One side of your face is in shadow. Face the light more evenly.";
    } else {
      lightState = "ok";
      lightDetail = "Even light";
    }
  }
  checks.push({ state: lightState, title: "Lighting", detail: lightDetail });
  let distanceState = "wait";
  let distanceDetail = "Waiting for a face";
  if (face) {
    if (face.w < 0.2) {
      distanceState = "warn";
      distanceDetail = "Too far away";
      tip ??= "Move a little closer to the camera.";
    } else if (face.w > 0.6) {
      distanceState = "warn";
      distanceDetail = "Too close";
      tip ??= "Move back a little.";
    } else {
      distanceState = "ok";
      distanceDetail = "Good distance";
    }
  }
  checks.push({ state: distanceState, title: "Distance", detail: distanceDetail });
  const aligned = checks[0].state === "ok" && distanceState === "ok";
  return { checks, tip, aligned };
}

async function runCheck() {
  const video = $("cam-video");
  if (camera.busy || !video.videoWidth) return;
  camera.busy = true;
  try {
    const context = grabFrame(video, camera.canvas);
    const blob = await new Promise((resolve) => camera.canvas.toBlob(resolve, "image/jpeg", 0.85));
    const response = await fetch("/api/camera/check", { method: "POST", body: blob });
    if (!response.ok) return;
    const result = await response.json();
    const brightness = result.face ? regionBrightness(context, result.face) : null;
    const { checks, tip, aligned } = evaluateFrame(result, brightness);
    camera.history.push(result.pupil_px !== null);
    if (camera.history.length > 12) camera.history.shift();
    const rate = camera.history.filter(Boolean).length / camera.history.length;
    renderChecks(checks);
    $("pupil-rate").textContent = `${Math.round(rate * 100)}%`;
    $("pupil-meter").style.width = `${Math.round(rate * 100)}%`;
    $("pupil-meter").style.background = rate >= 0.5 ? "var(--brand)" : "var(--accent)";
    $("oval").classList.toggle("ok", aligned);
    $("cam-hint").hidden = !tip;
    $("cam-hint").textContent = tip || "";
  } catch {
  } finally {
    camera.busy = false;
  }
}

function animateMic() {
  if (!camera.meter) return;
  const level = camera.meter.level();
  $("mic-meter").style.width = `${Math.round(level * 100)}%`;
  if (level > 0.15) $("mic-status").textContent = "Hearing you";
  camera.micFrame = requestAnimationFrame(animateMic);
}

async function startCameraCheck() {
  showCameraError("");
  const button = $("cam-continue");
  button.disabled = true;
  button.textContent = "Looks good, continue";
  camera.usable = false;
  camera.history = [];
  renderChecks([
    { state: "wait", title: "Face centred", detail: "Starting the camera" },
    { state: "wait", title: "Eyes found", detail: "" },
    { state: "wait", title: "Lighting", detail: "" },
    { state: "wait", title: "Distance", detail: "" },
  ]);
  const support = mediaSupport();
  if (!support.secure || !support.camera || !support.recorder) {
    showCameraError(
      !support.secure
        ? "The camera only works over https or on this computer's own address. Use the phone link shown on the setup page."
        : "This browser can't record video. Try Chrome, Edge or Safari."
    );
    button.disabled = false;
    button.textContent = "Continue without video";
    return;
  }
  try {
    if (!camera.stream) camera.stream = await openCamera(localStorage.getItem(CAMERA_KEY));
  } catch (error) {
    try {
      if (error.name === "OverconstrainedError" || error.name === "NotFoundError") {
        localStorage.removeItem(CAMERA_KEY);
        camera.stream = await openCamera(null);
      } else throw error;
    } catch (second) {
      showCameraError(
        second.name === "NotAllowedError"
          ? "Camera permission was blocked. Allow the camera in the browser's site settings, then try again."
          : `Could not start the camera: ${second.message}`
      );
      button.disabled = false;
      button.textContent = "Continue without video";
      return;
    }
  }
  const video = $("cam-video");
  video.srcObject = camera.stream;
  await video.play().catch(() => {});
  camera.usable = true;
  button.disabled = false;
  populateCameraSelect();
  if (selectedTasks().includes("fluency") && !camera.microphone) {
    try {
      camera.microphone = await openMicrophone();
      camera.meter = levelMeter(camera.microphone);
      await camera.meter?.resume();
    } catch {
      camera.microphone = null;
    }
  }
  $("mic-check").hidden = !camera.microphone;
  $("mic-status").textContent = camera.microphone ? "Say something" : "";
  if (selectedTasks().includes("fluency") && !camera.microphone) {
    showCameraError("The microphone isn't available, so Animal Sprint answers won't be recorded.");
  }
  animateMic();
  clearInterval(camera.timer);
  camera.timer = setInterval(runCheck, CHECK_INTERVAL_MS);
  runCheck();
}

$("cam-continue").addEventListener("click", () => startSession());

function deviceInfo() {
  const ua = navigator.userAgent;
  const touch = navigator.maxTouchPoints > 0;
  const shortSide = Math.min(screen.width, screen.height);
  let kind = "Computer";
  if (/iPad/.test(ua) || (touch && /Macintosh/.test(ua))) kind = "Tablet";
  else if (/iPhone|Android.+Mobile|Mobi/.test(ua)) kind = "Phone";
  else if (/Android/.test(ua) && touch && shortSide >= 600) kind = "Tablet";
  return {
    kind,
    touch,
    screen: [screen.width, screen.height],
    viewport: [innerWidth, innerHeight],
    pixel_ratio: devicePixelRatio,
  };
}

const session = {
  running: false,
  recorder: null,
  events: [],
  wakeLock: null,
};

function marker(label) {
  const t = performance.now();
  return api("/api/session/marker", { label, t: t / 1000, epoch_ms: performance.timeOrigin + t }).catch(() => {});
}

async function enterFullscreen() {
  try {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen({ navigationUI: "hide" });
    }
    await navigator.keyboard?.lock?.(["Escape"]);
  } catch {
  }
  try {
    session.wakeLock = await navigator.wakeLock?.request("screen");
  } catch {
  }
}

async function leaveFullscreen() {
  navigator.keyboard?.unlock?.();
  session.wakeLock?.release().catch(() => {});
  session.wakeLock = null;
  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
}

document.addEventListener("visibilitychange", () => {
  if (!session.running) return;
  const label = document.hidden ? "PAGE_HIDDEN" : "PAGE_VISIBLE";
  session.events.push({ event: label, t: performance.now() / 1000 });
  marker(label);
  if (!document.hidden && session.wakeLock === null) navigator.wakeLock?.request("screen").then((lock) => (session.wakeLock = lock)).catch(() => {});
});

window.addEventListener(
  "keydown",
  (event) => {
    if (session.running && event.key === "Escape" && event.shiftKey) {
      event.preventDefault();
      event.stopImmediatePropagation();
      requestAbort();
    }
  },
  true
);

let holdTimer = null;
document.addEventListener("pointerdown", (event) => {
  const button = event.target.closest("[data-hold-stop]");
  if (!button) return;
  event.preventDefault();
  button.classList.add("holding");
  clearTimeout(holdTimer);
  holdTimer = setTimeout(() => {
    button.classList.remove("holding");
    requestAbort();
  }, HOLD_TO_STOP_MS);
});
for (const type of ["pointerup", "pointercancel", "pointerleave"]) {
  document.addEventListener(
    type,
    (event) => {
      const button = event.target.closest?.("[data-hold-stop]");
      if (!button && type === "pointerleave") return;
      clearTimeout(holdTimer);
      document.querySelectorAll(".holding").forEach((b) => b.classList.remove("holding"));
    },
    true
  );
}

window.addEventListener("pagehide", () => {
  if (!session.running) return;
  navigator.sendBeacon(
    "/api/session/finish",
    JSON.stringify({ aborted: true, recording: session.recorder?.info(), events: session.events })
  );
});

function playScreen(html) {
  $("play").innerHTML = `<div class="play-light"><div class="content">${html}</div></div>`;
}

async function startSession() {
  const problem = validateSetup();
  if (problem) {
    showView("setup");
    return showSetupError(problem);
  }
  const subjectId = $("subject-id").value.trim();
  const tasks = selectedTasks();
  const record = camera.usable && camera.stream !== null;
  closeCameraCheck(false);
  camera.meter?.close();
  camera.meter = null;
  session.events = [];
  session.recorder = null;
  $("shell").hidden = true;
  $("play").hidden = false;
  let audioInfo = null;
  const ctx = {
    settings: { ...settings },
    tasks,
    microphone: camera.microphone,
    marker,
    saveResults: (task, rows) => api("/api/session/results", { task, rows }),
    begin: async () => {
      await enterFullscreen();
      const device = deviceInfo();
      if (record) {
        const { deviceId, groupId, ...settings } = camera.stream.getVideoTracks()[0]?.getSettings() || {};
        device.camera = settings;
      }
      await api("/api/session/start", {
        subject_id: subjectId,
        session: $("session-number").value.trim(),
        notes: $("session-notes").value.trim(),
        record_webcam: record,
        overwrite: overwriteChecked(),
        settings: ctx.settings,
        tasks,
        device,
      });
      session.running = true;
      if (record) {
        session.recorder = new Recorder(camera.stream, "video");
        await session.recorder.start();
      }
      marker("SESSION_BEGAN");
    },
  };
  let outcome = { aborted: true };
  let failure = null;
  try {
    outcome = await runBattery(ctx);
  } catch (error) {
    failure = error;
  }
  audioInfo = ctx.audioInfo ?? null;
  const started = session.running;
  if (started) await marker(outcome.aborted ? "SESSION_STOPPED" : "SESSION_COMPLETE");
  playScreen(`
    <div class="saving">
      <div class="done-badge">${ICON_OK}</div>
      <h1 class="h1">${failure ? "Something went wrong" : outcome.aborted ? "Session stopped" : "All done, thank you!"}</h1>
      <p class="lede" id="save-text">${started ? "Saving the recording…" : ""}</p>
      <div class="progress" ${started && session.recorder ? "" : "hidden"}><i id="save-bar"></i></div>
    </div>`);
  let progressTimer = null;
  let recording = null;
  if (session.recorder) {
    progressTimer = setInterval(() => {
      $("save-bar").style.width = `${Math.round(session.recorder.progress() * 100)}%`;
    }, 250);
    recording = await session.recorder.stop();
    clearInterval(progressTimer);
  }
  session.running = false;
  let saveError = null;
  if (started) {
    try {
      await api("/api/session/finish", {
        aborted: outcome.aborted || failure !== null,
        recording,
        audio: audioInfo,
        practice: ctx.practice,
        events: session.events,
      });
    } catch (error) {
      saveError = error.message;
    }
  }
  closeCameraCheck(true);
  await leaveFullscreen();
  const uploadProblem = recording?.upload_error || audioInfo?.upload_error;
  const messages = [];
  if (failure) messages.push(`The session stopped because of an error: ${escapeHtml(failure.message)}`);
  if (saveError) messages.push(`Could not save the session: ${escapeHtml(saveError)}`);
  if (uploadProblem) messages.push(`Part of the recording could not be uploaded: ${escapeHtml(uploadProblem)}`);
  if (!started) messages.push("The session was stopped before it began, so nothing was saved.");
  $("play").innerHTML = `
    <div class="play-light"><div class="content">
      <div class="saving">
        <div class="done-badge">${ICON_OK}</div>
        <h1 class="h1">${failure ? "Something went wrong" : outcome.aborted ? "Session stopped" : "All done, thank you!"}</h1>
        <p class="lede">${messages.length ? messages.join("<br>") : outcome.aborted ? "Finished tasks were saved." : "Please hand the device back to the researcher."}</p>
        <button class="btn btn-primary" type="button" id="play-close">${started ? "See results" : "Back to setup"}</button>
      </div>
    </div></div>`;
  await new Promise((resolve) => $("play-close").addEventListener("click", resolve, { once: true }));
  $("play").hidden = true;
  $("play").innerHTML = "";
  $("shell").hidden = false;
  existingFiles = [];
  $("subject-warning").hidden = true;
  if (!started) return showView("setup");
  $("subject-id").value = "";
  showView("results");
  openSession(subjectId);
}

const percent = (value) => (value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`);
const msText = (value) => (value === null || value === undefined ? "–" : `${value} ms`);
const formatDate = (iso) => (iso ? new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "–");

function qualityPill(row) {
  if (row.processing) return '<span class="pill plain">Processing…</span>';
  if (row.pupil_detect_rate !== null && row.pupil_detect_rate !== undefined) {
    const tone = row.pupil_detect_rate >= 0.5 ? "" : "warn";
    return `<span class="pill ${tone}">Pupil ${percent(row.pupil_detect_rate)}</span>`;
  }
  if (row.has_trace) return '<span class="pill">Pupil data</span>';
  if (row.has_video) return '<span class="pill plain">Video</span>';
  return '<span class="pill plain">No video</span>';
}

let selectedSubject = null;
let sessionsTimer = null;

async function loadSessions() {
  clearTimeout(sessionsTimer);
  let data;
  try {
    data = await api("/api/sessions");
  } catch (error) {
    $("sessions-list").innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    return;
  }
  $("data-folder").textContent = data.logs_dir;
  const list = $("sessions-list");
  if (!data.sessions.length) {
    list.innerHTML = '<div class="empty">No sessions recorded yet.</div>';
    return;
  }
  list.replaceChildren(
    ...data.sessions.map((row) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `srow${row.subject_id === selectedSubject ? " sel" : ""}`;
      button.dataset.subject = row.subject_id;
      const when = row.source === "psychopy" ? "PsychoPy run" : formatDate(row.started_at);
      const extra = [when, row.device, row.aborted ? "stopped early" : null].filter(Boolean).join(" · ");
      button.innerHTML = `
        <b>${escapeHtml(row.subject_id)}${row.session ? ` · session ${escapeHtml(row.session)}` : ""}</b>
        ${qualityPill(row)}
        <span class="sub">${escapeHtml(extra)}</span>`;
      button.addEventListener("click", () => openSession(row.subject_id));
      return button;
    })
  );
  if (data.sessions.some((row) => row.processing) && !$("view-results").hidden) {
    sessionsTimer = setTimeout(loadSessions, 3000);
  }
}

function tile(taskId, label, value, sub) {
  const task = TASKS.find((t) => t.id === taskId);
  return `
    <div class="mtile">
      <span class="label"><span class="ticon ${task.tone}">${taskIcon(taskId)}</span>${escapeHtml(label)}</span>
      <span class="value">${escapeHtml(value ?? "–")}</span>
      <span class="sub">${escapeHtml(sub)}</span>
    </div>`;
}

function scoreTiles(summary, files) {
  const tiles = [];
  if (summary.nback) {
    const n = summary.nback;
    tiles.push(tile("nback", "Letter Match", percent(n.accuracy), `hit rate ${percent(n.hit_rate)} · false alarms ${percent(n.false_alarm_rate)}`));
  }
  if (summary.digitspan) {
    const d = summary.digitspan;
    tiles.push(tile("digitspan", "Number Recall", `${d.forward_span} / ${d.backward_span}`, "forward / reverse span"));
  }
  if (summary.stroop) {
    const s = summary.stroop;
    const sign = s.interference_ms > 0 ? "+" : "";
    tiles.push(
      tile("stroop", "Color Clash", s.interference_ms === null ? "–" : `${sign}${s.interference_ms} ms`, `slower on clashing words · ${percent(s.accuracy)} correct`)
    );
  }
  if (files["_fluency_audio.webm"] || files["_fluency_audio.mp4"]) {
    tiles.push(tile("fluency", "Animal Sprint", "Audio saved", "count the animals from the recording"));
  }
  return tiles.length ? `<div class="tiles">${tiles.join("")}</div>` : '<p class="hint">No task results saved.</p>';
}

function metaLine(meta) {
  const parts = [];
  if (meta.started_at) parts.push(`Started ${formatDate(meta.started_at)}`);
  if (meta.device?.kind) parts.push(meta.device.kind);
  if (meta.webcam?.frames) {
    const fps = meta.webcam.mean_fps ? `${meta.webcam.mean_fps.toFixed(1)} fps` : "";
    const size = meta.webcam.frame_size ? `${meta.webcam.frame_size.join("×")}` : "";
    parts.push(`Video ${[size, fps].filter(Boolean).join(", ")}, ${Math.round(meta.webcam.duration_s)} s`);
  }
  if (meta.practice?.length) parts.push(`${meta.practice.length} practice round${meta.practice.length > 1 ? "s" : ""}`);
  if (meta.aborted) parts.push("Stopped early; finished tasks were saved");
  if (meta.notes) parts.push(`Notes: ${meta.notes}`);
  return parts.map(escapeHtml).join(" · ");
}

function fileLinks(subjectId, files) {
  return Object.entries(files)
    .filter(([, present]) => present)
    .map(([suffix]) => `<a href="/files/${encodeURIComponent(subjectId + suffix)}">${escapeHtml(subjectId + suffix)}</a>`)
    .join("");
}

let pollTimer = null;

async function openSession(subjectId) {
  clearTimeout(pollTimer);
  selectedSubject = subjectId;
  for (const row of document.querySelectorAll(".srow")) row.classList.toggle("sel", row.dataset.subject === subjectId);
  const detail = $("session-detail");
  let data;
  try {
    data = await api(`/api/sessions/${encodeURIComponent(subjectId)}`);
  } catch (error) {
    detail.innerHTML = `<div class="card">${escapeHtml(error.message)}</div>`;
    return;
  }
  const meta = data.meta || {};
  detail.innerHTML = `
    <div class="detail-head">
      <h2>${escapeHtml(subjectId)}${meta.session ? ` · session ${escapeHtml(meta.session)}` : ""}</h2>
    </div>
    <p class="meta-line">${metaLine(meta)}</p>
    ${scoreTiles(data.summary, data.files)}
    <section class="card">
      <div class="analysis-row">
        <h3 class="h2">Pupil size over the session</h3>
        <button class="btn btn-ghost btn-small" type="button" id="analyze" hidden></button>
      </div>
      <div class="progress" id="job-bar" hidden><i></i></div>
      <p class="status-line" id="job-status"></p>
      <div id="chart-wrap"></div>
    </section>
    <section class="card">
      <h3 class="h2">Files</h3>
      <div class="file-links">${fileLinks(subjectId, data.files)}</div>
    </section>`;
  renderAnalysis(subjectId, data);
}

function renderAnalysis(subjectId, data) {
  const files = data.files;
  const hasRaw = files["_webcam_raw.webm"] || files["_webcam_raw.mp4"];
  const hasVideo = hasRaw || (files["_webcam.mp4"] && files["_webcam_frame_timestamps.csv"]);
  const hasTrace = files["_pupil_trace.csv"];
  const button = $("analyze");
  const status = $("job-status");
  const bar = $("job-bar");
  if (!hasVideo && !hasTrace) {
    status.textContent = "No webcam recording for this session.";
    return;
  }
  button.hidden = !hasVideo;
  button.textContent = hasTrace ? "Run pupil analysis again" : "Run pupil analysis";
  if (data.trial_counts) {
    const names = { nback: "Letter Match", digit_span: "Number Recall", stroop: "Color Clash" };
    const counts = Object.entries(data.trial_counts).map(([task, n]) => `${names[task] || task} ${n}`).join(", ");
    status.textContent = `Trials matched to pupil samples: ${counts}.`;
  } else if (!hasTrace) {
    status.textContent = "The pupil analysis hasn't run yet.";
  }

  const poll = async () => {
    const job = await api(`/api/sessions/${encodeURIComponent(subjectId)}/job`).catch(() => null);
    if (!job || selectedSubject !== subjectId) return;
    const running = job.state === "running";
    bar.hidden = !running;
    bar.firstElementChild.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    status.textContent = running ? `${job.message}… ${Math.round((job.progress || 0) * 100)}%` : job.message;
    status.classList.toggle("error", job.state === "error");
    button.disabled = running;
    if (running) pollTimer = setTimeout(poll, 700);
    else if (job.state === "done") {
      await loadSessions();
      await openSession(subjectId);
    }
  };

  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/api/sessions/${encodeURIComponent(subjectId)}/analyze`, {});
      poll();
    } catch (error) {
      status.textContent = error.message;
      status.classList.add("error");
      button.disabled = false;
    }
  });

  if (data.job?.state === "running") {
    button.disabled = true;
    poll();
  } else if (data.job?.state === "error") {
    status.textContent = data.job.message;
    status.classList.add("error");
  }
  if (hasTrace) {
    api(`/api/sessions/${encodeURIComponent(subjectId)}/trace`)
      .then((trace) => renderTraceChart($("chart-wrap"), trace))
      .catch((error) => ($("chart-wrap").textContent = error.message));
  }
}

renderTaskList();
renderSettings();
renderDeviceChips();
loadInfo();
showView("setup");
