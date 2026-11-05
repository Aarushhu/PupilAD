import { DEFAULT_SETTINGS, SETTING_LABELS, TASKS, requestAbort, runBattery } from "./battery.js";
import { renderTraceChart } from "./chart.js";

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "pupilad.settings";
const TASKS_KEY = "pupilad.tasks";

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

async function syncClock(samples = 15) {
  let best = null;
  for (let i = 0; i < samples; i++) {
    const sent = performance.now();
    const { t } = await api("/api/clock");
    const received = performance.now();
    const rtt = received - sent;
    if (best === null || rtt < best.rtt_ms) {
      best = { rtt_ms: rtt, offset_s: t - (sent + received) / 2000 };
    }
  }
  return { ...best, samples };
}

function showView(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === name);
  }
  $("view-setup").hidden = name !== "setup";
  $("view-results").hidden = name !== "results";
  setPreview(name === "setup");
  if (name === "results") loadSessions();
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => showView(tab.dataset.view));
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

function renderTaskList() {
  $("task-list").replaceChildren(
    ...TASKS.map((task) => {
      const label = document.createElement("label");
      label.className = "check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = enabledTasks[task.id] !== false;
      box.addEventListener("change", () => {
        enabledTasks[task.id] = box.checked;
        localStorage.setItem(TASKS_KEY, JSON.stringify(enabledTasks));
      });
      label.append(box, task.name);
      return label;
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

renderTaskList();
renderSettings();

const preview = $("camera-preview");
let statusTimer = null;

function setPreview(on) {
  clearInterval(statusTimer);
  if (on) {
    preview.src = `/api/camera/stream?ts=${Date.now()}`;
    statusTimer = setInterval(refreshCameraStatus, 1000);
    refreshCameraStatus();
  } else {
    preview.removeAttribute("src");
  }
}

function setStats(entries) {
  $("camera-stats").replaceChildren(
    ...entries.flatMap(([term, value, className]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      if (className) dd.className = className;
      return [dt, dd];
    })
  );
}

async function refreshCameraStatus() {
  try {
    const status = await api("/api/camera/status");
    if (status.error) {
      setStats([["Status", status.error, "error"]]);
      return;
    }
    const rate = status.pupil_detect_rate;
    setStats([
      ["Status", status.running ? "Live" : "Starting…"],
      ["Resolution", status.frame_size ? status.frame_size.join(" × ") : "–"],
      ["Frame rate", status.fps ? `${status.fps.toFixed(1)} fps` : "–"],
      [
        "Pupil detected",
        rate === null ? "–" : `${Math.round(rate * 100)}% of recent checks`,
        rate === null ? "" : rate >= 0.5 ? "ok" : "error",
      ],
      [
        "Pupil size",
        status.last_diameter_px ? `${status.last_diameter_px.toFixed(1)} px` : "–",
      ],
    ]);
  } catch (error) {
    setStats([["Status", `Server unreachable: ${error.message}`, "error"]]);
  }
}

for (let index = 0; index < 5; index++) {
  const option = document.createElement("option");
  option.value = index;
  option.textContent = `Camera ${index}`;
  $("camera-index").append(option);
}
api("/api/camera/status").then((status) => ($("camera-index").value = status.index));
$("camera-index").addEventListener("change", async (event) => {
  try {
    await api("/api/camera/select", { index: Number(event.target.value) });
    setPreview(true);
  } catch (error) {
    showSetupError(error.message);
  }
});

let existingFiles = [];
let overwriteBox = null;

function showSetupError(message, extra = null) {
  const box = $("setup-error");
  box.hidden = !message;
  box.replaceChildren(message || "");
  if (extra) box.append(document.createElement("br"), extra);
}

let checkTimer = null;
$("subject-id").addEventListener("input", () => {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(checkSubject, 300);
});

async function checkSubject() {
  const subjectId = $("subject-id").value.trim();
  existingFiles = [];
  overwriteBox = null;
  showSetupError("");
  if (!subjectId) return;
  try {
    existingFiles = (await api("/api/session/check", { subject_id: subjectId })).existing;
  } catch (error) {
    showSetupError(error.message);
    return;
  }
  if (existingFiles.length) {
    const label = document.createElement("label");
    label.className = "check";
    overwriteBox = document.createElement("input");
    overwriteBox.type = "checkbox";
    label.append(overwriteBox, "Overwrite the existing files");
    showSetupError(`${subjectId} already has recorded data (${existingFiles.length} files).`, label);
  }
}

let sessionRunning = false;

window.addEventListener(
  "keydown",
  (event) => {
    if (sessionRunning && event.key === "Escape" && event.shiftKey) {
      event.preventDefault();
      event.stopImmediatePropagation();
      requestAbort();
    }
  },
  true
);

window.addEventListener("pagehide", () => {
  if (sessionRunning) {
    navigator.sendBeacon("/api/session/finish", JSON.stringify({ aborted: true }));
  }
});

async function enterStage() {
  $("stage").hidden = false;
  $("topbar").hidden = true;
  try {
    await document.documentElement.requestFullscreen();
    await navigator.keyboard?.lock?.(["Escape"]);
  } catch {
  }
}

async function leaveStage() {
  $("stage").hidden = true;
  $("topbar").hidden = false;
  navigator.keyboard?.unlock?.();
  if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
}

$("start-button").addEventListener("click", async () => {
  const subjectId = $("subject-id").value.trim();
  const tasks = TASKS.filter((t) => enabledTasks[t.id] !== false).map((t) => t.id);
  if (!subjectId) return showSetupError("Enter a subject ID first.");
  if (!tasks.length) return showSetupError("Select at least one task.");
  if (existingFiles.length && !overwriteBox?.checked) {
    return showSetupError(
      `${subjectId} already has recorded data. Tick "Overwrite" or use a different ID.`
    );
  }
  const button = $("start-button");
  button.disabled = true;
  showSetupError("");
  const stagePromise = enterStage();
  setPreview(false);
  let clockStart;
  try {
    clockStart = await syncClock();
    await api("/api/session/start", {
      subject_id: subjectId,
      session: $("session-number").value.trim(),
      notes: $("session-notes").value.trim(),
      record_webcam: $("record-webcam").checked,
      overwrite: Boolean(overwriteBox?.checked),
      settings,
      tasks,
      clock_sync: clockStart,
    });
  } catch (error) {
    await stagePromise;
    await leaveStage();
    button.disabled = false;
    setPreview(true);
    return showSetupError(error.message);
  }
  await stagePromise;
  sessionRunning = true;
  const toServer = (ms) => ms / 1000 + clockStart.offset_s;
  let outcome = { aborted: true };
  let failure = null;
  try {
    outcome = await runBattery({
      settings,
      tasks,
      toServer,
      marker: (label) =>
        api("/api/session/marker", { label, t: toServer(performance.now()) }).catch(() => {}),
      saveResults: (task, rows) => api("/api/session/results", { task, rows }),
    });
  } catch (error) {
    failure = error;
  }
  sessionRunning = false;
  const clockEnd = await syncClock().catch(() => null);
  await api("/api/session/finish", { aborted: outcome.aborted || failure !== null, clock_sync: clockEnd })
    .catch(() => {});
  await leaveStage();
  button.disabled = false;
  existingFiles = [];
  overwriteBox = null;
  $("subject-id").value = "";
  if (failure) {
    showView("setup");
    showSetupError(`Session stopped because of an error: ${failure.message}`);
    return;
  }
  showView("results");
  openSession(subjectId);
});

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text ?? "–";
  return td;
}

function formatDate(iso) {
  return iso ? new Date(iso).toLocaleString() : "–";
}

async function loadSessions() {
  const { sessions, logs_dir: logsDir } = await api("/api/sessions");
  $("data-folder").textContent = logsDir;
  const body = $("sessions-table").querySelector("tbody");
  $("sessions-empty").hidden = sessions.length > 0;
  $("sessions-table").hidden = sessions.length === 0;
  body.replaceChildren(
    ...sessions.map((row) => {
      const tr = document.createElement("tr");
      tr.dataset.subject = row.subject_id;
      const open = document.createElement("button");
      open.className = "secondary";
      open.textContent = "Open";
      open.addEventListener("click", () => openSession(row.subject_id));
      const actions = document.createElement("td");
      actions.append(open);
      tr.append(
        cell(row.subject_id + (row.aborted ? " (aborted)" : "")),
        cell(row.session),
        cell(row.source === "psychopy" ? "PsychoPy run" : formatDate(row.started_at)),
        cell(row.has_video ? "Yes" : "No"),
        cell(row.has_trace ? "Extracted" : "Not yet"),
        actions
      );
      return tr;
    })
  );
}

function tile(label, value, sub = "") {
  const element = document.createElement("div");
  element.className = "tile";
  element.innerHTML = `<div class="label"></div><div class="value"></div><div class="sub"></div>`;
  element.querySelector(".label").textContent = label;
  element.querySelector(".value").textContent = value ?? "–";
  element.querySelector(".sub").textContent = sub;
  return element;
}

const percent = (value) => (value === null || value === undefined ? "–" : `${Math.round(value * 100)}%`);
const ms = (value) => (value === null || value === undefined ? "–" : `${value} ms`);

function heading(text) {
  const h = document.createElement("h3");
  h.textContent = text;
  return h;
}

function scoreTiles(summary) {
  const tiles = [];
  if (summary.nback) {
    const n = summary.nback;
    tiles.push(
      tile("N-back accuracy", percent(n.accuracy), `${n.trials} trials`),
      tile("N-back hit rate", percent(n.hit_rate), `${n.targets} targets`),
      tile("N-back false alarms", percent(n.false_alarm_rate)),
      tile("N-back hit RT", ms(n.mean_hit_rt_ms))
    );
  }
  if (summary.digitspan) {
    const d = summary.digitspan;
    tiles.push(
      tile("Digit span forward", d.forward_span, "longest correct"),
      tile("Digit span backward", d.backward_span, "longest correct")
    );
  }
  if (summary.stroop) {
    const s = summary.stroop;
    tiles.push(
      tile("Stroop accuracy", percent(s.accuracy), `${s.missed} missed of ${s.trials}`),
      tile("Stroop interference", ms(s.interference_ms), `incongruent ${ms(s.incongruent_rt_ms)} vs congruent ${ms(s.congruent_rt_ms)}`)
    );
  }
  const grid = document.createElement("div");
  grid.className = "tiles";
  grid.append(...tiles);
  return tiles.length ? grid : document.createTextNode("No task results saved.");
}

function sessionNotes(meta) {
  const lines = [];
  if (meta.started_at) lines.push(`Started ${formatDate(meta.started_at)}`);
  if (meta.notes) lines.push(`Notes: ${meta.notes}`);
  if (meta.aborted) lines.push("Session was aborted; only completed tasks were saved.");
  if (meta.webcam) {
    const fps = meta.webcam.mean_fps ? `${meta.webcam.mean_fps.toFixed(1)} fps` : "–";
    lines.push(`Webcam: ${meta.webcam.frames} frames over ${meta.webcam.duration_s.toFixed(0)} s (${fps})`);
  }
  if (meta.clock_sync_start) {
    let text = `Clock sync: ±${(meta.clock_sync_start.rtt_ms / 2).toFixed(1)} ms`;
    if (meta.clock_sync_end) {
      const drift = (meta.clock_sync_end.offset_s - meta.clock_sync_start.offset_s) * 1000;
      text += `, drift over session ${drift.toFixed(1)} ms`;
    }
    lines.push(text);
  }
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = lines.join(" · ");
  return p;
}

function fileLinks(subjectId, files) {
  const wrap = document.createElement("div");
  wrap.className = "file-links";
  for (const [suffix, present] of Object.entries(files)) {
    if (!present) continue;
    const a = document.createElement("a");
    a.href = `/files/${encodeURIComponent(subjectId + suffix)}`;
    a.textContent = subjectId + suffix;
    wrap.append(a);
  }
  return wrap;
}

let pollTimer = null;

async function openSession(subjectId) {
  clearTimeout(pollTimer);
  for (const tr of $("sessions-table").querySelectorAll("tbody tr")) {
    tr.classList.toggle("selected", tr.dataset.subject === subjectId);
  }
  const detail = $("session-detail");
  detail.hidden = false;
  let data;
  try {
    data = await api(`/api/sessions/${encodeURIComponent(subjectId)}`);
  } catch (error) {
    detail.replaceChildren(error.message);
    return;
  }
  const title = document.createElement("h2");
  title.textContent = `${subjectId}${data.meta.session ? ` · session ${data.meta.session}` : ""}`;
  const analysis = document.createElement("div");
  detail.replaceChildren(
    title,
    sessionNotes(data.meta),
    heading("Task scores"),
    scoreTiles(data.summary),
    heading("Pupil data"),
    analysis,
    heading("Files"),
    fileLinks(subjectId, data.files)
  );
  renderAnalysis(analysis, subjectId, data);
}

function renderAnalysis(container, subjectId, data) {
  const hasVideo = data.files["_webcam.mp4"] && data.files["_webcam_frame_timestamps.csv"];
  const hasTrace = data.files["_pupil_trace.csv"];
  if (!hasVideo && !hasTrace) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "No webcam recording for this session.";
    container.replaceChildren(note);
    return;
  }
  const button = document.createElement("button");
  button.className = "secondary";
  button.hidden = !hasVideo;
  button.textContent = hasTrace ? "Re-run pupil extraction" : "Extract pupil data from video";
  const status = document.createElement("p");
  status.className = "hint";
  const bar = document.createElement("div");
  bar.className = "progress-bar";
  bar.hidden = true;
  bar.append(document.createElement("div"));
  const chartWrap = document.createElement("div");
  container.replaceChildren(button, bar, status, chartWrap);

  if (data.trial_counts) {
    const counts = Object.entries(data.trial_counts).map(([task, n]) => `${task} ${n}`).join(", ");
    status.textContent = `Trials matched to pupil samples: ${counts}.`;
  }

  const poll = async () => {
    const job = await api(`/api/sessions/${encodeURIComponent(subjectId)}/job`).catch(() => null);
    if (!job) return;
    bar.hidden = job.state !== "running";
    bar.firstChild.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    status.textContent = job.state === "running"
      ? `${job.message}… ${Math.round((job.progress || 0) * 100)}%`
      : job.message;
    status.className = job.state === "error" ? "error" : "hint";
    if (job.state === "running") pollTimer = setTimeout(poll, 500);
    else {
      button.disabled = false;
      if (job.state === "done") {
        await loadSessions();
        await openSession(subjectId);
      }
    }
  };

  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/api/sessions/${encodeURIComponent(subjectId)}/analyze`, {});
      poll();
    } catch (error) {
      status.textContent = error.message;
      status.className = "error";
      button.disabled = false;
    }
  });

  if (data.job?.state === "running") {
    button.disabled = true;
    poll();
  }
  if (hasTrace) {
    api(`/api/sessions/${encodeURIComponent(subjectId)}/trace`)
      .then((trace) => renderTraceChart(chartWrap, trace))
      .catch((error) => (chartWrap.textContent = error.message));
  }
}

showView("setup");
