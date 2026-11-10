import { Recorder, levelMeter } from "./media.js";

export const DEFAULT_SETTINGS = {
  nbackN: 2,
  nbackTrials: 20,
  nbackLetterDurationS: 0.5,
  nbackSoaS: 2.0,
  nbackMatchProbability: 0.3,
  digitSpanStartLength: 3,
  digitSpanMaxLength: 9,
  digitPresentDurationS: 1.0,
  digitIsiS: 0.5,
  digitPreSequenceS: 1.0,
  stroopTrials: 30,
  stroopResponseWindowS: 3.0,
  stroopIsiS: 0.5,
  fluencyDurationS: 60,
  countdownS: 5,
  breakS: 15,
  minPupilPercent: 50,
};

export const SETTING_LABELS = {
  nbackN: "Letter Match: N (how many back)",
  nbackTrials: "Letter Match: letters",
  nbackLetterDurationS: "Letter Match: letter on screen (s)",
  nbackSoaS: "Letter Match: time between letters (s)",
  nbackMatchProbability: "Letter Match: match probability",
  digitSpanStartLength: "Number Recall: start length",
  digitSpanMaxLength: "Number Recall: max length",
  digitPresentDurationS: "Number Recall: digit on screen (s)",
  digitIsiS: "Number Recall: gap between digits (s)",
  digitPreSequenceS: "Number Recall: pause before each sequence (s)",
  stroopTrials: "Color Clash: words",
  stroopResponseWindowS: "Color Clash: time to answer (s)",
  stroopIsiS: "Color Clash: gap between words (s)",
  fluencyDurationS: "Animal Sprint: duration (s)",
  countdownS: "Settle time before each task (s)",
  breakS: "Break between tasks (s)",
  minPupilPercent: "Camera setup: pupil found (%) needed to start",
};

const ICONS = {
  nback: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="5" width="7" height="14" rx="2"/><rect x="14" y="5" width="7" height="14" rx="2"/></svg>',
  digitspan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 7h4M6 5v12M11 9a2 2 0 1 1 4 0c0 2-4 3-4 6h4M18 7h3l-2 4a2 2 0 1 1-1 4"/></svg>',
  stroop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="9" cy="10" r="5"/><circle cx="15" cy="14" r="5"/></svg>',
  fluency: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>',
};
const CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" aria-hidden="true"><path d="M5 12l5 5L20 7"/></svg>';
const ARROW_RIGHT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';
const ARROW_LEFT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6"/></svg>';
const DELETE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 6h11v12H9l-6-6zM12 10l4 4M16 10l-4 4"/></svg>';

export const TASKS = [
  {
    id: "nback",
    name: "Letter Match",
    science: "2-back",
    blurb: "Spot letters that repeat 2 steps back",
    tone: "t1",
    minutes: (s) => (s.nbackTrials * s.nbackSoaS) / 60,
  },
  {
    id: "digitspan",
    name: "Number Recall",
    science: "Digit span",
    blurb: "Remember digits, then tap them back",
    tone: "t2",
    minutes: () => 4,
  },
  {
    id: "stroop",
    name: "Color Clash",
    science: "Stroop",
    blurb: "Pick the ink colour, not the word",
    tone: "t3",
    minutes: (s) => (s.stroopTrials * (1.2 + s.stroopIsiS)) / 60,
  },
  {
    id: "fluency",
    name: "Animal Sprint",
    science: "Semantic fluency",
    blurb: "Name as many animals as you can",
    tone: "t4",
    minutes: (s) => s.fluencyDurationS / 60,
  },
];

export const taskIcon = (id) => ICONS[id];

const STROOP_COLORS = ["red", "blue", "green", "yellow"];
const STROOP_KEYS = { r: "red", b: "blue", g: "green", y: "yellow" };
const COLOR_NAMES = { red: "Red", blue: "Blue", green: "Green", yellow: "Yellow" };
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const FRAME_LEAD_MS = 8;
const PRACTICE_NBACK_TRIALS = 8;
const PRACTICE_STROOP_TRIALS = 6;
const PRACTICE_FEEDBACK_MS = 1300;

export class Aborted extends Error {}

let abortState = null;

function resetAbort() {
  let reject;
  const promise = new Promise((_, r) => (reject = r));
  promise.catch(() => {});
  abortState = { promise, reject, aborted: false };
}

export function requestAbort() {
  if (abortState && !abortState.aborted) {
    abortState.aborted = true;
    abortState.reject(new Aborted("Session stopped"));
  }
}

function guard(promise) {
  return Promise.race([promise, abortState.promise]);
}

const sleep = (ms) => guard(new Promise((resolve) => setTimeout(resolve, Math.max(0, ms))));
const sleepUntil = (perfMs) => sleep(perfMs - performance.now());
const randomItem = (items) => items[Math.floor(Math.random() * items.length)];
const minutesLabel = (minutes) => (minutes < 1.5 ? "1 min" : `${Math.round(minutes)} min`);
const root = () => document.getElementById("play");
const byId = (id) => document.getElementById(id);

function paint(update) {
  return guard(
    new Promise((resolve) =>
      requestAnimationFrame((frameTime) => {
        update();
        resolve(frameTime);
      })
    )
  );
}

function normalKey(event) {
  if (event.key === " ") return " ";
  return event.key.length === 1 ? event.key.toLowerCase() : event.key;
}

function waitResponse(keys, container, maxMs = null) {
  return guard(
    new Promise((resolve) => {
      let timer = null;
      const done = (value) => {
        window.removeEventListener("keydown", onKey);
        container?.removeEventListener("pointerdown", onPointer);
        clearTimeout(timer);
        resolve(value);
      };
      const onKey = (event) => {
        if (event.repeat || (event.shiftKey && event.key === "Escape")) return;
        const answer = keys[normalKey(event)];
        if (answer === undefined) return;
        event.preventDefault();
        done({ answer, t: event.timeStamp });
      };
      const onPointer = (event) => {
        const button = event.target.closest("[data-answer]");
        if (!button || button.disabled) return;
        event.preventDefault();
        done({ answer: button.dataset.answer, t: event.timeStamp });
      };
      window.addEventListener("keydown", onKey);
      container?.addEventListener("pointerdown", onPointer);
      if (maxMs !== null) timer = setTimeout(() => done(null), Math.max(0, maxMs));
      abortState.promise.catch(() => done(null));
    })
  );
}

function waitAction(keys = {}) {
  return guard(
    new Promise((resolve) => {
      const container = root();
      const done = (action) => {
        container.removeEventListener("click", onClick);
        window.removeEventListener("keydown", onKey);
        resolve(action);
      };
      const onClick = (event) => {
        const element = event.target.closest("[data-action]");
        if (!element || element.disabled) return;
        done(element.dataset.action);
      };
      const onKey = (event) => {
        if (event.repeat || event.shiftKey) return;
        const action = keys[normalKey(event)];
        if (!action) return;
        const element = container.querySelector(`[data-action="${action}"]`);
        if (!element || element.disabled) return;
        event.preventDefault();
        done(action);
      };
      container.addEventListener("click", onClick);
      window.addEventListener("keydown", onKey);
      abortState.promise.catch(() => done(null));
    })
  );
}

const STOP_BUTTON = '<button class="stop-btn" type="button" data-hold-stop aria-label="Hold to stop the session"><span>Hold to stop</span></button>';

function lightScreen(title, right, body) {
  root().innerHTML = `
    <div class="play-light">
      <header class="appbar">
        <div class="logo">${title}</div>
        <div class="appbar-right"><span class="who">${right}</span>${STOP_BUTTON}</div>
      </header>
      <div class="content">${body}</div>
    </div>`;
  root().querySelector(".play-light").scrollTop = 0;
}

function stageScreen(task, { practice = false, actions = "", actionsClass = "" } = {}) {
  root().innerHTML = `
    <div class="stage">
      <div class="stage-top">
        <span class="name">${task.name}</span>
        ${practice ? '<span class="practice-tag">Practice</span>' : ""}
        <div class="bar"><i id="st-bar"></i></div>
        <span class="count" id="st-count"></span>
        ${STOP_BUTTON}
      </div>
      <div class="stage-center" id="st-center"></div>
      <div class="feedback" id="st-feedback" aria-live="polite"></div>
      <div class="stage-actions ${actionsClass}" id="st-actions">${actions}</div>
    </div>`;
}

function setProgress(fraction, text) {
  byId("st-bar").style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  byId("st-count").textContent = text;
}

function setFeedback(text, tone = "") {
  const element = byId("st-feedback");
  element.textContent = text;
  element.className = `feedback ${tone}`;
}

const fixation = () => {
  const dot = document.createElement("div");
  dot.className = "fix";
  return dot;
};

async function countdown(task, seconds) {
  const center = byId("st-center");
  for (let remaining = Math.round(seconds); remaining > 0; remaining--) {
    await paint(() => {
      center.innerHTML = `<div class="ready"><span>Get ready</span><b>${remaining}</b></div>`;
    });
    await sleep(1000);
  }
  await paint(() => center.replaceChildren(fixation()));
}

function nbackLetters(s, count) {
  const letters = Array.from({ length: count }, () => randomItem(LETTERS));
  for (let index = s.nbackN; index < count; index++) {
    if (Math.random() < s.nbackMatchProbability) letters[index] = letters[index - s.nbackN];
  }
  return letters;
}

function practiceLetters(s) {
  for (let attempt = 0; attempt < 50; attempt++) {
    const letters = nbackLetters({ ...s, nbackMatchProbability: 0.35 }, PRACTICE_NBACK_TRIALS);
    const matches = letters.filter((letter, i) => i >= s.nbackN && letter === letters[i - s.nbackN]).length;
    if (matches >= 2 && matches <= 3) return letters;
  }
  return nbackLetters(s, PRACTICE_NBACK_TRIALS);
}

async function nbackBlock(ctx, task, letters, practice) {
  const s = ctx.settings;
  stageScreen(task, {
    practice,
    actions: `
      <button class="sbtn" type="button" data-answer="no_match"><span class="lab">No match</span><kbd>N</kbd></button>
      <button class="sbtn" type="button" data-answer="match"><span class="lab">Match</span><kbd>Space</kbd></button>`,
  });
  const actions = byId("st-actions");
  setProgress(0, `0 / ${letters.length}`);
  await countdown(task, practice ? 3 : s.countdownS);
  const card = document.createElement("div");
  card.className = "lettercard";
  card.append(fixation());
  await paint(() => byId("st-center").replaceChildren(card));
  if (!practice) ctx.marker("NBACK_START");
  const results = [];
  let nextOnset = performance.now() + 600;
  for (const [index, letter] of letters.entries()) {
    await sleepUntil(nextOnset - FRAME_LEAD_MS);
    const onset = await paint(() => {
      card.className = "lettercard";
      card.textContent = letter;
      actions.querySelectorAll(".chosen").forEach((b) => b.classList.remove("chosen"));
      setProgress((index + 1) / letters.length, `${index + 1} / ${letters.length}`);
    });
    const letterEnd = onset + s.nbackLetterDurationS * 1000;
    const windowEnd = onset + s.nbackSoaS * 1000 - FRAME_LEAD_MS * 2;
    const offset = (async () => {
      await sleepUntil(letterEnd - FRAME_LEAD_MS);
      await paint(() => card.replaceChildren(fixation()));
    })();
    const isMatch = index >= s.nbackN && letter === letters[index - s.nbackN];
    const response = await waitResponse({ " ": "match", n: "no_match" }, actions, windowEnd - performance.now());
    const saidMatch = response?.answer === "match";
    const correct = saidMatch === isMatch;
    if (response) actions.querySelector(`[data-answer="${response.answer}"]`)?.classList.add("chosen");
    if (practice) {
      if (!response) setFeedback("Answer on every letter, before the next one appears", "warn");
      else if (correct) setFeedback(isMatch ? "Correct, that was a match" : "Correct, no match", "good");
      else setFeedback(isMatch ? `That was a match: same as ${s.nbackN} back` : `Not a match: ${s.nbackN} back was ${letters[index - s.nbackN] ?? "nothing"}`, "warn");
      if (response) card.classList.add(correct ? "fb-good" : "fb-miss");
    }
    await offset;
    results.push({
      trial: index,
      letter,
      is_match: isMatch,
      responded: saidMatch,
      rt: response ? (response.t - onset) / 1000 : null,
      correct,
      onset_time: onset / 1000,
      response: response ? response.answer : "",
    });
    nextOnset = onset + s.nbackSoaS * 1000;
  }
  await sleepUntil(nextOnset);
  if (!practice) ctx.marker("NBACK_END");
  return results;
}

function keypadMarkup() {
  const keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "del", "0", "ok"];
  return keys
    .map((key) => {
      if (key === "del") return `<button class="key" type="button" data-answer="del" aria-label="Delete">${DELETE}</button>`;
      if (key === "ok") return '<button class="key done" type="button" data-answer="ok">Done</button>';
      return `<button class="key" type="button" data-answer="${key}">${key}</button>`;
    })
    .join("");
}

function directionBadge(direction, recall) {
  const forward = direction === "forward";
  const text = recall
    ? forward ? "Tap them in the same order" : "Tap them in reverse order"
    : forward ? "Same order" : "Reverse order";
  return `<span class="direction">${forward ? ARROW_RIGHT : ARROW_LEFT}${text}</span>`;
}

async function presentDigits(s, digits, direction, progressText) {
  const center = byId("st-center");
  center.innerHTML = `<div class="present">${directionBadge(direction, false)}<div class="digitbig" id="ds-digit"></div></div>`;
  const slot = byId("ds-digit");
  slot.replaceChildren(fixation());
  setProgress(0, progressText);
  await sleep(s.digitPreSequenceS * 1000);
  let firstOnset = null;
  for (const [index, digit] of digits.entries()) {
    const onset = await paint(() => {
      slot.textContent = digit;
      setProgress((index + 1) / digits.length, progressText);
    });
    firstOnset ??= onset;
    await sleepUntil(onset + s.digitPresentDurationS * 1000 - FRAME_LEAD_MS);
    const blank = await paint(() => slot.replaceChildren(fixation()));
    await sleepUntil(blank + s.digitIsiS * 1000 - FRAME_LEAD_MS);
  }
  return firstOnset;
}

async function collectDigits(length, direction) {
  const center = byId("st-center");
  center.innerHTML = `
    <div class="recall">
      ${directionBadge(direction, true)}
      <div class="slots" id="ds-slots"></div>
      <div class="keypad" id="ds-keypad">${keypadMarkup()}</div>
    </div>`;
  const keypad = byId("ds-keypad");
  const slots = byId("ds-slots");
  const done = keypad.querySelector('[data-answer="ok"]');
  let typed = "";
  const draw = () => {
    slots.innerHTML = Array.from({ length }, (_, i) => `<div class="slot${typed[i] ? " filled" : ""}">${typed[i] ?? ""}</div>`).join("");
    done.disabled = typed.length === 0;
  };
  draw();
  const keys = { Backspace: "del", Enter: "ok" };
  for (const digit of "0123456789") keys[digit] = digit;
  while (true) {
    const press = await waitResponse(keys, keypad);
    if (press.answer === "ok") {
      if (typed.length) return { response: typed, t: press.t };
    } else if (press.answer === "del") typed = typed.slice(0, -1);
    else if (typed.length < length) typed += press.answer;
    draw();
  }
}

const randomDigits = (length) => Array.from({ length }, () => String(Math.floor(Math.random() * 10)));

async function digitSequence(s, direction, digits, label) {
  const stimOnset = await presentDigits(s, digits, direction, label);
  const { response, t } = await collectDigits(digits.length, direction);
  const expected = (direction === "forward" ? digits : [...digits].reverse()).join("");
  return {
    direction,
    length: digits.length,
    digits: digits.join(""),
    response,
    correct: response === expected,
    stim_onset_time: stimOnset / 1000,
    recall_end_time: t / 1000,
    expected,
  };
}

async function partTwoCard() {
  byId("st-center").innerHTML = `
    <div class="stage-card">
      <span class="direction">${ARROW_LEFT}Part 2</span>
      <h2>Now in reverse order</h2>
      <p>If you see <b>4 7 2</b>, tap <b>2 7 4</b>.</p>
      <button class="btn btn-stage" type="button" data-action="go">Start part 2</button>
    </div>`;
  setProgress(0, "Part 2");
  await waitAction({ Enter: "go", " ": "go" });
}

async function digitPractice(ctx, task) {
  const s = ctx.settings;
  stageScreen(task, { practice: true });
  await countdown(task, 3);
  let correct = 0;
  for (const direction of ["forward", "backward"]) {
    if (direction === "backward") await partTwoCard();
    const row = await digitSequence(s, direction, randomDigits(3), direction === "forward" ? "Same order" : "Reverse order");
    if (row.correct) correct++;
    byId("st-center").innerHTML = row.correct
      ? `<div class="stage-card"><div class="big-check good">${CHECK}</div><h2>Correct</h2></div>`
      : `<div class="stage-card"><h2>Not quite</h2><p>The answer was <b>${row.expected.split("").join(" ")}</b></p></div>`;
    await sleep(PRACTICE_FEEDBACK_MS + 400);
  }
  return { correct, total: 2 };
}

async function digitMeasured(ctx, task) {
  const s = ctx.settings;
  stageScreen(task);
  await countdown(task, s.countdownS);
  ctx.marker("DIGITSPAN_START");
  const results = [];
  for (const direction of ["forward", "backward"]) {
    if (direction === "backward") await partTwoCard();
    for (let length = s.digitSpanStartLength; length <= s.digitSpanMaxLength; length++) {
      const label = `${direction === "forward" ? "Part 1" : "Part 2"} · ${length} digits`;
      const row = await digitSequence(s, direction, randomDigits(length), label);
      delete row.expected;
      results.push(row);
      if (!row.correct) break;
    }
  }
  ctx.marker("DIGITSPAN_END");
  return results;
}

function stroopTrials(count, practice) {
  return Array.from({ length: count }, (_, index) => {
    const word = randomItem(STROOP_COLORS);
    let ink = randomItem(STROOP_COLORS);
    if (practice && index % 3 !== 1) while (ink === word) ink = randomItem(STROOP_COLORS);
    return { word, ink };
  });
}

async function stroopBlock(ctx, task, trials, practice) {
  const s = ctx.settings;
  stageScreen(task, {
    practice,
    actionsClass: "four",
    actions: STROOP_COLORS.map(
      (color) => `<button class="sbtn" type="button" data-answer="${color}"><span class="lab"><i class="sw ink-${color}"></i>${COLOR_NAMES[color]}</span><kbd>${color[0].toUpperCase()}</kbd></button>`
    ).join(""),
  });
  const actions = byId("st-actions");
  setProgress(0, `0 / ${trials.length}`);
  await countdown(task, practice ? 3 : s.countdownS);
  const word = document.createElement("div");
  word.className = "stroopword";
  await paint(() => byId("st-center").replaceChildren(word));
  word.append(fixation());
  if (!practice) ctx.marker("STROOP_START");
  await sleep(600);
  const results = [];
  for (const [index, trial] of trials.entries()) {
    const onset = await paint(() => {
      word.textContent = trial.word.toUpperCase();
      word.className = `stroopword ink-${trial.ink}`;
      setProgress((index + 1) / trials.length, `${index + 1} / ${trials.length}`);
    });
    const response = await waitResponse(STROOP_KEYS, actions, onset + s.stroopResponseWindowS * 1000 - performance.now());
    const blank = await paint(() => {
      word.className = "stroopword";
      word.replaceChildren(fixation());
    });
    const responseColor = response ? response.answer : null;
    const correct = responseColor === trial.ink;
    if (practice) {
      if (!response) setFeedback("Too slow, try to answer a bit faster", "warn");
      else if (correct) setFeedback(`Correct, the ink was ${trial.ink}`, "good");
      else setFeedback(`The ink was ${trial.ink}. Ignore what the word says.`, "warn");
    }
    results.push({
      trial: index,
      word: trial.word,
      ink_color: trial.ink,
      is_congruent: trial.word === trial.ink,
      responded: response !== null,
      response_color: responseColor,
      rt: response ? (response.t - onset) / 1000 : null,
      correct,
      onset_time: onset / 1000,
    });
    await sleepUntil(blank + (practice ? PRACTICE_FEEDBACK_MS : s.stroopIsiS * 1000) - FRAME_LEAD_MS);
    if (practice) setFeedback("");
  }
  if (!practice) ctx.marker("STROOP_END");
  return results;
}

async function fluencyMeasured(ctx, task) {
  const s = ctx.settings;
  stageScreen(task);
  await countdown(task, s.countdownS);
  const circumference = 289;
  byId("st-center").innerHTML = `
    <div class="fluency">
      <div class="ring">
        <svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="46" class="ring-track"/><circle id="fl-arc" cx="50" cy="50" r="46" class="ring-arc" stroke-dasharray="${circumference}" stroke-dashoffset="0"/></svg>
        <div class="inner"><span class="secs">Say out loud</span><span class="cat">Animals</span><span class="secs" id="fl-secs"></span></div>
      </div>
      <div class="mic" id="fl-mic" aria-hidden="true">${"<i></i>".repeat(14)}</div>
      <span class="fluency-note" id="fl-note">Any animal counts. Keep going until the circle closes.</span>
    </div>`;
  const bars = [...byId("fl-mic").children];
  let recorder = null;
  let meter = null;
  if (ctx.microphone) {
    try {
      recorder = new Recorder(ctx.microphone, "audio");
      await recorder.start();
      meter = levelMeter(ctx.microphone);
      await meter?.resume();
    } catch (error) {
      ctx.audioInfo = { error: error.message };
      recorder = null;
    }
  }
  if (!recorder) {
    byId("fl-mic").hidden = true;
    byId("fl-note").textContent = "Any animal counts. The experimenter will note your answers.";
  } else {
    setProgress(0, "Recording");
    byId("st-count").classList.add("recdot");
  }
  const start = performance.now();
  const end = start + s.fluencyDurationS * 1000;
  ctx.marker("FLUENCY_START");
  try {
    while (performance.now() < end) {
      const left = Math.max(0, end - performance.now());
      await paint(() => {
        byId("fl-secs").textContent = `${Math.ceil(left / 1000)} s left`;
        byId("fl-arc").style.strokeDashoffset = String(circumference * (1 - left / (s.fluencyDurationS * 1000)));
        byId("st-bar").style.width = `${(1 - left / (s.fluencyDurationS * 1000)) * 100}%`;
        if (meter) {
          const level = meter.level();
          bars.forEach((bar, i) => {
            const shape = 0.45 + 0.55 * Math.abs(Math.sin(i * 1.7 + performance.now() / 180));
            bar.style.height = `${6 + level * 34 * shape}px`;
          });
        }
      });
      await sleep(Math.min(100, end - performance.now()));
    }
  } finally {
    ctx.marker("FLUENCY_END");
    meter?.close();
    if (recorder) ctx.audioInfo = await recorder.stop();
  }
  return null;
}

function exampleFor(task, s) {
  if (task.id === "nback") {
    return `
      <div class="howto" aria-label="Example: K, M, K. The second K matches the K two steps back.">
        <div class="tile src">K<small>2 back</small></div>
        <div class="tile">M<small>1 back</small></div>
        <div class="tile hit">K<small>now</small></div>
      </div>
      <p class="lede center">The new <b>K</b> matches the <b>K</b> ${s.nbackN} steps back, so tap <b>Match</b>.</p>`;
  }
  if (task.id === "digitspan") {
    return `
      <div class="example-rows">
        <div class="example-row"><span class="example-label">You see</span><div class="mini-tiles"><span>4</span><span>7</span><span>2</span></div></div>
        <div class="example-row"><span class="example-label">${ARROW_RIGHT}Part 1</span><div class="mini-tiles good"><span>4</span><span>7</span><span>2</span></div></div>
        <div class="example-row"><span class="example-label">${ARROW_LEFT}Part 2</span><div class="mini-tiles accent"><span>2</span><span>7</span><span>4</span></div></div>
      </div>`;
  }
  if (task.id === "stroop") {
    return `
      <div class="stroop-example"><span class="ink-blue">GREEN</span></div>
      <p class="lede center">The word says green, but the ink is <b>blue</b>, so tap <b>Blue</b>.</p>`;
  }
  return `
    <div class="chips-example"><span>dog</span><span>eagle</span><span>shark</span><span>ant</span><span>zebra</span><span>…</span></div>
    <p class="lede center">Say them out loud. No typing or tapping needed.</p>`;
}

function stepsFor(task, s, ctx) {
  if (task.id === "nback") {
    return [
      `Letters appear one at a time, every ${s.nbackSoaS} seconds.`,
      `Tap <b>Match</b> if the letter is the same as ${s.nbackN} steps back. Otherwise tap <b>No match</b>.`,
      "Answer every letter, before the next one appears.",
    ];
  }
  if (task.id === "digitspan") {
    return [
      "Digits appear one at a time. Remember them.",
      "Then tap them back on the keypad and press <b>Done</b>.",
      `Part 1 is the same order, part 2 is reverse order. Each part starts at ${s.digitSpanStartLength} digits and grows until you miss one.`,
    ];
  }
  if (task.id === "stroop") {
    return [
      "Colour words appear in coloured ink.",
      "Tap the colour of the <b>ink</b>. Ignore what the word says.",
      "Answer as quickly and accurately as you can.",
    ];
  }
  return [
    `You have ${s.fluencyDurationS} seconds.`,
    "Say as many different animals as you can, out loud.",
    ctx.microphone ? "Your voice is recorded so your answers can be counted afterwards." : "The experimenter will note your answers.",
  ];
}

const PRACTICE = {
  nback: {
    label: (s) => `Practice (${PRACTICE_NBACK_TRIALS} letters)`,
    run: async (ctx, task) => {
      const rows = await nbackBlock(ctx, task, practiceLetters(ctx.settings), true);
      return { correct: rows.filter((r) => r.correct).length, total: rows.length };
    },
  },
  digitspan: {
    label: () => "Practice (2 short rounds)",
    run: digitPractice,
  },
  stroop: {
    label: () => `Practice (${PRACTICE_STROOP_TRIALS} words)`,
    run: async (ctx, task) => {
      const rows = await stroopBlock(ctx, task, stroopTrials(PRACTICE_STROOP_TRIALS, true), true);
      return { correct: rows.filter((r) => r.correct).length, total: rows.length };
    },
  },
};

async function introScreen(ctx, task, number, total) {
  const s = ctx.settings;
  const practice = PRACTICE[task.id];
  let note = "";
  let practiced = false;
  while (true) {
    lightScreen(
      task.name,
      `Challenge ${number} of ${total}`,
      `
      <div>
        <div class="eyebrow">How it works</div>
        <h1 class="h1">${task.id === "nback" ? `Does this letter match the one ${s.nbackN} steps back?` : task.blurb}</h1>
      </div>
      ${note}
      <div class="grid2">
        <section class="card">${exampleFor(task, s)}</section>
        <section class="card"><ol class="steps">${stepsFor(task, s, ctx).map((step) => `<li>${step}</li>`).join("")}</ol></section>
      </div>
      <div class="actions">
        ${practice ? `<button class="btn btn-ghost" type="button" data-action="practice">${practiced ? "Practice again" : practice.label(s)}</button>` : ""}
        <button class="btn btn-primary" type="button" data-action="go">I'm ready</button>
      </div>`
    );
    const action = await waitAction({ Enter: "go" });
    if (action === "go") return;
    const result = await practice.run(ctx, task);
    practiced = true;
    ctx.practice.push({ task: task.id, ...result, at: new Date().toISOString() });
    note = `<div class="notice good"><b>Practice: ${result.correct} of ${result.total} right.</b> Practice again, or start when you're ready.</div>`;
  }
}

function bestStreak(rows) {
  let best = 0;
  let run = 0;
  for (const row of rows) {
    run = row.correct ? run + 1 : 0;
    best = Math.max(best, run);
  }
  return best;
}

function meanRt(rows) {
  const times = rows.filter((r) => r.correct && r.rt !== null).map((r) => r.rt);
  return times.length ? times.reduce((a, b) => a + b, 0) / times.length : null;
}

function taskStats(task, rows, ctx) {
  if (task.id === "digitspan") {
    const span = (direction) => Math.max(0, ...rows.filter((r) => r.direction === direction && r.correct).map((r) => r.length));
    return {
      stats: [[span("forward"), "digits, same order"], [span("backward"), "digits, reverse"], [rows.length, "sequences"]],
      good: span("forward") >= 5,
    };
  }
  if (task.id === "fluency") {
    return {
      stats: [[`${ctx.settings.fluencyDurationS} s`, "of talking"], [ctx.audioInfo?.mime ? "Saved" : "Done", ctx.audioInfo?.mime ? "recording" : "countdown"]],
      good: true,
    };
  }
  const correct = rows.filter((r) => r.correct).length;
  const rt = meanRt(rows);
  return {
    stats: [[`${correct}/${rows.length}`, "correct"], [rt === null ? "–" : `${rt.toFixed(2)} s`, "avg answer"], [bestStreak(rows), "best streak"]],
    good: correct / Math.max(1, rows.length) >= 0.75,
  };
}

async function breakScreen(ctx, task, rows, next, doneCount, total) {
  const s = ctx.settings;
  const { stats, good } = taskStats(task, rows ?? [], ctx);
  let remaining = Math.round(s.breakS);
  lightScreen(
    "Nice work",
    `${doneCount} of ${total} done`,
    `
    <div class="row-head">
      <div class="done-badge">${CHECK}</div>
      <div><div class="eyebrow">${task.name} complete</div><h1 class="h1 small">${good ? "You kept up the whole way" : "Nice effort, that one is tricky"}</h1></div>
    </div>
    <div class="grid2">
      <section class="card">
        <div class="stats">${stats.map(([value, label]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`).join("")}</div>
        <p class="lede small">Scores are just for you. Researchers look at the full data.</p>
      </section>
      <section class="card center">
        <div class="breathe"><span>Breathe in…<br>and out</span></div>
        <p class="lede">Relax your eyes for <b id="br-secs">${remaining}</b> s before the next challenge.</p>
      </section>
    </div>
    <section class="card next-card">
      <div class="ticon ${next.tone}">${ICONS[next.id]}</div>
      <div><b>Up next: ${next.name}</b><div class="muted">${next.blurb} · ${minutesLabel(next.minutes(s))}</div></div>
    </section>
    <div class="actions"><button class="btn btn-primary" type="button" data-action="go" id="br-go" ${remaining > 0 ? "disabled" : ""}>${remaining > 0 ? `Continue in ${remaining} s` : `Continue to ${next.name}`}</button></div>`
  );
  while (remaining > 0) {
    await sleep(1000);
    remaining--;
    byId("br-secs").textContent = String(remaining);
    const button = byId("br-go");
    button.disabled = remaining > 0;
    button.textContent = remaining > 0 ? `Continue in ${remaining} s` : `Continue to ${next.name}`;
  }
  await waitAction({ Enter: "go" });
}

export function journeyMarkup(tasks, settings, nextId = null) {
  return tasks
    .map(
      (task) => `
      <div class="station${task.id === nextId ? " next" : ""}">
        <div class="ticon ${task.tone}">${ICONS[task.id]}</div>
        <div><b>${task.name}</b><span>${task.blurb}</span>
          <div class="status"><span class="pill${task.id === nextId ? "" : " plain"}">${task.id === nextId ? "Up next · " : ""}${minutesLabel(task.minutes(settings))}</span></div>
        </div>
      </div>`
    )
    .join("");
}

const MEASURED = {
  nback: (ctx, task) => nbackBlock(ctx, task, nbackLetters(ctx.settings, ctx.settings.nbackTrials), false),
  digitspan: digitMeasured,
  stroop: (ctx, task) => stroopBlock(ctx, task, stroopTrials(ctx.settings.stroopTrials, false), false),
  fluency: fluencyMeasured,
};

export async function runBattery(ctx) {
  resetAbort();
  ctx.practice = [];
  const tasks = TASKS.filter((task) => ctx.tasks.includes(task.id));
  const total = tasks.length;
  try {
    const totalMinutes = tasks.reduce((sum, task) => sum + task.minutes(ctx.settings), 0) + ((total - 1) * ctx.settings.breakS) / 60;
    lightScreen(
      "Your session",
      `About ${Math.max(1, Math.round(totalMinutes + total))} minutes`,
      `
      <div>
        <h1 class="h1">${total === 1 ? "One short challenge" : `${["", "", "Two", "Three", "Four"][total]} short challenges`}</h1>
        <p class="lede">Each one starts with a quick explanation and an optional practice round. You get a short break after every challenge.</p>
      </div>
      <div class="journey">${journeyMarkup(tasks, ctx.settings, tasks[0].id)}</div>
      <p class="lede small">Sit comfortably, keep your head still and look at the screen. The camera records your eyes during the challenges.</p>
      <div class="actions"><button class="btn btn-primary" type="button" data-action="go">Start</button></div>`
    );
    await waitAction({ Enter: "go" });
    await guard(ctx.begin());
    for (const [index, task] of tasks.entries()) {
      await introScreen(ctx, task, index + 1, total);
      const rows = await MEASURED[task.id](ctx, task);
      if (rows !== null) await ctx.saveResults(task.id, rows);
      if (index + 1 < total) await breakScreen(ctx, task, rows, tasks[index + 1], index + 1, total);
    }
    return { aborted: false };
  } catch (error) {
    if (error instanceof Aborted) return { aborted: true };
    throw error;
  }
}
