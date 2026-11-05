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
  stroopTrials: 30,
  stroopResponseWindowS: 3.0,
  stroopIsiS: 0.5,
  fluencyDurationS: 60,
};

export const SETTING_LABELS = {
  nbackN: "N-back: N",
  nbackTrials: "N-back: trials",
  nbackLetterDurationS: "N-back: letter on screen (s)",
  nbackSoaS: "N-back: letter interval (s)",
  nbackMatchProbability: "N-back: match probability",
  digitSpanStartLength: "Digit span: start length",
  digitSpanMaxLength: "Digit span: max length",
  digitPresentDurationS: "Digit span: digit on screen (s)",
  digitIsiS: "Digit span: gap between digits (s)",
  stroopTrials: "Stroop: trials",
  stroopResponseWindowS: "Stroop: response window (s)",
  stroopIsiS: "Stroop: gap between trials (s)",
  fluencyDurationS: "Fluency: duration (s)",
};

export const TASKS = [
  { id: "nback", name: "N-Back" },
  { id: "digitspan", name: "Digit Span" },
  { id: "stroop", name: "Stroop" },
  { id: "fluency", name: "Semantic Fluency" },
];

const STROOP_COLORS = ["red", "blue", "green", "yellow"];
const STROOP_KEYS = { r: "red", b: "blue", g: "green", y: "yellow" };
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const FRAME_LEAD_MS = 8;

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
    abortState.reject(new Aborted("Session aborted"));
  }
}

function guard(promise) {
  return Promise.race([promise, abortState.promise]);
}

const sleep = (ms) => guard(new Promise((resolve) => setTimeout(resolve, Math.max(0, ms))));
const sleepUntil = (perfMs) => sleep(perfMs - performance.now());
const randomItem = (items) => items[Math.floor(Math.random() * items.length)];

function keyName(event) {
  if (event.key === " ") return "space";
  if (event.key === "Enter") return "return";
  if (event.key === "Backspace") return "backspace";
  if (event.key === "Escape") return "escape";
  if (event.key.length === 1) return event.key.toLowerCase();
  return null;
}

function waitKey(keys = null, maxMs = null) {
  return guard(
    new Promise((resolve) => {
      let timer = null;
      const done = (value) => {
        window.removeEventListener("keydown", onKey);
        clearTimeout(timer);
        resolve(value);
      };
      const onKey = (event) => {
        if (event.repeat || event.shiftKey && event.key === "Escape") return;
        const key = keyName(event);
        if (key === null || (keys && !keys.includes(key))) return;
        event.preventDefault();
        done({ key, t: event.timeStamp });
      };
      window.addEventListener("keydown", onKey);
      if (maxMs !== null) timer = setTimeout(() => done(null), Math.max(0, maxMs));
      abortState.promise.catch(() => done(null));
    })
  );
}

const stage = {
  center: () => document.getElementById("stage-center"),
  progress: () => document.getElementById("stage-progress"),
  reminder: () => document.getElementById("stage-reminder"),
};

function block(className, text, style = {}) {
  const element = document.createElement("div");
  element.className = className;
  element.textContent = text;
  Object.assign(element.style, style);
  return element;
}

function setScreen({ center = [], progress = "", reminder = "" } = {}) {
  stage.center().replaceChildren(...(Array.isArray(center) ? center : [center]));
  stage.progress().textContent = progress;
  stage.reminder().textContent = reminder;
}

function paint(screen) {
  return guard(
    new Promise((resolve) =>
      requestAnimationFrame((frameTime) => {
        setScreen(screen);
        resolve(frameTime);
      })
    )
  );
}

async function showInstructionPages(pages) {
  for (const [index, text] of pages.entries()) {
    const footer = index === pages.length - 1 ? "begin" : "continue";
    await paint({ center: block("stage-page", `${text}\n\nPress SPACE to ${footer}.`) });
    await waitKey(["space"]);
  }
}

async function showTaskTransition(number, total, name, pauseS = 2.0) {
  await paint({ center: block("stage-title", `Task ${number} of ${total}\n\n${name}`) });
  await sleep(pauseS * 1000);
}

async function showCountdown(seconds = 3) {
  for (let remaining = seconds; remaining > 0; remaining--) {
    await paint({ center: block("stage-countdown", `Get ready...\n\n${remaining}`) });
    await sleep(1000);
  }
}

function nbackPages(s) {
  return [
    "N-BACK TASK\n\n" +
      `You will see a series of letters appear one at a time, about every ${s.nbackSoaS.toFixed(0)} seconds.\n\n` +
      `Press SPACEBAR whenever the current letter is the SAME as the letter that appeared ${s.nbackN} letters before it.`,
    "N-BACK TASK (example)\n\n" +
      "Sequence: K  M  K  T  M  K\n\n" +
      "  - 3rd letter (K) vs 1st letter (K) → match → press SPACE\n" +
      "  - 4th letter (T) vs 2nd letter (M) → no match → don't press\n" +
      "  - 5th letter (M) vs 3rd letter (K) → no match → don't press\n" +
      "  - 6th letter (K) vs 4th letter (T) → no match → don't press\n\n" +
      "If unsure, it's fine to skip rather than guess. Keep your head still and eyes on the screen.",
  ];
}

function digitSpanPages(s) {
  return [
    "DIGIT SPAN TASK\n\n" +
      "You'll see a sequence of digits, one at a time, each shown briefly then disappearing before the next appears.\n\n" +
      "PART 1 — FORWARD: type the digits back in the SAME order, then press ENTER.\n\n" +
      "PART 2 — BACKWARD: type the digits back in REVERSE order, then press ENTER.",
    "DIGIT SPAN TASK (example)\n\n" +
      "If you saw: 4, 7, 2\n" +
      "  - Forward, you would type: 472\n" +
      "  - Backward, you would type: 274\n\n" +
      `Sequences start at ${s.digitSpanStartLength} digits and get one digit longer each time you answer correctly. ` +
      "Each direction ends as soon as you miss one — just do your best.\n\n" +
      "Use BACKSPACE to fix a typo before pressing ENTER.",
  ];
}

const STROOP_PAGES = [
  "STROOP TASK\n\n" +
    "You'll see color words (RED, BLUE, GREEN, YELLOW) printed in colored ink. " +
    "The ink color and the word will often NOT match.\n\n" +
    "Respond to the INK COLOR the word is printed in — ignore what the word says.\n\n" +
    "R = red    B = blue    G = green    Y = yellow",
  "STROOP TASK (example)\n\n" +
    "If the word 'GREEN' appears printed in blue ink, press B (for blue), not G.\n\n" +
    "Each trial gives you a few seconds to respond before moving on. " +
    "Try to respond as fast and accurately as you can.",
];

function fluencyPages(s) {
  return [
    "SEMANTIC FLUENCY TASK\n\n" +
      `You will have ${s.fluencyDurationS} seconds to name out loud as many different ANIMALS as you can think of ` +
      "(e.g. 'dog, cat, elephant, tiger...').\n\n" +
      "Say the names out loud; you do not type anything for this task.",
    "SEMANTIC FLUENCY TASK (continued)\n\n" +
      "Try not to repeat an animal you've already said. If you get stuck, keep thinking — " +
      "any animal counts, including unusual ones.\n\n" +
      "Your responses are recorded separately by the experimenter/microphone — " +
      "this screen just shows a countdown timer.",
  ];
}

function nbackLetters(s) {
  const letters = Array.from({ length: s.nbackTrials }, () => randomItem(LETTERS));
  for (let index = s.nbackN; index < s.nbackTrials; index++) {
    if (Math.random() < s.nbackMatchProbability) letters[index] = letters[index - s.nbackN];
  }
  return letters;
}

async function runNback(ctx) {
  const s = ctx.settings;
  await showInstructionPages(nbackPages(s));
  await showCountdown();
  ctx.marker("NBACK_START");
  const letters = nbackLetters(s);
  const reminder = `Press SPACE if this letter matches the one from ${s.nbackN} letters back`;
  const results = [];
  let nextOnset = performance.now();
  for (const [index, letter] of letters.entries()) {
    const progress = `Trial ${index + 1} of ${s.nbackTrials}`;
    await sleepUntil(nextOnset - FRAME_LEAD_MS);
    const onset = await paint({ center: block("stage-big", letter), progress, reminder });
    const letterEnd = onset + s.nbackLetterDurationS * 1000;
    const key = await waitKey(["space"], letterEnd - performance.now());
    await sleepUntil(letterEnd - FRAME_LEAD_MS);
    await paint({ progress, reminder });
    const isMatch = index >= s.nbackN && letter === letters[index - s.nbackN];
    const responded = key !== null;
    results.push({
      trial: index,
      letter,
      is_match: isMatch,
      responded,
      rt: responded ? (key.t - onset) / 1000 : null,
      correct: responded === isMatch,
      onset_time: ctx.toServer(onset),
    });
    nextOnset = onset + s.nbackSoaS * 1000;
  }
  await sleepUntil(nextOnset);
  ctx.marker("NBACK_END");
  return results;
}

async function presentDigits(s, digits, progress) {
  let firstOnset = null;
  for (const digit of digits) {
    const onset = await paint({ center: block("stage-digit", digit), progress });
    firstOnset ??= onset;
    await sleepUntil(onset + s.digitPresentDurationS * 1000 - FRAME_LEAD_MS);
    const blank = await paint({ progress });
    await sleepUntil(blank + s.digitIsiS * 1000 - FRAME_LEAD_MS);
  }
  return firstOnset;
}

async function collectTypedDigits(promptText, progress) {
  let response = "";
  const screen = () => ({
    center: [block("stage-prompt", promptText), block("stage-typed", response)],
    progress,
  });
  await paint(screen());
  while (true) {
    const key = await waitKey();
    if (key.key === "return") return { response, t: key.t };
    if (key.key === "backspace") response = response.slice(0, -1);
    else if (/^[0-9]$/.test(key.key)) response += key.key;
    await paint(screen());
  }
}

async function runDigitSpan(ctx) {
  const s = ctx.settings;
  await showInstructionPages(digitSpanPages(s));
  await showCountdown();
  ctx.marker("DIGITSPAN_START");
  const results = [];
  for (const direction of ["forward", "backward"]) {
    const orderHint = direction === "forward" ? "same order" : "REVERSE order";
    const promptText = `Type the digits you saw, then press ENTER\n(${orderHint})`;
    for (let length = s.digitSpanStartLength; length <= s.digitSpanMaxLength; length++) {
      const digits = Array.from({ length }, () => String(Math.floor(Math.random() * 10)));
      const progress = `${direction.toUpperCase()} — sequence length ${length}`;
      const stimOnset = await presentDigits(s, digits, progress);
      const { response, t: recallEnd } = await collectTypedDigits(promptText, progress);
      const expected = (direction === "forward" ? digits : [...digits].reverse()).join("");
      const correct = response === expected;
      results.push({
        direction,
        length,
        digits: digits.join(""),
        response,
        correct,
        stim_onset_time: ctx.toServer(stimOnset),
        recall_end_time: ctx.toServer(recallEnd),
      });
      if (!correct) break;
    }
  }
  ctx.marker("DIGITSPAN_END");
  return results;
}

async function runStroop(ctx) {
  const s = ctx.settings;
  await showInstructionPages(STROOP_PAGES);
  await showCountdown();
  ctx.marker("STROOP_START");
  const reminder = "R = red      B = blue      G = green      Y = yellow";
  const results = [];
  for (let index = 0; index < s.stroopTrials; index++) {
    const word = randomItem(STROOP_COLORS);
    const inkColor = randomItem(STROOP_COLORS);
    const progress = `Trial ${index + 1} of ${s.stroopTrials}`;
    const onset = await paint({
      center: block("stage-big", word.toUpperCase(), { color: inkColor }),
      progress,
      reminder,
    });
    const key = await waitKey(
      Object.keys(STROOP_KEYS),
      onset + s.stroopResponseWindowS * 1000 - performance.now()
    );
    const responded = key !== null;
    const responseColor = responded ? STROOP_KEYS[key.key] : null;
    results.push({
      trial: index,
      word,
      ink_color: inkColor,
      is_congruent: word === inkColor,
      responded,
      response_color: responseColor,
      rt: responded ? (key.t - onset) / 1000 : null,
      correct: responseColor === inkColor,
      onset_time: ctx.toServer(onset),
    });
    const blank = await paint({ progress, reminder });
    await sleepUntil(blank + s.stroopIsiS * 1000 - FRAME_LEAD_MS);
  }
  ctx.marker("STROOP_END");
  return results;
}

async function runSemanticFluency(ctx) {
  const s = ctx.settings;
  await showInstructionPages(fluencyPages(s));
  await showCountdown();
  ctx.marker("FLUENCY_START");
  const start = performance.now();
  const end = start + s.fluencyDurationS * 1000;
  let escaped = false;
  const onKey = (event) => {
    if (event.key === "Escape" && !event.shiftKey) escaped = true;
  };
  window.addEventListener("keydown", onKey);
  try {
    while (!escaped && performance.now() < end) {
      const remaining = Math.round((end - performance.now()) / 1000);
      await paint({ center: block("stage-fluency", `Name animals!\n\n${remaining}s remaining`) });
      await sleep(Math.min(100, end - performance.now()));
    }
  } finally {
    window.removeEventListener("keydown", onKey);
  }
  ctx.marker("FLUENCY_END");
  return null;
}

const RUNNERS = {
  nback: runNback,
  digitspan: runDigitSpan,
  stroop: runStroop,
  fluency: runSemanticFluency,
};

export async function runBattery(ctx) {
  resetAbort();
  const tasks = TASKS.filter((task) => ctx.tasks.includes(task.id));
  try {
    for (const [index, task] of tasks.entries()) {
      await showTaskTransition(index + 1, tasks.length, task.name);
      const results = await RUNNERS[task.id](ctx);
      if (results !== null) await ctx.saveResults(task.id, results);
    }
    await paint({ center: block("stage-title", "Session complete. Thank you!") });
    await sleep(2000);
    return { aborted: false };
  } catch (error) {
    if (error instanceof Aborted) return { aborted: true };
    throw error;
  }
}
