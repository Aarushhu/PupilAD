const SVG = "http://www.w3.org/2000/svg";
const WIDTH = 900;
const HEIGHT = 300;
const MARGIN = { top: 28, right: 12, bottom: 28, left: 44 };
const TASK_NAMES = {
  NBACK: "N-back",
  DIGITSPAN: "Digit span",
  STROOP: "Stroop",
  FLUENCY: "Fluency",
};

function svgElement(name, attributes = {}, text = null) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}

function niceStep(span, targetTicks) {
  const raw = span / targetTicks;
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const nice = fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10;
  return nice * power;
}

function ticks(min, max, target) {
  const step = niceStep(max - min || 1, target);
  const values = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) values.push(+v.toFixed(6));
  return values;
}

function clockLabel(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function taskAt(intervals, t) {
  const hit = intervals.find((interval) => t >= interval.start && t <= interval.end);
  return hit ? TASK_NAMES[hit.task] || hit.task : "Between tasks";
}

export function renderTraceChart(container, trace) {
  const finite = trace.diameter.filter((v) => v !== null);
  const caption = document.createElement("p");
  caption.className = "hint";
  const rate = trace.detect_rate === null ? "–" : `${Math.round(trace.detect_rate * 100)}%`;
  caption.textContent =
    `Pupil diameter (px), median per ${(trace.t.length > 1 ? trace.t[1] - trace.t[0] : 0).toFixed(2)} s bin. ` +
    `Pupil found in ${rate} of ${trace.frames} frames. Shaded = task periods. ` +
    "Webcam pixel sizes depend on distance to the camera, so compare within a session only.";
  if (!finite.length) {
    container.replaceChildren(caption, "No frames with a detected pupil.");
    return;
  }

  const tMax = trace.t[trace.t.length - 1] || 1;
  let yMin = Math.min(...finite);
  let yMax = Math.max(...finite);
  const pad = (yMax - yMin) * 0.08 || 1;
  yMin -= pad;
  yMax += pad;
  const plotW = WIDTH - MARGIN.left - MARGIN.right;
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const x = (t) => MARGIN.left + (t / tMax) * plotW;
  const y = (v) => MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  const svg = svgElement("svg", {
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    role: "img",
    "aria-label": "Pupil diameter over the session",
  });

  for (const interval of trace.intervals) {
    const x0 = x(Math.max(0, interval.start));
    const x1 = x(Math.min(tMax, interval.end));
    svg.append(
      svgElement("rect", { class: "band", x: x0, y: MARGIN.top, width: Math.max(0, x1 - x0), height: plotH }),
      svgElement("text", { class: "band-label", x: x0 + 4, y: MARGIN.top - 8 }, TASK_NAMES[interval.task] || interval.task)
    );
  }

  for (const value of ticks(yMin, yMax, 5)) {
    svg.append(
      svgElement("line", { class: "gridline", x1: MARGIN.left, x2: WIDTH - MARGIN.right, y1: y(value), y2: y(value) }),
      svgElement("text", { class: "tick", x: MARGIN.left - 6, y: y(value) + 4, "text-anchor": "end" }, String(value))
    );
  }
  const timeStep = tMax > 600 ? 120 : tMax > 240 ? 60 : 30;
  for (let t = 0; t <= tMax; t += timeStep) {
    svg.append(
      svgElement("text", { class: "tick", x: x(t), y: HEIGHT - 8, "text-anchor": "middle" }, clockLabel(t))
    );
  }
  svg.append(
    svgElement("line", {
      class: "baseline",
      x1: MARGIN.left,
      x2: WIDTH - MARGIN.right,
      y1: MARGIN.top + plotH,
      y2: MARGIN.top + plotH,
    })
  );

  let path = "";
  let penDown = false;
  trace.t.forEach((t, index) => {
    const value = trace.diameter[index];
    if (value === null) {
      penDown = false;
      return;
    }
    path += `${penDown ? "L" : "M"}${x(t).toFixed(1)},${y(value).toFixed(1)}`;
    penDown = true;
  });
  svg.append(svgElement("path", { class: "series", d: path }));

  const crosshair = svgElement("line", { class: "crosshair", y1: MARGIN.top, y2: MARGIN.top + plotH, visibility: "hidden" });
  const dot = svgElement("circle", { class: "hover-dot", r: 5, visibility: "hidden" });
  const hitArea = svgElement("rect", {
    x: MARGIN.left,
    y: MARGIN.top,
    width: plotW,
    height: plotH,
    fill: "transparent",
  });
  svg.append(crosshair, dot, hitArea);

  const wrap = document.createElement("div");
  wrap.className = "chart";
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.hidden = true;
  wrap.append(svg, tooltip);

  hitArea.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const scale = WIDTH / box.width;
    const t = ((event.clientX - box.left) * scale - MARGIN.left) / plotW * tMax;
    let index = Math.round((t / tMax) * (trace.t.length - 1));
    index = Math.max(0, Math.min(trace.t.length - 1, index));
    const pointT = trace.t[index];
    const value = trace.diameter[index];
    crosshair.setAttribute("x1", x(pointT));
    crosshair.setAttribute("x2", x(pointT));
    crosshair.setAttribute("visibility", "visible");
    if (value === null) dot.setAttribute("visibility", "hidden");
    else {
      dot.setAttribute("cx", x(pointT));
      dot.setAttribute("cy", y(value));
      dot.setAttribute("visibility", "visible");
    }
    tooltip.hidden = false;
    tooltip.innerHTML = "";
    const line1 = document.createElement("div");
    line1.textContent = `${clockLabel(pointT)} · ${taskAt(trace.intervals, pointT)}`;
    const line2 = document.createElement("div");
    const bold = document.createElement("b");
    bold.textContent = value === null ? "no pupil detected" : `${value.toFixed(1)} px`;
    line2.append(bold);
    tooltip.append(line1, line2);
    const left = x(pointT) / scale;
    const flip = left > box.width * 0.7;
    tooltip.style.left = `${flip ? left - tooltip.offsetWidth - 12 : left + 12}px`;
    tooltip.style.top = `${MARGIN.top / scale}px`;
  });
  hitArea.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
    tooltip.hidden = true;
  });

  container.replaceChildren(caption, wrap);
}
