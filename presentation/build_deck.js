// Builds snn_scaling_talk.pptx from deck_data.json (which extract_data.py derives from logged results).
// Run:  NODE_PATH=<dir with pptxgenjs + jszip> node build_deck.js
// No simulation code is executed; every number on a slide comes from deck_data.json or a cited file.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "snn_scaling_talk.pptx");

// ---------------------------------------------------------------- theme
const HEX = {
  ink: "1A1F2B", navy: "0B2545", paper: "FFFFFF", mist: "EEF2F6",
  denseCPU: "1F3A5F", denseGPU: "5B8FD4", sparseCPU: "1B7F6E", sparseGPU: "4DB89A",
  event: "D1495B", amber: "C98A2B", grid: "D9DEE5", muted: "5E6878",
};
const THEME = {
  name: "SNN Scaling",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.paper, dk2: HEX.navy, lt2: HEX.mist,
    accent1: HEX.denseCPU, accent2: HEX.denseGPU, accent3: HEX.sparseCPU,
    accent4: HEX.sparseGPU, accent5: HEX.event, accent6: HEX.amber,
    hlink: HEX.denseGPU, folHlink: "6B4E9B",
  },
};
const BACKENDS = ["Dense CPU", "Dense GPU", "Sparse CPU", "Sparse GPU", "Event-driven CPU"];
const BCOLOR = {
  "Dense CPU": HEX.denseCPU, "Dense GPU": HEX.denseGPU, "Sparse CPU": HEX.sparseCPU,
  "Sparse GPU": HEX.sparseGPU, "Event-driven CPU": HEX.event,
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "One network, three execution strategies";
pres.author = "Cannon Wilson";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
const FOOT = "Spiking-network execution strategies on CPU and GPU";
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.2, fontSize: 20, color: C.background2, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "tag", type: "body", x: 0.6, y: 0.32, w: 8, h: 0.34, fontSize: 12, bold: true, color: C.accent3, charSpacing: 1.5, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.68, w: 12.13, h: 1.05, fontSize: 28, bold: true, color: C.text2, valign: "top", align: "left", margin: 0, fontFace: THEME.headFontFace }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 7, h: 0.3, fontSize: 10, color: C.text1, transparency: 45, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "5E6878", align: "right" },
});

// ---------------------------------------------------------------- helpers
let shapeId = 0;
const nm = (s) => `${s}-${++shapeId}`;
const addedSections = new Set(["Question"]);
function content(tag, title, section) {
  if (!addedSections.has(section)) { pres.addSection({ title: section }); addedSections.add(section); }
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(tag.toUpperCase(), { placeholder: "tag" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, fontSize: 16, color: C.text1, margin: 0, valign: "top", objectName: nm("txt") }, o));
}
function box(s, x, y, w, h, text, o = {}) {
  const fill = o.fill || C.background2;
  s.addText(text, Object.assign({
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w, h,
    fill: { color: fill, transparency: o.transparency || 0 },
    line: o.line ? { color: o.line, width: o.lineWidth || 1.25 } : { type: "none" },
    fontSize: o.fontSize || 14, color: o.color || C.text1, bold: !!o.bold,
    align: o.align || "center", valign: o.valign || "middle", margin: o.margin ?? 0.08,
    isTextBox: true, objectName: nm("box"),
  }, o.extra || {}));
}
function arrow(s, x1, y1, x2, y2, o = {}) {
  const flipH = x2 < x1, flipV = y2 < y1;
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH, flipV,
    line: { color: o.color || C.text1, width: o.width || 1.5, dashType: o.dash || "solid", endArrowType: o.noHead ? undefined : "triangle" },
    objectName: nm("arrow"),
  });
}
function bullets(s, items, x, y, w, h, o = {}) {
  const arr = items.map((it, i) => {
    const t = typeof it === "string" ? [{ text: it }] : it;
    return t.map((run, j) => ({
      text: run.text,
      options: Object.assign({ bullet: j === 0 ? (o.bullet === false ? false : { indent: 16 }) : undefined, breakLine: j === t.length - 1 && i < items.length - 1 }, run.options || {}),
    }));
  }).flat();
  s.addText(arr, { x, y, w, h, fontSize: o.fontSize || 16, color: o.color || C.text1, valign: "top", margin: 0, paraSpaceAfter: o.space ?? 8, isTextBox: true, objectName: nm("list") });
}
function code(s, text, x, y, w, h, fs = 12) {
  s.addText(text, {
    x, y, w, h, fontFace: "Courier New", fontSize: fs, color: C.text1, valign: "top", align: "left",
    fill: { color: C.background2 }, margin: 0.15, isTextBox: true, objectName: nm("code"),
    shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06,
  });
}
function caption(s, text, x, y, w, h = 0.45) {
  txt(s, text, { x, y, w, h, fontSize: 11, color: C.text1, transparency: 25, italic: false });
}
function stat(s, big, small, x, y, w, color) {
  txt(s, big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || C.text2 });
  txt(s, small, { x, y: y + 0.78, w, h: 0.75, fontSize: 13, color: C.text1 });
}
const fmt = (v, d = 2) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 0.1 ? v.toFixed(d) : v.toPrecision(2));
const get = (name, N, p) => D.sweep[name].find((r) => r.N === N && Math.abs(r.p - p) < 1e-9);
const wkey = (N, p) => `${N}|${Number.isInteger(p) ? p.toFixed(1) : p}`;
const PS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0];
const NS = [10, 100, 1000, 10000];

// Log-log scatter (x axis log scale is added in post-processing; pptxgenjs only exposes a log value axis)
function logChart(s, { xs, series, x, y, w, h, title, xTitle, yTitle, ymin, ymax, legend = true, ylinear = false, markersOnly = false, ystep }) {
  const data = [{ name: "X", values: xs }].concat(series.map((sr) => ({ name: sr.name, values: sr.values })));
  s.addChart(pres.charts.SCATTER, data, {
    x, y, w, h, objectName: nm("chart"),
    chartColors: series.map((sr) => sr.color),
    lineSize: markersOnly ? 0 : 2.25, lineDataSymbol: "circle", lineDataSymbolSize: markersOnly ? 10 : 7,
    valAxisLogScaleBase: ylinear ? undefined : 10, valAxisMinVal: ymin, valAxisMaxVal: ymax, valAxisMajorUnit: ystep,
    showValAxisTitle: true, valAxisTitle: yTitle, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.ink, valAxisTitleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: xTitle, catAxisTitleFontSize: 12, catAxisTitleColor: HEX.ink, catAxisTitleFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valAxisLabelFormatCode: "[<1]0.#####;#,##0", catAxisLabelFormatCode: "[<1]0.#####;#,##0",
    valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { color: HEX.grid, size: 0.5 },
    showLegend: legend, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.ink,
    showTitle: !!title, title, titleFontSize: 14, titleColor: HEX.ink, titleFontFace: "+mn-lt",
  });
}
function sweepSeries(N, names = BACKENDS) {
  return names.map((n) => ({ name: n, color: BCOLOR[n], values: PS.map((p) => { const r = get(n, N, p); return r ? r.t : null; }) }));
}

const SECT = { intro: "Question", bg: "Background", model: "Cost models", impl: "Implementation", val: "Validation", res: "Results", lessons: "Lessons", close: "Close", backup: "Backup" };
const notes = []; // collected for script.md
function addNotes(s, num, title, minutes, body) {
  const text = `[${minutes} min] ${body}`;
  s.addNotes(text);
  notes.push({ num, title, minutes, body });
}

// ================================================================= SLIDES
let n = 0;

// 1 ---------------------------------------------------------------- title
{
  pres.addSection({ title: SECT.intro });
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: SECT.intro });
  s.addText("One network, three execution strategies: where spiking-network simulation crosses over on CPU and GPU", { placeholder: "title" });
  s.addText([
    { text: "Cannon Wilson", options: { bold: true, breakLine: true } },
    { text: "M.S. Computer Science, UW–Madison · Software Engineer, CACI", options: { breakLine: true } },
    { text: "Technical presentation · ARL:UT Environmental Sciences Laboratory · November 20, 2026", options: { fontSize: 14 } },
  ], { placeholder: "body" });
  addNotes(s, ++n, "Title", 0.5, `- Good morning. I'm Cannon; I build software that has to be correct and fast — most recently test and data tooling at CACI, before that an M.S. in CS at UW–Madison with an HPC focus.
- This talk is a personal project: the same simulation run three different ways, measured on CPU and GPU.
- I'll keep the neuroscience to a minimum — the core of it is a recursive filter, a sparse matrix, and a priority queue.`);
}

// 2 ---------------------------------------------------------------- question
{
  const s = content("Question", "Same network, three execution strategies: where are the crossovers as N and p vary?", SECT.intro);
  box(s, 0.6, 2.75, 2.6, 1.9, "One network\n\nN neurons\nconnection probability p\nsynaptic delays", { fill: C.text2, color: C.background1, fontSize: 15, bold: false });
  const rows = [
    ["Clock-driven dense", "batched dense mat-vec", C.accent1, "CPU · GPU"],
    ["Clock-driven sparse", "CSR sparse mat-vec", C.accent3, "CPU · GPU"],
    ["Event-driven", "priority queue (heap)", C.accent5, "CPU"],
  ];
  rows.forEach(([name, prim, col, dev], i) => {
    const y = 1.95 + i * 1.25;
    arrow(s, 3.25, 3.7, 4.05, y + 0.5, { color: C.text1 });
    box(s, 4.1, y, 3.6, 1.0, [{ text: name, options: { bold: true, breakLine: true, fontSize: 16 } }, { text: prim, options: { fontSize: 14 } }], { fill: col, color: C.background1 });
    box(s, 7.85, y + 0.25, 1.3, 0.5, dev, { fill: C.background2, fontSize: 13, bold: true });
  });
  txt(s, "The questions", { x: 9.6, y: 1.95, w: 3.1, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    "Which strategy is fastest as network size N and connection probability p change, on CPU vs GPU?",
    "Can the computational primitive underneath each one predict where the crossovers fall?",
  ], 9.6, 2.45, 3.1, 3.5, { fontSize: 16, space: 12 });
  addNotes(s, ++n, "Question", 2, `- Same network, same parameters; only the execution strategy and device change.
- Clock-driven: advance every neuron every time step — like sampling every channel at a fixed rate. Dense stores every possible connection; sparse stores only real ones (CSR).
- Event-driven: only do work when a spike arrives, scheduled in a priority queue.
- Two questions: where are the crossovers, and can I predict them from the primitive — dense GEMV, sparse SpMV, heap operations.
- If asked "why does anyone care": choosing a simulator today is mostly folklore; I wanted a cost model I could check against measurements.`);
}

// 3 ---------------------------------------------------------------- preview: winner map
{
  const s = content("Answer up front", "Preview: event-driven wins small and sparse, the GPU wins at scale, and the cost models explain the shape", SECT.intro);
  const short = { "Dense CPU": "Dense CPU", "Dense GPU": "Dense GPU", "Sparse CPU": "Sparse CPU", "Sparse GPU": "Sparse GPU", "Event-driven CPU": "Event" };
  const textOn = { "Dense CPU": "FFFFFF", "Dense GPU": "FFFFFF", "Sparse CPU": "FFFFFF", "Sparse GPU": HEX.ink, "Event-driven CPU": "FFFFFF" };
  const header = [{ text: "N \\ p", options: { bold: true, color: HEX.ink, fill: { color: HEX.mist } } }].concat(PS.map((p) => ({ text: String(p), options: { bold: true, color: HEX.ink, fill: { color: HEX.mist }, align: "center" } })));
  const rows = [header];
  NS.slice().reverse().forEach((N) => {
    const row = [{ text: N.toLocaleString("en-US"), options: { bold: true, color: HEX.ink, fill: { color: HEX.mist } } }];
    PS.forEach((p) => {
      const w = D.derived.winner[wkey(N, p)];
      const noEvent = !get("Event-driven CPU", N, p);
      let label = short[w.winner];
      if (N === 10 && p === 0.01) label = "no event data";
      row.push({ text: label + (noEvent && !(N === 10 && p === 0.01) ? "*" : ""), options: { fill: { color: N === 10 && p === 0.01 ? "FFFFFF" : BCOLOR[w.winner] }, color: N === 10 && p === 0.01 ? HEX.muted : textOn[w.winner], align: "center", fontSize: 11, bold: true } });
    });
    rows.push(row);
  });
  s.addTable(rows, { x: 0.6, y: 1.95, w: 8.3, colW: [0.86].concat(Array(9).fill(0.826)), rowH: 0.62, fontSize: 12, fontFace: "Calibri", border: { type: "solid", color: "FFFFFF", pt: 2 }, valign: "middle", objectName: nm("winnermap") });
  caption(s, "Fastest backend per (N, p) in the Apr 30 sweep: Δt = 1 ms, 1 s simulated, one run per cell. *Event-driven not recorded at N = 10⁴, p ≥ 0.1. Source: logs/04-30/results.csv", 0.6, 5.2, 8.3, 0.6);
  bullets(s, [
    [{ text: "N ≤ 100: ", options: { bold: true } }, { text: "event-driven, 8.7× to 1,600× ahead — clock-driven pays a fixed cost every step" }],
    [{ text: "N = 10³: ", options: { bold: true } }, { text: "event-driven up to p = 0.05, then dense GPU" }],
    [{ text: "N = 10⁴: ", options: { bold: true } }, { text: "sparse GPU up to p = 0.5, dense GPU above" }],
    [{ text: "Cost models predict the shapes; ", options: { bold: true } }, { text: "the crossover points depend on constants: launch overhead, bytes per edge, cost per heap operation" }],
  ], 9.3, 1.95, 3.45, 4.8, { fontSize: 15, space: 10 });
  addNotes(s, ++n, "Preview", 1.5, `- Here's the answer before the details. Each cell is the fastest backend for one network size and connection probability.
- Small networks: event-driven wins by orders of magnitude, because clock-driven pays a fixed cost every step even when nothing happens.
- Large networks: the GPU wins everywhere; sparse up to p = 0.5, dense above that.
- The cost models get the shapes right; the exact crossover depends on hardware constants I'll show later.
- Caveat to say out loud: one run per cell, single machine. These are first measurements, not a survey.
- If asked "what's the blank cell": at N = 10, p = 0.01 nothing spikes and there's no event-driven row in the log.`);
}

// 4 ---------------------------------------------------------------- LIF = one-pole IIR
{
  const s = content("Background", "A spiking neuron is a leaky integrator: a one-pole IIR low-pass filter plus a threshold", SECT.bg);
  txt(s, "x[n]\ninput spikes × weights", { x: 0.6, y: 2.95, w: 1.75, h: 0.9, fontSize: 14, align: "left" });
  arrow(s, 2.1, 3.4, 2.6, 3.4);
  s.addText("Σ", { shape: pres.shapes.OVAL, x: 2.6, y: 3.05, w: 0.7, h: 0.7, fill: { color: C.background2 }, line: { color: C.text1, width: 1.25 }, fontSize: 20, bold: true, align: "center", valign: "middle", isTextBox: true, objectName: nm("sum") });
  arrow(s, 3.3, 3.4, 5.6, 3.4);
  txt(s, "V[n]", { x: 4.55, y: 2.95, w: 0.8, h: 0.4, fontSize: 15, bold: true });
  box(s, 5.6, 2.9, 1.9, 1.0, "V ≥ θ ?", { fill: C.accent5, color: C.background1, fontSize: 17, bold: true });
  arrow(s, 7.5, 3.4, 8.1, 3.4);
  txt(s, "spike\ns[n]", { x: 8.15, y: 3.05, w: 0.9, h: 0.75, fontSize: 14, bold: true });
  s.addShape(pres.shapes.LINE, { x: 4.4, y: 3.4, w: 0.001, h: 1.35, line: { color: C.text1, width: 1.5 }, objectName: nm("fb") });
  box(s, 3.95, 4.75, 0.9, 0.6, "z⁻¹", { fill: C.background2, line: HEX.ink, fontSize: 16, bold: true });
  arrow(s, 3.95, 5.05, 3.4, 5.05);
  box(s, 2.5, 4.75, 0.9, 0.6, "× α", { fill: C.background2, line: HEX.ink, fontSize: 16, bold: true });
  arrow(s, 2.95, 4.75, 2.95, 3.75);
  arrow(s, 6.55, 3.9, 6.55, 4.75, { dash: "dash" });
  box(s, 5.45, 4.75, 2.2, 0.75, "reset V → V_rest\nhold for refractory period", { fill: C.background2, fontSize: 12 });
  // equation card
  box(s, 9.25, 1.95, 3.5, 4.5, "", { fill: C.background2 });
  txt(s, "V[n+1] = α · V[n] + x[n]", { x: 9.45, y: 2.15, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  txt(s, "α = e^(−Δt / τ)", { x: 9.45, y: 2.7, w: 3.2, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    "Pole at z = α: a leaky integrator; cutoff ≈ 1/(2πτ) ≈ 16 Hz for τ = 10 ms",
    "τ = 10 ms: α = 0.905 at Δt = 1 ms, 0.990 at Δt = 0.1 ms",
    "Exact solution between inputs, not an Euler step",
    "Nonlinear part: threshold, reset, refractory hold",
  ], 9.45, 3.3, 3.15, 3.1, { fontSize: 14, space: 8 });
  caption(s, "Source: simulations/clock_driven_dense/backend.cpp (voltage update), shared/simulation_config.py (membrane_decay = exp(−Δt/τ))", 0.6, 6.45, 8.4);
  addNotes(s, ++n, "LIF as a one-pole filter", 2, `- If you remember one thing about the neuron: it's a one-pole IIR low-pass filter with a comparator on the output.
- Input is a sum of weighted spikes; the state decays by alpha each step, alpha = e^(−dt/tau). That's the exact discrete-time solution of the RC circuit, not an Euler approximation.
- When the voltage crosses threshold it emits a spike, resets, and holds for a refractory period. That's the only nonlinearity.
- Tau = 10 ms gives a cutoff around 16 Hz — it smooths the input spike train.
- If asked "why exact instead of Euler": with a constant input between steps the exponential is the closed form; it costs the same and removes a stability limit on dt.`);
}

// 5 ---------------------------------------------------------------- 2-pole model
{
  const s = content("Background", "The benchmarked model cascades two one-pole filters: synaptic current, then membrane voltage", SECT.bg);
  const y = 2.6;
  txt(s, "spikes\nΣ w·s", { x: 0.6, y: y + 0.15, w: 1.2, h: 0.8, fontSize: 14, bold: true });
  arrow(s, 1.75, y + 0.55, 2.3, y + 0.55);
  box(s, 2.3, y, 3.2, 1.1, [{ text: "Synaptic current I", options: { bold: true, breakLine: true } }, { text: "one pole, β = e^(−Δt/τs), τs = 5 ms", options: { fontSize: 13 } }], { fill: C.accent3, color: C.background1, fontSize: 16 });
  arrow(s, 5.5, y + 0.55, 6.1, y + 0.55);
  txt(s, "R · I", { x: 5.55, y: y + 0.05, w: 0.6, h: 0.4, fontSize: 13, bold: true });
  box(s, 6.1, y, 3.2, 1.1, [{ text: "Membrane voltage V", options: { bold: true, breakLine: true } }, { text: "one pole, α = e^(−Δt/τm), τm = RC = 10 ms", options: { fontSize: 13 } }], { fill: C.accent1, color: C.background1, fontSize: 16 });
  arrow(s, 9.3, y + 0.55, 9.9, y + 0.55);
  box(s, 9.9, y + 0.05, 1.5, 1.0, "V ≥ θ ?", { fill: C.accent5, color: C.background1, fontSize: 16, bold: true });
  arrow(s, 11.4, y + 0.55, 11.9, y + 0.55);
  txt(s, "spike", { x: 11.95, y: y + 0.35, w: 0.8, h: 0.4, fontSize: 14, bold: true });
  code(s, "I[n+1] = β·I[n] + ring[now] + Poisson[n]\nV[n+1] = α·V[n] + (1 − α)·(R·I[n+1] + V_rest)", 0.6, 4.2, 7.6, 0.95, 14);
  bullets(s, [
    "Two real poles: β = 0.819 and α = 0.905 at Δt = 1 ms",
    "(1 − α) gives unity DC gain from R·I to V",
    "This is the model in the Apr 30 benchmark sweep. The Brian2 cross-check (later) uses the one-pole version from the previous slide",
  ], 8.55, 4.2, 4.2, 2.4, { fontSize: 15 });
  caption(s, "Source: backend.cpp loops at commit 63ab99a (Apr 30): synaptic_currents.mul_(synaptic_decay); new_voltages = alpha·(1−decay) + V·decay", 0.6, 6.45, 12);
  addNotes(s, ++n, "Two-pole model [CUT IF LONG]", 1.5, `- The benchmark sweep used a slightly richer neuron: the input first goes through a synaptic-current filter, then the membrane filter. Two cascaded one-pole low-pass filters.
- Poles at 0.82 and 0.90 for a 1 ms step. The (1 − alpha) term gives unity DC gain.
- I want to be upfront: the big April sweep used this model, and later I moved the dense backend to the simpler one-pole model to match Brian2 exactly. I'll be clear about which results use which.
- If asked "does the model change the performance results": the per-step work is the same element-wise ops plus one extra multiply-add per neuron; the dominant costs — mat-vec, SpMV, heap — don't change.`);
}

// 6 ---------------------------------------------------------------- network
{
  const s = content("Background", "The network: a random directed graph with delayed connections and Poisson input", SECT.bg);
  // random graph drawing (deterministic layout)
  const pts = [[1.2, 2.3], [2.4, 2.0], [3.5, 2.5], [3.9, 3.6], [3.3, 4.7], [2.1, 5.0], [1.0, 4.3], [0.9, 3.3], [2.3, 3.4], [2.8, 4.1]];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [8, 2], [0, 8], [8, 5], [9, 3], [6, 9], [1, 9], [4, 8], [9, 7], [2, 6]];
  edges.forEach(([a, b]) => {
    const [x1, y1] = pts[a], [x2, y2] = pts[b];
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), r = 0.2;
    arrow(s, x1 + 0.2 + dx / L * r, y1 + 0.2 + dy / L * r, x2 + 0.2 - dx / L * r, y2 + 0.2 - dy / L * r, { color: "8A94A6", width: 1 });
  });
  pts.forEach(([x, y], i) => s.addShape(pres.shapes.OVAL, { x, y, w: 0.4, h: 0.4, fill: { color: i % 5 === 4 ? C.accent5 : C.accent3 }, line: { color: C.background1, width: 1.5 }, objectName: nm("node") }));
  caption(s, "Each directed edge present with probability p; each edge has its own delay. (Red: inhibitory, used in the Brian2 setup only.)", 0.6, 5.65, 4.0, 0.8);
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rows = [
    [H(""), H("Benchmark sweep (Apr 30)"), H("Brian2 cross-check (May 21)")],
    ["Neurons N", "10, 10², 10³, 10⁴", "1,000"],
    ["Connection prob. p", "0.01 … 1 (9 values)", "0.01, 0.1, 0.3"],
    ["Edges", "exactly ⌊N²p⌋, equal weights", "same (Brian2: Bernoulli per pair)"],
    ["Excitatory / inhibitory", "all excitatory", "80% / 20%, inhibitory weight −5×"],
    ["Neuron model", "two-pole (current-based)", "one-pole (voltage jump)"],
    ["Synaptic delays", "uniform 2–4 ms", "uniform 2–4 ms"],
    ["External input", "Poisson, 50 Hz per neuron", "Poisson, 2× threshold rate (Brunel 2000)"],
    ["Refractory period", "20 ms", "2 ms"],
    ["Δt / simulated time", "1 ms / 1 s (1,000 steps)", "0.1 ms / 1 s (10,000 steps)"],
    ["Runs per point", "1", "3"],
  ];
  s.addTable(rows, { x: 4.9, y: 1.95, w: 7.83, colW: [2.1, 2.7, 3.03], fontSize: 13, fontFace: "Calibri", color: HEX.ink, rowH: 0.4, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("params") });
  caption(s, "Sources: simulations/run_all.py at 63ab99a (sweep) and at HEAD eb02eb9 (Brian2); shared/graph_creation.py; simulations/brian2/runner.py", 4.9, 6.45, 7.8);
  addNotes(s, ++n, "Network", 2, `- The network is a random directed graph: every possible edge exists independently with probability p, and each edge gets its own delay, uniform between 2 and 4 ms.
- External drive is Poisson spike input — think of it as a noise floor that keeps neurons near threshold.
- Two configurations, and I'm keeping them separate on purpose. Left: the big benchmark sweep, all-excitatory, 1 ms step. Right: a Brunel-2000-style excitatory/inhibitory network I used to validate against Brian2, at 0.1 ms.
- The 20 ms refractory period in the sweep caps firing at 50 Hz — at large N neurons run near that cap (about 48 Hz). That makes spike counts nearly constant across p, which matters for reading the event-driven results.
- If asked "why exact edge counts": it removes edge-count variance between backends so runtime differences aren't graph noise.`);
}

// 7 ---------------------------------------------------------------- clock vs event
{
  const s = content("Background", "Clock-driven means sampling every neuron at a fixed rate; event-driven means doing work only when a spike arrives", SECT.bg);
  const x0 = 2.3, x1 = 9.3, rowsY = [2.25, 2.55, 2.85, 3.15];
  txt(s, "Clock-driven", { x: 0.6, y: 2.35, w: 1.6, h: 0.8, fontSize: 16, bold: true, color: C.accent1 });
  for (let k = 0; k <= 20; k++) {
    const xx = x0 + k * (x1 - x0) / 20;
    rowsY.forEach((yy) => s.addShape(pres.shapes.OVAL, { x: xx - 0.06, y: yy, w: 0.12, h: 0.12, fill: { color: C.accent1 }, line: { type: "none" }, objectName: nm("tick") }));
  }
  arrow(s, x0 - 0.1, 3.5, x1 + 0.3, 3.5, { color: "8A94A6" });
  txt(s, "time → every Δt, every neuron", { x: x0, y: 3.55, w: 5, h: 0.35, fontSize: 12, color: C.text1 });
  txt(s, "Event-driven", { x: 0.6, y: 4.45, w: 1.6, h: 0.8, fontSize: 16, bold: true, color: C.accent5 });
  const ev = [[0, 0.9], [0, 4.6], [1, 2.2], [1, 5.3], [2, 1.0], [2, 3.1], [3, 2.8], [3, 6.4], [1, 6.6], [2, 6.9]];
  const rows2 = [4.35, 4.65, 4.95, 5.25];
  ev.forEach(([r, t]) => s.addShape(pres.shapes.OVAL, { x: x0 + t - 0.07, y: rows2[r], w: 0.14, h: 0.14, fill: { color: C.accent5 }, line: { type: "none" }, objectName: nm("ev") }));
  arrow(s, x0 + 0.9, 4.43, x0 + 2.12, 4.77, { color: C.accent5, width: 1 });
  arrow(s, x0 + 0.9, 4.43, x0 + 2.73, 5.32, { color: C.accent5, width: 1 });
  arrow(s, x0 - 0.1, 5.6, x1 + 0.3, 5.6, { color: "8A94A6" });
  txt(s, "time → only at spike arrivals, irregular", { x: x0, y: 5.65, w: 5, h: 0.35, fontSize: 12 });
  box(s, 9.9, 1.95, 2.85, 1.85, [{ text: "Work ∝ (T / Δt) × N", options: { bold: true, breakLine: true, fontSize: 16 } }, { text: "even when nothing happens. Like sampling every channel at a fixed rate.", options: { fontSize: 13 } }], { fill: C.background2, align: "left", valign: "top", margin: 0.15 });
  box(s, 9.9, 4.05, 2.85, 1.95, [{ text: "Work ∝ spike deliveries × log(queue)", options: { bold: true, breakLine: true, fontSize: 16 } }, { text: "Each neuron's state is advanced by e^(−Δt/τ) with its own variable Δt.", options: { fontSize: 13 } }], { fill: C.background2, align: "left", valign: "top", margin: 0.15 });
  caption(s, "Event-driven update: simulations/event_driven/backend.cpp lines 160–172 (per-neuron dt, std::exp per event)", 0.6, 6.45, 9);
  addNotes(s, ++n, "Clock vs event", 1.5, `- Clock-driven: every neuron, every step. Cost scales with steps times neurons, whether or not anything happens. Fixed-rate sampling.
- Event-driven: keep a queue of future spike arrivals; pop the earliest, update only that neuron. Because each neuron is updated at irregular times, the decay factor is computed per event with its own dt.
- The trade: clock-driven has regular, vectorizable work — great for GPUs. Event-driven skips idle time but does irregular, pointer-chasing work that's hard to parallelize.
- If asked about accuracy: event-driven has no time grid, so spike times are not rounded to dt; clock-driven rounds delays and spike times to the step.`);
}

// 8 ---------------------------------------------------------------- primitives
{
  const s = content("Cost models", "Each strategy reduces to one computational primitive", SECT.model);
  const cols = [
    { x: 0.6, title: "Clock-driven dense", col: C.accent1, prim: "Batched dense mat-vec", sub: "[B, N, N] × [N] every step (one N×N matrix per delay bucket B)", codeTxt: "bucketized_weights.matmul(spikes)" },
    { x: 4.75, title: "Clock-driven sparse", col: C.accent3, prim: "CSR sparse mat-vec", sub: "B calls of SpMV per step, over every stored edge — spiking or not", codeTxt: "torch::mv(W[b], spikes)" },
    { x: 8.9, title: "Event-driven", col: C.accent5, prim: "Binary-heap push / pop", sub: "Pop earliest event; on a spike push N·p future arrivals", codeTxt: "event_queue.push({t + d[j], col[j], w[j]})" },
  ];
  cols.forEach((c, i) => {
    box(s, c.x, 1.95, 3.85, 0.6, c.title, { fill: c.col, color: C.background1, bold: true, fontSize: 17 });
    // mini diagram
    const gx = c.x + 0.4, gy = 2.8;
    if (i < 2) {
      const filled = (r, k) => (i === 0 ? true : (r * 7 + k * 3) % 5 === 0);
      for (let r = 0; r < 6; r++) for (let k = 0; k < 6; k++) s.addShape(pres.shapes.RECTANGLE, { x: gx + k * 0.24, y: gy + r * 0.24, w: 0.21, h: 0.21, fill: { color: filled(r, k) ? c.col : C.background2, transparency: filled(r, k) ? 20 : 0 }, line: { type: "none" }, objectName: nm("cell") });
      txt(s, "×", { x: gx + 1.5, y: gy + 0.5, w: 0.3, h: 0.4, fontSize: 18, bold: true, align: "center" });
      for (let r = 0; r < 6; r++) s.addShape(pres.shapes.RECTANGLE, { x: gx + 1.85, y: gy + r * 0.24, w: 0.21, h: 0.21, fill: { color: [1, 4].includes(r) ? C.accent5 : C.background2 }, line: { type: "none" }, objectName: nm("vec") });
      txt(s, "=", { x: gx + 2.1, y: gy + 0.5, w: 0.3, h: 0.4, fontSize: 18, bold: true, align: "center" });
      for (let r = 0; r < 6; r++) s.addShape(pres.shapes.RECTANGLE, { x: gx + 2.45, y: gy + r * 0.24, w: 0.21, h: 0.21, fill: { color: c.col, transparency: 50 }, line: { type: "none" }, objectName: nm("out") });
      txt(s, "W[b]        s", { x: gx, y: gy + 1.48, w: 2.4, h: 0.3, fontSize: 11, color: C.text1 });
    } else {
      const nodes = [[1.45, 0], [0.75, 0.6], [2.15, 0.6], [0.35, 1.2], [1.15, 1.2], [1.75, 1.2], [2.55, 1.2]];
      [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]].forEach(([a, b]) => arrow(s, gx + nodes[a][0] + 0.2, gy + nodes[a][1] + 0.4, gx + nodes[b][0] + 0.2, gy + nodes[b][1], { noHead: true, color: "8A94A6", width: 1 }));
      nodes.forEach(([nx, ny], k) => s.addText(k === 0 ? "t min" : "", { shape: pres.shapes.OVAL, x: gx + nx, y: gy + ny, w: 0.4, h: 0.4, fill: { color: k === 0 ? C.accent5 : C.background2 }, line: { color: C.accent5, width: 1 }, fontSize: 8, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: nm("heap") }));
    }
    txt(s, c.prim, { x: c.x, y: 4.75, w: 3.85, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    txt(s, c.sub, { x: c.x, y: 5.15, w: 3.85, h: 0.75, fontSize: 14 });
    code(s, c.codeTxt, c.x, 5.95, 3.85, 0.45, 11);
  });
  addNotes(s, ++n, "Primitives", 2, `- Strip away the neuroscience and each backend is one primitive.
- Dense: one N×N weight matrix per delay bucket, multiplied by the 0/1 spike vector every step — a batched GEMV. Same work at p = 0.01 as at p = 1.
- Sparse: the same, but each bucket is a CSR matrix, so work scales with the number of stored edges. Note it still multiplies the full spike vector — it exploits sparse connectivity, not sparse activity.
- Event-driven: a binary heap. Each spike pushes N·p future arrivals; each arrival is a pop plus a neuron update.
- If asked "why not skip non-spiking columns in the dense case": you could gather the spiking columns first; at 25–48 Hz with 1 ms steps only 2–5% of neurons spike per step, so that's a real optimization I didn't try.`);
}

// 9 ---------------------------------------------------------------- cost models
{
  const s = content("Cost models", "Cost models: dense is flat in p, sparse grows with N²p, event-driven grows with spikes × N·p", SECT.model);
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rows = [
    [H("Strategy"), H("Work for T seconds simulated"), H("Grows with p?"), H("Grows with firing rate?"), H("Bytes read per step")],
    [{ text: "Dense", options: { bold: true, color: HEX.denseCPU } }, "(T/Δt) · B · N²", "no", "no", "4 · B · N²  (1.2 GB at N = 10⁴)"],
    [{ text: "Sparse (CSR)", options: { bold: true, color: HEX.sparseCPU } }, "(T/Δt) · N²p", "linearly", "no", "≈ 12 · N²p  (4 B value + 8 B index)"],
    [{ text: "Event-driven", options: { bold: true, color: HEX.event } }, "S · N·p · log Q,  S = N·r·T spikes", "linearly", "linearly", "16 B per queued event"],
    [{ text: "All clock-driven", options: { bold: true } }, "+ (T/Δt) × ops per step × per-op overhead", "no", "no", "— the fixed floor"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 12.13, colW: [1.9, 3.6, 1.45, 1.95, 3.23], fontSize: 14, fontFace: "Calibri", color: HEX.ink, rowH: 0.5, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("cost") });
  txt(s, "Checked against the Apr 30 log", { x: 0.6, y: 4.65, w: 6, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  const chk = [
    ["Dense GPU, N = 10⁴", "1.923–1.924 s at all 9 values of p", C.accent2],
    ["Sparse CPU, N = 10⁴", "12.5 s → 173 s from p = 0.1 to 1  (×13.8 for ×10 in p)", C.accent3],
    ["Event-driven, N = 10³", "0.045 s → 11.6 s from p = 0.01 to 1  (×256; firing also rose 23 → 45 Hz)", C.accent5],
  ];
  chk.forEach(([a, b, col], i) => box(s, 0.6 + i * 4.1, 5.1, 3.93, 1.25, [{ text: a, options: { bold: true, breakLine: true, fontSize: 15 } }, { text: b, options: { fontSize: 13 } }], { fill: col, transparency: 82, align: "left", valign: "top", margin: 0.14 }));
  caption(s, "B = 3 delay buckets at Δt = 1 ms. Byte counts are analytic (float32 weights, int64 CSR column indices); nothing on this slide measured memory. Q = queue length, r = firing rate.", 0.6, 6.45, 12.1);
  addNotes(s, ++n, "Cost models", 2.5, `- Dense: steps × buckets × N². No p, no firing rate. Bytes per step are 4·B·N² — 1.2 GB per step at N = 10,000 with 3 buckets.
- Sparse: proportional to stored edges, N²p. About 12 bytes per edge: a float value plus a 64-bit column index.
- Event-driven: number of spikes times out-degree times log of the queue length.
- Clock-driven also has a fixed floor: steps × ops per step × per-op overhead. That floor decides small-N behavior.
- Three checks from the log: dense GPU dead flat in p; sparse CPU roughly linear in p at large N; event-driven grows faster than p because the firing rate also rose.
- If asked "where's log Q": the per-event cost rises from about 120 to 300 ns as N grows — consistent with a deeper heap and more cache misses, but I haven't separated the two.`);
}

// 10 ---------------------------------------------------------------- architecture
{
  const s = content("Implementation", "Architecture: Python sets up the experiment, C++ runs the loop, every backend has the same interface", SECT.impl);
  box(s, 0.6, 1.95, 12.13, 0.38, "Python orchestration", { fill: C.background1, color: C.text2, bold: true, fontSize: 14, align: "left", margin: 0 });
  const py = [["run_all.py", "sweep N × p × repeats\ndeterministic seed per config"], ["SimulationConfig", "dataclass, validated in __post_init__\n(Δt < τ, delays ≥ Δt, …)"], ["runner.py (per backend)", "build graph, delays, buckets;\npreallocate state tensors"]];
  py.forEach(([a, b], i) => { box(s, 0.6 + i * 4.1, 2.35, 3.8, 1.05, [{ text: a, options: { bold: true, breakLine: true, fontSize: 15 } }, { text: b, options: { fontSize: 12 } }], { fill: C.background2 }); if (i < 2) arrow(s, 4.4 + i * 4.1, 2.88, 4.7 + i * 4.1, 2.88); });
  s.addShape(pres.shapes.LINE, { x: 0.6, y: 3.7, w: 12.13, h: 0, line: { color: C.accent6, width: 1.5, dashType: "dash" }, objectName: nm("boundary") });
  txt(s, "pybind11 boundary: torch::Tensor handles cross by reference — the C++ side writes into the same storage Python allocated", { x: 0.6, y: 3.78, w: 12.13, h: 0.35, fontSize: 13, bold: true, color: C.accent6 });
  const cpp = [["ClockDrivenDense", C.accent1], ["ClockDrivenSparse", C.accent3], ["EventDriven", C.accent5]];
  cpp.forEach(([a, col], i) => box(s, 0.6 + i * 4.1, 4.25, 3.8, 0.75, [{ text: a, options: { bold: true, breakLine: true } }, { text: "constructor(tensors, params) · run() → dict", options: { fontSize: 12 } }], { fill: col, color: C.background1, fontSize: 15 }));
  box(s, 0.6, 5.2, 7.9, 0.6, "libtorch tensor ops: CPU threads or CUDA kernels", { fill: C.background2, fontSize: 14 });
  box(s, 8.8, 5.2, 3.93, 0.6, "plain C++ loop over accessor<float,1>", { fill: C.background2, fontSize: 14 });
  const gl = [["Brian2", "established Python SNN simulator; outside reference"], ["libtorch", "PyTorch's C++ tensor library — same kernels, no Python"], ["pybind11", "C++ ↔ Python bindings; same role as a MATLAB MEX gateway"], ["CSR", "row pointers + column indices + values"]];
  gl.forEach(([a, b], i) => txt(s, [{ text: a + ": ", options: { bold: true } }, { text: b }], { x: 0.6 + i * 3.08, y: 6.0, w: 2.95, h: 0.75, fontSize: 12 }));
  addNotes(s, ++n, "Architecture", 2, `- Python does what Python is good at: the sweep, config validation, building the graph, allocating every tensor up front.
- The time loop is C++. Each backend is a small C++ class with the same constructor shape and a run() that returns spike counts and a timed-out flag — one interface, so the comparison is fair.
- Tensors cross the pybind11 boundary by reference; no copies. If you've written MEX files, pybind11 plays the same role as the mexFunction gateway.
- Timing wraps only run(), with a CUDA synchronize on both sides, so setup cost isn't counted.
- If asked "why move to C++": in the Python-loop version, interpreter overhead per step was a measurable part of runtime; moving the loop removed that confound across backends.`);
}

// 11 ---------------------------------------------------------------- per-step flow
{
  const s = content("Implementation", "One clock-driven time step is five stages, and the state never leaves the device", SECT.impl);
  const st = [
    ["1  Drive", "noise.uniform_()\nPoisson = noise < p_in"],
    ["2  Filter", "I ← β·I + ring[now] + Poisson\nV ← α·V + (1−α)(R·I + V_rest)\n(only outside refractory)"],
    ["3  Threshold", "s = (V ≥ θ)\n0/1 spike vector"],
    ["4  Propagate", "for each bucket b:\nring[t + k_b] += W[b] · s"],
    ["5  Reset, record", "V[s] ← V_rest;  t_last[s] ← t\nring[now] ← 0;  counts += s"],
  ];
  st.forEach(([a, b], i) => {
    const x = 0.6 + i * 2.47;
    box(s, x, 2.05, 2.25, 0.65, a, { fill: i === 3 ? C.accent2 : C.accent1, color: C.background1, bold: true, fontSize: 16 });
    box(s, x, 2.8, 2.25, 1.75, b, { fill: C.background2, fontSize: 12, align: "left", valign: "top", margin: 0.12, extra: { fontFace: "Courier New" } });
    if (i < 4) arrow(s, x + 2.27, 2.37, x + 2.45, 2.37);
  });
  bullets(s, [
    [{ text: "About 24 small tensor ops per step ", options: { bold: true } }, { text: "→ on the GPU, about 24 kernel launches per step, 1,000 steps per run" }],
    [{ text: "Dense loop has no host–device copies or .item() calls; ", options: { bold: true } }, { text: "counters accumulate on the device and are read once at the end" }],
    [{ text: "Stage 4 is the only part that grows with N² ", options: { bold: true } }, { text: "— everything else is O(N) element-wise" }],
  ], 0.6, 4.85, 12.1, 1.6, { fontSize: 15 });
  caption(s, "Source: run() in simulations/clock_driven_*/backend.cpp at 63ab99a. Op count is my count of tensor operations in the loop body.", 0.6, 6.45, 12);
  addNotes(s, ++n, "Per-step data flow", 2, `- Five stages per step. Draw the Poisson drive, run the two filters, threshold, propagate through the weights into the ring buffer, then reset and record.
- Stage 4 is the matrix work; everything else is element-wise over N neurons.
- I counted roughly 24 tensor ops per step. On a GPU each is a kernel launch — keep that number in mind for the dense-GPU results.
- The dense loop never copies to the host or calls .item() — spike counters live on the device and are read once at the end. That's the "keep data on the device" rule.
- If asked "what about the sparse loop": it indexes the ring buffer with a 0-dim device tensor; I believe libtorch turns that into an .item() call, meaning a device sync three times per step. That's a hypothesis I'd check with Nsight Systems.`);
}

// 12 ---------------------------------------------------------------- delay buckets + ring buffer
{
  const s = content("Implementation", "Delay buckets feed a ring buffer, so delayed spikes are scheduled without a queue", SECT.impl);
  txt(s, "1. Quantize each synapse's delay to whole steps", { x: 0.6, y: 1.95, w: 5.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, "d ~ U(2, 4) ms,  Δt = 1 ms  →  bucket b = ⌊d / Δt⌋ − 2", { x: 0.6, y: 2.4, w: 5.6, h: 0.4, fontSize: 14 });
  const bc = [C.accent1, C.accent2, C.accent3];
  [0, 1, 2].forEach((b) => {
    const x = 0.9 + b * 0.45, y = 3.0 + b * 0.45;
    s.addText(`W[${b}]`, { shape: pres.shapes.RECTANGLE, x, y, w: 1.9, h: 1.5, fill: { color: bc[b], transparency: b === 2 ? 70 : 10 }, line: { color: C.background1, width: 1.5 }, fontSize: 15, bold: true, color: b === 2 ? C.text1 : C.background1, align: "left", valign: "top", margin: 0.08, isTextBox: true, objectName: nm("bucket") });
  });
  bullets(s, [
    "One N×N slice per bucket: W[b] holds the edges whose delay falls in bucket b",
    "B = 3 buckets → dense weights are 3 · N² · 4 bytes",
    "W[2] (exactly 4 ms) is always empty when d < 4 ms — a third of the dense work is wasted",
  ], 3.6, 3.0, 2.75, 3.2, { fontSize: 13 });
  // ring buffer
  txt(s, "2. Scatter into a ring buffer of future inputs", { x: 6.75, y: 1.95, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const slots = 5, sx = 6.95, sy = 3.0, sw = 1.05;
  for (let k = 0; k < slots; k++) {
    const lab = ["now", "now+1", "now+2", "now+3", "now+4"][k];
    s.addText(lab, { shape: pres.shapes.RECTANGLE, x: sx + k * (sw + 0.08), y: sy, w: sw, h: 0.75, fill: { color: k === 0 ? C.accent5 : k >= 2 ? C.accent2 : C.background2, transparency: k >= 2 ? 40 : 0 }, line: { color: C.text1, width: 1 }, fontSize: 13, bold: true, color: k === 0 ? C.background1 : C.text1, align: "center", valign: "middle", isTextBox: true, objectName: nm("slot") });
  }
  txt(s, "read, then zero", { x: sx - 0.05, y: 3.8, w: 1.4, h: 0.35, fontSize: 12, color: C.accent5, bold: true });
  ["W[0]·s", "W[1]·s", "W[2]·s"].forEach((t, i) => {
    const cx = sx + (i + 2) * (sw + 0.08) + sw / 2;
    arrow(s, cx, 2.55, cx, 2.98, { color: C.accent2 });
    txt(s, t, { x: cx - 0.5, y: 2.3, w: 1.0, h: 0.28, fontSize: 12, bold: true, align: "center" });
  });
  arrow(s, sx + 5 * (sw + 0.08) - 0.05, 4.05, sx + 0.3, 4.05, { color: "8A94A6", dash: "dash" });
  txt(s, "wraps around: slot index = (t + k) mod 5", { x: sx + 1.6, y: 4.1, w: 4, h: 0.35, fontSize: 12, color: C.text1 });
  bullets(s, [
    "Size = ⌊max delay / Δt⌋ + 1 = 5 slots × N floats (200 KB at N = 10⁴)",
    "One index_add_ per step writes all buckets at once (dense)",
    "Same structure as a tapped delay line or an overlap-add output buffer",
  ], 6.95, 4.6, 5.8, 1.6, { fontSize: 14 });
  box(s, 6.95, 5.95, 5.78, 0.5, "Found while preparing this talk: reads landed one step early (delay k − 1). Now fixed; a single-spike test confirms k.", { fill: C.accent6, transparency: 80, fontSize: 12, align: "left", margin: 0.1 });
  caption(s, "Source: shared/clock_driven.py (_compute_delay_buckets, create_ring_buffer, create_lookup_tensors); backend.cpp run() loops", 0.6, 6.5, 6);
  addNotes(s, ++n, "Delay buckets and ring buffer (key slide)", 2.5, `- This is the key data structure. Delays are continuous, but a clock-driven sim can only deliver on step boundaries, so I quantize each delay to whole steps and group synapses into buckets — one weight matrix per bucket.
- Each step: multiply each bucket's matrix by the spike vector and add the result into a ring buffer slot that many steps in the future. Read the current slot, zero it, advance. No queue, no sorting — just a circular buffer of future inputs, like a tapped delay line.
- Two honest findings from re-reading the code for this talk. One: with delays uniform on [2, 4) ms, the third bucket is always empty, so a third of the dense mat-vec work is multiplying zeros. Two: the read pointer advanced before the read, so contributions arrived one step early — effective delays of 1–2 ms instead of 2–3 in the April sweep. Dense and sparse both had it; the old OpenMP version did not.
- I've fixed the second one: the read slot is now simply t mod B. A one-spike test — one synapse, known delay — failed before the fix and now arrives on exactly step k at both 1 ms and 0.1 ms.
- Neither changes the scaling story. The lesson is that a unit test on a single known spike would have caught it on day one.
- If asked "how big can the buffer get": slots = max delay / dt + 1 — 41 slots at dt = 0.1 ms. It's O(N) per slot, tiny next to the weights.`);
}

// 13 ---------------------------------------------------------------- event loop
{
  const s = content("Implementation", "Event-driven: one min-heap of 16-byte events, and each neuron is updated only when touched", SECT.impl);
  const steps = [
    ["Pop earliest event", "(time, target, weight, is_poisson)", C.accent5],
    ["Advance target neuron", "I ← I·e^(−Δt/τs) + w;  V toward R·I using its own Δt", C.accent1],
    ["V ≥ θ ?", "if yes: push N·p arrivals at t + d_j, reset, start refractory", C.accent2],
    ["Poisson inputs re-arm", "push next arrival at t + Exp(rate)", C.accent3],
  ];
  steps.forEach(([a, b, col], i) => {
    const y = 1.95 + i * 1.08;
    box(s, 0.6, y, 2.6, 0.85, a, { fill: col, color: C.background1, bold: true, fontSize: 15 });
    txt(s, b, { x: 3.35, y: y + 0.12, w: 3.6, h: 0.75, fontSize: 14 });
    if (i < 3) arrow(s, 1.9, y + 0.86, 1.9, y + 1.07);
  });
  arrow(s, 0.55, 5.35, 0.55, 2.35, { color: "8A94A6", dash: "dash" });
  code(s, "struct Event {\n    float time;\n    int   neuron_idx;\n    float weight;\n    bool  is_poisson;\n};                      // 16 bytes\n\nusing EventQueue = std::priority_queue<\n    Event, std::vector<Event>, std::greater<Event>>;", 7.3, 1.95, 5.43, 3.0, 13);
  stat(s, "≈120–300 ns", "per delivered spike, derived from logged runtime ÷ (spikes × out-degree), N ≥ 10³; rises with N", 7.3, 5.15, 5.4, C.accent5);
  caption(s, "Source: simulations/event_driven/backend.cpp. Single-threaded. Per-event cost is arithmetic on logs/04-30/results.csv, not a separate measurement.", 0.6, 6.45, 6.5, 0.6);
  addNotes(s, ++n, "Event-driven loop [CUT IF LONG]", 1.5, `- The whole event-driven engine is a std::priority_queue of 16-byte Event structs, ordered by time.
- Pop the earliest, advance just that neuron using its own elapsed time, check threshold. A spike pushes one event per outgoing edge at t plus that edge's delay.
- Poisson inputs are events too: each one schedules the next arrival with an exponential inter-arrival time — that's an exact Poisson process, not a per-step coin flip.
- Dividing runtime by spike deliveries gives about 120–300 ns per delivery, rising with N — a bigger heap and more cache misses.
- If asked "why float time": float32 has about 60 ns resolution at t = 1 s, fine for 1 s runs; for longer runs I'd use double or integer ticks.`);
}

// 14 ---------------------------------------------------------------- engineering details
{
  const s = content("Implementation", "Engineering details that keep the comparison fair and the runs reproducible", SECT.impl);
  const items = [
    ["Preallocate, then update in place", "State, ring buffer and counters are allocated in Python before the loop; the loop uses mul_, add_, index_add_, zero_. Some temporaries remain (comparisons, torch::where) — _out variants are on the list."],
    ["Share, don't copy", "Tensors cross pybind11 as handles to the same storage; the event-driven loop reads raw elements through accessor<float,1>."],
    ["Deterministic seeds", "Seed = blake2b(repeat, N, p). The Apr 30 sweep used Python's hash(), which is salted per process — those seeds are not reproducible."],
    ["Bounded runs", "SIGALRM + alarm(300 s) sets a flag the loop checks every step or event; timed-out runs are not written to the CSV."],
    ["Honest timing", "Only run() is timed; torch.cuda.synchronize() before and after."],
  ];
  items.forEach(([a, b], i) => txt(s, [{ text: a, options: { bold: true, breakLine: true, color: C.text2 } }, { text: b, options: { fontSize: 13 } }], { x: 0.6, y: 1.95 + i * 0.92, w: 6.4, h: 0.88, fontSize: 15 }));
  code(s, "volatile sig_atomic_t timed_out = 0;\n\npy::dict run() {\n  signal(SIGALRM, [](int) { timed_out = 1; });\n  alarm(max_runtime);              // 300 s\n  for (int t = 0; t < num_timesteps\n                  && !timed_out; t++) {\n    ...                            // one step\n  }\n  alarm(0);\n  return py::dict(..., \"timed_out\"_a = (bool)timed_out);\n}", 7.35, 1.95, 5.38, 3.65, 12);
  caption(s, "Sources: simulations/*/backend.cpp; shared/utils.py (get_deterministic_seed); shared/monitoring.py; run_all.py at 63ab99a (hash-based seed).", 7.35, 5.75, 5.38, 0.7);
  addNotes(s, ++n, "Engineering details", 1.5, `- Five habits. Allocate everything before the loop and update in place. Pass tensors by handle, never copy. Seed deterministically. Bound every run. Time only the part you mean to time.
- The timeout is a classic signal pattern: SIGALRM sets a sig_atomic_t flag and the loop checks it — no threads, no exceptions across the C++/Python boundary.
- Honest note: the April sweep's seeds came from Python's built-in hash, which is randomized per process, so those exact graphs can't be regenerated. I switched to a blake2b hash afterward.
- If asked "what happens to timed-out runs": they're dropped, which is why some event-driven points at N = 10,000 are missing. Better practice is to record them as censored at 300 s; that's on the list.`);
}

// 15 ---------------------------------------------------------------- validation 1
{
  const s = content("Validation", "My five backends agree within 1.3% on spike counts; event-driven runs 4–7% lower", SECT.val);
  logChart(s, { xs: PS, series: BACKENDS.map((b) => ({ name: b, color: BCOLOR[b], values: PS.map((p) => get(b, 1000, p).spn) })), x: 0.5, y: 1.85, w: 7.8, h: 4.6, title: "N = 1,000", xTitle: "Connection probability p (log scale)", yTitle: "Mean spikes per neuron in 1 s (= rate, Hz)", ymin: 0, ymax: 50, ystep: 10, ylinear: true, legend: true });
  stat(s, "≤ 1.3%", "max spread in mean spike count across the four clock-driven backends (all 18 points with N ≥ 10³)", 8.6, 2.0, 4.1, C.accent3);
  stat(s, "4.3–7.4%", "fewer spikes from event-driven than the clock-driven mean (all 11 points where it ran, N ≥ 10³)", 8.6, 3.75, 4.1, C.accent5);
  caption(s, "Apr 30 sweep: Δt = 1 ms, 1 s simulated, one run per point, different seed per backend. Source: logs/04-30/results.csv (mean_spikes_per_neuron).", 8.6, 5.5, 4.1, 0.85);
  addNotes(s, ++n, "Cross-backend validation", 1.5, `- First check: do my own backends agree? Same parameters, different code paths — dense versus sparse, CPU versus GPU.
- The four clock-driven backends agree within 1.3% on mean spike count at every point with N ≥ 1,000. They use different seeds, so that includes graph-to-graph variation.
- Event-driven is consistently lower, 4 to 7%. Consistent and one-sided means a systematic modeling difference, not noise — next two slides.
- Caveat: mean rate is a coarse statistic. Next steps are rate distributions and a single-neuron test against the closed form.
- If asked "why not compare spike trains": different seeds and float rounding make spike-by-spike comparison meaningless; statistics are the right level.`);
}

// 16 ---------------------------------------------------------------- Brian2
{
  const s = content("Validation", "Against Brian2 as an outside reference, mean firing rates differ by 1.6–2.7%", SECT.val);
  const b2 = D.brian2;
  const pv = ["0.01", "0.1", "0.3"];
  logChart(s, { xs: pv.map(Number), series: [
    { name: "My dense CPU backend (mean of 3)", color: HEX.denseCPU, values: pv.map((p) => b2.clock_driven_dense[p].spn_mean) },
    { name: "Brian2 2.9 (mean of 3)", color: HEX.amber, values: pv.map((p) => b2.clock_driven_brian2[p].spn_mean) },
  ], x: 0.5, y: 1.85, w: 7.1, h: 4.5, title: "Brunel-style E/I network, N = 1,000, Δt = 0.1 ms", xTitle: "Connection probability p (log scale)", yTitle: "Mean spikes per neuron in 1 s (Hz)", ymin: 0, ymax: 120, ystep: 20, ylinear: true, legend: true });
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rng = (a) => `${Math.min(...a).toFixed(1)}–${Math.max(...a).toFixed(1)}`;
  const d = D.derived.brian2_diff_pct;
  const rows = [[H("p"), H("Diff."), H("Mine, 3 runs"), H("Brian2, 3 runs")]].concat(pv.map((p) => [p, `${d[p] > 0 ? "+" : "−"}${Math.abs(d[p]).toFixed(1)}%`, rng(b2.clock_driven_dense[p].spn_all), rng(b2.clock_driven_brian2[p].spn_all)]));
  s.addTable(rows, { x: 7.9, y: 1.95, w: 4.83, colW: [0.7, 0.9, 1.6, 1.63], fontSize: 13, fontFace: "Calibri", color: HEX.ink, rowH: 0.4, border: { type: "solid", color: HEX.grid, pt: 0.75 }, objectName: nm("b2tab") });
  bullets(s, [
    "Brunel-style E/I network, one-pole model, Δt = 0.1 ms (setup on slide 6)",
    "At p = 0.3 the gap is inside run-to-run spread; at p = 0.01 and 0.1 it is not",
    "Brian2 was ~20× faster on this run: 1.2–1.6 s vs 33.4–33.7 s (it propagates only actual spikes; I multiply 21 N×N buckets every step)",
    "After the ring-buffer fix (rerun, same seeds): +1.5%, +2.4%, −2.5% — the gap is not that bug",
  ], 7.9, 3.75, 4.83, 2.7, { fontSize: 13, space: 6 });
  caption(s, "Chart: results.csv at commit eb02eb9 (pre-fix). Machine not logged; rerunning the same seeds on my Mac reproduced every count exactly, so it was almost certainly that machine.", 0.6, 6.4, 7, 0.55);
  addNotes(s, ++n, "Brian2 reference", 1.5, `- Second check: an independent simulator. Brian2 is a widely used Python SNN simulator. I matched its model and parameters to the classic Brunel 2000 network.
- Mean rates agree within 1.6 to 2.7%. At p = 0.3 the difference is inside the spread across seeds; at the lower p values it's small but outside it, so there's still a systematic piece I haven't pinned down.
- Brian2 was about 20 times faster on this setup. My dense backend multiplies 21 bucket matrices every step at dt = 0.1 ms; Brian2 propagates only actual spikes. That's the same primitive story as the rest of the talk.
- After I fixed the ring-buffer offset I reran these exact seeds: rates moved by less than 0.2% and the gap stayed at +1.5, +2.4 and −2.5%. So that bug isn't the explanation.
- If asked about the gap: remaining candidates are exact edge count versus Bernoulli connections, where the inhibitory neurons sit, and floor-versus-round delay quantization.`);
}

// 17 ---------------------------------------------------------------- discretization
{
  const s = content("Validation", "The event-driven gap is most likely discretization, and a Δt sweep would confirm it", SECT.val);
  // timeline diagram
  txt(s, "One synapse, delay d = 3.6 ms", { x: 0.6, y: 1.95, w: 6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  const tx = 0.9, scale = 1.25; // inches per ms
  arrow(s, tx, 3.3, tx + 5 * scale + 0.3, 3.3, { color: "8A94A6" });
  for (let k = 0; k <= 5; k++) { s.addShape(pres.shapes.LINE, { x: tx + k * scale, y: 3.2, w: 0.001, h: 0.2, line: { color: "8A94A6", width: 1 }, objectName: nm("tick") }); txt(s, `${k} ms`, { x: tx + k * scale - 0.3, y: 3.42, w: 0.6, h: 0.3, fontSize: 11, align: "center" }); }
  const mark = (t, y, label, col) => { s.addShape(pres.shapes.OVAL, { x: tx + t * scale - 0.1, y: 3.2, w: 0.2, h: 0.2, fill: { color: col }, line: { type: "none" }, objectName: nm("m") }); if (y < 3) arrow(s, tx + t * scale, y + 0.35, tx + t * scale, 3.15, { color: col, width: 1 }); else arrow(s, tx + t * scale, y, tx + t * scale, 3.45, { color: col, width: 1 }); if (label) txt(s, label, { x: tx + t * scale - 0.85, y: y < 3 ? y : y + 0.02, w: 1.7, h: 0.35, fontSize: 12, bold: true, align: "center", color: col }); };
  mark(3.6, 2.4, "event-driven: 3.6", C.accent5);
  mark(3.0, 3.85, "", C.accent1);
  txt(s, "intended: ⌊3.6⌋ = 3", { x: tx + 3 * scale - 0.85, y: 3.88, w: 1.7, h: 0.35, fontSize: 12, bold: true, align: "center", color: C.accent1 });
  mark(2.0, 3.85, "", C.accent6);
  txt(s, "Apr 30 code: 2", { x: tx + 2 * scale - 0.85, y: 3.88, w: 1.7, h: 0.35, fontSize: 12, bold: true, align: "center", color: C.accent6 });
  txt(s, "Candidate causes found in the code", { x: 0.6, y: 4.45, w: 6.6, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, [
    "Delays floored to whole steps: mean 3.0 → 2.5 ms",
    "Ring-buffer offset (now fixed): the Apr 30 sweep delivered one step earlier still → mean 1.5 ms",
    "Event-driven holds the synaptic current constant over each interval when advancing V — an approximation for the two-pole model",
    "Input: Bernoulli coin flip per step vs a true Poisson process",
  ], 0.6, 4.9, 6.6, 1.9, { fontSize: 13, space: 4 });
  box(s, 7.7, 1.95, 5.03, 4.45, "", { fill: C.background2 });
  txt(s, "How I'd confirm it", { x: 7.95, y: 2.15, w: 4.6, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
  bullets(s, [
    [{ text: "Δt sweep (not run yet): ", options: { bold: true } }, { text: "1, 0.5, 0.2, 0.1 ms at fixed N, p. If it's discretization, the clock-driven rate converges toward event-driven roughly in proportion to Δt" }],
    [{ text: "Unit tests: ", options: { bold: true } }, { text: "one neuron, one input spike, compare V(t) to the closed form; one synapse, check the arrival step" }],
    [{ text: "Done: offset fixed and tested. ", options: { bold: true } }, { text: "At Δt = 0.1 ms it moved rates < 0.2%. Next: rerun the Δt = 1 ms sweep, where the shift is a full step" }],
  ], 7.95, 2.65, 4.6, 3.6, { fontSize: 14, space: 10 });
  caption(s, "Sources: shared/clock_driven.py (bucketize, bucket offsets), backend.cpp loops (read index advanced before read), event_driven/backend.cpp lines 160–172.", 0.6, 6.5, 12);
  addNotes(s, ++n, "Why event-driven differs [CUT IF LONG]", 1, `- Why is event-driven 4 to 7% lower? Event-driven delivers a 3.6 ms delay at 3.6 ms. The clock-driven code floors it to 3 ms, and in the April sweep an indexing offset delivered it at 2. That offset is now fixed.
- Two other differences: event-driven treats the synaptic current as constant over each interval when it updates the voltage, and the clock-driven input is a coin flip per step rather than a true Poisson process.
- I haven't separated these yet. The clean experiment is a dt sweep — discretization error should shrink with dt. The offset fix is done and tested; at 0.1 ms it barely moved rates, which is expected because the shift there is only 0.1 ms.
- If asked "which do you think dominates": at 1 ms the offset was a full step out of a 2.5 ms delay, so it could matter there; I won't know until the 1 ms sweep is rerun.`);
}

// 18 ---------------------------------------------------------------- MAIN FIGURE
{
  const s = content("Results", "Runtime vs connection probability: the winner changes with network size and device", SECT.res);
  logChart(s, { xs: PS, series: sweepSeries(1000), x: 0.5, y: 1.8, w: 6.15, h: 4.3, title: "N = 1,000", xTitle: "Connection probability p (log scale)", yTitle: "Wall-clock runtime (s, log scale)", ymin: 0.01, ymax: 1000, legend: false });
  logChart(s, { xs: PS, series: sweepSeries(10000), x: 6.65, y: 1.8, w: 6.15, h: 4.3, title: "N = 10,000", xTitle: "Connection probability p (log scale)", yTitle: "Wall-clock runtime (s, log scale)", ymin: 0.01, ymax: 1000, legend: false });
  // manual legend
  BACKENDS.forEach((b, i) => {
    const x = 0.9 + i * 2.4;
    s.addShape(pres.shapes.OVAL, { x, y: 6.19, w: 0.16, h: 0.16, fill: { color: BCOLOR[b] }, line: { type: "none" }, objectName: nm("leg") });
    txt(s, b, { x: x + 0.22, y: 6.12, w: 2.1, h: 0.3, fontSize: 13 });
  });
  caption(s, "Δt = 1 ms, 1 s simulated (1,000 steps), delays U(2,4) ms, one run per point. Xeon E5-2696 v4 + RTX 4070 Ti Super. Event-driven at N = 10⁴, p ≥ 0.1 not recorded (300 s limit / run stopped). Source: logs/04-30/results.csv", 0.6, 6.48, 12.1, 0.45);
  addNotes(s, ++n, "MAIN FIGURE", 2.5, `- This is the main result. Runtime against connection probability, both axes log, N = 1,000 left and 10,000 right. One simulated second, 1 ms steps.
- Left: event-driven (red) starts lowest and rises with p — at p = 0.01 it's 45 ms. Dense GPU sits flat at about 0.37 s and takes over from p = 0.1.
- Right: everything moves up. Dense CPU is about 85–94 s regardless of p. Sparse CPU rises from 2 s to 173 s. Sparse GPU is the fastest up to p = 0.5; dense GPU is flat at 1.92 s and wins above that.
- Event-driven at N = 10,000 is already 8 s at p = 0.01 and 67 s at 0.05; larger p weren't recorded.
- Say clearly: one run per point. I'd put confidence intervals on this before publishing anything.
- If asked "why does dense CPU at N = 10⁴ wander between 70 and 94 s": single runs on a shared-memory Xeon — I'd expect run-to-run variance from memory bandwidth contention; repeats would show it.`);
}

// 19 ---------------------------------------------------------------- event-driven scaling with N
{
  const s = content("Results", "Event-driven wins when there's little work: small N, or N = 10³ with p ≤ 0.05", SECT.res);
  const p = 0.05;
  const series = BACKENDS.map((b) => ({ name: b, color: BCOLOR[b], values: NS.map((N) => { const r = get(b, N, p); return r ? r.t : null; }) }));
  logChart(s, { xs: NS, series, x: 0.5, y: 1.85, w: 7.3, h: 4.6, title: "p = 0.05", xTitle: "Number of neurons N (log scale)", yTitle: "Wall-clock runtime (s, log scale)", ymin: 0.00001, ymax: 1000, legend: true });
  bullets(s, [
    [{ text: "N ≤ 100: ", options: { bold: true } }, { text: "event-driven is 8.7× to 1,600× faster than the best clock-driven backend; clock-driven is paying a per-step floor" }],
    [{ text: "N = 10³: ", options: { bold: true } }, { text: "wins at p ≤ 0.05 (0.23 s vs 0.37 s dense GPU); loses from p = 0.1" }],
    [{ text: "N = 10⁴: ", options: { bold: true } }, { text: "never wins. At p = 0.01: 8.0 s vs 0.62 s sparse GPU and 2.1 s sparse CPU" }],
    [{ text: "Why: ", options: { bold: true } }, { text: "deliveries grow as N²p·r while the GPU's per-step cost stays near its floor" }],
  ], 8.1, 1.95, 4.65, 4.5, { fontSize: 15, space: 10 });
  caption(s, "Δt = 1 ms, 1 s simulated, one run per point. Dense CPU at N = 100 (3.6 s) is slower than at N = 10³ (1.0 s) — unexplained; likely threading overhead on tiny matrices. Source: logs/04-30/results.csv", 0.6, 6.5, 12.1, 0.45);
  addNotes(s, ++n, "Event-driven crossover", 1.5, `- Same data sliced the other way: runtime versus N at p = 0.05.
- At small N, event-driven is two to three orders of magnitude ahead. Clock-driven backends sit on a floor — a fixed cost per step for launching ops — no matter how little there is to do.
- At N = 1,000 it still wins at low p. At N = 10,000 it never wins: the number of spike deliveries grows like N squared times p times the firing rate, and each costs 120–300 ns on one core, while the GPU stays near its floor.
- There's an oddity: dense CPU is slower at N = 100 than at 1,000. I haven't investigated; my guess is thread-pool overhead on tiny matrices. I'd rather show it than hide it.
- If asked "would a parallel event-driven backend change this": yes, potentially — see the backup slide on lookahead parallelism.`);
}

// 20 ---------------------------------------------------------------- dense GPU
{
  const s = content("Results", "Dense GPU is launch-bound below N = 10⁴ and bandwidth-bound at N = 10⁴", SECT.res);
  const series = NS.map((N, i) => ({ name: `N = ${N.toLocaleString("en-US")}`, color: [HEX.sparseGPU, HEX.sparseCPU, HEX.denseGPU, HEX.denseCPU][i], values: PS.map((p) => get("Dense GPU", N, p).t) }));
  logChart(s, { xs: PS, series, x: 0.5, y: 1.85, w: 6.6, h: 4.6, title: "Dense GPU runtime", xTitle: "Connection probability p (log scale)", yTitle: "Wall-clock runtime (s, log scale)", ymin: 0.1, ymax: 10, legend: true });
  box(s, 7.4, 1.95, 5.33, 2.15, "", { fill: C.background2 });
  txt(s, "Floor: 0.36–0.40 s for N ≤ 10³, any p", { x: 7.6, y: 2.05, w: 5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, ["≈ 0.37 ms per step ÷ ~24 ops ≈ 15 µs per op", "Same time for 10 neurons as for 1,000 → per-op overhead (launch + dispatch), not arithmetic", "Fix: fuse ops, or capture the step as a CUDA Graph"], 7.6, 2.5, 4.95, 1.6, { fontSize: 13, space: 4 });
  box(s, 7.4, 4.25, 5.33, 2.15, "", { fill: C.background2 });
  txt(s, "N = 10⁴: 1.92 s at every p", { x: 7.6, y: 4.35, w: 5, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  bullets(s, ["Weights: 3 × 10⁸ × 4 B = 1.2 GB read per step", "1.2 TB in 1.92 s ≈ 624 GB/s — close to the card's ~672 GB/s spec-sheet bandwidth", "One bucket is always empty → ~⅓ of that traffic is avoidable (prediction, not measured)"], 7.6, 4.8, 4.95, 1.6, { fontSize: 13, space: 4 });
  caption(s, "Δt = 1 ms, 1,000 steps, one run per point. 624 GB/s is derived (bytes ÷ runtime); 672 GB/s is the vendor spec, not measured. Op count from backend.cpp. Source: logs/04-30/results.csv", 0.6, 6.5, 12.1, 0.45);
  addNotes(s, ++n, "Dense GPU", 2, `- Dense GPU is the most instructive curve because it's flat — p doesn't matter to a dense mat-vec.
- For N up to 1,000 it takes 0.36 to 0.40 seconds regardless of size: about 0.37 ms per step. With about 24 ops per step, that's roughly 15 microseconds each. The GPU is mostly waiting on launches and dispatch, not computing.
- At N = 10,000 it's 1.92 s. Each step reads 1.2 GB of weights; divide and you get about 624 GB/s, close to the card's published ~672 GB/s. So at that size it's memory-bandwidth bound — the right kind of bound for a GEMV.
- Two cheap wins: CUDA Graphs or fusion for the floor, and dropping the always-empty bucket for a third of the bandwidth. Neither is measured yet.
- If asked "is 15 µs per launch realistic": raw launch latency is a few microseconds; libtorch dispatch, the caching allocator and small temporaries add the rest. I'd confirm with Nsight Systems rather than assert it.`);
}

// 21 ---------------------------------------------------------------- sparse GPU
{
  const s = content("Results", "At N = 10⁴, sparse GPU is fastest up to p = 0.5 — 142× faster than dense CPU at p = 0.01", SECT.res);
  logChart(s, { xs: PS, series: sweepSeries(10000, ["Dense CPU", "Sparse CPU", "Dense GPU", "Sparse GPU"]), x: 0.5, y: 1.85, w: 6.6, h: 4.6, title: "N = 10,000", xTitle: "Connection probability p (log scale)", yTitle: "Wall-clock runtime (s, log scale)", ymin: 0.1, ymax: 1000, legend: true });
  stat(s, "142×", "sparse GPU (0.62 s) vs dense CPU (88.7 s) at p = 0.01; 3.4× vs sparse CPU (2.1 s)", 7.4, 1.95, 5.33, C.accent4);
  bullets(s, [
    [{ text: "Byte model: ", options: { bold: true } }, { text: "dense reads 12·N² B per step, CSR ≈ 12·N²p B → equal only at p = 1" }],
    [{ text: "Measured GPU crossover: ", options: { bold: true } }, { text: "between p = 0.5 (1.50 vs 1.92 s) and 0.75 (1.97 vs 1.92 s). Irregular CSR access and a higher floor (0.6 s) move it left" }],
    [{ text: "CPU crossover: ", options: { bold: true } }, { text: "sparse wins to p = 0.4 (75.7 vs 92.0 s); dense from p = 0.5 (87.4 vs 98.4 s)" }],
    [{ text: "Hypothesis for the 0.6 s floor: ", options: { bold: true } }, { text: "a device sync per bucket per step (0-dim tensor index) — to check with Nsight Systems" }],
  ], 7.4, 3.6, 5.33, 2.9, { fontSize: 13, space: 7 });
  caption(s, "Δt = 1 ms, 1,000 steps, one run per point. Byte counts analytic (float32 values, int64 indices). Source: logs/04-30/results.csv", 0.6, 6.5, 12.1, 0.4);
  addNotes(s, ++n, "Sparse GPU [CUT IF LONG]", 1.5, `- At N = 10,000 the sparse GPU backend is the fastest thing I built: 0.62 seconds at p = 0.01, 142 times faster than dense CPU and 3.4 times faster than sparse CPU.
- The byte model says dense and CSR read the same number of bytes only at p = 1, so naively sparse should win everywhere. It loses above p ≈ 0.6 because CSR access is irregular and the sparse loop has a higher fixed floor.
- On the CPU the crossover is lower, between 0.4 and 0.5.
- My hypothesis for the higher sparse floor is an implicit device sync from indexing with a device tensor. That's testable in an afternoon with Nsight.
- If asked "why int64 indices": that's PyTorch's default for CSR. Switching to int32 would cut index bytes in half; another cheap experiment.`);
}

// 22 ---------------------------------------------------------------- what didn't work
{
  const s = content("Lessons", "What didn't work: torch.compile, and an OpenMP backend that was never actually parallel", SECT.lessons);
  const tc = D.torch_compile;
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  box(s, 0.6, 1.95, 5.95, 4.45, "", { fill: C.background2 });
  txt(s, "torch.compile (Apr 16)", { x: 0.85, y: 2.1, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    `First call: ${Math.round(Math.min(tc["clock_driven_dense_cpu_compiled|num_neurons=10"][0], tc["clock_driven_dense_cpu_compiled|num_neurons=10000"][0]) / 60)}–${Math.round(tc["clock_driven_dense_cpu_compiled|num_neurons=10000"][0] / 60)} min to compile on CPU, per N`,
    `Warm runs, N = 10: ${mean(tc["clock_driven_dense_cpu_compiled|num_neurons=10"].slice(1)).toFixed(3)} s vs ${mean(tc["clock_driven_dense_cpu|num_neurons=10"]).toFixed(2)} s eager — fusion removes per-op overhead`,
    `Warm runs, N = 10⁴: ${mean(tc["clock_driven_dense_cpu_compiled|num_neurons=10000"].slice(1)).toFixed(0)} s vs ${mean(tc["clock_driven_dense_cpu|num_neurons=10000"]).toFixed(1)} s eager — 2× slower where it matters`,
    "GPU: CUDA Graphs skipped (in-place zero_ on a dynamically indexed slice); N = 100 crashed in Inductor (RecursionError)",
    "Decision: drop it; move the loop to C++ instead",
  ], 0.85, 2.6, 5.5, 3.4, { fontSize: 13, space: 6 });
  caption(s, "Colab, earlier Python-loop version, p = 1, 5 runs per N. Source: logs/04-16/output.txt, logs/04-16/README.md", 0.85, 5.85, 5.5, 0.5);
  box(s, 6.78, 1.95, 5.95, 4.45, "", { fill: C.background2 });
  txt(s, "OpenMP backend (May 1, deleted May 5)", { x: 7.03, y: 2.1, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    "Push model: loop over spiking neurons, scatter into the ring buffer with #pragma omp atomic",
    "N = 10³, p = 1: 70.7 s — same as event-driven on that run (68.5 s); the matmul backends took ~1 s on Apr 30 (different run)",
    "Root cause 1: setup.py never passed -fopenmp, so the pragmas were ignored — it ran serially",
    "Root cause 2: per-synapse libtorch indexing (ring_buffer[a][b] += w) and .item() per neuron — microseconds each",
    "Lesson: check the build flags and profile before parallelizing. A corrected version is planned",
  ], 7.03, 2.6, 5.5, 3.4, { fontSize: 13, space: 6 });
  caption(s, "Sources: git show 084dd31 (backend.cpp, setup.py, results.csv); commit 4494bfc deletes it. Machine for that run not recorded.", 7.03, 5.85, 5.5, 0.5);
  addNotes(s, ++n, "What didn't work", 3, `- Two things that didn't work, and I think the reasons matter more than the results.
- torch.compile: compiling took 24 to 28 minutes per network size. It helped tiny networks — 35 times faster at N = 10 because fusion removes per-op overhead — but at N = 10,000 it was twice as slow as eager. On the GPU it couldn't use CUDA Graphs because I mutate a dynamically indexed slice of the ring buffer, and at N = 100 the compiler crashed. I dropped it and moved the loop to C++.
- OpenMP: I wrote a push-style backend that scatters each spike's targets into the ring buffer with atomics. It was as slow as the single-threaded event-driven code, so I deleted it. Re-reading it for this talk: setup.py never passed -fopenmp, so it was never parallel. And the inner loop did a libtorch tensor index per synapse — microseconds each. Adding the flag alone wouldn't fix that; it needs raw pointers.
- Lesson I took: verify the build actually did what you think, and profile before parallelizing.
- If asked "how would you do it now": see backup — pull model over target neurons, no atomics.`);
}

// 23 ---------------------------------------------------------------- limitations & next steps
{
  const s = content("Lessons", "What I measured, what I didn't, and what comes next", SECT.lessons);
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const g = (t) => ({ text: t, options: { bold: true, color: HEX.sparseCPU } });
  const a = (t) => ({ text: t, options: { bold: true, color: HEX.amber } });
  const r = (t) => ({ text: t, options: { bold: true, color: HEX.event } });
  const rows = [
    [H("Quantity"), H("Status"), H("Detail")],
    ["Wall-clock runtime", g("Measured"), "run() only, CUDA-synchronized; 1 run per point (sweep), 3 (Brian2)"],
    ["Memory", a("Analytic"), "weights 4·B·N² B dense, ≈12·N²p B CSR; never measured"],
    ["Energy", r("Planned"), "RAPL / NVML; only a MacBook prototype in March, no results on this hardware"],
    ["Unit tests", a("Started"), "one single-spike delay test (ring-buffer fix); no suite yet"],
    ["SpiNNaker", r("Not run"), "runner written; virtual board only, never on hardware"],
    ["Known bugs", a("Partly fixed"), "ring-buffer offset fixed + tested; empty bucket and stale sparse/event runners remain"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 7.6, colW: [1.75, 1.55, 4.3], fontSize: 13, fontFace: "Calibri", color: HEX.ink, rowH: 0.55, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("limits") });
  txt(s, "Next steps", { x: 8.6, y: 1.95, w: 4.1, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    "Round delays to the nearest step (fills the empty bucket); grow tests into a suite",
    "Repeat runs with confidence intervals; record timeouts as censored",
    "Δt sweep to settle the event-driven gap",
    "Memory and energy: peak allocation, RAPL, NVML",
    "CUDA Graphs for the launch floor",
    "Calendar queue: delays are bounded (2–4 ms), so O(1) bucketed scheduling",
    "Correct OpenMP: pull model, -fopenmp, raw pointers",
  ], 8.6, 2.45, 4.13, 4.0, { fontSize: 14, space: 5 });
  addNotes(s, ++n, "Limitations and next steps", 2, `- I want to be precise about scope. What I measured is wall-clock runtime of the simulation loop. Memory I can compute exactly but didn't measure. Energy was planned — I prototyped it on a MacBook early on, but I have no energy numbers from this hardware.
- Line to use if someone points at the résumé: "My résumé lists latency, throughput, memory and energy as the project's axes. What I've measured so far is runtime — throughput falls out of that as events per second. Memory is analytic, and energy instrumentation is the next step."
- No test suite yet, which is how an indexing bug survived. I've fixed the ring-buffer offset this week with a single-spike test; the empty bucket and the stale sparse and event runners are next.
- SpiNNaker: the runner exists but has only run against a virtual board, never real hardware.
- Next steps in priority order: tests and the bug fixes, repeats with confidence intervals, the dt sweep, then memory and energy, then performance work — CUDA Graphs, a calendar queue, a correct parallel CPU backend.`);
}

// 24 ---------------------------------------------------------------- takeaways for array processing
{
  const s = content("Close", "Takeaways that carry over to array processing", SECT.close);
  const rows = [
    ["Ring buffer of future inputs", "Circular buffers of sensor blocks; delay lines"],
    ["Delays quantized to Δt (and an off-by-one)", "Integer-sample vs fractional delays in time-domain beamforming — test delay paths explicitly"],
    ["Per-step overhead dominated small problems", "Block processing: batch enough work per launch, keep data on the device"],
    ["Pick the primitive by data density", "Dense GEMV vs sparse vs event list ↔ full-band processing vs sparse subarrays vs detection-driven processing"],
    ["Validate against a reference before trusting a speedup", "Brian2 here; a MATLAB reference implementation in a sonar chain"],
  ];
  rows.forEach(([a, b], i) => {
    const y = 1.95 + i * 0.86;
    box(s, 0.6, y, 0.6, 0.6, String(i + 1), { fill: C.accent3, color: C.background1, bold: true, fontSize: 18 });
    txt(s, a, { x: 1.4, y: y + 0.1, w: 4.6, h: 0.6, fontSize: 16, bold: true, color: C.text2 });
    arrow(s, 6.05, y + 0.3, 6.6, y + 0.3, { color: "8A94A6" });
    txt(s, b, { x: 6.75, y: y + 0.1, w: 6.0, h: 0.7, fontSize: 15 });
  });
  txt(s, "Measure, model, validate — and say what you didn't measure.", { x: 0.6, y: 6.35, w: 12.1, h: 0.45, fontSize: 18, bold: true, italic: true, color: C.accent3 });
  addNotes(s, ++n, "Takeaways", 2, `- Five things I'd bring to array-processing work.
- A ring buffer of future contributions is the same structure as a circular buffer of sensor blocks or a delay line.
- Quantizing delays to the sample grid is the integer-versus-fractional-delay problem in time-domain beamforming — and my off-by-one is the kind of bug that only a test on a known delay catches.
- Small problems drowned in per-step overhead: batch work into blocks big enough to amortize launches, and keep data on the device.
- Match the primitive to the data's density.
- And validate against a reference — Brian2 here, a MATLAB reference in a sonar chain — before trusting any speedup.
- Close: measure, model, validate, and be explicit about what you didn't measure. Thank you.`);
}

// 25 ---------------------------------------------------------------- closing
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: SECT.close });
  s.addText("Thank you — questions?", { placeholder: "title" });
  s.addText([
    { text: "142×  ", options: { bold: true, color: HEX.sparseGPU } }, { text: "sparse GPU vs dense CPU at N = 10⁴, p = 0.01", options: { breakLine: true } },
    { text: "≤ 2.7%  ", options: { bold: true, color: HEX.sparseGPU } }, { text: "mean-rate difference vs Brian2", options: { breakLine: true } },
    { text: "~15 µs  ", options: { bold: true, color: HEX.sparseGPU } }, { text: "per-op floor on the GPU (derived)", options: {} },
  ], { placeholder: "body" });
  addNotes(s, ++n, "Questions", 0, `- Leave this up during Q&A. Backup slides follow: memory math, ring-buffer sizing, full runtime tables, OpenMP redesign, torch.compile numbers, Brian2 parameter matching, parallel event-driven, floating-point reproducibility.`);
}

// ================================================================= BACKUP
let bnum = 0;
function backup(title) { const s = content(`Backup ${++bnum}`, title, SECT.backup); return s; }

// B1 memory math
{
  const s = backup("Memory math: dense weights grow as B·N², so Δt drives memory as much as N does");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const gb = (b) => (b >= 1e9 ? (b / 1e9).toFixed(1) + " GB" : (b / 1e6).toFixed(0) + " MB");
  const rows = [[H("Structure"), H("Formula"), H("N = 10³"), H("N = 10⁴")],
    ["Dense weights, Δt = 1 ms (B = 3)", "4 · B · N²", gb(4 * 3 * 1e6), gb(4 * 3 * 1e8)],
    ["Dense weights, Δt = 0.1 ms (B = 21)", "4 · B · N²", gb(4 * 21 * 1e6), gb(4 * 21 * 1e8)],
    ["CSR, p = 0.01", "12 · N²p + 8 · B(N+1)", "0.1 MB", gb(12 * 1e6 + 8 * 3 * 10001)],
    ["CSR, p = 1", "12 · N²p + 8 · B(N+1)", gb(12 * 1e6), gb(12 * 1e8)],
    ["Event-driven edges (int32 index, float w, float d)", "12 · N²p", "—", "1.2 GB at p = 1"],
    ["Ring buffer, Δt = 1 ms", "(⌊max d/Δt⌋ + 1) · N · 4", "20 KB", "200 KB"],
    ["Graph construction temporaries", "N² int64 randperm + N² masks", "8 MB +", "800 MB +"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 12.13, colW: [4.6, 3.4, 2.0, 2.13], fontSize: 14, fontFace: "Calibri", color: HEX.ink, rowH: 0.5, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("mem") });
  caption(s, "All analytic from the code (shared/clock_driven.py, shared/graph_creation.py); no memory was measured. B = ⌊max d/Δt⌋ + 1 − ⌊min d/Δt⌋.", 0.6, 6.2, 12.1);
  addNotes(s, ++n, "Backup: memory", 0, `- Dense memory is 4·B·N², and B grows as dt shrinks: 3 buckets at 1 ms, 21 at 0.1 ms. At N = 10,000 and 0.1 ms that's 8.4 GB just for weights.
- Construction is worse than steady state: building the graph allocates N² int64 permutations and masks.
- None of this was measured — peak-allocation measurement is on the next-steps list.`);
}

// B2 ring buffer sizing + indexing
{
  const s = backup("Ring buffer and bucket sizing, and the one-step offset (now fixed)");
  bullets(s, [
    "buffer_size = ⌊max_delay / Δt⌋ + 1 = 5 at Δt = 1 ms; 41 at Δt = 0.1 ms",
    "Bucket offsets k = ⌊min_delay/Δt⌋ … buffer_size − 1 = {2, 3, 4} at Δt = 1 ms",
    "bucketize(right=True) − 1 maps d ∈ [2,3) → 0, [3,4) → 1, [4,5) → 2; with d ~ U(2,4) bucket 2 is empty",
    "Apr 30 code: step t read slot (t + 1) mod 5 (index advanced before the read) but wrote slots (t + k) mod 5",
    "So a write with offset k was read at step t + k − 1: delays of 1–2 ms instead of 2–3 ms",
    "Fixed: buffer_index = t % buffer_size (dense and sparse). Single-spike test: arrival at exactly k steps at Δt = 1 and 0.1 ms",
    "The deleted OpenMP backend wrote to (read index + d), so its delays were correct",
  ], 0.6, 1.95, 7.2, 4.6, { fontSize: 15, space: 8 });
  code(s, "// before the fix (B = buffer_size)\nfor (int t = 0; t < T; t++) {\n  idx = (idx + 1) % B;    // reads t + 1\n  I.add_(ring[idx]);\n  ...\n  ring.index_add_(0,\n      slots[t],           // writes t + k\n      W.matmul(s));\n  ring[idx].zero_();\n}\n\n// after the fix\nint idx = t % B;          // reads t", 8.1, 1.95, 4.63, 4.1, 12);
  caption(s, "Sources: shared/clock_driven.py (create_ring_buffer, create_lookup_tensors, _compute_delay_buckets); git show eb02eb9:simulations/clock_driven_dense/backend.cpp lines 94–127 (before the fix).", 0.6, 6.45, 12);
  addNotes(s, ++n, "Backup: ring buffer", 0, `- Walk through the indexing if asked. The read index advances before the read, the write uses t plus offset, so arrival is one step early.
- I found this re-reading the code for the talk and fixed it: the read slot is now t mod B. The single-spike test failed before (arrival at k − 1) and passes after.`);
}

// B3 full runtime table
{
  const s = backup("Full runtime table, N = 10³ and 10⁴ (seconds, one run each)");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy }, align: "center" } });
  const head = [H("N"), H("Backend")].concat(PS.map((p) => H(String(p))));
  const rows = [head];
  [1000, 10000].forEach((N) => BACKENDS.forEach((b, i) => {
    rows.push([i === 0 ? { text: N.toLocaleString("en-US"), options: { bold: true, rowspan: 5, valign: "middle" } } : null, { text: b, options: { bold: true, color: BCOLOR[b] } }].filter((x) => x !== null).concat(PS.map((p) => { const r = get(b, N, p); const w = D.derived.winner[wkey(N, p)].winner === b; return { text: r ? fmt(r.t) : "—", options: { align: "center", bold: w, fill: w ? { color: "E3F1EC" } : undefined } }; })));
  }));
  s.addTable(rows, { x: 0.6, y: 1.9, w: 12.13, colW: [0.8, 1.9].concat(Array(9).fill(1.048)), fontSize: 11, fontFace: "Calibri", color: HEX.ink, rowH: 0.34, margin: 0.03, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("rt") });
  caption(s, "Bold/tinted = fastest in that column. Δt = 1 ms, 1 s simulated. Dense CPU uses the first dense-CPU block only; a second block labelled dense CPU (file lines 38–67) is excluded — it scales like event-driven and is unexplained. The README plot averages both blocks. Source: logs/04-30/results.csv", 0.6, 6.25, 12.1, 0.7);
  addNotes(s, ++n, "Backup: runtime table", 0, `- Every number behind the main figure. Fastest per column is tinted.
- Be ready for: why is there a dash — event-driven at N = 10,000, p ≥ 0.1 wasn't recorded.
- Data-hygiene note if asked: the CSV has a second block labelled dense CPU that can't be dense CPU; I excluded it. The README plot averages it in, so that plot's dense-CPU curve is wrong.`);
}

// B4 full table small N
{
  const s = backup("Full runtime table, N = 10 and 100 (seconds, one run each)");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy }, align: "center" } });
  const rows = [[H("N"), H("Backend")].concat(PS.map((p) => H(String(p))))];
  [10, 100].forEach((N) => BACKENDS.forEach((b, i) => {
    rows.push([i === 0 ? { text: String(N), options: { bold: true, rowspan: 5, valign: "middle" } } : null, { text: b, options: { bold: true, color: BCOLOR[b] } }].filter((x) => x !== null).concat(PS.map((p) => { const r = get(b, N, p); return { text: r ? fmt(r.t, 3) : "—", options: { align: "center" } }; })));
  }));
  s.addTable(rows, { x: 0.6, y: 1.9, w: 12.13, colW: [0.8, 1.9].concat(Array(9).fill(1.048)), fontSize: 11, fontFace: "Calibri", color: HEX.ink, rowH: 0.34, margin: 0.03, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("rt2") });
  caption(s, "At this size every clock-driven backend sits on its fixed per-step floor. Event-driven N = 10, p = 0.01 missing from the log. Source: logs/04-30/results.csv", 0.6, 6.25, 12.1, 0.5);
  addNotes(s, ++n, "Backup: small-N table", 0, `- Small networks: clock-driven runtimes are floors — 0.1 to 4 s — while event-driven is microseconds to milliseconds.`);
}

// B5 OpenMP code + better design
{
  const s = backup("The OpenMP backend as written, and the design I'd use now");
  code(s, "// setup.py: extra_compile_args = [\"-O3\", \"-march=native\"]\n//   (no -fopenmp: the pragmas below are ignored)\n\ntorch::set_num_threads(1);\n#pragma omp parallel for schedule(static)\nfor (int src = 0; src < num_neurons; src++) {\n  if (!spikes_bool[src].item<bool>()) continue;\n  for (int j = crow[src]; j < crow[src+1]; j++) {\n    int slot = (buffer_index + d[j]) % buffer_size;\n#pragma omp atomic\n    ring_buffer[slot][col[j]] += w[j];  // Tensor op\n  }\n}", 0.6, 1.95, 6.6, 4.3, 12);
  txt(s, "Better design", { x: 7.5, y: 1.95, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  bullets(s, [
    "Build with -fopenmp in both compile and link args; check omp_get_max_threads() at startup",
    "Raw pointers (data_ptr<float>()) — no per-element tensor dispatch or .item()",
    "Pull model: parallelize over target neurons using the transposed (CSC) graph; each thread owns its targets, so no atomics",
    "Or keep push, with per-thread private ring-buffer slices and one reduction per step",
    "Compact the spike list first (2–5% of neurons spike per step here)",
    "Pin threads, first-touch allocate per thread",
  ], 7.5, 2.45, 5.23, 4.0, { fontSize: 14, space: 7 });
  caption(s, "Source: git show 084dd31:simulations/clock_driven_openmp/backend.cpp (lines 158–193) and 084dd31:setup.py", 0.6, 6.45, 12);
  addNotes(s, ++n, "Backup: OpenMP", 0, `- As written, without -fopenmp the pragmas are ignored. Even with the flag, omp atomic on a Tensor expression isn't a scalar update, and every inner iteration goes through libtorch indexing.
- Redesign: raw pointers, pull model over targets with a transposed graph so no atomics, or per-thread buffers with a reduction.`);
}

// B6 torch.compile numbers
{
  const s = backup("torch.compile numbers (Colab, Python-loop version, p = 1, 5 runs per N)");
  const tc = D.torch_compile;
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rows = [[H("N"), H("Eager CPU, mean of 5 (s)"), H("Compiled CPU, first call (s)"), H("Compiled CPU, warm mean of 4 (s)"), H("Compiled / eager")]];
  [10, 100, 1000, 10000].forEach((N) => {
    const e = mean(tc[`clock_driven_dense_cpu|num_neurons=${N}`]);
    const c = tc[`clock_driven_dense_cpu_compiled|num_neurons=${N}`];
    const w = mean(c.slice(1));
    rows.push([N.toLocaleString("en-US"), fmt(e, 3), c[0].toFixed(0), fmt(w, 3), (w / e).toFixed(2) + "×"]);
  });
  s.addTable(rows, { x: 0.6, y: 1.95, w: 12.13, colW: [1.4, 2.6, 2.8, 3.0, 2.33], fontSize: 14, fontFace: "Calibri", color: HEX.ink, rowH: 0.5, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("tc") });
  bullets(s, [
    `GPU compiled: first call at N = 10 took ${tc["clock_driven_dense_gpu_compiled|num_neurons=10"][0].toFixed(0)} s; warm ≈ 0.05 s vs ≈ 0.6 s eager GPU`,
    "N = 100 on GPU: Inductor RecursionError in reorder_for_peak_memory after 1,111 s",
    "CUDA Graphs skipped: \"mutated inputs\" from ring_buffer[buffer_index].zero_()",
  ], 0.6, 4.75, 12.1, 1.4, { fontSize: 14 });
  caption(s, "Source: logs/04-16/output.txt (first of each 5 runs is the compile call), logs/04-16/README.md. Colab hardware, not the Xeon/4070 machine.", 0.6, 6.3, 12);
  addNotes(s, ++n, "Backup: torch.compile", 0, `- Compile helps tiny problems, hurts big ones, and the compile itself costs tens of minutes. On GPU it couldn't use CUDA Graphs because of the in-place zero of a dynamically indexed slice.`);
}

// B7 Brian2 parameter matching
{
  const s = backup("Brian2 parameter matching (Brunel-2000-style network)");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rows = [[H("Parameter"), H("My dense backend"), H("Brian2 runner")],
    ["Membrane τ, threshold, reset", "10 ms, 10 mV, 0 mV", "same; method = \"exact\""],
    ["Refractory", "2 ms (V held)", "2 ms (unless refractory)"],
    ["Excitatory weight J", "0.1 mV voltage jump", "on_pre: v += J"],
    ["Inhibitory weight", "−5 J; 20% of neurons, random positions", "−5 J; last 20% of neurons"],
    ["Connectivity", "exactly ⌊N²p⌋ edges", "connect(p = p): Bernoulli per pair"],
    ["Delays", "U(2, 4) ms, quantized to Δt", "U(2, 4) ms, rounded to Δt by Brian2"],
    ["External input", "Poisson count per step, mean = rate·Δt·p·N_E", "PoissonInput: p·N_E sources at rate"],
    ["Input rate", "2 × θ / (J · p·N_E · τ)", "same"],
    ["Δt, T, repeats", "0.1 ms, 1 s, 3", "0.1 ms, 1 s, 3"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 12.13, colW: [3.2, 4.6, 4.33], fontSize: 13, fontFace: "Calibri", color: HEX.ink, rowH: 0.42, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("b2p") });
  caption(s, "Sources: simulations/brian2/runner.py, simulations/clock_driven_dense/backend.cpp, shared/simulation_config.py (poisson_rate, poisson_prob), simulations/run_all.py at HEAD.", 0.6, 6.3, 12);
  addNotes(s, ++n, "Backup: Brian2 matching", 0, `- Differences that remain: exact versus Bernoulli edge counts, inhibitory neuron placement, and how each rounds delays to the grid. Any of these could explain a 2% rate difference.`);
}

// B8 parallel event-driven
{
  const s = backup("How I'd parallelize the event-driven backend");
  bullets(s, [
    [{ text: "Minimum-delay lookahead: ", options: { bold: true } }, { text: "no spike can affect another neuron sooner than d_min = 2 ms, so all events in [t, t + d_min) are independent across neurons" }],
    [{ text: "Partition neurons ", options: { bold: true } }, { text: "across threads or MPI ranks; each owns a local queue and processes its window without locks" }],
    [{ text: "Exchange spikes at window boundaries: ", options: { bold: true } }, { text: "MPI_Alltoallv of (source, time) pairs; receivers expand them into local events (the approach NEST takes)" }],
    [{ text: "Calendar queue / time wheel: ", options: { bold: true } }, { text: "delays are bounded, so bucket events by time slot for O(1) insert instead of O(log Q)" }],
    [{ text: "Determinism: ", options: { bold: true } }, { text: "break time ties by (time, neuron id) so results don't depend on thread count" }],
  ], 0.6, 1.95, 12.1, 4.5, { fontSize: 16, space: 12 });
  addNotes(s, ++n, "Backup: parallel event-driven", 0, `- The classic answer is conservative synchronization with lookahead equal to the minimum delay. With 2 ms minimum delay and ~120–300 ns per event, each window has plenty of work per thread at N = 10⁴.`);
}

// B9 FP reproducibility
{
  const s = backup("Why results aren't bit-reproducible across thread counts or devices");
  bullets(s, [
    "Float addition isn't associative: (a + b) + c ≠ a + (b + c) in float32",
    "Multi-threaded GEMV, SpMV and index_add_ split the sums differently for different thread counts; CUDA index_add_ uses atomics, so the order varies run to run",
    "A threshold turns a 1-ulp difference into spike / no spike, and the network amplifies it — trajectories diverge within milliseconds",
    "Even the graph differs by device here: edges come from a CPU randperm, but delays are drawn with the device's RNG",
    "So compare statistics (rates, distributions), not spike trains; for regression tests, fix threads = 1 and use deterministic algorithms",
  ], 0.6, 1.95, 12.1, 4.5, { fontSize: 16, space: 12 });
  caption(s, "Sources: shared/graph_creation.py (torch.randperm without device), shared/utils.py (create_delays on sim device).", 0.6, 6.3, 12);
  addNotes(s, ++n, "Backup: FP reproducibility", 0, `- Chaotic dynamics plus non-associative float sums means bitwise reproducibility is the wrong target. Statistics are.`);
}

// B10 model timeline
{
  const s = backup("Which code produced which result");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy } } });
  const rows = [[H("Date"), H("Commit"), H("What"), H("Machine")],
    ["Apr 5 – Apr 11", "logs/04-05, 04-11", "first agreement check; first sweep (Python loops)", "Colab"],
    ["Apr 16", "logs/04-16", "torch.compile vs eager", "Colab"],
    ["Apr 30", "63ab99a", "main sweep, all 5 C++ backends, two-pole model", "Vast.ai: Xeon E5-2696 v4 + RTX 4070 Ti Super"],
    ["May 1–5", "084dd31 → 4494bfc", "OpenMP backend added, measured, deleted", "not recorded"],
    ["May 8", "1058ec7", "Brian2 docs examples: constant vs uniform delay", "not recorded"],
    ["May 21", "eb02eb9", "dense moved to one-pole Brunel model; Brian2 comparison", "my Mac (reproduced exactly)"],
    ["Nov 17", "ring-buffer fix", "read slot = t mod B; single-spike test; Brunel rerun", "my Mac"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.95, w: 12.13, colW: [1.7, 2.4, 5.0, 3.03], fontSize: 14, fontFace: "Calibri", color: HEX.ink, rowH: 0.5, border: { type: "solid", color: HEX.grid, pt: 0.75 }, valign: "middle", objectName: nm("tl") });
  caption(s, "At HEAD the sparse and event-driven runners still reference config fields removed in eb02eb9 (resistance, poisson_weight, synaptic_decay, synaptic_time_constant), so only dense and Brian2 run as committed.", 0.6, 6.2, 12.1, 0.6);
  addNotes(s, ++n, "Backup: provenance", 0, `- If someone asks "could I rerun this": the Apr 30 sweep corresponds to commit 63ab99a, which still has the ring-buffer offset; HEAD has moved the dense backend to the Brunel model and the other runners need updating.`);
}

// ================================================================= write + post-process
(async () => {
  await pres.writeFile({ fileName: OUT });
  // post-process: log-scale X axis on scatter charts; drop empty points so missing runs render as gaps
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  const charts = Object.keys(zip.files).filter((f) => /^ppt\/charts\/chart\d+\.xml$/.test(f));
  for (const f of charts) {
    let x = await zip.file(f).async("string");
    if (x.includes("<c:barChart>")) {
      x = x.replace(/<c:multiLvlStrRef>\s*(<c:f>[^<]*<\/c:f>)\s*<c:multiLvlStrCache>\s*(<c:ptCount val="\d+"\/>)<c:lvl>([\s\S]*?)<\/c:lvl>\s*<\/c:multiLvlStrCache>\s*<\/c:multiLvlStrRef>/g, "<c:strRef>$1<c:strCache>$2$3</c:strCache></c:strRef>");
      x = x.replace(/<c:noMultiLvlLbl val="[01]"\/>/g, "");
      // pptxgenjs emits a third <c:axId> (a series axis that is never declared); barChart allows exactly two
      x = x.replace(/(<c:axId val="\d+"\/>\s*<c:axId val="\d+"\/>)\s*<c:axId val="\d+"\/>(\s*<\/c:barChart>)/g, "$1$2");
      zip.file(f, x);
      continue;
    }
    if (!x.includes("<c:scatterChart>")) continue;
    const xs = [...x.matchAll(/<c:xVal>[\s\S]*?<\/c:xVal>/g)][0][0];
    const vals = [...xs.matchAll(/<c:v>([^<]+)<\/c:v>/g)].map((m) => +m[1]);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...vals))));
    x = x.replace(/<c:valAx>([\s\S]*?)<\/c:valAx>/g, (blk) => {
      if (!/<c:axPos val="b"\/>/.test(blk)) return blk;
      return blk.replace(/<c:scaling>[\s\S]*?<\/c:scaling>/, `<c:scaling><c:logBase val="10"/><c:orientation val="minMax"/><c:max val="${hi}"/><c:min val="${lo}"/></c:scaling>`);
    });
    x = x.replace(/<c:pt idx="\d+"><c:v><\/c:v><\/c:pt>/g, "");
    x = x.replace(/<c:dispBlanksAs val="[a-z]+"\/>/, '<c:dispBlanksAs val="gap"/>');
    zip.file(f, x);
  }
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  await applyTheme(OUT, THEME);
  fs.writeFileSync(path.join(__dirname, "notes.json"), JSON.stringify(notes, null, 1));
  console.log("wrote", OUT, "slides:", n);
})();
