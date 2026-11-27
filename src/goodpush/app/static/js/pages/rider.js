// Rider: what you can do now - your numbers today, best efforts, the physiology model, the power-curve heatmap, what
// training does for you (RiderPage).
import {Lazy, Pref, Segmented} from "../controls.js";
import {Page} from "./page.js";
import {S} from "../api.js";
import {$, css, esc, fmt} from "../util.js";
import {hint, niceDate, tile} from "../widgets.js";
import {router} from "../main.js";
import {ZOOM_HELP} from "./fitness.js";
import {EChart, TimeWindow} from "../charts.js";

const HEAT_RAMP = ["#00224e" , "#2a3f6d" , "#575d6d", "#7d7c78" , "#a59c74" , "#d2c060" , "#fee838"];   // cividis, dark -> bright
// observed as % of predicted (the curve heatmap's third view): diverging, centred on 100 %, clamped to lo..hi
const PERCENT_RAMP = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"];   // below .. 100 % .. above
const PERCENT = {lo: 60, mid: 100, hi: 115};
function percentColour(v) {
  const {lo, mid, hi} = PERCENT;
  return heatColour(v < mid ? 0.5 * (v - lo) / (mid - lo) : 0.5 + 0.5 * (v - mid) / (hi - mid), PERCENT_RAMP);
}
function heatColour(f, r = HEAT_RAMP) {
  const x = Math.max(0, Math.min(1, f)) * (r.length - 1), i = Math.min(Math.floor(x), r.length - 2), t = x - i;
  const a = parseInt(r[i].slice(1), 16), b = parseInt(r[i + 1].slice(1), 16);
  const ch = (k) => Math.round(((a >> k) & 255) * (1 - t) + ((b >> k) & 255) * t);
  return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}
// what each kind of training does for you (models.physio.TrainingResponse): per domain a colour (the zone scale,
// every other step), a short name and the label colour on it; its watts today come
// from the model's LT1 / CP / MAP
export const DOMAIN = {z1: ["--z1", "Easy", "--on-z1"], z2: ["--z3", "Tempo – threshold", "--on-z3"], z3: ["--z5", "VO2max", "--on-z5"],
  z4: ["--z7", "Short hard efforts", "--on-z7"], sprint: ["--muted", "Sprints", "--surface"]};
export const DOMAIN_ORDER = ["z1", "z2", "z3", "z4", "sprint"];   // long rides: endurance hours (the model has no own effect)
const LT1_TIP = "First (aerobic) threshold: below it you can ride for hours; easy rides stay below it.\n• From DFA alpha1 crossing 0.75 in steady riding (chest strap), or a ramp test\n• With the physiology model: from all your evidence\n• Each reading: Model > Threshold readings";
const CP_TIP = "Critical power: the most you can hold without tiring quickly, about LT2.\n• Above it you use up W′, a limited store\n• FTP (~1-hour power) sets plan targets and each ride's TSS\n• From the physiology model, else your best 3-20 min of 90 days (a floor unless all-out)";
const BEST_TIP = "Your best efforts.\n• ✓ verified: heart rate shows it was all-out (near your max at its end, steady effort), last 12 months\n• No tick: best of 90 days, maybe not all-out\n• The plan adds a timed max effort now and then to keep them current";
const BEST_S = [[15, "15 s"], [60, "1 min"], [300, "5 min"], [1200, "20 min"]];
const POWER_S = [[3600, "1 h"], [1200, "20 min"], [300, "5 min"], [60, "1 min"]];   // models.power_history.PowerHistory.DURATIONS, long to short
const POWER_TIP = "• Predicted (line): the most the model says you could hold this long, fresh\n• Observed (points): your best that day; most rides aren't all-out, so they sit below\n• Filled: a test or heart rate shows it was all-out\n• Clearly above the line: the model is too low\n• Dashed: the forecast, if you ride the plan\n• Click a point for its ride\n";
const CONFIRMED = ["test", "maxed"];       // effort kinds (physio KINDS) known to be all-out
const KIND_WORDS = {easy: "an easy ride", hard: "a hard workout or race", test: "a test", maxed: "all-out: heart rate reached your maximum"};

class RiderPage extends Page {
  showParts = new Pref("rider_parts", false);   // the response table's LT1 / CP / MAP columns

  lastOf(a) { for (let i = (a || []).length - 1; i >= 0; i--) if (a[i] != null) return a[i]; return null; }

  // the model's 5-95 % range today, in its own unit (k: a physio column; div: e.g. 1000 for kJ)
  rangeOf(k, unit = "W", div = 1, digits = 0) {
    const P = S.physio || {}, lo = this.lastOf(P[k + "_p05"]), hi = this.lastOf(P[k + "_p95"]);
    return lo == null || hi == null ? "" : `${fmt(lo / div, digits)}–${fmt(hi / div, digits)} ${unit}`;
  }

  // one row: LT1, CP / FTP, MAP, 20 min (predicted), W' - the physio model's where fitted, else what the rides show
  numbersSection() {
    const T = S.thresholds || {}, P = S.physio, p = S.power || {}, C = T.cp || {}, F = T.ftp || {}, l1 = T.lt1 || {};
    const p20 = this.predictedToday(1200);
    const mcp = P ? this.lastOf(P.cp) : null, mmap = P ? this.lastOf(P.map) : null, mwp = P ? this.lastOf(P.w_prime) : null;
    const lt1Sub = !l1.w ? "not enough steady riding with alphaHRV" : l1.lo != null ? `${fmt(l1.lo)}–${fmt(l1.hi)} W` : l1.hr ? `at ${fmt(l1.hr)} bpm` : "";
    const own = F.since && F.model_w != null && Math.round(F.w) !== Math.round(F.model_w);
    const ftpSub = own ? `FTP ${fmt(F.w)} W: your setting (model ${fmt(F.model_w)} W)` : `FTP ${fmt(p.ftp_now)} W · ${fmt(p.wkg_now, 2)} W/kg`;
    return `<div class="tiles">
    ${tile("LT1", l1.w ? fmt(l1.w) : "–", l1.w ? "W" : "", lt1Sub, "", LT1_TIP)}
    ${tile("CP · FTP", fmt(mcp ?? C.cp_90d), "W", [mcp != null ? this.rangeOf("cp") : "best efforts of 90 days (a floor)", ftpSub].filter(Boolean).join(" · "), "", CP_TIP)}
    ${mmap != null ? tile("MAP (5 min)", fmt(mmap), "W", this.rangeOf("map"), "", "Maximal aerobic power: about your best 5 minutes. The ceiling the other numbers sit under.") : ""}
    ${p20 != null ? tile("20 min", fmt(p20), "W", "predicted, fresh", "", "The most the model says you could hold for 20 minutes today, fresh and all-out.\n• Over time: Power over time below") : ""}
    ${mwp != null ? tile("W′", fmt(mwp / 1000, 1), "kJ", this.rangeOf("w_prime", "kJ", 1000, 1), "", "W′ (W prime): the work you can do above CP before you must ease off.\n• Like a battery that recharges below CP")
        : C.wprime_90d_kj != null ? tile("W′ (90 d)", fmt(C.wprime_90d_kj, 1), "kJ", "from best efforts", "", "W′ (W prime): the work you can do above CP before you must ease off.\n• From your best efforts of 90 days") : ""}
  </div>`;
  }

  // power over time (models.power_history.PowerHistory, /api/power) at the durations picked (one or more, each its own cividis
  // colour): the physio model's all-out power, fresh (predicted), each ride day's best (observed: filled with a ±1 sd
  // bar where a test or heart rate shows it was all-out, else a ring as opaque as the model's odds that it was) and the
  // plan's forecast (dashed). One y axis for all durations (whichever are on), so it holds still while you pick
  power = new Lazy("/api/power", () => { if (router.route() === "rider") this.powerReady(); });
  powerS = new Pref("rider_power_s", "1200");      // the durations shown (s, comma-separated)

  powerSecs() {
    const want = String(this.powerS.value).split(",").map(Number);
    const on = POWER_S.map(([s]) => s).filter((s) => want.includes(s));       // in POWER_S order, long to short
    return on.length ? on : [1200];
  }

  powerName(s) { return POWER_S.find(([t]) => t === s)[1]; }

  // cividis, fixed per duration (1 h darkest .. 1 min, the highest power, yellow), leaving out the end that would
  // vanish into the page
  powerColour(s) {
    const t = document.documentElement.dataset.theme, dark = t === "dark" || (!t && matchMedia("(prefers-color-scheme: dark)").matches);
    const [a, b] = dark ? [0.3, 1] : [0, 0.8], j = POWER_S.findIndex(([q]) => q === s);
    return heatColour(a + (b - a) * j / (POWER_S.length - 1));
  }

  predictedToday(s) {        // the model's predicted power for s seconds today (null before /api/power is back)
    const P = this.power.data, j = P?.s ? P.s.indexOf(s) : -1;
    return j < 0 ? null : P.predicted[j][P.date.indexOf(P.today)];
  }

  powerSection() {
    if (!S.physio) return "";
    this.power.load();
    const on = this.powerSecs();
    const picks = POWER_S.map(([s, n]) => `<button data-s="${s}" class="${on.includes(s) ? "on" : ""}"><span class="sw" style="background:${this.powerColour(s)}"></span>${n}</button>`).join("");
    const ink = "var(--ink2)";
    return `<h2>Power over time${hint(POWER_TIP + ZOOM_HELP)} <a class="small" href="#model" style="margin-left:10px;font-weight:400">How it was tested →</a></h2>
  <div class="card"><div class="row"><div class="seg" id="powerS" title="Show one or more durations">${picks}</div>
    <div class="legend"><span><i style="background:${ink}"></i>Predicted</span><span><i class="dash" style="border-color:${ink}"></i>Forecast</span><span><i class="dot" style="background:${ink}"></i>Observed, all-out</span><span><i class="dot" style="background:none;border:1.5px solid ${ink}"></i>Observed</span></div>
    <span class="spacer"></span><div class="seg" id="powerRange"><button data-d="90">3 m</button><button data-d="365" class="on">1 y</button><button data-d="0">All</button></div></div>
    <div id="powerChart" class="chart"></div></div>`;
  }

  powerReady() {             // /api/power is back: the 20-min tile that reads it, then the chart
    const nums = $("#riderNumbers");
    if (nums) nums.innerHTML = this.numbersSection();
    this.chartPower();
  }

  powerPoints(P, j) {        // one duration's observed points on the date axis: filled (all-out known) with bars, rings
    const n = P.date.length, O = P.observed[j], none = () => new Array(n).fill(null);
    const pts = {sure: none(), lo: none(), hi: none(), ring: none(), alpha: new Array(n).fill(0), effort: new Map()};
    O.i.forEach((d, q) => {
      pts.effort.set(d, q);
      if (CONFIRMED.includes(O.kind[q])) {
        pts.sure[d] = O.w[q]; pts.lo[d] = O.w[q] * Math.exp(-P.all_out_sd); pts.hi[d] = O.w[q] * Math.exp(P.all_out_sd);
      } else { pts.ring[d] = O.w[q]; pts.alpha[d] = 0.15 + 0.85 * (O.all_out ? O.all_out[q] / 100 : 0); }
    });
    return pts;
  }

  chartPower(days) {
    const P = this.power.data;
    if (!P?.s || !$("#powerChart")) return;
    const ms = (q) => new Date(q + "T12:00:00").getTime(), W = (v) => fmt(v) + " W", n = P.date.length;
    const day0 = ms(P.date[0]), dayAt = (x) => Math.round((x - day0) / 864e5);     // one date a day
    const pts = P.s.map((_, j) => this.powerPoints(P, j)), on = this.powerSecs().map((s) => [s, P.s.indexOf(s)]);
    const tw = TimeWindow.of("power");       // the opening year only once: picking durations keeps your zoom
    tw.showLast(days === undefined && tw.opened ? null : days, ms(P.today), ms(P.date[n - 1]), 365);
    tw.opened = true;
    const lines = [], dots = [];
    on.forEach(([s, j]) => {
      const c = this.powerColour(s), name = this.powerName(s), p = pts[j];
      lines.push({kind: "line", y: P.predicted[j], color: c, name: `Predicted, ${name}`, fmt: W, width: 1.6});
      if (P.forecast.length) lines.push({kind: "line", y: P.forecast[j], color: c, dash: true, name: `Forecast, ${name}`, planned: true, fmt: W});
      dots.push({kind: "dots", r: 3, hollow: true, alpha: p.alpha, y: p.ring, color: c, name: `Observed, ${name}`, fmt: W},
        {kind: "dots", r: 3, lo: p.lo, hi: p.hi, y: p.sure, color: c, name: `Observed, ${name}`, tipName: `Observed, ${name}, all-out`, fmt: W});
    });
    const yExtra = P.s.flatMap((_, j) => [P.predicted[j], P.forecast[j] || [], pts[j].ring, pts[j].sure]);   // every duration
    const tipNote = (x) => {          // each effort that day: share of predicted, its kind, the model's odds it was all-out
      const d = dayAt(x);
      const rows = on.map(([s, j]) => {
        const O = P.observed[j], q = pts[j].effort.get(d), pred = P.predicted[j][d];
        if (q == null || !pred) return "";
        const odds = O.all_out && !CONFIRMED.includes(O.kind[q]) ? ` · ${fmt(O.all_out[q])} % likely all-out` : "";
        return `<div class="small">${this.powerName(s)}: ${fmt(100 * O.w[q] / pred)} % of predicted · ${esc(KIND_WORDS[O.kind[q]] || O.kind[q])}${odds}</div>`;
      }).join("");
      return rows ? rows + '<div class="small muted">click for the ride</div>' : "";
    };
    const onPick = (x) => {           // the ride of the first duration shown with an effort that day
      const d = dayAt(x);
      for (const [, j] of on) { const q = pts[j].effort.get(d), id = q == null ? null : P.observed[j].id[q]; if (id) { location.hash = "#ride/" + encodeURIComponent(id); return; } }
    };
    EChart.draw("powerChart", {x: P.date, xType: "date", yFmt: (v) => v + " W", group: "power", scrollbar: true, yExtra,
      vline: P.forecast.length ? ms(P.today) : null, planFrom: ms(P.today), series: lines.concat(dots), tipNote, onPick});
  }

  // 15 s, 1 min, 5 min, 20 min: the heart-rate-verified best (models.thresholds.measured) with a tick and its ride, else
  // the best of the last 90 days; then the ramp tests, folded
  effortsSection() {
    const T = S.thresholds;
    if (!T) return "";
    const G = T.glycolytic || {}, C = T.cp || {}, M = T.measured || {};
    const best90 = {15: G.p15, 60: G.p60, 300: C.best_5min_90d, 1200: C.best_20min_90d};
    const tiles = BEST_S.map(([s, name]) => {
      const m = M[String(s)];
      if (m) return tile(name, fmt(m.w), "W", `✓ <a href="#ride/${encodeURIComponent(m.id)}">${niceDate(m.date, {day: "numeric", month: "short", year: "numeric"})}</a> · HR ${fmt(m.hr)} of ${fmt(m.max_hr)}`);
      return best90[s] != null ? tile(name, fmt(best90[s]), "W", "best of 90 days, not verified") : "";
    }).join("");
    const ramps = (T.ramps || []).slice().reverse().map((r) => `<tr class="click" data-href="#ride/${encodeURIComponent(r.id)}">
    <td>${esc(r.date)}</td><td><a href="#ride/${encodeURIComponent(r.id)}">${esc(r.name || "Ride")}</a></td>
    <td class="n">${fmt(r.lt1_w)} W · ${fmt(r.lt1_hr)}</td><td class="n">${fmt(r.lt2_w)} W · ${fmt(r.lt2_hr)}${r.lt2_loose ? `<div class="small muted">only a loose cap: ${esc(r.lt2_loose)}</div>` : ""}</td><td class="n">${fmt(r.top_w)} W</td></tr>`).join("");
    return `<h2>Best efforts${hint(BEST_TIP)}</h2>
  <div class="tiles">${tiles}</div>
  ${ramps ? `<details class="fold"><summary>Ramp tests</summary><div class="card">${hint("Rides where the power stepped up until you couldn't go on.\n• LT1 and LT2 come from them for 8 weeks (ahead of ordinary rides)\n• Top: the last step you finished")}
    <div class="tw"><table><thead><tr><th>Date</th><th>Ride</th><th class="n">LT1 · bpm</th><th class="n">LT2 · bpm</th><th class="n">Top</th></tr></thead><tbody>${ramps}</tbody></table></div></div></details>` : ""}`;
  }

  // the whole power curve over time (models.power_history.CurveHistory, /api/curve): weeks across, durations 5 s .. 2 h down,
  // colour = power relative to that duration's best in the shown span (rows comparable), watts on hover; the
  // "percent" view = observed as % of predicted on a fixed diverging scale (PERCENT)
  curve = new Lazy("/api/curve", () => { if (router.route() === "rider") this.chartCurve(); });
  curveView = "measured";
  curveSpan = 365 * 3;

  curveSection() {
    this.curve.load();
    const views = [["measured", "Observed (6-week best)"], ["model", "Predicted"], ["percent", "Observed ÷ predicted"]];
    return `<h2>Your power curve over time${hint("Durations 5 s to 2 h (down) over the weeks (across). Hover for the watts; grey: nothing that long ridden.\n• Observed: your best in the 6 weeks around that week\n• Predicted: the model's all-out curve, fresh\n• In these two, colour is per duration (5-95 % of the span shown)\n• Observed ÷ predicted: white = 100 %, blue below (most riding), red above = model too low\n" + ZOOM_HELP)}</h2>
  <div class="card"><div class="row"><div class="seg" id="curveView">${views.map(([v, n]) => `<button data-v="${v}" class="${this.curveView === v ? "on" : ""}">${n}</button>`).join("")}</div>
    <span class="spacer"></span><div class="seg" id="curveSpan">${[[365, "1 y"], [1095, "3 y"], [0, "All"]].map(([d, n]) => `<button data-d="${d}" class="${d === this.curveSpan ? "on" : ""}">${n}</button>`).join("")}</div></div>
    <div id="curveHeat" class="chart" style="margin-top:8px"></div>
    <div class="row small muted" id="curveKey" style="margin-top:6px">${this.curveKey()}</div></div>`;
  }

  curveKey() {      // the colour key under the heatmap, for the view shown
    const bar = (ramp) => `<span style="flex:0 0 160px;height:8px;margin:0 8px;border-radius:4px;background:linear-gradient(90deg,${ramp.join(",")})"></span>`;
    return this.curveView === "percent"
      ? `<span>${PERCENT.lo} % or less</span>${bar(PERCENT_RAMP)}<span>${PERCENT.hi} % or more · white = 100 %</span>`
      : `<span>lower</span>${bar(HEAT_RAMP)}<span>higher</span>`;
  }

  // weeks across (x = week index, each cell x .. x + 1), durations down; zoom like the other charts (TimeWindow "curveHeat",
  // in weeks), and after each zoom the rows are coloured again for the weeks shown (onView); the 1 y / 3 y / All buttons
  // set the window
  chartCurve() {
    const C = this.curve.data;
    if (!C || !C.date || !$("#curveHeat")) return;
    const Z = C[this.curveView] || C.measured, fixed = this.curveView === "percent", key = $("#curveKey");
    if (key) key.innerHTML = this.curveKey();
    const n = C.date.length, start = C.measured.reduce((m, row) => Math.min(m, row.findIndex((v) => v != null) >= 0 ? row.findIndex((v) => v != null) : n), n);
    const tw = TimeWindow.of("curveHeat"), open = this.curveSpan ? [Math.max(start, n - this.curveSpan / 7), n] : null;
    const [a, b] = tw.view || (!tw.opened && open) || [start, n], i0 = Math.max(start, Math.floor(a)), i1 = Math.min(n, Math.ceil(b));
    const label = (t) => t < 60 ? t + " s" : t < 3600 ? t / 60 + " min" : t / 3600 + " h", none = css("--line");
    const cells = [];
    C.s.forEach((t, r) => {
      const vals = Z[r].slice(i0, i1).filter((v) => v != null).sort((p, q) => p - q);
      const lo = vals[Math.floor(0.05 * (vals.length - 1))], hi = vals[Math.ceil(0.95 * (vals.length - 1))];   // 5-95 % of the weeks shown: one odd week can't flatten the rest
      for (let c = start; c < n; c++) {
        const v = Z[r][c];
        cells.push({value: [c, r], itemStyle: {color: v == null ? none : fixed ? percentColour(v) : heatColour(hi > lo ? (v - lo) / (hi - lo) : 1)}});
      }
    });
    const cols = b - a, ticks = [];
    let last = "";
    for (let c = start; c < n; c++) {        // a label each year, half-year or month, by how much is shown
      const d = C.date[c], y = d.slice(0, 4), m = d.slice(5, 7), k = cols > 160 ? y : cols > 40 ? y + (m >= "07") : y + m;
      if (k !== last && c > start) ticks.push([c, cols > 160 ? y : niceDate(d, {month: "short", year: "2-digit"})]);
      last = k;
    }
    const w = (v) => v == null ? "–" : fmt(v) + " W";
    EChart.draw("curveHeat", {xType: "num", xDomain: [start, n], view: open, rows: C.s.map(label), xTicks: ticks,
      onView: () => this.chartCurve(),
      tipItem: (c, r) => `<div class="th">${label(C.s[r])}, week to ${esc(niceDate(C.date[c], {day: "numeric", month: "short", year: "numeric"}))}</div>`
        + `<div>observed <b>${w(C.measured[r][c])}</b> · predicted <b>${w(C.model[r][c])}</b>${C.percent?.[r]?.[c] == null ? "" : ` · <b>${fmt(C.percent[r][c])} %</b>`}</div>`,
      series: [{kind: "heat", x: [start, n], cells}]});
    const [s0, s1] = tw.shown();
    document.querySelectorAll("#curveSpan button").forEach((y) => {
      const d = Number(y.dataset.d);
      y.classList.toggle("on", d ? !!tw.view && Math.abs(s1 - s0 - d / 7) < 1 && s1 >= n - 0.5 : !tw.view);
    });
  }

  html() {
    const p = S.power || {};
    return `
  <h1>You, the rider${hint("What you can do now, from your own rides:\n• Thresholds and best efforts today\n• A model of your physiology\n• What kinds of training worked for you")}</h1>
  <p class="sub">${esc(p.phenotype || "")}${p.phenotype_reason ? " – " + esc(p.phenotype_reason) : ""}</p>
  <h2>Your numbers today</h2>
  <div id="riderNumbers">${this.numbersSection()}</div>
  ${this.powerSection()}
  ${this.effortsSection()}
  ${S.physio ? `<h2>Physiology (model)<a class="small" href="#model" style="margin-left:10px;font-weight:400">How sure is the model? →</a>${hint("A model of your body, fitted to all your rides, tests and readings: what you could do fresh and all-out each day.\n• MAP: about your best 5 minutes\n• CP: the most you can hold without tiring quickly\n• LT1: the top of easy riding\n• Band: the 5-95 % range · dashed: if you ride the plan\n" + ZOOM_HELP)}</h2>
  <div class="card">
    ${S.physio.cp_p05 ? this.rangesAge() : ""}
    <div class="row"><div class="legend"><span><i style="background:var(--s3)"></i>LT1</span><span><i style="background:var(--s1)"></i>CP</span><span><i style="background:var(--s2)"></i>MAP (5 min)</span>${S.physio.cp_p05 ? '<span><i class="band" style="background:var(--s1)"></i>5–95 % range</span>' : ""}<span><i class="dash" style="border-color:var(--muted)"></i>If you ride the plan</span></div><span class="spacer"></span>
    <div class="seg" id="phyrange"><button data-d="90">3 m</button><button data-d="365" class="on">1 y</button><button data-d="0">All</button></div></div>
    <div id="phycap" class="chart"></div>
  </div>` : ""}
  ${this.curveSection()}
  <div id="riderResponse">${this.responseSection()}</div>
  <p class="small muted" style="margin-top:16px">Where these models come from: <a href="#sources">Sources</a>.</p>`;
  }

  domainWatts() {
    const P = S.physio || {}, now = (k) => { const v = this.lastOf(P[k]); return v == null ? null : Math.round(v); };
    const [l, c, m] = [now("lt1"), now("cp"), now("map")];
    if (l == null || c == null || m == null) return {};
    return {z1: `below ${l} W`, z2: `${l}–${c} W`, z3: `${c}–${m} W`, z4: `above ${m} W`};
  }

  swatch(d) { return `<span class="sw" style="background:var(${DOMAIN[d][0]})"></span>`; }

  domainMix(shares) {      // which kinds of riding a session works: a labelled bar and the main parts in words
    const parts = DOMAIN_ORDER.filter((d) => shares[d] > 0.005).map((d) => [d, shares[d]]);
    const words = parts.slice().sort((a, b) => b[1] - a[1]).filter(([, v]) => v >= 0.1)
      .map(([d, v]) => `${this.swatch(d)}${esc(DOMAIN[d][1])} ${fmt(v * 100)} %`).join(" · ");
    return `<div class="stack lab">${parts.map(([d, v]) => `<i style="width:${(v * 100).toFixed(1)}%;background:var(${DOMAIN[d][0]});color:var(${DOMAIN[d][2]})" title="${esc(DOMAIN[d][1])}: ${fmt(v * 100)} %">${v >= 0.18 ? fmt(v * 100) + " %" : ""}</i>`).join("")}</div>
    <div class="small" style="margin-top:3px">${words}</div>`;
  }

  responseSection() {
    const R = S.response; if (!R || !R.workouts) return "";
    const E = R.effects || {}, dw = this.domainWatts(), parts = this.showParts.value;
    const main = (w) => Object.entries(w.domains).sort((a, b) => b[1] - a[1])[0]?.[0];
    const top = Math.max(...R.workouts.map((w) => w.gain.p20), 0.01);
    const pct = (v) => (v > 0 ? "+" : "") + fmt(v, 1) + " %";
    const rng = (w, k) => w.range ? ` <span class="small muted">(${fmt(w.range[k][0], 1)} – ${fmt(w.range[k][1], 1)})</span>` : "";
    // hardest first; "long" only in a response saved before long rides counted as endurance (2026-11-13)
    const rows = R.workouts.filter((w) => w.key !== "long").sort((a, b) => b.watts - a.watts).map((w) => {
      const guess = E[main(w)]?.prior_like;
      return `<tr${guess ? ' class="muted"' : ""}><td><b>${esc(w.label)}</b><div class="small muted">${esc(w.session)} · ${fmt(w.watts)} W</div></td>
      <td style="min-width:200px">${this.domainMix(w.domains)}</td>
      <td style="min-width:150px"><div class="bar"><i style="width:${Math.max(0, w.gain.p20 / top * 100).toFixed(0)}%"></i></div>
        <div class="small">${pct(w.gain.p20)}${rng(w, "p20")}${guess ? " · assumed" : ""}</div></td>
      ${parts ? `<td class="n">${pct(w.gain.lt1)}</td><td class="n">${pct(w.gain.cp)}</td><td class="n">${pct(w.gain.map)}</td>` : ""}</tr>`;
    }).join("");
    const legend = DOMAIN_ORDER.filter((d) => d !== "sprint").map((d) => `<span><i style="background:var(${DOMAIN[d][0]})"></i>${DOMAIN[d][1]}${dw[d] ? ` <span class="muted">${dw[d]}</span>` : ""}</span>`).join("");
    return `
  <h2>What training does for you${hint("How you respond to each kind of riding, as the model learned it.\n• From your own rides only: good bets to test, not facts\n• Changes as new rides come in\n• How each kind did for you: Model page")}</h2>
  <div class="legend" style="margin:0 0 8px">${legend}</div>
  <div class="card">
    <div class="row"><h3 style="margin:0">Where your gains are${hint(`One more session a week for ${R.weeks} weeks, on top of your last ${R.weeks}: what it would add to your 20-minute power (fresh, at the end).\n• LT1 · CP · MAP: the gain in each threshold\n• What it works: how the session splits over the kinds of riding, at today's watts\n• Greyed, "assumed": your rides barely moved the model's assumption\n• Brackets: the likely range`)}</h3><span class="spacer"></span>
      <div class="seg small"><button id="riderParts" class="${parts ? "on" : ""}" title="Show what each session adds to LT1, CP and MAP">LT1 · CP · MAP</button></div></div>
    ${R.draws ? this.rangesAge() : `<p class="small muted" style="margin:0 0 8px">Single best-fit numbers; ranges come with a full model run (goodpush physio --nuts --samples 2500 --ess 400 --max-minutes 540).</p>`}
    <div class="tw"><table><thead><tr><th>Add one a week</th><th>What it works</th><th>20-min power after ${R.weeks} weeks</th>${parts ? '<th class="n">LT1</th><th class="n">CP</th><th class="n">MAP</th>' : ""}</tr></thead><tbody>${rows}</tbody></table></div>
  </div>`;
  }

  rangesAge() {     // the date of the run the 5-95 % ranges come from (models.physio: draws_day; re-made about weekly)
    const P = S.physio || {}, last = (P.date || [])[(P.date || []).length - 1], d = x => esc(niceDate(x, {day: "numeric", month: "short"}));
    return P.draws_day && last && P.draws_day < last
      ? `<p class="small muted" style="margin:0 0 8px">Ranges as of ${d(P.draws_day)} (they move slowly and are re-made about weekly); best-fit numbers up to ${d(last)}.</p>` : "";
  }

  chartPhysio(days) {   // the model's LT1, CP and MAP, then the plan's forecast (dashed)
    const P = S.physio; if (!P) return;
    const F = P.forecast || {date: []}, d = P.date, last = d[d.length - 1];
    const fut = F.date.map((x, i) => [x, i]).filter(([x]) => x > last);
    const x = d.concat(fut.map(([q]) => q)), pad = (a) => (a || []).concat(fut.map(() => null));
    const ahead = (k) => d.map((_, i) => (i === d.length - 1 ? P[k][i] : null)).concat(fut.map(([, i]) => F[k][i]));
    const ms = (q) => new Date(q + "T12:00:00").getTime(), W = (v) => fmt(v) + " W";
    TimeWindow.of("physio").showLast(days, ms(last), ms(x[x.length - 1]), 365);
    const vline = fut.length ? ms(last) : null, lines = [["lt1", "--s3", "LT1"], ["cp", "--s1", "CP"], ["map", "--s2", "MAP (5 min)"]];   // low to high
    const series = [];
    lines.forEach(([k, c]) => { if (P[k + "_p05"]) series.push({kind: "band", y: pad(P[k + "_p05"]), y2: pad(P[k + "_p95"]), color: css(c)}); });
    lines.forEach(([k, c, n]) => series.push({kind: "line", y: pad(P[k]), color: css(c), name: n, fmt: W},
                                             {kind: "line", y: ahead(k), color: css(c), dash: true, name: n + ", planned", planned: true, fmt: W}));
    // hover: highest first (MAP ... LT1), as they sit on the chart
    EChart.draw("phycap", {x, xType: "date", yFmt: (v) => v + " W", group: "physio", scrollbar: true, vline, planFrom: ms(last), tipReverse: true, series});
  }

  wire() {
    requestAnimationFrame(() => { this.chartPower(); this.chartCurve(); this.chartPhysio(); });
    new Segmented("#powerS", () => {           // one or more durations on
      this.powerS.set([...document.querySelectorAll("#powerS button.on")].map((x) => x.dataset.s).join(","));
      this.chartPower();
    }, true);
    new Segmented("#curveView", (b) => { this.curveView = b.dataset.v; this.chartCurve(); });
    new Segmented("#curveSpan", (b) => {      // chartCurve marks the button that matches the window shown
      this.curveSpan = Number(b.dataset.d); const n = this.curve.data?.date?.length; if (n) TimeWindow.of("curveHeat").zoomTo(this.curveSpan ? [n - this.curveSpan / 7, n] : null);
    });
    new Segmented("#powerRange", (b) => this.chartPower(Number(b.dataset.d)));
    new Segmented("#phyrange", (b) => this.chartPhysio(Number(b.dataset.d)));
    this.wireParts();
  }

  wireParts() {          // the LT1 / CP / MAP columns: only the response block is redrawn (the page keeps its place)
    $("#riderParts")?.addEventListener("click", () => {
      this.showParts.set(!this.showParts.value);
      $("#riderResponse").innerHTML = this.responseSection();
      this.wireParts();
    });
  }
}
export const riderPage = new RiderPage();
