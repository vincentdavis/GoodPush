// HTML pieces several pages use (tiles, workout profiles, steps, session cards) and the floating tips.
import {api, S} from "./api.js";
import {$, esc, fmt, hhmm} from "./util.js";

export const ICON = {
  green: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
  amber: '<svg viewBox="0 0 24 24"><path d="M12 7v6M12 17h.01"/><circle cx="12" cy="12" r="9.5"/></svg>',
  red: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};
export const STATUS_LABEL = {green: "Good to go", amber: "Go easier", red: "Rest / easy only"};
export const TYPE_LABEL = {rest: "Rest", recovery: "Recovery", endurance: "Endurance", long: "Long ride", threshold: "Threshold",
  vo2max: "VO2max", sweetspot: "Sweet spot", tempo: "Tempo", test: "FTP test", long_endurance: "Long ride",
  anaerobic: "Anaerobic", race: "Race", ramp: "Ramp test", durability: "Durability", lsct: "LSCT"};
// helpers
// 💡 an explanation kept out of the way: shown on hover, keyboard focus or a tap
export const hint = (text) => `<span class="hint" tabindex="0" role="note" aria-label="${esc(text)}" data-tip="${esc(text)}">💡</span>`;
// easier / harder (planning.workouts.Adjuster): buttons data-adjust="-1" / "1", off at the limit; `steps` = how
// far from `base` ("as planned", "as built") it is now
export function adjustRow(steps, easier, harder, base, tip) {
  const n = Math.abs(steps);
  const now = !steps ? base : `${n} step${n > 1 ? "s" : ""} ${steps < 0 ? "easier" : "harder"}`;
  const btn = (d, on, label, off) =>
    `<button class="btn sm" data-adjust="${d}"${on ? "" : ` disabled title="${off}"`}>${label}</button>`;
  return `<div class="row adjust">${btn(-1, easier, "− Easier", "As easy as it gets")}
    <span class="small muted">${esc(now)}</span>${btn(1, harder, "Harder +", "As hard as it gets")}${hint(tip)}</div>`;
}
export const today = () => S.plan?.generated || S.today?.date;
export const sessions = () => S.plan?.sessions || [];
export function sessionOn(ds) { return sessions().find((s) => s.date === ds); }
// every morning sign checked (planning.readiness.MorningSigns): green fine, amber a bit off, red well off; one off on
// its own changes nothing, 2 points together make the day amber, 4 red
export function signsRow(rd) {
  const G = rd.signs || [], cls = ["green", "amber", "red"];
  if (!G.length) return "";
  return `<div class="signs">${G.map((g) => `<span class="status sm ${cls[Math.min(g.points, 2)]}" title="${esc(g.text)}">${esc(g.name)}</span>`).join("")}
    ${hint("This morning's signs against your usual (hover one for its numbers).\n• Green: normal · amber: a bit off (1 point) · red: well off (2 points)\n• 2 points together: an amber day; 4: red\n• One sign alone changes nothing\n• How you feel and your illness markers count on their own")}</div>`;
}
// the last 7 mornings' readiness, today last: one oval each in the colour-blind-safe status colours (--good / --warn / --bad),
// the day, verdict and reasons in its tooltip and label, so the colour never carries it alone
export function morningsStrip(rd) {
  const M = rd.mornings;
  if (!M?.length) return "";
  const n = M.filter((m) => m.status !== "green").length;
  const dot = (m) => {
    const t = `${niceDate(m.date, {weekday: "short", day: "numeric", month: "short"})}: ${STATUS_LABEL[m.status] || m.status}`
      + ` – ${m.reasons.length ? m.reasons.join("; ") : "nothing flagged"}`;
    return `<span class="mdot7 ${esc(m.status)}" role="img" title="${esc(t)}" aria-label="${esc(t)}"></span>`;
  };
  return `<div class="mornings"><span>Last 7 mornings</span><span class="mdots">${M.map(dot).join("")}</span>
    <span class="muted">${n ? `${n} not green` : "all green"}${rd.streak ? " – a run: tomorrow easy too, until a green morning" : ""}</span></div>`;
}
export function niceDate(ds, opts = {weekday: "short", day: "numeric", month: "short"}) {
  return new Date(ds + "T12:00:00").toLocaleDateString(undefined, opts);
}
export function tile(label, value, unit, sub, cls = "", tip = "") {   // tip: a 💡 explaining it
  return `<div class="tile"><div class="l">${esc(label)}${tip ? hint(tip) : ""}</div><div class="v">${value}${unit ? `<small>${esc(unit)}</small>` : ""}</div>${sub ? `<div class="s ${cls}">${sub}</div>` : ""}</div>`;
}
export function zoneVar(frac) {   // %FTP -> sequential cividis ramp
  const z = frac < .56 ? 1 : frac < .76 ? 2 : frac < .91 ? 3 : frac < 1.06 ? 4 : frac < 1.21 ? 5 : 6;
  return `var(--z${z + (z >= 2 ? 0 : 0)})`;
}
export function profileSVG(s, cls = "profile") {
  const st = s.steps || [];
  if (!st.length) return "";
  const ftp = S.plan.ftp || 200;
  const tot = st.reduce((a, x) => a + x.min, 0);
  const maxF = Math.max(1.2, ...st.map((x) => Math.max(x.hi, x.hi_end || 0) / ftp));
  let x = 0, rects = "";
  for (const p of st) {
    const f = (p.lo + p.hi) / 2 / ftp, f1 = p.lo_end != null ? (p.lo_end + p.hi_end) / 2 / ftp : f;
    const w = p.min / tot * 1000, h = Math.max(6, f / maxF * 100), h1 = Math.max(6, f1 / maxF * 100);
    const data = `fill="${zoneVar((f + f1) / 2)}" data-n="${esc(p.name)}" data-m="${p.min}" data-w="${esc(stepW(p))}"`;
    rects += p.lo_end != null       // a ramp: a slope
      ? `<polygon points="${(x + 0.6).toFixed(1)},100 ${(x + 0.6).toFixed(1)},${(100 - h).toFixed(1)} ${(x + w - 0.6).toFixed(1)},${(100 - h1).toFixed(1)} ${(x + w - 0.6).toFixed(1)},100" ${data}></polygon>`
      : `<rect x="${(x + 0.6).toFixed(1)}" y="${(100 - h).toFixed(1)}" width="${Math.max(w - 1.2, 1).toFixed(1)}" height="${h.toFixed(1)}" rx="3" ${data} data-lo="${p.lo}" data-hi="${p.hi}"></rect>`;
    x += w;
  }
  const yF = 100 - 1 / maxF * 100;
  return `<svg class="${cls}" viewBox="0 0 1000 100" preserveAspectRatio="none" role="img" aria-label="Workout profile">${rects}<line x1="0" x2="1000" y1="${yF}" y2="${yF}" stroke="var(--muted)" stroke-dasharray="6 6" stroke-width="1" vector-effect="non-scaling-stroke"/></svg>`;
}
function stepDur(m) {  // 30 s, 2 min, 4:45, 1:30 h
  if (m < 1) return `${Math.round(m * 60)} s`;
  const s = Math.round(m * 60);
  return s >= 3600 ? hhmm(s / 60) : s % 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s / 60} min`;
}
function stepItem(b) {  // a step, or {reps, steps}: repeats nest (a set of micro intervals in its sets)
  if (b.steps) return b.reps > 1 ? `<li class="rep"><b>${b.reps} ×</b><ul class="steps">${b.steps.map(stepItem).join("")}</ul></li>`
    : b.steps.map(stepItem).join("");
  return `<li><span>${stepDur(b.min)} ${esc(b.name)}${b.cadence ? ` · ${b.cadence} rpm` : ""}${b.cue ? ` <small class="muted">${esc(b.cue)}</small>` : ""}</span><span class="w">${stepW(b)}</span></li>`;
}
// a step's watts: 200–215 W, a ramp 150–160 → 250–260 W
const stepW = (b) => `${b.lo}–${b.hi}${b.lo_end != null ? ` → ${b.lo_end}–${b.hi_end}` : ""} W`;
export function stepsList(s) {
  const bl = s.blocks || [];
  return bl.length ? `<ul class="steps">${bl.map(stepItem).join("")}</ul>` : "";
}
export function statusChip(s) {
  if (s.date === today()) return '<span class="chip today">Today</span>';
  if (s.status) return `<span class="chip ${esc(s.status)}">${esc(s.status)}</span>`;
  return "";
}
// today's session with steps: ride it with `goodpush ride` (the trainer and the overlay on this computer)
export const rideable = (p, ds) => ds === today() && p.type !== "rest" && p.steps?.length && p.status !== "done";
export const rideButton = (key) => `<button class="btn sm primary" data-ride-key="${esc(key)}" title="Opens the goodpush ride overlay on this computer">▶ Ride it now</button>`;
export function rideIt(key) {
  api.action({action: "ride", key}, "Opening the ride window – it stays on top; your trainer takes the watts.");
}
export function markButtons(s) {
  if (s.type === "rest") return "";
  const done = s.status === "done", sk = s.status === "skipped";
  return `<div class="row" style="margin-top:12px">
    ${rideable(s, s.date) ? rideButton(s.key || s.date) : ""}
    <button class="btn sm ${done ? "primary" : ""}" data-mark="${s.date}" data-v="${done ? "" : "done"}">${done ? "✓ Done" : "Mark done"}</button>
    <button class="btn sm" data-mark="${s.date}" data-v="${sk ? "" : "skipped"}">${sk ? "Skipped (undo)" : "Skipped"}</button>
    ${s.fit ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(s.fit)}" download>Workout (.fit)</a>` : ""}
    ${s.zwo ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(s.zwo)}" download title="For Incyclist, Zwift, TrainingPeaks, the MyWhoosh Workout Builder">Zwift / Incyclist file</a>` : ""}
    ${s.icu ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(s.icu)}" download title="Paste into an intervals.icu workout">intervals.icu text</a>` : ""}
  </div>`;
}
export function sessionCard(s, big = true) {
  if (!s && sessions().length) return `<div class="card"><div class="sess-head"><div class="sess-title">Rest day</div></div><div class="why">Nothing planned today.</div></div>`;
  if (!s) return `<div class="card muted">No plan yet – run <code>goodpush plan</code>.</div>`;
  if (s.type === "rest") {
    return `<div class="card"><div class="sess-head"><div class="sess-title">Rest day</div></div><div class="why">${esc(s.why)}</div>${s.adjusted ? `<div class="adj">${esc(s.adjusted)}</div>` : ""}</div>`;
  }
  return `<div class="card">
    <div class="sess-head"><div class="sess-title">${esc(s.title)}</div>${statusChip(s)}</div>
    <div class="meta">${TYPE_LABEL[s.type] || s.type} · ${hhmm(s.minutes)} · ${fmt(s.tss)} TSS · FTP ${fmt(S.plan.ftp)} W</div>
    ${profileSVG(s)}
    ${big ? stepsList(s) : ""}
    ${s.note ? `<div class="why">${esc(s.note)}</div>` : ""}
    <div class="why">${esc(s.why)}</div>
    ${s.adjusted ? `<div class="adj">⚠ ${esc(s.adjusted)}</div>` : ""}
    ${markButtons(s)}
  </div>`;
}
export function delta(v, base, unit, goodUp, nd = 0) {
  if (v == null || base == null) return "";
  const d = v - base;
  if (Math.abs(d) < (nd ? 0.15 : 1)) return `normal (${fmt(base, nd)}${unit})`;
  const good = goodUp ? d > 0 : d < 0;
  return `<span class="${good ? "up" : "down"}">${d > 0 ? "+" : ""}${fmt(d, nd)}${unit}</span> vs usual ${fmt(base, nd)}`;
}

class HintTip {
  // one floating box for every 💡, placed below it (above when there's no room) and kept inside the window. A tip's
  // text is lines: "• " lines make a bulleted list, the others short paragraphs (ending in ":" = a small heading)
  show(h) {
    const t = $("#hintTip") || document.body.appendChild(Object.assign(document.createElement("div"), {id: "hintTip"}));
    t.replaceChildren(...this.parts(h.dataset.tip)); t.hidden = false;
    const r = h.getBoundingClientRect(), w = t.offsetWidth, ht = t.offsetHeight, pad = 8;
    t.style.left = Math.max(pad, Math.min(innerWidth - w - pad, r.left - 8)) + "px";
    t.style.top = (r.bottom + 6 + ht <= innerHeight - pad ? r.bottom + 6 : Math.max(pad, r.top - 6 - ht)) + "px";
  }

  parts(text) {
    const out = [];
    let list = null;
    for (const line of text.split("\n").map((s) => s.trim()).filter(Boolean)) {
      if (line.startsWith("• ")) {
        if (!list) out.push(list = document.createElement("ul"));
        list.appendChild(Object.assign(document.createElement("li"), {textContent: line.slice(2)}));
      } else {
        list = null;
        out.push(Object.assign(document.createElement("p"), {textContent: line, className: line.endsWith(":") ? "head" : ""}));
      }
    }
    return out;
  }

  hide = () => { const t = $("#hintTip"); if (t) t.hidden = true; };
}
const hintTip = new HintTip();

class ProfileTip {
  // one floating tip for every workout profile (Plan, Today, Calendar): the step under the pointer,
  // solid (unlike the chart tips), above the profile so it never covers the workout
  box = null;

  follow(e) {
    const r = e.target.closest ? e.target.closest("[data-m]") : null;
    if (!r) { if (this.box) this.box.hidden = true; return; }
    if (!this.box) { this.box = document.body.appendChild(document.createElement("div")); this.box.className = "tip proftip"; }
    const d = r.dataset, ftp = S?.plan?.ftp || 200, pct = (v) => Math.round(v / ftp * 100);
    this.box.innerHTML = `<div class="th">${esc(d.n)}</div>${stepDur(+d.m)}<b>${esc(d.w)}</b>
    ${d.lo ? `<span class="muted">${pct(+d.lo)}–${pct(+d.hi)} % FTP</span>` : ""}`;
    this.box.hidden = false;
    const W = this.box.offsetWidth, H = this.box.offsetHeight, box = r.ownerSVGElement.getBoundingClientRect();
    this.box.style.left = Math.max(8, Math.min(e.clientX - W / 2, innerWidth - W - 8)) + "px";
    this.box.style.top = (box.top - H - 8 >= 0 ? box.top - H - 8 : box.bottom + 8) + "px";   // above the workout, never over it
  }
}
const profileTip = new ProfileTip();

for (const [on, off] of [["mouseover", "mouseout"], ["focusin", "focusout"]]) {
  document.addEventListener(on, (e) => { const h = e.target.closest?.(".hint"); if (h) hintTip.show(h); });
  document.addEventListener(off, (e) => { if (e.target.closest?.(".hint")) hintTip.hide(); });
}

window.addEventListener("scroll", hintTip.hide, true);

window.addEventListener("hashchange", hintTip.hide);

document.addEventListener("pointermove", (e) => profileTip.follow(e));

document.addEventListener("scroll", () => { if (profileTip.box) profileTip.box.hidden = true; }, true);
