// Calendar: the TSS strip and the week rows, scrolled together (CalendarPage).
import {Page} from "../pages/page.js";
import {api, S, toast} from "../api.js";
import {$, addDays, css, esc, fmt, hhmm} from "../util.js";
import {ask} from "../controls.js";
import {hint, niceDate, today} from "../widgets.js";
import {router} from "../main.js";
import {sessionModal} from "./session.js";
import {sessionMoves} from "./moves.js";
import {AWAY_SEL, awayDays} from "./away.js";
import {el, MON, niceTicks} from "../charts.js";
import {builder} from "../builder.js";
import {seasonBlocks} from "./season.js";

const TYPE_COLOR = {recovery: "--z2", endurance: "--z3", long: "--z3", long_endurance: "--z3", tempo: "--z4",
  sweetspot: "--z5", threshold: "--z6", vo2max: "--z7", anaerobic: "--z7", race: "--s7", test: "--s7", ramp: "--s7",
  lsct: "--s7"};
export const typeColor = (t) => `var(${TYPE_COLOR[t] || "--muted"})`;
export const actKind = (a) => a.sport === "ride" ? a.workout : a.sport;    // a ride's workout type, else its sport
const biggest = (xs) => xs.reduce((b, x) => (x.tss || 0) > (b?.tss || 0) ? x : b, null);   // the day's main session
// calTop: the calendar's scroll position in weeks (fractional), kept across re-renders
// a loading week that uses (nearly) all its hours: why its TSS stops rising
const RACE_STAR = {A: "#d4a62a", B: "#b8bcc4", C: "#b5793e"};   // a race's star on the TSS strip: gold, silver, bronze
export const HOURS_LIMITED_TIP ="This week uses nearly all the hours it may (weekly hours, each day's time, or the build-up from recent weeks).\n• TSS levels off: progress comes from the interval levels\n• More hours a week or a day (Settings > Plan) lets it grow";
export const PUSH_WEEK_TIP = "The block's last loading week, before its recovery week: the overload.\n• A harder load target (more fatigue)\n• Hours still at most +10 % a week on your recent weeks";
// a week whose 4-week CTL ramp is outside your range (RampRange): why, and where the range comes from
const rampTip = (flag, r, rr) => {
  if (!rr) return "";
  const lo = `${rr.lo} (${rr.lo_learned ? "where your history shows gains" : "a usual figure"})`;
  const hi = `${rr.hi} (${rr.hi_learned ? "the fastest you've built often without getting ill more" : "a usual limit"})`;
  return flag === "fast"
    ? `Load (CTL) builds ${r} a week over 4 weeks here, above ${hi}. Watch your mornings: easy days follow if they turn amber.`
    : `Load (CTL) builds only ${r} a week over 4 weeks here, below ${lo}. More hours (Settings > Plan) or harder intervals would build faster.`;
};
export const rampChip = (flag, r, rr) => flag
  ? `<span class="ramp-${flag}" title="${esc(rampTip(flag, r, rr))}">${flag === "fast" ? "fast build" : "slow build"}</span>` : "";
export const calGone = {minutes: 0, tss: 0};   // a session taken off its day
const calIndex = (ds) => Math.round((new Date(ds + "T12:00:00") - new Date(calendarPage.CAL.start + "T12:00:00")) / 864e5);
export const REP_RESULT = {hit: ["done", "on target"], partial: ["amber", "partly"], short: ["amber", "short"], low: ["missed", "low"],
  high: ["amber", "above"], missed: ["missed", "not done"]};
const CAL_HELP = "Chart and calendar follow each other.\n" +
  "• Scroll or drag either; hover a day to see it in both\n" +
  "• Click a week (or its total) to focus it; Go to picks a date\n" +
  "• Drag a ride onto a planned session: it counts for it\n" +
  "• Drag a session to another day to move it there (phone: hold first)\n" +
  "• Hold an easy ride over another one: one longer ride\n" +
  "• Click a session to see, resize or delete it\n" +
  "The lock on a session:\n" +
  "• 🔓 the plan's: click to lock it (🔒 kept exactly as it is)\n" +
  "• Blue 🔓 your change: the plan starts from it but may change it\n" +
  "• Click it to lock; a locked change, click: back to the plan's version\n" +
  "• Clear my changes: the unlocked ones go back to the plan\n" +
  "• Days away (✈): the plan moves around them; click tags (Shift+click: a range), then Remove";

class CalendarPage extends Page {
  wire() { requestAnimationFrame(() => this.wireCalendar()); }

  // calendar
  // TSS strip (scrolls sideways) above a week-per-row calendar (scrolls down); scrolling either
  // one moves the other, and the shaded box in the strip marks the weeks the calendar shows.
  DAY_W = 14;
  WEEK_W = 7 * this.DAY_W;     // the calendar chart's day width: Ctrl + wheel / pinch zooms it (6..40 px)

  CAL = null;
  calKey = "";
  calLoading = false;
  calAgain = false;
  calTop = null;
  calResize = null;

  // where the calendar was scrolled: the first week row in view and its pixel offset (a redraw puts that row back
  // exactly, however much rows above or below grew or shrank - a move changes a week's height)
  calAnchor = null;

  calFocus = null;                        // the week you picked (index): marked in the chart and the calendar

  calKeyNow = () => `${S.generated}|${S.plan_version}|${S.data_version}`;   // the plan, the rides

  loadCal(redraw = true) {                   // redraw = false: the caller renders once it's back
    if (this.calLoading) { this.calAgain = true; return Promise.resolve(); }   // asked again mid-fetch: fetch again when this one is back
    this.calLoading = true;
    const key = this.calKeyNow();
    return fetch("/api/calendar", {cache: "no-store"}).then((r) => r.json()).then((j) => { this.CAL = j; })
      .catch(() => { this.CAL = this.CAL || {}; })
      .finally(() => {
        this.calLoading = false;
        if (this.calAgain) { this.calAgain = false; this.loadCal(); return; }
        this.calKey = key; if (redraw && router.route() === "calendar") router.render();
      });
  }

  // show a change straight away (greyed, "…ing"); the re-plan replaces it. patch: its new minutes / tss,
  // so the week's hours and TSS follow at once; what = null: undo (the old values back)
  calPending(key, what, patch) {
    const p = this.CAL?.planned[key];
    if (!p) return;
    if (what) {
      p.pending = what;
      if (patch) { p._was ||= {minutes: p.minutes, tss: p.tss}; Object.assign(p, patch); }
    } else {
      delete p.pending;
      if (p._was) { Object.assign(p, p._was); delete p._was; }
      if (p._temp) delete this.CAL.planned[key];
    }
    this.CAL._m = null;
    if (router.route() === "calendar") router.render();
  }

  async clearMine() {                 // your unlocked changes from today on back to the plan (locked ones stay)
    const mine = Object.entries(this.CAL.planned).filter(([k, p]) => k.slice(0, 10) >= this.CAL.today && p.custom
      && !p.locked);
    if (!mine.length) { toast.show("No changes of yours to clear (locked ones stay: unlock them first)"); return; }
    if (!await ask.confirm({title: `Give ${mine.length} day(s) back to the plan?`, yes: "Give them back", danger: true,
      text: "The days you changed from today on go back to the plan's version. Locked days stay as they are."})) return;
    api.action({action: "clear_own"}, "Your changes cleared – updating the plan…");
  }

  calPlaceholder(day, src, what) {   // a session about to land on `day` (move / paste): its key
    const key = `${day}#~${Date.now()}`;
    this.CAL.planned[key] = {...this.CAL.planned[src], ride: undefined, status: undefined, compliance: undefined, custom: true, _temp: true};
    this.calPending(key, what);
    return key;
  }

  calModel() {
    if (this.CAL._m) return this.CAL._m;
    const n = this.CAL.tss.length, nw = n / 7;
    const dates = Array.from({length: n}, (_, i) => addDays(this.CAL.start, i));
    const ti = dates.indexOf(this.CAL.today);
    const byDay = {}, byId = {}, linked = {};
    for (const a of this.CAL.acts) { (byDay[a.date] ||= []).push(a); byId[a.id] = a; }
    const planByDay = {};                   // a day can have more than one session ("day#2" ...)
    for (const [k, p] of Object.entries(this.CAL.planned)) { if (p.ride) linked[p.ride] = k.slice(0, 10); (planByDay[k.slice(0, 10)] ||= []).push(k); }
    for (const ks of Object.values(planByDay)) ks.sort();
    // planned TSS per day from the sessions (as the server sums it), so a change shows in the weeks at once
    const planTss = dates.map((ds) => planByDay[ds] ? planByDay[ds].reduce((t, k) => t + (this.CAL.planned[k].tss || 0), 0) : null);
    const phase = new Array(n).fill(null);           // the season phase each day is in (null in gaps)
    for (const p of this.CAL.phases || []) for (let i = Math.max(0, calIndex(p.start)); i <= Math.min(n - 1, calIndex(p.end)); i++) phase[i] = p;
    // each day's type for the TSS chart's colours: the biggest ride, and the biggest session still planned
    const doneKind = dates.map((ds) => { const a = biggest(byDay[ds] || []); return a ? actKind(a) : null; });
    const planKind = dates.map((ds) => biggest((planByDay[ds] || []).map((k) => this.CAL.planned[k]))?.type || null);
    const weeks = [];
    for (let w = 0; w < nw; w++) {
      let act = 0, rest = 0, min = 0, pmin = 0, last = -1;
      const run = [], prun = [];
      for (let k = 0; k < 7; k++) {
        const i = w * 7 + k, t = this.CAL.tss[i], p = planTss[i];
        act += t || 0;
        if (t != null) last = k;
        if ((i > ti || (i === ti && !t)) && p) rest += p;     // still to come: count the plan
        if (i > ti || (i === ti && !t)) for (const key of planByDay[dates[i]] || []) pmin += this.CAL.planned[key].minutes || 0;
        run.push(act); prun.push(act + rest);
        for (const a of byDay[dates[i]] || []) min += a.min || 0;
      }
      weeks.push({start: dates[w * 7], act, rest, total: act + rest, min, pmin, run, prun, last, target: this.CAL.weeks?.[dates[w * 7]],
        kind: this.CAL.week_kind?.[dates[w * 7]]});
    }
    return (this.CAL._m = {n, nw, dates, ti, byDay, byId, linked, planByDay, planTss, weeks, phase, doneKind, planKind});
  }

  dayIndex(ds) { return calIndex(ds); }      // a date's day on the chart (0 = the calendar's first day)

  calReach(end) {                            // empty weeks up to `end` (a season block past the plan); true if it grew
    const n = this.CAL.tss.length, need = Math.ceil((calIndex(end) + 1) / 7) * 7;
    if (need <= n) return false;
    const pad = new Array(need - n).fill(null);
    this.CAL.tss = [...this.CAL.tss, ...pad];
    if (this.CAL.plan_tss) this.CAL.plan_tss = [...this.CAL.plan_tss, ...pad];
    this.CAL._m = null;
    return true;
  }

  phaseNow() {   // "Phase now: Build · week 3 of 8", or the next phase when today is in a gap
    const M = this.calModel(), p = M.ti >= 0 ? M.phase[M.ti] : null;
    if (p) {
      const kind = M.weeks[Math.floor(M.ti / 7)]?.kind;
      return `Phase now: <b>${esc(p.name)}</b> · week ${Math.floor((M.ti - calIndex(p.start)) / 7) + 1} of ${(calIndex(p.end) - calIndex(p.start) + 1) / 7}${kind ? ` (${kind} week)` : ""}`;
    }
    const next = (this.CAL.phases || []).find((q) => q.start > this.CAL.today);
    return next ? `No phase today · next: <b>${esc(next.name)}</b> from ${esc(niceDate(next.start))}` : "No phases planned (add them on the Plan page)";
  }

  calRows() {
    const M = this.calModel(), cw = Math.floor(M.ti / 7);
    let h = "";
    for (let w = 0; w < M.nw; w++) {
      const W = M.weeks[w];
      let cells = "";
      for (let k = 0; k < 7; k++) {
        const i = w * 7 + k, ds = M.dates[i], d = new Date(ds + "T12:00:00"), dn = d.getDate();
        const mon = dn === 1 || i === 0 ? `<b>${MON[d.getMonth()]}${d.getMonth() === 0 || i === 0 ? " " + d.getFullYear() : ""}</b> ` : "";
        const away = this.CAL.away?.[ds];
        const plans = (away != null ? `<button class="awaytag${AWAY_SEL.has(ds) ? " sel" : ""}" data-away="${ds}" title="You can't ride this day. Click to pick it (Shift+click: a range), then Remove">✈ Away${away ? ` · ${esc(away)}` : ""}</button>` : "") +
          (M.planByDay[ds] || []).filter((k) => away == null || this.CAL.planned[k].type !== "rest").map((k) => sessionModal.calPlan(this.CAL.planned[k], k)).join("") +
          (this.CAL.events || []).filter((e) => e.date === ds).map((e) => `<div class="ev" title="${esc(e.series)}">🏁 ${esc(e.priority)} · ${esc(e.name)}</div>`).join("");
        cells += `<div class="cd${i === M.ti ? " today" : ""}${i > M.ti ? " fut" : ""}${d.getMonth() % 2 ? " alt" : ""}${away != null ? " away" : ""}" data-day="${ds}" data-i="${i}">
        <div class="dn">${mon}${dn}</div>${plans}${(M.byDay[ds] || []).map((a) => sessionModal.calAct(a)).join("")}</div>`;
      }
      const hrs = hhmm;
      const sum = W.act ? `<b>${fmt(W.act)}</b><span>TSS${W.min ? ` · ${hrs(W.min)}` : ""}</span>${W.rest ? `<span>plan ${fmt(W.total)} · ${hrs(W.min + W.pmin)}</span>` : ""}`
        : W.rest ? `<b class="pln">${fmt(W.total)}</b><span>TSS · ${hrs(W.pmin)} planned</span>` : '<span class="muted">–</span>';
      const rf = (this.CAL.ramp_flags || {})[W.start];
      const lim = ((this.CAL.push_weeks || []).includes(W.start) ? `<span class="pushwk" title="${PUSH_WEEK_TIP}">push week</span>` : "") +
        ((this.CAL.hours_limited || []).includes(W.start) ? `<span class="hrlim" title="${HOURS_LIMITED_TIP}">at your hours</span>` : "") +
        (rf ? rampChip(rf[0], rf[1], this.CAL.ramp_range) : "");
      h += `<div class="wk${w === cw ? " cur" : ""}${w === this.calFocus ? " focus" : ""}" data-w="${w}">${cells}<div class="ws" title="Click to focus this week in the chart and the calendar">${sum}${W.target ? `<span>target ${fmt(W.target)}</span>` : ""}${lim}</div></div>`;
    }
    return h;
  }

  html() {
    if (!this.CAL || this.calKey !== this.calKeyNow()) this.loadCal();
    if (!this.CAL) return '<h1>Calendar</h1><div class="loading">Loading…</div>';
    if (!this.CAL.tss) return '<h1>Calendar</h1><div class="card">Couldn\'t load the calendar. Run <code>goodpush run</code> first.</div>';
    return `
  <div class="row"><h1>Calendar</h1>${hint(CAL_HELP)}</div>
  <div class="card calchart">
    <div class="small" style="margin-bottom:6px">${this.phaseNow()}</div>
    <div class="legend"><span><i style="background:linear-gradient(90deg,var(--z3),var(--z5),var(--z7))"></i>Daily TSS, coloured by workout type</span><span><i class="band" style="background:var(--s1)"></i>Weekly TSS</span>
      <span><i style="background:var(--s2)"></i>Running weekly TSS</span><span><i class="dash" style="border-color:var(--s1)"></i>Planned</span>
      <span><i class="dash" style="border-color:var(--s3)"></i>Week target</span></div>
    <div class="tssbox"><div id="tssAxis"></div><div class="tss-scroll" id="tssScroll"></div></div>
    ${seasonBlocks.tools()}
  </div>
  ${sessionModal.proposalCard()}
  <div class="clipbar pop" id="awaybar" role="dialog" aria-label="Days away picked" hidden><span id="awaysel"></span><span class="spacer"></span><button class="btn sm primary" id="awayrm">Remove</button><button class="btn sm" id="awaynone">Cancel</button></div>
  ${sessionMoves.CLIP ? `<div class="clipbar"><span>📋 <b>${esc(sessionMoves.CLIP.title)}</b> copied · click a day to paste it (or Cmd / Ctrl+V over a day); hold Shift to paste on more days</span><span class="spacer"></span><button class="btn sm" id="clipdone">Done</button></div>` : ""}
  <div class="row caltools">
    <button class="btn sm" id="caltoday">Today</button>
    <label class="goto">Go to <input type="date" id="calgoto" min="${this.CAL.start}" max="${addDays(this.CAL.start, this.calModel().n - 1)}" value="${this.calFocus != null ? this.calModel().dates[this.calFocus * 7] : this.CAL.today}"></label>
    <button class="btn sm" id="calaway" title="Days you can't ride (travel, work): the plan works around them">Days away</button>
    <button class="btn sm" id="calclear" title="Your changes from today on that aren't locked go back to the plan">
      Clear my changes</button>
    <span class="spacer"></span>
    <button class="btn sm" id="calnew" title="Build your own workout: drag blocks or write it as text">New workout</button>
  </div>
  <div class="cal${sessionMoves.CLIP ? " pasting" : ""}" id="cal">
    <div class="cal-head">${["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d) => `<div>${d}</div>`).join("")}<div>Week</div></div>
    ${this.calRows()}
  </div>
  <div class="legend" style="margin-top:8px"><span><i class="dash" style="border-color:var(--muted)"></i>Planned session</span>
    <span><i class="band" style="background:var(--muted)"></i>What you did (click to open)</span></div>`;
  }

  drawCalChart() {
    const M = this.calModel(), sc = $("#tssScroll"), ax = $("#tssAxis");
    const H = 232, mt = 40, mb = 22, W = M.nw * this.WEEK_W;
    const top = Math.max(100, ...M.weeks.map((w) => Math.max(w.total, w.target || 0))) * 1.08;
    const yt = niceTicks(0, top, 4);
    while (yt[yt.length - 1] < top) yt.push(yt[yt.length - 1] + yt[1] - yt[0]);
    const ymax = yt[yt.length - 1];
    const fy = (v) => (mt + (1 - v / ymax) * (H - mt - mb)).toFixed(1);
    const c = {s1: css("--s1"), s2: css("--s2"), s3: css("--s3"), grid: css("--grid"), ink: css("--ink"), ink2: css("--ink2"), muted: css("--muted"), accent: css("--accent")};
    let g = "", wk = "", bars = "", ln = "", lab = "";
    for (const t of yt) g += `<line x1="0" x2="${W}" y1="${fy(t)}" y2="${fy(t)}" stroke="${c.grid}"/>`;
    const band = seasonBlocks.band();          // season phases along the top: you edit them there (season.js)
    M.dates.forEach((ds, i) => {
      if (!ds.endsWith("-01")) return;
      const m = Number(ds.slice(5, 7)) - 1, x = i * this.DAY_W;
      g += `<line x1="${x}" x2="${x}" y1="${mt}" y2="${H - mb + 4}" stroke="${c.grid}"/>`;
      lab += `<text x="${x + 4}" y="${H - 6}" font-size="11" fill="${m === 0 ? c.ink : c.ink2}" font-weight="${m === 0 ? 650 : 400}">${MON[m]}${m === 0 ? " " + ds.slice(0, 4) : ""}</text>`;
    });
    M.weeks.forEach((w, j) => {
      const x = j * this.WEEK_W;
      if (w.act > 0) wk += `<rect x="${x + 1.5}" y="${fy(w.act)}" width="${this.WEEK_W - 3}" height="${(fy(0) - fy(w.act)).toFixed(1)}" rx="3" fill="${c.s1}" fill-opacity=".13"/>`;
      if (w.rest > 0) wk += `<rect x="${x + 1.5}" y="${fy(w.total)}" width="${this.WEEK_W - 3}" height="${(fy(w.act) - fy(w.total)).toFixed(1)}" rx="3" fill="none" stroke="${c.s1}" stroke-opacity=".6" stroke-dasharray="4 3"/>`;
      if (w.target) wk += `<line x1="${x + 3}" x2="${x + this.WEEK_W - 3}" y1="${fy(w.target)}" y2="${fy(w.target)}" stroke="${c.s3}" stroke-width="2" stroke-dasharray="5 3"/>`;
      if (w.total > 0) lab += `<text x="${x + this.WEEK_W / 2}" y="${fy(Math.max(w.total, w.target || 0)) - 5}" text-anchor="middle" font-size="11" fill="${c.ink2}">${!w.rest ? Math.round(w.act) : w.act ? `${Math.round(w.act)} → ${Math.round(w.total)}` : `plan ${Math.round(w.total)}`}</text>`;
      // running total through the week: solid while it's what you did, dashed where it's the plan
      const pt = (k, v) => `${x + (k + 1) * this.DAY_W},${fy(v)}`;
      if (w.last >= 0 && w.act > 0) ln += `<polyline points="${x},${fy(0)} ${w.run.slice(0, w.last + 1).map((v, k) => pt(k, v)).join(" ")}" fill="none" stroke="${c.s2}" stroke-width="2" stroke-linejoin="round"/>`;
      if (w.rest > 0) {
        const k0 = Math.max(w.last, 0);
        ln += `<polyline points="${w.last >= 0 ? pt(w.last, w.run[w.last]) : `${x},${fy(0)}`} ${w.prun.slice(k0 + (w.last >= 0 ? 1 : 0)).map((v, k) => pt(k + k0 + (w.last >= 0 ? 1 : 0), v)).join(" ")}" fill="none" stroke="${c.s2}" stroke-width="2" stroke-dasharray="4 3"/>`;
      }
    });
    const kc = {}, kindFill = (k) => (kc[k] ??= css(TYPE_COLOR[k] || "--muted"));   // workout-type colours, as on the calendar
    for (let i = 0; i < M.n; i++) {
      const t = this.CAL.tss[i], p = M.planTss[i], x = i * this.DAY_W + 3;
      if (t > 0) bars += `<rect x="${x}" y="${fy(t)}" width="${this.DAY_W - 6}" height="${(fy(0) - fy(t)).toFixed(1)}" rx="1.5" fill="${kindFill(M.doneKind[i])}"/>`;
      else if (p > 0 && i >= M.ti) bars += `<rect x="${x + .5}" y="${fy(p)}" width="${this.DAY_W - 7}" height="${(fy(0) - fy(p)).toFixed(1)}" rx="1.5" fill="none" stroke="${kindFill(M.planKind[i])}" stroke-dasharray="3 2"/>`;
    }
    const r = Math.max(4, Math.min(7, this.DAY_W / 2 - 1)), dayAt = Object.fromEntries(M.dates.map((ds, i) => [ds, i]));
    for (const e of this.CAL.events || []) {                   // races: a star at the base of the day's bar
      const i = dayAt[e.date];
      if (i == null) continue;
      bars += `<polygon points="${this.starPoints(i * this.DAY_W + this.DAY_W / 2, fy(0) - r - 1, r)}" fill="${RACE_STAR[e.priority] || RACE_STAR.C}"
        stroke="${c.ink}" stroke-opacity=".45" stroke-width=".8" stroke-linejoin="round"/>`;
    }
    const tx = M.ti * this.DAY_W + this.DAY_W / 2;
    const today = M.ti >= 0 ? `<line x1="${tx}" x2="${tx}" y1="${mt - 6}" y2="${H - mb}" stroke="${c.ink}" stroke-opacity=".55" stroke-dasharray="2 3"/><text x="${tx}" y="${mt - 8}" text-anchor="middle" font-size="10.5" fill="${c.ink}">today</text>` : "";
    sc.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="Daily and weekly TSS">${g}
    <rect id="tssWin" x="0" y="${mt - 2}" width="0" height="${H - mt - mb + 4}" rx="5" fill="${c.ink}" fill-opacity=".07" stroke="${c.muted}" stroke-opacity=".7"/>
    <rect id="tssFocus" x="${(this.calFocus ?? 0) * this.WEEK_W}" y="${mt - 2}" width="${this.WEEK_W}" height="${H - mt - mb + 4}" rx="4" fill="${css("--accent")}" fill-opacity=".1" stroke="${css("--accent")}" stroke-width="1.5" visibility="${this.calFocus != null ? "visible" : "hidden"}"/>
    ${band}${wk}${bars}${ln}${today}${lab}
    <rect id="tssHover" x="0" y="${mt}" width="${this.DAY_W}" height="${H - mt - mb}" fill="${c.accent || c.ink}" fill-opacity=".18" visibility="hidden"/></svg>
    <div class="tip" hidden></div>`;
    ax.innerHTML = `<svg width="40" height="${H}">${yt.map((t) => `<text x="34" y="${+fy(t) + 4}" text-anchor="end" font-size="11" fill="${c.ink2}">${t}</text>`).join("")}</svg>`;
  }

  starPoints(cx, cy, r) {    // a five-pointed star's polygon points around (cx, cy), radius r
    return Array.from({length: 10}, (_, k) => {
      const a = -Math.PI / 2 + k * Math.PI / 5, q = k % 2 ? r * 0.45 : r;
      return `${(cx + q * Math.cos(a)).toFixed(1)},${(+cy + q * Math.sin(a)).toFixed(1)}`;
    }).join(" ");
  }

  calTip(ev) {
    const M = this.calModel(), sc = $("#tssScroll"), svg = $("svg", sc), tip = $(".tip", sc), hv = $("#tssHover");
    const x = ev.clientX - svg.getBoundingClientRect().left, i = Math.floor(x / this.DAY_W);
    if (i < 0 || i >= M.n) { tip.hidden = true; this.calMark(null); return; }
    const w = Math.floor(i / 7), k = i % 7, W = M.weeks[w], t = this.CAL.tss[i], p = M.planTss[i], fut = i > M.ti || (i === M.ti && !t);
    const row = (col, name, v) => `<div><i style="background:${col}"></i>${name} <b>${v}</b></div>`;
    const ahead = fut && p != null;            // a day still to come: "planned" once, by the date
    const races = (this.CAL.events || []).filter((e) => e.date === M.dates[i])
      .map((e) => `<div><i style="background:${RACE_STAR[e.priority] || RACE_STAR.C}"></i>${esc(e.priority)} race <b>${esc(e.name)}</b></div>`).join("");
    tip.innerHTML = `<div class="th">${niceDate(M.dates[i], {weekday: "short", day: "numeric", month: "short", year: "numeric"})}${ahead ? ' <span class="muted">· planned</span>' : ""}</div>${races}
    ${t != null ? row(css("--s1"), "TSS", fmt(t)) : ""}${p != null && (fut || !t) ? row("transparent", ahead ? "TSS" : "Planned", fmt(p)) : ""}
    ${row(css("--s2"), "Week so far", fmt(fut ? W.prun[k] : W.run[k]))}${row(css("--s1"), "Week total", fmt(W.act) + (W.rest ? ` → ${fmt(W.total)}` : ""))}
    ${W.target ? row(css("--s3"), "Week target", fmt(W.target)) : ""}
    ${M.phase[i] ? `<div class="muted">${esc(M.phase[i].name)}${W.kind ? ` · ${W.kind} week` : ""}</div>` : ""}`;
    tip.hidden = false;
    this.calMark(i);
    const tw = tip.offsetWidth, px = i * this.DAY_W + this.DAY_W + 8;
    tip.style.left = Math.max(sc.scrollLeft, Math.min(px + tw > sc.scrollLeft + sc.clientWidth ? i * this.DAY_W - tw - 8 : px, sc.scrollLeft + sc.clientWidth - tw)) + "px";
    tip.style.top = "4px";
  }

  // the day under the pointer, in both the chart (a band) and the calendar (the cell); null = none
  calMarked = null;

  calMark(i) {
    const hv = $("#tssHover");
    if (hv) { if (i == null) hv.setAttribute("visibility", "hidden"); else { hv.setAttribute("x", i * this.DAY_W); hv.setAttribute("visibility", "visible"); } }
    this.calMarked?.classList.remove("hl");
    this.calMarked = i == null ? null : document.querySelector(`#cal .cd[data-i="${i}"]`);
    this.calMarked?.classList.add("hl");
  }

  wireCalendar() {
    const sc = $("#tssScroll"), cal = $("#cal");
    if (!sc || !cal || !this.CAL?.tss) return;
    this.drawCalChart();
    seasonBlocks.wire();
    const M = this.calModel(), rows = [...cal.querySelectorAll(".wk")];
    let win = $("#tssWin");
    let head = 0, tops = [], hs = [];
    const measure = () => { head = $(".cal-head", cal).offsetHeight; tops = rows.map((r) => r.offsetTop); hs = rows.map((r) => r.offsetHeight); };
    const yToW = (y) => {               // pixel in the calendar -> position in weeks (fractional)
      let lo = 0, hi = rows.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (tops[mid] <= y) lo = mid; else hi = mid - 1; }
      return lo + Math.min(1, Math.max(0, (y - tops[lo]) / hs[lo]));
    };
    const wToY = (w) => { const i = Math.max(0, Math.min(rows.length - 1, Math.floor(w))); return tops[i] + Math.min(1, Math.max(0, w - i)) * hs[i]; };
    const visible = () => [yToW(cal.scrollTop + head), yToW(cal.scrollTop + cal.clientHeight)];
    const showWin = () => { const [a, b] = visible(); win.setAttribute("x", a * this.WEEK_W); win.setAttribute("width", Math.max(2, (b - a) * this.WEEK_W)); };
    // a scroll we set on one side remembers where it landed, so its scroll event isn't fed back
    const put = (el, prop, v) => { const old = el[prop]; el[prop] = v; if (el[prop] !== old) el._want = el[prop]; };
    const echo = (el, prop) => { const hit = el._want != null && Math.abs(el[prop] - el._want) < 1; el._want = null; return hit; };
    // The window sits in the middle of the chart, sliding to the chart's edge over the last
    // g weeks at either end, so both reach the first / last week together (in weeks:
    // L = chart left edge, t = calendar top, g = half the chart width not covered by the window).
    const geo = () => {
      const [a, b] = visible();
      return {a, g: (sc.clientWidth / this.WEEK_W - (b - a)) / 2, lmax: Math.max(0, (sc.scrollWidth - sc.clientWidth) / this.WEEK_W)};
    };
    const fromCal = () => {
      const {a, g, lmax} = geo();
      const L = g <= 0 ? a - g : a < 2 * g ? a / 2 : a > lmax ? (a - 2 * g + lmax) / 2 : a - g;
      showWin(); put(sc, "scrollLeft", L * this.WEEK_W);
    };
    const fromChart = () => {
      const {g, lmax} = geo(), L = sc.scrollLeft / this.WEEK_W;
      const t = g <= 0 ? L + g : L + g - Math.max(0, g - L) + Math.max(0, g - (lmax - L));
      put(cal, "scrollTop", wToY(t) - head); showWin();
    };
    cal.addEventListener("scroll", () => {
      // a re-render replaces the calendar; a scroll still due on the old one (trackpad momentum) reads 0 there
      if (!cal.isConnected || !cal.clientHeight) return;
      this.calTop = yToW(cal.scrollTop + head);
      const i = Math.min(rows.length - 1, Math.floor(this.calTop) + (tops[Math.floor(this.calTop)] < cal.scrollTop + head ? 1 : 0));
      this.calAnchor = {i, off: tops[i] - cal.scrollTop};
      if (echo(cal, "scrollTop")) showWin(); else fromCal();
    }, {passive: true});
    // the pointer resting on the chart while it scrolls under it (a click's jump): the bar now under it
    let pt = null;
    sc.addEventListener("scroll", () => { if (!echo(sc, "scrollLeft")) fromChart(); if (pt && !drag) this.calTip(pt); }, {passive: true});
    sc.addEventListener("wheel", (e) => {             // mouse wheel over the chart scrolls it sideways
      if (e.ctrlKey || e.metaKey) {                     // zoom around the pointer, as the other charts do
        e.preventDefault();
        const x = e.clientX - sc.getBoundingClientRect().left, at = (sc.scrollLeft + x) / this.DAY_W;
        const w = Math.max(6, Math.min(40, this.DAY_W * (e.deltaY > 0 ? 1 / 1.25 : 1.25)));
        if (w === this.DAY_W) return;
        this.DAY_W = w; this.WEEK_W = 7 * w;
        this.drawCalChart(); win = $("#tssWin");
        put(sc, "scrollLeft", at * this.DAY_W - x); showWin();
        return;
      }
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
      e.preventDefault(); sc.scrollLeft += e.deltaY * (e.deltaMode === 1 ? 16 : 1);
    }, {passive: false});
    const jumpTo = (w) => {
      cal.scrollTo({top: wToY(w + 0.5) - head - (cal.clientHeight - head) / 2, behavior: "smooth"});
    };
    let drag = null;
    // a click (press and release both on the chart) jumps to that week; a mouse drags the chart sideways, a finger
    // scrolls it natively (pointercancel). A release that started elsewhere (e.g. closing a dialog) does nothing.
    sc.addEventListener("pointerdown", (e) => {
      if (e.button) return;
      const r = sc.getBoundingClientRect();
      if (e.clientY >= r.top + sc.clientTop + sc.clientHeight) return;   // on its scrollbar: the browser's own drag
      drag = {x: e.clientX, l: sc.scrollLeft, moved: false, mouse: e.pointerType === "mouse"};
      if (!drag.mouse) return;
      e.preventDefault();                               // no text / image selection flashing while dragging (macOS)
      sc.setPointerCapture(e.pointerId);
    });
    sc.addEventListener("pointercancel", () => { drag = null; sc.classList.remove("drag"); });
    sc.addEventListener("pointermove", (e) => {
      if (drag?.mouse) {
        const dx = e.clientX - drag.x;
        if (Math.abs(dx) > 3) { drag.moved = true; sc.classList.add("drag"); }
        if (drag.moved) { sc.scrollLeft = drag.l - dx; return; }
      }
      pt = {clientX: e.clientX, clientY: e.clientY};
      this.calTip(e);
    });
    sc.addEventListener("pointerup", (e) => {
      const click = drag && !drag.moved;
      drag = null; sc.classList.remove("drag");
      if (click) { const x = e.clientX - $("svg", sc).getBoundingClientRect().left; focusWeek(Math.floor(x / this.WEEK_W)); this.calTip(e); }
    });
    sc.addEventListener("pointerleave", () => { pt = null; if (drag) return; $(".tip", sc).hidden = true; this.calMark(null); });
    const toToday = () => { put(cal, "scrollTop", tops[Math.max(0, Math.floor(M.ti / 7) - 1)] - head); fromCal(); };   // last week on top
    // a week you pick: marked in both, the calendar centred on it (the chart follows); the same week again: off
    const focusWeek = (w, toggle = true, fromInput = false) => {   // fromInput: the date box's own pick, left as is
      if (w < 0 || w >= M.nw) return;
      this.calFocus = toggle && this.calFocus === w ? null : w;
      rows.forEach((r, j) => r.classList.toggle("focus", j === this.calFocus));
      const f = $("#tssFocus");
      f.setAttribute("x", (this.calFocus ?? 0) * this.WEEK_W); f.setAttribute("visibility", this.calFocus != null ? "visible" : "hidden");
      if (this.calFocus == null) return;
      jumpTo(this.calFocus);
      if (!fromInput) $("#calgoto").value = M.dates[this.calFocus * 7];   // (writing it while its picker is open
      // reset the picker: choosing a month in the month / year view never got back to the days)
    };
    $("#caltoday").addEventListener("click", () => focusWeek(Math.floor(M.ti / 7), false));
    $("#calgoto").addEventListener("change", (e) => {
      const i = calIndex(e.target.value);
      if (!e.target.value || i < 0 || i >= M.n) { toast.show("That date isn't in the calendar"); return; }
      focusWeek(Math.floor(i / 7), false, true);
      const cell = cal.querySelector(`.cd[data-i="${i}"]`);
      if (cell) { cell.classList.remove("flash"); void cell.offsetWidth; cell.classList.add("flash"); }
    });
    $("#calaway").addEventListener("click", () => awayDays.awayModal());
    cal.addEventListener("click", (e) => { const ws = e.target.closest(".ws"); if (ws) focusWeek(Number(ws.parentElement.dataset.w)); });
    cal.addEventListener("pointerover", (e) => { const cd = e.target.closest(".cd[data-i]"); if (cd && !sessionMoves.calDrag?.on) this.calMark(Number(cd.dataset.i)); });
    $("#calnew").addEventListener("click", () => builder.open({date: this.CAL.today}));
    $("#calclear").addEventListener("click", () => this.clearMine());
    sessionMoves.wireCalDrag(cal);
    document.querySelectorAll("[data-decide]").forEach((b) => b.addEventListener("click", () => sessionModal.decide(b.dataset.decide, b.dataset.key)));
    document.querySelectorAll(".pr-row[data-goto]").forEach((r) => r.addEventListener("click", (e) => {   // a proposed change: to its day
      if (e.target.closest("button, a")) return;
      const ds = r.dataset.goto.slice(0, 10), cell = cal.querySelector(`.cd[data-day="${ds}"]`);
      cal.scrollIntoView({block: "nearest", behavior: "smooth"});
      jumpTo(Math.floor(calIndex(ds) / 7));
      if (cell) { cell.classList.remove("flash"); void cell.offsetWidth; cell.classList.add("flash"); }
    }));
    cal.addEventListener("pointerover", (e) => { sessionMoves.calHover = e.target; });
    cal.addEventListener("pointerleave", () => { sessionMoves.calHover = null; this.calMark(null); });
    $("#clipdone")?.addEventListener("click", () => sessionMoves.clipEnd());
    $("#awayrm").addEventListener("click", () => awayDays.awayRemove());
    $("#awaynone").addEventListener("click", () => awayDays.awayClear());
    awayDays.awayShow();
    cal.addEventListener("click", (e) => {             // pasting: the day; a day away: pick it; else a planned session: open it
      const at = !sessionMoves.CLIP && e.target.closest(".awaytag[data-away]");
      if (at) { awayDays.awayPick(at.dataset.away, e.shiftKey); return; }
      const cd = sessionMoves.CLIP && !e.defaultPrevented && !e.target.closest("a") && e.target.closest(".cd[data-day]");
      if (cd) { sessionMoves.calPaste(cd.dataset.day, e.shiftKey); return; }
      const lk = !e.defaultPrevented && e.target.closest("[data-lock]");
      if (lk) { sessionModal.toggleLock(lk.dataset.lock); return; }
      const pl = e.target.closest(".pl[data-ds]");
      if (!pl || e.defaultPrevented || e.target.closest("a, button")) return;
      sessionModal.open(pl.dataset.ds);
    });
    const bottom = () => innerHeight - (matchMedia("(max-width:760px)").matches ? parseFloat(css("--nav-h")) || 0 : 0);   // above the phone's tab bar
    const fit = () => {                       // to the bottom of the window with the page scrolled until the phase
      // band on top of the TSS chart is just under the (sticky) top bar
      const bar = document.querySelector(".top")?.offsetHeight || 0, band = $("#tssScroll").getBoundingClientRect().top;
      cal.style.height = Math.max(360, bottom() - bar - (cal.getBoundingClientRect().top - band) - 8) + "px";
    };
    cal.addEventListener("wheel", (e) => {      // scrolling down: the page first, until the calendar is all in view
      const room = cal.getBoundingClientRect().bottom - bottom();
      if (e.deltaY <= 0 || room <= 1 || e.ctrlKey) return;
      e.preventDefault(); scrollBy(0, Math.min(room, e.deltaY * (e.deltaMode === 1 ? 16 : 1)));
    }, {passive: false});
    this.calResize = () => { fit(); measure(); fromCal(); };
    const restore = () => {
      if (!cal.isConnected) return;
      if (!cal.clientHeight) { requestAnimationFrame(restore); return; }   // not laid out yet (page hidden)
      fit(); measure();
      if (this.calAnchor && this.calAnchor.i < rows.length) { put(cal, "scrollTop", tops[this.calAnchor.i] - this.calAnchor.off); fromCal(); }
      else if (this.calTop != null) { put(cal, "scrollTop", wToY(this.calTop) - head); fromCal(); } else toToday();
    };
    restore();
  }
}
export const calendarPage = new CalendarPage();
