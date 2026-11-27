// The season editor on the Calendar (SeasonBlocks): the phase band on top of the TSS chart, edited by drag and drop.
//
// Phases are blocks back to back from the season's start (BlockTimeline, as in the workout builder): drag a chip in
// (Base, Build, ...), drag a block along to reorder, its right edge to change its weeks; a Recovery chip dropped onto
// a phase is a recovery week in it, and its recovery weeks drag within it. Tapers come from your races. Edits stay
// here (the lock shows open) until you lock the season: then it is sent whole (action "season_blocks":
// planning/season.py Season.set_blocks) and re-planned once; the server asks first when days you set yourself are in
// weeks that changed. "Auto-plan" rebuilds the season back from your A races (Season.autoplan; with none ahead
// "Lay out a general plan": Season.GENERIC). A new block's
// recovery weeks come from the server (cadence).
import {api, S, toast} from "../api.js";
import {$, addDays, css, esc} from "../util.js";
import {ask} from "../controls.js";
import {hint, niceDate} from "../widgets.js";
import {BlockTimeline} from "../timeline.js";
import {calendarPage} from "./calendar.js";
import {router} from "../main.js";

const KINDS = ["base", "build", "specialty", "maintenance", "recovery"];
const PHASE_COLOR = {base: "--s3", build: "--s4", specialty: "--s2", maintenance: "--muted", recovery: "--s1"};
const NEW_WEEKS = {base: 6, build: 6, specialty: 4, maintenance: 4, recovery: 1};
const BAND_Y = 4, BAND_H = 17;
const SEASON_HELP = "Your season: blocks back to back\n" +
  "• Drag a block in from the chips; double-click one to add it at the end\n" +
  "• Drag a block along to move it, its right edge to change its weeks\n" +
  "• Drop Recovery on a phase: a recovery week in it; drag recovery weeks within it\n" +
  "• Tapers come from your races; click a block to rename, set its hours or delete it\n" +
  "• 🔓 = changes not planned yet: lock 🔒 to plan them (Discard goes back)";
const AUTO_ASK = "Your season is replaced with one planned back from your A races (base, build, specialty, "
  + "recovery). You can drag it around after.";
const GENERIC_TIP = "No A race ahead: 8 weeks of Base, then 4 of Build, from next Monday";
const GENERIC_ASK = "Your season is replaced with a general 12-week plan from next Monday: 8 weeks of Base, then 4 "
  + "of Build, with a recovery week every 4th. You can drag it around after, or add a race and auto-plan from it.";

class SeasonBlocks {
  blocks = null;        // edits not back from the server yet: [{id, type, name, weeks, recovery_weeks, hours}]
  start = null;                   // the season's first Monday while editing
  sel = null;                     // the picked block's index
  forCal = null;                  // the calendar data the edits started from (a reload starts over unless edited)
  dirty = false;                  // edits not sent yet (unlocked): kept across reloads until you lock or discard
  want = null;                    // a phase id to pick when the calendar opens (from the Plan page)

  timeline = new BlockTimeline({
    items: () => this.edit(), at: (p) => this.edit()[p[0]], size: (b) => b.weeks,
    scale: () => calendarPage.WEEK_W, canvas: () => $("#tssScroll"), left: () => this.left(),
    resize: (b, w) => this.resize(b, w), handles: {rec: (b, target) => this.recDrag(b, Number(target.dataset.r))},
    dropped: (b, px) => this.recDrop(b, px), select: (p) => { this.sel = p ? p[0] : null; },
    preview: () => this.redraw(), done: (changed) => { if (changed) this.changed(); this.settle(); },
  });

  phases() {                      // the phases to draw: your edits, else the calendar's
    const C = calendarPage.CAL;
    if (this.forCal !== C) { if (!this.dirty) { this.blocks = null; this.sel = null; } this.forCal = C; }
    if (!this.blocks) return C.phases || [];
    let at = this.start;
    return this.blocks.map((b) => {
      const p = {...b, start: at, end: addDays(at, 7 * b.weeks - 1)};
      at = addDays(at, 7 * b.weeks);
      return p;
    });
  }

  edit() {                        // start editing: a copy of the phases as blocks
    if (!this.blocks) {
      const ph = this.phases();
      this.start = ph[0]?.start || this.nextMonday();
      this.blocks = ph.map((p) => ({id: p.id, type: p.type, name: p.name, weeks: p.weeks, hours: p.hours,
        recovery_weeks: [...(p.recovery_weeks || [])]}));
    }
    return this.blocks;
  }

  nextMonday() {
    const t = calendarPage.CAL.today, wd = (new Date(t + "T12:00:00").getDay() + 6) % 7;
    return addDays(t, wd ? 7 - wd : 0);
  }

  racing() {                      // an A race ahead: Auto-plan plans back from it, else lays out the general block
    const t = calendarPage.CAL.today;
    return (S.season?.events || []).some((e) => e.priority === "A" && e.date >= t);
  }

  left() {                        // client x of the season's first day on the chart
    const svg = $("#tssScroll svg"), start = this.start || this.phases()[0]?.start || this.nextMonday();
    return svg.getBoundingClientRect().left + calendarPage.dayIndex(start) * calendarPage.DAY_W;
  }

  resize(b, w) {                  // a block's weeks; recovery weeks past its end go (a new block's stay to come)
    b.weeks = Math.max(1, Math.min(52, Math.round(w)));
    if (b.recovery_weeks) b.recovery_weeks = b.recovery_weeks.filter((k) => k < b.weeks);
  }

  recDrag(b, k0) {                // a recovery week dragged within its phase
    let cur = k0;
    return {move: (dx) => {
      const k = Math.max(0, Math.min(b.weeks - 1, Math.round(k0 + dx / calendarPage.WEEK_W)));
      if (k === cur || b.recovery_weeks.includes(k)) return;
      b.recovery_weeks = [...b.recovery_weeks.filter((x) => x !== cur), k].sort((x, y) => x - y);
      cur = k;
    }};
  }

  recDrop(b, px) {                // a Recovery chip dropped onto a phase: a recovery week there
    if (b.type !== "recovery") return false;
    const w = Math.floor(px / calendarPage.WEEK_W);
    let at = 0;
    for (const [i, x] of this.edit().entries()) {
      if (w >= at && w < at + x.weeks) {
        if (x.type === "recovery") return false;
        const recs = x.recovery_weeks || [];
        if (!recs.includes(w - at)) x.recovery_weeks = [...recs, w - at].sort((p, q) => p - q);
        this.sel = i;
        return true;
      }
      at += x.weeks;
    }
    return false;
  }

  marks(p) {                      // a phase's recovery weeks (draggable) and taper weeks (from your races)
    const WW = calendarPage.WEEK_W, ink = css("--ink"), ink2 = css("--ink2");
    const label = (k, t) => k ? `<text x="${k * WW + WW / 2}" y="12" text-anchor="middle" font-size="10"
      fill="${ink2}" pointer-events="none">${t}</text>` : "";
    let out = "";
    for (const k of p.recovery_weeks || []) {
      out += `<rect data-h="rec" data-r="${k}" class="sb-rec" x="${k * WW + 1}" width="${WW - 2}" height="${BAND_H}"
        fill="${ink}" fill-opacity=".14"><title>Recovery week · drag it within the phase</title></rect>`;
      out += label(k, "rec");
    }
    for (let k = 0; k < p.weeks; k++) {
      if (calendarPage.CAL.week_kind?.[addDays(p.start, 7 * k)] !== "taper") continue;
      out += `<rect x="${k * WW + 1}" width="${WW - 2}" height="${BAND_H}" fill="${ink}" fill-opacity=".07"
        pointer-events="none"/>${label(k, "taper")}`;
    }
    return out;
  }

  band() {                        // the phase band (SVG), one group per block
    const M = calendarPage.calModel(), DW = calendarPage.DAY_W;
    return `<g id="seasonBand">${this.phases().map((p, j) => {
      const i0 = calendarPage.dayIndex(p.start), i1 = calendarPage.dayIndex(p.end);
      if (i1 < 0 || i0 >= M.n) return "";
      const x = i0 * DW, w = (i1 - i0 + 1) * DW, now = M.ti >= i0 && M.ti <= i1, sel = j === this.sel;
      const name = p.name.length * 6.4 + 12 > w ? p.name.split(/[ /]/)[0] : p.name;   // short name when narrow
      const ring = sel ? ` stroke="${css("--accent")}" stroke-width="2"` : "";
      return `<g data-p="${j}" class="sb" transform="translate(${x + 1},${BAND_Y})">
        <title>${esc(p.name)}: ${esc(niceDate(p.start))} – ${esc(niceDate(p.end))} · drag to move</title>
        <rect width="${Math.max(w - 2, 1)}" height="${BAND_H}" rx="4" fill="${css(PHASE_COLOR[p.type] || "--muted")}"
          fill-opacity="${now || sel ? .45 : .2}"${ring}/>${this.marks(p)}
        <text x="6" y="12.5" font-size="11" fill="${css("--ink")}" font-weight="${now ? 650 : 400}"
          pointer-events="none">${esc(name)}</text>
        <rect data-h="right" class="sb-edge" x="${Math.max(w - 8, 0)}" width="7" height="${BAND_H}" fill="transparent">
          <title>Drag to change its weeks</title></rect></g>`;
    }).join("")}</g>`;
  }

  redraw() { const g = $("#seasonBand"); if (g) g.outerHTML = this.band(); }

  settle() {                      // after an edit: the calendar grows to the season's end (a block appended past it)
    const end = this.phases().at(-1)?.end;
    if (end && calendarPage.calReach(end)) router.render(); else { this.redraw(); this.editor(); }
  }

  label(k) { return S.season_meta?.types?.[k]?.label || k; }

  tools() {                       // the chips and Auto-plan (under the chart), the picked block's settings
    const chips = KINDS.map((k) => `<span class="b-chip" data-season="${k}" style="--c:var(${PHASE_COLOR[k]})"
      title="Drag onto the chart, or double-click to add at the end">${esc(this.label(k).split(" /")[0])}</span>`)
      .join("");
    return `<div class="row season-tools"><span class="small">Season</span>${hint(SEASON_HELP)}${chips}
      <button class="btn sm" id="sbLock"></button>
      <button class="btn sm" id="sbDiscard" title="Back to the season as planned" hidden>Discard</button>
      <span class="spacer"></span>
      <a class="small" href="#plan" title="Your races and the plan's weeks">Races on the Plan page</a>
      ${this.racing() ? `<button class="btn sm" id="sbAuto" title="Plan the whole season back from your A races">
        Auto-plan from my races</button>` : `<button class="btn sm" id="sbAuto" title="${esc(GENERIC_TIP)}">
        Lay out a general plan</button>`}
    </div><div id="seasonEdit"></div>`;
  }

  editor() {                      // the picked block: name, type, weeks, hours, its recovery weeks, delete
    const box = $("#seasonEdit"); if (!box) return;
    const b = this.sel != null ? this.edit()[this.sel] : null;
    if (!b) { box.innerHTML = ""; return; }
    const p = this.phases()[this.sel], recs = b.recovery_weeks || [];
    const types = KINDS.map((k) =>
      `<option value="${k}"${k === b.type ? " selected" : ""}>${esc(this.label(k))}</option>`);
    const start = this.sel === 0
      ? `<label class="f">Season starts<input id="sbStart" type="date" value="${esc(this.start)}"></label>` : "";
    const dates = `${esc(niceDate(p.start))} – ${esc(niceDate(p.end))}${recs.length ? " · recovery weeks:" : ""}`;
    const recButtons = recs.map((k) =>
      `<button class="btn sm" data-rec="${k}" title="Take this recovery week out">week ${k + 1} ×</button>`).join("");
    box.innerHTML = `<div class="b-fields season-edit">
      <label class="f">Name<input id="sbName" value="${esc(b.name || "")}" style="width:150px"></label>
      <label class="f">Type<select id="sbType">${types.join("")}</select></label>
      <label class="f">Weeks<input id="sbWeeks" type="number" min="1" max="52" value="${b.weeks}"
        style="width:64px"></label>
      <label class="f">Hours / week<input id="sbHours" type="number" min="0" max="40" step=".5" value="${b.hours ?? ""}"
        placeholder="usual" style="width:80px"></label>
      ${start}<span class="small muted">${dates}</span>${recButtons}
      <span class="spacer"></span><button class="btn sm danger" id="sbDel">Delete block</button></div>`;
    const on = (id, f) => $(id, box)?.addEventListener("change", (e) => {
      f(e.target.value); this.changed(); this.settle();
    });
    on("#sbName", (v) => { b.name = v; });
    on("#sbType", (v) => { b.type = v; b.name = ""; if (v === "recovery") b.recovery_weeks = []; });
    on("#sbWeeks", (v) => this.resize(b, Number(v) || b.weeks));
    on("#sbHours", (v) => { b.hours = v === "" ? null : Number(v); });
    on("#sbStart", (v) => { if (v) this.start = v; });
    box.querySelectorAll("[data-rec]").forEach((x) => x.addEventListener("click", () => {
      b.recovery_weeks = recs.filter((k) => k !== Number(x.dataset.rec));
      this.redraw(); this.editor(); this.changed();
    }));
    $("#sbDel", box).addEventListener("click", () => {
      this.edit().splice(this.sel, 1); this.sel = null; this.redraw(); this.editor(); this.changed();
    });
  }

  wire() {                        // after the chart is drawn: the band's pointer, the chips, Auto-plan
    const sc = $("#tssScroll"); if (!sc) return;
    sc.addEventListener("pointerdown", (e) => {     // before the chart's own drag-to-scroll
      if (e.button || !e.target.closest("#seasonBand")) return;
      e.stopImmediatePropagation(); this.timeline.pointer(e);
    }, {capture: true});
    document.querySelectorAll("[data-season]").forEach((c) => {
      const type = c.dataset.season;
      this.timeline.palette(c, () => ({id: null, type, name: "", weeks: NEW_WEEKS[type], hours: null,
        recovery_weeks: null}));
    });
    $("#sbAuto")?.addEventListener("click", async () => {
      const racing = this.racing();
      if (!await ask.confirm({title: racing ? "Plan your season from your races?" : "Lay out a general plan?",
        text: racing ? AUTO_ASK : GENERIC_ASK, yes: "Replace my season", danger: true})) return;
      this.discard();
      api.action({action: "autoplan_season"},
        `${racing ? "Season planned from your races" : "General plan laid out"} – updating the plan…`);
    });
    $("#sbLock").addEventListener("click", () => { if (this.dirty) this.save(); });
    $("#sbDiscard").addEventListener("click", () => this.discard());
    this.pickWanted();
    this.lockState(); this.editor();
  }

  pickWanted() {                  // the phase the Plan page asked for: picked, the calendar scrolled to its start
    if (!this.want) return;
    const ph = this.phases(), j = ph.findIndex((p) => p.id === this.want);
    this.want = null;
    if (j < 0) return;
    this.sel = j; this.redraw();
    calendarPage.calTop = Math.max(0, Math.floor(calendarPage.dayIndex(ph[j].start) / 7));
  }

  changed() { this.dirty = true; this.lockState(); }

  lockState() {                   // the lock: closed = the plan follows the season, open = edits not planned yet
    const b = $("#sbLock"), d = $("#sbDiscard"); if (!b) return;
    b.textContent = this.dirty ? "🔓 Lock and plan" : "🔒 Planned";
    b.title = this.dirty ? "Your season changes aren't planned yet: lock them to re-plan"
      : "The plan follows this season · change a block to unlock it";
    b.classList.toggle("primary", this.dirty); b.disabled = !this.dirty; d.hidden = !this.dirty;
  }

  discard() {                     // back to the season as planned
    this.dirty = false; this.blocks = null; this.sel = null; this.redraw(); this.editor(); this.lockState();
  }

  async save() {                  // the whole season to the server; the re-plan brings it back
    if (!this.blocks) return;
    if (!this.blocks.length && !await ask.confirm({title: "Remove every block?", yes: "Remove them all", danger: true,
      text: "Nothing is planned without a season."})) return;
    const body = {action: "season_blocks", start: this.start, blocks: this.blocks};
    this.dirty = false; this.lockState();         // the re-plan's reload brings the season back from the server
    api.action(body, "Season locked – updating the plan…")
      .then((ok) => {
        if (ok) return;
        this.dirty = true; this.lockState(); toast.show("Season not changed: still unlocked");
      });
  }
}

export const seasonBlocks = new SeasonBlocks();
