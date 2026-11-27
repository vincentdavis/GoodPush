// The workout builder (WorkoutBuilder): blocks on a timeline, or text (planning/builder.py does the parsing).
import {api, S, toast} from "./api.js";
import {esc, fmt, hhmm} from "./util.js";
import {ask} from "./controls.js";
import {adjustRow, niceDate, TYPE_LABEL, zoneVar} from "./widgets.js";
import {calendarPage} from "./calendar/calendar.js";
import {el} from "./charts.js";
import {BlockTimeline} from "./timeline.js";

// workout builder
// Drag blocks onto a timeline (bars: width = time, height = power), drag a bar's top edge for power, its right
// edge for time, drag a bar to move it; or write it as text (planning/builder.py: the server turns text into
// steps and back). Steps as saved: {name, minutes, lo, hi (fraction of FTP), kind, lo_end?, hi_end?, cadence?}
// and repeats {reps, steps}.
const BLOCKS = {
  warmup: ["Warm-up", {name: "Warm up", minutes: 10, lo: .5, hi: .5, lo_end: .7, hi_end: .7, kind: "warmup"}],
  endurance: ["Endurance", {name: "Endurance", minutes: 20, lo: .6, hi: .7, kind: "active"}],
  tempo: ["Tempo", {name: "Tempo", minutes: 15, lo: .8, hi: .85, kind: "active"}],
  sweetspot: ["Sweet spot", {name: "Sweet spot", minutes: 12, lo: .88, hi: .93, kind: "active"}],
  threshold: ["Threshold", {name: "Threshold", minutes: 10, lo: .95, hi: 1, kind: "active"}],
  vo2: ["VO2max set", {reps: 5, steps: [{name: "VO2max", minutes: 3, lo: 1.1, hi: 1.15, kind: "active"}, {name: "Recover", minutes: 3, lo: .5, hi: .5, kind: "rest"}]}],
  micro: ["30/30s", {reps: 10, steps: [{name: "On", minutes: .5, lo: 1.2, hi: 1.3, kind: "active"}, {name: "Off", minutes: .5, lo: .5, hi: .5, kind: "rest"}]}],
  ramp: ["Ramp", {name: "Ramp", minutes: 10, lo: .6, hi: .6, lo_end: .9, hi_end: .9, kind: "active"}],
  recover: ["Recovery", {name: "Recover", minutes: 5, lo: .5, hi: .5, kind: "rest"}],
  cooldown: ["Cool-down", {name: "Cool down", minutes: 10, lo: .45, hi: .5, kind: "cooldown"}],
};
const BLD_TYPES = ["recovery", "endurance", "long", "tempo", "sweetspot", "threshold", "vo2max", "anaerobic", "openers"];
const B_H = 200;                          // the timeline's height (px)
const ADJUST_TIP = "Make the workout a step easier or harder: its main set changes, its shape stays.\n"
  + "• Harder: a rep, a rep, then a set more with a rep fewer; then shorter rests – same power\n"
  + "• Easier: fewer reps, then a set fewer (3x5, 3x4, 3x3, 2x4); then longer recovery, then power\n"
  + "• Edit it yourself and that is the new start";

class WorkoutBuilder {
  BLD = null;

  bFlat(items) { return items.flatMap((x) => x.reps ? Array(x.reps).fill(this.bFlat(x.steps)).flat() : [x]); }

  bMid(s) { return (s.lo + s.hi) / 2; }

  bMidEnd(s) { return s.lo_end != null ? (s.lo_end + s.hi_end) / 2 : this.bMid(s); }

  bDur(m) {                       // minutes -> "1h5m30s"
    let s = Math.round(m * 60); const h = Math.floor(s / 3600); s -= h * 3600; const mi = Math.floor(s / 60); s -= mi * 60;
    return [[h, "h"], [mi, "m"], [s, "s"]].filter(([v]) => v).map(([v, u]) => v + u).join("") || "0s";
  }

  bMin(t) {                       // "1:30", "90s", "10m", "1h30m", "12" (minutes) -> minutes
    t = String(t).trim().toLowerCase();
    let m = t.match(/^(\d+):([0-5]\d)$/); if (m) return +m[1] + m[2] / 60;
    if (/^\d+(\.\d+)?$/.test(t)) return +t;
    m = t.match(/^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?:in)?)?(?:(\d+(?:\.\d+)?)s)?$/);
    return m && (m[1] || m[2] || m[3]) ? (+m[1] || 0) * 60 + (+m[2] || 0) + (+m[3] || 0) / 60 : null;
  }

  bShown(items) { return items.reduce((a, x) => a + (x.reps ? this.bShown(x.steps) : x.minutes), 0); }   // a set's steps once

  bTss(items) { return this.bFlat(items).reduce((a, s) => { const x = this.bMid(s), y = this.bMidEnd(s); return a + s.minutes / 60 * (x * x + x * y + y * y) / 3 * 100; }, 0); }

  bAt(path) {                     // the item at a path [i] or [i, j]; its list and index
    let list = this.BLD.items, x = null;
    for (let d = 0; d < path.length; d++) { x = list[path[d]]; if (d < path.length - 1) list = x.steps; }
    return {x, list, i: path[path.length - 1]};
  }

  open(o = {}) {
    this.BLD = {items: [], sel: null, title: o.title || "", type: o.type || "", typeAuto: !o.type, date: o.date || calendarPage.CAL?.today || "",
           key: o.key || null, wid: o.wid || null, text: "", errors: [], tab: "bars"};
    const el = document.createElement("div");
    el.className = "modal";
    el.innerHTML = `<div class="modal-box wide" role="dialog" aria-modal="true" aria-label="Workout builder">
    <button class="modal-x" aria-label="Close">×</button>
    <div class="row" style="padding-right:40px"><h2 style="margin:0">Workout builder</h2><span class="spacer"></span>
      <div class="seg" id="b-tab"><button data-t="bars" class="on">Blocks</button><button data-t="text">Text</button></div></div>
    <div id="b-body"></div></div>`;
    const close = () => { el.remove(); document.removeEventListener("keydown", key_); this.BLD = null; };
    const key_ = (e) => {
      if (e.key === "Escape") close();
      else if ((e.key === "Delete" || e.key === "Backspace") && this.BLD?.sel && !e.target.closest("input, textarea, select")) { this.bDelete(); }
    };
    el.addEventListener("click", (e) => { if (e.target === el || e.target.closest(".modal-x")) close(); });
    document.addEventListener("keydown", key_);
    el.querySelectorAll("#b-tab button").forEach((b) => b.addEventListener("click", () => {
      this.BLD.tab = b.dataset.t; el.querySelectorAll("#b-tab button").forEach((x) => x.classList.toggle("on", x === b)); this.bRender();
    }));
    document.body.appendChild(el);
    this.BLD.el = el; this.BLD.close = close;
    this.timeline = new BlockTimeline({        // drag in / move / resize (timeline.js); the top edge is power
      items: () => this.BLD.items, at: (p) => this.bAt(p).x, size: (x) => this.bShown([x]),
      scale: () => this.BLD.scale, canvas: () => this.BLD.el.querySelector("#b-canvas"),
      left: () => this.BLD.el.querySelector("#b-canvas").getBoundingClientRect().left,
      resize: (x, m) => this.bResize(x, m), handles: {top: (x) => this.bPowerDrag(x)},
      select: (p) => { this.BLD.sel = p; }, preview: () => { this.bCanvas(); this.bSummary(); },
      done: (changed) => { if (changed) this.bChanged(); else { this.bCanvas(); this.bEditor(); } },
    });
    const start = o.wid ? (S.my_workouts || {})[o.wid]?.steps : o.steps;
    if (o.key) this.bLoadSteps({action: "workout_format", key: o.key});
    else if (start) this.bLoadSteps({action: "workout_format", steps: start});
    else this.bRender();
  }

  async bPost(body) {
    const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
      body: JSON.stringify({ftp: S.plan?.ftp || 0, ...body})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return null; }
    return j;
  }

  async bLoadSteps(body) {
    const j = await this.bPost(body); if (!j || !this.BLD) return;
    this.BLD.items = j.steps; this.BLD.text = j.text; this.BLD.errors = []; this.BLD.adjust = null;
    if (this.BLD.typeAuto && j.type) this.BLD.type = j.type;
    this.bRender();
  }

  bFormatT = 0;

  async bAdjust(d) {               // easier / harder: steps from the workout as it was before the first click
    const a = this.BLD.adjust ||= {base: structuredClone(this.BLD.items), offset: 0};
    const j = await this.bPost({action: "workout_adjust", steps: a.base, offset: a.offset + d});
    if (!j || !this.BLD) return;
    Object.assign(a, {offset: j.offset, easier: j.easier, harder: j.harder});
    this.BLD.items = j.steps; this.BLD.text = j.text; this.BLD.sel = null;
    this.bRender();
  }

  bChanged() {                     // the blocks changed: redraw now, the text a moment later
    this.BLD.adjust = null;        // your edit is the new starting point for easier / harder
    this.bRender();
    clearTimeout(this.bFormatT);
    this.bFormatT = setTimeout(async () => {
      const j = await this.bPost({action: "workout_format", steps: this.BLD.items}); if (!j || !this.BLD) return;
      this.BLD.text = j.text; if (this.BLD.typeAuto && j.type) { this.BLD.type = j.type; const t = this.BLD.el.querySelector("#b-type"); if (t) t.value = j.type; }
    }, 250);
  }

  bParseT = 0;

  bTextInput(t) {                  // the text changed: blocks from the server's parse
    this.BLD.text = t;
    clearTimeout(this.bParseT);
    this.bParseT = setTimeout(async () => {
      const j = await this.bPost({action: "workout_text", text: t}); if (!j || !this.BLD) return;
      this.BLD.errors = j.errors;
      if (!j.errors.length) {
        this.BLD.items = j.steps; this.BLD.adjust = null;
        if (this.BLD.typeAuto && j.type) this.BLD.type = j.type;
      }
      this.bSummary(); this.bErrors();
    }, 350);
  }

  bRender() {
    const box = this.BLD.el.querySelector("#b-body"), ftp = S.plan?.ftp || 0;
    const mine = Object.entries(S.my_workouts || {});
    box.innerHTML = `
    ${this.BLD.tab === "bars" ? `<div class="b-palette">${Object.entries(BLOCKS).map(([k, [n, b]]) =>
        `<span class="b-chip" data-b="${k}"
          style="--c:${zoneVar(b.reps ? this.bMid(b.steps[0]) : Math.max(this.bMid(b), this.bMidEnd(b)))}"
          title="Drag onto the timeline, or double-click to add at the end">${esc(n)}</span>`).join("")}</div>
    <div class="b-canvas" id="b-canvas"></div>
    <div id="b-edit"></div>` : `<textarea id="b-text" class="snip b-text" rows="12" spellcheck="false" placeholder="Warm up 10m 50%>70%&#10;3x&#10;  Threshold 8m 95-100%&#10;  Recover 4m 55%&#10;Cool down 10m 50%">${esc(this.BLD.text)}</textarea>
    <div id="b-errors"></div>
    <details class="small"><summary>How to write it</summary><div class="small muted" style="margin-top:6px">One step per line: a name (optional), a duration
      (90s, 10m, 1h30m, 1:30) and a power: 95% or 90-95% of FTP, a ramp 50%>70%, a zone Z1-Z7, or watts 200W / 200-220W; optionally 90rpm.
      <b>3x</b> repeats the indented lines under it (or the lines up to a blank one); <b>5x 30s 120%, 30s 50%</b> on one line.
      Names starting with "warm" / "cool" make a warm-up / cool-down; an easy step inside a repeat is the recovery. # starts a comment.</div></details>`}
    <div class="b-sum" id="b-sum"></div>
    ${this.BLD.items.length ? adjustRow(this.BLD.adjust?.offset || 0, this.BLD.adjust?.easier ?? true,
      this.BLD.adjust?.harder ?? true, "as built", ADJUST_TIP) : ""}
    <div class="b-fields" style="margin-top:10px">
      <label class="f">Name<input id="b-title" value="${esc(this.BLD.title)}" placeholder="My workout" style="width:220px"></label>
      <label class="f">Type<select id="b-type">${BLD_TYPES.map((t) => `<option value="${t}"${t === this.BLD.type ? " selected" : ""}>${esc(TYPE_LABEL[t] || t)}</option>`).join("")}</select></label>
      <label class="f">Day<input type="date" id="b-date" value="${esc(this.BLD.date)}"></label>
    </div>
    <div class="row" style="margin-top:10px">
      <button class="btn sm primary" id="b-add">${this.BLD.key ? "Replace the session" : "Add to the calendar"}</button>
      ${this.BLD.key ? '<button class="btn sm" id="b-addnew">Add as another session</button>' : ""}
      <button class="btn sm" id="b-save">${this.BLD.wid ? "Update in my workouts" : "Save to my workouts"}</button>
      ${this.BLD.wid ? '<button class="btn sm danger" id="b-del">Delete from my workouts</button>' : ""}
      <span class="spacer"></span>
      ${mine.length ? `<select id="b-load"><option value="">Open one of my workouts…</option>${mine.map(([id, w]) => `<option value="${esc(id)}">${esc(w.title)} · ${hhmm(w.minutes)}</option>`).join("")}</select>` : ""}
    </div>`;
    if (this.BLD.tab === "bars") { this.bCanvas(); this.bEditor(); this.bWirePalette(); }
    else { const t = box.querySelector("#b-text"); t.addEventListener("input", () => this.bTextInput(t.value)); this.bErrors(); }
    this.bSummary();
    box.querySelectorAll("[data-adjust]").forEach((b) =>
      b.addEventListener("click", () => this.bAdjust(Number(b.dataset.adjust))));
    box.querySelector("#b-title").addEventListener("input", (e) => { this.BLD.title = e.target.value; });
    box.querySelector("#b-type").addEventListener("change", (e) => { this.BLD.type = e.target.value; this.BLD.typeAuto = false; });
    box.querySelector("#b-date").addEventListener("change", (e) => { this.BLD.date = e.target.value; });
    box.querySelector("#b-add").addEventListener("click", () => this.bAdd(this.BLD.key));
    box.querySelector("#b-addnew")?.addEventListener("click", () => this.bAdd(null));
    box.querySelector("#b-save").addEventListener("click", () => this.bSave());
    box.querySelector("#b-del")?.addEventListener("click", async () => {
      if (!await ask.confirm({title: `Delete ${this.BLD.title || "this workout"}?`, text: "It goes from your workouts.",
        yes: "Delete", danger: true})) return;
      if (await this.bPost({action: "delete_workout", id: this.BLD.wid})) { delete S.my_workouts[this.BLD.wid]; this.BLD.wid = null; toast.show("Deleted"); this.bRender(); }
    });
    box.querySelector("#b-load")?.addEventListener("change", (e) => {
      const w = S.my_workouts[e.target.value]; if (!w) return;
      Object.assign(this.BLD, {wid: e.target.value, title: w.title, type: w.type, typeAuto: false});
      this.bLoadSteps({action: "workout_format", steps: w.steps});
    });
  }

  bSummary() {
    const el = this.BLD.el.querySelector("#b-sum"); if (!el) return;
    const st = this.bFlat(this.BLD.items), min = st.reduce((a, s) => a + s.minutes, 0), tss = this.bTss(this.BLD.items), ftp = S.plan?.ftp || 0;
    const IF = min ? Math.sqrt(tss / (min / 60) / 100) : 0;
    el.innerHTML = min ? `<b>${hhmm(min)}</b> · ${fmt(tss)} TSS · IF ${fmt(IF, 2)}${ftp ? ` · at FTP ${fmt(ftp)} W` : ""}${this.BLD.type ? ` · ${esc(TYPE_LABEL[this.BLD.type] || this.BLD.type)}` : ""}`
      : '<span class="muted">Drag blocks onto the timeline, or write it on the Text tab.</span>';
  }

  bErrors() {
    const el = this.BLD.el.querySelector("#b-errors"); if (!el) return;
    el.innerHTML = this.BLD.errors.length ? `<ul class="b-errs">${this.BLD.errors.map((e) => `<li>Line ${e.line}: ${esc(e.message)}</li>`).join("")}</ul>` : "";
  }

  bCanvas() {
    const cv = this.BLD.el.querySelector("#b-canvas"), W = cv.clientWidth || 800;
    const total = Math.max(this.bShown(this.BLD.items), 30);
    const maxF = Math.max(1.3, ...this.bFlat(this.BLD.items).map((s) => Math.max(s.hi, s.hi_end || 0) + .05));
    this.BLD.scale = W / total; this.BLD.maxF = maxF;
    let x = 0, html = "";
    const bar = (s, path) => {
      const w = s.minutes * this.BLD.scale, h0 = this.bMid(s) / maxF * B_H, h1 = this.bMidEnd(s) / maxF * B_H, top = Math.max(h0, h1);
      const sel = this.BLD.sel && this.BLD.sel.join() === path.join() ? " sel" : "";
      const clip = s.lo_end != null ? `clip-path:polygon(0 ${(top - h0).toFixed(1)}px,100% ${(top - h1).toFixed(1)}px,100% 100%,0 100%);` : "";
      html += `<div class="b-bar${sel}" data-p="${path.join(".")}" style="left:${x.toFixed(1)}px;width:${Math.max(w - 2, 3).toFixed(1)}px;height:${top.toFixed(1)}px"
      title="${esc(s.name)} · ${this.bDur(s.minutes)} · ${Math.round(this.bMid(s) * 100)}${s.lo_end != null ? `→${Math.round(this.bMidEnd(s) * 100)}` : ""} %">
      <span class="b-fill" style="background:${zoneVar((this.bMid(s) + this.bMidEnd(s)) / 2)};${clip}"></span><i class="b-top" data-h="top"></i><i class="b-right" data-h="right"></i></div>`;
      x += w;
    };
    const draw = (items, path, depth) => items.forEach((it, i) => {       // a set's steps once, under a bracket ×N
      const p = [...path, i];
      if (!it.reps) return bar(it, p);
      const x0 = x;
      draw(it.steps, p, depth + 1);
      const sel = this.BLD.sel && this.BLD.sel.join() === p.join() ? " sel" : "";
      html += `<div class="b-rep${sel}" data-p="${p.join(".")}" style="left:${x0.toFixed(1)}px;width:${(x - x0 - 2).toFixed(1)}px;top:${depth * 18}px">×${it.reps}</div>`;
    });
    draw(this.BLD.items, [], 0);
    const ftpY = B_H - 1 / maxF * B_H;
    cv.innerHTML = `<div class="b-ftp" style="top:${ftpY.toFixed(0)}px"><span>FTP</span></div>${html}`;
    cv.onpointerdown = (e) => this.timeline.pointer(e);
  }

  bEditor() {
    const el = this.BLD.el.querySelector("#b-edit"); if (!el) return;
    if (!this.BLD.sel) { el.innerHTML = '<p class="small muted" style="margin:6px 0 0">Click a bar to change it: drag its top for power, its right edge for time, the bar itself to move it. Delete removes it.</p>'; return; }
    const {x} = this.bAt(this.BLD.sel), inRep = this.BLD.sel.length > 1;
    const pct = (v) => v == null ? "" : Math.round(v * 1000) / 10;
    el.innerHTML = x.reps ? `<div class="b-fields">
      <label class="f">Repeats<input type="number" id="be-reps" min="1" max="50" value="${x.reps}"></label>
      <button class="btn sm" id="be-addstep">+ step in the set</button><button class="btn sm" id="be-unwrap">Undo the set</button>
      <button class="btn sm" data-mv="-1">←</button><button class="btn sm" data-mv="1">→</button>
      <button class="btn sm" id="be-dup">Duplicate</button><button class="btn sm danger" id="be-del">Delete</button></div>`
      : `<div class="b-fields">
      <label class="f">Name<input id="be-name" value="${esc(x.name)}" style="width:150px"></label>
      <label class="f">Time<input id="be-min" value="${this.bDur(x.minutes)}" style="width:80px"></label>
      <label class="f">% FTP<input id="be-lo" type="number" step="1" value="${pct(x.lo)}" style="width:64px"></label>
      <label class="f">to %<input id="be-hi" type="number" step="1" value="${pct(x.hi)}" style="width:64px"></label>
      <label class="f">Ramp to %<input id="be-end" type="number" step="1" value="${pct(x.lo_end != null ? this.bMidEnd(x) : null)}" placeholder="–" style="width:64px"></label>
      <label class="f">rpm<input id="be-cad" type="number" step="5" value="${x.cadence || ""}" placeholder="–" style="width:60px"></label>
      <label class="f">Kind<select id="be-kind">${["warmup", "active", "rest", "cooldown"].map((k) => `<option${k === x.kind ? " selected" : ""}>${k}</option>`).join("")}</select></label></div>
    <div class="row" style="margin-top:6px"><button class="btn sm" data-mv="-1">←</button><button class="btn sm" data-mv="1">→</button>
      <button class="btn sm" id="be-dup">Duplicate</button>${inRep ? "" : '<button class="btn sm" id="be-wrap">Repeat it…</button>'}
      <button class="btn sm danger" id="be-del">Delete</button></div>`;
    const on = (id, f) => el.querySelector(id)?.addEventListener("change", (e) => { f(e.target.value); this.bChanged(); });
    on("#be-reps", (v) => { x.reps = Math.max(1, Math.round(+v || 1)); });
    on("#be-name", (v) => { x.name = v; });
    on("#be-min", (v) => { const m = this.bMin(v); if (m > 0) x.minutes = m; else toast.show("A time like 8m, 30s, 1:30"); });
    on("#be-lo", (v) => { x.lo = +v / 100; if (x.hi < x.lo) x.hi = x.lo; });
    on("#be-hi", (v) => { x.hi = +v / 100; if (x.lo > x.hi) x.lo = x.hi; });
    on("#be-end", (v) => { if (v === "") { delete x.lo_end; delete x.hi_end; } else { const w = (x.hi - x.lo) / 2; x.lo_end = +v / 100 - w; x.hi_end = +v / 100 + w; } });
    on("#be-cad", (v) => { if (v) x.cadence = Math.round(+v); else delete x.cadence; });
    on("#be-kind", (v) => { x.kind = v; });
    el.querySelectorAll("[data-mv]").forEach((b) => b.addEventListener("click", () => {
      const {list, i} = this.bAt(this.BLD.sel), j = i + Number(b.dataset.mv);
      if (j < 0 || j >= list.length) return;
      [list[i], list[j]] = [list[j], list[i]]; this.BLD.sel = [...this.BLD.sel.slice(0, -1), j]; this.bChanged();
    }));
    el.querySelector("#be-dup")?.addEventListener("click", () => {
      const {list, i} = this.bAt(this.BLD.sel); list.splice(i + 1, 0, structuredClone(list[i])); this.BLD.sel = [...this.BLD.sel.slice(0, -1), i + 1]; this.bChanged();
    });
    el.querySelector("#be-del")?.addEventListener("click", () => this.bDelete());
    el.querySelector("#be-wrap")?.addEventListener("click", async () => {
      const n = Number(await ask.prompt({title: "Repeat this step", label: "How many times?", value: "3", yes: "Repeat",
        text: "Each time is followed by an easy recovery half as long.",
        input: {type: "number", min: 1, max: 50, step: 1, required: "", inputmode: "numeric"}}));
      if (!(n >= 1)) return;
      const {list, i} = this.bAt(this.BLD.sel), s = list[i];
      list[i] = {reps: Math.round(n), steps: [s, {name: "Recover", minutes: Math.max(1, s.minutes / 2), lo: .5, hi: .5, kind: "rest"}]};
      this.BLD.sel = [i]; this.bChanged();
    });
    el.querySelector("#be-unwrap")?.addEventListener("click", () => {
      const {list, i} = this.bAt(this.BLD.sel); list.splice(i, 1, ...list[i].steps); this.BLD.sel = null; this.bChanged();
    });
    el.querySelector("#be-addstep")?.addEventListener("click", () => {
      x.steps.push({name: "Recover", minutes: 2, lo: .5, hi: .5, kind: "rest"}); this.bChanged();
    });
  }

  bDelete() {
    const {list, i} = this.bAt(this.BLD.sel); list.splice(i, 1);
    if (this.BLD.sel.length > 1 && !list.length) this.BLD.items.splice(this.BLD.sel[0], 1);     // an emptied set goes too
    this.BLD.sel = null; this.bChanged();
  }

  bWirePalette() {
    this.BLD.el.querySelectorAll(".b-chip").forEach((c) =>
      this.timeline.palette(c, () => structuredClone(BLOCKS[c.dataset.b][1])));
  }

  bResize(x, m) {                  // a step's right edge dragged: 15 s steps under 5 min, else minutes (not a set)
    if (x.reps) return;
    m = Math.max(1 / 6, m);
    x.minutes = m < 5 ? Math.round(m * 4) / 4 : Math.round(m);
  }

  bPowerDrag(x) {                  // a step's top edge dragged: whole % of FTP, a ramp's end along
    if (x.reps) return null;
    const lo0 = x.lo, hi0 = x.hi, le0 = x.lo_end, he0 = x.hi_end;
    return {move: (dx, dy) => {
      const d = Math.round(-dy / B_H * this.BLD.maxF * 100) / 100;
      x.lo = Math.max(0, lo0 + d); x.hi = Math.max(x.lo, hi0 + d);
      if (le0 != null) { x.lo_end = Math.max(0, le0 + d); x.hi_end = Math.max(x.lo_end, he0 + d); }
    }};
  }

  bWorkout() {
    return {type: this.BLD.type || "endurance", title: this.BLD.title.trim() || "My workout", steps: this.BLD.items};
  }

  async bAdd(key) {
    if (!this.BLD.items.length) { toast.show("Add some blocks first"); return; }
    if (!this.BLD.date) { toast.show("Pick a day"); return; }
    const day = this.BLD.date, close = this.BLD.close, w = this.bWorkout();     // before close(): it drops BLD
    if (key && key.slice(0, 10) !== day) { toast.show("A session is replaced on its own day – or add it as another session"); return; }
    close();
    await api.action({action: "add_workout", workout: w, date: day, key: key || undefined},
      `${w.title} ${key ? "replaces the session" : "added to " + niceDate(day)} – updating the plan…`);
  }

  async bSave() {
    if (!this.BLD.items.length) { toast.show("Add some blocks first"); return; }
    const w = this.bWorkout();
    const j = await this.bPost({action: "save_workout", id: this.BLD.wid || undefined, title: w.title, type: w.type, steps: w.steps});
    if (!j) return;
    this.BLD.wid = j.id;
    (S.my_workouts ||= {})[j.id] = {title: w.title, type: w.type, steps: structuredClone(w.steps), minutes: this.bFlat(w.steps).reduce((a, s) => a + s.minutes, 0)};
    toast.show(`Saved to your workouts: ${w.title}`); this.bRender();
  }
}
export const builder = new WorkoutBuilder();
