// A day's session in the calendar: details, proposals, accept / decline (SessionModal).
import {api, S, toast} from "../api.js";
import {esc, fmt, hhmm, mmss} from "../util.js";
import {ask} from "../controls.js";
import {adjustRow, niceDate, profileSVG, rideable, rideButton, rideIt, stepsList, today, TYPE_LABEL}
  from "../widgets.js";
import {router} from "../main.js";
import {markers} from "../markers.js";
import {actKind, calendarPage, calGone, REP_RESULT, typeColor} from "./calendar.js";
import {sessionMoves} from "./moves.js";
import {el} from "../charts.js";
import {builder} from "../builder.js";

const RESULT = {hit: ["done", "on target"], partial: ["amber", "partly"], over: ["amber", "above target"], low: ["missed", "fell short"],
  short: ["amber", "short"],
  other: ["amber", "did other"], easy: ["missed", "rode easy"], none: ["missed", "missed"], done: ["done", "done"]};
const ADJUST_TIP = (f) => `Too hard or too easy? Moves every ${f} session from now on.\n`
  + "• Harder: a rep, a rep, then a set more with a rep fewer (3x3, 3x4, 3x5, 4x4);"
  + " then shorter rests – same power\n"
  + "• Easier: fewer reps, then a set fewer (3x5, 3x4, 3x3, 2x4); then longer recovery (30/15, 30/30, 30/45), then power\n"
  + "• Your level stays as it is; the session keeps its day";

// Easier / harder in the open session window: each click shows the session as it would be (nothing saved);
// closing the window saves the steps once (one re-plan).
class SessionTune {
  constructor(box, key, p, s) { Object.assign(this, {box, key, p, s, step: 0}); }

  meta(x) {
    return `${hhmm(x.minutes)}${x.tss != null ? ` · ${fmt(x.tss)} TSS` : ""} · FTP ${fmt(S.plan?.ftp)} W`;
  }

  row(a) {
    return adjustRow(a.offset, a.easier, a.harder, "as planned", ADJUST_TIP(a.format));
  }

  async click(d) {
    const j = await api.query({action: "adjust_preview", key: this.key, step: this.step + d, ftp: S.plan?.ftp || 0});
    if (!j || !this.box.isConnected) return;
    this.step = j.adjust.offset - this.p.adjust.offset;
    const x = {...this.s, ...j};
    this.box.querySelector(".sess-title").textContent = j.title;
    this.box.querySelector(".meta").textContent = this.meta(x);
    this.box.querySelector(".sm-wk").innerHTML = profileSVG(x) + stepsList(x);
    this.box.querySelector(".sm-adj").innerHTML = this.row(j.adjust) + (this.step
      ? '<div class="small muted">Saved when you close this window</div>' : "");
  }

  save() {                                   // on close: the steps you clicked, as one change
    if (!this.step) return;
    const n = Math.abs(this.step), way = this.step < 0 ? "easier" : "harder";
    calendarPage.calPending(this.key, "updating");
    api.action({action: "adjust_session", key: this.key, step: this.step},
      `${this.p.adjust.format}: ${n} step${n > 1 ? "s" : ""} ${way} from now on – updating the plan…`)
      .then((ok) => ok || calendarPage.calPending(this.key, null));
  }
}

class SessionModal {
  repTable(reps) {                  // a scored session rep by rep: planned vs what you rode
    if (!reps?.length) return "";
    const rows = reps.map((r, i) => { const [cls, txt] = REP_RESULT[r.result] || ["", r.result];
      return `<tr><td class="n">${i + 1}</td><td class="n">${mmss(r.planned_s)} @ ${r.lo}–${r.hi} W</td>
      <td class="n">${r.watts != null ? `${mmss(r.ride_s)} @ <b>${r.watts}</b> W <span class="muted">at ${mmss(r.start)}</span>` : "–"}</td>
      <td class="n">${r.watts != null ? mmss(r.on_s) : ""}</td><td>${markers.chipFor(cls, txt)}</td></tr>`; }).join("");
    return `<div class="tw" style="margin-top:8px"><table class="small"><thead><tr><th class="n">Rep</th><th class="n">Planned</th><th class="n">You rode</th><th class="n">On target</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  calScore(p) {                     // how the session went (time on target from the ride's power)
    const c = p.compliance;
    if (!c || !RESULT[c.result] || (p.manual && !p.ride)) return "";
    const [cls, txt] = RESULT[c.result];
    const pct = c.ratio != null && c.result !== "none" ? ` ${Math.round(c.ratio * 100)}%` : "";
    const tip = c.minutes != null ? `rode ${hhmm(c.minutes)} of ${hhmm(p.minutes)}` + (c.ride_workout ? ` (${c.ride_workout})` : "") :
      c.ratio != null ? `${hhmm(c.on_s / 60)} of ${hhmm(c.planned_s / 60)} on target` +
      (c.avg_pct ? `, efforts at ${Math.round(c.avg_pct * 100)}% of the target middle` : "") : (c.ride_workout ? `rode: ${c.ride_workout}` : "");
    return ` <span class="chip ${cls}" title="${esc(tip)}">${txt}${pct}</span>`;
  }

  calProposed(ds) {
    const c = (calendarPage.CAL.proposal || []).find((x) => x.kind === "session" && x.date === ds);
    return c ? `<div class="prop" title="${esc(c.why)}">↻ proposed: ${esc(c.to.title)} · ${hhmm(c.to.minutes)} · ${fmt(c.to.tss)} TSS</div>` : "";
  }

  // clicking a planned session opens it: the workout, why, how it went, files and buttons
  open(key) {
    const p = calendarPage.CAL.planned[key], ds = key.slice(0, 10);
    if (!p) return;
    const M = calendarPage.calModel(), r = p.ride && M.byId[p.ride], c = p.compliance || {};
    const prop = (calendarPage.CAL.proposal || []).find((x) => x.kind === "session" && x.date === key);
    const s = {...p, date: ds};
    const past = ds < calendarPage.CAL.today, today = ds === calendarPage.CAL.today;
    let went = "";
    if (c.result && RESULT[c.result]) {
      const [cls, txt] = RESULT[c.result];
      const detail = c.minutes != null ? `rode ${hhmm(c.minutes)} of ${hhmm(p.minutes)}`
        : c.ratio != null ? `${hhmm(c.on_s / 60)} of ${hhmm(c.planned_s / 60)} on target${c.avg_pct ? `, efforts at ${Math.round(c.avg_pct * 100)} % of the target middle` : ""}` : "";
      const ride = c.ride && M.byId[c.ride];
      went = `<div class="card sm"><div class="row">${markers.chipFor(cls, txt)}<span class="small">${esc(detail)}</span></div>
      ${ride ? `<div class="small" style="margin-top:6px">Ride: <a href="#ride/${encodeURIComponent(ride.id)}">${esc(ride.name || "ride")}</a> · ${esc(TYPE_LABEL[ride.workout] || ride.workout || "")} · ${hhmm(ride.min)} · ${fmt(ride.tss)} TSS${p.ride ? " · linked by you" : ""}</div>` : ""}
      ${this.repTable(c.reps)}</div>`;
    } else if (r) {
      went = `<div class="card sm small">Linked: <a href="#ride/${encodeURIComponent(r.id)}">${esc(r.name || "ride")}</a> – scoring it…</div>`;
    }
    const files = [p.fit ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(p.fit)}" download>Workout (.fit)</a>` : "",
      p.zwo ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(p.zwo)}" download>Zwift / Incyclist (.zwo)</a>` : "",
      p.icu ? `<a class="btn sm" href="/files/workouts/${encodeURIComponent(p.icu)}" download title="Paste into an intervals.icu workout">intervals.icu (.txt)</a>` : ""].join("");
    const future = !past;
    const edit = !future ? "" : `<div class="edit-row">
    ${p.type !== "rest" ? `<label class="f">Duration (min)<input type="number" id="sm-min" min="20" max="480" step="5" value="${esc(Math.round(p.minutes || 0))}"></label>
      <button class="btn sm" id="sm-setmin">Set duration</button>` : ""}
    ${p.type !== "rest" ? `<label class="f">Add to day<input type="date" id="sm-day" min="${calendarPage.CAL.today}" value="${ds}"></label>
      <button class="btn sm" id="sm-move">Move there</button>` : ""}
    </div>
    <div class="row" style="margin-top:8px">
      ${p.type !== "rest" ? '<button class="btn sm danger" id="sm-del">Delete</button>' : ""}
      ${p.type !== "rest" ? '<button class="btn sm" id="sm-build">Edit in the builder</button>' : '<button class="btn sm" id="sm-build">Build a workout for this day</button>'}
      ${p.custom && key === ds ? `<button class="btn sm" id="sm-reset">Back to the plan's version</button>` : ""}</div>`;
    const marks = p.type === "rest" || !(past || today) ? "" : `
    <button class="btn sm" data-do="done">${p.status === "done" && !p.ride ? "✓ Done (undo)" : "Mark done"}</button>
    <button class="btn sm" data-do="skipped">${p.status === "skipped" ? "Skipped (undo)" : "Skipped"}</button>
    ${p.ride || p.compliance?.ride ? '<button class="btn sm" data-do="unlink">Unlink ride</button>' : ""}`;
    const el = document.createElement("div");
    const tune = new SessionTune(el, key, p, s);
    el.className = "modal";
    el.innerHTML = `<div class="modal-box" role="dialog" aria-modal="true" aria-label="${esc(p.title)}">
    <button class="modal-x" aria-label="Close">×</button>
    <div class="small muted">${esc(niceDate(ds, {weekday: "long", day: "numeric", month: "long"}))}${today ? " · today" : ""}</div>
    <div class="sess-head"><div class="sess-title">${esc(p.type === "rest" ? "Rest day" : p.title)}</div>${p.type !== "rest" ? `<span class="chip">${esc(TYPE_LABEL[p.type] || p.type)}</span>` : ""}</div>
    ${p.type !== "rest" ? `<div class="meta">${tune.meta(p)}</div>` : ""}
    <div class="sm-wk">${p.steps?.length ? profileSVG(s) : ""}
    ${p.blocks?.length ? stepsList(s) : p.workout ? `<p class="small">${esc(p.workout)}</p>` : ""}</div>
    ${p.why ? `<div class="why">${esc(p.why)}</div>` : ""}${p.note ? `<div class="why">${esc(p.note)}</div>` : ""}
    ${p.adjusted ? `<div class="adj">⚠ ${esc(p.adjusted)}</div>` : ""}
    ${p.adjust && !past ? `<div class="sm-adj">${tune.row(p.adjust)}</div>` : ""}
    ${went}
    ${prop ? `<div class="card proposal sm"><b>Proposed instead:</b> ${this.sessionLine(prop.to, prop.from)}
      <div class="small muted">${esc(prop.why)}</div>
      <div class="row" style="margin-top:8px"><button class="btn sm primary" data-decide="accept" data-key="${key}">Accept</button><button class="btn sm" data-decide="reject" data-key="${key}">Reject</button></div></div>` : ""}
    ${files || marks || p.type !== "rest" ? `<div class="row" style="margin-top:12px">${files}${marks}${p.type !== "rest" ? '<button class="btn sm" id="sm-copy" title="Then click a day to paste it (Shift+click: more days)">Copy</button>' : ""}${rideable(p, ds) ? rideButton(key) : ""}</div>` : ""}
    ${edit ? `<details class="edit"${p.custom ? " open" : ""}><summary class="small">Change this session</summary>${edit}</details>` : ""}
    ${p.locked ? '<p class="small muted" style="margin:10px 0 0">🔒 Locked: the plan keeps it exactly as it is.</p>'
      : p.custom ? `<p class="small muted" style="margin:10px 0 0">You changed this day (blue 🔓): the plan starts
        from it but may change it. Click the lock on the calendar to keep it as it is.</p>` : ""}
  </div>`;
    const close = () => { el.remove(); document.removeEventListener("keydown", esc_); tune.save(); };
    const esc_ = (e) => { if (e.key === "Escape") close(); };
    el.addEventListener("click", (e) => { if (e.target === el || e.target.closest(".modal-x")) close(); });
    document.addEventListener("keydown", esc_);
    el.querySelectorAll("[data-decide]").forEach((b) => b.addEventListener("click", () => { close(); this.decide(b.dataset.decide, b.dataset.key); }));
    el.querySelectorAll("[data-do]").forEach((b) => b.addEventListener("click", async () => {
      close();
      const m = b.dataset.do;
      if (m === "unlink") return sessionMoves.linkRide(ds, null);
      const undo = (m === "done" && p.status === "done" && !p.ride) || (m === "skipped" && p.status === "skipped");
      await api.action({action: "mark", date: key, value: undo ? "" : m}, undo ? "Undone" : m === "done" ? "Nice – marked done" : "Marked skipped");
    }));
    el.querySelector("#sm-setmin")?.addEventListener("click", () => {
      const v = Number(el.querySelector("#sm-min").value);
      if (!v) { toast.show("Enter the minutes"); return; }
      close(); calendarPage.calPending(key, hhmm(v), {minutes: v, tss: p.tss != null && p.minutes ? Math.round(p.tss * v / p.minutes) : p.tss});
      api.action({action: "edit_session", key, minutes: v}, `${p.title}: ${hhmm(v)} – updating the plan…`).then((ok) => ok || calendarPage.calPending(key, null));
    });
    el.addEventListener("click", (e) => {        // redrawn on every click: listened to on the window
      const b = e.target.closest("[data-adjust]");
      if (b) tune.click(Number(b.dataset.adjust));
    });
    el.querySelector("#sm-copy")?.addEventListener("click", () => { close(); sessionMoves.calCopy(key); });
    el.querySelector("#sm-build")?.addEventListener("click", () => {
      close(); builder.open(p.type === "rest" ? {date: ds} : {date: ds, key, title: p.title, type: p.type});
    });
    el.querySelector("[data-ride-key]")?.addEventListener("click", () => { close(); rideIt(key); });
    el.querySelector("#sm-move")?.addEventListener("click", () => {
      const day = el.querySelector("#sm-day").value;
      if (!day || day === ds) { toast.show("Pick another day"); return; }
      close(); sessionMoves.moveSession(key, day);
    });
    el.querySelector("#sm-del")?.addEventListener("click", () => {
      close(); calendarPage.calPending(key, "deleting", calGone);
      api.action({action: "delete_session", key}, `${p.title} deleted – updating the plan…`).then((ok) => ok || calendarPage.calPending(key, null));
    });
    el.querySelector("#sm-reset")?.addEventListener("click", () => {
      close(); calendarPage.calPending(key, "updating");
      api.action({action: "edit_session", key, reset: true}, "Back to the plan's version – updating…").then((ok) => ok || calendarPage.calPending(key, null));
    });
    document.body.appendChild(el);
    el.querySelector(".modal-x").focus();
  }

  sessionLine(s, other) {                     // name · time · TSS; the parts that differ from `other` in bold
    const part = (v, o) => (v === o ? esc(v) : `<b>${esc(v)}</b>`);
    const name = (x) => x.title || x.type || "";
    const tss = (x) => (x.tss == null ? "" : `${fmt(x.tss)} TSS`);
    return [part(name(s), name(other)), part(hhmm(s.minutes), hhmm(other.minutes)), part(tss(s), tss(other))]
      .filter((x) => x).join(" · ");
  }

  proposalCard() {
    const P = calendarPage.CAL.proposal || [];
    if (!P.length) return "";
    const row = (c) => {
      const key = c.kind === "ftp" ? "ftp" : c.date;
      const what = c.kind === "ftp"
        ? `<div class="pr-what"><b>FTP</b> <s>${fmt(c.from)} W</s> → <b>${fmt(c.to)} W</b></div>`
        : `<div class="pr-what"><b>${esc(niceDate(c.date.slice(0, 10)))}</b>
          <div class="pr-was">Now: ${this.sessionLine(c.from, c.to)}</div>
          <div class="pr-new">Proposed: ${this.sessionLine(c.to, c.from)}</div></div>`;
      return `<div class="pr-row"${c.kind === "ftp" ? "" : ` data-goto="${esc(c.date.slice(0, 10))}" title="Show it in the calendar"`}>${what}<div class="small muted">${esc(c.why)}</div>
      <div class="pr-btns"><button class="btn sm primary" data-decide="accept" data-key="${esc(key)}">Accept</button>
        <button class="btn sm" data-decide="reject" data-key="${esc(key)}">Reject</button></div></div>`;
    };
    return `<div class="card proposal">
    <div class="row"><h3 style="margin:0">Plan update: ${P.length} proposed change${P.length > 1 ? "s" : ""}</h3><span class="spacer"></span>
      ${P.length > 1 ? `<button class="btn sm primary" data-decide="accept" data-key="all">Accept all</button><button class="btn sm" data-decide="reject" data-key="all">Reject all</button>` : ""}</div>
    <p class="small muted" style="margin:4px 0 8px">From how your recent sessions went, or a session of your own the plan would now set differently. Until you accept, your plan (and its workout files) stays as it is.</p>
    ${P.map(row).join("")}</div>`;
  }

  async decide(decision, key) {
    const keys = key === "all" ? null : [key];
    calendarPage.CAL.proposal = (calendarPage.CAL.proposal || []).filter((c) => keys && !keys.includes(c.kind === "ftp" ? "ftp" : c.date));
    router.render();                                          // gone from the list straight away
    await api.action({action: "plan_decide", decision, dates: keys},
      decision === "accept" ? "Accepted – updating your plan…" : "Rejected – keeping your plan");
  }

  // a day's session from today on, one button through its states: the plan's 🔓 -> 🔒 -> 🔓; your change (blue 🔓)
  // -> 🔒 -> back to the plan's version. A day with a ride (today) shows it only while locked, so you can unlock it
  lockButton(p, key) {
    if (key.length !== 10 || key < calendarPage.CAL.today || ((p.ride || p.compliance?.ride) && !p.locked)) return "";
    const [tip, label] = p.locked && p.custom ? ["Locked, your version: the plan keeps it exactly as it is. "
      + "Click to go back to the plan's version", "Back to the plan's version"]
      : p.locked ? ["Locked: the plan keeps it exactly as it is. Click to unlock", "Unlock this day"]
      : p.custom ? ["Your change: the plan starts from it but may change it. Click to lock it", "Lock your change"]
      : ["Click to lock: the plan keeps it exactly as it is", "Lock this day"];
    const cls = `lk${p.locked ? " on" : ""}${p.custom ? " mine" : ""}`;
    return `<button class="${cls}" data-lock="${key}" aria-pressed="${!!p.locked}" title="${tip}"
      aria-label="${label}">${p.locked ? "🔒" : "🔓"}</button>`;
  }

  async toggleLock(key) {           // straight from the calendar (no modal): shown at once, then re-planned
    const p = calendarPage.CAL.planned[key];
    if (!p) return;
    if (p.locked && p.custom) {     // your locked change: off, back to the plan's version
      if (!await ask.confirm({title: "Back to the plan's version?", yes: "Use the plan's version", danger: true,
        text: "Your change to this day goes."})) return;
      p.locked = false; p.custom = false;
      calendarPage.calPending(key, "updating");
      api.action({action: "edit_session", key, reset: true}, "Back to the plan's version – updating…")
        .then((ok) => ok || calendarPage.calPending(key, null));
      return;
    }
    p.locked = !p.locked;
    router.render();
    api.action({action: "lock_session", key, locked: p.locked},
      p.locked ? "Locked – the plan keeps it as it is" : "Unlocked – the plan may change it when it next re-plans");
  }

  calPlan(p, key) {                 // key: the day, or "day#2" for a session you added
    const ds = key.slice(0, 10), main = key === ds;
    const lock = this.lockButton(p, key);     // (your change: a blue lock; ✎ where there is no lock)
    const mine = p.custom && !lock ? ' <span class="mine" title="You moved / added / changed this">✎</span>' : "";
    if (p.type === "rest") {
      const drop = p.custom && ds >= calendarPage.CAL.today ? " drop-day" : "";
      return `<div class="pl rest${drop}"${p.custom ? ` data-ds="${key}"` : ""}>${lock}Rest day${mine}</div>`
        + (main ? this.calProposed(ds) : "");
    }
    const rid = p.ride || p.compliance?.ride, M = calendarPage.calModel(), r = rid && M.byId[rid];   // linked, or matched automatically
    const st = this.calScore(p) || (p.status && p.status !== "n/a" ? ` <span class="chip ${esc(p.status)}">${esc(p.status)}</span>` : "");
    const via = r ? `<div class="via">✓ <a href="#ride/${encodeURIComponent(r.id)}">${esc(r.name || "ride")}</a>${r.date !== ds ? ` (${niceDate(r.date, {weekday: "short"})})` : ""}
    <button class="unlink" data-unlink="${key}" title="Unlink this ride" aria-label="Unlink this ride">×</button></div>` : "";
    const drop = ds <= calendarPage.CAL.today ? ` drop" data-date="${key}` : "";
    const pending = p.pending ? ` <span class="chip">${esc(p.pending)}…</span>` : "";
    return `<div class="pl${p.status === "done" || p.ride ? " done" : ""}${p.pending ? " pending" : ""}${drop}" data-ds="${key}" style="--c:${typeColor(p.type)}" title="Click to open, drag to move: ${esc(p.workout || p.title)}">
    <div class="nm"><span class="tt">${esc(p.title)}</span>${mine}${lock}</div>
    <div class="mt">${hhmm(p.minutes)}${p.tss != null ? ` · ${fmt(p.tss)} TSS` : ""}${st}${pending}</div>
    ${p.steps?.length ? profileSVG(p, "calprof") : ""}${via}</div>${main ? this.calProposed(ds) : ""}`;
  }

  calAct(a) {
    const kind = actKind(a);
    const label = TYPE_LABEL[a.workout] || (a.sport === "ride" ? "Ride" : a.sport[0]?.toUpperCase() + a.sport.slice(1));
    const to = calendarPage.calModel().linked[a.id];
    return `<a class="ac${to ? " linked" : ""}" href="#ride/${encodeURIComponent(a.id)}" draggable="false" data-ride="${esc(a.id)}" style="--c:${typeColor(kind)}"
    title="${esc(a.name)}${to ? ` – done for the plan on ${niceDate(to)}` : " – drag onto a planned session to mark it done"}">
    <div class="nm">${to ? "✓ " : ""}${esc(a.name || label)}</div><div class="mt">${esc(label)} · ${hhmm(a.min)}${a.tss != null ? ` · ${fmt(a.tss)} TSS` : ""}</div></a>`;
  }
}
export const sessionModal = new SessionModal();
