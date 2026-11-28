// The top bar's two small status controls: the sync icon (state at a glance, a tooltip, a popover per data type,
// Update now, a link to the Sync page) and the setup checklist pill (Connect Garmin, import history, the full
// model run, a ride app), each task snoozed or dismissed in this browser (SyncStatus, SetupTasks, StatusBar).
import {Pref} from "./controls.js";
import {api, jobs, S} from "./api.js";
import {$, esc, fmt, LOCAL} from "./util.js";
import {niceDate, today} from "./widgets.js";
import {exportDrop} from "./export_drop.js";

// two arrows round the state's mark: the data on this computer brought up to date (not a cloud: nothing goes to one)
const ARROWS = '<path d="M19 9a7.5 7.5 0 0 0-13.2-1.8M5 15a7.5 7.5 0 0 0 13.2 1.8"/><path d="M19 4.5V9h-4.5M5 19.5V15h4.5"/>';
const MARK = {ok: '<path d="M9.5 12l1.8 1.8 3.2-3.3"/>', warn: '<path d="M12 9.5v3M12 14.8v.01"/>',
  bad: '<path d="M10 10l4 4M14 10l-4 4"/>', off: '<path d="M4 4l16 16"/>', run: ""};
export const syncIcon = (level) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ARROWS}${MARK[level] || ""}</svg>`;
const LEVEL_LABEL = {ok: "Up to date", warn: "Garmin is behind", bad: "Last update had a problem", off: "Garmin not connected",
  run: "Updating…"};

const dayDiff = (a, b) => Math.round((new Date(a + "T12:00:00") - new Date(b + "T12:00:00")) / 864e5);
// days between today and the newest morning (sleep / HRV) goodpush has; null: none at all
export function wellnessAge() {
  const t = S.today || {}, td = today();
  return t.wellness_date && td ? dayDiff(td, t.wellness_date) : null;
}
// "this morning", "yesterday", "Tue, 24 Nov"
export function morningLabel(ds) {
  const td = today(), d = ds && td ? dayDiff(td, ds) : null;
  return d == null ? "–" : d <= 0 ? "this morning" : d === 1 ? "yesterday" : niceDate(ds);
}
// an ISO time ("2026-11-27T07:52:10") as "07:52", "yesterday 07:52", "Nov 25 07:52"
export function whenLabel(iso) {
  if (!iso) return "never";
  const ds = iso.slice(0, 10), hm = iso.slice(11, 16), td = today(), d = td ? dayDiff(td, ds) : null;
  if (d === 0) return hm || "today";
  if (d === 1) return `yesterday${hm ? " " + hm : ""}`;
  return `${niceDate(ds, {day: "numeric", month: "short"})}${hm ? " " + hm : ""}`;
}

class SyncStatus {
  // one state for the icon: run (a job is going), off (no Garmin and no mornings), bad (the last update stopped or
  // failed), warn (the newest morning is older than today), ok
  level() {
    const G = S.garmin_api || {}, ls = G.last_sync || {}, J = jobs.JOB || {}, age = wellnessAge();
    if (J.running) return "run";
    if (!G.configured && age == null) return "off";
    const failedJob = J.ok === false && J.t && Date.now() / 1000 - J.t < 12 * 3600;
    if (failedJob || (G.configured && ls.stopped)) return "bad";
    return age === 0 ? "ok" : "warn";
  }

  tip(level) {
    const G = S.garmin_api || {}, ls = G.last_sync || {}, J = jobs.JOB || {}, age = wellnessAge();
    if (level === "run") return `${J.name || "Updating"}…`;
    if (level === "off") return "Garmin not connected · no sleep or HRV yet";
    const morning = age == null ? "no sleep or HRV yet" : `sleep & HRV from ${morningLabel(S.today.wellness_date)}`;
    if (level === "bad") return `Last update had a problem · ${morning}`;
    return `${level === "ok" ? "Up to date" : "Garmin is behind"} · ${morning}${ls.at ? ` · checked ${whenLabel(ls.at)}` : ""}`;
  }

  // [state, what, newest] for each kind of data goodpush brings in or sends out
  rows() {
    const G = S.garmin_api || {}, age = wellnessAge(), R = S.rides?.[0], out = [];
    const ls = G.last_sync || {};
    out.push([age == null ? "off" : age === 0 ? "ok" : ls.stopped ? "bad" : "warn", "Sleep, HRV, resting HR",
      age == null ? "no data yet" : morningLabel(S.today.wellness_date)]);
    out.push([R ? (ls.retry?.length ? "warn" : "ok") : "off", "Rides", R ? whenLabel(R.start || R.date) : "no rides yet"]);
    const I = S.intervals_icu || {}, inc = S.incyclist || {}, gh = S.google_health || {};
    if (I.configured) out.push([I.last_push?.at ? "ok" : "warn", "Plan → intervals.icu", I.last_push?.at ? whenLabel(I.last_push.at) : "not sent yet"]);
    if (inc.sync) out.push([inc.written ? "ok" : "warn", "Plan → Incyclist", inc.written ? whenLabel(inc.written) : "not written yet"]);
    if (gh.connected) out.push([gh.last_sync?.at ? "ok" : "warn", "Fitbit (Google Health)", gh.last_sync?.at ? whenLabel(gh.last_sync.at) : "not synced yet"]);
    for (const c of this.peers()) out.push([c.last ? "ok" : "warn", `Other computer: ${c.label || c.name}`, c.last ? whenLabel(c.last) : "not synced yet"]);
    return out;
  }

  peers() { return (S.peers?.computers || []).filter((c) => c.can_start); }

  note() {
    const G = S.garmin_api || {}, ls = G.last_sync || {};
    if (!G.configured) return "Add your Garmin Personal API key in Settings once. After that goodpush checks Garmin by itself.";
    if (ls.stopped) return esc(ls.stopped);
    const parts = [G.auto_hours > 0 ? `Checks Garmin every ${fmt(G.auto_hours)} h while the app runs` : "Automatic checks are off"];
    if (ls.skipped) parts.push(`Garmin didn't answer ${ls.skipped} request${ls.skipped > 1 ? "s" : ""}: fetched again next update`);
    return parts.join(" · ");
  }

  popover() {
    const lv = this.level(), G = S.garmin_api || {}, peers = this.peers();
    return `<div class="stpop" role="dialog" aria-label="Sync status"><h4>${LEVEL_LABEL[lv]}</h4>
      <div class="strows">${this.rows().map(([s, k, w]) => `<div class="strow"><span class="led ${s}"></span><span>${esc(k)}</span><span class="when">${esc(w)}</span></div>`).join("")}</div>
      <div class="small muted">${this.note()}</div>
      <div class="row">${G.configured
        ? `<button class="btn sm primary" data-st="garmin_sync"${lv === "run" ? " disabled" : ""}>Update now</button>`
        : '<a class="btn sm primary" href="#settings/keys">Connect Garmin</a>'}
        ${peers.length && LOCAL ? `<button class="btn sm" data-st="peer_sync_all">Sync other computer</button>` : ""}
        <span class="spacer"></span><a class="small" href="#sync">All sync details →</a></div></div>`;
  }
}
export const syncStatus = new SyncStatus();

// the setup checklist: [key, title, what it gives, done?, action html, required]
class SetupTasks {
  later = new Pref("setup_later", "{}");       // key -> the day it shows again
  hidden = new Pref("setup_hidden", "{}");     // key -> true: don't show

  read(p) { try { return JSON.parse(p.value) || {}; } catch (e) { return {}; } }

  all() {
    const G = S.garmin_api || {}, P = S.physio, C = P?.full_run_cores, I = S.intervals_icu || {}, inc = S.incyclist || {};
    return [
      {key: "garmin", title: "Connect Garmin", text: "Your Personal API key, once. Sleep, HRV and rides then come in by themselves.",
        done: G.configured || wellnessAge() != null, act: '<a class="btn sm primary" href="#settings/keys">Connect</a>', required: true},
      {key: "history", title: "Bring in your whole history",
        text: `goodpush started with your last 90 days${S.import_guide?.first ? ` (rides from ${esc(niceDate(S.import_guide.first))})` : ""}. Your account export makes fitness, thresholds and the plan much better: <a href="https://www.garmin.com/account/datamanagement/exportdata/" target="_blank" rel="noopener">ask Garmin for it</a> (or Strava: Settings → My Account → Download your archive), then drop the .zip anywhere in the app.`,
        done: !S.import_guide?.show, act: '<label class="btn sm">Choose the .zip<input type="file" id="setupzip" accept=".zip,application/zip" hidden></label>'},
      {key: "model", title: "Run the full model",
        text: `Your thresholds and the plan's sessions come from a quick best fit with no ranges. The full run takes about an hour (2 h at most)${C ? ` and keeps ${C.chains} of ${C.all} cores busy` : ""} at low priority: best started overnight.`,
        done: !P || !!P.cp_p05, act: '<a class="btn sm" href="#model">Model page</a>'},
      {key: "rideapp", title: "Send your plan to a ride app", text: "intervals.icu or Incyclist get each planned session with its steps.",
        done: !!I.configured || !!inc.sync, act: '<a class="btn sm" href="#settings/icu">Settings</a>', optional: true},
    ];
  }

  // the tasks the checklist counts: not dismissed; open ones not snoozed
  shown() {
    const later = this.read(this.later), hidden = this.read(this.hidden), td = today() || "";
    return this.all().filter((t) => t.required || (!hidden[t.key] && !(later[t.key] && later[t.key] > td && !t.done)));
  }

  snooze(key, days) {
    const later = this.read(this.later), d = new Date(); d.setDate(d.getDate() + days);
    later[key] = d.toISOString().slice(0, 10);
    this.later.set(JSON.stringify(later));
  }

  dismiss(key) {
    const hidden = this.read(this.hidden); hidden[key] = true;
    this.hidden.set(JSON.stringify(hidden));
  }

  popover() {
    const T = this.shown(), first = T.find((t) => !t.done);
    return `<div class="stpop" role="dialog" aria-label="Setup"><h4>Finish setting up</h4>
      ${T.map((t) => `<div class="task${t.done ? " done" : ""}"><span class="chk">${t.done ? "✓" : ""}</span><div>
        <b>${esc(t.title)}</b>${!t.done && t.required ? '<span class="tag">needed</span>' : !t.done && t.optional ? '<span class="tag opt">optional</span>' : ""}
        ${t.done ? "" : `<p>${t.text}</p><div class="acts">${t === first ? t.act.replace('class="btn sm"', 'class="btn sm primary"') : t.act}
          ${t.required ? "" : `<button class="btn sm ghost" data-later="${t.key}">Later</button><button class="btn sm ghost" data-hide="${t.key}">Don't show</button>`}</div>`}
      </div></div>`).join("")}</div>`;
  }
}
export const setupTasks = new SetupTasks();

class StatusBar {
  open = null;                 // "sync" | "setup" | null
  key = "";

  ring(done, total) {
    const c = 2 * Math.PI * 9, f = total ? c * done / total : 0;
    return `<svg class="setupring" viewBox="0 0 24 24" aria-hidden="true"><circle class="bgc" cx="12" cy="12" r="9"/>${done ? `<circle class="fgc" cx="12" cy="12" r="9" stroke-dasharray="${f.toFixed(1)} ${c.toFixed(1)}"/>` : ""}</svg>`;
  }

  render() {
    const box = $("#statusbar");
    if (!box || !S) return;
    const lv = syncStatus.level(), tip = syncStatus.tip(lv);
    const T = setupTasks.shown(), done = T.filter((t) => t.done).length, showSetup = done < T.length;
    if (this.open === "setup" && !showSetup) this.open = null;
    box.innerHTML = `${showSetup ? `<button class="setuppill" id="setupbtn" type="button" aria-haspopup="dialog" aria-expanded="${this.open === "setup"}"
        title="Setup: ${done} of ${T.length} done">${this.ring(done, T.length)}<span class="lbl">Setup</span><span class="n">${done}/${T.length}</span></button>` : ""}
      <span class="syncwrap${this.open === "sync" ? " open" : ""}">
        <button class="btn sm ghost circle syncicon s-${lv}" id="syncicon" type="button" aria-haspopup="dialog" aria-expanded="${this.open === "sync"}"
          aria-label="Sync status: ${esc(tip)}">${syncIcon(lv)}<i class="dot"></i></button>
        <span class="sttip" role="tooltip">${esc(tip)}</span></span>
      ${this.open === "sync" ? syncStatus.popover() : this.open === "setup" ? setupTasks.popover() : ""}`;
    this.key = this.stateKey();
    this.wirePanel(box);
  }

  stateKey() { const J = jobs.JOB || {}; return [J.running, J.ok, J.t].join("|"); }

  // from the job poll: redraw only when the job state changed (not every 0.3 s)
  refresh() { if (this.stateKey() !== this.key) this.render(); }

  toggle(which) { this.open = this.open === which ? null : which; this.render(); }

  wirePanel(box) {
    $("#syncicon", box)?.addEventListener("click", () => this.toggle("sync"));
    $("#setupbtn", box)?.addEventListener("click", () => this.toggle("setup"));
    box.querySelectorAll("[data-st]").forEach((b) => b.addEventListener("click", () => {
      api.action({action: b.dataset.st}, b.dataset.st === "garmin_sync" ? "Checking Garmin…" : "Syncing…");
    }));
    box.querySelectorAll("[data-later]").forEach((b) => b.addEventListener("click", () => { setupTasks.snooze(b.dataset.later, 7); this.render(); }));
    box.querySelectorAll("[data-hide]").forEach((b) => b.addEventListener("click", () => { setupTasks.dismiss(b.dataset.hide); this.render(); }));
    $("#setupzip", box)?.addEventListener("change", (e) => { if (e.target.files[0]) { exportDrop.uploadExport(e.target.files[0]); this.toggle(null); } });
    box.querySelectorAll(".stpop a[href^='#']").forEach((a) => a.addEventListener("click", () => { this.open = null; setTimeout(() => this.render()); }));
  }

  wire() {
    window.addEventListener("hashchange", () => { if (this.open) { this.open = null; this.render(); } });   // a new page
    document.addEventListener("click",(e) => { if (this.open && !e.composedPath().some((n) => n.id === "statusbar")) { this.open = null; this.render(); } });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.open) { const w = this.open; this.open = null; this.render(); $(w === "sync" ? "#syncicon" : "#setupbtn")?.focus(); }
    });
  }
}
export const statusBar = new StatusBar();
