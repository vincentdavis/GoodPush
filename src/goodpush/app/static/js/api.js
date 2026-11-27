// Server calls and the state S (Api), the job chip / log / restart watch (JobWatch), Toast.
import {$, esc} from "./util.js";
import {router} from "./main.js";
import {dips} from "./dips.js";
import {modelPage} from "./pages/model.js";
import {calendarPage} from "./calendar/calendar.js";
import {el} from "./charts.js";
import {statusBar} from "./status_bar.js";

export let S = null;                 // last state from the server
// while a job runs: every 0.3 s; otherwise every few seconds, so a job the app starts by itself (the
// automatic update a few seconds after start-up) shows up without a reload
const JOB_IDLE_MS = 4000, JOB_IDLE_HIDDEN_MS = 30000;

class Api {
  // data + actions
  async load() {
    const r = await fetch("/api/state", {cache: "no-store"});
    if (!r.ok) throw new Error(await r.text());
    S = await r.json();
    if (router.route() === "calendar" && calendarPage.CAL && calendarPage.calKey !== calendarPage.calKeyNow()) await calendarPage.loadCal(false);   // one redraw with the new plan
    router.render();
    if (S.refreshing) jobs.pollJob();
  }

  async action(body, msg) {
    const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
      body: JSON.stringify(body)});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return false; }
    if (j.busy) { toast.show("Busy with another job, try again in a moment"); return false; }
    // a question first (merge two phases? re-plan your own days?): the answer under the key the server names
    if (j.ask) return this.action({...body, [j.answer || "merge"]: confirm(j.ask)}, msg);
    if (j.queued) toast.show("Saved – the plan updates as soon as the current job finishes");
    else if (msg) toast.show(msg);
    jobs.pollJob();
    return true;
  }

  async query(body) {              // an answer only (nothing saved, no job): its JSON, or null after a toast
    const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
      body: JSON.stringify(body)});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return null; }
    return j;
  }

  async saveOnly(body, msg) {
    const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
      body: JSON.stringify(body)});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return; }
    toast.show(msg); this.load();
  }
}
export const api = new Api();

class JobWatch {
  jobTimer = null;
  JOB = null;   // JOB: the last /api/job answer, so a redraw keeps its log

  APP_RUN = null;                // the app run this page came from (/api/job "started")

  pollIdle = () => { this.jobTimer = setTimeout(() => this.pollJob(), document.hidden ? JOB_IDLE_HIDDEN_MS : JOB_IDLE_MS); };

  // held open while the page is: a restarted app finds this tab (no new one) and its first message reloads the page.
  // Opened after the page loaded: one opened before holds the load event back (Chromium) and the app never starts
  stayAlive() {
    new EventSource("/api/alive").onmessage = (e) => { if (this.APP_RUN && e.data !== this.APP_RUN) location.reload(); else this.APP_RUN = this.APP_RUN || e.data; };
  }

  pollJob() {
    clearTimeout(this.jobTimer);
    fetch("/api/job", {cache: "no-store"}).then((r) => r.json()).then((j) => {
      if (this.APP_RUN && j.started && j.started !== this.APP_RUN) { location.reload(); return; }   // the app was restarted: its new page
      this.APP_RUN = this.APP_RUN || j.started;
      this.JOB = j;
      statusBar.refresh();
      if (S && j.code_stale !== S.code_stale) { S.code_stale = j.code_stale; router.render(); }   // goodpush updated: say so now
      if (S && (j.peer_waiting || null) !== (S.peer_waiting || null)) { S.peer_waiting = j.peer_waiting || null; router.render(); }
      const chip = $("#jobchip");
      if (j.running) {
        chip.hidden = false;
        const st = j.step, pct = st ? Math.round(st.done / st.total * 100) : 0;
        chip.innerHTML = `<span class="spin"></span>${esc(j.name)}…` + (st ? `<span class="jbar"><i style="width:${pct}%"></i></span>
        <span class="small">${esc(st.label)} ${st.done}/${st.total}</span>` : "") + (j.queued ? `<span class="small muted">then: ${esc(j.queued)}</span>` : "");
        const lg = $("#joblog");
        if (lg) { lg.hidden = false; lg.textContent = j.log.join("\n"); lg.scrollTop = 1e6; }
        this.jobTimer = setTimeout(() => this.pollJob(), 300);
      } else {
        chip.hidden = true;
        const lg = $("#joblog");
        if (lg && j.log.length) { lg.hidden = false; lg.textContent = j.log.join("\n"); }
        const ended = j.name && j.t && Date.now() / 1000 - j.t < 900 && j.ok !== null && !this.pollJob.seen?.[j.t];
        if (ended) {
          this.pollJob.seen = {...(this.pollJob.seen || {}), [j.t]: true};
          toast.show(j.ok ? "Updated" : `Failed: ${j.log.at(-1)?.replace(/^failed: /, "") || "no details"} – the log is on the Sync page`);
        }
        // a job's end, or newer rides / plan than this page has (it missed that end: a throttled window, a sync)
        // (an app not restarted since an update doesn't send them: no reload every poll)
        const newer = S && "data_version" in j && (j.data_version !== S.data_version || j.plan_version !== S.plan_version);
        if (ended || newer) this.reload();
        this.pollIdle();
      }
    }).catch(this.pollIdle);                                   // the app restarting: keep trying
  }

  reload() {                                   // everything again: the calendar re-reads the new plan too; a refit: the Model page
    dips.drops.forget(); modelPage.model.forget(); calendarPage.calKey = ""; api.load();
  }

  jobLog() {                                   // the last job's log (kept across redraws)
    const lines = this.JOB?.log || [];
    return `<div class="log" id="joblog"${lines.length ? "" : " hidden"}>${esc(lines.join("\n"))}</div>`;
  }
}
export const jobs = new JobWatch();

class Toast {
  timer;

  show(t) {
    const el = $("#toast"); el.textContent = t; el.classList.add("show");
    clearTimeout(this.timer); this.timer = setTimeout(() => el.classList.remove("show"), 3200);
  }
}
export const toast = new Toast();
