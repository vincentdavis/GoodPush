// goodpush app entry: the page registry, Router (render + wire), FormMemory, ThemeSwitch, start-up listeners.
import {Pref} from "./controls.js";
import {api, jobs, S} from "./api.js";
import {$, esc, LOCAL} from "./util.js";
import {todayPage} from "./pages/today.js";
import {settingsPage} from "./pages/settings.js";
import {planPage} from "./pages/plan.js";
import {fitnessPage} from "./pages/fitness.js";
import {sourcesPage} from "./pages/sources.js";
import {helpPage} from "./pages/help.js";
import {riderPage} from "./pages/rider.js";
import {recoveryPage} from "./pages/recovery.js";
import {modelPage} from "./pages/model.js";
import {ridesPage} from "./pages/rides.js";
import {ridePage} from "./pages/ride.js";
import {calendarPage} from "./calendar/calendar.js";
import {TimeWindow} from "./charts.js";
import {athleteMenu} from "./athlete_menu.js";

// routing
const PAGES = {today: todayPage, plan: planPage, calendar: calendarPage, fitness: fitnessPage, rider: riderPage, recovery: recoveryPage,
  settings: settingsPage, rides: ridesPage, ride: ridePage, sources: sourcesPage, model: modelPage, help: helpPage};
const TAB_OF = {rides: "fitness", ride: "fitness", sources: "settings", help: "settings"};
// colour theme: auto (follows Windows / the phone) -> light -> dark, remembered in this browser; the daisyUI themes
// "light" and "dark" (styles/app.css), none set follows the system. Icons: [svg paths, label]
const THEMES = {auto: ['<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17A8.5 8.5 0 0 0 12 3.5z" fill="currentColor"/>', "Theme: automatic (follows your system)"],
  light: ['<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/>', "Theme: light"],
  dark: ['<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/>', "Theme: dark"]};

class FormMemory {
  // what you've typed (and which expanders are open) survives a redraw of the same page, e.g. the data
  // refresh when you come back to the tab; a successful Add clears its fields (clearFields)
  edited = new Set();

  formState() {
    const st = {};
    this.edited.forEach((id) => { const e = document.getElementById(id); if (e) st[id] = e.type === "checkbox" ? e.checked : e.value; });
    document.querySelectorAll("#main details[id]").forEach((d) => { st["@" + d.id] = d.open; });
    return st;
  }

  restoreForm(st) {
    for (const [k, v] of Object.entries(st)) {
      const e = document.getElementById(k.replace(/^@/, ""));
      if (!e) continue;
      if (k.startsWith("@")) e.open = v;
      else if (e.type === "checkbox") e.checked = v;
      else e.value = v;
    }
  }

  clearFields(...ids) {
    ids.forEach((id) => { this.edited.delete(id); const e = document.getElementById(id); if (e && e.tagName !== "SELECT") e.value = ""; });
  }
}
export const forms = new FormMemory();

class ThemeSwitch {
  lastScheme = matchMedia("(prefers-color-scheme: dark)");
  saved = new Pref("theme", null);   // index.html applies it before the page draws

  theme() { return document.documentElement.dataset.theme || "auto"; }

  setTheme(t) {
    if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    this.saved.set(t === "auto" ? null : t);
    const b = $("#themebtn");
    b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${THEMES[t][0]}</svg>`;
    b.title = THEMES[t][1] + " (click to change)"; b.setAttribute("aria-label", THEMES[t][1]);
    if (S) router.render();                                   // charts take their colours when drawn
  }
}
const themes = new ThemeSwitch();

// phones: the bottom bar's More opens the rest of the pages (and the guide) above it
class MoreMenu {
  open(on) {
    $("#morepanel").hidden = !on;
    $("#morebtn").setAttribute("aria-expanded", on ? "true" : "false");
  }

  wire() {
    const btn = $("#morebtn"), panel = $("#morepanel");
    btn.addEventListener("click", () => this.open(panel.hidden));
    panel.addEventListener("click", (e) => { if (e.target.closest("a")) this.open(false); });
    document.addEventListener("click", (e) => { if (!panel.hidden && !e.target.closest("#morepanel, #morebtn")) this.open(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) { this.open(false); btn.focus(); } });
  }
}
const moreMenu = new MoreMenu();

// the top bar's Quit (on the computer running goodpush): stops the app, tray icon included. A busy app asks first.
// The tab stays with a note; started again, goodpush reloads it (JobWatch.stayAlive)
class QuitButton {
  async quit() {
    let j = await api.query({action: "quit"});
    if (j?.ask) j = confirm(j.ask) ? await api.query({action: "quit", force: true}) : null;
    if (!j?.ok) return;
    document.title = "goodpush stopped";
    $("#main").innerHTML = `<div class="card"><h2>goodpush has stopped</h2>
      <p>You can close this tab. Start goodpush again and this page comes back by itself.</p></div>`;
    document.querySelectorAll("#tabs, .topright").forEach((e) => { e.hidden = true; });
  }

  wire() {
    const b = $("#quitbtn");
    b.hidden = !LOCAL;
    b.addEventListener("click", () => this.quit());
  }
}
const quitButton = new QuitButton();

class Router {
  route() { const p = location.hash.replace("#", "").split("/")[0]; return PAGES[p] ? p : "today"; }

  routeArg() { return decodeURIComponent(location.hash.replace("#", "").split("/").slice(1).join("/")); }

  lastRoute = "";

  render() {
    if (!S) return;
    const p = this.route();
    const same = location.hash === this.lastRoute, form = same ? forms.formState() : {};
    if (!same) forms.edited.clear();
    document.querySelectorAll("#tabs a").forEach((a) => a.classList.toggle("on", a.dataset.page === (TAB_OF[p] || p)));
    document.body.dataset.page = p;
    calendarPage.calResize = null;
    if (location.hash !== this.lastRoute) TimeWindow.forgetAll();
    // a redraw of the same page keeps its scroll: hold the old height until the charts (drawn a frame later)
    // are back, else a page that just fits jumps to the top
    const main = $("#main"), y = window.scrollY;
    if (same) main.style.minHeight = main.offsetHeight + "px";
    this.syncButton();
    athleteMenu.label();
    main.innerHTML = (S.code_stale ? `<div class="card proposal" style="margin-bottom:12px"><b>goodpush was updated since this app started.</b>
    Restart it (stop it, then <code>goodpush app</code>) to use the new version; until then the app, rides started
    from it and the plan run the old one.</div>` : "") + (p !== "settings" ? this.peerAsk("card proposal") : "") + PAGES[p].html();
    forms.restoreForm(form);
    this.wire(p);
    if (!same) window.scrollTo(0, 0);
    else requestAnimationFrame(() => { main.style.minHeight = ""; if (window.scrollY !== y) window.scrollTo(0, y); });
    this.lastRoute = location.hash;
  }

  // a computer that knew the pairing code waits for your OK: at the top of every page, on Settings next to the list
  peerAsk(cls) {
    const who = S.peer_waiting || S.peers?.waiting;
    return who ? `<div class="${cls}" style="margin:0 0 12px">
    <b>${esc(who)} wants to sync with this computer.</b> Allow it only if that's your other computer joining
    with the code shown here: it will read and change your training data.
    <div class="row" style="margin-top:8px"><button class="btn primary" data-peer-allow="1">Allow</button>
      <button class="btn" data-peer-allow="0">Deny</button></div></div>` : "";
  }

  // the top bar's Sync: on a computer that joined another one (Settings > Other computers)
  syncButton() {
    const b = $("#syncbtn"), joined = (S.peers?.computers || []).filter((c) => c.can_start);
    b.hidden = !joined.length || !LOCAL;
    const last = joined.map((p) => p.last).filter(Boolean).sort().pop();
    b.title = `Sync with ${joined.map((p) => p.label || p.name).join(", ")} now` + (last ? ` (last ${last.replace("T", " ")})` : "");
  }

  // wiring
  wire(page) {
    document.querySelectorAll("[data-mark]").forEach((b) => b.addEventListener("click", (e) => {
      e.preventDefault();
      api.action({action: "mark", date: b.dataset.mark, value: b.dataset.v}, b.dataset.v === "done" ? "Nice – marked done" : "Updated");
    }));
    document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => api.action({action: b.dataset.act}, "Started")));
    document.querySelectorAll("[data-peer-allow]").forEach((b) => b.addEventListener("click", () => {
      const allow = b.dataset.peerAllow === "1";
      S.peer_waiting = null;
      api.saveOnly({action: "peer_allow", allow}, allow ? "Allowed – it syncs now" : "Denied");
    }));
    document.querySelectorAll("tr.click").forEach((tr) => tr.addEventListener("click", (e) => {
      if (e.target.closest("a")) return; location.hash = tr.dataset.href;
    }));
    PAGES[page].wire();
  }
}
export const router = new Router();

document.addEventListener("input", (e) => { if (e.target.id && e.target.closest("#main")) forms.edited.add(e.target.id); });

window.addEventListener("hashchange", () => router.render());

window.addEventListener("load", () => { api.load().catch((e) => { $("#main").innerHTML = `<div class="card">Couldn't load data: ${esc(e.message)}<br>Run <code>goodpush run</code> first.</div>`; }); jobs.pollJob(); jobs.stayAlive(); });

// pick up changes made outside the app (e.g. `goodpush garmin-import`) when you come back to it
document.addEventListener("visibilitychange", () => { if (!document.hidden && S) api.load().catch(() => {}); });

themes.lastScheme.addEventListener?.("change", () => router.render());

document.addEventListener("DOMContentLoaded", () => {
  themes.setTheme(themes.theme());
  $("#themebtn").addEventListener("click", () => { const k = Object.keys(THEMES); themes.setTheme(k[(k.indexOf(themes.theme()) + 1) % k.length]); });
  $("#syncbtn").addEventListener("click", () => api.action({action: "peer_sync_all"}, "Syncing…"));
  athleteMenu.wire();
  quitButton.wire();
  moreMenu.wire();
});
