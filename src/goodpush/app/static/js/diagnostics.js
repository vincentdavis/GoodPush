// Settings > Diagnostics (DiagnosticsCard): versions, the model, athletes, data, sources, folders and libraries from
// /api/diagnostics (app/diagnostics.py), and a viewer for the app log and the run logs (/api/log). Loaded when the
// section is opened, and drawn into its own card, so the log you're reading stays put while the rest of Settings
// redraws.
import {api, toast} from "./api.js";
import {$, esc, LOCAL} from "./util.js";

const LOG_LINES = [100, 400, 2000];
const LOG_NAMES = {"app.log": "App log (this run)", "app.previous.log": "App log (the run before)"};

class DiagnosticsCard {
  data = null;          // the last /api/diagnostics answer
  log = null;           // the last /api/log answer
  pick = "";            // the log shown: "where|name"
  lines = 400;
  loading = false;

  // the section's body: drawn from what's loaded, or a placeholder until the section is opened
  card() {
    return `<div class="card" id="diagcard">${this.inner()}</div>`;
  }

  inner() {
    const D = this.data;
    if (!D) return `<p class="small muted" style="margin:0">${this.loading ? "Loading…" : "Open this section to see the details."}</p>`;
    return `
    <div class="row" style="margin-bottom:12px">
      <button class="btn sm" id="diagcopy" type="button">Copy for a bug report</button>
      <button class="btn sm" id="diagreload" type="button">Refresh</button>
      <span class="small muted">As of ${esc(D.made || "")}</span></div>
    <p class="small muted" style="margin:0 0 14px">The copy leaves out your folders, names and sign-ins, so it can go in a
      public bug report. Folders and logs below stay on this computer.</p>
    <div class="diaggrid">${D.groups.map((g) => this.group(g)).join("")}</div>
    <h3 style="margin:20px 0 6px">Logs</h3>
    ${this.logPicker()}
    <pre class="log tall" id="diaglogtext" tabindex="0" aria-label="Log text">${esc(this.logText())}</pre>
    <p class="small muted" style="margin:6px 0 0">Logs can hold your folders and the names of your rides: check one before you
      post it anywhere public.</p>`;
  }

  group(g) {
    if (g.id === "packages")                    // versions only: one line
      return `<div class="wide"><h3>${esc(g.title)}</h3><p class="small" style="margin:0">${g.facts.map((f) =>
        `${esc(f.label)} <span class="muted">${esc(f.value)}</span>`).join(" · ")}</p></div>`;
    const rows = g.facts.map((f) => `<tr><td>${esc(f.label)}</td><td>${f.folder ? this.folder(f) : esc(f.value)}</td></tr>`).join("");
    return `<div${g.id === "folders" ? ' class="wide"' : ""}><h3>${esc(g.title)}</h3><div class="tw"><table><tbody>${rows}</tbody></table></div></div>`;
  }

  folder(f) {
    return `<code class="diagpath">${esc(f.value)}</code>${LOCAL ? ` <button class="btn sm" type="button" data-reveal-folder="${esc(f.folder)}">Show</button>` : ""}`;
  }

  logPicker() {
    const logs = this.data?.logs || [];
    if (!logs.length) return `<p class="small muted" style="margin:0">No logs yet.</p>`;
    const kb = (n) => n < 1024 ? `${n} bytes` : `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
    const opt = (l) => {
      const v = `${l.where}|${l.name}`;
      return `<option value="${esc(v)}"${v === this.pick ? " selected" : ""}>${esc(LOG_NAMES[l.name] || l.name)} · ${esc(l.modified)} · ${kb(l.size)}</option>`;
    };
    const app = logs.filter((l) => l.where === "app"), runs = logs.filter((l) => l.where === "run");
    return `<div class="row">
      <label class="f" style="flex:1;min-width:220px">Log
        <select id="diaglog">${app.length ? `<optgroup label="The app">${app.map(opt).join("")}</optgroup>` : ""}
          ${runs.length ? `<optgroup label="Runs (fits, refreshes, rides)">${runs.map(opt).join("")}</optgroup>` : ""}</select></label>
      <label class="f" style="max-width:140px">Last
        <select id="diaglines">${LOG_LINES.map((n) => `<option value="${n}"${n === this.lines ? " selected" : ""}>${n} lines</option>`).join("")}</select></label>
      <button class="btn sm" id="diaglogload" type="button" style="align-self:flex-end">Reload log</button>
      <button class="btn sm" id="diaglogcopy" type="button" style="align-self:flex-end">Copy log</button>
      ${LOCAL ? '<button class="btn sm" id="diaglogshow" type="button" style="align-self:flex-end">Show file</button>' : ""}</div>`;
  }

  logText() {
    if (!this.data?.logs?.length) return "";
    if (!this.log) return "Loading…";
    if (this.log.error) return this.log.error;
    return (this.log.cut ? "… (earlier lines left out)\n" : "") + (this.log.text || "(empty)");
  }

  async load() {
    if (this.loading) return;
    this.loading = true;
    this.draw();
    try {
      const r = await fetch("/api/diagnostics", {cache: "no-store"});
      this.data = await r.json();
      const logs = this.data.logs || [];
      if (!logs.some((l) => `${l.where}|${l.name}` === this.pick)) this.pick = logs.length ? `${logs[0].where}|${logs[0].name}` : "";
    } catch {
      toast.show("Couldn't read the diagnostics");
    }
    this.loading = false;
    this.draw();
    this.loadLog();
  }

  async loadLog() {
    if (!this.pick) return;
    const [where, name] = this.pick.split("|");
    const q = new URLSearchParams({where, name, lines: String(this.lines)});
    try {
      const r = await fetch(`/api/log?${q}`, {cache: "no-store"});
      this.log = await r.json();
    } catch {
      this.log = {error: "Couldn't read this log"};
    }
    const pre = $("#diaglogtext");
    if (!pre) return;
    pre.textContent = this.logText();
    pre.scrollTop = pre.scrollHeight;          // the newest lines
  }

  draw() {
    const c = $("#diagcard");
    if (!c) return;
    c.innerHTML = this.inner();
    this.wireCard();
    const pre = $("#diaglogtext");
    if (pre) pre.scrollTop = pre.scrollHeight;
  }

  async copy(text, what) {
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch { ok = false; }
    if (!ok) {                                  // not a secure page (the phone over Wi-Fi): the old way
      const t = document.createElement("textarea");
      t.value = text; document.body.append(t); t.select();
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      t.remove();
    }
    toast.show(ok ? `${what} copied` : "Couldn't copy: select the text and copy it yourself");
  }

  // the section (details#set-diagnostics) loads when opened; its card's buttons
  wire(section) {
    if (!section) return;
    section.addEventListener("toggle", () => { if (section.open && !this.data) this.load(); });
    if (section.open && !this.data) this.load();
    else this.wireCard();
  }

  wireCard() {
    $("#diagreload")?.addEventListener("click", () => this.load());
    $("#diagcopy")?.addEventListener("click", () => this.copy(this.data?.shareable_text || "", "Diagnostics"));
    $("#diaglog")?.addEventListener("change", (e) => { this.pick = e.target.value; this.log = null; this.loadLog(); });
    $("#diaglines")?.addEventListener("change", (e) => { this.lines = Number(e.target.value); this.loadLog(); });
    $("#diaglogload")?.addEventListener("click", () => this.loadLog());
    $("#diaglogcopy")?.addEventListener("click", () => this.copy(this.log?.text || "", "Log"));
    $("#diaglogshow")?.addEventListener("click", () => {
      const [where, name] = this.pick.split("|");
      api.query({action: "reveal", where, name});
    });
    document.querySelectorAll("[data-reveal-folder]").forEach((b) => b.addEventListener("click", () =>
      api.query({action: "reveal", folder: b.dataset.revealFolder})));
  }
}
export const diagnostics = new DiagnosticsCard();
