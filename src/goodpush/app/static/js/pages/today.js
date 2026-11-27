// Today: the session, readiness, the update card, the history guide (TodayPage); export drops (ExportDrop).
import {Pref, Segmented} from "../controls.js";
import {Page} from "./page.js";
import {api, jobs, S, toast} from "../api.js";
import {$, esc, fmt, hhmm} from "../util.js";
import {delta, ICON, morningsStrip, niceDate, rideIt, sessionCard, sessionOn, sessions, signsRow, STATUS_LABEL, tile,
  today} from "../widgets.js";
import {router} from "../main.js";
import {markers} from "../markers.js";

class ExportDrop {
  // pages
  // account exports (Strava / Garmin .zip) dropped anywhere on the app, or chosen on Today's guide: sent to
  // /api/upload (progress in the job chip), then imported by a job
  uploadExport(file) {
    if (!/\.zip$/i.test(file.name)) { toast.show("Drop the .zip from Garmin or Strava, still zipped"); return; }
    const chip = $("#jobchip"), xhr = new XMLHttpRequest();
    chip.hidden = false;
    chip.innerHTML = `<span class="spin"></span>Uploading ${esc(file.name)}…`;
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("X-GoodPush", "1");
    xhr.setRequestHeader("X-Filename", file.name);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) chip.innerHTML = `<span class="spin"></span>Uploading ${esc(file.name)}… ${Math.round(e.loaded / e.total * 100)}%`; };
    xhr.onload = () => {
      let j = {};
      try { j = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
      if (xhr.status >= 400 || j.error) { chip.hidden = true; toast.show(j.error || "Upload failed"); return; }
      toast.show(`${j.kind === "garmin" ? "Garmin" : "Strava"} export received – importing your history (a few minutes)…`);
      jobs.pollJob();
    };
    xhr.onerror = () => { chip.hidden = true; toast.show("Upload failed"); };
    xhr.send(file);
  }

  dropDepth = 0;

  hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
}
const exportDrop = new ExportDrop();

class TodayPage extends Page {
  // a new user starts with their last 90 days: how to bring in the rest (account exports, imported here)
  historyLater = new Pref("history_later", false);   // "later" on the history guide

  historyCard() {
    const h = S.import_guide;
    if (!h?.show || this.historyLater.value) return "";
    return `<div class="card guide">
    <h3 style="margin-top:0">Bring in your whole history</h3>
    <p class="small" style="margin-top:0">goodpush started with your last 90 days${h.first ? ` (rides from ${esc(niceDate(h.first))})` : ""}. Your fitness, thresholds and plan get much better with everything before that:</p>
    <ol class="small steps">
      <li><b>Ask Garmin for your data.</b> On the web, <a href="https://www.garmin.com/account/datamanagement/exportdata/" target="_blank" rel="noopener">Garmin account → Data Management → Export Your Data</a>. Garmin emails you a link, usually within a day or two.</li>
      <li><b>Drop the zip onto this page</b> (as you downloaded it, still zipped), or
        <label class="btn sm primary">Choose the file<input type="file" id="histfile" accept=".zip,application/zip" hidden></label></li>
    </ol>
    <p class="small muted">Use Strava? Its export works the same way (strava.com → Settings → My Account → Download or delete your account → Request your archive). Everything stays on this computer.</p>
    <button class="btn sm" id="histlater">Not now</button>
  </div>`;
  }

  html() {
    const t = S.today || {}, pl = S.plan || {}, rd = pl.readiness || {status: "green", reasons: []};
    const td = today(), s = sessionOn(td);
    const tomorrow = sessions().find((x) => x.date > td);
    const ov = pl.override || "";
    const age = t.wellness_date ? Math.round((new Date(td) - new Date(t.wellness_date)) / 864e5) : null;
    const hrvRange = t.hrv_base_low && t.hrv_base_high ? `range ${fmt(t.hrv_base_low)}–${fmt(t.hrv_base_high)}` : "";
    const hrvCls = t.hrv == null || !t.hrv_base_low ? "" : t.hrv < t.hrv_base_low ? "down" : t.hrv > t.hrv_base_high ? "mid" : "up";
    const greet = S.name ? `Hi ${esc(S.name)}` : "Today";
    return `
  <h1>${greet}</h1>
  <p class="sub">${niceDate(td, {weekday: "long", day: "numeric", month: "long"})}${age != null ? ` · Garmin data ${age === 0 ? "from this morning" : age === 1 ? "from yesterday" : age + " days old"}` : ""}</p>
  ${this.updateCard(age)}
  ${this.historyCard()}
  ${pl.proposal?.length ? `<div class="card proposal" style="margin-bottom:14px"><div class="row"><b>Plan update proposed</b>
    <span class="small muted">${pl.proposal.length} change${pl.proposal.length > 1 ? "s" : ""} from how your recent sessions went</span><span class="spacer"></span>
    <a class="btn sm primary" href="#calendar">Review</a></div></div>` : ""}
  ${this.modelCard()}
  <div class="grid g2">
    <div class="card">
      <div class="row"><span class="status ${rd.status}">${ICON[rd.status]}${STATUS_LABEL[rd.status]}</span></div>
      ${signsRow(rd)}
      ${morningsStrip(rd)}
      <ul class="reasons">${(rd.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("") || "<li>Nothing flagged in sleep, HRV, resting HR or load.</li>"}
        ${(rd.notes || []).map((r) => `<li class="muted">${esc(r)}</li>`).join("")}</ul>
      ${markers.illWatchLine()}${markers.overWatchLine()}${markers.alphaLine()}
      <div style="margin-top:14px" class="small muted">How do you feel?</div>
      <div class="seg" style="margin-top:6px" id="feel">
        <button data-v="" class="${ov === "" ? "on" : ""}">Normal</button>
        <button data-v="off" class="${ov === "off" ? "on" : ""}">A bit off</button>
        <button data-v="sick" class="${ov === "sick" ? "on" : ""}">Sick</button>
      </div>
    </div>
    ${sessionCard(s)}
  </div>
  <h2>This morning</h2>
  <div class="tiles">
    ${tile("Sleep", hhmm(t.sleep_h * 60, false), "h", delta(t.sleep_h, t.sleep_h_base, " h", true, 1))}
    ${tile("Sleep score", fmt(t.sleep_score), "", t.sleep_score_base ? `usual ${fmt(t.sleep_score_base)}` : "")}
    ${tile("HRV", fmt(t.hrv), "ms", hrvRange, hrvCls)}
    ${tile("Resting HR", fmt(t.rhr), "bpm", delta(t.rhr, t.rhr_base, " bpm", false))}
    ${tile("Body battery", fmt(t.body_battery_max), "", "overnight high")}
    ${tile("Form (TSB)", (t.tsb > 0 ? "+" : "") + fmt(t.tsb), "", `load ${fmt(t.ctl)} · fatigue ${fmt(t.atl)}`)}
  </div>
  ${tomorrow ? `<h2>Next: ${niceDate(tomorrow.date)}</h2>${sessionCard(tomorrow, false)}` : ""}
`;
  }

  // the physio model has a best fit but no ranges: no full sampling (NUTS) run yet, or none since the model changed
  modelCard() {
    const P = S.physio, C = P?.full_run_cores;
    if (!P || P.cp_p05) return "";
    return `<div class="card proposal" style="margin-bottom:14px"><div class="row"><b>The model still needs its full run</b>
    <span class="spacer"></span><a class="btn sm primary" href="#model">Model page</a></div>
    <p class="small" style="margin:6px 0 0">No full sampling run (NUTS) of the current model yet.
    Your thresholds, what training does for you and the plan's choice of sessions come from a quick best fit only, with no ranges.
    <b>Run it at some point:</b> Model page → <i>Run sampling + backtest</i> (about an hour, 2 h at most)${P.full_run ? `, or
    <code>${esc(P.full_run)}</code>` : ""}.</p>
    ${C ? `<p class="small muted" style="margin:6px 0 0">It keeps ${C.chains} of this computer's ${C.all} cores busy the whole time
    (all ${C.all} for a few moments now and then). It runs at low priority, so the app stays usable, but other heavy work slows
    it and it slows that: best started when you can leave the computer to it, e.g. overnight.</p>` : ""}</div>`;
  }

  updateCard(age) {
    const fresh = age === 0, G = S.garmin_api || {};
    const last = G.last_sync?.at ? G.last_sync.at.slice(5, 16).replace("T", " ") : null;
    const status = age == null ? ["amber", "No Garmin data yet"] : fresh ? ["done", "Up to date"] :
      ["amber", age === 1 ? "Newest Garmin data is from yesterday" : `Newest Garmin data is ${age} days old`];
    const rideLink = S.rides?.length ? `last ride <a href="#ride/${encodeURIComponent(S.rides[0].id)}">${esc(S.rides[0].name || "ride")}, ${esc(S.rides[0].date.slice(5))}</a>` : "";
    const body = G.configured ? `<div class="row" style="margin-top:10px">
        <button class="btn primary" data-act="garmin_sync">Update now</button>
        <span class="small muted">${G.auto_hours > 0 ? `Checks Garmin every ${fmt(G.auto_hours)} h while the app runs` : "Automatic checks are off"}${last ? ` · last check ${esc(last)}` : ""} · <a href="#settings/garmin">Settings</a></span></div>`
      : `<div class="row" style="margin-top:10px">
        <a class="btn primary" href="#settings/keys">Connect Garmin</a>
        <span class="small muted">Add your Garmin Personal API key in Settings once – after that, <b>Update now</b> here pulls what's new.</span></div>`;
    return `<div class="card upd ${fresh ? "" : "stale"}">
    <div class="row">${markers.chipFor(status[0], status[1])}<span class="small muted">${rideLink}</span></div>
    ${body}
    ${jobs.jobLog()}
  </div>`;
  }

  wire() {
    document.querySelectorAll("[data-ride-key]").forEach((b) => b.addEventListener("click", () => rideIt(b.dataset.rideKey)));
    $("#histfile")?.addEventListener("change", (e) => { if (e.target.files[0]) exportDrop.uploadExport(e.target.files[0]); });
    $("#histlater")?.addEventListener("click", () => {
      this.historyLater.set(true);
      router.render();
    });
    new Segmented("#feel", (b) => api.action({action: "override", value: b.dataset.v}, "Adjusting today's plan…"));
  }
}
export const todayPage = new TodayPage();

window.addEventListener("dragenter", (e) => { if (exportDrop.hasFiles(e)) { exportDrop.dropDepth++; document.body.classList.add("dropping"); } });

window.addEventListener("dragleave", (e) => { if (exportDrop.hasFiles(e) && --exportDrop.dropDepth <= 0) { exportDrop.dropDepth = 0; document.body.classList.remove("dropping"); } });

window.addEventListener("dragover", (e) => { if (exportDrop.hasFiles(e)) e.preventDefault(); });

window.addEventListener("drop", (e) => {
  if (!e.dataTransfer?.files?.length) return;
  e.preventDefault();
  exportDrop.dropDepth = 0; document.body.classList.remove("dropping");
  exportDrop.uploadExport(e.dataTransfer.files[0]);
});
