// Sync (#sync): what came in and went out, per kind of data, and the last update's log (SyncPage). The top bar's
// sync icon links here.
import {Page} from "./page.js";
import {jobs, S} from "../api.js";
import {esc, fmt, LOCAL} from "../util.js";
import {niceDate} from "../widgets.js";
import {morningLabel, syncStatus, whenLabel, wellnessAge, cloudIcon} from "../status_bar.js";

const STATE = {ok: "Up to date", warn: "Behind", bad: "Problem", off: "Not connected", run: "Updating"};

class SyncPage extends Page {
  rows() {
    const G = S.garmin_api || {}, ls = G.last_sync || {}, age = wellnessAge(), R = S.rides?.[0], H = S.import_guide || {};
    const I = S.intervals_icu || {}, inc = S.incyclist || {}, gh = S.google_health || {};
    const garmin = G.configured ? "Garmin (Personal API)" : S.garmin_import ? "Garmin" : "Garmin (not connected)";
    const checked = ls.at ? whenLabel(ls.at) : "–";
    const L = (s, t) => `<span class="stline"><span class="led ${s}"></span>${esc(t)}</span>`;
    const out = [
      ["Sleep, HRV, resting HR, body battery", garmin, age == null ? "–" : `${morningLabel(S.today.wellness_date)} (${S.today.wellness_date})`, checked,
        age == null ? L("off", "No data yet") : ls.stopped ? L("bad", "Stopped: Garmin limits requests") : age === 0 ? L("ok", "Up to date")
          : L("warn", `${age} day${age > 1 ? "s" : ""} behind: your watch may not have synced to Garmin Connect yet`)],
      ["Rides (activities and FIT files)", garmin, R ? `${niceDate(R.date)}: ${R.name || "ride"}` : "–", checked,
        !R ? L("off", "No rides yet") : ls.retry?.length ? L("warn", `${ls.retry.length} day${ls.retry.length > 1 ? "s" : ""} to fetch again next update`) : L("ok", "Up to date")],
      ["Your whole history", "Garmin or Strava export (.zip)", H.first ? `rides from ${niceDate(H.first)}` : "–", "–",
        H.show ? L("warn", "Only recent rides: drop your export anywhere in the app") : L("ok", "Imported")],
    ];
    out.push(["Planned sessions out", "intervals.icu", I.configured ? `next ${I.days || 14} days` : "–",
      I.last_push?.at ? whenLabel(I.last_push.at) : "–", I.configured ? (I.last_push?.at ? L("ok", `Sent ${fmt(I.last_push.sent || 0)}`) : L("warn", "Not sent yet")) : L("off", "Not connected")]);
    out.push(["Planned sessions out", "Incyclist", inc.entries ? `${inc.entries} sessions` : "–", inc.written ? whenLabel(inc.written) : "–",
      inc.sync ? (inc.written ? L("ok", "Written") : L("warn", "Not written yet")) : L("off", "Off")]);
    if (gh.client || gh.connected) out.push(["Sleep, HRV, resting HR", "Fitbit (Google Health)", "–", gh.last_sync?.at ? whenLabel(gh.last_sync.at) : "–",
      gh.connected ? L("ok", "Connected") : L("off", "Not connected")]);
    for (const c of syncStatus.peers()) out.push(["Everything", `Other computer: ${c.label || c.name}`, "–", c.last ? whenLabel(c.last) : "never", c.last ? L("ok", "Synced") : L("warn", "Not synced yet")]);
    return out;
  }

  html() {
    const G = S.garmin_api || {}, ls = G.last_sync || {}, lv = syncStatus.level();
    const last = ls.at ? `Last check ${esc(whenLabel(ls.at))}: ${fmt(ls.days || 0)} day${ls.days === 1 ? "" : "s"} and ${fmt(ls.activities || 0)} ride${ls.activities === 1 ? "" : "s"} updated${ls.calls ? ` (${ls.calls} requests)` : ""}.` : "";
    return `<h1>Sync</h1>
    <p class="sub">What goodpush has brought in from Garmin and the others, and what it sends out. Everything stays on this computer.</p>
    <div class="card syncsum"><div class="row"><span class="syncicon big s-${lv}">${cloudIcon(lv)}</span>
      <div class="grow"><b>${esc(syncStatus.tip(lv))}</b><div class="small muted">${syncStatus.note()}${last ? `<br>${last}` : ""}</div></div>
      ${G.configured ? `<button class="btn primary" data-act="garmin_sync"${lv === "run" ? " disabled" : ""}>Update now</button>` : '<a class="btn primary" href="#settings/keys">Connect Garmin</a>'}
      ${syncStatus.peers().length && LOCAL ? '<button class="btn" data-act="peer_sync_all">Sync other computer</button>' : ""}</div></div>
    <div class="tablewrap card flush"><table class="synctable">
      <thead><tr><th>Data</th><th>From / to</th><th>Newest</th><th>Last checked</th><th>State</th></tr></thead>
      <tbody>${this.rows().map((r) => `<tr>${r.map((c, i) => `<td${i === 2 || i === 3 ? ' class="num"' : ""}>${i < 4 ? esc(c) : c}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    <p class="small muted">Keys, automatic checks and the ride apps are set in Settings: <a href="#settings/keys">API keys</a>, <a href="#settings/garmin">Garmin connection</a>, <a href="#settings/icu">intervals.icu</a>, <a href="#settings/incyclist">Incyclist</a>.</p>
    <h2>Last update log</h2>
    ${(jobs.JOB?.log || []).length ? "" : '<p class="small muted">Nothing has run since the app started.</p>'}
    ${jobs.jobLog()}`;
  }

  wire() {}
}
export const syncPage = new SyncPage();
