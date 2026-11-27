// Settings: Garmin / Google Health, other computers, Incyclist, the goodpush.toml sections (SettingsPage).
import {Page} from "./page.js";
import {api, jobs, S, toast} from "../api.js";
import {$, esc, fmt, LOCAL} from "../util.js";
import {hint, niceDate} from "../widgets.js";
import {router} from "../main.js";
import {DAYS, SAVE_LABEL, WIDE_KINDS} from "./plan.js";
import {markers} from "../markers.js";
import {el} from "../charts.js";
import {ask, Pref} from "../controls.js";

// what each goodpush.toml section holds, shown beside its title while it is closed
const SECTION_NOTE = {
  plan: "Hours, rest, key and long-ride days, how hard days are chosen",
  athlete: "Name, weight, heart rates, birth year",
  ftp: "Where your FTP comes from",
  load: "CTL / ATL and TSS without power",
  physio: "Which of your rides it fits on, how often its ranges are refreshed",
  events: "Illness, injury, travel",
  analysis: "Windows for progress and detection",
  ride: "The trainer overlay",
  ingest: "Reading ride files",
  paths: "Where your data lives",
};

const PEERS_TIP = "Pair your computers once; goodpush keeps rides, health data, plan and settings the same.\n" +
  "• Changes on both survive; the same thing changed on both: home wins\n" +
  "• Your own network only, encrypted, paired with a code\n" +
  "• Your Garmin key and other sign-ins never travel";

class SettingsPage extends Page {
  html() {
    const G = S.garmin_api || {}, st = S.settings || {}, I = S.intervals_icu || {}, GH = S.google_health || {};
    const last = G.last_sync?.at ? G.last_sync.at.slice(0, 16).replace("T", " ") : null;
    const ls = G.last_sync || {};
    const garmin = G.configured ? `
      <div class="row">${markers.chipFor("done", "Connected")}<span class="small muted">${last ? `last check ${esc(last)}: ${fmt(ls.days)} day(s), ${fmt(ls.activities)} activit${ls.activities === 1 ? "y" : "ies"}${ls.stopped ? " – " + esc(ls.stopped) : ""}` : "no check yet"}</span></div>
      <div class="row" style="margin-top:12px"><button class="btn primary" data-act="garmin_sync">Update now</button></div>`
      : `<p class="small muted" style="margin:0">Add your Garmin key under ${this.jump("keys", "API keys")}.</p>`;
    const peers = (S.peers?.computers || []).length, inc = S.incyclist || {};
    const lastRun = S.generated ? new Date(S.generated).toLocaleString(undefined, {weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit"}) : "";
    const toml = (secs) => secs.map((sec) => this.settingsCard(sec)).join("");
    this.firstOpen = G.configured ? [] : ["keys"];     // nothing set up yet: start at the keys
    return `
  <h1>Settings</h1>
  <div class="sethead">
    <p class="sub">Saved in <code>goodpush.toml</code> in ${esc(st.root || "your project folder")}; your API keys in your user folder.</p>
    <button class="btn sm" id="setall" type="button">Expand all</button></div>

  <h2 class="setgroup">Connections</h2>
  ${this.section("keys", "API keys", `${this.state(G.configured, "Garmin ✓", "Garmin: not set")} ${this.state(I.configured, "intervals.icu ✓", "intervals.icu: not set")}`, this.keysCard())}
  ${this.section("garmin", "Garmin connection", G.configured ? (last ? `last check ${esc(last)}` : "Connected") : this.state(false, "", "Not connected"), `<div class="card">
    ${garmin}
    <div class="row" style="margin-top:16px">
      <label class="f" style="max-width:240px">Also check Garmin by itself every (hours, 0 = off: only when you press Update now)
        <input type="number" id="autoh" min="0" max="48" step="1" value="${esc(st.auto_sync_hours ?? 0)}"></label>
      <button class="btn sm" id="saveauto" style="align-self:flex-end">Save</button></div>
    <p class="small muted" style="margin:8px 0 0">Off by default; never when the app opens. When on, it only runs while <code>goodpush app</code> is open. Pulls sleep, resting HR, HRV, stress, Body Battery, weight and new rides with their .fit files.</p>
    ${LOCAL ? `<details style="margin-top:12px"><summary class="small">Browser script (fallback; also gets Garmin's Training Readiness score)</summary>
      <div class="steps3" style="margin-top:8px">
        <div><span class="num">1</span><button class="btn sm" id="copysnip">Copy Garmin script</button></div>
        <div><span class="num">2</span><span class="small">Open <a href="https://connect.garmin.com" target="_blank" rel="noopener">connect.garmin.com</a> → F12 → Console → paste → Enter</span></div>
        <div><span class="num">3</span><button class="btn sm" data-act="garmin_import">Import latest download</button></div>
      </div>
      <textarea class="snip" id="snip" hidden readonly></textarea>
      <p class="small muted">Downloads folder: ${esc(st.downloads || "")}</p></details>` : ""}
    ${jobs.jobLog()}
  </div>`)}
  ${GH.client || GH.connected ? this.section("fitbit", "Fitbit (Google Health)", this.state(GH.connected, "Connected", "Not connected"), this.googleHealthCard()) : ""}
  ${this.section("icu", "intervals.icu", I.configured ? (I.last_push?.at ? `last sent ${esc(I.last_push.at.slice(0, 10))}` : "Nothing sent yet") : this.state(false, "", "No key yet"), this.icuCard())}
  ${this.section("incyclist", "Incyclist", !inc.found ? "Not found on this computer" : inc.sync ? "Writing your plan" : "Off", this.incyclistCard())}
  ${this.section("peers", `Other computers ${hint(PEERS_TIP)}`, peers ? `${peers} paired` : "None paired", this.peersCard())}
  ${this.section("phone", "Phone", "Open the app on your phone", `<div class="card small">Start the app with <code>goodpush app --lan</code> and open the link it prints on your phone (same Wi-Fi). Keep that link private – it opens your training and health data.</div>`)}

  <h2 class="setgroup">Training and analysis</h2>
  ${toml(["plan", "athlete", "ftp", "load", "physio", "events", "analysis"])}

  <h2 class="setgroup">App and data</h2>
  ${toml(["ride", "ingest", "paths"])}
  ${this.section("data", "Data", lastRun ? `last analysis ${esc(lastRun)}` : "Not analysed yet", `<div class="card">
    <div class="maint">
      <div><button class="btn" data-act="rebuild">Rebuild everything</button></div>
      <p class="small muted">Re-read every ride file and re-analyse. Use it after importing a Strava or Garmin export.</p>
      <div><button class="btn" id="physiofresh">Fit the model from scratch</button></div>
      <p class="small muted">Fit the physiological model again from its starting assumptions instead of the last fit, then
        update the analysis and plan (the same as <code>goodpush physio --fresh</code>). Use it after a model update or when
        its numbers look implausible. Takes a few minutes, and the 5–95 % ranges are cleared until the next full run on the
        <a href="#model">Model</a> page.</p>
    </div>
    <p class="small muted" style="margin:12px 0 0">Last analysis: ${lastRun ? esc(lastRun) : "never"}</p>
    ${(S.notes || []).length ? `<h3 style="margin:14px 0 4px">Data notes</h3><ul class="small muted" style="margin:0">${S.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
  </div>`)}
  <p class="small muted" style="margin-top:16px">GoodPush ${esc(S.about?.version || "")} (beta) · <a href="#help">Help &amp; feedback</a> · where the methods come from: <a href="#sources">Sources</a>.</p>`;
  }

  // one collapsible section: its title, a one-line status that stays visible when closed, and the card
  section(id, title, status, body) {
    const open = SettingsPage.opened.value !== null ? SettingsPage.openSet().has(id) : this.firstOpen.includes(id);
    return `<details class="setsec" id="set-${id}" data-sec-id="${id}"${open ? " open" : ""}>
    <summary><span class="settitle">${title}</span><span class="setstat small muted">${status || ""}</span>
      <span class="setdirty small" hidden>Unsaved changes</span></summary>
    <div class="setbody">${body}</div></details>`;
  }

  state(ok, yes, no) { return markers.chipFor(ok ? "done" : "amber", ok ? yes : no); }

  jump(id, text) { return `<a href="#settings/${id}">${esc(text)}</a>`; }

  static opened = new Pref("settings.open", null);    // the sections you left open (comma-separated ids)

  static openSet() { return new Set((SettingsPage.opened.value || "").split(",").filter(Boolean)); }

  rememberOpen() {
    SettingsPage.opened.set([...document.querySelectorAll("details.setsec[open]")].map((d) => d.dataset.secId).join(","));
  }

  peersCard() {
    const P = S.peers || {};
    const when = (t) => t ? esc(t.replace("T", " ")) : "not yet";
    // what you call it here, its own computer name under it once renamed
    const named = (x) => `<b>${esc(x.label || x.name)}</b>${x.label && x.label !== x.name ? `<div class="small muted">${esc(x.name)}</div>` : ""}`;
    const rename = (x) => `<button class="btn sm" data-peer-rename="${esc(x.name)}" data-label="${esc(x.label || x.name)}">Rename</button>`;
    // one row per paired computer: send to it, get from it, or both ways (a pairing from before both-ways linking
    // only lets the other one start)
    const rows = (P.computers || []).map((c) => `<tr><td>${named(c)}${c.home ? ' <span class="small muted">home</span>' : ""}</td>
      <td class="small muted">${esc(c.address || "")}</td>
      <td class="small">${c.can_start ? `Last sync ${when(c.last)}${c.result && c.result !== "ok" ? `<div class="small muted">${esc(c.result)}</div>` : ""}`
          : "Syncs with this one; pair again to start syncs from here too"}</td>
      <td>${LOCAL ? `${c.can_start ? `<button class="btn sm" data-peer-sync="${esc(c.name)}" data-dir="send" title="Your rides and data from here to ${esc(c.label)}">Send to</button>
        <button class="btn sm" data-peer-sync="${esc(c.name)}" data-dir="get" title="Its rides, data and analysis to here">Get from</button>
        <button class="btn sm" data-peer-sync="${esc(c.name)}" data-dir="both">Both ways</button> ` : ""}${rename(c)}
        <button class="btn sm" data-peer-forget="${esc(c.name)}">Forget</button>` : ""}</td></tr>`).join("");
    const joined = (P.computers || []).some((c) => c.can_start), devices = (P.computers || []).some((c) => c.lets_in);
    const home = P.home_here
      ? (P.home ? "This computer is the home machine: it downloads from Garmin and does the analysis; the others get the results from it." : "")
      : `The home machine is <b>${esc(P.home_label || P.home)}</b>: it downloads from Garmin and does the analysis; this one gets the results from it.
       ${LOCAL ? `<button class="btn sm" data-peer-home="${esc(P.name)}">Make this one home</button>` : ""}`;
    const code = P.code ? `<div class="adj" style="margin-top:10px">On the other computer: Settings → Other computers → Join, and enter<br>
      address <b>${(P.addresses || []).map(esc).join("</b> or <b>")}</b><br>
      code <b style="font-size:1.25em;letter-spacing:.08em">${esc(P.code)}</b> <span class="small muted">(until ${esc(P.until)})</span><br>
      <span class="small">This computer then asks you to allow it.</span></div>` : "";
    return `<div class="card">
    ${home ? `<p class="small" style="margin:0">${home}</p>` : ""}
    ${P.error ? `<div class="adj">${esc(P.error)}</div>` : ""}
    ${router.peerAsk("adj")}
    ${rows ? `<div class="tw" style="margin-top:10px"><table><tbody>${rows}</tbody></table></div>
      <p class="small muted" style="margin:6px 0 0">Send / Get move rides, health data and the analysis one way; your plan,
        settings and other choices always merge both ways (nothing of yours is lost).</p>` : ""}
    ${joined && LOCAL ? `<div class="row" style="margin-top:10px">
      <label class="f" style="max-width:260px">Sync by itself every (hours, 0 = only when you press Sync)
        <input type="number" id="peerevery" min="0" max="48" step="0.5" value="${esc(P.every_hours ?? 2)}"></label>
      <button class="btn sm" id="saveevery" style="align-self:flex-end">Save</button></div>
      <p class="small muted" style="margin:4px 0 0">Also after each ride, and whenever you press <b>Sync</b> at the top. This computer only.</p>` : ""}
    ${devices && LOCAL ? `<label class="small" style="display:block;margin-top:10px"><input type="checkbox" id="peerserve" ${P.serve ? "checked" : ""}>
      Let the computers paired with this one sync with it</label>
      <p class="small muted" style="margin:4px 0 0">${P.listening ? `Listening on ${esc((P.addresses || [])[0] || "")} – your home network and Tailscale only, never the open internet.`
          : "Not listening: the paired computers can't reach this one."}</p>` : ""}
    ${LOCAL ? `<div class="grid g2" style="margin-top:14px">
      <div><h3>Pair a computer with this one</h3>
        <p class="small muted" style="margin:0 0 8px">Best on the computer that's on most: the others sync with it.</p>
        <button class="btn primary" id="peerpair">Show a code</button>${code}</div>
      <div><h3>Join another computer</h3>
        <label class="f">Address it shows<input type="text" id="peeraddr" placeholder="192.168.1.20:8865" autocomplete="off"></label>
        <label class="f">Code<input type="text" id="peercode" placeholder="ABCDE-FGHJK" autocomplete="off"></label>
        <button class="btn primary" id="peerjoin" style="margin-top:8px">Join</button></div></div>
      <p class="small muted" style="margin:12px 0 0">The home computer passes on whatever it gets (from another computer or from Garmin) to the others it can reach; the others sync when you press Sync, after each ride, and every 2 hours (change it there).
        Windows may ask to let goodpush through its firewall: allow it on private networks.</p>`
      : `<p class="small muted" style="margin:0">Pair computers from the app on each computer.</p>`}
  </div>`;
  }

  incyclistCard() {
    const I = S.incyclist || {};
    const custom = I.folder !== I.default_folder;
    const chip = !I.found ? markers.chipFor("missed", "Incyclist not found") : I.sync ? markers.chipFor("done", "Writing your plan") : markers.chipFor("amber", "Off");
    const days = I.days || [];
    const now = I.entries ? `${fmt(I.entries)} goodpush workouts in its calendar (${esc(niceDate(days[0], {day: "numeric", month: "short"}))} – ${esc(niceDate(days[days.length - 1], {day: "numeric", month: "short"}))})${I.written ? `, file written ${esc(I.written.slice(0, 16).replace("T", " "))}` : ""}.` : "No goodpush workouts in its calendar yet.";
    return `<div class="card">
    <div class="row">${chip}<span class="small muted">${esc(I.calendar || "")}</span></div>
    <p class="small" style="margin:10px 0 0">After every plan update goodpush writes your sessions into Incyclist's own workout calendar on this PC
      (no account, nothing online). They show up in Incyclist under <i>Workouts → Scheduled Workouts</i> for this week and the next 7 days.</p>
    <p class="small muted" style="margin:6px 0 0">${now}</p>
    ${I.running ? `<div class="adj">Incyclist is open: it reads its calendar when it starts, so restart it to see changes.</div>` : ""}
    ${LOCAL ? `
    <div class="grid g2" style="margin-top:14px">
      <label class="f">Incyclist data folder (empty = the usual place)
        <input type="text" id="incfolder" value="${esc(custom ? I.folder : "")}" placeholder="${esc(I.default_folder || "")}" spellcheck="false"></label>
      <label class="f" style="flex-direction:row;align-items:center;gap:8px;align-self:end;min-height:40px">
        <input type="checkbox" id="incsync" ${I.sync ? "checked" : ""} style="width:18px;height:18px"> Write my plan into Incyclist</label>
    </div>
    <div class="row" style="margin-top:12px">
      <button class="btn primary" id="saveinc">Save</button>
      ${I.sync ? '<button class="btn" data-act="incyclist_sync">Send plan now</button>' : ""}</div>` :
      `<p class="small muted" style="margin:10px 0 0">Change these on the PC running goodpush.</p>`}
    <ul class="reasons small" style="margin-top:12px">
      <li>Targets are % of FTP: set your FTP in Incyclist too (${fmt(S.plan?.ftp)} W here).</li>
      <li>Only goodpush's own entries are replaced; past days of the last two weeks stay.</li>
      <li>If you connect an online calendar for workout downloads inside Incyclist, its sync replaces this list.</li>
      <li>The same workouts are also in <code>reports/plan/workouts</code> as .zwo files to import by hand.</li></ul>
  </div>`;
  }

  icuCard() {
    const I = S.intervals_icu || {}, lp = I.last_push || {};
    const sent = lp.at ? `Last sent ${esc(lp.at.slice(0, 16).replace("T", " "))}: ${fmt(lp.sent)} workout(s)${lp.deleted ? `, ${fmt(lp.deleted)} removed` : ""}.` : "Nothing sent yet.";
    const body = I.configured ? `
    <div class="row"><button class="btn primary" data-act="icu_push">Send plan now</button><span class="small muted">${esc(sent)}</span></div>`
      : `<p class="small muted" style="margin:0">Add your intervals.icu key under ${this.jump("keys", "API keys")}.</p>`;
    return `<div class="card">
    ${body}
    <p class="small" style="margin:12px 0 0">This is how your workouts get to your watch, bike computer and training apps: goodpush sends the next
      ${fmt(I.days || 14)} days of your plan to your intervals.icu calendar, and intervals.icu passes them on.</p>
    <ol class="small" style="margin:8px 0 0 18px;padding:0;line-height:1.6">
      <li>Make a free account at <a href="https://intervals.icu" target="_blank" rel="noopener">intervals.icu</a> and add its API key under ${this.jump("keys", "API keys")}.</li>
      <li>In intervals.icu → <a href="https://intervals.icu/settings" target="_blank" rel="noopener">Settings</a> → <i>Connections</i>, connect your
        Garmin, Wahoo, Zwift or MyWhoosh account and switch on sending planned workouts to it.</li>
      <li>Press <b>Send plan now</b> here, and again after the plan changes.</li></ol>
    <ul class="reasons small" style="margin-top:8px">
      <li>Sending again updates goodpush's workouts there; ones the plan dropped are removed.</li>
      <li>Past days and workouts you made in intervals.icu yourself are left alone.</li>
      <li>MyWhoosh: connect it in intervals.icu; it takes the next 7 days (ramps ride as a steady average, no cadence targets).</li>
      <li>Targets are % of FTP: keep your FTP in intervals.icu at ${fmt(S.plan?.ftp)} W too.</li></ul>
    <p class="small muted" style="margin:8px 0 0">intervals.icu is free to use and is paid for by the riders who support it. If it gets your plan to
      your watch, please consider <a href="https://intervals.icu" target="_blank" rel="noopener">becoming a supporter</a>.</p>
  </div>`;
  }

  googleHealthCard() {
    const G = S.google_health || {}, ls = G.last_sync || {};
    const body = G.connected ? `
    <div class="row">${markers.chipFor("done", "Connected")}<span class="small muted">${ls.at ? `last update ${esc(ls.at.slice(0, 16).replace("T", " "))}: ${fmt(ls.new)} new data point(s)` : "first update running"}</span></div>
    <div class="row" style="margin-top:12px"><button class="btn primary" data-act="gh_sync">Update now</button>
      ${LOCAL ? '<button class="btn" id="ghforget">Disconnect</button>' : ""}</div>`
      : LOCAL ? `
    <ol class="small" style="margin:0 0 10px 18px;padding:0;line-height:1.6">
      <li>In the <b>Google Health</b> app, sign in with a personal Google account (move the Fitbit account to it when asked).</li>
      <li>At <a href="https://console.cloud.google.com" target="_blank" rel="noopener">console.cloud.google.com</a> (free): create a project and enable the <b>Google Health API</b>.
        Set up the OAuth consent screen (External, add yourself as a test user), then <b>Publish app</b> – left in Testing, Google signs you out every 7 days; under 100 users it needs no review.
        Then Credentials → <b>OAuth client ID</b>, type <b>Desktop app</b>.</li>
      <li>Paste its ID and secret here and connect. They stay on this PC, in goodpush's own folder (not the project).</li></ol>
    <div class="row"><input id="ghid" placeholder="Client ID (…apps.googleusercontent.com)" autocomplete="off" spellcheck="false" style="flex:2;min-width:220px">
      <input type="password" id="ghsecret" placeholder="Client secret" autocomplete="off" spellcheck="false" style="flex:1;min-width:140px">
      <button class="btn primary" id="ghconnect">Connect</button></div>
    ${G.client ? '<p class="small muted" style="margin:8px 0 0">A client is saved: <a href="/oauth/google/start">connect with it</a>.</p>' : ""}`
      : '<p class="small muted" style="margin:0">Connect from the app on your PC.</p>';
    return `<div class="card">${body}
    <p class="small muted" style="margin:10px 0 0">For a Fitbit (the Fitbit API itself shuts down on 2 Nov 2026). Sleep and stages, resting HR, HRV,
      breathing rate, SpO2 and skin temperature, updated with the Garmin check. Where Garmin has a value it wins; Fitbit fills the gaps
      and adds deep-sleep HRV, SpO2 and skin temperature.</p></div>`;
  }

  keysCard() {
    const G = S.garmin_api || {}, I = S.intervals_icu || {};
    const saved = (k) => `${markers.chipFor("done", "Saved")}<span class="small muted">${esc((k.saved || "").slice(0, 10))}</span>`;
    const missing = markers.chipFor("amber", "Not set");
    if (!LOCAL) return `<div class="card small muted">Garmin ${G.configured ? "✓" : "–"} · intervals.icu ${I.configured ? "✓" : "–"}.
      Add or change keys from the app on your PC (they never travel over Wi-Fi).</div>`;
    return `<div class="card">
    <p class="small muted" style="margin:0 0 12px">Stored only on this PC, in goodpush's own folder, never in the project or sent to other computers.</p>
    <h3 style="margin:0 0 6px">Garmin Personal API</h3>
    <div class="row">${G.configured ? saved(G) + '<span class="small muted">· Garmin emails you before it expires</span>' : missing}</div>
    ${G.configured ? `<div class="row" style="margin-top:8px"><button class="btn sm" id="testkey">Test key</button><button class="btn sm" id="forgetkey">Remove key</button></div>
      <details style="margin-top:8px"><summary class="small">Replace the key</summary>${this.keyInput("Save new key")}</details>` : `
      <ol class="small" style="margin:8px 0 10px 18px;padding:0;line-height:1.6">
        <li>Open <a href="https://connect.garmin.com/app/settings/personalApiToken" target="_blank" rel="noopener">Garmin Connect → Settings → Personal API Key</a> (signed in).</li>
        <li>Create a key (pick 365 days) and copy it – Garmin shows it only once.</li>
        <li>Paste it below.</li></ol>
      ${this.keyInput("Save &amp; update")}`}
    <h3 style="margin:18px 0 6px">intervals.icu</h3>
    <div class="row">${I.configured ? saved(I) : missing}</div>
    ${I.configured ? `<div class="row" style="margin-top:8px"><button class="btn sm" id="icuforget">Remove key</button></div>
      <details style="margin-top:8px"><summary class="small">Replace the key</summary>${this.icuKeyInput()}</details>` : `
      <ol class="small" style="margin:8px 0 10px 18px;padding:0;line-height:1.6">
        <li>Open <a href="https://intervals.icu/settings" target="_blank" rel="noopener">intervals.icu → Settings</a> (signed in), scroll to <i>Developer Settings</i>.</li>
        <li>Click <i>API key</i>, even if it says (none), then generate the key and copy it.</li>
        <li>Paste it below.</li></ol>
      ${this.icuKeyInput()}`}
  </div>`;
  }

  icuKeyInput() {
    return `<div class="row"><input type="password" id="icukey" placeholder="intervals.icu API key" autocomplete="off" spellcheck="false" style="flex:1;min-width:200px">
    <button class="btn primary" id="icusave">Save key</button></div>`;
  }

  keyInput(label) {
    return `<div class="row"><input type="password" id="gkey" placeholder="Garmin Personal API key" autocomplete="off" spellcheck="false" style="flex:1;min-width:200px">
    <button class="btn primary" id="savekey">${label}</button></div>`;
  }

  settingsCard(sec) {
    const spec = S.settings?.schema?.[sec]; if (!spec) return "";
    const vals = S.settings.values?.[sec] || {};
    return this.section(sec, esc(spec.title), esc(SECTION_NOTE[sec] || ""), `<div class="card">
    <div class="grid g3">${spec.fields.map((f) => this.settingField(sec, f, vals[f.key])).join("")}</div>
    <div class="row" style="margin-top:14px"><button class="btn primary" data-save="${sec}">${SAVE_LABEL[spec.then]}</button>
      ${spec.then === "nothing" ? '<span class="small muted">Used from the next import / ride</span>' : ""}</div></div>`);
  }

  settingField(sec, f, v) {
    const attrs = `data-sec="${sec}" data-key="${f.key}" data-kind="${f.kind}"`;
    const help = f.help ? `<span class="small muted">${esc(f.help)}</span>` : "";
    const num = (val, extra = "") => `<input type="number" min="${f.zero ? 0 : f.lo ?? ""}" max="${f.hi ?? ""}" step="${f.step ?? 1}" value="${esc(val ?? "")}" ${extra}>`;
    const wide = WIDE_KINDS.includes(f.kind) ? ' style="grid-column:1/-1"' : "";
    let input;
    if (f.kind === "bool") return `<div class="setting" ${attrs}><label><input type="checkbox" ${v ? "checked" : ""}> ${esc(f.label)}</label>${help}</div>`;
    if (["int", "float"].includes(f.kind)) input = num(v);
    else if (f.kind === "choice" || f.kind === "day") {
      const opts = f.kind === "day" ? DAYS.map((d) => [d, d]) : f.choices;
      input = `<select>${opts.map(([k, l]) => `<option value="${esc(k)}" ${k === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
    } else if (f.kind === "days") input = `<div class="days">${DAYS.map((d) =>
      `<button type="button" data-d="${d}" class="${(v || []).includes(d) ? "on" : ""}">${d[0].toUpperCase() + d.slice(1)}</button>`).join("")}</div>`;
    else if (f.kind === "week") input = `<div class="avail">${DAYS.map((d, i) =>
      `<label class="f">${d[0].toUpperCase() + d.slice(1)}${num((v || [])[i])}</label>`).join("")}</div>`;
    else if (f.kind === "numbers") input = `<div class="avail">${f.keys.map((k) =>
      `<label class="f">${esc(k)}${num((v || {})[k], `data-n="${k}"`)}</label>`).join("")}</div>`;
    else if (f.kind === "history") input = `<textarea class="snip" rows="4">${esc(Object.entries(v || {}).map(([d, w]) => `${d} ${w}`).join("\n"))}</textarea>`;
    else if (f.kind === "events") input = `<textarea class="snip" rows="4">${esc((v || []).map((e) =>
      [e.date, e.end || e.date, e.type, e.note || ""].join(", ")).join("\n"))}</textarea>`;
    else input = `<input type="text" value="${esc(v ?? "")}" ${f.hi ? `maxlength="${f.hi}"` : ""}>`;
    return wide ? `<div class="setting" ${attrs}${wide}><span>${esc(f.label)}</span>${input}${help}</div>`
      : `<div class="setting" ${attrs}><label class="f">${esc(f.label)}${input}</label>${help}</div>`;
  }

  settingValues(sec) {          // what the card's inputs hold, in the shapes the server checks
    const out = {};
    document.querySelectorAll(`[data-sec="${sec}"]`).forEach((el) => {
      const kind = el.dataset.kind, inputs = [...el.querySelectorAll("input, select, textarea")];
      out[el.dataset.key] =
        kind === "bool" ? inputs[0].checked
        : kind === "days" ? [...el.querySelectorAll("button.on")].map((b) => b.dataset.d)
        : kind === "week" ? (inputs.every((x) => x.value === "") ? [] : inputs.map((x) => Number(x.value || 0)))
        : kind === "numbers" ? Object.fromEntries(inputs.map((x) => [x.dataset.n, Number(x.value || 0)]))
        : ["int", "float"].includes(kind) ? Number(inputs[0].value || 0)
        : inputs[0].value;
    });
    return out;
  }

  wire() {
    const secs = [...document.querySelectorAll("details.setsec")];
    const allBtn = $("#setall");
    const label = () => { allBtn.textContent = secs.every((d) => d.open) ? "Collapse all" : "Expand all"; };
    secs.forEach((d) => d.addEventListener("toggle", () => { this.rememberOpen(); label(); }));
    allBtn.addEventListener("click", () => {
      const open = !secs.every((d) => d.open);
      secs.forEach((d) => { d.open = open; });
    });
    label();
    // an edit not saved yet shows on the section's title, so closing it doesn't hide it
    secs.forEach((d) => {
      const mark = () => { d.querySelector(".setdirty").hidden = false; };
      d.querySelector(".setbody").addEventListener("input", mark);
      d.querySelectorAll(".days button").forEach((b) => b.addEventListener("click", mark));
    });
    // #settings/<section>: open that section and scroll to it (links from Today, Plan and this page)
    const want = router.routeArg();
    const target = want && $(`#set-${CSS.escape(want)}`);
    if (target && this.jumped !== location.hash) {
      this.jumped = location.hash;
      target.open = true;
      requestAnimationFrame(() => target.scrollIntoView({block: "start"}));
    }
    if (!want) this.jumped = "";
    $("#physiofresh")?.addEventListener("click", async () => {
      if (!await ask.confirm({title: "Fit the model from scratch?", yes: "Fit from scratch", danger: true,
        text: "It starts from the model's priors instead of the last fit (the same as goodpush physio --fresh), then "
          + "updates the analysis and plan.\n\nA stopped sampling run's checkpoint is dropped, and the 5–95 % ranges "
          + "are cleared until the next full run on the Model page."})) return;
      api.action({action: "physio_fresh"}, "Fitting the model from scratch…");
    });
    const sk = $("#savekey");
    if (sk) sk.addEventListener("click", async () => {
      const k = $("#gkey").value.trim();
      if (!k) { toast.show("Paste your key first"); return; }
      sk.disabled = true;
      await api.action({action: "garmin_key", key: k}, "Key accepted – fetching your Garmin data…");
      sk.disabled = false; $("#gkey").value = "";
      api.load();
    });
    const ghc = $("#ghconnect");
    if (ghc) ghc.addEventListener("click", async () => {
      const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
        body: JSON.stringify({action: "gh_client", values: {client_id: $("#ghid").value, client_secret: $("#ghsecret").value}})});
      const j = await r.json().catch(() => ({}));
      if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return; }
      location.href = j.go;                       // Google's consent page, then back to Settings
    });
    $("#ghforget")?.addEventListener("click", async () => {
      await api.action({action: "gh_forget"}, "Disconnected from Google Health");
      api.load();
    });
    const fk = $("#forgetkey");
    if (fk) fk.addEventListener("click", async () => {
      await api.action({action: "garmin_forget"}, "Garmin key removed from this PC");
      api.load();
    });
    const cp = $("#copysnip");
    if (cp) {
      let snip = null;
      fetch("/api/snippet").then((r) => r.text()).then((t) => { snip = t; }).catch(() => {});
      cp.addEventListener("click", async () => {
        // 1) browser clipboard, synchronously inside the click (needs the text preloaded)
        let ok = false;
        if (snip) {
          const t = $("#snip"); t.hidden = false; t.value = snip; t.select();
          try { ok = document.execCommand("copy"); } catch { ok = false; }
          if (!ok && navigator.clipboard) { try { await navigator.clipboard.writeText(snip); ok = true; } catch { ok = false; } }
          if (ok) t.hidden = true;
        }
        // 2) let the goodpush server put it on the PC clipboard
        if (!ok) {
          try {
            const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
              body: JSON.stringify({action: "copy_snippet"})});
            ok = !!(await r.json()).copied;
            if (ok) $("#snip").hidden = true;
          } catch { ok = false; }
        }
        toast.show(ok ? "Script copied – paste it in the Garmin Connect console" : "Couldn't copy – select the text below and press Ctrl+C");
        if (!ok) { const t = $("#snip"); t.hidden = false; t.value = snip || ""; t.focus(); t.select(); }
      });
    }
    $("#icusave")?.addEventListener("click", async () => {
      const k = $("#icukey").value.trim();
      if (!k) { toast.show("Paste your key first"); return; }
      await api.action({action: "icu_key", key: k}, "Key accepted by intervals.icu");
      api.load();
    });
    $("#icuforget")?.addEventListener("click", async () => {
      await api.action({action: "icu_forget"}, "intervals.icu key removed from this PC");
      api.load();
    });
    $("#saveinc")?.addEventListener("click", () => {
      const on = $("#incsync").checked;
      const body = {action: "incyclist_settings", values: {sync: on, folder: $("#incfolder").value.trim()}};
      if (on) api.action(body, "Saved – writing your plan into Incyclist…");
      else api.saveOnly(body, "Saved – the plan isn't written into Incyclist any more");
    });
    $("#saveevery")?.addEventListener("click", () => {
      const h = Number($("#peerevery").value || 0);
      api.saveOnly({action: "peer_every", hours: h}, h ? `Syncs by itself every ${h} h` : "Syncs only when you press Sync (and after a ride)");
    });
    $("#peerserve")?.addEventListener("change", (e) => api.saveOnly({action: "peer_serve", on: e.target.checked},
      e.target.checked ? "Paired computers can sync with this one" : "Not listening for other computers any more"));
    $("#peerpair")?.addEventListener("click", () => api.saveOnly({action: "peer_pair"}, "Enter the code on the other computer"));
    $("#peerjoin")?.addEventListener("click", async () => {
      const b = $("#peerjoin"); b.disabled = true; b.textContent = "Allow it on the other computer…";
      await api.action({action: "peer_join", address: $("#peeraddr").value.trim(), code: $("#peercode").value.trim()},
        "Paired – syncing now");
      api.load();
    });
    document.querySelectorAll("[data-peer-sync]").forEach((b) => b.addEventListener("click", () =>
      api.action({action: "peer_sync", name: b.dataset.peerSync, direction: b.dataset.dir || "both"},
        {send: "Sending…", get: "Getting…", both: "Syncing…"}[b.dataset.dir || "both"])));
    document.querySelectorAll("[data-peer-rename]").forEach((b) => b.addEventListener("click", async () => {
      const label = await ask.prompt({title: `Rename ${b.dataset.peerRename}`, label: "Its name on this computer",
        text: "Leave it empty to use its own name.", value: b.dataset.label, yes: "Rename"});
      if (label !== null) api.saveOnly({action: "peer_rename", name: b.dataset.peerRename, label: label.trim()}, "Renamed");
    }));
    document.querySelectorAll("[data-peer-forget]").forEach((b) => b.addEventListener("click", async () => {
      if (await ask.confirm({title: `Stop syncing with ${b.dataset.peerForget}?`, yes: "Stop syncing", danger: true,
        text: "Nothing is deleted. Pair the computers again to sync again."}))
        api.saveOnly({action: "peer_forget", name: b.dataset.peerForget}, "Done");
    }));
    document.querySelectorAll("[data-peer-home]").forEach((b) => b.addEventListener("click", () =>
      api.saveOnly({action: "peer_home", name: b.dataset.peerHome}, "This computer is home now")));
    document.querySelectorAll(".days button").forEach((b) => b.addEventListener("click", () => b.classList.toggle("on")));
    $("#testkey")?.addEventListener("click", async () => {
      const r = await fetch("/api/action", {method: "POST", headers: {"Content-Type": "application/json", "X-GoodPush": "1"},
        body: JSON.stringify({action: "garmin_test"})});
      const j = await r.json().catch(() => ({}));
      toast.show(j.error || j.message || "Checked");
    });
    $("#saveauto").addEventListener("click", () => api.saveOnly({action: "settings", section: "garmin",
      values: {auto_sync_hours: Number($("#autoh").value || 0)}}, "Saved"));
    document.querySelectorAll("[data-save]").forEach((b) => b.addEventListener("click", () => {
      const sec = b.dataset.save, body = {action: "settings", section: sec, values: this.settingValues(sec)};
      if (["reanalyse", "replan"].includes(S.settings.schema[sec].then)) api.action(body, "Saved – updating…");
      else api.saveOnly(body, "Saved");
    }));
  }
}
export const settingsPage = new SettingsPage();
