// Today: one day at a time (#today, #today/2026-11-25): the verdict and the session in one card, that morning's numbers,
// the next session; ‹ › (and the arrow keys, a swipe on a phone) step through the days (TodayPage). Sync state and setup
// tasks are in the top bar (status_bar.js).
import {Segmented} from "../controls.js";
import {Page} from "./page.js";
import {api, S} from "../api.js";
import {$, addDays, esc, fmt, hhmm} from "../util.js";
import {delta, ICON, morningsStrip, niceDate, profileSVG, rideIt, markButtons, sessionOn, sessions, signsRow,
  STATUS_LABEL, stepsList, TYPE_LABEL, today} from "../widgets.js";
import {router} from "../main.js";
import {markers} from "../markers.js";
import {cloudIcon, morningLabel, wellnessAge} from "../status_bar.js";

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;
const NO_VERDICT = '<svg viewBox="0 0 24 24"><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17.5v.01"/></svg>';

class TodayPage extends Page {
  // the day shown: #today/YYYY-MM-DD, else today
  day() {
    const a = router.routeArg(), td = today();
    return ISO_DAY.test(a) && this.range() && a >= this.range()[0] && a <= this.range()[1] ? a : td;
  }

  // the days there is something to show: the first day with morning numbers or load .. the last planned session
  range() {
    const td = today();
    if (!td) return null;
    const first = S.series?.date?.[0] || S.pmc?.date?.[0] || td, last = sessions().at(-1)?.date || td;
    return [first < td ? first : td, last > td ? last : td];
  }

  dayHref(ds) { return ds === today() ? "#today" : `#today/${ds}`; }

  step(n) {
    const r = this.range(), d = this.day();
    if (!r) return;
    const to = addDays(d, n);
    if (to >= r[0] && to <= r[1]) location.hash = this.dayHref(to);
  }

  header(d) {
    const td = today(), r = this.range(), off = d === td ? 0 : d < td ? -1 : 1;
    const rel = d === td ? null : d === addDays(td, -1) ? "Yesterday" : d === addDays(td, 1) ? "Tomorrow" : null;
    const title = d === td ? (S.name ? `Hi ${esc(S.name)}` : "Today") : rel || niceDate(d, {weekday: "long"});
    const prev = addDays(d, -1), next = addDays(d, 1);
    const arrow = (to, dir, label) => {
      const ok = r && to >= r[0] && to <= r[1];
      return `<a class="btn sm ghost circle dayarrow" ${ok ? `href="${this.dayHref(to)}"` : 'aria-disabled="true"'} aria-label="${label}" title="${label} (${dir === -1 ? "←" : "→"})">${dir === -1 ? "‹" : "›"}</a>`;
    };
    return `<div class="dayhead">
      ${arrow(prev, -1, "Previous day")}
      <div class="daytitle"><h1>${title}</h1><p class="sub">${niceDate(d, {weekday: "long", day: "numeric", month: "long"})}</p></div>
      ${arrow(next, 1, "Next day")}
      ${off ? `<a class="chip today backtoday" href="#today">Back to today</a>` : ""}
    </div>`;
  }

  // no Garmin and no mornings: the one setup step that keeps today's verdict away gets a line here
  nudge(d) {
    if (d !== today() || wellnessAge() != null || S.garmin_api?.configured) return "";
    return `<div class="nudge">${cloudIcon("off")}<span><b>Connect Garmin</b> to get a readiness verdict: your sleep, HRV and resting HR
      each morning. The Personal API key goes in once.</span><span class="spacer"></span><a class="btn sm primary" href="#settings/keys">Connect Garmin</a></div>`;
  }

  feel() {
    const ov = S.plan?.override || "";
    return `<div class="feel"><span>How do you feel?</span><div class="seg" id="feel">
      <button data-v="" class="${ov === "" ? "on" : ""}">Normal</button>
      <button data-v="off" class="${ov === "off" ? "on" : ""}">A bit off</button>
      <button data-v="sick" class="${ov === "sick" ? "on" : ""}">Sick</button></div></div>`;
  }

  noVerdict(title, basis, why, feel = "") {
    return `<div class="verdict v-none"><div class="vword"><span class="vbadge">${NO_VERDICT}</span>
      <div><h2>${title}</h2><div class="basis">${basis}</div></div></div><p class="vwhy">${why}</p>${feel}</div>`;
  }

  verdict(d) {
    const td = today(), rd = S.plan?.readiness || {};
    if (d > td) return this.noVerdict("Not yet", "The verdict comes from that morning", "goodpush judges each day from that morning's sleep, HRV and resting HR, and adjusts the session if it needs to.");
    if (d < td) {
      const m = (rd.mornings || []).find((x) => x.date === d);
      if (!m) return this.noVerdict("No verdict kept", "goodpush keeps the last 7 mornings' verdicts", "That morning's numbers are below, when there are any.");
      return `<div class="verdict v-${esc(m.status)}"><div class="vword"><span class="vbadge">${ICON[m.status] || ""}</span>
        <div><h2>${esc(STATUS_LABEL[m.status] || m.status)}</h2><div class="basis">That morning's verdict</div></div></div>
        <p class="vwhy">${m.reasons.length ? m.reasons.map(esc).join("; ") : "Nothing flagged in sleep, HRV, resting HR or load."}</p></div>`;
    }
    const age = wellnessAge();
    if (age == null) return this.noVerdict("No verdict yet", "Readiness needs your mornings",
      "goodpush judges readiness from your sleep, HRV and resting HR. The first verdict comes with your first morning from Garmin.", this.feel());
    const st = rd.status || "green", reasons = rd.reasons || [];
    const basis = age === 0 ? "From this morning's sleep and HRV" : `From ${morningLabel(S.today.wellness_date)}'s data · this morning isn't in yet`;
    const more = `${signsRow(rd)}${morningsStrip(rd)}
      ${reasons.length > 1 || rd.notes?.length ? `<ul class="reasons">${reasons.slice(1).map((r) => `<li>${esc(r)}</li>`).join("")}${(rd.notes || []).map((r) => `<li class="muted">${esc(r)}</li>`).join("")}</ul>` : ""}
      ${markers.illWatchLine()}${markers.overWatchLine()}${markers.alphaLine()}`;
    return `<div class="verdict v-${esc(st)}"><div class="vword"><span class="vbadge">${ICON[st] || ""}</span>
      <div><h2>${esc(STATUS_LABEL[st] || st)}</h2><div class="basis">${basis}</div></div></div>
      <p class="vwhy">${esc(reasons[0] || "Nothing flagged in sleep, HRV, resting HR or load.")}</p>
      ${this.feel()}
      <details class="dayfold" id="todaywhy"><summary>Why</summary><div class="whybody">${more}</div></details></div>`;
  }

  session(d) {
    const td = today(), pl = S.plan || {};
    const proposal = d === td && pl.proposal?.length ? `<div class="planline"><b>Plan update proposed</b>
      <span class="muted">${pl.proposal.length} change${pl.proposal.length > 1 ? "s" : ""} from how your recent sessions went</span>
      <span class="spacer"></span><a href="#calendar">Review</a></div>` : "";
    const label = d === td ? "Today's session" : d < td ? "What you rode" : "Planned";
    const s = sessionOn(d);
    if (d < td && !s) return `<div class="daysession"><div class="eyebrow">${label}</div>${this.ridden(d)}</div>`;
    if (!s) {
      const body = sessions().length ? `<h3>Rest day</h3><p class="why">Nothing planned.</p>`
        : `<h3>No plan yet</h3><p class="why">The plan fills in once goodpush has rides to learn from${S.rides?.length ? "" : ": connect Garmin or import your history"}. Or run <code>goodpush plan</code>.</p>`;
      return `<div class="daysession"><div class="eyebrow">${label}</div>${body}${proposal}</div>`;
    }
    if (s.type === "rest") return `<div class="daysession"><div class="eyebrow">${label}</div><h3>Rest day</h3>
      <p class="why">${esc(s.why)}</p>${s.adjusted ? `<div class="adj">${esc(s.adjusted)}</div>` : ""}${proposal}</div>`;
    const steps = stepsList(s);
    return `<div class="daysession"><div class="eyebrow">${label}</div>
      <h3>${esc(s.title)}</h3>
      <div class="meta">${TYPE_LABEL[s.type] || s.type} · ${hhmm(s.minutes)} · ${fmt(s.tss)} TSS · FTP ${fmt(pl.ftp)} W</div>
      ${profileSVG(s)}
      ${s.adjusted ? `<div class="adj">⚠ ${esc(s.adjusted)}</div>` : ""}
      ${d === td ? markButtons(s) : ""}
      <details class="dayfold" id="todaysteps"><summary>Steps and why</summary><div class="whybody">${steps}
        ${s.note ? `<p class="why">${esc(s.note)}</p>` : ""}<p class="why">${esc(s.why)}</p>
        ${d !== td ? markButtons(s) : ""}</div></details>
      ${proposal}</div>`;
  }

  // a past day: its rides (the load table), linked to the ride page when it's among the recent rides
  ridden(d) {
    const P = S.pmc || {}, i = (P.date || []).indexOf(d);
    const rides = (S.rides || []).filter((r) => r.date === d);
    if (rides.length) return rides.map((r) => `<h3><a href="#ride/${encodeURIComponent(r.id)}">${esc(r.name || "Ride")}</a></h3>
      <div class="meta">${hhmm(r.min)} · ${fmt(r.tss)} TSS${r.np ? ` · NP ${fmt(r.np)} W` : ""}</div>`).join("");
    if (i >= 0 && P.tss?.[i] > 0) return `<h3>${esc(P.title?.[i] || "Ride")}</h3><div class="meta">${hhmm((P.hours?.[i] || 0) * 60)} · ${fmt(P.tss[i])} TSS</div>`;
    return `<h3>No ride</h3><p class="why">Nothing recorded that day.</p>`;
  }

  // that morning's numbers: today's from S.today (with your usual), another day's from the daily series
  strip(d) {
    const td = today();
    let t = null;
    if (d === td) t = wellnessAge() === 0 ? S.today : null;
    else if (d < td) {
      const D = S.series || {}, i = (D.date || []).indexOf(d);
      if (i >= 0) t = Object.fromEntries(Object.keys(D).map((k) => [k, D[k][i]]));
    }
    if (!t || [t.sleep_h, t.hrv, t.rhr].every((v) => v == null)) return "";
    const hrvRange = t.hrv_base_low && t.hrv_base_high ? `range ${fmt(t.hrv_base_low)}–${fmt(t.hrv_base_high)}` : "";
    const hrvCls = t.hrv == null || !t.hrv_base_low ? "" : t.hrv < t.hrv_base_low ? "down" : t.hrv > t.hrv_base_high ? "mid" : "up";
    const cell = (k, v, u, sub, cls = "") => `<div class="cell"><div class="k">${k}</div><div class="v">${v}${u ? `<small> ${u}</small>` : ""}</div><div class="s ${cls}">${sub || "&nbsp;"}</div></div>`;
    return `<div><h2 class="striptitle">${d === td ? "This morning" : "That morning"}</h2><div class="strip">
      ${cell("Sleep", hhmm(t.sleep_h * 60, false), "h", delta(t.sleep_h, t.sleep_h_base, " h", true, 1))}
      ${cell("Sleep score", fmt(t.sleep_score), "", t.sleep_score_base ? `usual ${fmt(t.sleep_score_base)}` : "")}
      ${cell("HRV", fmt(t.hrv), "ms", hrvRange, hrvCls)}
      ${cell("Resting HR", fmt(t.rhr), "bpm", delta(t.rhr, t.rhr_base, " bpm", false))}
      ${cell("Body battery", fmt(t.body_battery_max), "", "overnight high")}
      ${cell("Form (TSB)", (t.tsb > 0 ? "+" : "") + fmt(t.tsb), "", `load ${fmt(t.ctl)} · fatigue ${fmt(t.atl)}`)}
    </div></div>`;
  }

  nextLine(d) {
    if (d < today()) return "";
    const n = sessions().find((x) => x.date > d && x.type !== "rest");
    if (!n) return "";
    return `<div class="nextline"><span class="muted">Next</span><a href="${this.dayHref(n.date)}"><b>${esc(niceDate(n.date))}: ${esc(n.title)}</b></a>
      <span class="muted">${hhmm(n.minutes)} · ${fmt(n.tss)} TSS</span><span class="spacer"></span><a href="#plan">Week plan →</a></div>`;
  }

  html() {
    const td = today();
    if (!td) return `<h1>${S.name ? `Hi ${esc(S.name)}` : "Today"}</h1>${this.nudge(td)}<div class="card muted">No data yet. Connect Garmin
      (the cloud icon at the top) or drop your Garmin or Strava export (.zip) anywhere on this page.</div>`;
    const d = this.day();
    return `${this.header(d)}
      ${this.nudge(d)}
      <div class="daycard">${this.verdict(d)}${this.session(d)}</div>
      ${this.strip(d)}
      ${this.nextLine(d)}`;
  }

  wire() {
    document.querySelectorAll("[data-ride-key]").forEach((b) => b.addEventListener("click", () => rideIt(b.dataset.rideKey)));
    if ($("#feel")) new Segmented("#feel", (b) => api.action({action: "override", value: b.dataset.v}, "Adjusting today's plan…"));
  }
}
export const todayPage = new TodayPage();

// ← / → step through the days; on a phone, a sideways swipe on the page
document.addEventListener("keydown", (e) => {
  if (router.route() !== "today" || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable], .seg, .stpop")) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); todayPage.step(-1); }
  if (e.key === "ArrowRight") { e.preventDefault(); todayPage.step(1); }
});
let touch = null;
document.addEventListener("touchstart", (e) => {
  touch = router.route() === "today" && e.touches.length === 1 && e.target.closest("#main") && !e.target.closest("svg, .seg, details")
    ? {x: e.touches[0].clientX, y: e.touches[0].clientY} : null;
}, {passive: true});
document.addEventListener("touchend", (e) => {
  if (!touch) return;
  const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
  touch = null;
  if (Math.abs(dx) > 70 && Math.abs(dy) < Math.abs(dx) / 2) todayPage.step(dx < 0 ? 1 : -1);
}, {passive: true});
