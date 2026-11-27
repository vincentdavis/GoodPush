// One ride: charts, selection, intervals, ramp / alpha1 cards (RidePage).
import {ask, Pref} from "../controls.js";
import {Page} from "./page.js";
import {api, S, toast} from "../api.js";
import {$, boxSigma, css, expSmooth, esc, fmt, hhmm, mmss} from "../util.js";
import {hint, niceDate, tile, zoneVar} from "../widgets.js";
import {router} from "../main.js";
import {markers} from "../markers.js";
import {HR_ZONE_NAMES, ZONE_NAMES} from "./rides.js";
import {rideMap} from "../ride_map.js";
import {calendarPage} from "../calendar/calendar.js";
import {EChart, el, niceTicks, selectGroups, SelectionPanel, TimeWindow} from "../charts.js";

// interval zones (models/intervals.py): label and colour
const IZONE = {recovery: ["Recovery", "--z1"], endurance_low: ["Lower endurance", "--z2"], endurance_high: ["Upper endurance", "--z3"],
  tempo: ["Tempo", "--z4"], sweetspot: ["Sweet spot", "--z5"], threshold: ["Threshold", "--z6"], vo2max: ["VO2max", "--z7"], anaerobic: ["Anaerobic", "--s7"]};
const STAMINA_TIP = "What you had left as the ride went:\n"
  + "• Stamina: the share of your W′ left (efforts above CP); 0 % = nothing left above CP\n"
  + "• Potential: your 5-min power now vs fresh; long riding above LT1 wears it down\n"
  + "• An estimate from published models (Sources), not a measurement";
const izoneOf = (IV, w) => IV.zones[Math.max(0, IV.zone_floors.findLastIndex((f) => f <= w))];

class RidePage extends Page {
  RIDE = null;
  rideSmooth = 0;                        // power smoothing window (s), 0 = raw

  // ride charts: a selection (minutes, its stats from the server) and a zoomed window with finer data
  rideSel = {range: null, onChange: (r) => this.rideSelect(r)};

  rideViewSt = null;
  rideSelStats = null;
  selPanel = null;                          // SelectionPanel, made on first draw (charts.js is no leaf module)
  rideZoomT;   // finer streams for the zoomed window

  rideIv = new Pref("ride_iv", false);     // intervals highlighted on the ride charts
  STRAVA_LINE = "#f4a7b9";                 // the power line on the Strava picture: pastel pink
  smoothLabel(s) {             // the smoothing slider's value: Raw, 25 s, 1:30 min
    return s ? s < 60 ? `${s} s` : `${mmss(s)} min` : "Raw";
  }

  selectStretch(r) {          // select [from, to] (chart minutes) on every ride chart
    this.rideSel.range = r;
    (selectGroups.ride || []).forEach((h) => h.show(r));
    this.rideSelect(r);
  }

  // a click on a highlighted interval selects it; on an interval with steps (over-unders) that is already
  // selected (or one of its steps), the step under the click; a click on the selected step goes back to the whole
  pickInterval(IV, tMin) {
    const t = this.rideS(tMin * 60), e = (IV?.intervals || []).find((x) => x.start <= t && t < x.end);
    if (!this.rideIv.value || !e) return;
    const whole = this.ivRange(e.start, e.end), step = this.ivStepRange(e, t), r = this.rideSel.range;
    const inside = !!r && r[0] >= whole[0] - 2e-4 && r[1] <= whole[1] + 2e-4;
    this.selectStretch(step && inside && !this.sameRange(r, step) ? step : whole);
  }

  ivRange(a, b) {             // ride seconds [a, b] -> chart minutes
    return [this.rideT(a) / 60, this.rideT(b) / 60];
  }

  ivStepRange(e, t) {         // the step of interval e at ride second t (chart minutes), if it has more than one
    if (e.steps.length < 2) return null;
    let at = e.start;
    for (const [sec] of e.steps) { if (t < at + sec) return this.ivRange(at, at + sec); at += sec; }
    return null;
  }

  sameRange(r, q) {           // two selections [from, to] (minutes) the same, to a hundredth of a second
    return !!r && Math.abs(r[0] - q[0]) < 2e-4 && Math.abs(r[1] - q[1]) < 2e-4;
  }

  hasMap(a) {                 // a route to draw (beside the charts, above the selection's numbers)
    return (a.track || []).length > 1;
  }

  rideTipNote(IV, tMin) {     // the hover's line for the interval (and its step) at tMin, if any
    const t = this.rideS(tMin * 60), k = (IV?.intervals || []).findIndex((e) => e.start <= t && t < e.end);
    if (k < 0) return "";
    const e = IV.intervals[k];
    let z = e.zone, at = e.start, step = 0;
    for (const [sec, w] of e.steps) { if (t < at + sec) { z = izoneOf(IV, w); step = sec; break; } at += sec; }
    const len = mmss(e.end - e.start), part = z !== e.zone && step ? `${mmss(step)} ` : "";
    return `<div><i style="background:${css(IZONE[z][1])}"></i>Interval ${k + 1} <b>${esc(IZONE[z][0])}</b> ${len}`
      + `${z !== e.zone ? ` (${part}in a ${esc(IZONE[e.zone][0].toLowerCase())} interval)` : ""}</div>`;
  }

  // the planned session's targets as bands on the power chart, in the interval zone colours (a missed rep fainter);
  // steps [from min, to min, low W, high W, 0 easy / 1 work / 2 missed]. They don't stretch the y axis: a sprint's
  // open-ended target is cut at the top of the ride's own power.
  ridePlanBands(plan, IV) {
    return (plan?.steps || []).map(([a, b, lo, hi, k]) => ({kind: "band", x: [this.rideT(a * 60) / 60, this.rideT(b * 60) / 60], y: [lo, lo], y2: [hi, hi], range: false,
      color: css(IZONE[IV ? izoneOf(IV, lo) : "threshold"]?.[1] || "--z6"), opacity: k === 2 ? 0.1 : 0.3}));
  }

  ridePlanColor(a) {           // the session's hardest work step's zone colour (the legend)
    const w = Math.max(0, ...(a.plan?.steps || []).filter((s) => s[4]).map((s) => s[2]));
    return css(IZONE[a.intervals ? izoneOf(a.intervals, w) : "threshold"]?.[1] || "--z6");
  }

  ridePlanNote(plan, tMin) {   // the hover's line for the planned step at tMin (riding time)
    const m = this.rideS(tMin * 60) / 60, s = (plan?.steps || []).find((q) => q[0] <= m && m < q[1]);
    if (!s) return "";
    const what = s[4] === 2 ? "missed rep" : s[4] ? "rep" : "easy";
    const z = this.RIDE?.intervals ? izoneOf(this.RIDE.intervals, s[2]) : "threshold";
    return `<div><i style="background:${css(IZONE[z]?.[1] || "--z6")}"></i>Planned ${what} <b>${s[2] === s[3] ? s[2] : `${s[2]}–${s[3]}`} W</b></div>`;
  }

  rideBands(IV) {             // [from min, to min, colour] per interval, per step for over-unders
    const m = (s) => this.rideT(s) / 60;
    return (IV?.intervals || []).flatMap((e) => {
      if (!e.steps.length) return [[m(e.start), m(e.end), css(IZONE[e.zone][1])]];
      let t = e.start;
      return e.steps.map(([sec, w]) => { const b = [m(t), m(t + sec), css(IZONE[izoneOf(IV, w)][1])]; t += sec; return b; });
    });
  }

  // the ride charts leave stops out (RIDE.stops: [ride second, seconds]); intervals, plan steps, alpha1 and the
  // server's windows count ride seconds: rideT = ride second -> riding second, rideS = back
  rideT(s) {
    let t = s;
    for (const [a, n] of this.RIDE?.stops || []) { if (a >= s) break; t -= Math.min(n, s - a); }
    return t;
  }

  rideS(t) {
    let s = t;
    for (const [a, n] of this.RIDE?.stops || []) { if (a > s) break; s += n; }
    return s;
  }

  rideSource(a) {             // where the ride's data comes from (its id: goodpush_ / garmin_ / Strava's number)
    return a.id.startsWith("goodpush_") ? "Recorded by goodpush" : a.id.startsWith("garmin_") ? "From Garmin Connect" : "From Strava";
  }

  // the line under the title: source, other copies, day and time, type, workout
  rideLine(a) {
    const day = niceDate(a.date, {weekday: "long", day: "numeric", month: "long", year: "numeric"});
    const time = a.start && a.start.length > 10 ? " · " + esc(a.start.slice(11)) : "";
    const kind = `${esc(a.type || "")}${a.indoor ? " (indoor)" : ""}`;
    const workout = a.workout ? ` · <span class="chip">${esc(a.type_name || a.workout)}</span>` : "";
    const source = `<span class="chip" title="Where this ride's data comes from">${this.rideSource(a)}</span>`;
    return `${source}${this.rideAlso(a)} ${day}${time} · ${kind}${workout}`;
  }

  // the same ride from other sources (SameRide on the server): left out, so it counts once
  rideAlso(a) {
    const ids = (a.also || "").split(",").filter((x) => x);
    if (!ids.length) return "";
    const names = [...new Set(ids.map((x) => x.startsWith("goodpush_") ? "goodpush"
      : x.startsWith("garmin_") ? "Garmin" : "Strava"))].join(" and ");
    const tip = "The copies were merged: its time, TSS and hours count once";
    return ` <span class="chip" title="${tip}">also recorded by ${names}</span>`;
  }

  // goodpush's own recording that Strava doesn't have yet (Strava's ids are plain numbers)
  stravaMissing(a) {
    return a.id.startsWith("goodpush_") && !(a.also || "").split(",").some((x) => /^\d+$/.test(x));
  }

  // Strava's upload page opens in the browser, the FIT file (and the chart as a PNG) in Explorer / Finder; the title
  // goes on the clipboard first (a FIT can't carry it), while this page still has the focus the browser asks for
  async sendToStrava() {
    const copied = navigator.clipboard ? navigator.clipboard.writeText(this.stravaTitle(this.RIDE)).then(() => true, () => false)
      : Promise.resolve(false);
    const png = await this.stravaPng().catch(() => null);
    const r = await fetch(`/api/strava_upload?id=${encodeURIComponent(this.RIDE.id)}`, {method: "POST",
      headers: {"Content-Type": "image/png", "X-GoodPush": "1"}, body: png || ""});
    const j = await r.json().catch(() => ({}));
    if (!r.ok || j.error) { toast.show(j.error || "Something went wrong"); return; }
    toast.show(await copied ? "Title copied to the clipboard – drag the .fit onto Strava's page, then paste the title"
      : "Strava's upload page is open – drag the .fit onto it");
  }

  stravaTitle(a) {            // the title for Strava: the workout's name, or the ride's
    return `GoodPush - ${a.plan?.title || a.name || "Ride"}`;   // like report/plan_out.outbound_title
  }

  // the picture for Strava, drawn like the calendar's workout profile (widgets.profileSVG): the planned steps as
  // blocks in their zone colours, a dashed FTP line, and the power ridden over them in pastel; whole ride, 2x sharp
  async stravaPng() {
    const a = this.RIDE, st = a.streams;
    if (!st) return null;
    const W = 1200, H = 520, font = getComputedStyle(document.body).fontFamily;
    const power = expSmooth(st.power, boxSigma(10 / (st.bucket_s || 1)));
    const steps = a.plan?.steps || [], ftp = a.ftp || 200;
    const top = Math.max(1.2 * ftp, ...steps.map((s) => s[3]), ...power.filter((v) => v != null)) * 1.05;
    const box = {l: 56, r: W - 20, t: 76, b: H - 30, x1: this.rideEnd(), y1: top};
    const fx = (m) => box.l + m / box.x1 * (box.r - box.l), fy = (w) => box.b - w / box.y1 * (box.b - box.t);
    const cv = document.createElement("canvas");
    cv.width = W * 2; cv.height = H * 2;
    const g = cv.getContext("2d");
    g.scale(2, 2);
    g.fillStyle = css("--bg"); g.fillRect(0, 0, W, H);
    g.fillStyle = css("--ink"); g.font = `600 24px ${font}`;
    g.fillText(a.plan ? a.plan.title : a.name || "Ride", 20, 36);
    g.fillStyle = css("--ink2"); g.font = `15px ${font}`;
    const day = niceDate(a.date, {day: "numeric", month: "long", year: "numeric"});
    g.fillText(`${day} · ${hhmm((a.moving_s || 0) / 60)} · ${fmt(a.avg_power)} W average · TSS ${fmt(a.tss)}`, 20, 60);
    this.stravaAxes(g, box, fx, fy, font);
    this.stravaBlocks(g, steps, ftp, fx, fy);
    g.strokeStyle = css("--muted"); g.lineWidth = 1; g.setLineDash([6, 6]);           // FTP
    g.beginPath(); g.moveTo(box.l, fy(ftp)); g.lineTo(box.r, fy(ftp)); g.stroke(); g.setLineDash([]);
    this.stravaLine(g, st.t_min, power, fx, fy);
    return await new Promise((ok) => cv.toBlob(ok, "image/png"));
  }

  stravaAxes(g, box, fx, fy, font) {             // faint watt grid, time along the bottom (h:mm)
    g.font = `12px ${font}`; g.fillStyle = css("--ink2"); g.strokeStyle = css("--grid"); g.lineWidth = 1;
    g.textAlign = "right";
    for (const w of niceTicks(0, box.y1, 4)) {
      if (w > box.y1) continue;
      g.beginPath(); g.moveTo(box.l, fy(w)); g.lineTo(box.r, fy(w)); g.stroke();
      g.fillText(`${w} W`, box.l - 8, fy(w) + 4);
    }
    g.textAlign = "center";
    for (const m of niceTicks(0, box.x1, 8)) {
      if (m <= box.x1) g.fillText(`${Math.floor(m / 60)}:${String(Math.round(m % 60)).padStart(2, "0")}`, fx(m), box.b + 20);
    }
    g.textAlign = "left";
  }

  // the planned steps (ride clock: [from min, to min, low W, high W, kind]): a block at the target's middle, its
  // zone colour as on the calendar; a missed rep fainter
  stravaBlocks(g, steps, ftp, fx, fy) {
    for (const [s0, s1, lo, hi, k] of steps) {
      const x0 = fx(this.rideT(s0 * 60) / 60), x1 = fx(this.rideT(s1 * 60) / 60), y = fy((lo + hi) / 2);
      g.globalAlpha = k === 2 ? 0.35 : 0.9;
      g.fillStyle = css(zoneVar((lo + hi) / 2 / ftp).slice(4, -1));
      g.beginPath(); g.roundRect(x0 + 1, y, Math.max(x1 - x0 - 2, 1), fy(0) - y, [5, 5, 0, 0]); g.fill();
    }
    g.globalAlpha = 1;
  }

  stravaLine(g, t, power, fx, fy) {              // the power ridden: pastel, on a thin halo so it reads on any block
    const path = new Path2D();
    let on = false;
    power.forEach((v, i) => {
      if (v == null) { on = false; return; }
      if (on) path.lineTo(fx(t[i]), fy(v)); else path.moveTo(fx(t[i]), fy(v));
      on = true;
    });
    g.lineJoin = g.lineCap = "round";
    g.strokeStyle = css("--bg"); g.lineWidth = 5; g.stroke(path);
    g.strokeStyle = this.STRAVA_LINE; g.lineWidth = 2.5; g.stroke(path);
  }

  // a ride goodpush recorded is deleted for good; one from Strava / Garmin is left out (a sync can't bring it back)
  async deleteRide() {
    const a = this.RIDE, mine = a.id.startsWith("goodpush_");
    if (!await ask.confirm({title: `Delete ${a.name || "this ride"} (${a.date})?`, yes: "Delete the ride", danger: true,
      text: mine ? "Its file is removed for good."
        : "It stays on Strava / Garmin, but goodpush leaves it out from now on."})) return;
    if (!await api.action({action: "delete_activity", id: a.id}, "Ride deleted – re-analysing…")) return;
    this.RIDE = null;
    if (calendarPage.CAL) { calendarPage.CAL.acts = calendarPage.CAL.acts.filter((x) => x.id !== a.id); calendarPage.CAL._m = null; }   // gone now; the re-analysis's
    location.hash = "#calendar";                                                   // new plan reloads the rest
  }

  html() {
    const id = router.routeArg();
    if (!this.RIDE || this.RIDE.id !== id) {
      fetch(`/api/activity/${encodeURIComponent(id)}`).then((r) => r.json()).then((j) => {
        if (router.route() !== "ride" || router.routeArg() !== id) return;
        this.RIDE = j.error ? {id, error: j.error} : j;
        this.rideViewSt = this.rideSelStats = this.rideSel.range = null;
        TimeWindow.of("ride").view = null;
        router.render();
      }).catch(() => {});
      return `<p><a href="#rides">← Rides</a></p><div class="loading">Loading ride…</div>`;
    }
    if (this.RIDE.error) return `<p><a href="#rides">← Rides</a></p><div class="card">${esc(this.RIDE.error)}</div>`;
    const a = this.RIDE, st = a.streams;
    const zoneBars = (secs, names, colors) => {
      const tot = secs.reduce((p, v) => p + (v || 0), 0);
      if (!tot) return '<div class="muted small">No data</div>';
      return `<div class="zbars">${secs.map((v, i) => `<div class="zrow"><span>${names[i]}</span><div class="zb"><i style="width:${(v / tot * 100).toFixed(1)}%;background:var(${colors ? colors[i] : `--z${i + 1}`})"></i></div><span class="n">${mmss(v || 0)}</span><span class="n muted">${Math.round((v || 0) / tot * 100)}%</span></div>`).join("")}</div>`;
    };
    const IV = a.intervals || {intervals: [], sets: []}, ZN = Object.fromEntries(Object.entries(IZONE).map(([k, v]) => [k, v[0]]));
    const zc = (z) => `var(${IZONE[z]?.[1] || "--muted"})`;
    const zoneRange = (i) => `${IV.zone_floors[i]}${i + 1 < IV.zone_floors.length ? `–${IV.zone_floors[i + 1]}` : "+"} W`;
    const setLines = IV.sets.map((x) => `<li><span class="sw" style="background:${zc(x.zone)}"></span><b>${x.count > 1 ? `${x.count} × ` : ""}${mmss(x.seconds)}</b> @ ${x.watts_hi - x.watts_lo < 6 ? x.watts_lo : `${x.watts_lo}–${x.watts_hi}`} W
    <span class="muted">${ZN[x.zone] || x.zone}${x.rest ? ` · ${mmss(x.rest)} between at ${x.rest_watts} W (${(ZN[x.rest_zone] || "").toLowerCase()})` : ""}</span></li>`).join("");
    const eff = IV.intervals.map((e, i) => `<tr class="pick" data-a="${this.rideT(e.start) / 60}" data-b="${this.rideT(e.end) / 60}" title="Select this interval"><td class="n">${i + 1}</td><td class="n">${mmss(this.rideT(e.start))}</td><td class="n">${mmss(e.seconds)}</td><td class="n"><b>${fmt(e.watts)}</b> W</td><td class="n">${fmt(e.pct_cp)} %</td><td><span class="sw" style="background:${zc(e.zone)}"></span>${ZN[e.zone] || e.zone}</td><td class="small muted">${e.steps.map(([t, w]) => `${mmss(t)} @ ${w}`).join(" · ")}</td>${IV.w_prime ? `<td class="n">${e.w_prime_used > 0 ? fmt(e.w_prime_used) + " %" : "–"}</td>` : ""}<td class="n">${fmt(e.avg_hr)}</td><td class="n">${fmt(e.max_hr)}</td><td class="n">${e.hrr_60 != null ? `−${fmt(e.hrr_60)} <span class="small muted">at ${fmt(e.rec_watts)} W</span>` : "–"}</td><td class="n">${e.hrr_share != null ? `${fmt(e.hrr_share * 100)} %` : "–"}</td></tr>`).join("");
    const kmh = a.distance_m && a.moving_s ? a.distance_m / a.moving_s * 3.6 : null;
    const notes = [a.description, a.private_note].filter((x) => x).map((x) => `<p style="margin:6px 0 0">${esc(x)}</p>`).join("");
    return `
  <div class="row"><a href="#rides">← Rides</a><span class="spacer"></span>
    ${a.prev ? `<a class="btn sm" href="#ride/${encodeURIComponent(a.prev)}">← Older</a>` : ""}${a.next ? `<a class="btn sm" href="#ride/${encodeURIComponent(a.next)}">Newer →</a>` : ""}
    ${this.stravaMissing(a) ? `<button class="btn sm" id="rstrava" title="Opens Strava's upload page and a folder with this ride's file (drag it onto the page) and a picture of the workout (add it to the activity as a photo); its title goes on the clipboard">Send to Strava</button>` : ""}
    <button class="btn sm danger" id="rdel">Delete</button></div>
  <h1 style="margin-top:12px">${esc(a.name || "Ride")}</h1>
  <p class="sub">${this.rideLine(a)}</p>
  ${notes ? `<div class="card small">${notes}</div>` : ""}
  <div class="tiles" style="margin-top:12px">
    ${tile("Time", mmss(a.duration_s || 0), "", a.moving_s && a.moving_s < a.duration_s - 60 ? `moving ${mmss(a.moving_s)}` : "")}
    ${a.distance_m ? tile("Distance", fmt(a.distance_m / 1000, 1), "km", kmh ? `${fmt(kmh, 1)} km/h` : "") : ""}
    ${a.elev_gain ? tile("Climbing", fmt(a.elev_gain), "m", "") : ""}
    ${tile("Normalized power", fmt(a.np_used), "W", `avg ${fmt(a.avg_power)} W · max ${fmt(a.max_power)} W`)}
    ${tile("Intensity", fmt(a.if_, 2), "IF", `FTP ${fmt(a.ftp)} W`)}
    ${tile("Training load", fmt(a.tss), "TSS", a.tss_source && a.tss_source !== "power" ? `from ${esc(a.tss_source)}` : `${fmt(a.work_kj)} kJ`)}
    ${tile("Heart rate", fmt(a.avg_hr), "bpm", `max ${fmt(a.max_hr)}`)}
    ${a.ef ? tile("Efficiency", fmt(a.ef, 2), "NP/HR", a.decoupling != null ? `HR drift ${fmt(a.decoupling, 1)} %` : "") : ""}
    ${a.avg_cadence ? tile("Cadence", fmt(a.avg_cadence), "rpm", "") : ""}
  </div>
  ${st ? `
  <h2>Power and heart rate</h2>
  <div class="card">
    <div class="row"><div class="legend"><span><i style="background:var(--s1)"></i>Power</span><span><i style="background:var(--muted)"></i>FTP</span>${a.plan ? `<span title="The planned targets in their zone colours, lined up with the reps the ride was scored on"><i style="background:${this.ridePlanColor(a)};opacity:.4"></i>Planned: <a href="#calendar">${esc(a.plan.title)}</a></span>${a.plan.steps.some((s) => s[4] === 2) ? `<span><i style="background:${this.ridePlanColor(a)};opacity:.15"></i>Missed rep (fainter)</span>` : ""}` : ""}</div><span class="spacer"></span>
      <label class="chk smooth" title="Average the power over a moving window">Smoothing <input type="range" id="smooth" min="0" max="120" step="5" value="${this.rideSmooth}"> <span id="smoothv">${this.smoothLabel(this.rideSmooth)}</span></label>
      <div class="seg" id="rzoom"><button data-z="in" title="Zoom in" aria-label="Zoom in">+</button><button data-z="out" title="Zoom out" aria-label="Zoom out">−</button><button data-z="left" title="Earlier" aria-label="Earlier">◀</button><button data-z="right" title="Later" aria-label="Later">▶</button><button data-z="fit">Whole ride</button></div>
      ${(a.intervals?.intervals || []).length ? `<label class="chk"><input type="checkbox" id="rivshow" ${this.rideIv.value ? "checked" : ""}> Highlight intervals</label>` : ""}</div>
    <div class="small muted" style="margin:4px 0 2px"><span id="rview">Whole ride</span> · drag across a chart to select a stretch, drag its edges to adjust · Ctrl + wheel or pinch to zoom, Shift + drag to move along</div>
    <div class="selgrid ridegrid"><div>
    <div class="withhist"><div id="rpow" class="chart sm"></div><div id="rpowH" class="sidehist"></div></div>
    <div class="legend stack"><span><i style="background:var(--s2)"></i>Heart rate</span>${a.lthr ? '<span><i style="background:var(--muted)"></i>Threshold HR</span>' : ""}<span class="spacer"></span><span class="small muted">right: time at each value, in the stretch shown</span></div>
    <div class="withhist"><div id="rhr2" class="chart sm"></div><div id="rhr2H" class="sidehist"></div></div>
    ${st.stamina ? `<div class="legend stack">
      <span><i style="background:var(--s3)"></i>Stamina${hint(STAMINA_TIP)}</span>
      <span><i class="dash" style="border-color:var(--muted)"></i>Potential</span></div>
    <div class="withhist"><div id="rsta" class="chart xs"></div><div id="rstaH" class="sidehist"></div></div>` : ""}
    ${(st.altitude || []).some((v) => v != null) && !a.indoor ? `<div class="legend stack"><span><i style="background:var(--s3)"></i>Elevation</span></div><div id="ralt" class="chart xs"></div>` : ""}
    ${(st.cadence || []).some((v) => v) ? `<div class="legend stack"><span><i style="background:var(--s7)"></i>Cadence</span></div><div class="withhist"><div id="rcad" class="chart xs"></div><div id="rcadH" class="sidehist"></div></div>` : ""}
    ${a.alpha ? `<div class="legend stack"><span><i style="background:var(--s4)"></i>Alpha1</span><span><i class="dash" style="border-color:var(--muted)"></i>0.75 aerobic threshold · 0.5 second threshold</span></div><div class="withhist"><div id="ra1" class="chart xs"></div><div id="ra1H" class="sidehist"></div></div>` : ""}
    </div><aside class="selpanel">${this.hasMap(a) ? `<div id="rmap" class="map side"></div>` : ""}<div id="rsel"></div></aside></div>
  </div>` : `<div class="card muted" style="margin-top:12px">No second-by-second data for this activity yet. If it was just imported, press <b>Rebuild everything</b> on the Today page (or run <code>goodpush run</code>) so the ride file gets read.</div>`}
  ${a.ramp ? this.rampCard(a) : ""}
  ${a.lsct ? this.lsctCard(a) : ""}
  ${a.alpha ? this.alphaCard(a) : ""}
  <div class="grid g2" style="margin-top:12px">
    ${this.hasMap(a) && !st ? `<div class="card"><h3>Route</h3><div id="rmap" class="map"></div></div>` : ""}
    <div class="card"><h3>Power curve</h3><div class="legend"><span><i style="background:var(--s1)"></i>This ride</span><span><i style="background:var(--muted)"></i>Your best, last 90 days</span></div><div id="rpdc" class="chart sm"></div></div>
  </div>
  <div class="grid g2" style="margin-top:12px">
    ${IV.zone_seconds ? `<div class="card"><h3>Time in power zones</h3><p class="small muted" style="margin-top:0">From your thresholds that day (the interval zones).</p>
      ${zoneBars(IV.zone_seconds, IV.zones.map((z, i) => `${ZN[z]} <span class="muted">${zoneRange(i)}</span>`), IV.zones.map((z) => IZONE[z][1]))}</div>`
        : `<div class="card"><h3>Time in power zones</h3>${zoneBars(a.zones_power || [], ZONE_NAMES)}</div>`}
    <div class="card"><h3>Time in heart-rate zones</h3>${zoneBars(a.zones_hr || [], HR_ZONE_NAMES)}</div>
  </div>
  ${eff ? `<h2>Intervals</h2><div class="card"><ul class="ivsets">${setLines}</ul>
    <p class="small muted">Found where the power steps up or down${IV.bursty ? " (on 30-s power: this ride's power is bursty, so short surges don't count)" : ""}; tempo and harder.
      Zones from your thresholds that day${IV.anchors ? ` (LT1 ${IV.anchors.lt1} W, CP ${IV.anchors.cp} W, MAP ${IV.anchors.map} W${IV.w_prime ? `, W′ ${fmt(IV.w_prime / 1000, 1)} kJ` : ""})` : ""}, as in your plan: threshold at CP, VO2max between CP and MAP, anaerobic above MAP.
      ${IV.w_prime ? "W′ used: the most of your W′ spent by the end of the interval (W′ balance, as in the stamina chart). " : ""}Click one to select it in the charts.</p>
    <div class="tw"><table><thead><tr><th class="n">#</th><th class="n">At</th><th class="n">Duration</th><th class="n">Avg power</th><th class="n">of CP</th><th>Zone</th><th>Steps</th>${IV.w_prime ? '<th class="n">W′ used</th>' : ""}<th class="n">Avg HR</th><th class="n">Max HR</th><th class="n" title="How far your heart rate fell in the minute after the interval, and your average power in that minute (– when the next interval came sooner)">HR drop 1 min</th><th class="n" title="How much of the way down to your steady heart rate at that recovery power your heart rate fell in the minute: comparable across recovery powers. Steady heart rate: a straight line from your resting HR to your LTHR at CP">Recovered</th></tr></thead><tbody>${eff}</tbody></table></div></div>` : ""}`;
  }

  // a ramp test ridden from the plan (server ramp_test_detail): the top, max HR, recovery, HR at your thresholds,
  // each step, and your ramp tests side by side
  rampCard(a) {
    const R = a.ramp, rec = R.recovery || {}, at = R.hr_at || {};
    const higher = R.max_hr != null && R.max_hr > (R.max_hr_set || R.max_hr_now || 0);
    const maxSub = `${R.max_hr_set ? `your setting: ${fmt(R.max_hr_set)}` : `in use: ${fmt(R.max_hr_now)} (from your rides)`}` +
      (higher ? ` <button class="btn sm" data-maxhr="${R.max_hr}">Use ${fmt(R.max_hr)} as my max HR</button>` : "");
    const atTile = (k, tip) => tile(`HR at ${k}`, at[k]?.hr != null ? fmt(at[k].hr) : "–", at[k]?.hr != null ? "bpm" : "",
      at[k]?.w ? `at ${fmt(at[k].w)} W (your ${k} now)` : `no ${k} yet`, "", tip);
    const steps = (R.steps || []).map((s) => `<tr${s.held ? "" : ' class="muted"'}><td>${esc(s.name)}${s.held ? "" : " (stopped)"}</td>
    <td class="n">${fmt(s.target_w)} W</td><td class="n">${s.power != null ? `${fmt(s.power)} W` : "–"}</td><td class="n">${mmss(s.seconds)}</td>
    <td class="n">${fmt(s.hr_end)}</td><td class="n">${fmt(s.hr_max)}</td><td class="n">${fmt(s.cadence)}</td></tr>`).join("");
    const hist = (R.history || []).slice().reverse().map((h) => `<tr class="${h.this ? "" : "click"}" ${h.this ? "" : `data-href="#ride/${encodeURIComponent(h.id)}"`}>
    <td>${esc(niceDate(h.date, {day: "numeric", month: "short", year: "numeric"}))}${h.this ? " (this one)" : ""}</td><td class="n">${fmt(h.top_w)} W</td>
    <td class="n">${fmt(h.hr_ftp)}</td><td class="n">${h.recovery_60 != null ? `−${fmt(h.recovery_60)}` : "–"}</td><td class="n">${fmt(h.max_hr)}</td>
    <td class="n">${h.lt1_w != null ? `${h.by_hr ? "≈ " : ""}${fmt(h.lt1_w)} W` : "–"}</td><td class="n">${h.lt2_w != null ? `${h.by_hr ? "≈ " : ""}${fmt(h.lt2_w)} W` : "–"}</td></tr>`).join("");
    const T = R.thresholds || {}, alpha = T.lt1?.w != null || T.lt2?.w != null;
    const ltTile = (k, lvl) => {                        // alpha1's crossing leads; else the power at your LT heart rate
      const t = T[k] || {}, K = k.toUpperCase(), all = S.thresholds?.[k]?.w;
      const now = all != null ? ` · all your evidence: ${fmt(all)} W` : "";
      if (t.w != null) return tile(`${K} from this test`, fmt(t.w), "W", `at ${fmt(t.hr)} bpm · alpha1 ${lvl} · ramps read ${K} high${now}`, "",
        `Where DFA alpha1 dropped through ${lvl} on the ramp. A ramp reads ${K} high (alpha1 lags the rising power), so this is one reading: the physio model weighs it with your others and its figure is what the plan uses (Rider page).`);
      if (t.hr_w != null) return tile(`${K} from this test`, `≈ ${fmt(t.hr_w)}`, "W", `where you reached your ${K} heart rate (${fmt(t.at_hr)} bpm)${now}`, "",
        `Without alpha1: the ramp's power when your heart rate reached your ${K} heart rate (from alpha1 on your earlier rides). Heart rate lags a ramp, so this reads high; the physio model learns by how much.`);
      return tile(`${K} from this test`, "–", "", "no alpha1, and no heart-rate anchor to read it from");
    };
    return `<h2>Ramp test${hint("A ramp to failure: the target rose every 2 minutes until you couldn't hold it.\n• Step heart rate: its last 30 s (heart rate lags power)\n• Held: within 10 s of the step's end\n• Top of the ramp: Kuipers et al. 1985 · HR recovery: Cole et al. 1999")}</h2>
  <div class="card">
    <div class="tiles">
      ${tile("Top of the ramp", fmt(R.top_w), "W", `${R.top_ftp != null ? `${fmt(R.top_ftp, 2)} × FTP · ` : ""}${R.last_held ? `held ${esc(R.last_held)} to the end` : "no step held to the end"} · best minute ${fmt(R.best_1min)} W`, "",
          "The last step you held plus the share of the next one you rode. A slower ramp (this one: about 3 % of FTP a minute) tops out lower than a 1-minute ramp, so compare it with your other ramp tests, not with apps' ramp tests.")}
      ${tile("Max heart rate", fmt(R.max_hr), "bpm", maxSub)}
      ${tile("Heart-rate recovery", rec["60"] != null ? `−${fmt(rec["60"])}` : "–", "bpm", `in the first minute after you stopped${rec["120"] != null ? ` · −${fmt(rec["120"])} after 2 min` : ""}`, "",
          "How fast your heart rate drops once you stop. A bigger drop goes with better fitness; a smaller one than usual can mean fatigue. Compare it between your own tests.")}
      ${ltTile("lt1", "0.75")}${ltTile("lt2", "0.5")}
      ${atTile("LT1", "Heart rate at your first threshold's power, from this test's steps. The same watts at a lower heart rate next time = fitter.")}
      ${atTile("FTP", "Heart rate at your FTP's power, from this test's steps.")}
      ${atTile("LT2", "Heart rate at your second threshold's power, from this test's steps.")}
    </div>
    ${alpha ? "" : `<p class="small muted" style="margin:6px 0 0">No alpha1 on this ride, so LT1 / LT2 are read from your heart rate instead (they read high). For the validated ones, record the ride with the alphaHRV field on your Garmin too: goodpush takes its alpha1 for this ride.</p>`}
    <h3 style="margin-top:14px">Steps</h3>
    <div class="tw"><table><thead><tr><th>Step</th><th class="n">Target</th><th class="n">Your power</th><th class="n">Time</th><th class="n">HR at end</th><th class="n">Max HR</th><th class="n">Cadence</th></tr></thead><tbody>${steps}</tbody></table></div>
    <h3 style="margin-top:14px">Your ramp tests${hint(`Heart rate at ${R.ftp_w ? `${R.ftp_w} W (your FTP now)` : "your FTP"} in each test.\n• Same watts at a lower heart rate = fitter\n• LT1 / LT2 from alpha1 where recorded, else (≈) from heart rate`)}</h3>
    <div class="tw"><table><thead><tr><th>Date</th><th class="n">Top</th><th class="n">HR at ${R.ftp_w ? `${fmt(R.ftp_w)} W` : "FTP"}</th><th class="n">Recovery 1 min</th><th class="n">Max HR</th><th class="n">LT1</th><th class="n">LT2</th></tr></thead><tbody>${hist}</tbody></table></div>
    ${(R.history || []).length < 2 ? `<p class="small muted" style="margin:6px 0 0">Your next ramp test shows up here next to this one.</p>` : ""}
    <p class="small muted" style="margin:6px 0 0">How these are worked out, and the papers: <a href="#sources">Sources</a>.</p>
  </div>`;
  }

  // an LSCT ridden from the plan (server lsct_detail): power and heart rate per stage, heart-rate recovery, and your
  // LSCTs side by side
  lsctCard(a) {
    const L = a.lsct, n = L.stages.length;
    const stages = L.stages.map((s) => `<tr><td>${esc(s.name)}</td><td class="n">${s.power != null ? `${fmt(s.power)} W` : "–"}</td>
    <td class="n">${fmt(s.hr)}</td><td class="n">${mmss(s.seconds)}</td></tr>`).join("");
    const hist = (L.history || []).slice().reverse().map((h) => `<tr class="${h.this ? "" : "click"}" ${h.this ? "" : `data-href="#ride/${encodeURIComponent(h.id)}"`}>
    <td>${esc(niceDate(h.date, {day: "numeric", month: "short", year: "numeric"}))}${h.this ? " (this one)" : ""}</td>
    ${[1, 2, 3].map((k) => `<td class="n">${h[`lsct_w_${k}`] != null ? `${fmt(h[`lsct_w_${k}`])} W` : "–"}</td>`).join("")}
    <td class="n">${h.lsct_hrr_60 != null ? `−${fmt(h.lsct_hrr_60)}` : "–"}</td></tr>`).join("");
    const last = n ? L.stages[n - 1] : {};
    return `<h2>LSCT${hint("Submaximal test: heart rate held at 60, 80 and 90 % of max, then 90 s sitting still.\n• Stage power and heart rate: its second half\n• More power at the same heart rates = fitter\n• Recovery: the fall in the first minute sitting still\n• Lamberts et al. 2011")}</h2>
  <div class="card">
    <div class="tiles">
      ${tile("Last stage", last.power != null ? fmt(last.power) : "–", "W", last.hr != null ? `at ${fmt(last.hr)} bpm` : "no heart rate")}
      ${tile("Heart-rate recovery", L.recovery_60 != null ? `−${fmt(L.recovery_60)}` : "–", "bpm", "in the first minute sitting still", "",
          "How far heart rate fell in the first minute after the last stage. A bigger fall goes with better fitness, but heavy fatigue can speed it up too: read it with the stage powers.")}
    </div>
    <h3 style="margin-top:14px">Stages</h3>
    <div class="tw"><table><thead><tr><th>Stage</th><th class="n">Power</th><th class="n">Heart rate</th><th class="n">Time</th></tr></thead><tbody>${stages}</tbody></table></div>
    <h3 style="margin-top:14px">Your LSCTs</h3>
    <div class="tw"><table><thead><tr><th>Date</th><th class="n">60 %</th><th class="n">80 %</th><th class="n">90 %</th><th class="n">Recovery 1 min</th></tr></thead><tbody>${hist}</tbody></table></div>
    ${(L.history || []).length < 2 ? `<p class="small muted" style="margin:6px 0 0">Your next LSCT (each recovery week) shows up here next to this one.</p>` : ""}
  </div>`;
  }

  alphaCard(a) {
    const c = a.alpha_check || {};
    const part = (name, label) => {
      const v = c[`a1_${name}`]; if (v == null) return `<div class="tile"><div class="l">${label}</div><div class="v">–</div><div class="s">no steady easy riding in that window</div></div>`;
      const [cls, txt] = markers.a1Verdict(c[`a1_${name}_z`]);
      const hz = c[`a1_${name}_hr_z`];
      return `<div class="tile"><div class="l">${label}</div><div class="v">${fmt(v, 2)}<small>α1</small></div>
      <div class="s">at ${fmt(c[`a1_${name}_p`])} W, HR ${fmt(c[`a1_${name}_hr`])}${c[`a1_${name}_exp`] != null ? ` · your normal ≈ ${fmt(c[`a1_${name}_exp`], 2)}` : ""}</div>
      <div style="margin-top:4px">${markers.chipFor(cls, "alpha1 " + txt)}${hz != null && Math.abs(hz) >= 2 ? " " + markers.chipFor(hz > 0 ? "missed" : "amber", hz > 0 ? "HR high for the power" : "HR low for the power") : ""}</div></div>`;
    };
    return `<h2>DFA alpha1 (${a.alpha_source === "own" ? "goodpush, from your strap's beat intervals" : "alphaHRV"})${hint(a.alpha_source === "own"
      ? "Worked out by goodpush from your chest strap's beat-to-beat intervals:\n• Odd beats found against the local spread; missed or extra beats repaired (never invented)\n• Slow heart-rate drift taken out\n• DFA over 4-16-beat boxes in the last 2 minutes, every 5 s\nIndoor rides use this; outdoor rides your Garmin's alphaHRV."
      : "From the alphaHRV field on your Garmin (outdoor rides). Indoors goodpush works it out from your strap's beats.")}</h2>
  <div class="card">
    <div class="tiles">${part("warm", "Warm-up (min 5–20)")}${part("session", "Session getting going (min 20–40)")}
      ${c.hrvt1_w != null && c.ramp_top_w == null ? tile("Aerobic threshold this ride", fmt(c.hrvt1_w), "W", "where alpha1 crossed 0.75") : ""}
      ${c.ramp_top_w != null && !a.ramp ? tile("Ramp test: LT1", c.ramp_lt1_w != null ? fmt(c.ramp_lt1_w) : "–", "W", c.ramp_lt1_hr != null ? `at ${fmt(c.ramp_lt1_hr)} bpm · alpha1 0.75` : "alpha1 didn't cross 0.75") +
          tile("Ramp test: LT2", c.ramp_lt2_w != null ? fmt(c.ramp_lt2_w) : "–", "W", (c.ramp_lt2_hr != null ? `at ${fmt(c.ramp_lt2_hr)} bpm · alpha1 0.5` : "alpha1 didn't cross 0.5") + (c.ramp_lt2_loose ? ` · only a loose cap: ${c.ramp_lt2_loose}` : "")) +
          tile("Ramp top", fmt(c.ramp_top_w), "W", `${fmt(c.ramp_min)} min ramp · max HR ${fmt(c.ramp_max_hr)}`) : ""}</div>
    <p class="small muted" style="margin:6px 0 0">How alpha1 is worked out, and the papers: <a href="#sources">Sources</a>. The alpha1 chart is with the power chart above. Only steady stretches (2 minutes at a similar power) with clean data count for the checks; your normal comes from your last 6 weeks of rides at the same power. Alpha1 well below or above normal in the warm-up tends to go with a flat or tired day.</p>
  </div>`;
  }

  rideEnd() {   // minutes
    const s = this.RIDE.streams;
    return s.t_min.length ? s.t_min[s.t_min.length - 1] + s.bucket_s / 120 : 0;
  }

  rideOnView(v) {   // after a zoom: redraw, then fetch finer data for the window
    this.drawRide();
    clearTimeout(this.rideZoomT);
    const id = this.RIDE.id;
    if (v) this.rideZoomT = setTimeout(() => fetch(`/api/activity/${encodeURIComponent(id)}/window?from=${Math.floor(this.rideS(v[0] * 60))}&to=${Math.ceil(this.rideS(v[1] * 60))}`)
      .then((r) => r.json()).then((j) => {
        if (this.RIDE?.id === id && TimeWindow.of("ride").view === v && !j.error) { this.rideViewSt = {...j.streams, range: v}; this.drawRide(); }
      }).catch(() => {}), 200);
  }

  rideSelect(r) {
    this.rideSelStats = r ? {loading: true} : null;
    this.drawSelection();
    if (!r) return;
    const id = this.RIDE.id;
    fetch(`/api/activity/${encodeURIComponent(id)}/window?from=${Math.round(this.rideS(r[0] * 60))}&to=${Math.round(this.rideS(r[1] * 60))}`)
      .then((x) => x.json()).then((j) => {
        if (this.RIDE?.id !== id || this.rideSel.range !== r) return;
        this.rideSelStats = j.error ? {error: j.error} : j.stats; this.drawSelection();
      }).catch(() => {});
  }

  drawSelection() {   // the numbers for the selected stretch, beside the charts
    const r = this.rideSel.range, s = this.rideSelStats;
    this.selPanel ??= new SelectionPanel("rsel", "ride", this.rideSel, "Drag across a chart to see the numbers for that stretch.");
    if (!r) { this.selPanel.draw(null); return; }
    const body = !s || s.loading ? '<div class="muted small">Working it out…</div>' : s.error ? `<div class="muted small">${esc(s.error)}</div>` : `<div class="tiles">
    ${tile("Avg power", fmt(s.avg_power), "W", `${fmt(s.pct_ftp)} % of FTP · max ${fmt(s.max_power)} W`)}
    ${tile("Normalized power", fmt(s.np), "W", `IF ${fmt(s.if, 2)}`)}
    ${tile("Training load", fmt(s.tss), "TSS", `${fmt(s.work_kj)} kJ`)}
    ${s.avg_hr ? tile("Heart rate", fmt(s.avg_hr), "bpm", `max ${fmt(s.max_hr)}${s.decoupling != null ? ` · HR drift ${fmt(s.decoupling, 1)} %` : ""}`) : ""}
    ${s.avg_cadence ? tile("Cadence", fmt(s.avg_cadence), "rpm", "") : ""}
    ${s.distance_m ? tile("Distance", fmt(s.distance_m / 1000, 1), "km", s.speed_kmh ? `${fmt(s.speed_kmh, 1)} km/h` : "") : ""}
    ${s.climb_m ? tile("Climbing", fmt(s.climb_m), "m", "") : ""}
    ${s.alpha1 != null ? tile("Alpha1", fmt(s.alpha1, 2), "avg", `${fmt(s.alpha1_min, 2)} – ${fmt(s.alpha1_max, 2)}`) : ""}</div>`;
    this.selPanel.draw(`<b>${mmss(r[0] * 60)} – ${mmss(r[1] * 60)}</b> <span class="muted">(${mmss((r[1] - r[0]) * 60)})</span>`, body);
  }

  // time at each value beside a chart (#<id>H), on that chart's y axis: bars of `bin` wide, length = time (each
  // value `secs` seconds); hovering a bar's row shows its range, time and share of the stretch
  sideHist(id, vals, bin, secs, colour, f) {
    const box = document.getElementById(id + "H"), Y = EChart.live[id]?.spec.yScale;
    if (!box) return;
    box.innerHTML = "";
    if (!Y || !vals.length) return;
    const W = box.clientWidth || 80, k0 = Math.floor(Y.y0 / bin), n = Math.max(Math.ceil(Y.y1 / bin) - k0, 1);
    const t = new Array(n).fill(0);
    for (const v of vals) { const k = Math.floor(v / bin) - k0; if (k >= 0 && k < n) t[k] += secs; }
    const top = Math.max(...t) || 1, all = t.reduce((p, q) => p + q, 0) || 1;
    const fy = (v) => Y.top + (1 - (v - Y.y0) / (Y.y1 - Y.y0)) * (Y.bottom - Y.top);
    const svg = el("svg", {width: W, height: Y.H, viewBox: `0 0 ${W} ${Y.H}`}, box);
    const bars = t.map((s, k) => {
      if (!s) return null;
      const lo = (k + k0) * bin, y = fy(lo + bin), h = Math.max(fy(lo) - y - 0.5, 0.5);
      return el("rect", {x: 0, y, width: Math.max(s / top * (W - 4), 1), height: h, fill: colour, "fill-opacity": 0.75}, svg);
    });
    const tip = document.createElement("div");
    tip.className = "tip histtip"; tip.hidden = true; box.appendChild(tip);
    let on = -1;
    const show = (k) => {                            // k: the bin under the pointer, -1 none
      if (k === on) return;
      if (bars[on]) bars[on].setAttribute("fill-opacity", 0.75);
      on = bars[k] ? k : -1;
      if (on < 0) { tip.hidden = true; return; }
      bars[on].setAttribute("fill-opacity", 1);
      const lo = (on + k0) * bin;
      tip.innerHTML = `<div class="th">${f(lo)} – ${f(lo + bin)}</div><div><i style="background:${colour}"></i>${mmss(t[on])}<b>${Math.round(t[on] / all * 100)} %</b></div>`;
      tip.hidden = false;
      tip.style.top = Math.min(Math.max(fy(lo + bin / 2) - tip.offsetHeight / 2, 0), Y.H - tip.offsetHeight) + "px";
    };
    svg.addEventListener("pointermove", (e) => {
      const y = e.clientY - svg.getBoundingClientRect().top;
      const v = Y.y0 + (1 - (y - Y.top) / (Y.bottom - Y.top)) * (Y.y1 - Y.y0);
      show(y < Y.top || y > Y.bottom ? -1 : Math.floor(v / bin) - k0);
    });
    svg.addEventListener("pointerleave", () => show(-1));
  }

  drawRide() {
    const a = this.RIDE; if (!a || a.error) return;
    const z = TimeWindow.of("ride").view, fine = this.rideViewSt?.range;          // the finer data, while it covers the window
    const st = z && fine && fine[0] <= z[0] && fine[1] >= z[1] ? this.rideViewSt : a.streams;
    const tl = (t) => `${mmss(t * 60)}`;
    const onHover = (t) => rideMap.moveMapMarker(t);
    const viewLab = document.getElementById("rview");
    const stopS = (a.stops || []).reduce((q, s) => q + s[1], 0);
    if (viewLab) viewLab.textContent = (z ? `Showing ${mmss(z[0] * 60)} – ${mmss(z[1] * 60)}` : "Whole ride")
      + (stopS >= 60 ? ` (riding time: ${mmss(stopS)} of stops left out)` : "");
    const tick = (v, step) => step < 1 ? mmss(v * 60) : `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, "0")}`;
    const xStep = (span, n) => [1 / 12, 0.25, 0.5, 1, 2, 5, 15, 30, 60, 120].find((q) => (span >= 60 ? q >= 15 : true) && span / q <= n) || 240;
    const view = {xDomain: [0, this.rideEnd()], select: this.rideSel, onView: (v) => this.rideOnView(v), group: "ride", xStep,
      onPick: (t) => this.pickInterval(a.intervals, t)};
    const last = ["ra1", "rcad", "ralt", "rsta", "rhr2", "rpow"].find((k) => document.getElementById(k));
    const axis = (id) => ({xAxis: id === last});              // one shared time axis, at the bottom
    if (st) {
      const sig = (s) => boxSigma(s / (st.bucket_s || 1));             // a smoothing window (s) as the filter's sigma in samples
      const flat = (v) => st.t_min.map(() => v);
      const common = {x: st.t_min, xType: "num", xTickFmt: tick, xLabel: tl, onHover, ...view, bands: this.rideIv.value ? this.rideBands(a.intervals) : [],
        tipNote: (t) => (this.rideIv.value ? this.rideTipNote(a.intervals, t) : "") + this.ridePlanNote(a.plan, t)};
      EChart.draw("rpow", {...common, ...axis("rpow"), yFmt: (v) => v + " W", zero: true, series: [
        ...this.ridePlanBands(a.plan, a.intervals),
        {kind: "line", y: expSmooth(st.power, sig(this.rideSmooth)), color: css("--s1"), name: "Power", fmt: (v) => fmt(v) + " W", width: 1.6, area: true},
        {kind: "line", y: flat(a.ftp), color: css("--muted"), dash: true, name: "FTP", fmt: (v) => fmt(v) + " W", width: 1}]});
      EChart.draw("rhr2", {...common, ...axis("rhr2"), yFmt: (v) => v, series: [
        {kind: "line", y: expSmooth(st.hr, sig(10)), color: css("--s2"), name: "Heart rate", fmt: (v) => fmt(v) + " bpm", width: 1.6},
        ...(a.lthr ? [{kind: "line", y: flat(a.lthr), color: css("--muted"), dash: true, name: "Threshold HR", fmt: (v) => fmt(v) + " bpm", width: 1}] : [])]});
      const shown = (z || [0, Infinity]), secs = st.bucket_s || 1;
      const inView = (ys) => ys.filter((v, i) => v != null && st.t_min[i] >= shown[0] && st.t_min[i] <= shown[1]);
      this.sideHist("rpow", inView(st.power).filter((v) => v > 0), 10, secs, css("--s1"), (v) => fmt(v) + " W");
      this.sideHist("rhr2", inView(st.hr), 2, secs, css("--s2"), (v) => fmt(v) + " bpm");
      const pct = (v) => fmt(v) + " %";
      if (document.getElementById("rsta") && st.stamina) EChart.draw("rsta", {...common, ...axis("rsta"), yFmt: pct,
        zero: true, yExtra: [flat(100)], series: [
          {kind: "line", y: st.potential, color: css("--muted"), dash: true, name: "Potential", fmt: pct, width: 1.2},
          {kind: "line", y: st.stamina, color: css("--s3"), name: "Stamina", fmt: pct, width: 1.6, area: true}]});
      if (st.stamina) this.sideHist("rsta", inView(st.stamina), 5, secs, css("--s3"), pct);
      if (document.getElementById("ralt")) EChart.draw("ralt", {...common, ...axis("ralt"), yFmt: (v) => v + " m", series: [
        {kind: "line", y: st.altitude, color: css("--s3"), name: "Elevation", fmt: (v) => fmt(v) + " m", width: 1.6, area: true}]});
      if (document.getElementById("rcad")) EChart.draw("rcad", {...common, ...axis("rcad"), yFmt: (v) => v, series: [
        {kind: "line", y: expSmooth(st.cadence, sig(10)), color: css("--s7"), name: "Cadence", fmt: (v) => fmt(v) + " rpm", width: 1.4}]});
      this.sideHist("rcad", inView(st.cadence || []).filter((v) => v > 0), 5, secs, css("--s7"), (v) => fmt(v) + " rpm");
    }
    if (a.alpha && document.getElementById("ra1")) {
      const al = a.alpha, flat = (v) => al.t_min.map(() => v), x = al.t_min.map((m) => this.rideT(m * 60) / 60);
      EChart.draw("ra1", {x, xType: "num", xTickFmt: tick, ...view, ...axis("ra1"),
        xLabel: (t) => mmss(t * 60), yFmt: (v) => v, shade: [5, 20], bands: this.rideIv.value ? this.rideBands(a.intervals) : [], series: [
          {kind: "line", y: flat(0.75), color: css("--muted"), dash: true, width: 1},
          {kind: "line", y: flat(0.5), color: css("--muted"), dash: true, width: 1},
          {kind: "line", y: expSmooth(al.a1, boxSigma(3)), color: css("--s4"), name: "Alpha1", fmt: (v) => fmt(v, 2), width: 1.8}]});
      const [z0, z1] = z || [0, Infinity];
      this.sideHist("ra1", al.a1.filter((v, i) => v != null && x[i] >= z0 && x[i] <= z1), 0.05, al.bucket_s || 5, css("--s4"), (v) => fmt(v, 2));
    }
    const pts = (o) => Object.entries(o || {}).map(([k, v]) => [Number(k), v]).filter(([k, v]) => v && k >= 5).sort((p, q) => p[0] - q[0]);
    const r = pts(a.mmp), b = pts(a.mmp_90d);
    const lab = (x) => x < 60 ? `${x} s` : x < 3600 ? `${Math.round(x / 60)} min` : hhmm(x / 60);
    if (r.length) EChart.draw("rpdc", {xType: "log", yFmt: (v) => v + " W", xLabel: lab,
      xTicks: [[5, "5s"], [15, "15s"], [60, "1m"], [300, "5m"], [1200, "20m"], [3600, "1h"], [7200, "2h"]], series: [
        {kind: "line", x: b.map((q) => q[0]), y: b.map((q) => q[1]), color: css("--muted"), dash: true, name: "90-day best", fmt: (v) => fmt(v) + " W"},
        {kind: "line", x: r.map((q) => q[0]), y: r.map((q) => q[1]), color: css("--s1"), name: "This ride", fmt: (v) => fmt(v) + " W"}]});
    rideMap.drawMap();
  }

  wire() {
    if (!(this.RIDE && this.RIDE.id === router.routeArg())) return;
    requestAnimationFrame(() => { this.drawRide(); this.drawSelection(); });
    $("#rdel").addEventListener("click", () => this.deleteRide());
    $("#rstrava")?.addEventListener("click", () => this.sendToStrava());
    $("[data-maxhr]")?.addEventListener("click", (e) => api.action({action: "settings", section: "athlete",
      values: {max_hr: +e.currentTarget.dataset.maxhr}}, "Max HR saved – re-analysing…"));
    $("#rivshow")?.addEventListener("change", (e) => {
      this.rideIv.set(e.target.checked);
      this.drawRide();
    });
    document.querySelectorAll("#rzoom button").forEach((b) => b.addEventListener("click", () => {
      const tw = TimeWindow.of("ride"), z = b.dataset.z, [a0, b0] = tw.shown(), r = this.rideSel.range;
      if (z === "fit") tw.zoomTo(null);
      else if (z === "in") tw.zoomAt(r ? (r[0] + r[1]) / 2 : (a0 + b0) / 2, 0.5);
      else if (z === "out") tw.zoomAt((a0 + b0) / 2, 2);
      else tw.pan(z === "left" ? -0.5 : 0.5);
    }));
    document.querySelectorAll("tr.pick").forEach((tr) => tr.addEventListener("click", () => {
      this.selectStretch([Number(tr.dataset.a), Number(tr.dataset.b)]);
      document.getElementById("rpow")?.closest(".card")?.scrollIntoView({block: "start", behavior: "smooth"});   // up to the charts
    }));
    $("#smooth")?.addEventListener("input", (e) => {
      this.rideSmooth = Number(e.target.value);
      $("#smoothv").textContent = this.smoothLabel(this.rideSmooth);
      this.drawRide();
    });
  }
}
export const ridePage = new RidePage();
