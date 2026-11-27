// Model: can you trust the physio model - a verdict, today's numbers with their ranges, how its predictions sit next to
// what you rode and measured, the sampler's details, folded (ModelPage).
import {ask, Lazy, Pref, Segmented} from "../controls.js";
import {Page} from "./page.js";
import {api, S} from "../api.js";
import {$, css, esc, fmt} from "../util.js";
import {hint, niceDate, tile} from "../widgets.js";
import {router} from "../main.js";
import {DOMAIN, DOMAIN_ORDER, riderPage} from "./rider.js";
import {EChart, TimeWindow} from "../charts.js";
import {GUIDE} from "./help.js";

const MODEL_Q = {cp: ["CP", "W", "--s1"], map: ["MAP (5 min)", "W", "--s2"], ftp: ["FTP (1-hour power)", "W", "--s4"], lt1: ["LT1", "W", "--s3"],
  w_prime: ["W′", "kJ", "--s7"], pmax: ["Pmax (sprint)", "W", "--muted"], fade: ["Fade past 30 min", "% of CP", "--ink2"]};
const KIND_COL = {easy: ["Easy ride", "--line"], hard: ["Hard ride", "--s3"], test: ["Test", "--s1"], maxed: ["Heart rate maxed", "--s2"]};
const CHAIN_COL = ["--s1", "--s2", "--s3", "--s4", "--s7"];
const READ_LINE = {lt1: ["LT1", "--s3"], cp: ["CP", "--s1"], map: ["MAP", "--s2"], ftp: ["FTP", "--s4"]};
const DOT_COL = ["--s1", "--s2", "--s3", "--s4", "--s7", "--ink2", "--muted"];
// threshold readings' sources (models.physio SOURCES) by the names people know
const READ_NAME = {my_ftp: "FTP you entered", alpha1_lt1: "alpha1 on rides (LT1)", ramp_lt1: "Ramp test, alpha1 (LT1)",
  ramp_lt2: "Ramp test, alpha1 (LT2)", ramp_hr_lt1: "Ramp test, heart rate (LT1)", ramp_hr_lt2: "Ramp test, heart rate (LT2)"};
const readName = (n) => READ_NAME[n] || n.replace(/_/g, " ");
const RUN_POLL_MS = 5000;
const STATUS_TIP = "Where the model stands.\n"
  + "• Version: of the stored fit, the code this app runs, the code on disk (all alike = fine)\n"
  + "• Fitted to: the best fit, re-made after every ride\n"
  + "• Backtest: when the forecast error was last checked";
const RUN_TIP = "A full sampling run for new 5-95 % ranges, then the backtest of the forecast error.\n"
  + "• Needed after a model change, or when the ranges fail their checks (the weekly refresh does the rest)\n"
  + "• Time: sampling usually under an hour (9 h at most), then the backtest (a few minutes); at low priority, so longer while the computer is busy\n"
  + "• Saved as it goes: stop it, and the next run goes on from there\n"
  + "• The plan is re-made after the sampling";

// the model's status (versions, fit, ranges, checks) and its full run: NUTS sampling, then the backtest (server
// FullModelRun). While it runs the card asks /api/model_run every few seconds and redraws only itself; done() when
// a run ends (the model's files changed)
class ModelRunCard {
  run = null;            // the last /api/model_run answer
  timer = null;

  constructor(done) { this.done = done; }

  // the status tiles: the model version (stored fit / this app / code on disk), what it's fitted to, its ranges' run,
  // the last backtest
  statusHtml(D, extra = "") {
    const V = D.versions || {}, B = D.backtest;
    const alike = V.fit === V.running && V.running === V.on_disk;
    const differ = `app v${esc(String(V.running))} · disk v${esc(String(V.on_disk))}`;
    return `<div class="tiles">
    ${tile("Version", `v${esc(String(V.fit ?? V.running ?? "?"))}`, "", alike ? "app and code alike" : differ,
      alike ? "up" : "down")}
    ${tile("Fitted to", D.day ? this.shortDate(D.day) : "–", "", D.day ? "best fit, after every ride" : "no fit yet")}
    ${tile("Backtest", B?.date ? this.shortDate(B.date) : "not run", "",
      B?.mae_pct != null ? `forecast error ${fmt(B.mae_pct, 1)} %` : "checks the forecast error")}
    ${extra}
  </div>`;
  }

  longDate(d) { return esc(niceDate(d, {day: "numeric", month: "short", year: "numeric"})); }

  shortDate(d) { return esc(niceDate(d, {day: "numeric", month: "short"})); }

  // verdict {state: "none" | "rough" | "ok" | "kept", say}: can the ranges be believed (ModelPage.rangesVerdict)
  html(D, verdict) {
    this.run = this.run || D.run || null;
    const V = verdict || {state: "none", say: ""};
    const [label, cls] = {none: ["Ranges: none yet", "gray"], rough: ["Ranges: rough", "amber"], ok: ["Ranges: trustworthy", "green"],
      kept: ["Ranges: from an earlier run", "gray"]}[V.state];
    return `<div class="card mdverdict"><div class="mdv-head"><span class="status ${cls}">${label}</span>
      <span class="mdv-say">${V.say}</span></div>
      <div id="mdRunCard">${this.cardHtml(V.state)}</div></div>`;
  }

  cardHtml(state = this.state) {
    const R = this.run;
    this.state = state;
    if (!R) return "";
    if (R.stage) {
      return `<div class="row"><span class="spin"></span><b>Step ${R.step} of ${R.steps}: ${esc(R.label)}</b>
      <span class="small muted">${this.span(R.stage_elapsed)} (all: ${this.span(R.elapsed)})</span>
      <span class="spacer"></span><button class="btn sm" id="mdRunStop">Stop</button></div>
      ${R.last ? `<div class="log">${esc(R.last)}</div>` : ""}`;
    }
    const end = R.result ? `<p class="small${R.result.ok ? "" : " muted"}" style="margin:8px 0 0">`
      + `Last run: ${esc(R.result.say)}</p>` : "";
    const O = R.other;
    if (O) {             // a sampling run started elsewhere (a terminal's, the weekly refresh): its progress
      const what = O.kind === "refresh" ? "The weekly ranges refresh" : "A sampling run started outside the app";
      return `<div class="row"><span class="spin"></span><b>${what} is going on</b>
        <span class="small muted">${this.span(O.elapsed)} so far${O.quiet > 600 ? ` · nothing new for ${this.span(O.quiet)}` : ""}</span></div>
        ${O.line ? `<div class="log">${esc(O.line)}</div>` : ""}
        <p class="small muted" style="margin:6px 0 0">This page updates when it ends. Stop it where it was started.</p>`;
    }
    const primary = this.state !== "ok";        // the ranges need a run: the main action; else a quiet one
    return `<div class="row">
      <button class="btn ${primary ? "primary" : "sm"}" id="mdRunGo"${R.busy ? " disabled" : ""}>Run sampling + backtest</button>
      ${hint(RUN_TIP)}
      <span class="small muted">${esc(R.busy || "")}</span></div>${end}`;
  }

  span(s) {             // seconds as "12 min" / "2 h 5 min"
    if (s == null) return "";
    if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min`;
    return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  }

  wire() {
    $("#mdRunGo")?.addEventListener("click", () => this.start());
    $("#mdRunStop")?.addEventListener("click", () => this.stop());
    this.poll();
  }

  async start() {
    if (!await ask.confirm({title: "Run sampling and the backtest?", yes: "Start the run",
      text: "A full sampling run first, then the backtest. It usually takes about an hour at low priority "
        + "(9 hours at most), and you can stop it any time."})) return;
    if (await api.action({action: "model_run"}, "Started: sampling, then the backtest")) this.poll();
  }

  async stop() {
    if (!await ask.confirm({title: "Stop the run?", yes: "Stop the run", no: "Keep it running",
      text: "The sampling is saved up to its last checkpoint and goes on from there next time."})) return;
    if (await api.action({action: "model_run_stop"}, "Stopping…")) this.poll();
  }

  // the run now: the card redrawn; again in a few seconds while it runs and the Model page is open
  poll() {
    clearTimeout(this.timer);
    fetch("/api/model_run", {cache: "no-store"}).then((r) => r.json()).then((j) => {
      const was = this.run?.stage || this.run?.other;
      this.run = j;
      if (was && !j.stage && !j.other) { this.done(); return; }
      const card = $("#mdRunCard");
      if (!card) return;
      card.innerHTML = this.cardHtml();
      $("#mdRunGo")?.addEventListener("click", () => this.start());
      $("#mdRunStop")?.addEventListener("click", () => this.stop());
      if (j.stage || j.other) this.timer = setTimeout(() => this.poll(), RUN_POLL_MS);
    }).catch(() => { this.timer = setTimeout(() => this.poll(), RUN_POLL_MS); });
  }
}

class ModelPage extends Page {
  // the Model page: how the physio fit is doing
  // (processed/physio_diagnostics.json, models.physio.PhysioDiagnostics): convergence of the last NUTS run, today's
  // thresholds over the draws, the model next to what you measured, and its checks (goodpush.checks: forecast error,
  // ranges checked)
  model = new Lazy("/api/model", () => { if (router.route() === "model") router.render(); });
  runCard = new ModelRunCard(() => { this.model.forget(); if (router.route() === "model") router.render(); });
  modelDur = 300;
  showEasy = new Pref("md_easy", false);       // predicted vs measured: easy rides' best powers (many, rarely all-out)
  readQ = "";                                  // threshold readings: the quantity shown ("" = the first there is)
  readFair = new Pref("md_read_fair", true);   // threshold readings with each source's learned bias taken out

  // the last NUTS run's checks: [all passed, what failed] (none: no run)
  samplerChecks() {
    const G = this.model.data.sampling;
    if (!G) return null;
    const div = G.divergences.reduce((a, b) => a + b, 0), n = G.chains * G.draws;
    const failed = [G.r_hat > 1.01 ? "chains disagree a little" : "", G.min_ess < 400 ? "too few effective draws" : "",
      div > 0.01 * n ? "many divergences" : ""].filter(Boolean);
    return [!failed.length, failed];
  }

  // can the ranges be believed: one verdict for the top of the page (ModelRunCard.html)
  rangesVerdict() {
    const D = this.model.data, ck = this.samplerChecks(), day = D.draws_day ? esc(niceDate(D.draws_day, {day: "numeric", month: "short"})) : "";
    if (!D.draws_day) {
      return {state: "none", say: "Today's numbers come from the best fit only, with no 5–95 % ranges. A sampling run adds them (usually under an hour, at low priority)."};
    }
    if (!ck) return {state: "kept", say: `The 5–95 % ranges are from the sampling run of ${day}.`};
    if (ck[0]) return {state: "ok", say: `The 5–95 % ranges are from the sampling run of ${day}, and its chains agree. They're refreshed about weekly.`};
    const why = ck[1].join(", ");
    return {state: "rough", say: `The sampling run of ${day} didn't pass its checks (${esc(why)}), so read the ranges below as rough. A longer run usually settles them.`};
  }

  // how far off its forecasts have been, and did its ranges hold on held-out data: tiles beside the status
  verdictSection() {
    const D = this.model.data, cov = D.coverage;
    const used = D.backtest;
    const err = used && used.mae_pct != null ? tile("Forecast error", fmt(used.mae_pct, 1), "%", "20-min power on your hard days, up to 6 weeks ahead", "",
      "Mean error predicting your near-maximal days (20-min equivalent) of the next 6 weeks.\n• Fitted up to many past cut-offs\n• Re-made with: goodpush physio --backtest") : "";
    const e = cov?.efforts;
    const held = e && e.n ? tile("Ranges checked", fmt(100 * e.inside_90), "%", `of ${fmt(e.n)} held-out best powers inside the 90 % range${cov.current ? "" : " (an older model version)"}`,
      Math.abs(e.inside_90 - 0.9) <= 0.05 ? "up" : "down",
      `At ${cov.folds} past cut-offs: sampled on the data before, then the next six weeks' best powers and readings placed in its forecast range.\n• About 90 % inside is right${cov.readings?.n ? `\n• Threshold readings: ${fmt(100 * cov.readings.inside_90)} % of ${fmt(cov.readings.n)}` : ""}`) : "";
    return `${err}${held}`;
  }

  // the last NUTS run, folded: its pass / fail on the summary line; the charts are drawn when it opens
  samplerSection() {
    const G = this.model.data.sampling;
    if (!G) return "";
    const [ok] = this.samplerChecks();
    const div = G.divergences.reduce((a, b) => a + b, 0), n = G.chains * G.draws;
    const rows = G.slowest.map((p) => `<tr><td><code>${esc(p.name)}</code>${p.judged ? "" : ' <span class="muted">(not judged)</span>'}</td><td class="n">${p.ess}</td><td class="n">${fmt(p.r_hat, 3)}</td><td class="n small">${p.chain_means.map((v) => fmt(v, 2)).join(" · ")}</td></tr>`).join("");
    return `<details class="fold" id="mdSampler"><summary>Sampler details: ${ok ? "passed" : "not passed yet"} <span class="small muted">(R-hat ${fmt(G.r_hat, 3)}, ESS ${fmt(G.min_ess)})</span></summary>
    ${hint("Is the sampler's picture of the uncertainty trustworthy? Several chains from different starts: if they agree, the ranges can be believed.\n• R-hat: how much chains disagree (1.00 none; above 1.01 worth a look)\n• ESS: independent draws they're worth (a few hundred is enough)\n• Divergences: a few harmless; many in one chain = it got stuck\n• Pmax's own parameters aren't judged (sprints are outside the plan)")}
    <p class="sub">NUTS run of ${esc(G.date)}: ${G.chains} chains × ${G.draws} draws after ${G.warmup} warm-up steps, ${G.parameters} parameters${this.model.data.draws_day && this.model.data.day !== this.model.data.draws_day ? ` · ranges as of ${esc(niceDate(this.model.data.draws_day, {day: "numeric", month: "short"}))} (re-made about weekly); best fit up to ${esc(niceDate(this.model.data.day, {day: "numeric", month: "short"}))}` : ""}</p>
    <div class="tiles">
      ${tile("Worst R-hat", fmt(G.r_hat, 3), "", G.r_hat <= 1.01 ? "chains agree" : "chains disagree a little", G.r_hat <= 1.01 ? "up" : "down", "The largest R-hat over the judged parameters.\n• 1.00 = chains agree; aim for 1.01 or less")}
      ${tile("Smallest ESS", fmt(G.min_ess), "", G.min_ess >= 400 ? "plenty for 5–95 % ranges" : "ranges still rough", G.min_ess >= 400 ? "up" : "down", "The fewest effective independent draws of any judged parameter.")}
      ${G.divergences.length ? tile("Divergences", fmt(div), "", `${G.divergences.join(" · ")} per chain`, div <= 0.01 * n ? "up" : "down", "Steps where the sampler couldn't follow the posterior.\n• Under 1 % of draws: fine\n• Many in one chain: it explored badly") : ""}
      ${G.steps.length ? tile("Steps a draw", G.steps.join(" · "), "", `step size ${fmt(G.step_size, 3)}`, "", "Leapfrog steps per draw, per chain.\n• Alike: the chains see the same shape\n• 1023: the tree's limit") : ""}
    </div>
    <div class="card"><div class="row"><div class="legend">${hint("Each chain along the run: its log posterior (how well it explains everything) and today's CP.\n• Well mixed: they overlap like fuzzy caterpillars\n• One apart: it hasn't found the same answer")}${Array.from({length: G.chains}, (_, c) => `<span><i style="background:var(${CHAIN_COL[c % 5]})"></i>Chain ${c + 1}</span>`).join("")}</div></div>
      <div class="small muted" style="margin-top:6px">Log posterior</div><div id="mdTraceLp" class="chart xs"></div>
      <div class="small muted" style="margin-top:6px">CP today</div><div id="mdTraceCp" class="chart xs"></div></div>
    <div class="card"><div class="row"><b>Slowest parameters</b>${hint("The parameters with the fewest effective draws, with each chain's mean: one apart = where they disagree.\n• x0: starting states\n• floor / log_tau / log_effect: how training builds and fades\n• eta: week-to-week changes nothing else explains\n• bias / log_sd: each reading kind's offset and spread")}</div>
      <div class="tw" style="margin-top:8px"><table class="small"><thead><tr><th>Parameter</th><th class="n">ESS</th><th class="n">R-hat</th><th class="n">Chain means</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="small muted" style="margin-top:10px">Effective draws, all ${G.parameters} parameters</div>${this.essBands(G)}</div>
    </details>`;
  }

  chartSampler() {
    const G = this.model.data?.sampling;
    if (!G || !$("#mdTraceLp")) return;
    const tr = G.trace, chainSeries = (k, f) => tr[k].map((y, c) => ({kind: "line", y, color: css(CHAIN_COL[c % 5]), name: `Chain ${c + 1}`, fmt: f, width: 1.2}));
    EChart.draw("mdTraceLp", {x: tr.draw, xType: "num", xLabel: (v) => `draw ${v}`, xAxis: false, yFmt: (v) => fmt(v), series: chainSeries("log_post", (v) => fmt(v, 1))});
    EChart.draw("mdTraceCp", {x: tr.draw, xType: "num", xLabel: (v) => `draw ${v}`, yFmt: (v) => v + " W", series: chainSeries("cp", (v) => fmt(v) + " W")});
  }

  // how many parameters have how many effective draws, in plain bands (from the log10 histogram's bin centres)
  essBands(G) {
    const bands = [[0, 100, "under 100", "too few: ranges unreliable", "--bad"], [100, 400, "100–400", "rough ranges", "--warn"],
      [400, 1000, "400–1,000", "enough for 5–95 % ranges", "--good"], [1000, Infinity, "1,000 or more", "plenty", "--good"]];
    const n = bands.map(([lo, hi]) => G.ess_hist.x.reduce((s, x, i) => s + (10 ** x >= lo && 10 ** x < hi ? G.ess_hist.n[i] : 0), 0));
    const top = Math.max(1, ...n);
    return `<div class="zbars notes">${bands.map(([, , label, what, col], i) => `<div class="zrow"><span>${label}</span>
      <div class="zb"><i style="width:${n[i] ? Math.max(1, n[i] / top * 100).toFixed(1) : 0}%;background:var(${col})"></i></div>
      <span class="n">${fmt(n[i])}</span><span class="small muted">${what}</span></div>`).join("")}</div>`;
  }

  // today over the draws: one row per quantity, its median, 5-95 % range and a bar of that range (around the median)
  uncertaintySection() {
    const P = this.model.data.posterior || {}, A = this.model.data.assumed || [], assumed = Object.fromEntries(A.map((a) => [a.name, a]));
    const rows = Object.entries(MODEL_Q).filter(([k]) => P[k]).map(([k, [name, unit]]) => {
      const q = P[k], d = ["w_prime", "fade"].includes(k) ? 1 : 0;
      const a = Math.min(q.p05, q.p50 * 0.85), b = Math.max(q.p95, q.p50 * 1.15), at = (v) => (100 * (v - a) / (b - a)).toFixed(1);
      return `<tr><td>${esc(name)}${assumed[k] ? ` <span class="chip">assumed</span>${hint(assumed[k].why)}` : ""}</td><td class="n">${fmt(q.p50, d)} ${unit}</td>
        <td class="n">${fmt(q.p05, d)}–${fmt(q.p95, d)} ${unit}</td>
        <td style="min-width:120px"><div class="rbar"><i style="left:${at(q.p05)}%;width:${(at(q.p95) - at(q.p05)).toFixed(1)}%;background:var(${MODEL_Q[k][2]})"></i><b style="left:${at(q.p50)}%"></b></div></td></tr>`;
    }).join("");
    const other = A.filter((x) => !MODEL_Q[x.name]), ck = this.samplerChecks();
    const rough = ck && !ck[0] ? `<p class="small" style="margin:0 0 8px;color:var(--warn)">Rough: the sampling run behind these ranges didn't pass its checks (see Sampler details).</p>` : "";
    return rows ? `<h2>Today, with its uncertainty${hint("Where the model puts you today: median and 5-95 % range over its draws.\n• Bar: the range around the median (track ±15 %, wider if needed)\n• Narrow: the data pin it down · wide: anywhere in there")}</h2>
  <div class="card">${rough}<div class="tw"><table><thead><tr><th>Today</th><th class="n">Median</th><th class="n">5–95 %</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
    ${other.length ? `<p class="small muted" style="margin:8px 0 0">Assumed, not learned: ${other.map((x) => `<b>${esc(x.what)}</b> (${esc(x.short)})${hint(x.why)}`).join(" · ")}</p>` : ""}</div>` : "";
  }

  fatigueAssumed() {    // the fit couldn't learn what fatigue costs on the day (models.physio.PhysioDiagnostics.assumed)
    return (S.physio?.assumed || []).some((a) => a.name === "log_s_f");
  }

  fatigueSection() {
    const P = S.physio;
    if (!P || !P.fatigue) return "";
    return `<h2>Fatigue${hint("Your recent hours at CP, fading over ~10 days.\n• Lowers what you can do on the day, not your fitness\n• Slows how fast training turns into fitness\n" + (this.fatigueAssumed() ? "• Its cost on the day is assumed: you ride all-out efforts fresh, so rides can't show it\n" : "• Its cost on the day is learned from your history\n") + "• The plan backs off when more fatigue costs more than it gains, and stays within 10 % above your highest so far")}</h2>
  <div class="card"><div class="legend"><span><i style="background:var(--s2)"></i>Model fatigue</span><span><i class="dash" style="border-color:var(--s2)"></i>Planned</span>${P.fatigue_cap ? '<span><i class="dash" style="border-color:var(--muted)"></i>Your highest so far</span>' : ""}</div>
    <div id="mdFatigue" class="chart sm"></div></div>`;
  }

  chartFatigue() {      // the model's fatigue, then the plan's forecast (dashed); opens on the last year
    const P = S.physio;
    if (!P || !P.fatigue || !$("#mdFatigue")) return;
    const F = P.forecast || {date: []}, d = P.date, last = d[d.length - 1];
    const fut = F.date.map((x, i) => [x, i]).filter(([x]) => x > last);
    const x = d.concat(fut.map(([q]) => q)), ms = (q) => new Date(q + "T12:00:00").getTime(), f1 = (v) => fmt(v, 1);
    const ahead = d.map((_, i) => (i === d.length - 1 ? P.fatigue[i] : null)).concat(fut.map(([, i]) => F.fatigue[i]));
    TimeWindow.of("mdFatigue").showLast(365, ms(last), ms(x[x.length - 1]), 365);
    EChart.draw("mdFatigue", {x, xType: "date", yFmt: f1, scrollbar: true, vline: fut.length ? ms(last) : null, planFrom: ms(last), hline: P.fatigue_cap, series: [
      {kind: "line", y: P.fatigue.concat(fut.map(() => null)), color: css("--s2"), name: "Fatigue", fmt: f1},
      {kind: "line", y: ahead, color: css("--s2"), dash: true, name: "Fatigue, planned", planned: true, fmt: f1}]});
  }

  // what each kind of riding did for you (models.physio.TrainingResponse; moved from the Rider page), folded
  ridingSection() {
    const R = S.response;
    if (!R || !R.workouts) return "";
    const E = R.effects || {}, src = R.sources || {}, hw = src.hours_week || {}, ex = R.experience || {}, dw = riderPage.domainWatts();
    const PARTS = [["f_e", "LT1, closer to 1-h power"], ["f_t", "CP, closer to MAP"], ["map", "MAP (5-min power)"]];   // threshold order
    const vs = (r) => r == null ? "" : `<div class="small muted">${r >= 1.5 ? `${fmt(r, 1)}× assumed` : r <= 0.67 ? `${fmt(r, 2)}× assumed` : "≈ as assumed"}</div>`;
    const share = (d, k) => {
      const v = src[k]?.[d], r = E[d]?.trains?.[k];
      if (v == null && r == null) return `<td class="muted small">doesn't train it</td>`;
      return `<td><div class="bar sm"><i style="width:${((v || 0) * 100).toFixed(0)}%;background:var(${DOMAIN[d][0]})"></i></div>
      <div class="small"><b>${fmt((v || 0) * 100)} %</b> of the gain</div>${vs(r)}</td>`;
    };
    const rows = DOMAIN_ORDER.filter((d) => E[d]).map((d) => `<tr><td>${riderPage.swatch(d)}<b>${esc(DOMAIN[d][1])}</b>${dw[d] ? `<div class="small muted">${dw[d]}</div>` : ""}</td>
      <td class="n">${fmt(hw[d], 1)} h<div class="small muted">${fmt(ex[d])} h in 2 years</div></td>
      ${PARTS.map(([k]) => share(d, k)).join("")}</tr>`).join("");
    return `<details class="fold" open><summary>What each kind of riding did for you (last 12 months)</summary>
  <div class="card">${hint("Each kind of riding: your hours a week and its share of each gain (how much you rode × how well it trains that part). A column adds up to 100 %.\n• Below a share: how well it trains that part vs the model's prior\n• \"≈ as assumed\": your data barely changed it (an assumption, not a finding)\n• LT1 closer to 1-h power: how high easy riding reaches\n• CP closer to MAP: how much of it you can sustain\n• MAP: 5-minute power, the ceiling")}
    <div class="tw"><table><thead><tr><th>Riding</th><th class="n">Hours a week</th>${PARTS.map(([, n]) => `<th style="min-width:130px">${n}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div></div></details>`;
  }

  html() {
    if (!this.model.data) { this.model.load(); return `<h1>The model</h1><p class="muted">Loading…</p>`; }
    if (!this.model.data.day) {
      return `<h1>The model</h1><div class="card">No model diagnostics yet: they're written with the next fit `
        + `(<code>goodpush physio</code>, or the app's next update).</div>${this.runCard.html(this.model.data)}`;
    }
    const R = this.model.data.readings || {sources: []};
    const src = R.sources.filter((s) => s.n).map((s) => `<tr><td>${esc(readName(s.name))}</td><td>${esc(s.quantity.toUpperCase())}${s.bound ? " (cap)" : ""}</td><td class="n">${s.n}</td><td class="n">${s.bias_pct > 0 ? "+" : ""}${fmt(s.bias_pct, 1)} %</td><td class="n">${s.sd_pct == null ? "–" : "±" + fmt(s.sd_pct, 1) + " %"}</td></tr>`).join("");
    const used = this.readQuantities(), q = used.includes(this.readQ) ? this.readQ : used[0];
    return `<h1>The model${hint("Can you trust the physiology model behind the Rider page and the plan?\n• Can its ranges be believed\n• How far off its forecasts have been\n• How sure it is of today's numbers\n• Its predictions next to what you rode and measured\nOne rider, observational: a check, not proof.")}</h1>
  <p class="sub">${this.versionLine()} · fitted to your data up to ${esc(niceDate(this.model.data.day, {day: "numeric", month: "short", year: "numeric"}))}. <a href="#rider">← Rider</a> · <a href="${GUIDE}model/overview.html" target="_blank" rel="noopener">How the model works ↗</a></p>
  ${this.versionWarning()}
  ${this.runCard.html(this.model.data, this.rangesVerdict())}
  <h2>Status${hint(STATUS_TIP)}</h2>
  ${this.runCard.statusHtml(this.model.data, this.verdictSection())}
  ${this.samplerSection()}
  ${this.uncertaintySection()}
  ${this.fatigueSection()}
  ${this.trainingSection()}
  ${this.ridingSection()}
  <h2>Predicted vs observed${hint("The model's all-out power on the day (fatigue included) against each ride's best, last year.\n• All-out efforts: on the line · most rides: below it\n• Well above the line: efforts the model didn't think you could do\n• Easy rides hidden unless you show them")}</h2>
  <div class="card"><div class="row"><div class="legend" id="mdKindLegend">${this.kindLegend()}</div>
    <span class="spacer"></span><div class="seg"><button id="mdEasy" class="${this.showEasy.value ? "on" : ""}" title="Show easy rides' best powers too">Easy rides</button></div>
    <div class="seg" id="mdDur">${[60, 300, 1200].map((d) => `<button data-s="${d}" class="${d === this.modelDur ? "on" : ""}">${d / 60} min</button>`).join("")}<button data-s="0" class="${this.modelDur === 0 ? "on" : ""}">All</button></div></div>
    <div id="mdEffort" class="chart"></div></div>
  ${used.length ? `<h2>Threshold readings${hint("Each reading (alpha1, ramp tests, heart-rate estimates, FTP you entered) next to the model's line for what it measures.\n• Each kind has its own learned offset (bias) and spread\n• Bias removed: each reading divided by its kind's offset, so it can be compared with the line\n• As read: the raw numbers\n• Ramp-test readings are caps: the threshold can sit below, not above")}</h2>
  <div class="card"><div class="row"><div class="legend" id="mdReadLegend"></div><span class="spacer"></span>
    <div class="seg" id="mdReadFair"><button data-f="1" class="${this.readFair.value ? "on" : ""}">Bias removed</button><button data-f="0" class="${this.readFair.value ? "" : "on"}">As read</button></div>
    <div class="seg" id="mdReadQ">${used.map((k) => `<button data-q="${k}" class="${k === q ? "on" : ""}">${READ_LINE[k][0]}</button>`).join("")}</div></div>
    <div id="mdRead" class="chart"></div>
    <div class="tw" style="margin-top:10px"><table class="small"><thead><tr><th>Reading</th><th>Measures</th><th class="n">Count</th><th class="n">Bias${hint("How far this kind of reading sits from the model on average (+ = reads high).")}</th><th class="n">Spread${hint("How noisy this kind of reading is: its typical difference from the model, after the bias.")}</th></tr></thead><tbody>${src}</tbody></table></div></div>` : ""}`;
  }

  // the model version: of the stored fit, of the code this app runs, of the code on disk (models.physio.PhysioRun)
  versionLine() {
    const V = this.model.data.versions || {};
    return `Model <b>v${esc(String(V.fit ?? V.running ?? "?"))}</b>`;
  }

  versionWarning() {   // they should all be one: an app on older code than the disk refits nothing until restarted
    const V = this.model.data.versions || {};
    const say = V.running != null && V.on_disk != null && V.running !== V.on_disk
      ? `This app runs model v${V.running}, the code on disk is v${V.on_disk}: restart the app. Until then it doesn't refit (a refit would write the old model over the fit).`
      : V.fit != null && V.running != null && V.fit !== V.running
        ? `The stored fit is model v${V.fit}, this app runs v${V.running}: the next update refits it; ranges need a new run (goodpush physio --nuts ...).`
        : "";
    return say ? `<div class="card" style="background:var(--warn-bg);color:var(--warn)">${esc(say)}</div>` : "";
  }

  kindLegend() {
    return Object.entries(KIND_COL).filter(([k]) => k !== "easy" || this.showEasy.value).map(([, [n, c]]) => `<span><i class="dot" style="background:var(${c})"></i>${n}</span>`).join("")
      + '<span><i class="dash" style="border-color:var(--muted)"></i>Observed = predicted</span>';
  }

  // the quantities with readings (LT1 | CP | MAP | FTP order), for the readings chart's buttons
  readQuantities() {
    const R = this.model.data.readings || {sources: []}, have = new Set(R.sources.filter((s) => s.n).map((s) => s.quantity));
    return Object.keys(READ_LINE).filter((k) => have.has(k));
  }

  trainingSection() {
    const T = this.model.data.training || {};
    const tiles = Object.entries(MODEL_Q).filter(([k]) => T[k] && T[k].share != null).map(([k, [name]]) => {
      const t = T[k], flat = t.training_change_pct < 0.6 * t.change_pct;
      const say = t.corr != null && t.corr >= 0.5 && flat ? "gains and losses at the right times, but too small"
        : t.share >= 0.5 ? "training accounts for most of the change" : t.share < 0.2 ? "training explains little of the change" : "training explains part of the change";
      const together = t.corr == null ? "" : ` The two rise and fall together ${fmt(t.corr, 2)} (1 = always).`;
      return tile(name, fmt(100 * Math.max(t.share, 0)), "% from training", say, flat ? "down" : "up",
        `Over ${T.days / 7}-week stretches your ${name} typically moved ${fmt(t.change_pct, 1)} % (from your rides `
        + `and tests). Predicted from your training alone it moves ${fmt(t.training_change_pct, 1)} %.${together}`);
    }).join("");
    const tip = "How well your training alone predicts your thresholds.\n"
      + "• % from training: share of each threshold's ups and downs it predicts\n"
      + "• The rest: changes seen in your rides and tests that training doesn't account for\n"
      + "• \"Right times, but too small\": it predicts when you gain or lose, but less than really happened\n"
      + "• Then the plan's forecast likely understates your change too\n"
      + "One rider, observational: a check on the model.";
    return tiles ? `<h2>What training explains${hint(tip)}</h2><div class="tiles">${tiles}</div>` : "";
  }

  chartEfforts() {
    const E = this.model.data.efforts;
    if (!E || !$("#mdEffort")) return;
    const pick = E.s.map((s, i) => (!this.modelDur || s === this.modelDur) ? i : -1).filter((i) => i >= 0)
      .filter((i) => this.showEasy.value || E.kind[i] !== "easy");
    if (!pick.length) return;
    // both axes over one range (watts on both), so "observed = predicted" runs corner to corner
    const all = pick.flatMap((i) => [E.predicted[i], E.observed[i]]), pad = 0.04 * (Math.max(...all) - Math.min(...all) || 50);
    const span = Math.max(...all) - Math.min(...all) + 2 * pad, tick = [10, 20, 25, 50, 100, 200].find((t) => span / t <= 7) || 500;
    const lo = Math.floor((Math.min(...all) - pad) / tick) * tick, hi = Math.ceil((Math.max(...all) + pad) / tick) * tick;
    EChart.draw("mdEffort", {x: [lo, hi], xType: "num", xDomain: [lo, hi], yDomain: [lo, hi], tick,
      xName: "Predicted: the model's all-out power that day", yName: "Observed: your best",
      xLabel: (v) => `model ${fmt(v)} W`, xTickFmt: (v) => fmt(v) + " W", yFmt: (v) => v + " W",
      series: [...Object.entries(KIND_COL).map(([k, [n, c]]) => {    // the dots first: hover follows the first series
          const ix = pick.filter((i) => E.kind[i] === k);
          return {kind: "dots", r: k === "easy" ? 2 : 3, x: ix.map((i) => E.predicted[i]), y: ix.map((i) => E.observed[i]), color: css(c), name: n, fmt: (v) => fmt(v) + " W"};
        }).filter((q) => q.x.length),
        {kind: "line", x: [lo, hi], y: [lo, hi], color: css("--muted"), dash: true, width: 1, range: false, name: ""}]});
  }

  // one quantity: its model line and the readings of the sources that measure it (dots never in the line's colour)
  chartReadings() {
    const R = this.model.data.readings, used = this.readQuantities();
    if (!R || !used.length || !$("#mdRead")) return;
    const q = used.includes(this.readQ) ? this.readQ : used[0], [qName, qCol] = READ_LINE[q];
    const of = Object.fromEntries(R.sources.map((s) => [s.name, s.quantity]));
    const names = [...new Set(R.source)].filter((n) => of[n] === q), cols = DOT_COL.filter((c) => c !== qCol);
    const D = R.model, ms = (v) => new Date(v + "T12:00:00").getTime(), fair = this.readFair.value;
    const bias = Object.fromEntries(R.sources.map((s) => [s.name, 1 + (s.bias_pct || 0) / 100]));
    const seen = (n, i) => fair ? R.observed[i] / bias[n] : R.observed[i];
    $("#mdReadLegend").innerHTML = `<span><i style="background:var(${qCol})"></i>Model ${qName}</span>` +
      names.map((n, j) => `<span><i class="dot" style="background:var(${cols[j % cols.length]})"></i>${esc(readName(n))}</span>`).join("");
    EChart.draw("mdRead", {x: D.date, xType: "date", yFmt: (v) => v + " W", series: [
      {kind: "line", y: D[q] || [], color: css(qCol), name: `Model ${qName}`, fmt: (v) => fmt(v) + " W", width: 1.6},
      ...names.map((n, j) => { const ix = R.source.map((s, i) => s === n ? i : -1).filter((i) => i >= 0);
        return {kind: "dots", r: 3, x: ix.map((i) => ms(R.date[i])), y: ix.map((i) => seen(n, i)), color: css(cols[j % cols.length]),
          name: readName(n) + (fair && Math.abs(bias[n] - 1) >= 0.005 ? ", bias removed" : ""), fmt: (v) => fmt(v) + " W"}; })]});
  }

  chartModel() {
    if (!this.model.data || !this.model.data.day) return;
    this.chartEfforts();
    this.chartReadings();
    this.chartFatigue();
    if ($("#mdSampler")?.open) this.chartSampler();
  }

  wire() {
    this.runCard.wire();
    requestAnimationFrame(() => { this.chartModel(); });
    new Segmented("#mdDur", (b) => { this.modelDur = Number(b.dataset.s); this.chartEfforts(); });
    new Segmented("#mdReadQ", (b) => { this.readQ = b.dataset.q; this.chartReadings(); });
    new Segmented("#mdReadFair", (b) => { this.readFair.set(b.dataset.f === "1"); this.chartReadings(); });
    $("#mdEasy")?.addEventListener("click", (e) => {
      this.showEasy.set(!this.showEasy.value);
      e.target.classList.toggle("on", this.showEasy.value);
      $("#mdKindLegend").innerHTML = this.kindLegend();
      this.chartEfforts();
    });
    $("#mdSampler")?.addEventListener("toggle", (e) => { if (e.target.open) requestAnimationFrame(() => this.chartSampler()); });
  }
}
export const modelPage = new ModelPage();
