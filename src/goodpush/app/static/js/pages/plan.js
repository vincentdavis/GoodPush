// Plan: phases and races, intensity, interval levels, the 20-minute test (PlanPage).
import {Page} from "./page.js";
import {api, S, toast} from "../api.js";
import {$, addDays, esc, fmt} from "../util.js";
import {ask} from "../controls.js";
import {niceDate, TYPE_LABEL} from "../widgets.js";
import {forms, router} from "../main.js";
import {HOURS_LIMITED_TIP, PUSH_WEEK_TIP, rampChip} from "../calendar/calendar.js";
import {seasonBlocks} from "../calendar/season.js";

// Settings cards built from SETTINGS in config.py (S.settings.schema): every goodpush.toml option, checked and
// saved by the server ("settings" action, one section per card)
export const DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
export const SAVE_LABEL = {reanalyse: "Save and re-analyse", replan: "Save and rebuild plan", reload: "Save", nothing: "Save"};
export const WIDE_KINDS = ["week", "days", "numbers", "history", "events", "lines"];
const STYLE_TEXT = {
  model: "Model: your fitted physiology picks each key day's session for the biggest gain on your race days "
    + "and at the end of the plan.",
  polarized: "Polarized: easy riding plus hard work above threshold only (VO2max, over-unders, anaerobic).",
  pyramidal: "Pyramidal: one sweet-spot day early (tempo only in long rides), threshold and VO2max later.",
};

class PlanPage extends Page {
  editingRace = null;          // the race being edited in the race form (its id), else the form adds one

  mixText(mix) {
    if (!mix) return "";
    if (mix.error) return ` The model couldn't choose (${esc(mix.error)}): pyramidal for now.`;
    const what = Object.entries(mix.sessions || {}).map(([k, n]) => `${TYPE_LABEL[k] || k} ${n}`).join(", ");
    const by = mix.draws ? `averaged over ${mix.draws} model draws` : "best fit";
    return what ? ` This plan: ${esc(what)} (${by}).` : "";
  }

  seasonCard() {
    const se = {phases: [], events: [], ...(S.plan?.season || {}), ...(S.season || {})};   // tables: your season file
    const meta = S.season_meta || {types: {}, kinds: []};
    const end = (p) => addDays(p.start, 7 * p.weeks - 1);
    const recs = (p) => (p.recovery_weeks || []).length
      ? "week " + p.recovery_weeks.map((k) => k + 1).join(", ") : "none";
    // edited on the Calendar (drag the blocks: calendar/season.js); listed here
    const phases = se.phases.map((p) => `<tr class="click" data-phase="${esc(p.id)}"
      title="Open it in the season editor on the Calendar"><td><a href="#calendar">${esc(p.name)}</a></td>
    <td>${esc(niceDate(p.start))} – ${esc(niceDate(end(p)))}</td><td class="n">${p.weeks} weeks</td><td>${recs(p)}</td>
    <td class="n">${p.hours ? fmt(p.hours, 1) + " h/week" : "your usual hours"}</td></tr>`).join("");
    const events = se.events.map((e) => `<tr><td>${esc(niceDate(e.date))}</td><td><b>${esc(e.priority)}</b></td><td>${esc(e.name)}</td>
    <td>${esc(e.kind)}</td><td class="muted">${esc(e.series)}</td><td><button class="btn sm" data-edit-event="${esc(e.id)}">Edit</button>
      <button class="btn sm" data-del-event="${esc(e.id)}" data-what="${esc(`the race ${e.name || ""} on ${niceDate(e.date)}`)}">Delete</button></td></tr>`).join("");
    const opts = (xs, sel) => xs.map(([v, l]) => `<option value="${esc(v)}" ${v === sel ? "selected" : ""}>${esc(l)}</option>`).join("");
    const ed = se.events.find((e) => e.id === this.editingRace);
    const sug = se.suggestion || [];
    const generic = sug.some((p) => p.generic);
    const sugHead = generic ? `<h3>A plan to start with</h3>
    <p class="small muted" style="margin-top:0">No race ahead and no phases planned: a general block from next Monday.
      Add a race any time and plan back from it instead.</p>` : `<h3>Suggested from your races</h3>
    <p class="small muted" style="margin-top:0">Weeks your A race needs that none of your phases cover, planned back from the race.</p>`;
    const sugBox = sug.length ? `<div class="card proposal" style="margin-bottom:12px">${sugHead}
    <div class="tw"><table><thead><tr><th>Phase</th><th>Dates</th><th class="n">Length</th><th>Why</th></tr></thead><tbody>${sug.map((p) =>
        `<tr><td>${esc((meta.types[p.type] || {}).label || p.type)}</td><td>${esc(niceDate(p.start))} – ${esc(niceDate(end(p)))}</td>
      <td class="n">${p.weeks} week${p.weeks > 1 ? "s" : ""}</td><td class="small">${esc(p.why)}</td></tr>`).join("")}</tbody></table></div>
    <div class="row" style="margin-top:10px"><button class="btn primary" id="acceptphases">Add these phases</button>
      <button class="btn" id="dismissphases" data-sig="${esc(se.suggestion_sig || "")}">Not now</button></div></div>` : "";
    const unlocked = seasonBlocks.dirty ? `<div class="adj" style="margin-bottom:12px">🔓 You changed the season on
      the <a href="#calendar">Calendar</a> but haven't locked it: this plan is still the old season's.</div>` : "";
    return `<h2>Phases &amp; races</h2>
  ${unlocked}${sugBox}
  <div class="card">
    <div class="tw"><table>${phases ? "<thead><tr><th>Phase</th><th>Dates</th><th class=\"n\">Length</th>"
      + "<th>Recovery weeks</th><th class=\"n\">Hours</th></tr></thead>" : ""}
      <tbody>${phases || '<tr><td class="muted">No phases: nothing is planned.</td></tr>'}</tbody></table></div>
    <p class="small muted">Build and change your season on the <a href="#calendar">Calendar</a> (click a phase to open
      it there): drag Base, Build, Specialty and Recovery blocks onto the chart, then lock 🔒 it to re-plan, or
      auto-plan it from your races.</p>
    <h3 style="margin-top:16px">Races</h3>
    <div class="tw"><table>${events ? "<thead><tr><th>Date</th><th>Priority</th><th>Name</th><th>Type</th><th>Series</th><th></th></tr></thead>" : ""}
      <tbody>${events || '<tr><td class="muted">No races yet.</td></tr>'}</tbody></table></div>
    ${ed ? `<p class="small" style="margin:10px 0 0"><b>Editing ${esc(ed.name)}</b> (${esc(niceDate(ed.date))})</p>` : ""}
    <div class="grid g3" style="margin-top:10px" id="ev-form">
      <label class="f">Date<input type="date" id="ev-date" value="${esc(ed?.date || "")}"></label>
      <label class="f">Priority<select id="ev-pr">${opts([["A", "A – peak for it"], ["B", "B – short taper"], ["C", "C – train through"]], ed?.priority || "B")}</select></label>
      <label class="f">Type<select id="ev-kind">${opts(meta.kinds.map((k) => [k, k]), ed?.kind || "road")}</select></label>
      <label class="f">Name<input type="text" id="ev-name" value="${esc(ed?.name || "")}"></label>
      <label class="f">Series (optional)<input type="text" id="ev-series" value="${esc(ed?.series || "")}"></label>
      <div class="row" style="align-self:end"><button class="btn primary" id="addevent">${ed ? "Save race" : "Add race"}</button>
        ${ed ? '<button class="btn" id="canceledit">Cancel</button>' : ""}</div>
    </div>
    <details id="ev-paste" style="margin-top:10px"><summary class="small">Paste several races</summary>
      <textarea class="snip" id="ev-text" placeholder="2026-11-08, B, crit, Race name&#10;2026-11-15, A, road, Final"></textarea>
      <div class="row"><input type="text" id="ev-tseries" placeholder="Series (optional)" style="max-width:240px">
        <button class="btn sm primary" id="addevents">Add all</button>
        <span class="small muted">One per line: date, priority, type, name (priority / type / name optional)</span></div></details>
  </div>`;
  }

  html() {
    const pl = S.plan || {};
    const weeks = (pl.weeks || []).slice(0, 4).map((w) => `
    <div class="tile">
      <div class="l">Week of ${niceDate(w.week, {day: "numeric", month: "short"})} · ${esc(w.phase || "")} ${w.block}/${w.week_in_block}${w.recovery ? " (recovery)" : w.taper ? " (taper)" : ""}${w.push ? ` <span class="pushwk" title="${PUSH_WEEK_TIP}">push week</span>` : ""}</div>
      <div class="v">${fmt(w.hours_planned, 1)}<small>h</small> · ${fmt(w.tss_planned)}<small>TSS</small></div>
      <div class="s">target ${fmt(w.tss_target)} TSS (CTL ${w.ramp_target > 0 ? "+" : ""}${w.ramp_target}/wk)${w.capped ? " – capped by your hours" : ""}</div>
      ${w.hours_limited ? `<div class="s hrlim" title="${HOURS_LIMITED_TIP}">at your hours: load can only grow with more time</div>` : ""}
      ${w.ramp_flag ? `<div class="s">${rampChip(w.ramp_flag, w.ramp_4wk, pl.ramp_range)} CTL ${w.ramp_4wk > 0 ? "+" : ""}${w.ramp_4wk}/wk over 4 weeks</div>` : ""}
    </div>`).join("");
    // in power order, lowest first (durability by its efforts: just under FTP, above it when polarized);
    // when training polarized the types it doesn't use are greyed in place
    const style = (S.plan_cfg || {}).intensity || "model";
    const polar = style === "polarized", POLAR = ["vo2max", "anaerobic", "durability"];
    const order = polar ? ["tempo", "sweetspot", "threshold", "durability", "vo2max", "anaerobic"]
      : ["tempo", "sweetspot", "durability", "threshold", "vo2max", "anaerobic"];
    const rank = (k) => order.indexOf(k);
    const lad = Object.entries(S.ladders || {}).sort(([a], [b]) => rank(a) - rank(b)).map(([k, steps]) => {
      const lv = Math.min((S.levels || {})[k] || 0, steps.length - 1), off = polar && !POLAR.includes(k);
      const opts = steps.map((s, i) => `<option value="${i}" title="${esc(s)}"${i === lv ? " selected" : ""}>${i + 1} / ${steps.length}</option>`).join("");
      return `<div class="ladder${off ? " unused" : ""}"><div>${TYPE_LABEL[k] || k}${off ? '<div class="small muted">not used (polarized)</div>' : ""}</div><div class="bar"><i style="width:${((lv + 1) / steps.length * 100).toFixed(0)}%"></i></div>
      <select data-level="${esc(k)}" aria-label="${esc(TYPE_LABEL[k] || k)} level">${opts}</select><div class="small muted lv">${esc(steps[lv])}</div></div>`;
    }).join("");
    return `
  <h1>Plan</h1>
  <p class="sub">FTP for targets <b>${fmt(pl.ftp)} W</b> (${esc(pl.ftp_source || "")}) · load (CTL) ${fmt(pl.ctl_now)} → ${fmt(pl.ctl_end)} by the end of this plan${pl.need_test ? " · FTP test due" : ""}</p>
  <div class="tiles weeks">${weeks}</div>
  <p class="small muted">The whole plan is on the <a href="#calendar">Calendar</a>.</p>
  ${this.seasonCard()}
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><div class="row"><h3>Interval progression</h3><span class="spacer"></span>
      <div class="seg" id="intensityseg" title="What the hard days are (also in Settings)">
        ${[["model", "Model"], ["polarized", "Polarized"], ["pyramidal", "Pyramidal"]].map(([v, t]) =>
          `<button data-v="${v}"${style === v ? ' class="on"' : ""}>${t}</button>`).join("")}</div></div>
      <div class="small muted" style="margin-bottom:6px">${STYLE_TEXT[style] || ""}${this.mixText(pl.session_mix)}
        Each type moves up a step when you complete it; pick a level yourself if it's off. The formats take turns week by week.</div>${lad}</div>
    <div class="card"><h3>Record an FTP test</h3>
      <div class="small muted">Enter the average power of your 20-minute effort; targets use 95 % of it.</div>
      <div class="row" style="margin-top:10px"><input type="number" id="p20" min="80" max="600" placeholder="20-min avg W" style="max-width:160px"><button class="btn primary" id="savetest">Save</button></div>
      <h3 style="margin-top:18px">Calendar</h3>
      <a class="btn sm" href="/files/plan.ics" download>Download plan.ics</a>
    </div>
  </div>
  <p class="small muted" style="margin-top:16px">Hours per week, rest / key / long-ride days and FTP override are in <a href="#settings/plan">Settings</a>.</p>
`;
  }

  wire() {
    document.querySelectorAll("tr[data-phase]").forEach((r) => r.addEventListener("click", () => {
      seasonBlocks.want = r.dataset.phase; location.hash = "#calendar";
    }));
    document.querySelectorAll("#intensityseg button").forEach((b) => b.addEventListener("click", async () => {
      if (b.classList.contains("on")) return;
      if (!await ask.confirm({title: "Change how your hard days are chosen?", yes: `Switch to ${b.textContent}`,
        text: `${STYLE_TEXT[b.dataset.v]}\n\nThe plan is re-made from today, so your coming hard sessions change.`})) return;
      api.action({action: "settings", values: {intensity: b.dataset.v}}, `${b.textContent} – updating the plan…`);
    }));
    document.querySelectorAll("select[data-level]").forEach((sel) => sel.addEventListener("change", () => {
      api.action({action: "set_level", kind: sel.dataset.level, level: Number(sel.value)}, "Level saved – updating the plan…");
    }));
    $("#savetest").addEventListener("click", () => {
      const v = Number($("#p20").value);
      if (!v) { toast.show("Enter your 20-minute average power"); return; }
      api.action({action: "test_result", p20: v}, `FTP set to ${Math.round(v * 0.95)} W – rebuilding plan…`);
    });
    const val = (id) => $(id).value.trim();
    $("#addevent").addEventListener("click", async () => {
      if (!val("#ev-date")) { toast.show("Pick the race date"); return; }
      const values = {date: val("#ev-date"), priority: val("#ev-pr"), kind: val("#ev-kind"), name: val("#ev-name"),
        series: val("#ev-series")};
      const id = this.editingRace;
      if (await api.action(id ? {action: "edit_event", id, values} : {action: "add_event", values},
        id ? "Race saved – re-planning…" : "Race added – re-planning…")) {
        this.editingRace = null;
        forms.clearFields("ev-date", "ev-pr", "ev-kind", "ev-name", "ev-series");
      }
    });
    $("#canceledit")?.addEventListener("click", () => {
      this.editingRace = null;
      forms.clearFields("ev-date", "ev-pr", "ev-kind", "ev-name", "ev-series");
      router.render();
    });
    document.querySelectorAll("[data-edit-event]").forEach((b) => b.addEventListener("click", () => {
      this.editingRace = b.dataset.editEvent;
      forms.clearFields("ev-date", "ev-pr", "ev-kind", "ev-name", "ev-series");   // the race's values, not what was typed
      router.render();
      $("#ev-form")?.scrollIntoView({block: "center", behavior: "smooth"});
    }));
    $("#addevents").addEventListener("click", async () => {
      if (await api.action({action: "add_events", values: {text: $("#ev-text").value, series: val("#ev-tseries")}},
        "Races added – re-planning…")) forms.clearFields("ev-text", "ev-tseries");
    });
    $("#acceptphases")?.addEventListener("click", () => api.action({action: "accept_phases"}, "Phases added – re-planning…"));
    $("#dismissphases")?.addEventListener("click", (e) => api.action({action: "dismiss_phases", sig: e.target.dataset.sig}, "Hidden until your races or phases change"));
    document.querySelectorAll("[data-del-event]").forEach((b) => b.addEventListener("click", async () => {
      if (!await ask.confirm({title: `Delete ${b.dataset.what}?`, text: "The plan no longer tapers for it.", yes: "Delete",
        danger: true})) return;
      api.action({action: "delete_event", id: b.dataset.delEvent}, "Race deleted – re-planning…");
    }));
  }
}
export const planPage = new PlanPage();
