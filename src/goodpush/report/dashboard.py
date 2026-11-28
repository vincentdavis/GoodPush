"""Self-contained HTML dashboard (Plotly.js loaded from CDN)."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from ..metrics.power import MMP_DURATIONS
from ..models.insights import FEATURES

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _clean(v):
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    return v


def _col(s: pd.Series):
    return _clean(s.tolist())


def _dates(idx):
    return [d.date().isoformat() for d in idx]


def build_payload(r: dict, res: dict) -> dict:
    d, W, acts = r["daily"], r["weekly"], r["acts"]
    rides = acts[acts["sport"].eq("ride") & acts["has_power"]]
    pdc = {"durations": MMP_DURATIONS}
    cols = [f"mmp_{x}" for x in MMP_DURATIONS]
    pdc["all"] = _clean([res["power"]["bests_all"][str(x)]["w"] for x in MMP_DURATIONS])
    last90 = rides[rides["date"] >= acts["date"].max() - pd.Timedelta(days=90)]
    pdc["last90"] = _clean(last90[cols].max().tolist())
    by_year = rides.groupby(rides["date"].dt.year)
    if len(rides):
        best_year = int(by_year["mmp_1200"].max().idxmax())
        pdc["best_year"] = best_year
        pdc["best_year_curve"] = _clean(by_year[cols].max().loc[best_year].tolist())
    zone_h = {f"Z{i}": _col(W[f"pz{i}_s"] / 3600) for i in range(1, 8)}
    unmeasured = (W["ride_hours"] - W[[f"pz{i}_s" for i in range(1, 8)]].sum(axis=1) / 3600
                  ).clip(lower=0) + (W["hours"] - W["ride_hours"]).clip(lower=0)
    tab = r["drivers"]["table"]
    drv = {}
    if len(tab):
        feats = [f for f in FEATURES if f in set(tab["feature"])]
        drv["labels"] = [FEATURES[f] for f in feats]
        for t in ("d_perf_pct", "d_ef_pct"):
            sub = tab[tab["target"] == t].set_index("feature")
            drv[t] = [None if f not in sub.index else round(float(sub.at[f, "rho"]), 3) for f in feats]
            drv[t + "_strong"] = [bool(sub.at[f, "strong"]) if f in sub.index else False for f in feats]
    ban = r.get("banister", {})
    return {
        "daily": {"date": _dates(d.index), "ctl": _col(d["ctl"]), "atl": _col(d["atl"]),
                  "tsb": _col(d["tsb"]), "tss": _col(d["tss"]), "ftp": _col(d["ftp"]),
                  "cp42": _col(d["cp42"]), "test": _col(d["eq20_test"]),
                  "banister": _col(ban["fitted"]) if ban.get("ok") else None},
        "weekly": {"week": _dates(W.index), "perf": _col(W["perf"]), "ef": _col(W["ef"]),
                   "ef8": _col(W["ef"].rolling(8, min_periods=3).mean()),
                   "hours": _col(W["hours"]), "zones": zone_h, "other": _col(unmeasured),
                   "sleep": _col(W["sleep_h"]), "rhr": _col(W["rhr"]), "hrv": _col(W["hrv"])},
        "pdc": pdc, "drivers": drv,
        "periods": _clean([{"start": str(p["start"]), "end": str(p["end"]),
                            "label": f"+{p['change_pct']}%"} for p in res["progress_periods"]]),
        "declines": _clean([{"start": str(p["start"]), "end": str(p["end"]),
                             "label": f"{p['change_pct']}%"} for p in res["decline_periods"]]),
        "illness": _clean([{"start": str(e["onset"]), "end": str(e["comeback_note"]),
                            "label": e.get("severity", "")}
                           for e in (res.get("illness") or {}).get("episodes", [])
                           if not str(e.get("severity", "")).startswith("unconfirmed")]),
        "ill_profile": _clean((res.get("illness") or {}).get("profile") or []),
        "sessions": _clean((res.get("buildup") or {}).get("sessions_by_group") or {}),
        "curve": _clean((res.get("recovery") or {}).get("after_hard_day") or []),
        "episodes": _clean([{"start": str(e["start"]), "end": str(e["end"]),
                             "verdict": e["verdict"]} for e in res["episodes"]]),
    }


def _table(rows, cols, fmt=None):
    fmt = fmt or {}
    if not rows:
        return "<p class='muted'>None found.</p>"
    h = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in cols)
    body = []
    for r in rows:
        tds = []
        for k, _ in cols:
            v = r.get(k)
            if isinstance(v, float) and v != v:
                v = None
            if k in fmt and v is not None:
                v = fmt[k](v)
            tds.append(f"<td>{html.escape('' if v is None else str(v))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<div class='tw'><table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def render(r: dict, res: dict) -> str:
    payload = build_payload(r, res)
    p, a, ld = res["power"], res["athlete"], res["load"]
    pct = lambda x: f"{x:.0%}"
    f1 = lambda x: f"{x:.1f}"
    f0 = lambda x: f"{x:.0f}"
    tiles = [
        ("FTP now", f"{p['ftp_now']:.0f} W", f"{p['wkg_now'] or '-'} W/kg"),
        ("Career-best FTP", f"{p['ftp_peak']:.0f} W", p["ftp_peak_date"]),
        ("CP / W' (90 d)", f"{p['cp_90d'] or 0:.0f} W", f"W' {p['wprime_90d_kj'] or '-'} kJ"),
        ("Load (CTL)", f"{ld['ctl_now']:.0f}", f"peak {ld['ctl_peak']:.0f}"),
        ("Form (TSB)", f"{ld['tsb_now']:+.0f}", f"fatigue ATL {ld['atl_now']:.0f}"),
        ("Rider type", p["phenotype"], ""),
    ]
    tiles_html = "".join(f"<div class='tile'><div class='tl'>{html.escape(t)}</div>"
                         f"<div class='tv'>{html.escape(str(v))}</div>"
                         f"<div class='ts'>{html.escape(str(s))}</div></div>" for t, v, s in tiles)
    periods_tbl = _table(res["progress_periods"], [
        ("start", "Start"), ("end", "End"), ("weeks", "Weeks"), ("perf_start_w", "From W"),
        ("perf_end_w", "To W"), ("change_pct", "Change %"), ("hours_wk", "h/wk"),
        ("hard_wk", "Hard/wk"), ("f_z12", "Z1-2"), ("f_z4", "Z4"), ("f_z5p", "Z5+"),
        ("what_was_different", "What was different")],
        {"hours_wk": f1, "hard_wk": f1, "f_z12": pct, "f_z4": pct, "f_z5p": pct})
    declines_tbl = _table(res["decline_periods"], [
        ("start", "Start"), ("end", "End"), ("weeks", "Weeks"), ("perf_start_w", "From W"),
        ("perf_end_w", "To W"), ("change_pct", "Change %"), ("hours_wk", "h/wk"),
        ("hard_wk", "Hard/wk"), ("what_was_different", "What was different")],
        {"hours_wk": f1, "hard_wk": f1})
    eps_tbl = _table(res["episodes"], [
        ("start", "Start"), ("weeks", "Weeks"), ("peak_atl", "Peak ATL"), ("min_tsb", "Min TSB"),
        ("max_acwr", "Max ACWR"), ("hours_wk", "h/wk"), ("load_flags", "Load flags"),
        ("recovery_flags", "Recovery flags"), ("change_after_pct", "Perf after %"),
        ("verdict", "Verdict")],
        {"peak_atl": f0, "min_tsb": f0, "max_acwr": f1, "hours_wk": f1, "change_after_pct": f1})
    years_tbl = _table(res["years"], [
        ("year", "Year"), ("hours", "Hours"), ("tss", "TSS"), ("avg_ctl", "Avg CTL"),
        ("peak_ctl", "Peak CTL"), ("hard_per_wk", "Hard/wk"), ("z12_share", "Z1-2"),
        ("z4_share", "Z4"), ("z5p_share", "Z5+"), ("hr_z12_share", "Z1-2 (HR)"),
        ("indoor_share", "Indoor"),
        ("best_perf", "Best tested W")],
        {"hours": f0, "tss": f0, "avg_ctl": f0, "peak_ctl": f0, "hard_per_wk": f1,
         "z12_share": pct, "hr_z12_share": pct, "z4_share": pct, "z5p_share": pct, "indoor_share": pct, "best_perf": f0})
    rec = res.get("recovery") or {}
    f2 = lambda x: f"{x:+.2f}"
    dose_tbl = _table(rec.get("dose_next_morning") or [], [
        ("bin", "Day's load (TSS)"), ("days", "Days"), ("hrv_z", "HRV z"), ("rhr_z", "Resting HR z"),
        ("sleep_h", "Sleep h"), ("sleep_score", "Sleep score"), ("readiness", "Readiness")],
        {"hrv_z": f2, "rhr_z": f2, "sleep_h": lambda x: f"{x:.2f}", "sleep_score": f0, "readiness": f0})
    ready_tbl = _table(rec.get("readiness_vs_performance") or [], [
        ("bin", "Morning readiness"), ("test_vs_90d_best_n", "Hard days"),
        ("test_vs_90d_best", "Power vs 90-d best"), ("ef_vs_6wk_n", "Steady rides"),
        ("ef_vs_6wk", "EF vs baseline")], {"test_vs_90d_best": pct, "ef_vs_6wk": pct})
    il = res.get("illness") or {}
    ill_table = _table(il.get("episodes") or [], [
        ("onset", "Onset"), ("comeback_note", "Last note"), ("days_off", "Days off"),
        ("rhr_peak_delta", "RHR peak (bpm)"), ("hrv_min_z", "HRV low (z)"), ("severity", "Severity"),
        ("perf_before", "Power before"), ("perf_after_6wk", "Best within 6 wk after"),
        ("days_to_regain_97pct", "Days to regain")],
        {"rhr_peak_delta": lambda x: f"{x:+.0f}", "hrv_min_z": f1, "perf_before": f0,
         "perf_after_6wk": f0, "days_to_regain_97pct": f0})
    ill_signals = _table(il.get("pre_signals") or [], [
        ("label", "2 weeks before onset"), ("before_illness", "Before"), ("normal", "Normal"),
        ("avg_percentile", "Percentile"), ("signal", "Verdict")],
        {"before_illness": lambda x: f"{x:.2f}", "normal": lambda x: f"{x:.2f}", "avg_percentile": f0})
    bu = res.get("buildup") or {}
    pk_cols = [("start", "4 weeks from"), ("hours_wk", "h/wk"), ("ctl_end", "CTL"), ("ramp_wk", "Ramp/wk"),
               ("best_test", "Best test W"), ("rest_days_wk", "Rest days"), ("recovery", "Rec"),
               ("endurance", "End"), ("long_endurance", "Long"), ("tempo", "Tempo"),
               ("sweetspot", "SS"), ("threshold", "Thr"), ("vo2max", "VO2"), ("test", "Test")]
    f2s = lambda x: f"{x:.2f}"
    peak_tbl = (f"<p class='muted'>The 52 weeks before your best tested power ({bu.get('peak_value')} W, "
                f"week of {bu.get('peak_week')}). Sessions per week by type.</p>" +
                _table(bu.get("peak_blocks") or [], pk_cols,
                       {"hours_wk": f1, "ctl_end": f0, "ramp_wk": lambda x: f"{x:+.1f}", "best_test": f0,
                        "rest_days_wk": f1, **{k: f2s for k in ("recovery", "endurance", "long_endurance",
                                                                 "tempo", "sweetspot", "threshold", "vo2max", "test")}}))
    ramp_tbl = _table(bu.get("ramp_table") or [], [
        ("ctl_ramp_per_week", "CTL ramp / week"), ("weeks", "Weeks"), ("illness_next_6wk", "Illness next 6 wk"),
        ("power_change_pct", "Power change %"), ("ef_change_pct", "Efficiency change %"), ("rhr_z_after", "RHR after (z)")],
        {"illness_next_6wk": pct, "power_change_pct": f1, "ef_change_pct": f1, "rhr_z_after": lambda x: f"{x:+.2f}"})
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in res["notes"])
    ban = res["banister"]
    ban_txt = (f"Fitness time-constant ≈ {ban['tau_fitness']:.0f} d, fatigue ≈ {ban['tau_fatigue']:.0f} d; "
               f"a training block pays off after ≈ {ban.get('days_to_positive_effect', 0):.0f} d and "
               f"peaks after ≈ {ban.get('days_to_peak_after_load', 0):.0f} d (R² {ban['r2']:.2f}, "
               f"n = {ban['n']}).") if ban.get("ok") else "Not enough tested days to fit."
    from .plan_out import html_section
    plan_html = html_section(r.get("plan"))
    title = f"Training profile – {a.get('name') or 'rider'}"
    return TEMPLATE.format(
        title=html.escape(title), generated=html.escape(res["generated"]), plotly=PLOTLY,
        tiles=tiles_html, plan=plan_html, periods=periods_tbl, declines=declines_tbl, episodes=eps_tbl,
        years=years_tbl, dose=dose_tbl, readiness=ready_tbl, ill_table=ill_table, ill_signals=ill_signals, peak_tbl=peak_tbl, ramp_tbl=ramp_tbl, notes=notes or "<li>None.</li>", banister=html.escape(ban_txt),
        payload=json.dumps(payload, separators=(",", ":")))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="{plotly}"></script>
<style>
:root {{ color-scheme: light; --bg:#f6f6f4; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984;
  --grid:#e6e5e0; --line:#d6d5cf; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s7:#6250d6;
  --good:#2b7d74; --bad:#9a0b33; --z1:#cde2fb; --z2:#9ec5f4; --z3:#6da7ec; --z4:#3987e5; --z5:#256abf; --z6:#184f95; --z7:#0d366b; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --bg:#121211; --surface:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --grid:#2c2c2a; --line:#3a3a37; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --s4:#c98500; --s7:#9085e9; --good:#6fd6c0; --bad:#e8505f; --z1:#104281; --z2:#184f95; --z3:#1c5cab; --z4:#256abf; --z5:#3987e5; --z6:#6da7ec; --z7:#b7d3f6; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg:#121211; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86;
  --grid:#2c2c2a; --line:#3a3a37; --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s7:#9085e9; --good:#6fd6c0; --bad:#e8505f;
  --z1:#104281; --z2:#184f95; --z3:#1c5cab; --z4:#256abf; --z5:#3987e5; --z6:#6da7ec; --z7:#b7d3f6; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px }}
h1 {{ font-size:26px; margin:0 0 4px }} h2 {{ font-size:19px; margin:36px 0 6px }}
.muted, .sub {{ color:var(--ink2) }} .sub {{ margin:0 0 18px; font-size:13px }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.tl {{ font-size:12px; color:var(--ink2) }} .tv {{ font-size:22px; font-weight:650; margin-top:2px }} .ts {{ font-size:12px; color:var(--muted) }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px; margin-top:10px }}
.chart {{ width:100%; height:380px }} .chart.sm {{ height:300px }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }} @media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr }} }}
.tw {{ overflow-x:auto }} table {{ border-collapse:collapse; width:100%; font-size:13px }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--grid); vertical-align:top }}
th {{ color:var(--ink2); font-weight:600; white-space:nowrap }}
p.expl {{ color:var(--ink2); font-size:13.5px; margin:4px 0 0 }}
</style></head>
<body><main>
<h1>{title}</h1><p class="sub">Generated {generated} by goodpush · regenerate with <code>goodpush run</code></p>
<div class="tiles">{tiles}</div>
{plan}

<h2>Load, fatigue and form</h2>
<p class="expl">CTL (42-day load) is long-term load, ATL (7-day load) is fatigue, TSB = CTL − ATL is form. Shaded: overload blocks — red = looked like too much, green = followed by improvement.</p>
<div class="card"><div id="pmc" class="chart"></div></div>

<h2>Power over time</h2>
<p class="expl">Purple ticks along the top: illnesses (from your ride notes + resting HR). Dots: 20-min-equivalent power on hard days (tests). Line: best test in the previous 6 weeks (the performance marker used below). Auto-FTP is the 90-day critical-power estimate. Shaded green/red: biggest progress and decline periods.</p>
<div class="card"><div id="perf" class="chart"></div></div>

<div class="grid2">
<div><h2>Weekly volume by power zone</h2><div class="card"><div id="zones" class="chart sm"></div></div></div>
<div><h2>Power-duration curve</h2><div class="card"><div id="pdc" class="chart sm"></div></div></div>
</div>

<h2>Aerobic efficiency</h2>
<p class="expl">Efficiency factor = normalized power ÷ average heart rate on steady endurance rides (IF &lt; 0.8, 45 min+). Rising EF means more power for the same heartbeat — aerobic fitness that doesn't require all-out tests.</p>
<div class="card"><div id="ef" class="chart sm"></div></div>

<h2>What drove progress</h2>
<p class="expl">Rank correlation (Spearman ρ) between each feature of the previous 8 weeks and the change in performance over those 8 weeks. Solid bars pass a rough significance test (effective n = weeks ÷ 8); faint bars don't. Correlation is not causation.</p>
<div class="card"><div id="drivers" class="chart" style="height:520px"></div></div>
<h2>Biggest gains</h2><div class="card">{periods}</div>
<h2>Biggest declines</h2><div class="card">{declines}</div>

<h2>When it was too much</h2>
<p class="expl">Overload blocks = weeks with ≥2 days of acute:chronic ratio &gt;1.5, deep negative form, fast CTL ramp (&gt;8/wk) or monotonous high strain. Verdict looks at the 6 weeks after: performance change, a forced break (&lt;40% usual volume), and recovery markers (resting HR, HRV, sleep, efficiency, fatigue words in ride descriptions).</p>
<div class="card">{episodes}</div>
<p class="expl"><b>Your fitness–fatigue response:</b> {banister}</p>

<div id="wellness-wrap"><h2>Sleep and recovery</h2>
<p class="expl">Weekly averages from Garmin. Resting HR drifting down with rising fitness is a good sign; HRV below your band and resting HR above it are the classic signs of accumulated fatigue or illness.</p>
<div class="grid2"><div class="card"><div id="rhr" class="chart sm"></div></div><div class="card"><div id="hrv" class="chart sm"></div></div></div>
<div class="grid2"><div class="card"><div id="sleep" class="chart sm"></div></div><div class="card"><div id="curve" class="chart sm"></div></div></div>
<h2>How a training day shows up the next morning</h2>
<p class="expl">Average next-morning values by the previous day's training load. z = standard deviations from your own 28-day baseline.</p>
<div class="card">{dose}</div>
<h2>Morning readiness vs performance that day</h2>
<p class="expl">Hard-effort days: best 20-min-equivalent power as a share of your previous-90-day best. Steady rides: efficiency factor vs your 6-week baseline.</p>
<div class="card">{readiness}</div></div>

<div id="ill-wrap"><h2>Illness</h2>
<p class="expl">Built from sickness notes in your ride descriptions. Notes are usually written on the comeback ride, so onset is reconstructed from the training gap and the resting-HR rise before the note. Left: average resting HR and HRV around onset (day 0) across all illnesses.</p>
<div class="grid2"><div class="card"><div id="illprof" class="chart sm"></div></div><div class="card">{ill_signals}</div></div>
<div class="card">{ill_table}</div></div>

<h2>What you trained: sessions per week by type</h2>
<p class="expl">Each ride is classified from its power stream (sustained efforts vs FTP, confirmed by heart rate; app workout names as a check). Bars show sessions per week, averaged over 4-week blocks.</p>
<div class="card"><div id="sessions" class="chart"></div></div>
<h2>Your peak build</h2><div class="card">{peak_tbl}</div>
<h2>How fast can you ramp?</h2>
<p class="expl">CTL change per week (averaged over 4 weeks) against what followed: illness in the next 6 weeks, change in tested power, change in aerobic efficiency. Weeks right after an illness are excluded.</p>
<div class="card">{ramp_tbl}</div>

<h2>Year by year</h2><div class="card">{years}</div>
<h2>Data-quality notes</h2><ul class="muted">{notes}</ul>
</main>
<script>
const D = {payload};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function base(extra) {{
  return Object.assign({{
    paper_bgcolor: css('--surface'), plot_bgcolor: css('--surface'),
    font: {{color: css('--ink2'), family: 'system-ui, sans-serif', size: 12}},
    margin: {{l: 52, r: 16, t: 10, b: 40}}, hovermode: 'x unified',
    legend: {{orientation: 'h', y: 1.08, x: 0}},
    xaxis: {{gridcolor: css('--grid'), linecolor: css('--line'), zeroline: false}},
    yaxis: {{gridcolor: css('--grid'), linecolor: css('--line'), zeroline: false}},
    hoverlabel: {{bgcolor: css('--surface'), bordercolor: css('--line'), font: {{color: css('--ink')}}}}
  }}, extra || {{}});
}}
const cfg = {{displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']}};
function bands(list, colorFn) {{
  return (list || []).map(p => ({{type: 'rect', xref: 'x', yref: 'paper', x0: p.start, x1: p.end, y0: 0, y1: 1,
    fillcolor: colorFn(p), opacity: 0.13, line: {{width: 0}}, layer: 'below'}}));
}}
function draw() {{
  const d = D.daily, w = D.weekly;
  const line = (x, y, name, c, extra) => Object.assign({{x, y, name, type: 'scatter', mode: 'lines', line: {{color: c, width: 2}}}}, extra || {{}});
  Plotly.react('pmc', [
    line(d.date, d.ctl, 'Load (CTL)', css('--s1')),
    line(d.date, d.atl, 'Fatigue (ATL)', css('--s2'), {{line: {{color: css('--s2'), width: 1.2}}}}),
    line(d.date, d.tsb, 'Form (TSB)', css('--s3'), {{line: {{color: css('--s3'), width: 1.2}}}})
  ], base({{shapes: bands(D.episodes, e => e.verdict === 'too much' ? css('--bad') : (e.verdict.startsWith('productive') ? css('--good') : css('--muted'))),
           yaxis: {{title: 'TSS / day', gridcolor: css('--grid'), zeroline: true, zerolinecolor: css('--line')}}}}), cfg);
  const perf = [
    {{x: d.date, y: d.test, name: 'Test (20-min eq.)', type: 'scatter', mode: 'markers', marker: {{color: css('--s1'), size: 6, opacity: 0.55}}}},
    line(w.week, w.perf, 'Performance marker (6-wk best)', css('--s1'), {{line: {{color: css('--s1'), width: 2, shape: 'hv'}}}}),
    line(d.date, d.ftp, 'Auto-FTP (90-d CP)', css('--s2'), {{line: {{color: css('--s2'), width: 1.5, dash: 'dot'}}}})
  ];
  if (d.banister) perf.push(line(d.date, d.banister, 'Banister model fit', css('--s7'), {{line: {{color: css('--s7'), width: 1.2}}}}));
  Plotly.react('perf', perf, base({{shapes: bands(D.periods, () => css('--good')).concat(bands(D.declines, () => css('--bad'))).concat(bands(D.illness, () => css('--s7')).map(b => Object.assign(b, {{y0: 0.92, y1: 1, opacity: 0.6}}))),
    yaxis: {{title: 'Watts', gridcolor: css('--grid')}}}}), cfg);
  const zn = ['Z1 recovery', 'Z2 endurance', 'Z3 tempo', 'Z4 threshold', 'Z5 VO2max', 'Z6 anaerobic', 'Z7 sprint'];
  const zt = zn.map((n, i) => ({{x: w.week, y: w.zones['Z' + (i + 1)], name: n, type: 'bar', marker: {{color: css('--z' + (i + 1))}}, hovertemplate: '%{{y:.1f}} h'}}));
  zt.push({{x: w.week, y: w.other, name: 'No power data', type: 'bar', marker: {{color: css('--grid')}}, hovertemplate: '%{{y:.1f}} h'}});
  Plotly.react('zones', zt, base({{barmode: 'stack', bargap: 0.05, yaxis: {{title: 'Hours / week', gridcolor: css('--grid')}}, legend: {{orientation: 'h', y: -0.2}}}}), cfg);
  const p = D.pdc, pd = p.durations;
  const lbl = s => s < 60 ? s + 's' : (s < 3600 ? (s / 60) + 'min' : (s / 3600) + 'h');
  const pc = [line(pd, p.all, 'All-time best', css('--s1')), line(pd, p.last90, 'Last 90 days', css('--s2'))];
  if (p.best_year_curve) pc.push(line(pd, p.best_year_curve, 'Best year (' + p.best_year + ')', css('--s3')));
  Plotly.react('pdc', pc, base({{hovermode: 'x unified', xaxis: {{type: 'log', tickvals: [1, 5, 15, 60, 300, 1200, 3600, 7200], ticktext: ['1s', '5s', '15s', '1min', '5min', '20min', '1h', '2h'], gridcolor: css('--grid')}},
    yaxis: {{title: 'Watts', gridcolor: css('--grid')}}}}), cfg);
  Plotly.react('ef', [
    {{x: w.week, y: w.ef, name: 'Weekly EF', type: 'scatter', mode: 'markers', marker: {{color: css('--s1'), size: 6, opacity: 0.5}}}},
    line(w.week, w.ef8, '8-week average', css('--s1'))
  ], base({{yaxis: {{title: 'NP / HR', gridcolor: css('--grid')}}}}), cfg);
  if (D.drivers.labels) {{
    const g = D.drivers, op = (s) => s.map(v => v ? 1 : 0.35);
    Plotly.react('drivers', [
      {{y: g.labels, x: g.d_perf_pct, name: 'Δ tested power', type: 'bar', orientation: 'h', marker: {{color: css('--s1'), opacity: op(g.d_perf_pct_strong)}}, hovertemplate: 'ρ = %{{x:.2f}}'}},
      {{y: g.labels, x: g.d_ef_pct, name: 'Δ aerobic efficiency', type: 'bar', orientation: 'h', marker: {{color: css('--s2'), opacity: op(g.d_ef_pct_strong)}}, hovertemplate: 'ρ = %{{x:.2f}}'}}
    ], base({{barmode: 'group', hovermode: 'y unified', margin: {{l: 260, r: 16, t: 10, b: 40}}, yaxis: {{autorange: 'reversed', gridcolor: css('--grid')}},
      xaxis: {{title: 'Spearman ρ', range: [-0.7, 0.7], gridcolor: css('--grid'), zeroline: true, zerolinecolor: css('--ink2')}}}}), cfg);
  }}
  const ip = D.ill_profile || [];
  if (ip.length) Plotly.react('illprof', [
    line(ip.map(r => r.day), ip.map(r => r.rhr_d), 'Resting HR vs baseline (bpm)', css('--s2')),
    line(ip.map(r => r.day), ip.map(r => r.hrv_z), 'HRV vs baseline (z)', css('--s3'))
  ], base({{hovermode: 'x unified', xaxis: {{title: 'Days from illness onset', gridcolor: css('--grid'), zeroline: true, zerolinecolor: css('--ink2')}},
    yaxis: {{title: 'Change vs your baseline', gridcolor: css('--grid'), zeroline: true, zerolinecolor: css('--line')}}}}), cfg);
  else document.getElementById('ill-wrap').style.display = 'none';
  const S = D.sessions || {{}};
  if (S.block) {{
    const grp = [['easy', '--z2'], ['long', '--s3'], ['tempo/sweet spot', '--s4'], ['threshold', '--s2'], ['vo2max/anaerobic', '--bad'], ['race/test', '--s7']];
    Plotly.react('sessions', grp.filter(g => S[g[0]]).map(g => ({{x: S.block, y: S[g[0]], name: g[0], type: 'bar', marker: {{color: css(g[1])}}, hovertemplate: '%{{y:.2f}}/wk'}})),
      base({{barmode: 'stack', bargap: 0.08, yaxis: {{title: 'Sessions / week', gridcolor: css('--grid')}}}}), cfg);
  }}
  const has = a => a && a.some(v => v !== null);
  if (has(w.sleep) || has(w.rhr) || has(w.hrv)) {{
    Plotly.react('sleep', [line(w.week, w.sleep, 'Sleep (h/night)', css('--s7'))], base({{yaxis: {{title: 'Sleep, hours / night', gridcolor: css('--grid')}}}}), cfg);
    Plotly.react('rhr', [line(w.week, w.rhr, 'Resting HR', css('--s2'))], base({{yaxis: {{title: 'Resting HR, bpm', gridcolor: css('--grid')}}}}), cfg);
    const firstHrv = w.hrv.findIndex(v => v !== null);
    Plotly.react('hrv', [line(w.week, w.hrv, 'Nightly HRV (weekly mean)', css('--s3'))], base({{yaxis: {{title: 'HRV, ms', gridcolor: css('--grid')}},
      xaxis: {{gridcolor: css('--grid'), range: firstHrv >= 0 ? [w.week[firstHrv], w.week[w.week.length - 1]] : undefined}}}}), cfg);
    const c = D.curve || [];
    if (c.length) Plotly.react('curve', [
      line(c.map(r => r.days_after), c.map(r => r.hrv_z), 'HRV z', css('--s3'), {{mode: 'lines+markers', marker: {{size: 8}}}}),
      line(c.map(r => r.days_after), c.map(r => r.rhr_z), 'Resting HR z', css('--s2'), {{mode: 'lines+markers', marker: {{size: 8}}}})
    ], base({{hovermode: 'x unified', xaxis: {{title: 'Days after a 150+ TSS day', dtick: 1, gridcolor: css('--grid')}},
      yaxis: {{title: 'vs your baseline (z)', gridcolor: css('--grid'), zeroline: true, zerolinecolor: css('--ink2')}}}}), cfg);
    }} else document.getElementById('wellness-wrap').style.display = 'none';
}}
if (window.Plotly) {{ draw(); window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw); }}
else document.querySelectorAll('.chart').forEach(e => e.innerHTML = '<p class="muted">Charts need an internet connection (Plotly loads from cdn.plot.ly). Tables below still work.</p>');
</script></body></html>
"""
