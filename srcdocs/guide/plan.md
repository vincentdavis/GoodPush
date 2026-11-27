# The training plan

GoodPush plans every day of your **phases**, from today to the end of the last one, and re-plans as your rides and
mornings come in. Turn it on in **Settings > Plan** (`[plan] enabled = true`). The app shows it on Today, Plan and
the Calendar; it is also written to `reports/plan/`:

| File | What it is |
|---|---|
| `plan.md` | today's readiness, the sessions with watt targets, the week targets |
| `plan.ics` | a calendar file: import it into Google / Apple / Outlook calendar (re-importing updates the same events) |
| `workouts/gp_*.fit`, `gp_*.zwo` | structured workouts for a Garmin watch or bike computer, and for Zwift / Incyclist |
| `plan.json` | the same plan for other tools |

How to ride it (watch, Incyclist, GoodPush's own trainer bar): [Riding the plan](riding.md).

## Phases and races

**Plan > Phases & races** (stored in `data/plans.json`). A phase has a type, a start and a length; phases can't
overlap; gaps are allowed and nothing is planned in them.

| Phase | What it plans |
|---|---|
| Base | endurance with sweet spot and tempo, no VO2max |
| Build | threshold every week, VO2max / sweet spot alternating by block |
| Specialty | threshold + VO2max, tapers into your races |
| Maintenance | one quality session a week, fitness held |
| Recovery / downtime | easy riding only, about half your hours |

Each phase has a cadence: loading weeks before a recovery week (3 = 3:1, 2 = 2:1, 0 = none) and optionally its own
weekly hours.

Races have a date, an A / B / C priority, a type (road, crit, tt, mtb, gravel, cx, hillclimb, fondo) and an optional
series; paste several at once, one per line: `2026-11-29, B, crit, Round 1`. A race day gets the race, the day
before openers, the day after an easy spin. An A race gets a taper of about two weeks (the week before 70 %, its
week 50 % of the load, intensity kept; {cite:alp}`mujika2003`), a B race a lighter week; no key session 1-2 days before an A race and no
tests in race weeks.

**Phases suggested from your races.** When your A races need weeks that none of your phases cover, the Plan page
shows "Suggested from your races", planned back from each A race the way coaches periodise {cite:p}`friel2018`: Specialty for the 8
weeks up to the race, Build before it, Base before that, recovery weeks after the last A race. Your own phases
always win. "Add these phases" adds them; "Not now" hides them until your races or phases change.

## Your time

**Settings > Plan**: hours per weekday (Mon..Sun). The longest day gets the long ride, a 0 day is rest, no session is
longer than its day. `hours_per_week` caps the week. The plan builds up to it gradually: a week at most 10 % above
your average week of the last 4, one ride at most 10 % above your longest recent ride.

## Settings

`[plan]` in `goodpush.toml`; the ones marked * are also in the app (Settings > Plan).

| Setting | Default | What it does |
|---|---|---|
| `enabled` | `false` | Make a plan at all. |
| `hours_per_week` * | `8.0` | Most hours in a week; a phase can set its own, recovery weeks get less. |
| `plan_starting_hours` * | `0` | Start the build-up from at least this many hours a week, if you can already ride more than your last 4 weeks show; `0` = your recent average. After a long break set it back to `0`. |
| `availability` * | `[]` | Hours you can ride each weekday, Mon..Sun; 0 = no riding. `[]` = use the day settings below. |
| `rest_days` * | `["mon"]` | Days off (with `availability`: the 0 days). |
| `key_days` * | `["tue", "thu"]` | The two days for hard sessions; tests go on the first. |
| `long_day` * | `"sat"` | The long ride, only used without `availability`. |
| `recovery_days` * | `["fri"]` | Easy spin days. |
| `intensity` * | `"model"` | What the hard days are: `"model"` (the physiological model picks each key day's session for the best forecast on your race mornings; pyramidal until it's fitted), `"polarized"` (easy below LT1 plus hard work above LT2 only; {cite:alp}`seiler2006,stoggl2014`) or `"pyramidal"` (tempo and sweet spot in base, threshold and VO2max in build and specialty). Every style has the same number of hard days: 2 a week in base, build and specialty, 1 in maintenance and recovery weeks. |
| `mix_draws` | `10` | With `intensity = "model"`: how many of the model's possible versions of you the choice is averaged over (robust to what your data barely pins down); 0 = the best fit alone (faster). |
| `load_model` * | `"ctl"` | How much to ride each week. `"physio"`: the physiological model picks each week's hours and hard work for the best forecast on your race mornings, learns from your data how much fatigue costs you, and reads your sleep / HRV / resting HR as signs of fatigue (needs about 4 months of rides). `"ctl"`: fixed rules (`min_ramp` / `max_ramp`, recovery weeks at 70 %, tapers by form). |
| `min_ramp`, `max_ramp` | `1.5`, `2.5` | `"ctl"` only: how many CTL points a loading week adds, at least / at most. |
| `test_every_weeks` | `6` | A 20-min FTP test when the last one is older than this. |
| `ramp_every_weeks` | `6` | A ramp test with alphaHRV for LT1 / LT2 (`0` = never). |
| `ftp` * | `0` | `0` = the FTP from your data; a number overrides it for the plan. |

## How it decides

* **Weekly load.** With `load_model = "physio"` the model's forecast decides it. With `"ctl"`, one optimisation over
  all weeks of your phases sets each week's load: base and build weeks raise CTL between `min_ramp` and `max_ramp`
  a week, recovery weeks are about 70 % of a steady week, everything fits your hours, and races set the taper (form
  +10..+25 on an A-race morning, at least 0 for a B race).
* **Daily sessions.** A second optimisation picks each day's session (rest, recovery spin, endurance of any length,
  or the phase's quality types) to hit the week's load within your hours. Two neighbouring days' hardness add to at
  most 3 (VO2 / threshold = 2, tempo / sweet spot = 1, easy = 0), so no two hard days in a row; at least one rest
  day in a full week; form (TSB) stays above -30 and a hard day needs at least -20. It sticks to what you accepted
  unless a rule or your readiness needs a change.
* **Interval progression.** Each type (threshold, VO2max, sweet spot, tempo, anaerobic, a durability finish on long
  rides, {cite:alp}`maunder2021`) has a ladder of formats and levels (4x5 → ... → 2x25 min threshold; 30/15s, step-down and over-under
  VO2max sessions...). Each session's note says what it trains. A session **on target** (at least 90 % of its work
  time on target) moves you a step up; **partly** or **above target** repeats the step; **fell short** goes one step
  back. Two weeks without a type's intervals take its level a step down, then one more every 10 days. Set a level
  yourself on the Plan page (Interval progression).
* **Push week.** The last loading week of a base or build block aims to end at form about -20: tired enough to want
  the recovery week, well short of overreaching {cite:p}`aubry2014,meeusen2013`.
* **Adapting.** A missed key session is made up on a free easy day later that week; after an unplanned hard ride the
  next key session moves back. If your threshold and sweet-spot sessions keep landing 3 % or more above target, it
  proposes FTP +3 % (well short twice: -3 %).
* **You decide.** Once a day is planned, a real change to it is a **proposal** you accept or reject on the Calendar.
  Until you accept, the plan, watch files and Incyclist keep what you had. Small volume tweaks, new days and
  readiness changes go straight in.
* **Morning readiness** (from Garmin plus your load; {cite:alp}`plews2013,fullagar2015`):
  * **red**: resting HR +3 bpm two mornings running, HRV well below normal, or under 5.5 h sleep. Today and tomorrow
    become rest or easy spins.
  * **amber**: low HRV, short sleep, a 3-night sleep debt, form below -30, or a recent sickness note in a ride. No hard
    session today (tempo / sweet spot at most) and a test is postponed. After a run of 4 of the last 7 mornings not
    green, tomorrow stays easy too.

## Targets in your watts

With `load_model = "physio"`, sessions are designed in textbook % of FTP and then mapped through your own thresholds
on the session's day: 75 % → your LT1, 100 % → CP, 120 % → MAP (5-min power), 150 % → 80 % of an all-out minute,
250 % → 90 % of your sprint, straight lines between. Endurance stays below your LT1 wherever it is, VO2max work sits
between your CP and MAP. Later sessions use the model's forecast for their day. The Plan page shows today's anchors.

## The physiological model

The model (Rider and Model pages, and the plan with `load_model = "physio"`) is fitted to your whole history: its
best fit takes about a minute, and the app keeps it current after each re-plan in about a second.

**Its ranges** (the 5-95 % bands) come from a longer run that draws thousands of possible versions of you (NUTS
sampling, {cite:alp}`hoffman2014`). Start it once from a terminal; it saves itself as it goes, so you can stop it (Ctrl-C or close the window)
and run the same command again to carry on:

```
goodpush physio --nuts --samples 2500 --ess 400 --max-minutes 540
```

It usually finishes within an hour, 9 hours at most. After that the app refreshes the ranges by itself about once a
week, in the background at low priority (`[physio] ranges_every_days`, 0 = never; Settings > Physiological model).
A full run is only needed again when the model's version changes, or when its first day moves (below).

**Only your recent rides.** The model always fits on rides only (runs, strength and the rest never reach it), from
your first ride with power. To leave out older years, set Settings > Physiological model > *Fit on the last (days)*
(`[physio] since_days`, e.g. 1825 = the last 5 years; 0 = all your data), or try it once with
`goodpush physio --since-days 1825`. The first day goes back to the start of its quarter (1 Jan, 1 Apr, 1 Jul,
1 Oct) and moves on four times a year rather than every day; each move is a full fit, and the ranges need the full
run above again.

Everything the model finds is about one person's data and observational: treat its training responses as hypotheses.
How it works: [the model pages](../model/overview.rst).
