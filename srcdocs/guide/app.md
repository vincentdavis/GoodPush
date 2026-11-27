# The app, page by page

Start it from the installer's icon, or with `goodpush app` (it opens <http://127.0.0.1:8765> in your browser). The
💡 next to a title explains that part of the page; hover or tap it.

## Today

This morning's readiness: sleep, HRV and resting heart rate against your normal, and your form, as green (good to
go), amber (go easier) or red (rest or easy only). Below it today's session with watt targets and its shape,
**How do you feel?** (Normal / A bit off / Sick, which adjusts the plan), **Mark done** / **Skipped**, the workout
file, and **▶ Ride it now** to ride it on the trainer. **Update now** pulls new data from Garmin. A new user also
finds the guide to bringing in their whole history here.

## Calendar

Every week since your first ride, one row per week: what was planned (dashed) and what you actually rode (click a
ride to open it). Above it a chart of daily and weekly load and the plan's weekly targets; scroll either and the
other follows.

* **Move a session**: drag it onto another day. Click it to change its length, delete it, or go back to the plan's
  version. Days you changed (✎) stay as you set them.
* **Link a ride to a session**: drag the ride onto it (today or earlier). The session is marked done and counts for
  the interval progression. On a phone, hold the ride a moment, then drag. × unlinks.
* **Days away**: days you can't ride; nothing is planned on them.
* **Plan update**: when the plan wants to change a day you already had, it asks first, showing what you have and
  what it proposes, with Accept / Reject.

## Fitness

What you did: training load (CTL), fatigue (ATL) and form (TSB) over time, FTP over the years, your power curve,
alpha1 trends and your thresholds next to Garmin's estimates. **Rides** lists every ride. **Sources** lists the
papers and books each method comes from.

## A ride

Power, heart rate, elevation, cadence and (when the ride has it) alpha1 on one time axis, and the route on a map.
Drag across a chart to select a stretch: average and normalized power, IF, TSS, heart-rate drift, cadence, climbing
and alpha1 for just that stretch. Click an effort in the table to select it. Stamina lines show how much of your
W' you had left through the ride {cite:p}`skiba2012,chorley2021,clark2019a`.

## Rider

What you can do now: your numbers today (CP, LT1 / LT2, MAP, W', sprint power, with their ranges when the model has
them), best efforts, power over time, your power curve over the seasons, the **physiological model**'s view of you,
and **what training does for you**: how each kind of riding has moved your numbers.

## Model

Can you trust the model behind the Rider page and the plan? Its status (version, what it's fitted to, when its
ranges were made), today's numbers with their uncertainty, how far off its forecasts have been, its predictions
next to what you rode and measured, and your fatigue. This is the page to look at when you
[compare GoodPush with what you know](../community/feedback.md#do-the-numbers-make-sense).

## Plan

The next two weeks, week targets, interval progression, **Phases & races**, "Record an FTP test", and the calendar
file. See [The training plan](plan.md).

## Recovery

How your body is doing:

* **Fitness drops**: every break long enough to cost fitness, with the weeks around it and an automatic cause
  (illness, overreaching, travel, life) and its evidence; correct it if it's wrong.
* **Your illness markers**: **Analyze my history** replays your past illnesses to find which morning signals
  (resting HR, HRV, sleep, stress, breathing rate, load spikes...) showed up in the week before you noticed, and how
  often they cried wolf. The chosen ones are checked every morning (Clear / Watch / Illness warning on Today).
* **Overreaching markers**, learned the same way from drops you labelled overreaching.
* HRV with your normal range, resting HR, sleep, illness history, and your training builds compared.

Illnesses come from words like "sick", "flu", "covid" or "cough" in your ride descriptions, and from illness events you add.

## Settings

Your connections (Garmin, Fitbit, Incyclist, your other computers), the plan's settings, your athlete details,
FTP, the model, the trainer, and **Rebuild everything** (after importing an export). Everything is saved in
`goodpush.toml` in your GoodPush folder.

## Help & feedback

**Beta** (top right): report a problem, tell us whether the numbers make sense, and links to this guide.

## Charts

Every chart over time zooms: Ctrl + wheel (or pinch) zooms, and so does dragging the ends of the bar under the
chart. Once zoomed, drag the chart (or Shift + wheel, or the bar) to move along. Double-click to see it all. Charts
that share a time axis zoom together and show the same day when you hover one. On the load chart and on a ride, where
a drag selects a stretch, Shift + drag moves along, double-click clears the stretch, and **All** / **Whole ride**
shows it all.

## On your phone

On a narrow screen the pages are in a bar at the bottom: **Today**, **Calendar**, **Fitness** and **Plan**, with
**Rider**, **Model**, **Recovery**, **Settings** and this guide under **More**.

`goodpush app --lan` prints a link with an access key (`http://192.168.x.x:8765/?key=...`). Open it once on your
phone (same Wi-Fi); the key is remembered, and you can add the page to your home screen. Keep the link private: it
opens your health data. Without `--lan` the app only answers on the computer itself. Windows asks once whether
Python may use the network: allow it on private networks only. The key lives in `app_key` in GoodPush's own folder
([where things are kept](privacy.md)); delete that file to make a new one.
