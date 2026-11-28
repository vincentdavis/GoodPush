Inputs
======

The model sees plain arrays (:class:`goodpush.physio.PhysioData`), built from goodpush's tables by
:class:`goodpush.models.physio.PhysioInputs`. Three kinds of data go in.

Training (known inputs, not modelled)
-------------------------------------

* **Daily power histograms**: hours per 10 W bin (0-1500 W, more in the top bin), summed over the day's rides;
  coasting (the 0 W bin) left out. Rides without power contribute their heart-rate zone seconds, placed at fixed
  fractions of that day's FTP (0.50, 0.60, 0.75, 0.88, 0.98, 1.08, 1.20 for zones 1-7).
* **Long-ride hours**: per day, the hours past 2 h in each ride, summed.

The training is taken as exactly known: power meter error and rides missing from the record are not modelled.

Ride best powers (observations of capacity)
-------------------------------------------

For every day with a ride, the best power over each of 12 durations: 5, 15, 30 s, 1, 3, 5, 10, 20, 40, 60, 90 and
120 min (only durations the ride lasted, only rides with power for more than 90 % of their time). Each day also
has a **ride kind**, the hardest of its rides: *easy*; *hard* (a threshold, VO2max, anaerobic, sweet-spot or tempo
workout, or a race); *test*. A race is a hard ride, not a test: its best powers come inside repeated
efforts, not from fresh steady ones. The kind matters for how likely a best power is all-out (see :doc:`likelihood`).

A best power whose ride shows it went all-out is **maxed**, whatever the day's kind: heart rate around the effort's
end (1 min: the 45 s after it, heart rate lagging a short effort; 3 to 20 min: the effort's last 15 to 60 s) came
within 10 bpm (1 min), 7 (3 min) or 5 (5 to 20 min) of your maximum heart rate then - the 99th percentile of your
rides' highest heart rates in the year around it, as it drifts with age and fitness - and an effort of 3 min or more
was steady (at most 5 % of its seconds without power). These are the measurements of the curve; FTP, CP and W' are
inferred from it (``metrics.effort_hr.EffortHeartRate``).

Most best powers are not all-out, so they are *censored* observations: they say capacity was at least this high.

**Power that is wrong** is set aside before the model sees it (``metrics.power_check.PowerCheck``):

* rides you mark (the Ride page's "Power is wrong", or goodpush.toml ``[ingest] bad_power``: ride ids, days or date
  ranges - a meter that read high for weeks) count as rides without power: their time and heart rate stay, their
  load comes from heart rate, and they give no best powers, threshold watts or steady stretches;
* a ride whose best of 3 to 20 min is more than 1.2 x your rides' 99th percentile at that length (the next-highest
  ride's, so a glitch never sets its own bar) while heart rate around its end stayed under 0.95 x threshold heart
  rate loses its best powers - minutes far above your usual at an easy heart rate are a broken meter, not an
  effort. Its time in each power zone stays. ``[ingest] keep_power`` lists rides to leave alone.

What gets past both is the likelihood's bad-reading share (:doc:`likelihood`).

Threshold readings (observations of thresholds)
-----------------------------------------------

Each reading measures one quantity (LT1, CP, MAP or 1-h power) through a **source** with its own bias and error,
both fitted (priors in :doc:`priors`):

.. list-table::
   :header-rows: 1
   :widths: 18 10 72

   * - Source
     - Measures
     - Where it comes from
   * - ``my_ftp``
     - CP
     - the FTP history you entered (``[ftp] history``)
   * - ``alpha1_lt1``
     - LT1
     - power where DFA alpha1 crosses 0.75 on ordinary rides (HRVT1, :cite:t:`rogers2021a`)
   * - ``ramp_lt1``, ``ramp_lt2``
     - LT1, CP
     - from ramp-to-failure tests: alpha1 = 0.75 and 0.5 :cite:p:`rogers2021a,rogers2021b` - upper bounds (:doc:`likelihood`). The ramp's top is no
       reading: the ride's best powers are efforts like any other

Garmin's own estimates (its FTP, its VO2max) are not readings: the analysis stands on your rides, tests and
entries alone. Garmin's FTP is shown for comparison only.

FTP-type readings are taken to measure CP (a threshold), not an all-out hour. Threshold heart rate is an
*input* (your setting or goodpush's estimate), never a fitted parameter.

**Steady stretches** are not readings: each steady ride (>= 30 min, NP / average power <= 1.15, power recorded
>= 90 %; its average power and heart rate) and each steady interval rep (>= 5 min, power within 12 %; heart rate
from its settled second half) goes in as *power at its heart rate*, with heart rate as a share of threshold heart
rate between 0.55 and 1.1. The model reads them through your own heart-rate curve (:doc:`likelihood`). Until
2026-11-09 each was scaled to threshold heart rate along a straight line from resting heart rate; that read 0 %
low from easy rides and about 11 % low from tempo rides, so a week of different riding looked like a change in
CP.

Recovery markers (sleep, HRV, resting heart rate, bedtime) are built but **not** used by the model; why is in
:doc:`limits`. They act through the plan's day-to-day readiness instead.
