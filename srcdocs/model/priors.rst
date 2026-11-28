Priors
======

Every parameter has an independent **Normal prior on its unconstrained scale** (log for positive quantities,
logit for shares), so the log posterior can be minimised and sampled directly on one flat vector
(:class:`goodpush.physio.model.ParamSpec`, values in :meth:`goodpush.physio.PhysioModel.blocks`). Below: the
prior's median on the natural scale and its 90 % range.

States
------

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - State
     - Start (first day), median
     - 90 % range
     - Detrained floor, median (90 %)
   * - MAP
     - the median of your first year's 8-week best 5-min powers (290 W without)
     - x 0.61 - 1.64
     - 0.86 x the start (x 0.72 - 1.39)
   * - LT1 / 1-h power
     - 0.75
     - 0.49 - 0.90
     - 0.70 (0.51 - 0.84)
   * - CP / MAP
     - 0.78
     - 0.68 - 0.85
     - 0.72 (0.61 - 0.81)
   * - Pmax
     - the median of your first year's 8-week best 5-s powers (900 W without)
     - x 0.52 - 1.93
     - 0.89 x the start (x 0.61 - 1.64)
   * - fade (:math:`1/\delta`, :doc:`overview`)
     - 0.08
     - 0.035 - 0.18
     - 0.12 (0.062 - 0.23)

MAP and Pmax are centred on the rider's own first year of rides (at least 10 with efforts, else the generic
watts): in fixed watts the priors sat two to three standard deviations from a much weaker or stronger rider.
Each block of 8 weeks gives its best 5 minutes and 5 seconds, and the centre is their median: most 8-week blocks
hold a near all-out effort, and a block a broken power meter set moves no median (the year's best did: one
minutes-long 1000 W reading put a synthetic rider's MAP centre at 3.3 x the truth). The ranges stay as wide as
before.

The floors are what the states decay toward without training; for Pmax and durability (no detraining) they
are prior only. The CP / MAP prior is the narrow one on purpose: with a wider one (0.57 - 0.89) the fit can explain
every effort as well below a huge, never-tested ceiling, as well as with the real one.

Training
--------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Parameter
     - Median
     - 90 % range
   * - Detraining time constant :math:`\tau`: MAP, CP/MAP
     - 60 days
     - 19 - 190 days
   * - :math:`\tau`: LT1 / 1-h power
     - 60 days
     - 31 - 116 days
   * - :math:`\tau`: Pmax, durability (prior only: they don't detrain)
     - 90 days
     - 28 - 285 days
   * - Training effect on log MAP, per dose (z1, z2)
     - 0.24
     - 0.016 - 3.7
   * - ... (z3 = severe: above CP by power and heart rate together; z4 = above MAP)
     - 0.24
     - 0.09 - 0.64
   * - Training effect on logit LT1 / 1-h power, per dose (z1, z2, long)
     - 1.2
     - 0.08 - 18
   * - Training effect on logit CP/MAP, per dose (z2, z3)
     - 0.6
     - 0.04 - 9.1
   * - Pool turnover :math:`\rho` (per day, fresh; :doc:`dynamics`): MAP
     - 1/10
     - 1/19 - 1/5
   * - ... LT1 / 1-h share
     - 1/35
     - 1/68 - 1/18
   * - ... CP / MAP share
     - 1/17
     - 1/33 - 1/9
   * - Cost of fatigue on adaptation :math:`\gamma`
     - 0.05
     - 0.010 - 0.26
   * - Weekly process noise :math:`\eta`
     - 0
     - standard normal (scale fixed, see :doc:`dynamics`)

The training effects are deliberately vague (log sd 1.66, a factor of 15 either way at 90 %): which domain
trains what, and how much, is what the model is meant to learn from the data. The *pattern* of the gain matrix
(which entries may be non-zero) is fixed. Except MAP's effects from work between CP and MAP (VO2max intervals) and
above MAP (log sd 0.6): a rider's minutes there always come together with their hours, so the data can't tell those
effects apart, and with the wide prior they slid to almost nothing (one athlete's data, 2026). They are held near
what studies of trained cyclists found (about 0.05 - 0.18 % of MAP per 10 minutes between CP and MAP, at a
detraining time constant of 200 - 300 days; :cite:alp:`ronnestad2014,seiler2013,stoggl2014`), and shown as *assumed* when the draws stay at the prior.

LT1's share of 1-h power has a narrower time constant (log sd 0.4, about 27 - 130 days at 95 %) than MAP and CP / MAP
(0.7). After complete cessation of training the threshold's share of the ceiling falls slowly and only partly, and
has levelled within 8 - 12 weeks: in endurance athletes who stopped, submaximal responses changed over the first 56
days and then held, and after 84 days the lactate threshold still sat at a higher share of VO2max than in untrained
people (:cite:alp:`coyle1984,coyle1985`; reviewed by :cite:alp:`mujika2000a,mujika2000b`). Everyday rides say little about this constant
(a rider rarely stops for long), so with the wider prior the data left it wider than the prior, and the sampler stalled
in its long tail.

Fatigue
-------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Parameter
     - Median
     - 90 % range
   * - Fatigue time constant :math:`\tau_F`
     - 7 days
     - 4.3 - 11.5 days
   * - Fatigue cost on the day :math:`s_F` (log power per unit fatigue)
     - 0.01
     - 0.002 - 0.05

Observations
------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Parameter
     - Median
     - 90 % range
   * - Chance a best power is all-out :math:`\psi_k`: easy ride
     - 1 %
     - 0.4 - 2.2 %
   * - ... hard workout
     - 5 %
     - 2.3 - 11 %
   * - ... test
     - 40 %
     - 11 - 78 %
   * - ... maxed (heart rate at your maximum)
     - 90 %
     - 79 - 96 %
   * - Deficit shape :math:`\alpha` (all durations; sampled as :math:`\log(\alpha - 2)`)
     - 4
     - 2.6 - 8.3
   * - Mean deficit :math:`\mu_\Delta` at the middle duration, :math:`e^a` (log power)
     - 0.40
     - 0.077 - 2.1
   * - ... its slope :math:`b` per sd of log duration
     - 0
     - -1.6 - 1.6
   * - Scale multiplier :math:`e^{\kappa_k}`, hard / test days
     - 1
     - 0.44 - 2.3
   * - Reading bias :math:`b_j`
     - 0
     - :math:`\pm 1.645\,` × the source's bias sd (below)
   * - Reading error :math:`s_j`
     - the source's sd
     - log sd 0.3 around it (plus a floor of 0.01)
   * - Heart-rate curve :math:`g` at 70 / 80 / 90 / 100 % of threshold HR (log power / CP)
     - the straight line from resting HR, :math:`\log\frac{h - \mathrm{RHR}}{\mathrm{LTHR} - \mathrm{RHR}}`
     - ± 0.33 (sd 0.2) around it
   * - ... a rep's offset :math:`o`
     - 0
     - ± 8 %
   * - ... fatigue holding HR down :math:`c` (per unit fatigue)
     - 0
     - ± 0.033
   * - ... a stretch's own error :math:`s` / the week's shared error :math:`w`
     - 0.05 / 0.03
     - log sd 0.5 / 0.7 around them

Per source (log power; 0.05 ≈ 5 %):

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Source
     - Measures
     - Bias sd
     - Error sd
   * - ``my_ftp``
     - CP
     - 0.03
     - 0.03
   * - ``alpha1_lt1``
     - LT1
     - 0.08
     - 0.08
   * - ``ramp_lt1`` (upper bound)
     - LT1
     - 0.01
     - 0.035
   * - ``ramp_lt2`` (upper bound)
     - CP
     - 0.01
     - 0.035
   * - ``ramp_hr_lt1`` (upper bound)
     - LT1
     - 0.01
     - 0.05
   * - ``ramp_hr_lt2`` (upper bound)
     - CP
     - 0.01
     - 0.05

Every source has a fitted bias, so the *level* of a threshold is set by the balance of these bias priors and the
best powers: a source with a tight bias prior (your own FTP entries, ramp tests) anchors it most.

Tests are the reference where they measure a threshold well, and a ramp measures its thresholds from above: its LT1
and LT2 read high (alpha1 and heart rate lag the rising power), so they are soft upper bounds (:doc:`likelihood`) -
the threshold is at most the reading, with about 3.5 % slack (5 % from heart rate), and no bias of their own beyond
a 1 % tolerance. The ramp's top is not a MAP reading: legs can give out before the aerobic ceiling, after all the
ramp has taken, so it can sit well below fresh 5-minute power. The ride's best powers enter as efforts instead
(a test may be all-out or not). A fit made with other weights is redone, starting from it (``PhysioRun.priors``).

A ramp LT2 at 92 % or more of the ramp's top, or within 3 bpm of the ride's highest heart rate, crossed only at
exhaustion (``RampTest.lt2_check``): as a bound it is loose and harmless, the pages say so, and it is no FTP value
for the fitted FTP model.

Fixed, not fitted
-----------------

Some quantities are constants because fitting them breaks the fit (each was tried):

* the all-out scatter, 3 % (fitted, it widens until "all-out" describes ordinary rides);
* the share of best powers that are bad readings, :math:`10^{-4}`, spread over 2 in log power (fitted, it would
  explain ordinary rides away);
* the process-noise size per state (fitted, the mode runs to large noise that matches every reading);
* the minimum deficit shape, 2 (below it the likelihood has infinite density or gradient at the curve);
* the soft domain-edge width, 5 % in log power;
* hard physical limits on the states (MAP 40-1500 W, LT1 / 1-h power 0.2-0.98, CP/MAP 0.4-0.98, Pmax 100-3000 W,
  fade 0.005-1), which keep wild optimiser trial points finite; nothing real is near them.
