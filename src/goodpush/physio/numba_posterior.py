"""The physio model's negative log posterior and its gradient in numba: a forward pass over the days and a
hand-written backward (adjoint) pass. Same numbers as `PhysioFit.potential` and its JAX gradient (checked in
the tests), several times faster per evaluation: the day loop is scalar code instead of hundreds of tiny
XLA operations a day.

Backward pass, in short (the forward pass is `PhysioModel.step` / `effort_terms` / `reading_terms`):
* likelihood terms are independent per effort / reading: their gradients go straight into the parameters and,
  as seeds, into the capacity of their day (MAP, CP, Pmax, fade, W', LT1, 1-h power) and that day's fatigue;
* the days are then swept backwards, carrying the adjoints of state x, pool and fatigue: each day adds its
  capacity seeds, pushes them through `capacity_backward` into x, and through the day's update rules;
* W' (an 8-step Newton solve in log y) is differentiated implicitly at its solution: the same as JAX's
  derivative through the unrolled steps once Newton has converged (it does in ~4).

The kernels are module-level functions: numba can't call methods of a class from compiled code.
`NumbaPosterior` prepares the data once and calls them.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from numba import config, njit, prange

from .model import (ALL_OUT_SD, BAD_READING_SHARE, BAD_READING_WIDTH, DOSE_UNIT_H, EDGE_WIDTH, EVIDENCE, GAIN_EFFECT, GAIN_INDEX, HR_KNOTS, HR_LOG_2PI, MAP_S, RATE_OF, MIN_READING_SD, MIN_SHAPE, NOISE_SD,
                    NOISE_WEEKS, STATE_HI, STATE_LO, TRAINED, PhysioModel)

# the parameter blocks the kernels read, in this order (offsets passed as one int array)
BLOCKS = ("x0", "floor", "log_tau", "log_effect", "eta", "log_rate", "log_gamma", "log_tau_f", "log_s_f",
          "log_shape_m2", "log_deficit", "log_scale_kind", "logit_pi", "bias", "log_sd",
          "hr_curve", "hr_rep", "hr_fatigue", "hr_log_sd", "hr_log_week_sd")
# per-day capacity kept by the forward pass (columns of `cap`)
MAP, PMAX, FADE, S1, S2, CP, WP, DWP_DP, DWP_DCP, DWP_DPMAX, P1H, DP1_DCP, DP1_DWP, DP1_DPMAX, DP1_DFADE, LT1,     W_U = range(17)                    # W_U: the W' Newton solution (log y), the next day's starting point
# per-day capacity adjoint seeds (columns of `gcap`)
G_MAP, G_CP, G_PMAX, G_FADE, G_WP, G_LT1, G_P1H = range(7)
T_CP = 1800.0
# parallel kernels (the Hessian) run on OpenMP where numba has it, else its other threading layers
config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
LOG_GOOD = math.log1p(-BAD_READING_SHARE)                          # an effort: a good reading's share, log
LOG_BAD = math.log(BAD_READING_SHARE / BAD_READING_WIDTH)          # ... a bad reading's density, log
SQRT2 = math.sqrt(2.0)


@njit(cache=True, nogil=True, error_model="numpy")
def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@njit(cache=True, nogil=True, error_model="numpy")
def log_sigmoid(x):
    return -math.log1p(math.exp(-x)) if x >= 0.0 else x - math.log1p(math.exp(x))


@njit(cache=True, nogil=True, error_model="numpy")
def good_or_bad(term):
    """an effort's log likelihood as a good reading (`term`) -> with the bad-reading share mixed in
    (PhysioModel.effort_terms), and the good reading's responsibility (what its gradients are scaled by)."""
    g = LOG_GOOD + term
    if g > LOG_BAD:
        e = math.exp(LOG_BAD - g)
        return g + math.log1p(e), 1.0 / (1.0 + e)
    e = math.exp(g - LOG_BAD)
    return LOG_BAD + math.log1p(e), e / (1.0 + e)


@njit(cache=True, nogil=True, error_model="numpy")
def digamma(x):
    """psi(x) for x > 0: recurrence up to 6, then the asymptotic series."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


@njit(cache=True, nogil=True, error_model="numpy")
def curve_power(t, cp, wp, pmax, fade):
    """PowerCurve.power and its partial derivatives (cp, W', Pmax, fade)."""
    span = pmax - cp
    act = span > 1.0
    if not act:
        span = 1.0
    e = math.exp(-t * span / wp)
    p = wp / t * (1.0 - e) + cp
    d_wp = (1.0 - e) / t - e * span / wp
    d_cp = 1.0 - (e if act else 0.0)
    d_pmax = e if act else 0.0
    d_fade = 0.0
    if t > T_CP:
        lt = math.log(t / T_CP)
        p -= fade * cp * lt
        d_cp -= fade * lt
        d_fade = -cp * lt
    return p, d_cp, d_wp, d_pmax, d_fade


@njit(cache=True, nogil=True, error_model="numpy")
def w_prime_solve(t, p, cp, pmax, u_start):
    """PowerCurve.w_prime_for and its implicit derivatives (p, cp, Pmax), and the solution u = log y.
    Newton in log y from u_start (yesterday's solution; <= 0: from 1 / rho like PowerCurve) until it stops
    moving (at most 8 steps, as PowerCurve): the same W' to machine precision."""
    span = pmax - cp
    act = span > 1.0
    if not act:
        span = 1.0
    rho = (p - cp) / span
    clipped = rho < 1e-3 or rho > 0.95
    rho = min(max(rho, 1e-3), 0.95)
    log_rho = math.log(rho)
    u = u_start if u_start > 0.0 else -log_rho
    for _ in range(8):
        y = math.exp(u)
        step = (math.log(-math.expm1(-y)) - u - log_rho) / (y / math.expm1(y) - 1.0)
        u = u - step
        if abs(step) < 1e-13:
            break
    y = math.exp(u)
    wp = t * span / math.exp(u)
    d_rho = 0.0 if clipped else -wp / (rho * (y / math.expm1(y) - 1.0))
    d_span = wp / span + d_rho * (-(p - cp) / (span * span))
    d_p = d_rho / span
    d_cp = -d_rho / span - (d_span if act else 0.0)
    d_pmax = d_span if act else 0.0
    return wp, d_p, d_cp, d_pmax, u


@njit(cache=True, nogil=True, error_model="numpy")
def capacity_forward(x, lo, hi, cap, mask, u_start):
    """PhysioModel.capacity of states x (5,) into cap (17,), with the clip mask (1 where x moves capacity).
    u_start: where W's Newton solve starts (the previous day's cap[W_U]; 0 = PowerCurve's start)."""
    for s in range(5):
        mask[s] = 1.0 if lo[s] < x[s] < hi[s] else 0.0
    cap[MAP] = math.exp(min(max(x[0], lo[0]), hi[0]))
    cap[S1] = sigmoid(min(max(x[1], lo[1]), hi[1]))
    cap[S2] = sigmoid(min(max(x[2], lo[2]), hi[2]))
    cap[PMAX] = math.exp(min(max(x[3], lo[3]), hi[3]))
    cap[FADE] = math.exp(-min(max(x[4], lo[4]), hi[4]))
    cap[CP] = cap[S2] * cap[MAP]
    cap[WP], cap[DWP_DP], cap[DWP_DCP], cap[DWP_DPMAX], cap[W_U] = w_prime_solve(MAP_S, cap[MAP], cap[CP],
                                                                               cap[PMAX], u_start)
    cap[P1H], cap[DP1_DCP], cap[DP1_DWP], cap[DP1_DPMAX], cap[DP1_DFADE] = curve_power(
        3600.0, cap[CP], cap[WP], cap[PMAX], cap[FADE])
    cap[LT1] = cap[S1] * cap[P1H]


@njit(cache=True, nogil=True, error_model="numpy")
def capacity_backward(cap, mask, g, gx):
    """adjoint of capacity_forward: seeds g (7,) on (MAP, CP, Pmax, fade, W', LT1, 1-h power) -> gx (5,) on x."""
    g_map, g_cp, g_pmax, g_fade, g_wp = g[G_MAP], g[G_CP], g[G_PMAX], g[G_FADE], g[G_WP]
    s1, s2 = cap[S1], cap[S2]
    g_p1 = g[G_LT1] * s1 + g[G_P1H]
    gx1 = g[G_LT1] * s1 * (1.0 - s1) * cap[P1H]
    g_cp += g_p1 * cap[DP1_DCP]
    g_wp += g_p1 * cap[DP1_DWP]
    g_pmax += g_p1 * cap[DP1_DPMAX]
    g_fade += g_p1 * cap[DP1_DFADE]
    g_map += g_wp * cap[DWP_DP]
    g_cp += g_wp * cap[DWP_DCP]
    g_pmax += g_wp * cap[DWP_DPMAX]
    g_map += g_cp * s2
    gx[0] = g_map * cap[MAP] * mask[0]
    gx[1] = gx1 * mask[1]
    gx[2] = g_cp * cap[MAP] * s2 * (1.0 - s2) * mask[2]
    gx[3] = g_pmax * cap[PMAX] * mask[3]
    gx[4] = -g_fade * cap[FADE] * mask[4]


@njit(cache=True, nogil=True, error_model="numpy")
def log_ndtr_and_mills(x):
    """log Phi(x) and phi(x) / Phi(x): erfc keeps its relative accuracy far into the tail; past -37 (Phi
    underflows) the asymptotic series."""
    if x > -37.0:
        p = 0.5 * math.erfc(-x / SQRT2)
        return math.log(p), math.exp(-0.5 * x * x - HALF_LOG_2PI) / p
    s = 1.0 - 1.0 / (x * x) + 3.0 / x ** 4 - 15.0 / x ** 6
    return -0.5 * x * x - HALF_LOG_2PI - math.log(-x) + math.log(s), -x / s


@njit(cache=True, nogil=True, error_model="numpy")
def potential_and_gradient(w, mean, sd, off, gain_row, gain_col, gain_effect, rate_of, lo, hi, trained, noise_sd, unit,
                           bin_scale, bin_start, bin_index, bin_hours, bin_evidence, hours, long_h, work, noise_size,
                           period,
                           effort_day, effort_s, log_effort_w, effort_kind, effort_x, effort_weight,
                           reading_day, log_reading_w, reading_source, reading_quantity, reading_bound,
                           hr_day, hr_logw, hr_x, hr_rep, hr_group, n_groups, knots, hr_fatigue_ref):
    """negative log posterior at standardised parameters w, and its gradient (PhysioFit.potential).
    bin_scale = exp(log power / EDGE_WIDTH) per power bin: a bin's soft-edge sigmoid is then
    q / (1 + q) with q = bin_scale x edge^(-1 / EDGE_WIDTH), without an exp per bin. gain_effect: each gain's
    entry of log_effect (GAIN_EFFECT); effort_x: each effort's standardised log duration (the deficit's scale line);
    effort_weight: each effort's weight (its log likelihood, and so all its gradients, scale by it).
    bin_*: each day's entries (power bin, hours, exp(heart-rate evidence)): its time without heart rate (evidence 0,
    exp 1) and its time with heart rate per evidence bin (PhysioData.hr_hist_h); the LT1-MAP band splits by
    s = logistic((log P - log CP) / EDGE_WIDTH + evidence) (PhysioModel.doses). hours: the day's total."""
    z = mean + sd * w
    gz = np.zeros(z.size)
    T, K = hours.size, gain_row.size
    o_x0, o_floor, o_tau, o_eff, o_eta, o_rate, o_gam, o_tauf, o_sf, o_shape, o_scale, o_kind, o_pi, o_bias, \
        o_lsd = off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7], off[8], off[9], off[10], off[11], \
        off[12], off[13], off[14]
    gam = math.exp(z[o_gam])
    rate = np.empty(5)                                     # each state's reservoir rate (RATE_OF)
    for s in range(5):
        rate[s] = math.exp(z[o_rate + rate_of[s]])
    tau_f = math.exp(z[o_tauf])
    decay = math.exp(-1.0 / tau_f)
    detr, floor = np.empty(5), np.empty(5)
    for s in range(5):
        detr[s] = trained[s] * math.exp(-z[o_tau + s])
        floor[s] = z[o_floor + s]
    gk = np.empty(K)                                       # gains, one per (state, dose) pair
    for k in range(K):
        gk[k] = math.exp(z[o_eff + gain_effect[k]] - z[o_tau + gain_row[k]])
    inv_w = 1.0 / EDGE_WIDTH

    # --- forward over the days: morning state, pool, fatigue; capacity; doses -------------------------
    X, P, F = np.empty((T, 5)), np.empty((T, 5)), np.empty(T)
    cap, mask = np.empty((T, 17)), np.empty((T, 5))
    dose, edge = np.empty((T, 6)), np.empty((T, 4))
    dd = np.empty((T, 6))          # d (sums) / d log edge: D0, D0s, B1, D2, D2s, D3 (see the backward pass)
    x, pool, xn, cj = np.empty(5), np.zeros(5), np.empty(5), np.empty(4)
    for s in range(5):
        x[s] = z[o_x0 + s]
    fat = 0.0
    for t in range(T):
        for s in range(5):
            X[t, s] = x[s]
            P[t, s] = pool[s]
        F[t] = fat
        c = cap[t]
        capacity_forward(x, lo, hi, c, mask[t], cap[t - 1, W_U] if t > 0 else 0.0)
        edge[t, 0], edge[t, 1], edge[t, 2] = c[LT1], c[CP], c[MAP]
        edge[t, 3] = c[MAP] + 0.5 * (c[PMAX] - c[MAP])
        for j in range(4):
            cj[j] = math.exp(-math.log(edge[t, j]) * inv_w)
        s_a0, s_a2, s_a3, s_band, s_bs = 0.0, 0.0, 0.0, 0.0, 0.0
        d0, d0s, b1, d2, d2s, d3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(bin_start[t], bin_start[t + 1]):          # the day's entries with time in them
            sc, h = bin_scale[bin_index[i]], bin_hours[i]
            q0, q2, q3 = sc * cj[0], sc * cj[2], sc * cj[3]
            a0, a2, a3 = q0 / (1.0 + q0), q2 / (1.0 + q2), q3 / (1.0 + q3)
            qs = sc * cj[1] * bin_evidence[i]
            sv = qs / (1.0 + qs)                                 # severe share: power + heart-rate evidence
            band = a0 - a2
            s_a0 += h * a0
            s_a2 += h * a2
            s_a3 += h * a3
            s_band += h * band
            s_bs += h * band * sv
            g0, g2 = -a0 * (1.0 - a0) * inv_w * h, -a2 * (1.0 - a2) * inv_w * h   # h x d a / d log edge
            d0 += g0
            d0s += g0 * sv
            d2 += g2
            d2s += g2 * sv
            b1 += -h * band * sv * (1.0 - sv) * inv_w
            d3 += -a3 * (1.0 - a3) * inv_w * h
        dd[t, 0], dd[t, 1], dd[t, 2], dd[t, 3], dd[t, 4], dd[t, 5] = d0, d0s, b1, d2, d2s, d3
        dose[t, 0] = (hours[t] - s_a0) / unit[0]
        dose[t, 1] = (s_band - s_bs) / unit[1]
        dose[t, 2] = s_bs / unit[2]
        dose[t, 3] = (s_a2 - s_a3) / unit[3]
        dose[t, 4] = s_a3 / unit[4]
        dose[t, 5] = long_h[t] / unit[5]
        slow = math.exp(-gam * fat)
        eta = o_eta + period[t] * 5
        for s in range(5):
            r = 1.0 - math.exp(-rate[s] * slow)
            xn[s] = x[s] + r * pool[s] - detr[s] * (x[s] - floor[s]) + noise_size[t] * noise_sd[s] * z[eta + s]
            pool[s] *= 1.0 - r
        for k in range(K):
            pool[gain_row[k]] += gk[k] * dose[t, gain_col[k]]
        fat = fat * decay + work[t] / (c[CP] * c[CP])
        for s in range(5):
            x[s] = xn[s]

    # --- likelihood: efforts, readings (seeds on each day's capacity and fatigue) --------------------
    loglik = 0.0
    gcap, gF = np.zeros((T, 7)), np.zeros(T)
    sf = math.exp(z[o_sf])
    var = ALL_OUT_SD * ALL_OUT_SD
    a = MIN_SHAPE + math.exp(z[o_shape])                       # the deficit's shape, all durations
    lgam, dig, log_a = math.lgamma(a), digamma(a), math.log(a)    # scale = mean / shape
    nk = o_bias - o_pi                                         # effort kinds (physio KINDS)
    lsp, lsm, sig_p, sig_m, g_pi = np.empty(nk), np.empty(nk), np.empty(nk), np.empty(nk), np.zeros(nk)
    for kd in range(nk):
        lsp[kd], lsm[kd] = log_sigmoid(z[o_pi + kd]), log_sigmoid(-z[o_pi + kd])
        sig_p[kd], sig_m[kd] = sigmoid(z[o_pi + kd]), sigmoid(-z[o_pi + kd])
    const = -math.log(ALL_OUT_SD) - HALF_LOG_2PI
    for i in range(effort_day.size):
        d, kd, ex, wt = effort_day[i], effort_kind[i], effort_x[i], effort_weight[i]
        c = cap[d]
        p, dp_cp, dp_wp, dp_pmax, dp_fade = curve_power(effort_s[i], c[CP], c[WP], c[PMAX], c[FADE])
        u = log_effort_w[i] - math.log(max(p, 1.0))
        uf = u + sf * F[d]
        ao = lsp[kd] - 0.5 * uf * uf / var + const
        if u < 0.0:
            log_sc = z[o_scale] + z[o_scale + 1] * ex + (z[o_kind + kd - 1] if kd > 0 else 0.0) - log_a
            sc = math.exp(log_sc)
            xg = max(-u, 1e-9)
            lxs = math.log(xg) - log_sc
            sm = lsm[kd] + (a - 1.0) * lxs - xg / sc - lgam - log_sc
            if ao > sm:
                e = math.exp(sm - ao)
                term = ao + math.log1p(e)
                ra = 1.0 / (1.0 + e)
            else:
                e = math.exp(ao - sm)
                term = sm + math.log1p(e)
                ra = e / (1.0 + e)
            rs = 1.0 - ra
            term, rg = good_or_bad(term)
            ra, rs = wt * rg * ra, wt * rg * rs          # weight x good reading: every gradient below is linear in ra, rs
            gu = -ra * uf / var
            gF[d] += gu * sf
            gz[o_sf] += gu * sf * F[d]
            if -u > 1e-9:
                gu -= rs * ((a - 1.0) / xg - 1.0 / sc)
            g_lsc = rs * (xg / sc - a)
            gz[o_shape] += (rs * (lxs - dig) - g_lsc / a) * (a - MIN_SHAPE)
            gz[o_scale] += g_lsc
            gz[o_scale + 1] += g_lsc * ex
            if kd > 0:
                gz[o_kind + kd - 1] += g_lsc
            g_pi[kd] += ra * sig_m[kd] - rs * sig_p[kd]
        else:
            term, rg = good_or_bad(ao)
            ra, rs = wt * rg, 0.0
            gu = -ra * uf / var
            gF[d] += gu * sf
            gz[o_sf] += gu * sf * F[d]
            g_pi[kd] += ra * sig_m[kd]
        loglik += wt * term
        if p > 1.0:
            gp = -gu / p
            gcap[d, G_CP] += gp * dp_cp
            gcap[d, G_WP] += gp * dp_wp
            gcap[d, G_PMAX] += gp * dp_pmax
            gcap[d, G_FADE] += gp * dp_fade
    for kd in range(nk):
        gz[o_pi + kd] += g_pi[kd]
    for m in range(reading_day.size):
        d, src, q = reading_day[m], reading_source[m], reading_quantity[m]
        c = cap[d]
        val = c[LT1] if q == 0 else (c[CP] if q == 1 else (c[MAP] if q == 2 else c[P1H]))
        e_sd = math.exp(z[o_lsd + src])
        s_r = MIN_READING_SD + e_sd
        res = log_reading_w[m] - math.log(val) - z[o_bias + src]
        if reading_bound[m]:                 # an upper bound: log Phi(res / s), only a state above it costs
            lc, mills = log_ndtr_and_mills(res / s_r)
            loglik += lc
            g_pred = -mills / s_r
            gz[o_bias + src] += g_pred
            gz[o_lsd + src] += -mills * res / (s_r * s_r) * e_sd
        else:
            loglik += -0.5 * res * res / (s_r * s_r) - math.log(s_r) - HALF_LOG_2PI
            g_pred = res / (s_r * s_r)
            gz[o_bias + src] += g_pred
            gz[o_lsd + src] += (-1.0 / s_r + res * res / (s_r * s_r * s_r)) * e_sd
        col = G_LT1 if q == 0 else (G_CP if q == 1 else (G_MAP if q == 2 else G_P1H))
        gcap[d, col] += g_pred / val
    # steady stretches: residuals, their week sums, the closed-form weekly-correlated likelihood (PhysioModel.hr_terms)
    nh = hr_day.size
    if nh > 0:
        o_hc, o_hrep, o_hfat, o_hsd, o_hwk = off[15], off[16], off[17], off[18], off[19]
        s2, w2 = math.exp(2.0 * z[o_hsd]), math.exp(2.0 * z[o_hwk])
        cf, rep_off = z[o_hfat], z[o_hrep]
        n_knots = knots.size
        hres, hseg, hfrac = np.empty(nh), np.empty(nh, np.int64), np.empty(nh)
        wS, wQ, wn = np.zeros(n_groups), np.zeros(n_groups), np.zeros(n_groups)
        for i in range(nh):
            hd, hx = hr_day[i], hr_x[i]
            hj = 0
            while hj < n_knots - 2 and hx > knots[hj + 1]:
                hj += 1
            ht = (hx - knots[hj]) / (knots[hj + 1] - knots[hj])
            hr_r = (hr_logw[i] - math.log(cap[hd, CP]) - ((1.0 - ht) * z[o_hc + hj] + ht * z[o_hc + hj + 1])
                    - hr_rep[i] * rep_off - cf * (F[hd] - hr_fatigue_ref))
            hres[i], hseg[i], hfrac[i] = hr_r, hj, ht
            wk = hr_group[i]
            wS[wk] += hr_r
            wQ[wk] += hr_r * hr_r
            wn[wk] += 1.0
        g_s2, g_w2 = 0.0, 0.0
        for wk in range(n_groups):
            if wn[wk] > 0.0:
                den = s2 + wn[wk] * w2
                loglik += -0.5 * ((wQ[wk] - w2 * wS[wk] * wS[wk] / den) / s2 + (wn[wk] - 1.0) * math.log(s2)
                                  + math.log(den) + wn[wk] * HR_LOG_2PI)
                g_s2 += -0.5 * (-wQ[wk] / (s2 * s2) + w2 * wS[wk] * wS[wk] * (den + s2) / (s2 * s2 * den * den)
                                + (wn[wk] - 1.0) / s2 + 1.0 / den)
                g_w2 += -0.5 * (-wS[wk] * wS[wk] / (den * den) + wn[wk] / den)
        gz[o_hsd] += g_s2 * 2.0 * s2
        gz[o_hwk] += g_w2 * 2.0 * w2
        for i in range(nh):
            hd, wk, hj, ht = hr_day[i], hr_group[i], hseg[i], hfrac[i]
            gm = (hres[i] - w2 * wS[wk] / (s2 + wn[wk] * w2)) / s2   # = -d loglik / d r = d loglik / d (prediction)
            gcap[hd, G_CP] += gm / cap[hd, CP]
            gF[hd] += gm * cf
            gz[o_hfat] += gm * (F[hd] - hr_fatigue_ref)
            gz[o_hrep] += gm * hr_rep[i]
            gz[o_hc + hj] += gm * (1.0 - ht)
            gz[o_hc + hj + 1] += gm * ht

    # --- backward over the days -------------------------------------------------------------------
    ax, ap, af, rr = np.zeros(5), np.zeros(5), 0.0, np.empty(5)
    g_gk, gdose, g, gxc = np.zeros(K), np.empty(6), np.empty(7), np.empty(5)
    for t in range(T - 1, -1, -1):
        c, ft = cap[t], F[t]
        slow = math.exp(-gam * ft)
        g_slow = 0.0                                       # d / d (rate x slow) summed over the states, x rate
        eta = o_eta + period[t] * 5
        for s in range(5):
            mq = rate[s] * slow
            em = math.exp(-mq)
            gq = (ax[s] - ap[s]) * P[t, s] * em * mq      # d / d log(rate_s x slow)
            gz[o_rate + rate_of[s]] += gq
            g_slow += gq
            rr[s] = 1.0 - em
            gz[o_floor + s] += ax[s] * detr[s]
            gz[o_tau + s] += ax[s] * detr[s] * (X[t, s] - floor[s])
            gz[eta + s] += ax[s] * noise_size[t] * noise_sd[s]
        for cl in range(6):
            gdose[cl] = 0.0
        for k in range(K):
            gdose[gain_col[k]] += gk[k] * ap[gain_row[k]]
            g_gk[k] += ap[gain_row[k]] * dose[t, gain_col[k]]
        gz[o_tauf] += af * ft * decay / tau_f
        for q in range(7):
            g[q] = gcap[t, q]
        g[G_CP] += af * work[t] * (-2.0) / (c[CP] * c[CP] * c[CP])
        gz[o_gam] += g_slow * (-gam * ft)
        g_fat = gF[t] + af * decay + g_slow * (-gam)
        # doses -> domain edges (log) -> capacity. dose 0 = (hours - sum a0) / u0, 1 = (band - band s) / u1,
        # 2 = band s / u2, 3 = (sum a2 - sum a3) / u3, 4 = sum a3 / u4 with band = a0 - a2
        u0, u1, u2, u3, u4 = gdose[0] / unit[0], gdose[1] / unit[1], gdose[2] / unit[2], gdose[3] / unit[3], gdose[4] / unit[4]
        d0, d0s, b1, d2, d2s, d3 = dd[t, 0], dd[t, 1], dd[t, 2], dd[t, 3], dd[t, 4], dd[t, 5]
        gl0 = -u0 * d0 + u1 * (d0 - d0s) + u2 * d0s
        gl1 = (u2 - u1) * b1
        gl2 = u1 * (d2s - d2) - u2 * d2s + u3 * d2
        gv3 = (u4 - u3) * d3 / edge[t, 3]
        g[G_LT1] += gl0 / edge[t, 0]
        g[G_CP] += gl1 / edge[t, 1]
        g[G_MAP] += gl2 / edge[t, 2] + 0.5 * gv3
        g[G_PMAX] += 0.5 * gv3
        capacity_backward(c, mask[t], g, gxc)
        # the day's pool feeds x_new (r pool) and pool_new ((1 - r) pool): from the day-after adjoints
        for s in range(5):
            a_next = ax[s]
            ap[s] = a_next * rr[s] + ap[s] * (1.0 - rr[s])
            ax[s] = a_next * (1.0 - detr[s]) + gxc[s]
        af = g_fat
    for s in range(5):
        gz[o_x0 + s] += ax[s]
    for k in range(K):
        gz[o_eff + gain_effect[k]] += g_gk[k] * gk[k]
        gz[o_tau + gain_row[k]] -= g_gk[k] * gk[k]

    # --- prior, sign: potential = -log posterior, gradient on w ------------------------------------------
    value = loglik
    gw = np.empty(w.size)
    for i in range(w.size):
        value += -0.5 * w[i] * w[i] - math.log(sd[i]) - HALF_LOG_2PI
        gw[i] = -(gz[i] * sd[i] - w[i])
    return -value, gw


@njit(cache=True, nogil=True, error_model="numpy", parallel=True)
def hessian_by_differences(w, step, args):
    """Hessian of the potential at w by central differences of its gradient (one pair of gradients per
    parameter, the parameters spread over the cores), symmetrised. args: NumbaPosterior.args."""
    n = w.size
    H = np.empty((n, n))
    for i in prange(n):
        wp = w.copy()                                     # each thread perturbs its own copy
        wp[i] = w[i] + step
        g_hi = potential_and_gradient(wp, *args)[1]
        wp[i] = w[i] - step
        g_lo = potential_and_gradient(wp, *args)[1]
        for j in range(n):
            H[i, j] = (g_hi[j] - g_lo[j]) / (2.0 * step)
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = 0.5 * (H[i, j] + H[j, i])
    return H


class NumbaPosterior:
    """`PhysioFit.potential` (negative log posterior at standardised w) and its gradient for one model's data,
    computed by the numba kernels. The data are laid out once: each day's non-empty power bins, the efforts and
    readings without padding. Effort / reading days past the last day count on the last day, as in JAX's
    indexing (the kernels don't check bounds)."""

    def __init__(self, model: PhysioModel):
        d, spec = model.data, model.spec
        T = len(d.days)
        total = np.asarray(d.hist_h, float)
        hr = d.hr_hours                                            # (T, B, E) time with heart rate by evidence
        # entries: (day, bin, evidence) - the time without heart rate at evidence 0, then each evidence bin's
        H = np.concatenate([np.maximum(total - hr.sum(2), 0.0)[:, :, None], hr], axis=2)
        ev = np.r_[1.0, np.exp(EVIDENCE)]
        day, col, k = np.nonzero(H)
        hr_group = np.asarray(model.arrays["hr_group"])[:len(d.hr_w)].astype(np.int64)
        self.args = (
            np.asarray(spec.mean, float), np.asarray(spec.sd, float),
            np.array([spec.offsets[b] for b in BLOCKS], np.int64),
            GAIN_INDEX[0].astype(np.int64), GAIN_INDEX[1].astype(np.int64), GAIN_EFFECT.astype(np.int64),
            RATE_OF.astype(np.int64),
            np.asarray(STATE_LO, float), np.asarray(STATE_HI, float), np.asarray(TRAINED, float),
            np.asarray(NOISE_SD, float) * np.sqrt(NOISE_WEEKS), np.asarray(DOSE_UNIT_H, float),
            np.exp(np.log(np.asarray(d.power_w, float)) / EDGE_WIDTH),
            np.r_[0, np.cumsum(np.bincount(day, minlength=T))].astype(np.int64), col.astype(np.int64),
            H[day, col, k], ev[k],
            H.sum((1, 2)), np.asarray(d.long_h, float), total @ np.asarray(d.power_w, float) ** 2,
            np.asarray(model.arrays["noise_size"])[:T, 0].astype(float),
            np.asarray(model.arrays["period"])[:T].astype(np.int64),
            np.minimum(np.asarray(d.effort_day, np.int64), T - 1), np.asarray(d.effort_s, float), np.log(np.asarray(d.effort_w, float)),
            np.asarray(d.effort_kind, np.int64), np.asarray(model.duration_x, float)[np.asarray(model.effort_t, np.int64)],
            d.weights,
            np.minimum(np.asarray(d.reading_day, np.int64), T - 1), np.log(np.asarray(d.reading_w, float)),
            np.asarray(d.reading_source, np.int64),
            np.asarray(model.arrays["reading_quantity"])[:len(d.reading_w)].astype(np.int64),
            np.asarray(model.arrays["reading_bound"])[:len(d.reading_w)].astype(np.int64),
            np.asarray(d.hr_day, np.int64), np.log(np.asarray(d.hr_w, float)), np.asarray(d.hr_x, float),
            np.asarray(d.hr_rep, float), hr_group, int(hr_group.max()) + 1 if hr_group.size else 1,
            np.asarray(HR_KNOTS, float), float(d.hr_fatigue_ref))

        size = int(spec.size)
        self.shapes = (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((size,), jnp.float64))
        # `potential` for JAX code (NumPyro's NUTS): the value from numba, and its gradient from the same call
        self.potential = jax.custom_vjp(self.jax_value)
        self.potential.defvjp(self.jax_value_and_grad, self.jax_vjp)

    def value_and_grad(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        return potential_and_gradient(np.asarray(w, float), *self.args)

    HESSIAN_STEP = 1e-7            # central-difference step (standardised units): error ~ step^2 (4e-8 at 1e-7)

    def hessian(self, w: np.ndarray) -> np.ndarray:
        """Hessian of the potential at w (from differences of the exact gradient)."""
        return hessian_by_differences(np.asarray(w, float), self.HESSIAN_STEP, self.args)

    def host_value_and_grad(self, w) -> tuple[np.ndarray, np.ndarray]:
        v, g = self.value_and_grad(np.asarray(w, float))
        return np.asarray(v, np.float64), g

    def jax_value(self, w):
        return jax.pure_callback(self.host_value_and_grad, self.shapes, w, vmap_method="sequential")[0]

    def jax_value_and_grad(self, w):
        return jax.pure_callback(self.host_value_and_grad, self.shapes, w, vmap_method="sequential")

    @staticmethod
    def jax_vjp(grad, ct):
        return (ct * grad,)
