// Pure helpers: DOM lookup, escaping, number / time formats, dates.
export const $ = (sel, el = document) => el.querySelector(sel);
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
export const fmt = (v, nd = 0) => (v == null || Number.isNaN(Number(v)) ? "–" : (Math.abs(v) < 0.5 * 10 ** -nd ? 0 : Number(v)).toFixed(nd));
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const LOCAL = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
// durations: seconds as m:ss, from an hour as h:mm ("1:05 h"); minutes under an hour as "45 min", else h:mm
export const mmss = (sec) => { sec = Math.round(sec); const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s2 = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")} h` : `${m}:${String(s2).padStart(2, "0")}`; };
export const hhmm = (min, unit = true) => { if (min == null || min !== min) return "–"; min = Math.round(min);
  return min < 60 && unit ? `${min} min` : `${Math.floor(min / 60)}:${String(min % 60).padStart(2, "0")}${unit ? " h" : ""}`; };
// Dave Hale's two-sided recursive exponential filter (Mines JTK RecursiveExponentialFilter): a causal then an
// anti-causal first-order pass, y = (1 - a)^2 / ((1 - a/z)(1 - a z)) x. Bell-shaped, no time shift, no ringing,
// O(n) at any width. a gives the impulse response the variance sigma^2 (samples); a box of w samples has
// sigma^2 = (w^2 - 1) / 12 (boxSigma). Ends: the input continued flat. Gaps (null) stay gaps; each stretch between
// them is filtered on its own.
export function expSmooth(arr, sigma) {
  const n = arr.length; if (!(sigma > 0) || n < 2) return arr;
  const ss = sigma * sigma, a = (1 + ss - Math.sqrt(1 + 2 * ss)) / ss, b = 1 - a, y = Array.from(arr);
  for (let i0 = 0; i0 < n; ) {
    if (arr[i0] == null) { i0++; continue; }
    let i1 = i0; while (i1 + 1 < n && arr[i1 + 1] != null) i1++;
    let u = arr[i0];                                                        // causal pass, started in steady state
    for (let i = i0; i <= i1; i++) y[i] = u = a * u + b * arr[i];
    const xe = arr[i1]; let z = xe + (y[i1] - xe) / (1 + a);               // anti-causal pass: exact flat-end state
    y[i1] = z;
    for (let i = i1 - 1; i >= i0; i--) y[i] = z = a * z + b * y[i];
    i0 = i1 + 1;
  }
  return y;
}
export const boxSigma = (w) => Math.sqrt(Math.max(0, w * w - 1) / 12);   // the sigma that smooths like a w-sample box
// the physio model's lines in the zone colour map (--z1 .. --z7), each where its intensity lands (planning.intensity:
// LT1 = 0.75, CP = 1.0, MAP = 1.2 of CP): LT1 at the edge where tempo starts (zone 2's own colour is too faint for a
// line, about 2:1), CP and FTP threshold, MAP the VO2max domain, W′ (work above CP) and Pmax the short hard efforts
export const INTENSITY_COL = {lt1: "--z3", cp: "--z4", ftp: "--z4", map: "--z5", w_prime: "--z6", pmax: "--z7"};
export function addDays(ds, n) {
  const d = new Date(ds + "T12:00:00"); d.setDate(d.getDate() + n);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
