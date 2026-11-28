// How the physio model's numbers moved (Trend): a number on a day and its change over the n days before, from the
// model's daily states (S.physio: date, lt1, cp, map, ftp, w_prime, p20). Today's "Your numbers" reads it.
import {addDays, fmt} from "./util.js";

export class Trend {
  static NEAR = 3;            // a day just past the states' last (today, before the morning's refit) reads the last

  constructor(P) {
    this.P = P || {};
    this.at = new Map((this.P.date || []).map((d, i) => [d, i]));
  }

  // the states' row for day ds: its own, else the last one when ds is at most NEAR days after it; -1 for none
  index(ds) {
    if (this.at.has(ds)) return this.at.get(ds);
    const d = this.P.date || [], last = d[d.length - 1];
    return last && ds > last && ds <= addDays(last, Trend.NEAR) ? d.length - 1 : -1;
  }

  value(k, ds) {
    const i = this.index(ds), v = i < 0 ? null : this.P[k]?.[i];
    return v == null ? null : v;
  }

  // the change in k over the n days up to ds (null when either end has no number)
  change(k, ds, n) {
    const now = this.value(k, ds), then = this.at.has(addDays(ds, -n)) ? this.P[k]?.[this.at.get(addDays(ds, -n))] : null;
    return now == null || then == null ? null : now - then;
  }

  // a change as text: "+6", "−3" or "±0" within half the last digit; its class: up (more), down (less) or none
  static words(d, digits = 0) {
    if (d == null) return ["–", ""];
    if (Math.abs(d) < 0.5 * 10 ** -digits) return ["±0", ""];
    return [`${d > 0 ? "+" : "−"}${fmt(Math.abs(d), digits)}`, d > 0 ? "up" : "down"];
  }
}
