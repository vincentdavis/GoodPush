// The charts: EChart (a chart spec drawn by Apache ECharts), TimeWindow (zoom / pan for every time axis),
// SelectionPanel, and small SVG helpers (el, niceTicks) for the app's own drawings (calendar, map, profiles).
import {css, esc, fmt} from "./util.js";
import {niceDate} from "./widgets.js";
import {router} from "./main.js";
import {rideMap} from "./ride_map.js";
import {calendarPage} from "./calendar/calendar.js";
import {init} from "../vendor/echarts.esm.min.js";

const SVGNS = "http://www.w3.org/2000/svg";
export const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const selectGroups = {};  // charts that share a selection band
export function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  return t;
}
export function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
// A zoomable x range: one per chart, or one per group of charts on the same axis (they zoom together); all
// forgotten on a page change. extent = all the data, view = what's shown (null = all of it). EChart keeps its zoom
// here, so range buttons, zoom buttons (zoomTo / zoomAt / pan) and redraws keep it.
export class TimeWindow {
  static windows = {};           // key (chart id or group) -> window

  static of(key) { return (TimeWindow.windows[key] ||= new TimeWindow(key)); }
  static forget(...keys) { for (const k of keys) delete TimeWindow.windows[k]; }
  static forgetAll() { TimeWindow.windows = {}; }

  constructor(key) {
    this.key = key;
    this.extent = null;          // [x0, x1] of all the data
    this.view = null;            // [x0, x1] shown, null = everything
    this.opened = false;         // its opening window (a chart's spec.view) was applied: showing all then stays
  }

  shown() { return this.view || this.extent; }

  // a range button (6 months / 1 y / All): the last `days` before `from` up to `to` (ms; 0 = all); undefined keeps
  // a zoom you made, else opens on `fallback` days. The extent is measured again by the next draw
  showLast(days, from, to, fallback) {
    this.extent = null;
    if (days === undefined) days = this.view ? null : fallback;
    if (days) this.view = [from - days * 864e5, to];
    else if (days === 0) this.view = null;
  }

  widen([a, b]) {                // the group's full extent (its charts may differ)
    this.extent = this.extent ? [Math.min(this.extent[0], a), Math.max(this.extent[1], b)] : [a, b];
  }

  zoomTo(r) {                    // r = [a, b] in x units, or null for everything
    const [e0, e1] = this.extent, span = r ? r[1] - r[0] : 0;
    if (!r || span >= e1 - e0) this.view = null;
    else if (span > (e1 - e0) / 1000) { const a = Math.max(e0, Math.min(e1 - span, r[0])); this.view = [a, a + span]; }
    this.changed();
  }

  zoomAt(at, f) {                // around x = at; f > 1 zooms out
    const [a, b] = this.shown();
    this.zoomTo([at - (at - a) * f, at + (b - at) * f]);
  }

  pan(frac) {                    // move a zoomed window by a fraction of its width
    const z = this.view;
    if (z) this.zoomTo([z[0] + frac * (z[1] - z[0]), z[1] + frac * (z[1] - z[0])]);
  }

  changed() {                    // redraw its charts (or let the page redraw them: onView, e.g. the ride page with finer data)
    const mine = Object.entries(EChart.live).filter(([cid, c]) => (c.spec.group || cid) === this.key);
    const hook = mine.find(([, c]) => c.spec.onView)?.[1].spec;
    if (hook) hook.onView(this.view);
    else mine.forEach(([cid, c]) => EChart.draw(cid, c.spec));
  }

}
// A stretch selected on a chart group (spec.select = `sel` {range, onChange}): its numbers in an aside beside the
// charts (.selgrid > aside.selpanel), with Zoom to it / Clear. The page works the numbers out; draw(head, body)
// shows them (head null: the `empty` line)
export class SelectionPanel {
  constructor(id, group, sel, empty) { Object.assign(this, {id, group, sel, empty}); }

  draw(head, body = "") {
    const box = document.getElementById(this.id);
    if (!box) return;
    if (head == null || !this.sel.range) { box.innerHTML = `<div class="small muted">${this.empty}</div>`; return; }
    box.innerHTML = `<div class="selstats"><div>${head}</div>
    <div class="row" style="margin-top:6px"><button class="btn sm" data-sel="zoom">Zoom to it</button><button class="btn sm" data-sel="clear">Clear</button></div>${body}</div>`;
    box.querySelector('[data-sel="zoom"]').addEventListener("click", () => this.zoom());
    box.querySelector('[data-sel="clear"]').addEventListener("click", () => this.clear());
  }

  zoom() {
    const r = this.sel.range, pad = (r[1] - r[0]) * 0.15;
    TimeWindow.of(this.group).zoomTo([r[0] - pad, r[1] + pad]);
  }

  clear() {
    this.sel.range = null;
    (selectGroups[this.group] || []).forEach((h) => h.show(null));
    this.sel.onChange(null);
  }
}
// EChart.draw(id, spec): spec {x:[...], xType:"date"|"num"|"log", series:[{y, kind:"line"|"bar"|"dots"|"band", y2 (band
//   top), color, name, fmt, dash, x, width, area, markers, connect, range (false: doesn't set the y axis), opacity (bands),
//   hollow / alpha [per point] / lo, hi [per point bar] (dots), planned, tipName}], zero, yFmt, xTickFmt, xLabel, xTicks}
// optional: group (charts sharing zoom, crosshair and selection), scrollbar, view [a, b] (the window it opens on),
// yExtra [arrays on x] (also set the y axis, not drawn), xAxis false (no x labels: for charts stacked above one that
// has them), xDomain [a, b] (the full x range, when it isn't just the data's), vline / hline, shade, bands, planFrom,
// tipNote(x), tipReverse, onPick(x), onHover(x), onView, select, xStep, xName / yName (axis titles), yDomain [a, b]
// (a fixed y range, e.g. the same as xDomain for a predicted-vs-observed chart), tick (one tick step on both axes),
// label (what the chart shows, in a sentence: its text alternative for screen readers). A dots series may set shape
// ("circle" | "diamond" | "triangle" | "rect"), so its kind isn't told by colour alone.
// Drawn by Apache ECharts (static/vendor), which does the axes, ticks, marks, crosshair, tooltip box, zoom / pan (dataZoom: Ctrl + wheel / pinch zooms,
// drag or Shift + wheel pans, the slider is the scrollbar, double-click shows all). A group's charts (spec.group) share
// their x range, zoom and crosshair: passed on here (pointer, zoomed), not by ECharts' connect, which shows a pointer only
// in a chart the mouse is in. The zoom is kept in the TimeWindow of the chart (or its group), so range buttons and
// redraws keep it. select {range, onChange}: drag across selects a stretch (drag an edge to move it, Shift + drag pans,
// double-click clears it), shown on the group's charts (selectGroups) as an overlay box.
// onView(range | null): called once a zoom has settled (the ride page redraws with finer data); xStep(span, n): the
// x tick step for the span shown, about n ticks (xTickFmt gets it too); spec.yScale {y0, y1, top, bottom, H}: the y
// axis and plot as drawn, for a histogram beside the chart (the ride page's sideHist).
// A heatmap: rows [labels, top down] and a series {kind: "heat", cells: [{value: [x, row], itemStyle: {color}}], x:
// [first, last]}, coloured by the page; tipItem(x, row) is the hover's html for one cell.
export class EChart {
  static live = {};              // id -> EChart drawn (disposed once its box has left the page)
  static PICK_MS = 300;          // a click waits this long in case it is the first of a double-click
  static VIEW_MS = 150;          // onView waits this long after the last zoom step
  static viewT = 0;
  static EDGE = 6;               // px from a selection's edge that grab the edge
  static echo = false;           // a group's charts are following one of them (pointer, zoom): not passed on again
  static selDrag = null;         // a selection being dragged: {chart, fixed, px0, edge, moved} (followed on the window)

  static draw(id, spec) {
    const box = document.getElementById(id);
    if (!box) return;
    if (spec.label) { box.setAttribute("role", "img"); box.setAttribute("aria-label", spec.label); }
    for (const [k, c] of Object.entries(EChart.live)) if (!c.chart.getDom().isConnected) { c.chart.dispose(); delete EChart.live[k]; }
    const c = EChart.live[id];
    if (c && c.chart.getDom() === box) c.update(spec);
    else { c?.chart.dispose(); EChart.live[id] = new EChart(box, spec); }
  }

  static resizeAll() { Object.values(EChart.live).forEach((c) => { c.chart.resize(); c.selShow(c.spec.select?.range); }); }

  static selFollow(ev) {         // a selection drag: the stretch from where it began (or the edge held) to the pointer
    const g = EChart.selDrag;
    if (!g) return;
    const c = g.chart, sel = c.spec.select, px = ev.clientX - c.box.getBoundingClientRect().left;
    if (!g.edge && Math.abs(px - g.px0) < 4) return;   // a click, not a drag
    const [x0, x1] = c.shownX(), v = Math.max(x0, Math.min(x1, c.xAt(px)));
    sel.range = [Math.min(g.fixed, v), Math.max(g.fixed, v)];
    g.moved = true;
    c.selAll(sel.range);
  }

  static selDrop() {
    const g = EChart.selDrag;
    EChart.selDrag = null;
    if (g?.moved) g.chart.spec.select.onChange(g.chart.spec.select.range);
  }

  constructor(box, spec) {
    this.box = box;
    this.downX = null;
    this.pickT = 0;
    box.innerHTML = "";
    box.style.position = "relative";
    this.chart = init(box, null, {renderer: spec.rows ? "canvas" : "svg"});   // a heatmap's thousands of cells: canvas
    this.selBox = box.appendChild(document.createElement("div"));
    this.selBox.className = "esel";
    this.selBox.hidden = true;
    box.addEventListener("wheel", (ev) => this.wheel(ev), {capture: true});
    this.chart.on("datazoom", () => this.zoomed());
    const zr = this.chart.getZr();
    zr.on("mousedown", (e) => this.down(e));
    zr.on("mousemove", (e) => this.hover(e));
    zr.on("click", (e) => this.click(e));
    zr.on("dblclick", () => this.dblclick());
    this.chart.on("updateAxisPointer", (e) => this.pointer(e.axesInfo?.[0]?.value ?? null));
    zr.on("globalout", () => this.pointer(null));
    this.update(spec);
  }

  update(spec) {
    this.spec = spec;
    this.xv = (spec.x || []).map((v) => spec.xType === "date" ? new Date(v + "T12:00:00").getTime() : v);
    const allX = spec.series.flatMap((s) => this.xs(s)).filter((v) => v != null);
    const has = spec.series.some((s) => s.cells?.length || (s.y || []).some((v) => v != null));
    if (!has) { this.chart.clear(); this.empty(true); return; }
    this.empty(false);
    this.zoomable = (spec.xType === "date" || spec.xType === "num") && allX.length > 1;
    this.tw = this.zoomable ? TimeWindow.of(spec.group || this.box.id) : null;
    if (this.tw && spec.view && !this.tw.view && !this.tw.opened) { this.tw.view = spec.view; this.tw.opened = true; }
    const before = this.tw?.extent?.join();
    if (this.tw) this.tw.widen(spec.xDomain || [Math.min(...allX), Math.max(...allX)]);
    this.chart.setOption(this.option(allX), {notMerge: true});
    if (spec.group && before !== this.tw?.extent?.join()) this.widenGroup();
    const H = this.chart.getHeight(), top = this.gridBox.top, bottom = H - this.gridBox.bottom;
    spec.yScale = {y0: this.chart.convertFromPixel({yAxisIndex: 0}, bottom), y1: this.chart.convertFromPixel({yAxisIndex: 0}, top), top, bottom, H};
    if (spec.select) {
      const key = spec.group || this.box.id;
      selectGroups[key] = (selectGroups[key] || []).filter((h) => h.id !== this.box.id && document.getElementById(h.id));
      selectGroups[key].push({id: this.box.id, show: (r) => this.selShow(r)});
    }
    this.selShow(spec.select?.range);
  }

  // a group's charts share one x range (the TimeWindow's extent), so stacked charts line up: one drawn later with
  // more data widens the ones already drawn
  widenGroup() {
    const [min, max] = this.tw.extent;
    this.others().filter((c) => c.tw).forEach((c) => c.chart.setOption({xAxis: {min, max}}));
  }

  others() {                     // the other charts of this one's group
    return this.spec.group ? Object.values(EChart.live).filter((c) => c !== this && c.spec.group === this.spec.group) : [];
  }

  // run fn on the group's other charts once: what they do in turn (their own events) isn't passed on again
  toGroup(fn) {
    if (EChart.echo) return;
    EChart.echo = true;
    try {
      this.others().forEach(fn);
    } finally {
      EChart.echo = false;
    }
  }

  empty(on) {
    this.box.querySelector(".nodata")?.remove();
    if (on) this.box.insertAdjacentHTML("beforeend", '<div class="nodata muted small" style="position:absolute;inset:0;padding:30px 0;text-align:center">No data yet</div>');
  }

  xs(s) { return s.x || this.xv; }

  option(allX) {
    const spec = this.spec, ink2 = css("--ink2"), grid = css("--grid");
    const slider = this.hasSlider();
    const labelled = (spec.bands || []).some((b) => b[3]);
    const dom = this.tw?.extent || spec.xDomain || [Math.min(...allX), Math.max(...allX)];
    const series = spec.series.flatMap((s) => this.series(s));
    (spec.yExtra || []).forEach((y) => series.push({type: "line", data: this.pairs(this.xv, y), symbol: "none", silent: true,
      lineStyle: {opacity: 0}, tooltip: {show: false}}));
    const view = this.tw?.view;
    if (series.length) Object.assign(series[0], this.marks(view || dom));
    this.gridBox = {top: this.zoomable || labelled ? 20 : 6, bottom: (spec.xAxis === false ? 4 : 22) + (slider ? 24 : 0) + (spec.xName ? 20 : 0)};
    const title = (name, gap) => name ? {name, nameLocation: "middle", nameGap: gap, nameTextStyle: {color: ink2, fontSize: 11.5}} : {};
    const ticks = spec.xTicks ? new Map(spec.xTicks) : null;    // fixed ticks [[x, label], ...]
    const win = view || dom, step = spec.xStep ? spec.xStep(win[1] - win[0], Math.max(4, Math.floor((this.chart.getWidth() - 54) / 90))) : undefined;
    const H = this.chart.getHeight();
    return {
      animation: false,
      grid: {left: spec.yName ? 62 : 44, right: 10, ...this.gridBox},
      xAxis: {type: spec.xType === "date" ? "time" : spec.xType === "log" ? "log" : "value", min: dom[0], max: dom[1],
        axisLine: {show: false}, axisTick: {show: false}, splitLine: {show: false}, ...title(spec.xName, 26),
        ...(spec.tick ? {interval: spec.tick} : {}),
        axisLabel: {show: spec.xAxis !== false, color: ink2, fontSize: 11, hideOverlap: true,
          ...(step ? {customValues: this.stepValues(dom, step)} : {}),
          ...(spec.xTickFmt ? {formatter: (v) => spec.xTickFmt(v, step)} : {}),
          ...(ticks ? {customValues: [...ticks.keys()], formatter: (v) => ticks.get(v) ?? ""} : {})}},
      yAxis: spec.rows
        ? {type: "category", data: spec.rows, inverse: true, axisLine: {show: false}, axisTick: {show: false}, splitLine: {show: false},
          axisLabel: {color: ink2, fontSize: 11}}
        : {type: "value", scale: !(spec.zero || spec.series.some((s) => s.kind === "bar")), ...this.yRange(view),
          ...title(spec.yName, 46), ...(spec.tick ? {interval: spec.tick} : {}),
          splitNumber: H < 150 ? 3 : H < 260 ? 4 : 5,             // fewer y ticks on a short chart
          axisLine: {show: false}, axisTick: {show: false}, splitLine: {lineStyle: {color: grid}}, axisLabel: {color: ink2, fontSize: 11, formatter: (v) => (spec.yFmt || String)(v)}},
      tooltip: {trigger: spec.tipItem ? "item" : "axis", confine: true, className: "etip", transitionDuration: 0, padding: [6, 9],
        backgroundColor: css("--surface"), borderColor: css("--line"), borderWidth: 1, textStyle: {color: css("--ink"), fontSize: 12.5},
        extraCssText: "border-radius:9px;box-shadow:none;white-space:nowrap",
        axisPointer: {type: "line", lineStyle: {color: css("--muted"), type: [3, 3]}},
        formatter: spec.tipItem ? (p) => spec.tipItem(p.value[0], p.value[1]) : (p) => this.tip(p)},
      dataZoom: this.zoomable ? [
        {type: "inside", xAxisIndex: 0, zoomOnMouseWheel: "ctrl", moveOnMouseWheel: "shift", moveOnMouseMove: spec.select ? "shift" : true,
          startValue: view?.[0], endValue: view?.[1]},
        ...(slider ? [{type: "slider", xAxisIndex: 0, height: 12, bottom: 4, showDetail: false, brushSelect: false,
          borderColor: "transparent", backgroundColor: css("--surface2"), fillerColor: css("--line"),
          dataBackground: {lineStyle: {opacity: 0}, areaStyle: {opacity: 0}}, startValue: view?.[0], endValue: view?.[1]}] : [])] : [],
      series,
    };
  }

  // every whole `step` across the axis [a, b] (xStep's ticks): exact, so ECharts adds none at the ends
  stepValues([a, b], step) {
    const out = [];
    for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + 1e-9 && out.length < 5000; t += step) out.push(+t.toFixed(6));
    return out;
  }

  // the scrollbar (slider): always with spec.scrollbar, else on a chart of its own once it is zoomed
  hasSlider() {
    return this.zoomable && (!!this.spec.scrollbar || (!this.spec.group && !!this.tw?.view));
  }

  pairs(X, y) { return X.map((x, i) => [x, y?.[i] ?? null]); }

  // {min, max} of the y axis when a series has range: false (ECharts can't leave one out): from the other series
  // and yExtra in the x window shown, padded and rounded out to whole ticks as drawChart did; {} = ECharts' own
  yRange(view) {
    const spec = this.spec;
    if (spec.yDomain) return {min: spec.yDomain[0], max: spec.yDomain[1]};
    if (!spec.series.some((s) => s.range === false)) return {};
    const seen = (x) => !view || (x != null && x >= view[0] && x <= view[1]);
    const ys = [];
    for (const s of spec.series) {
      if (s.range === false) continue;
      const X = this.xs(s);
      for (const y of [s.y, s.y2]) (y || []).forEach((v, i) => { if (v != null && seen(X[i])) ys.push(v); });
    }
    (spec.yExtra || []).forEach((y) => y.forEach((v, i) => { if (v != null && seen(this.xv[i])) ys.push(v); }));
    if (!ys.length) return {};
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (spec.zero || spec.series.some((s) => s.kind === "bar")) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
    const pad = (y1 - y0) * 0.06 || 1;
    if (!(spec.zero && y0 === 0)) y0 -= pad;
    y1 += pad;
    const t = niceTicks(y0, y1, 5), step = t.length > 1 ? t[1] - t[0] : 1;
    return {min: Math.floor(y0 / step) * step, max: Math.ceil(y1 / step) * step};   // whole ticks at both ends
  }

  // one spec series as ECharts series: line (area, dash, markers, connect), bar, dots (hollow, alpha, lo / hi bars),
  // band (y .. y2 filled: a transparent lower line with the gap stacked on it)
  series(s) {
    const X = this.xs(s), col = s.color;
    if (s.kind === "band") {
      const stack = "band" + Math.random();
      const lo = X.map((x, i) => [x, s.y[i] != null && s.y2[i] != null ? s.y[i] : null]);
      const gap = X.map((x, i) => [x, s.y[i] != null && s.y2[i] != null ? s.y2[i] - s.y[i] : null]);
      const quiet = {type: "line", symbol: "none", silent: true, stack, stackStrategy: "all", tooltip: {show: false}, lineStyle: {opacity: 0}};
      return [{...quiet, data: lo}, {...quiet, data: gap, areaStyle: {color: col, opacity: s.opacity ?? 0.16}}];
    }
    if (s.kind === "heat") {        // cells [{value: [x, row], itemStyle: {color}}]: each x .. x + 1 by its row (spec.rows)
      return [{type: "custom", data: s.cells, clip: true, encode: {x: 0, y: 1}, renderItem: (_, api) => this.heatCell(api)}];
    }
    if (s.kind === "bar") {
      return [{type: "bar", data: this.pairs(X, s.y), barMaxWidth: 24,
        itemStyle: {color: col, opacity: s.opacity ?? 0.85, borderRadius: 2}}];
    }
    if (s.kind === "dots") {
      const data = X.flatMap((x, i) => s.y[i] == null ? [] : [{value: [x, s.y[i]],
        itemStyle: {opacity: s.alpha ? s.alpha[i] : s.r ? 0.75 : 1}}]);
      const out = [{type: "scatter", data, symbol: s.shape || "circle", symbolSize: 2 * (s.r || 4) * (s.shape && s.shape !== "circle" ? 1.25 : 1),
        itemStyle: s.hollow
        ? {color: "transparent", borderColor: col, borderWidth: 1.2}
        : {color: col, borderColor: s.r ? "transparent" : css("--surface"), borderWidth: s.r ? 0 : 1.5}}];
      if (s.lo) {
        const bars = X.flatMap((x, i) => s.y[i] == null || s.lo[i] == null ? [] : [[x, s.lo[i], s.hi[i], s.alpha ? s.alpha[i] : 1]]);
        out.push({type: "custom", data: bars, encode: {x: 0, y: [1, 2]}, silent: true, tooltip: {show: false},
          renderItem: (_, api) => this.errorBar(api, col)});
      }
      return out;
    }
    return [{type: "line", data: this.pairs(X, s.y), connectNulls: !!s.connect, showSymbol: !!s.markers, symbolSize: 5,
      itemStyle: {color: col}, lineStyle: {color: col, width: s.width || 2, type: s.dash ? [5, 4] : "solid"},
      ...(s.area ? {areaStyle: {color: col, opacity: 0.14}} : {})}];
  }

  heatCell(api) {
    const p = api.coord([api.value(0), api.value(1)]), [w, h] = api.size([1, 1]);
    return {type: "rect", shape: {x: p[0], y: p[1] - h / 2, width: w + 0.5, height: h + 0.5}, style: {fill: api.visual("color")}};
  }

  errorBar(api, col) {
    const x = api.value(0), a = api.coord([x, api.value(1)]), b = api.coord([x, api.value(2)]);
    return {type: "line", shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]}, style: {stroke: col, lineWidth: 1.2, opacity: api.value(3)}};
  }

  // vline / hline, shade [from, to] and bands [[from, to, colour, label?, opacity?, inside?], ...] on the first series;
  // a band's label only where it fits in the x window `win` shown (about 7 px a letter)
  marks(win) {
    const spec = this.spec, ink2 = css("--ink2"), bandOp = spec.bandOpacity ?? 0.22;
    const perX = (this.chart.getWidth() - 54) / (win[1] - win[0] || 1);
    const fits = (a, b, label) => !!label && (Math.min(b, win[1]) - Math.max(a, win[0])) * perX >= 7 * label.length;
    const lines = [];
    if (spec.vline != null) lines.push({xAxis: spec.vline, lineStyle: {color: ink2, width: 1, type: [2, 3]}});
    if (spec.hline != null) lines.push({yAxis: spec.hline, lineStyle: {color: ink2, width: 1, type: "solid"}});
    const areas = [...(spec.shade ? [[...spec.shade, css("--ink"), null, 0.06]] : []), ...(spec.bands || [])].map(([a, b, col, label, op, inside]) => [
      {xAxis: a, itemStyle: {color: col, opacity: op ?? bandOp},
        label: {show: fits(a, b, label), formatter: label || "", position: inside ? "insideTopLeft" : "top", align: "left", fontSize: 10.5,
          fontWeight: 600, color: inside ? css("--ink") : col}},
      {xAxis: b}]);
    return {
      ...(lines.length ? {markLine: {silent: true, symbol: "none", label: {show: false}, animation: false, data: lines}} : {}),
      ...(areas.length ? {markArea: {silent: true, animation: false, data: areas}} : {}),
    };
  }

  nearest(X, x) {
    const f = this.spec.xType === "log" ? Math.log : (v) => v;
    let bi = -1, bd = Infinity;
    X.forEach((v, i) => { if (v == null) return; const d = Math.abs(f(v) - f(x)); if (d < bd) { bd = d; bi = i; } });
    return bi;
  }

  // the tooltip: the date (or x), then each named series at that x (planFrom: after the last day of data the planned
  // series show as plain "X" - before it they're left out), then tipNote(x)
  tip(params) {
    const spec = this.spec, at = params?.[0]?.axisValue;
    const base = spec.series.find((q) => q.kind !== "band") || spec.series[0], BX = this.xs(base);
    const i = at == null ? -1 : this.nearest(BX, at);
    if (i < 0) return "";
    const xx = BX[i];
    const head = spec.xType === "date" ? niceDate(spec.x[i], {weekday: "short", day: "numeric", month: "short", year: "numeric"}) : (spec.xLabel ? spec.xLabel(xx) : xx);
    const ahead = spec.planFrom != null && xx > spec.planFrom;
    const shown = spec.series.filter((q) => q.name && q.kind !== "band" && (!q.planned || ahead));
    const lines = (spec.tipReverse ? shown.reverse() : shown).map((q) => {
      const j = q.x ? this.nearest(q.x, xx) : i, v = q.y[j];
      const name = q.tipName || (q.planned ? q.name.replace(/, planned$/, "") : q.name);
      return v == null ? "" : `<div><i style="background:${q.color}"></i>${esc(name)} <b>${esc((q.fmt || ((z) => fmt(z)))(v))}</b></div>`;
    }).join("");
    const note = spec.tipNote ? spec.tipNote(xx) : "";
    if (!lines && !note) return "";
    return `<div class="th">${esc(head)}${ahead ? ' <span class="muted">· planned</span>' : ""}</div>${lines}${note}`;
  }

  zoomed() {                     // the zoom ECharts shows -> the TimeWindow (null: all of it)
    if (!this.tw) return;
    const dz = this.chart.getOption().dataZoom?.[0], e = this.tw.extent;
    if (!dz || !e) return;
    const had = this.hasSlider();
    this.tw.view = dz.startValue <= e[0] && dz.endValue >= e[1] ? null : [dz.startValue, dz.endValue];
    if (had !== this.hasSlider()) { setTimeout(() => this.update(this.spec)); return; }   // the scrollbar comes or goes
    const y = this.yRange(this.tw.view);
    if (y.min != null) this.chart.setOption({yAxis: y});
    if ((this.spec.bands || []).some((b) => b[3])) this.chart.setOption({series: [{markArea: this.marks(this.shownX()).markArea}]});
    this.selShow(this.spec.select?.range);
    if (EChart.echo) return;                      // following another chart: it calls onView
    this.toGroup((c) => c.chart.dispatchAction({type: "dataZoom", startValue: dz.startValue, endValue: dz.endValue}));
    if (this.spec.onView) {
      clearTimeout(EChart.viewT);
      EChart.viewT = setTimeout(() => this.spec.onView(this.tw.view), EChart.VIEW_MS);
    }
  }

  // the crosshair at x (null: gone): onHover(x), and the group's other charts show theirs and their tooltip there
  pointer(x) {
    this.spec.onHover?.(x);
    this.toGroup((c) => c.showAt(x));
  }

  showAt(x) {
    if (x == null) { this.chart.dispatchAction({type: "hideTip"}); return; }
    this.chart.dispatchAction({type: "showTip", x: this.chart.convertToPixel({xAxisIndex: 0}, x), y: this.chart.getHeight() / 2});
  }

  // a plain wheel scrolls the page: ECharts' inside zoom takes every wheel over the plot (also with zoomOnMouseWheel
  // "ctrl"), so it never sees one without Ctrl (pinch) or Shift
  wheel(ev) {
    if (!(ev.ctrlKey || ev.shiftKey)) ev.stopPropagation();
  }

  showAll() {
    clearTimeout(this.pickT);
    if (this.zoomable) this.chart.dispatchAction({type: "dataZoom", start: 0, end: 100});
  }

  shownX() { return this.tw?.view || this.tw?.extent || [0, 1]; }

  xAt(px) { return this.chart.convertFromPixel({xAxisIndex: 0}, px); }

  dblclick() {                   // clears the selection (with select), else shows it all
    const sel = this.spec.select;
    if (!sel) { this.showAll(); return; }
    clearTimeout(this.pickT);
    sel.range = null;
    this.selAll(null);
    sel.onChange(null);
  }

  selAll(r) { (selectGroups[this.spec.group || this.box.id] || []).forEach((h) => h.show(r)); }

  selShow(r) {                   // the selected stretch [a, b] as a box over the plot (null: none)
    const [x0, x1] = this.shownX();
    if (!r || !this.spec.select || !this.gridBox || r[1] < x0 || r[0] > x1) { this.selBox.hidden = true; return; }
    const a = this.chart.convertToPixel({xAxisIndex: 0}, Math.max(x0, r[0]));
    const b = this.chart.convertToPixel({xAxisIndex: 0}, Math.min(x1, r[1]));
    Object.assign(this.selBox.style, {left: a + "px", width: Math.max(b - a, 1) + "px", top: this.gridBox.top + "px",
      height: (this.chart.getHeight() - this.gridBox.top - this.gridBox.bottom) + "px"});
    this.selBox.hidden = false;
  }

  edgeAt(px) {                   // 0 / 1: the selection's start / end edge is under px
    const r = this.spec.select?.range;
    return r ? [0, 1].find((k) => Math.abs(this.chart.convertToPixel({xAxisIndex: 0}, r[k]) - px) <= EChart.EDGE) : undefined;
  }

  down(e) {                      // a press: remembered for click vs drag; with select, a selection drag begins
    this.downX = e.offsetX;
    const sel = this.spec.select;
    if (!sel || e.event.button > 0 || e.event.shiftKey) return;     // Shift + drag: ECharts pans
    if (!this.chart.containPixel({gridIndex: 0}, [e.offsetX, e.offsetY])) return;
    const k = this.edgeAt(e.offsetX);
    EChart.selDrag = {chart: this, fixed: k != null ? sel.range[1 - k] : this.xAt(e.offsetX), px0: e.offsetX, edge: k != null, moved: false};
  }

  hover(e) {                     // the cursor: resize at a selection's edge, a crosshair where a drag selects
    if (!this.spec.select || EChart.selDrag) return;
    this.chart.getZr().setCursorStyle(this.edgeAt(e.offsetX) != null ? "ew-resize" : e.event.shiftKey ? "grab" : "crosshair");
  }

  click(e) {                     // a click (not a drag) in the plot: onPick(x), unless it becomes a double-click
    if (!this.spec.onPick || this.downX == null || Math.abs(e.offsetX - this.downX) > 4) return;
    if (!this.chart.containPixel({gridIndex: 0}, [e.offsetX, e.offsetY])) return;
    const x = this.chart.convertFromPixel({xAxisIndex: 0}, e.offsetX);
    clearTimeout(this.pickT);
    this.pickT = setTimeout(() => this.spec.onPick(x), EChart.PICK_MS);
  }
}
let rsT;
export const tail = (arr, n) => (arr || []).slice(Math.max(0, (arr || []).length - n));

// capture: ECharts' inside zoom stops the moves of every drag over a chart (also one it doesn't pan)
window.addEventListener("pointermove", (ev) => EChart.selFollow(ev), true);
window.addEventListener("pointerup", () => EChart.selDrop(), true);

window.addEventListener("resize", () => { clearTimeout(rsT); rsT = setTimeout(() => {
  EChart.resizeAll();
  if (router.route() === "ride" && document.getElementById("rmap")) rideMap.drawMap();
  if (calendarPage.calResize) calendarPage.calResize();
}, 150); });
