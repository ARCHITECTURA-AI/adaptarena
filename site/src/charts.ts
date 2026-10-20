import { el } from "./types";

export interface Series {
  name: string;
  xs: number[];
  ys: (number | null)[];
  color: string;
  dash?: string;
  marker?: "circle" | "square" | "diamond" | "triangle";
  hollowAt?: boolean[];   // saturated points are drawn hollow
  band?: { lo: (number | null)[]; hi: (number | null)[] };
}

export interface ChartOpts {
  title: string;
  xLabel: string;
  yLabel: string;
  yMin?: number;
  yMax?: number;
  xTicks?: number[];
  vlines?: { x: number; label: string }[];
  hlines?: { y: number; label: string; dash?: string }[];
  shade?: { x0: number; x1: number; label: string }[];
  width?: number;
  height?: number;
}

const NS = "http://www.w3.org/2000/svg";
function s<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

export const PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)", "var(--c8)"];

function markerPath(kind: string, x: number, y: number, hollow: boolean, color: string): SVGElement {
  let n: SVGElement;
  if (kind === "square") n = s("rect", { x: x - 4, y: y - 4, width: 8, height: 8 });
  else if (kind === "diamond") n = s("path", { d: `M${x} ${y - 5}L${x + 5} ${y}L${x} ${y + 5}L${x - 5} ${y}Z` });
  else if (kind === "triangle") n = s("path", { d: `M${x} ${y - 5}L${x + 5} ${y + 4}L${x - 5} ${y + 4}Z` });
  else n = s("circle", { cx: x, cy: y, r: 4 });
  n.setAttribute("fill", hollow ? "var(--bg)" : color);
  n.setAttribute("stroke", color);
  n.setAttribute("stroke-width", "1.6");
  return n;
}

/** A small dependency-free line chart plus a data table behind it (details/summary). */
export function lineChart(host: HTMLElement, series: Series[], o: ChartOpts): void {
  const W = o.width ?? 640, H = o.height ?? 340, m = { l: 56, r: 16, t: 34, b: 48 };
  const xs = series.flatMap((x) => x.xs);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ys = series.flatMap((x) => x.ys.filter((v): v is number => v !== null));
  const y0 = o.yMin ?? Math.min(0, ...ys), y1 = o.yMax ?? Math.max(...ys) * 1.05;
  const px = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const py = (y: number) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": `${o.title}. Data table below.` });
  svg.append(Object.assign(s("title"), { textContent: o.title }));
  const t = s("text", { x: m.l, y: 18, class: "chart-title" });
  t.textContent = o.title;
  svg.append(t);
  for (const sh of o.shade ?? []) {
    const r = s("rect", { x: px(sh.x0), y: m.t, width: Math.max(0, px(sh.x1) - px(sh.x0)), height: H - m.t - m.b, class: "shade" });
    svg.append(r);
    const lt = s("text", { x: px(sh.x0) + 4, y: m.t + 12, class: "tick" });
    lt.textContent = sh.label;
    svg.append(lt);
  }
  const yt = 5;
  for (let i = 0; i <= yt; i++) {
    const v = y0 + ((y1 - y0) * i) / yt;
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: py(v), y2: py(v), class: "gridline" }));
    const tx = s("text", { x: m.l - 6, y: py(v) + 4, class: "tick", "text-anchor": "end" });
    tx.textContent = Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2);
    svg.append(tx);
  }
  for (const v of o.xTicks ?? [x0, x1]) {
    const tx = s("text", { x: px(v), y: H - m.b + 16, class: "tick", "text-anchor": "middle" });
    tx.textContent = String(v);
    svg.append(tx);
  }
  svg.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }));
  svg.append(s("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" }));
  const xl = s("text", { x: (m.l + W - m.r) / 2, y: H - 8, class: "axis-label", "text-anchor": "middle" });
  xl.textContent = o.xLabel;
  const yl = s("text", { x: 14, y: (m.t + H - m.b) / 2, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
  yl.textContent = o.yLabel;
  svg.append(xl, yl);
  for (const h of o.hlines ?? []) {
    const l = s("line", { x1: m.l, x2: W - m.r, y1: py(h.y), y2: py(h.y), class: "refline", "stroke-dasharray": h.dash ?? "5 4" });
    svg.append(l);
    const lt = s("text", { x: W - m.r - 2, y: py(h.y) - 4, class: "tick", "text-anchor": "end" });
    lt.textContent = h.label;
    svg.append(lt);
  }
  for (const v of o.vlines ?? []) {
    svg.append(s("line", { x1: px(v.x), x2: px(v.x), y1: m.t, y2: H - m.b, class: "refline", "stroke-dasharray": "2 3" }));
    const lt = s("text", { x: px(v.x) + 3, y: m.t + 24, class: "tick" });
    lt.textContent = v.label;
    svg.append(lt);
  }
  for (const se of series) {
    if (se.band) {
      let d = "";
      const pts: [number, number][] = [];
      se.xs.forEach((x, i) => { const hi = se.band!.hi[i]; if (hi !== null) pts.push([px(x), py(hi)]); });
      const back: [number, number][] = [];
      se.xs.forEach((x, i) => { const lo = se.band!.lo[i]; if (lo !== null) back.push([px(x), py(lo)]); });
      d = "M" + [...pts, ...back.reverse()].map((p) => p.join(" ")).join("L") + "Z";
      const b = s("path", { d, class: "band" });
      b.setAttribute("fill", se.color);
      svg.append(b);
    }
    let d = "";
    let pen = false;
    se.xs.forEach((x, i) => {
      const y = se.ys[i];
      if (y === null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${px(x).toFixed(1)} ${py(y).toFixed(1)}`;
      pen = true;
    });
    const path = s("path", { d, fill: "none", stroke: se.color, "stroke-width": 2 });
    if (se.dash) path.setAttribute("stroke-dasharray", se.dash);
    svg.append(path);
    if (se.marker) se.xs.forEach((x, i) => {
      const y = se.ys[i];
      if (y !== null) svg.append(markerPath(se.marker!, px(x), py(y), !!se.hollowAt?.[i], se.color));
    });
  }
  const wrap = el("figure", { class: "chart-wrap" });
  wrap.append(svg);
  const legend = el("div", { class: "legend" });
  for (const se of series) {
    const sw = el("span", { class: "swatch" });
    sw.style.background = se.color;
    legend.append(el("span", { class: "legend-item" }, sw, se.name));
  }
  wrap.append(legend);
  const det = el("details", { class: "data-table" }, el("summary", {}, `Data table: ${o.title}`));
  const tbl = el("table");
  const head = el("tr", {}, el("th", {}, o.xLabel));
  for (const se of series) head.append(el("th", {}, se.name));
  tbl.append(head);
  const allx = Array.from(new Set(series.flatMap((x) => x.xs))).sort((a, b) => a - b);
  for (const x of allx) {
    const row = el("tr", {}, el("td", {}, String(x)));
    for (const se of series) {
      const i = se.xs.indexOf(x);
      row.append(el("td", {}, i >= 0 && se.ys[i] !== null ? (se.ys[i] as number).toFixed(3) : "—"));
    }
    tbl.append(row);
  }
  det.append(tbl);
  wrap.append(det);
  host.append(wrap);
}

export function heatmap(host: HTMLElement, data: number[][], title: string): void {
  const n = data.length, cell = 34;
  const max = Math.max(...data.flat(), 1e-9);
  const svg = s("svg", { viewBox: `0 0 ${n * cell} ${n * cell + 24}`, class: "chart heat", role: "img", "aria-label": `${title}. Data table below.` });
  data.forEach((row, y) => row.forEach((v, x) => {
    const r = s("rect", { x: x * cell, y: y * cell + 24, width: cell - 1, height: cell - 1 });
    r.setAttribute("fill", "var(--accent)");
    r.setAttribute("fill-opacity", String(0.05 + 0.95 * (v / max)));
    svg.append(r);
    const t = s("text", { x: x * cell + cell / 2, y: y * cell + 24 + cell / 2 + 3, class: "tick", "text-anchor": "middle" });
    t.textContent = v.toFixed(2);
    svg.append(t);
  }));
  const tt = s("text", { x: 0, y: 14, class: "chart-title" });
  tt.textContent = title;
  svg.append(tt);
  const wrap = el("figure", { class: "chart-wrap" });
  wrap.append(svg);
  const det = el("details", { class: "data-table" }, el("summary", {}, `Data table: ${title}`));
  const tbl = el("table");
  data.forEach((row) => tbl.append(el("tr", {}, ...row.map((v) => el("td", {}, v.toFixed(3))))));
  det.append(tbl);
  wrap.append(det);
  host.append(wrap);
}
