import { lineChart, PALETTE, Series } from "./charts";
import { el, getJSON } from "./types";

interface Rung { rung: number; mean_survival: (number | null)[]; saturated: (boolean | null)[];
  km: ({ t: number[]; S: number[]; censored: number } | null)[] | null; breaking_point: number | null; threshold: number | null; }
interface Stress { levels: number; n_scenarios: number; stressors: Record<string, { solvable_fraction: number[]; rungs: Record<string, Rung> }>;
  rules: Record<string, string>; familiar_survival: Record<string, number>; }

const NICE: Record<string, string> = { water: "Water sites remaining (and slower respawn)", cost: "Movement-cost multiplier", food: "Food sites remaining (and slower respawn)" };
const MARKERS = ["circle", "square", "diamond", "triangle"] as const;

export async function initStress(root: HTMLElement): Promise<void> {
  const d = await getJSON<Stress>("data/stress.json");
  root.append(el("p", { class: "note" }, `Rules fixed before the run — breaking point: ${d.rules.breaking_point}. Saturation: ${d.rules.saturation} (drawn hollow, never counted as a win). Unsolvable: ${d.rules.unsolvable} (shaded, never counted against a policy). ${d.n_scenarios} scenarios per level.`));
  for (const [st, res] of Object.entries(d.stressors)) {
    const h = el("h3", {}, NICE[st] ?? st);
    root.append(h);
    const xs = Array.from({ length: d.levels }, (_, i) => i);
    const series: Series[] = Object.entries(res.rungs).filter(([k]) => k !== "planner").map(([name, r], i) => ({
      name: `Rung ${r.rung}: ${name}`, xs, ys: r.mean_survival, color: PALETTE[i % PALETTE.length],
      marker: MARKERS[i % MARKERS.length], hollowAt: r.saturated.map((x) => !!x),
    }));
    const unsolv = res.solvable_fraction.map((f, i) => ({ f, i })).filter((x) => x.f === 0).map((x) => ({ x0: x.i - 0.5, x1: x.i + 0.5, label: "unsolvable" }));
    lineChart(root, series, { title: `Survival vs intensity: ${NICE[st] ?? st}`, xLabel: "Stress level (0 = training-typical, 9 = hardest)",
      yLabel: "Mean steps survived (max 500)", yMin: 0, yMax: 520, xTicks: xs, hlines: [{ y: 500, label: "Rung 9: planner ceiling" }], shade: unsolv });
    const bp = el("table", { class: "metric-strip" }, el("tr", {}, el("th", {}, "Policy"), el("th", {}, "Familiar survival"), el("th", {}, "Threshold (½)"), el("th", {}, "Breaking point")));
    for (const [name, r] of Object.entries(res.rungs)) {
      if (name === "planner") continue;
      bp.append(el("tr", {}, el("td", {}, name), el("td", {}, (d.familiar_survival[name] ?? NaN).toFixed(1)),
        el("td", {}, r.threshold === null ? "—" : r.threshold.toFixed(1)), el("td", {}, r.breaking_point === null ? "none within 10 levels" : `level ${r.breaking_point}`)));
    }
    root.append(bp);
    // Kaplan-Meier
    const names = Object.keys(res.rungs).filter((k) => res.rungs[k].km);
    const selL = el("select", { "aria-label": "Level" });
    for (let i = 0; i < d.levels; i++) selL.append(el("option", { value: String(i) }, `level ${i}`));
    const kmHost = el("div");
    root.append(el("label", {}, "Kaplan-Meier at ", selL), kmHost);
    const drawKM = (): void => {
      kmHost.innerHTML = "";
      const L = Number(selL.value);
      const ks: Series[] = names.flatMap((n, i) => {
        const k = res.rungs[n].km![L];
        return k ? [{ name: `${n} (censored ${(k.censored * 100).toFixed(0)}%)`, xs: k.t, ys: k.S, color: PALETTE[i % PALETTE.length] }] : [];
      });
      if (ks.length) lineChart(kmHost, ks, { title: `Kaplan-Meier survival, ${NICE[st] ?? st}, level ${L}`, xLabel: "Environment step",
        yLabel: "P(still alive)", yMin: 0, yMax: 1, xTicks: [0, 100, 200, 300, 400, 500] });
      else kmHost.append(el("p", { class: "note" }, "No solvable scenarios at this level."));
    };
    selL.onchange = drawKM;
    drawKM();
  }
}
