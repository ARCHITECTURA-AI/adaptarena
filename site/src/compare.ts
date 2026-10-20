import { Arena, Timeline } from "./arena";
import { el, getJSON, Replay } from "./types";

interface Index {
  scenarios: { id: string; label: string }[];
  interventions: { id: string; label: string; step: number | null }[];
  policies: { id: string; label: string; rung: number }[];
}

export async function initCompare(root: HTMLElement): Promise<void> {
  const idx = await getJSON<Index>("data/compare/index.json");
  const controls = el("div", { class: "controls" });
  const selScn = el("select", { "aria-label": "Scenario" });
  idx.scenarios.forEach((s) => selScn.append(el("option", { value: s.id }, s.label)));
  const selIv = el("select", { "aria-label": "Intervention" });
  idx.interventions.forEach((i) => selIv.append(el("option", { value: i.id }, i.label)));
  const selView = el("select", { "aria-label": "View" });
  selView.append(el("option", { value: "observer" }, "Observer view (full map)"), el("option", { value: "agent" }, "Agent view (what the policy sees)"));
  controls.append(el("label", {}, "Scenario ", selScn), el("label", {}, "Intervention ", selIv), el("label", {}, "View ", selView));
  root.append(controls);
  const grid = el("div", { class: "arena-grid" });
  root.append(grid);
  const tl = new Timeline(root, 500);
  const strip = el("table", { class: "metric-strip" });
  root.append(strip);
  const defaults = ["random", "memory_script", idx.policies.find((p) => p.id.startsWith("ppo_ff_armD"))?.id ?? idx.policies[idx.policies.length - 1].id];
  const arenas: { arena: Arena; sel: HTMLSelectElement }[] = [];
  for (let i = 0; i < 3; i++) {
    const sel = el("select", { "aria-label": `Policy for arena ${i + 1}` });
    idx.policies.forEach((p) => sel.append(el("option", { value: p.id }, p.label)));
    sel.value = defaults[i];
    const holder = el("div", { class: "arena-col" }, sel);
    grid.append(holder);
    const a = new Arena(holder);
    a.showProbs = true;
    arenas.push({ arena: a, sel });
    tl.arenas.push(a);
  }
  let cache: Record<string, Replay> = {};
  const refresh = (): void => {
    const mode = selView.value as "observer" | "agent";
    arenas.forEach(({ arena, sel }) => {
      const rep = cache[sel.value];
      arena.set(rep ?? null, mode);
      arena.title.textContent = idx.policies.find((p) => p.id === sel.value)?.label ?? "";
      const learned = sel.value.startsWith("ppo");
      arena.title.append(learned ? el("span", { class: "badge" }, "Frozen policy") : "");
    });
    tl.seek(tl.t);
    strip.innerHTML = "";
    strip.append(el("tr", {}, el("th", {}, "Arena"), el("th", {}, "Policy"), el("th", {}, "Steps survived"), el("th", {}, "Outcome")));
    arenas.forEach(({ sel }, i) => {
      const r = cache[sel.value];
      if (!r) return;
      strip.append(el("tr", {}, el("td", {}, String(i + 1)), el("td", {}, sel.options[sel.selectedIndex].text),
        el("td", {}, String(r.survival)), el("td", {}, r.alive ? "survived the 500-step horizon" : `died of ${r.cause}`)));
    });
  };
  const load = async (): Promise<void> => {
    cache = await getJSON<Record<string, Replay>>(`data/compare/${selScn.value}__${selIv.value}.json`);
    refresh();
  };
  selScn.onchange = load;
  selIv.onchange = load;
  selView.onchange = refresh;
  arenas.forEach(({ sel }) => (sel.onchange = refresh));
  await load();
}
