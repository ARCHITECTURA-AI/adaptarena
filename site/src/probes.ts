import { Arena, Timeline } from "./arena";
import { heatmap, lineChart, PALETTE } from "./charts";
import { el, getJSON, Replay } from "./types";

interface TreeNode { feature: string | null; threshold: number; left: number; right: number; class: number | null; n: number }
interface Probes { policies: Record<string, {
  hydration_threshold: { levels: number[]; p_head_for_water: number[]; n_states: number; reading: string };
  distilled_tree: { agreement_heldout: number; tree_survival: number; policy_survival: number; nodes: TreeNode[]; action_names: string[]; reading: string };
  occlusion: { heatmap: number[][]; n_states: number; reading: string };
  time_machine: { checkpoints: { pct: number; survival: number; cause: string; replay: Replay }[]; reading: string };
}>; script_overlay: { reactive_threshold: number } }

function treeList(nodes: TreeNode[], names: string[], i: number): HTMLElement {
  const n = nodes[i];
  if (n.class !== null) return el("li", {}, `→ ${names[n.class]} (n=${n.n})`);
  const li = el("li", {}, `${n.feature} ≤ ${n.threshold.toFixed(2)}?`);
  const ul = el("ul", {});
  ul.append(el("li", {}, "yes: ", el("ul", {}, treeList(nodes, names, n.left))), el("li", {}, "no: ", el("ul", {}, treeList(nodes, names, n.right))));
  li.append(ul);
  return li;
}

export async function initProbes(root: HTMLElement): Promise<void> {
  const d = await getJSON<Probes>("data/probes.json");
  const names = Object.keys(d.policies);
  const sel = el("select", { "aria-label": "Policy" });
  names.forEach((n) => sel.append(el("option", { value: n }, n.replace("arm_", "Arm "))));
  root.append(el("label", {}, "Policy (seed 0) ", sel));
  const body = el("div");
  root.append(body);
  const draw = (): void => {
    body.innerHTML = "";
    const p = d.policies[sel.value];
    body.append(el("h3", {}, "Hydration threshold"));
    lineChart(body, [{ name: "Learned policy: P(heads for visible water)", xs: p.hydration_threshold.levels, ys: p.hydration_threshold.p_head_for_water, color: PALETTE[1], marker: "circle" }],
      { title: "Does the policy head for water as hydration drops?", xLabel: "Hydration stat overwritten to (0–100)", yLabel: "P(moves toward visible water)", yMin: 0, yMax: 1,
        xTicks: [0, 20, 40, 60, 80, 100], vlines: [{ x: d.script_overlay.reactive_threshold, label: "reactive script rule: hydration < 40" }] });
    body.append(el("p", { class: "note" }, `Read it correctly: ${p.hydration_threshold.reading} (${p.hydration_threshold.n_states} states).`));
    body.append(el("h3", {}, "Distilled depth-3 tree"));
    const dt = p.distilled_tree;
    body.append(el("p", {}, `Agrees with the policy on ${(dt.agreement_heldout * 100).toFixed(1)}% of held-out states. Tree survival ${dt.tree_survival.toFixed(1)} vs policy ${dt.policy_survival.toFixed(1)} steps. ${dt.reading}.`));
    body.append(el("ul", { class: "tree" }, treeList(dt.nodes, dt.action_names, 0)));
    body.append(el("h3", {}, "Occlusion sensitivity"));
    heatmap(body, p.occlusion.heatmap, "Action-distribution shift when each tile is masked (agent at centre)");
    body.append(el("p", { class: "note" }, `Read it correctly: ${p.occlusion.reading}.`));
    body.append(el("h3", {}, "Training time machine"));
    const tmHost = el("div", { class: "arena-grid four" });
    body.append(tmHost);
    const tl = new Timeline(body, 500);
    for (const c of p.time_machine.checkpoints) {
      const holder = el("div", { class: "arena-col" });
      tmHost.append(holder);
      const a = new Arena(holder);
      a.title.textContent = `${c.pct}% of training`;
      a.title.append(el("span", { class: "badge" }, "Frozen policy"));
      a.set(c.replay, "observer");
      tl.arenas.push(a);
    }
    tl.seek(0);
    body.append(el("p", { class: "note" }, `Read it correctly: ${p.time_machine.reading}.`));
  };
  sel.onchange = draw;
  draw();
}
