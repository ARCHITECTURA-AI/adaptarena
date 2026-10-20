import { Arena, Timeline } from "./arena";
import { el, getJSON, Replay } from "./types";

interface Entry { label: string; survival: number; cause: string; point_of_no_return: { step: number | null; valid_for?: string };
  blind_or_blunder: string | null; narration: string[]; replay: Replay; }

export async function initMuseum(root: HTMLElement): Promise<void> {
  const d = await getJSON<{ policy: string; entries: Entry[]; selection: string }>("data/museum.json");
  root.append(el("p", { class: "note" }, `Policy: ${d.policy}. ${d.selection}. Every scenario below was certified solvable by the reference planner, so each failure is verified.`));
  d.entries.forEach((e, i) => {
    const card = el("section", { class: "card" }, el("h3", {}, `Exhibit ${i + 1}: died of ${e.cause} at step ${e.survival}`));
    const host = el("div", { class: "arena-grid one" });
    card.append(host);
    const tl = new Timeline(card, e.replay.frames.length - 1);
    const a = new Arena(host);
    a.set(e.replay, "observer");
    a.showProbs = true;
    tl.arenas.push(a);
    const toggle = el("button", { type: "button", class: "btn" }, "Switch to agent view");
    toggle.onclick = () => { a.mode = a.mode === "observer" ? "agent" : "observer"; toggle.textContent = a.mode === "observer" ? "Switch to agent view" : "Switch to observer view"; tl.seek(tl.t); };
    card.append(toggle);
    const pnr = e.point_of_no_return.step;
    if (pnr !== null) {
      const jump = el("button", { type: "button", class: "btn" }, `Jump to point of no return (step ${pnr})`);
      jump.onclick = () => tl.seek(pnr);
      card.append(jump, el("p", { class: "note" }, `Point of no return: step ${pnr}. Label: ${e.blind_or_blunder} (heuristic suggestion). Valid for the planner's search power, not a proof that no plan exists.`));
    }
    const ol = el("ol", { class: "narration" });
    e.narration.forEach((s) => ol.append(el("li", {}, s)));
    card.append(el("h4", {}, "Templated narration (no language model; every sentence traces to a logged value)"), ol);
    root.append(card);
    tl.seek(0);
  });
}
