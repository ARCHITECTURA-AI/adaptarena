import { initCompare } from "./compare";
import { initLedger } from "./ledger";
import { initMuseum } from "./museum";
import { initPlay } from "./play";
import { initProbes } from "./probes";
import { initStress } from "./stress";
import { el } from "./types";

const pages: Record<string, (root: HTMLElement) => Promise<void>> = {
  play: initPlay, compare: initCompare, stress: initStress, probes: initProbes, museum: initMuseum, ledger: initLedger,
};

const root = document.getElementById("app");
const page = document.body.dataset.page ?? "";
if (root && pages[page]) {
  pages[page](root).catch((e: Error) => root.append(el("p", { class: "error", role: "alert" }, `Could not load this page's data: ${e.message}`)));
}
