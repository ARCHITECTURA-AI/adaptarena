import { el, getJSON } from "./types";

interface Claim { id: string; claim: string; status: string; pre_declared: boolean; plan?: string; tier?: string; evidence: unknown; link: string }
interface Ledger { claims: Claim[]; plan_hash: string; git_commit: string | null; tier: string; decision: { outcome: string; text: string; saturated_conditions: string[] };
  predictions: { shift: string; predicted: string; status: string }[]; amendments: unknown[]; access_log: unknown[]; ladder: Record<string, Record<string, number>>; notes: string[] }

const ICON: Record<string, string> = { supported: "✔ supported", mixed: "◐ mixed", "not supported": "✖ not supported", hit: "✔ hit", partial: "◐ partial", miss: "✖ miss", descriptive: "· descriptive" };

export async function initLedger(root: HTMLElement): Promise<void> {
  const d = await getJSON<Ledger>("data/ledger.json");
  root.append(el("p", {}, `Tier: ${d.tier}. Analysis plan SHA-256 ${d.plan_hash}${d.git_commit ? ` (committed in git at ${d.git_commit.slice(0, 10)} before the locked suite existed)` : ""}.`));
  root.append(el("p", { class: "decision" }, `Pre-declared outcome: ${d.decision.text}`));
  const t = el("table", { class: "ledger" }, el("tr", {}, el("th", {}, "Claim"), el("th", {}, "Plan"), el("th", {}, "Tier"), el("th", {}, "Pre-declared"), el("th", {}, "Status"), el("th", {}, "Evidence")));
  for (const c of d.claims) {
    const ev = el("details", {}, el("summary", {}, "numbers"), el("pre", {}, JSON.stringify(c.evidence, null, 1)));
    ev.append(el("a", { href: `data/${c.link}` }, c.link));
    t.append(el("tr", {}, el("td", {}, c.claim), el("td", {}, c.plan ?? "v2"), el("td", { class: `tier ${c.tier ?? ""}` }, c.tier ?? ""), el("td", {}, c.pre_declared ? "yes" : "no"), el("td", { class: `status ${c.status.replace(/[ ()]/g, "-")}` }, ICON[c.status] ?? c.status), el("td", {}, ev)));
  }
  root.append(t);
  if (d.predictions?.length) {
    root.append(el("h3", {}, "Memory-track predictions, scored"));
    const p = el("table", { class: "ledger" }, el("tr", {}, el("th", {}, "Shift"), el("th", {}, "Predicted (locked before the run)"), el("th", {}, "Result")));
    d.predictions.forEach((x) => p.append(el("tr", {}, el("td", {}, x.shift), el("td", {}, x.predicted), el("td", {}, x.status))));
    root.append(p);
  }
  root.append(el("h3", {}, "Locked analysis plan"), el("p", {}, el("a", { href: "data/analysis_plan.yaml" }, "analysis_plan.yaml"), ` · amendments: ${d.amendments.length} · test-suite accesses logged: ${d.access_log.length}`));
  root.append(el("details", {}, el("summary", {}, "Access log"), el("pre", {}, d.access_log.map((x) => JSON.stringify(x)).join("\n"))));
  for (const n of d.notes) root.append(el("p", { class: "note" }, n));
}
