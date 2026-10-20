import { Arena, Timeline } from "./arena";
import { drawCodes } from "./draw";
import { el, getJSON, Replay, reducedMotion, STAT_MAX } from "./types";

interface Challenge { id: string; label: string; spec: Record<string, unknown>; rungs: Record<string, number>; layout_hash: string }
interface Ladder { challenges: Challenge[]; rung_labels: Record<string, string>; wheel: string; pyodide: string }
interface Frame { cells: number[]; energy: number; hyd: number; cold: number; t: number; done: boolean; alive: boolean; survived: number; cause: string; valid: number }

const GLUE = `
import json
import numpy as np
from adaptarena.core import SurvivalEnv, ScenarioSpec
from adaptarena.lab.replays import replay_from_actions
_env = SurvivalEnv()
_spec = None
def _pack(obs, info=None):
    g = obs["grid"]
    code = ((g[0] > 0) * 1 + (g[1] > 0) * 2 + (g[2] > 0) * 4 + (g[3] > 0) * 8 + (g[4] > 0) * 16 + (g[5] > 0) * 32).astype(int)
    return json.dumps({"cells": [int(v) for v in code.reshape(-1)], "energy": int(_env.energy), "hyd": int(_env.hydration),
        "cold": int(obs["stats"][2]), "t": int(_env.t), "done": bool(_env.done), "alive": bool(_env.alive),
        "survived": int(_env.survived), "cause": __import__("adaptarena.core.params", fromlist=["x"]).primary_cause(_env.flags),
        "valid": int(info["valid"]) if info else 1})
def layout_hash(spec_json):
    import hashlib
    from adaptarena.core import build_scenario
    sc = build_scenario(ScenarioSpec.from_json(json.loads(spec_json)))
    h = hashlib.sha256()
    for a in (sc.terrain, sc.food_pos, sc.water_pos, np.array(sc.start), sc.food_delay, sc.water_delay, sc.cold):
        h.update(np.ascontiguousarray(a, dtype="<i4").tobytes())
    return h.hexdigest()
def reset(spec_json):
    global _spec
    _spec = ScenarioSpec.from_json(json.loads(spec_json))
    return _pack(_env.reset(_spec))
def step(a):
    obs, r, te, tr, info = _env.step(int(a))
    return _pack(obs, info)
def observer_replay(actions_json):
    return json.dumps(replay_from_actions(_spec, json.loads(actions_json), "You"))
`;

const KEYS: Record<string, number> = { ArrowUp: 0, w: 0, W: 0, ArrowDown: 1, s: 1, S: 1, ArrowLeft: 2, a: 2, A: 2, ArrowRight: 3, d: 3, D: 3, e: 4, E: 4, " ": 4, x: 5, X: 5, q: 5, Q: 5 };

declare global { interface Window { loadPyodide?: (o: { indexURL: string }) => Promise<any>; ADAPTARENA_PYODIDE?: string } }

function loadScript(src: string): Promise<void> {
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = () => res(); s.onerror = () => rej(new Error(`could not load ${src}`)); document.head.append(s); });
}

function bestKey(id: string): string { return `aa-best-${id}`; }
function readBest(id: string): number | null { try { const v = localStorage.getItem(bestKey(id)); return v === null ? null : Number(v); } catch { return null; } }
function writeBest(id: string, v: number): void { try { localStorage.setItem(bestKey(id), String(v)); } catch { /* ignore */ } }

export async function initPlay(root: HTMLElement): Promise<void> {
  const ladder = await getJSON<Ladder>("data/ladder.json");
  const status = el("div", { class: "loading", role: "status", "aria-live": "polite" }, "Loading Python (Pyodide) — this runs the same simulator that trains the agents, in your browser. No server, no account.");
  root.append(status);
  const sel = el("select", { "aria-label": "Challenge scenario" });
  ladder.challenges.forEach((c) => sel.append(el("option", { value: c.id }, c.label)));
  const rm = el("input", { type: "checkbox", id: "rm" }) as HTMLInputElement;
  try { rm.checked = localStorage.getItem("aa-reduced-motion") === "1"; } catch { /* ignore */ }
  rm.onchange = () => { try { localStorage.setItem("aa-reduced-motion", rm.checked ? "1" : "0"); } catch { /* ignore */ } };
  const controls = el("div", { class: "controls" }, el("label", {}, "Challenge ", sel), el("label", {}, rm, " Reduced motion"));
  const game = el("div", { class: "game", tabindex: "0", role: "application", "aria-label": "AdaptArena game. Arrow keys or WASD move, E or Space interact, X waits." });
  const canvas = el("canvas", { role: "img", "aria-label": "Your view" });
  const hud = el("div", { class: "hud" });
  const help = el("p", { class: "note" }, "Fog of war: you see exactly what the policies see — a 9×9 window around you. Arrows/WASD move · E or Space drink or eat on a ready site · X wait. Water = circle W, food = triangle F, wall = hatch. Each key press is one environment step; the episode ends at step 500 or when you die.");
  game.append(canvas, hud);
  root.append(controls, help, game);
  const result = el("div", { class: "result", "aria-live": "polite" });
  root.append(result);

  // --- load Pyodide lazily, only on this page
  const t0 = performance.now();
  let py: any = null;
  let loadFailed = false;
  try {
    const base = window.ADAPTARENA_PYODIDE ?? ladder.pyodide;
    await loadScript(`${base}pyodide.js`);
    py = await window.loadPyodide!({ indexURL: base });
    await py.loadPackage("numpy");
    const buf = await (await fetch(`py/${ladder.wheel}`)).arrayBuffer();
    py.unpackArchive(buf, "wheel");
    await py.runPythonAsync(GLUE);
    const bytes = performance.getEntriesByType("resource").reduce((a, r) => a + ((r as PerformanceResourceTiming).transferSize || 0), 0);
    status.textContent = `Ready. Pyodide + numpy + the AdaptArena core loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s (${(bytes / 1e6).toFixed(1)} MB transferred).`;
  } catch (e) {
    loadFailed = true;
    status.textContent = `Could not load the in-browser simulator (${(e as Error).message}). Showing a curated replay of a trained policy instead.`;
  }
  if (loadFailed) {
    const rep = await getJSON<Replay>("data/fallback_replay.json");
    const host = el("div", { class: "arena-grid one" });
    root.append(host);
    const tl = new Timeline(root, rep.frames.length - 1);
    const a = new Arena(host);
    a.set(rep, "agent");
    tl.arenas.push(a);
    tl.seek(0);
    return;
  }

  let actions: number[] = [];
  let frame: Frame;
  let challenge: Challenge = ladder.challenges[0];
  const render = (): void => {
    drawCodes(canvas, frame.cells, !frame.alive, frame.cold === 1);
    hud.innerHTML = "";
    const row = (name: string, v: number): HTMLElement => {
      const track = el("div", { class: "bar-track", role: "progressbar", "aria-label": name, "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(v * 100)) });
      const fill = el("div", { class: "bar-fill" });
      fill.style.width = `${Math.round(v * 100)}%`;
      track.append(fill);
      return el("div", { class: "bar-row" }, el("span", { class: "bar-name" }, name), track, el("span", { class: "bar-num" }, String(Math.round(v * 100))));
    };
    hud.append(row("Energy", Math.max(0, frame.energy) / STAT_MAX), row("Hydration", Math.max(0, frame.hyd) / STAT_MAX));
    hud.append(el("div", { class: "hud-line" }, `Step ${frame.t} / 500 · ${frame.cold ? "COLD weather: energy drains faster" : "temperate"}${actions.length && !frame.valid ? " · last action had no effect" : ""}`));
    const b = readBest(challenge.id);
    hud.append(el("div", { class: "hud-line" }, b === null ? "No personal best yet on this challenge." : `Your best here: ${b} steps.`));
  };
  const start = (): void => {
    challenge = ladder.challenges.find((c) => c.id === sel.value)!;
    actions = [];
    result.innerHTML = "";
    const lh = py.globals.get("layout_hash")(JSON.stringify(challenge.spec));
    if (lh !== challenge.layout_hash) {
      result.append(el("p", { class: "error", role: "alert" }, "Warning: this browser's NumPy generates a different scenario than the training machine, so ranks on this challenge are not comparable."));
    }
    frame = JSON.parse(py.globals.get("reset")(JSON.stringify(challenge.spec)));
    render();
    game.focus();
  };
  const finish = async (): Promise<void> => {
    const you = frame.survived;
    const prev = readBest(challenge.id);
    if (prev === null || you > prev) writeBest(challenge.id, you);
    const rows = Object.entries(challenge.rungs).map(([k, v]) => ({ name: ladder.rung_labels[k] ?? k, v, you: false }));
    rows.push({ name: "You (rung 4: human)", v: you, you: true });
    rows.sort((a, b) => b.v - a.v);
    const rank = 1 + rows.filter((r) => r.v > you).length;
    result.innerHTML = "";
    result.append(el("h3", {}, frame.alive ? `You survived all 500 steps.` : `You died of ${frame.cause} after ${you} steps.`),
      el("p", {}, `Ranked #${rank} of ${rows.length} on the ladder for this scenario. Policy numbers are their precomputed survival on exactly this scenario; learned policies are medians over training seeds.`));
    const t = el("table", { class: "metric-strip" }, el("tr", {}, el("th", {}, "Rank"), el("th", {}, "Policy"), el("th", {}, "Steps survived")));
    rows.forEach((r, i) => t.append(el("tr", { class: r.you ? "you" : "" }, el("td", {}, String(i + 1)), el("td", {}, r.name), el("td", {}, r.v.toFixed(r.v % 1 ? 1 : 0)))));
    result.append(t);
    const best = rows.filter((r) => !r.you && r.v < you).length;
    if (rows.some((r) => !r.you && r.v > you && /baseline|script|random/i.test(r.name))) result.append(el("p", { class: "note" }, "Honesty rule: when a baseline beats a learned policy, this page shows it. Nothing is hidden or rounded."));
    const reveal = el("button", { type: "button", class: "btn" }, "Reveal the full map and replay my run (observer view)");
    reveal.onclick = () => {
      const rep: Replay = JSON.parse(py.globals.get("observer_replay")(JSON.stringify(actions)));
      const host = el("div", { class: "arena-grid one" });
      result.append(host);
      const tl = new Timeline(result, rep.frames.length - 1);
      const a = new Arena(host);
      a.set(rep, "observer");
      tl.arenas.push(a);
      tl.seek(0);
      reveal.remove();
    };
    result.append(reveal, el("p", { class: "note" }, `Scores stay in this browser (localStorage). No accounts, no server. ${best} policies you beat.`));
  };
  game.addEventListener("keydown", (e) => {
    const a = KEYS[(e as KeyboardEvent).key];
    if (a === undefined) return;
    e.preventDefault();
    if (frame.done) return;
    actions.push(a);
    frame = JSON.parse(py.globals.get("step")(a));
    render();
    if (frame.done) void finish();
  });
  const touch = el("div", { class: "touch", "aria-label": "On-screen controls" });
  ([["↑", 0], ["←", 2], ["↓", 1], ["→", 3], ["Interact", 4], ["Wait", 5]] as [string, number][]).forEach(([lab, a]) => {
    const b = el("button", { type: "button", class: "btn" }, lab);
    b.onclick = () => { game.dispatchEvent(new KeyboardEvent("keydown", { key: ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "e", "x"][a] })); };
    touch.append(b);
  });
  root.append(touch);
  sel.onchange = start;
  void reducedMotion;
  start();
}
