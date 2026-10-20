export interface ScenarioPayload {
  spec: Record<string, unknown>;
  sid: string;
  horizon: number;
  terrain: string[];
  start: [number, number];
  food: [number, number][];
  water: [number, number][];
  water2: [number, number][];
  reloc_step: number;
  food_until: number[];
  water_until: number[];
  hazard: [number, number][];
  hazard_start: number;
  cold: string;
}

/** frames[t] = [row, col, energy, hydration, cold, foodReadyMask, waterReadyMask] before action t. */
export interface Replay {
  label: string;
  scenario: ScenarioPayload;
  actions: number[];
  valid: number[];
  frames: number[][];
  survival: number;
  alive: boolean;
  cause: string;
  final_hash: string;
  probs?: number[][] | null;
}

export const ACTIONS = ["up", "down", "left", "right", "interact", "wait"];
export const STAT_MAX = 10000;
export const VIEW = 9;

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, attrs: Record<string, string> = {}, ...kids: (string | Node)[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) e.append(k);
  return e;
}

export const reducedMotion = (): boolean => {
  try {
    if (localStorage.getItem("aa-reduced-motion") === "1") return true;
  } catch { /* storage may be blocked */ }
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
};
