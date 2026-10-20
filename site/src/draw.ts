import { Replay, STAT_MAX, VIEW } from "./types";

export type ViewMode = "observer" | "agent";
export interface CellState { wall: boolean; food: 0 | 1 | 2; water: 0 | 1 | 2; hazard: boolean; }

const css = (n: string): string => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";

/** Cheap deterministic hash so every tile keeps the same subtle texture between frames. */
function noise(x: number, y: number): number {
  let h = (x * 374761393 + y * 668265263) | 0;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967295;
}

function rr(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function floorTile(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, gx: number, gy: number): void {
  const n = noise(gx, gy);
  ctx.fillStyle = css("--floor");
  ctx.fillRect(x, y, s, s);
  ctx.fillStyle = `rgba(255,255,255,${0.012 + n * 0.03})`;
  ctx.fillRect(x, y, s, s);
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
}

/** Walls: raised slate block with a bevel, plus diagonal hatching so they never depend on colour alone. */
function wallTile(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, gx: number, gy: number): void {
  const g = ctx.createLinearGradient(x, y, x, y + s);
  g.addColorStop(0, css("--wall-hi"));
  g.addColorStop(1, css("--wall"));
  ctx.fillStyle = g;
  rr(ctx, x + 1, y + 1, s - 2, s - 2, s * 0.14);
  ctx.fill();
  ctx.save();
  rr(ctx, x + 1, y + 1, s - 2, s - 2, s * 0.14);
  ctx.clip();
  ctx.strokeStyle = css("--wall-line");
  ctx.lineWidth = 1;
  const step = Math.max(5, s / 4.5);
  for (let i = -s; i < s; i += step) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + s);
    ctx.lineTo(x + i + s, y);
    ctx.stroke();
  }
  ctx.restore();
  ctx.strokeStyle = "rgba(255,255,255,.10)";
  rr(ctx, x + 1.5, y + 1.5, s - 3, s - 3, s * 0.14);
  ctx.stroke();
  void gx; void gy;
}

function glow(ctx: CanvasRenderingContext2D, color: string, blur: number, fn: () => void): void {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  fn();
  ctx.restore();
}

function food(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, ready: boolean, tm: number): void {
  const c = css("--food");
  const p = ready ? 1 + 0.06 * Math.sin(tm / 380) : 1;
  const cx = x + s / 2, cy = y + s / 2 + s * 0.04, r = s * 0.34 * p;
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx + r * 1.05, cy + r * 0.8);
    ctx.lineTo(cx - r * 1.05, cy + r * 0.8);
    ctx.closePath();
  };
  if (ready) {
    glow(ctx, c, s * 0.55, () => { path(); ctx.fillStyle = c; ctx.fill(); });
    ctx.fillStyle = "rgba(255,255,255,.35)";
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.85);
    ctx.lineTo(cx - r * 0.35, cy - r * 0.05);
    ctx.lineTo(cx + r * 0.1, cy - r * 0.05);
    ctx.closePath();
    ctx.fill();
  } else {
    path();
    ctx.strokeStyle = c;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.65;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = ready ? css("--on-accent") : c;
  ctx.font = `800 ${Math.round(s * 0.32)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("F", cx, cy + r * 0.62);
}

function water(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, ready: boolean, tm: number): void {
  const c = css("--water");
  const cx = x + s / 2, cy = y + s / 2, r = s * 0.33;
  if (ready) {
    const ph = (tm / 1400) % 1;
    ctx.strokeStyle = c;
    ctx.globalAlpha = 0.5 * (1 - ph);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r + ph * s * 0.2, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
    glow(ctx, c, s * 0.6, () => {
      const g = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 0.1, cx, cy, r);
      g.addColorStop(0, "#c9f1ff");
      g.addColorStop(1, c);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    });
  } else {
    ctx.strokeStyle = c;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.65;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = ready ? css("--on-accent") : c;
  ctx.font = `800 ${Math.round(s * 0.32)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("W", cx, cy + s * 0.11);
}

function hazard(ctx: CanvasRenderingContext2D, x: number, y: number, s: number): void {
  const c = css("--hazard");
  ctx.fillStyle = "rgba(255,70,60,.14)";
  ctx.fillRect(x, y, s, s);
  glow(ctx, c, s * 0.35, () => {
    ctx.strokeStyle = c;
    ctx.lineWidth = Math.max(2, s * 0.1);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x + s * 0.24, y + s * 0.24); ctx.lineTo(x + s * 0.76, y + s * 0.76);
    ctx.moveTo(x + s * 0.76, y + s * 0.24); ctx.lineTo(x + s * 0.24, y + s * 0.76);
    ctx.stroke();
  });
}

function agent(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, alive: boolean, tm: number): void {
  const cx = x + s / 2, cy = y + s / 2, r = s * 0.31;
  const col = alive ? css("--agent") : css("--dead");
  if (alive) {
    const ph = (tm / 1100) % 1;
    ctx.strokeStyle = css("--accent");
    ctx.globalAlpha = 0.55 * (1 - ph);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r + ph * s * 0.34, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  glow(ctx, alive ? css("--accent") : "#000", s * 0.5, () => {
    const g = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 0.1, cx, cy, r);
    g.addColorStop(0, "#ffffff");
    g.addColorStop(1, col);
    ctx.fillStyle = alive ? g : col;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = css("--on-accent");
  ctx.font = `800 ${Math.round(s * 0.3)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText(alive ? "A" : "×", cx, cy + s * 0.105);
}

/** State of a map cell at step t from a replay (observer truth). food/water: 0 none, 1 ready, 2 depleted. */
export function cellAt(rep: Replay, r: number, c: number, t: number): CellState {
  const sc = rep.scenario;
  const fr = rep.frames[Math.min(t, rep.frames.length - 1)];
  const wall = sc.terrain[r][c] === "1";
  let f: 0 | 1 | 2 = 0, w: 0 | 1 | 2 = 0;
  sc.food.forEach((p, i) => { if (p[0] === r && p[1] === c && t < sc.food_until[i]) f = (fr[5] >> i) & 1 ? 1 : 2; });
  const wp = sc.reloc_step >= 0 && t >= sc.reloc_step ? sc.water2 : sc.water;
  wp.forEach((p, i) => { if (p[0] === r && p[1] === c && t < sc.water_until[i]) w = (fr[6] >> i) & 1 ? 1 : 2; });
  const hz = t >= sc.hazard_start && sc.hazard.some((p) => p[0] === r && p[1] === c);
  return { wall, food: f, water: w, hazard: hz };
}

function drawCell(ctx: CanvasRenderingContext2D, st: CellState, x: number, y: number, s: number, gx: number, gy: number, tm: number): void {
  if (st.wall) { floorTile(ctx, x, y, s, gx, gy); wallTile(ctx, x, y, s, gx, gy); return; }
  floorTile(ctx, x, y, s, gx, gy);
  if (st.hazard) hazard(ctx, x, y, s);
  if (st.food) food(ctx, x, y, s, st.food === 1, tm);
  if (st.water) water(ctx, x, y, s, st.water === 1, tm);
}

/** Soft vignette: the edge of the 9x9 window fades into darkness, like fog of war. */
function fog(ctx: CanvasRenderingContext2D, size: number): void {
  const g = ctx.createRadialGradient(size / 2, size / 2, size * 0.22, size / 2, size / 2, size * 0.74);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, "rgba(4,6,12,.62)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
}

function setup(canvas: HTMLCanvasElement, size: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(size * dpr)) {
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  return ctx;
}

export function canvasSize(mode: ViewMode, cell = 24): number {
  return (mode === "observer" ? 15 : VIEW) * cell;
}

export function drawReplay(canvas: HTMLCanvasElement, rep: Replay, t: number, mode: ViewMode, cell = mode === "observer" ? 26 : 38): void {
  const size = canvasSize(mode, cell);
  const ctx = setup(canvas, size);
  const tm = performance.now();
  const tt = Math.min(t, rep.frames.length - 1);
  const fr = rep.frames[tt];
  const dead = !rep.alive && tt >= rep.frames.length - 1;
  if (mode === "observer") {
    for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) drawCell(ctx, cellAt(rep, r, c, tt), c * cell, r * cell, cell, c, r, tm);
    agent(ctx, fr[1] * cell, fr[0] * cell, cell, !dead, tm);
  } else {
    const half = (VIEW - 1) / 2;
    for (let dy = -half; dy <= half; dy++) for (let dx = -half; dx <= half; dx++) {
      const r = fr[0] + dy, c = fr[1] + dx;
      const x = (dx + half) * cell, y = (dy + half) * cell;
      if (r < 0 || r >= 15 || c < 0 || c >= 15) drawCell(ctx, { wall: true, food: 0, water: 0, hazard: false }, x, y, cell, c, r, tm);
      else drawCell(ctx, cellAt(rep, r, c, tt), x, y, cell, c, r, tm);
    }
    agent(ctx, half * cell, half * cell, cell, !dead, tm);
    fog(ctx, size);
  }
  overlays(ctx, size, fr[4] === 1, dead, rep.cause);
}

function overlays(ctx: CanvasRenderingContext2D, size: number, cold: boolean, dead: boolean, cause: string): void {
  if (cold) {
    const g = ctx.createLinearGradient(0, 0, 0, size);
    g.addColorStop(0, "rgba(140,210,255,.30)");
    g.addColorStop(1, "rgba(86,180,233,.10)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = "rgba(200,235,255,.55)";
    ctx.lineWidth = 3;
    ctx.strokeRect(1.5, 1.5, size - 3, size - 3);
  }
  if (dead) {
    ctx.fillStyle = css("--dead-tint");
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = css("--fg");
    ctx.font = `800 ${Math.round(size * 0.07)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(`DEAD · ${cause}`.toUpperCase(), size / 2, size / 2);
  }
}

export function statBars(rep: Replay, t: number): { energy: number; hydration: number } {
  const fr = rep.frames[Math.min(t, rep.frames.length - 1)];
  return { energy: Math.max(0, fr[2]) / STAT_MAX, hydration: Math.max(0, fr[3]) / STAT_MAX };
}

/** Draw an agent-view frame from the packed cell codes the Python core returns (bit flags). */
export function drawCodes(canvas: HTMLCanvasElement, codes: number[], dead: boolean, cold: boolean, cell = 44, cause = ""): void {
  const size = VIEW * cell;
  const ctx = setup(canvas, size);
  const tm = performance.now();
  const half = (VIEW - 1) / 2;
  for (let y = 0; y < VIEW; y++) for (let x = 0; x < VIEW; x++) {
    const k = codes[y * VIEW + x];
    drawCell(ctx, { wall: !!(k & 1), food: k & 2 ? 1 : k & 8 ? 2 : 0, water: k & 4 ? 1 : k & 16 ? 2 : 0, hazard: !!(k & 32) }, x * cell, y * cell, cell, x, y, tm);
  }
  agent(ctx, half * cell, half * cell, cell, !dead, tm);
  fog(ctx, size);
  overlays(ctx, size, cold, dead, cause);
}
