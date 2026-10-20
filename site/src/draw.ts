import { Replay, STAT_MAX, VIEW } from "./types";

const CELL = 24;
export type ViewMode = "observer" | "agent";

const css = (n: string): string => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888";

/** Draw glyphs so terrain never depends on colour alone: wall = hatch, food = triangle + F, water = waves + W. */
function glyphWall(ctx: CanvasRenderingContext2D, x: number, y: number, s: number): void {
  ctx.fillStyle = css("--wall");
  ctx.fillRect(x, y, s, s);
  ctx.strokeStyle = css("--wall-line");
  ctx.lineWidth = 1;
  for (let i = -s; i < s; i += 6) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + s);
    ctx.lineTo(x + i + s, y);
    ctx.stroke();
  }
}

function glyphFood(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, ready: boolean): void {
  ctx.beginPath();
  ctx.moveTo(x + s / 2, y + 4);
  ctx.lineTo(x + s - 4, y + s - 4);
  ctx.lineTo(x + 4, y + s - 4);
  ctx.closePath();
  if (ready) {
    ctx.fillStyle = css("--food");
    ctx.fill();
  }
  ctx.strokeStyle = css("--food");
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = ready ? css("--on-accent") : css("--food");
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("F", x + s / 2, y + s - 7);
}

function glyphWater(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, ready: boolean): void {
  ctx.beginPath();
  ctx.arc(x + s / 2, y + s / 2, s / 2 - 4, 0, Math.PI * 2);
  if (ready) {
    ctx.fillStyle = css("--water");
    ctx.fill();
  }
  ctx.strokeStyle = css("--water");
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = ready ? css("--on-accent") : css("--water");
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("W", x + s / 2, y + s / 2 + 4);
}

function glyphHazard(ctx: CanvasRenderingContext2D, x: number, y: number, s: number): void {
  ctx.strokeStyle = css("--hazard");
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x + 5, y + 5); ctx.lineTo(x + s - 5, y + s - 5);
  ctx.moveTo(x + s - 5, y + 5); ctx.lineTo(x + 5, y + s - 5);
  ctx.stroke();
}

function glyphAgent(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, alive: boolean): void {
  ctx.beginPath();
  ctx.arc(x + s / 2, y + s / 2, s / 2 - 3, 0, Math.PI * 2);
  ctx.fillStyle = alive ? css("--agent") : css("--dead");
  ctx.fill();
  ctx.strokeStyle = css("--bg");
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = css("--on-accent");
  ctx.font = "bold 12px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(alive ? "A" : "×", x + s / 2, y + s / 2 + 4);
}

export interface CellState { wall: boolean; food: 0 | 1 | 2; water: 0 | 1 | 2; hazard: boolean; }

/** State of a map cell at step t from a replay (observer truth). food/water: 0 none, 1 ready, 2 depleted. */
export function cellAt(rep: Replay, r: number, c: number, t: number): CellState {
  const sc = rep.scenario;
  const fr = rep.frames[Math.min(t, rep.frames.length - 1)];
  const wall = sc.terrain[r][c] === "1";
  let food: 0 | 1 | 2 = 0, water: 0 | 1 | 2 = 0;
  sc.food.forEach((p, i) => { if (p[0] === r && p[1] === c && t < sc.food_until[i]) food = (fr[5] >> i) & 1 ? 1 : 2; });
  const wp = sc.reloc_step >= 0 && t >= sc.reloc_step ? sc.water2 : sc.water;
  wp.forEach((p, i) => { if (p[0] === r && p[1] === c && t < sc.water_until[i]) water = (fr[6] >> i) & 1 ? 1 : 2; });
  const hazard = t >= sc.hazard_start && sc.hazard.some((p) => p[0] === r && p[1] === c);
  return { wall, food, water, hazard };
}

function drawCell(ctx: CanvasRenderingContext2D, st: CellState, x: number, y: number): void {
  ctx.fillStyle = css("--floor");
  ctx.fillRect(x, y, CELL, CELL);
  ctx.strokeStyle = css("--grid");
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
  if (st.wall) glyphWall(ctx, x, y, CELL);
  else {
    if (st.food) glyphFood(ctx, x, y, CELL, st.food === 1);
    if (st.water) glyphWater(ctx, x, y, CELL, st.water === 1);
    if (st.hazard) glyphHazard(ctx, x, y, CELL);
  }
}

export function canvasSize(mode: ViewMode): number {
  return (mode === "observer" ? 15 : VIEW) * CELL;
}

export function drawReplay(canvas: HTMLCanvasElement, rep: Replay, t: number, mode: ViewMode): void {
  const size = canvasSize(mode);
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const tt = Math.min(t, rep.frames.length - 1);
  const fr = rep.frames[tt];
  const dead = !rep.alive && tt >= rep.frames.length - 1;
  ctx.clearRect(0, 0, size, size);
  const cold = fr[4] === 1;
  if (mode === "observer") {
    for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) drawCell(ctx, cellAt(rep, r, c, tt), c * CELL, r * CELL);
    glyphAgent(ctx, fr[1] * CELL, fr[0] * CELL, CELL, !dead);
  } else {
    const half = (VIEW - 1) / 2;
    for (let dy = -half; dy <= half; dy++) for (let dx = -half; dx <= half; dx++) {
      const r = fr[0] + dy, c = fr[1] + dx;
      const x = (dx + half) * CELL, y = (dy + half) * CELL;
      if (r < 0 || r >= 15 || c < 0 || c >= 15) drawCell(ctx, { wall: true, food: 0, water: 0, hazard: false }, x, y);
      else drawCell(ctx, cellAt(rep, r, c, tt), x, y);
    }
    glyphAgent(ctx, half * CELL, half * CELL, CELL, !dead);
  }
  if (cold) {
    ctx.fillStyle = css("--cold-tint");
    ctx.fillRect(0, 0, size, size);
  }
  if (dead) {
    ctx.fillStyle = css("--dead-tint");
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = css("--fg");
    ctx.font = "bold 14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`DEAD · ${rep.cause}`, size / 2, size / 2);
  }
}

export function statBars(rep: Replay, t: number): { energy: number; hydration: number } {
  const fr = rep.frames[Math.min(t, rep.frames.length - 1)];
  return { energy: Math.max(0, fr[2]) / STAT_MAX, hydration: Math.max(0, fr[3]) / STAT_MAX };
}

/** Draw an agent-view frame from the packed cell codes the Python core returns (bit flags). */
export function drawCodes(canvas: HTMLCanvasElement, codes: number[], dead: boolean, cold: boolean): void {
  const size = VIEW * CELL, dpr = window.devicePixelRatio || 1;
  if (canvas.width !== size * dpr) {
    canvas.width = size * dpr; canvas.height = size * dpr;
    canvas.style.width = `${size}px`; canvas.style.height = `${size}px`;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const half = (VIEW - 1) / 2;
  for (let y = 0; y < VIEW; y++) for (let x = 0; x < VIEW; x++) {
    const k = codes[y * VIEW + x];
    drawCell(ctx, { wall: !!(k & 1), food: k & 2 ? 1 : k & 8 ? 2 : 0, water: k & 4 ? 1 : k & 16 ? 2 : 0, hazard: !!(k & 32) }, x * CELL, y * CELL);
  }
  glyphAgent(ctx, half * CELL, half * CELL, CELL, !dead);
  if (cold) { ctx.fillStyle = css("--cold-tint"); ctx.fillRect(0, 0, size, size); }
  if (dead) { ctx.fillStyle = css("--dead-tint"); ctx.fillRect(0, 0, size, size); }
}
