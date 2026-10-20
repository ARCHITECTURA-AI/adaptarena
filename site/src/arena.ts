import { drawReplay, statBars, ViewMode } from "./draw";
import { ACTIONS, el, Replay, reducedMotion } from "./types";

/** One arena: a canvas, a title, stat bars, a status line and optional action-distribution bars. */
export class Arena {
  root: HTMLElement;
  canvas: HTMLCanvasElement;
  title: HTMLElement;
  status: HTMLElement;
  bars: HTMLElement;
  probs: HTMLElement;
  rep: Replay | null = null;
  mode: ViewMode = "observer";
  showProbs = false;
  markStep: number | null = null;

  constructor(host: HTMLElement, titleEl?: HTMLElement) {
    this.root = el("div", { class: "arena" });
    this.title = titleEl ?? el("div", { class: "arena-title" });
    this.canvas = el("canvas", { role: "img", "aria-label": "arena" });
    this.status = el("div", { class: "arena-status", "aria-live": "off" });
    this.bars = el("div", { class: "bars" });
    this.probs = el("div", { class: "probs" });
    this.root.append(this.title, this.canvas, this.bars, this.status, this.probs);
    host.append(this.root);
  }

  set(rep: Replay | null, mode?: ViewMode): void {
    this.rep = rep;
    if (mode) this.mode = mode;
  }

  viewLabel(): string {
    return this.mode === "observer" ? "Observer view (full map)" : "Agent view (what the policy sees)";
  }

  render(t: number): void {
    const rep = this.rep;
    if (!rep) return;
    drawReplay(this.canvas, rep, t, this.mode);
    const tt = Math.min(t, rep.frames.length - 1);
    const { energy, hydration } = statBars(rep, tt);
    this.bars.innerHTML = "";
    for (const [name, v] of [["Energy", energy], ["Hydration", hydration]] as const) {
      const row = el("div", { class: "bar-row" }, el("span", { class: "bar-name" }, name));
      const track = el("div", { class: "bar-track", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
        "aria-valuenow": String(Math.round(v * 100)), "aria-label": name });
      const fill = el("div", { class: "bar-fill" });
      fill.style.width = `${Math.round(v * 100)}%`;
      track.append(fill);
      row.append(track, el("span", { class: "bar-num" }, `${Math.round(v * 100)}`));
      this.bars.append(row);
    }
    const dead = !rep.alive && tt >= rep.frames.length - 1;
    this.status.textContent = `${this.viewLabel()} · step ${Math.min(t, rep.frames.length - 1)} · ` +
      (dead ? `died of ${rep.cause} after ${rep.survival} steps` : tt >= rep.frames.length - 1 ? `survived the horizon (${rep.survival})` : "alive");
    this.canvas.setAttribute("aria-label", `${rep.label}. ${this.status.textContent}`);
    this.probs.innerHTML = "";
    if (this.showProbs && rep.probs && rep.probs[tt]) {
      this.probs.append(el("div", { class: "probs-title" }, "Frozen policy: output distribution over actions (not confidence, not reasoning)"));
      rep.probs[tt].forEach((p, i) => {
        const row = el("div", { class: "bar-row" }, el("span", { class: "bar-name" }, ACTIONS[i]));
        const track = el("div", { class: "bar-track" });
        const fill = el("div", { class: "bar-fill prob" });
        fill.style.width = `${Math.round(p * 100)}%`;
        track.append(fill);
        row.append(track, el("span", { class: "bar-num" }, p.toFixed(2)));
        this.probs.append(row);
      });
    }
  }
}

/** Shared timeline measured in environment steps. Dead policies freeze on their last frame. */
export class Timeline {
  t = 0;
  max: number;
  playing = false;
  speed = 1;
  arenas: Arena[] = [];
  slider: HTMLInputElement;
  label: HTMLElement;
  playBtn: HTMLButtonElement;
  private last = 0;
  private acc = 0;
  onTick: (t: number) => void = () => {};

  constructor(host: HTMLElement, max = 500) {
    this.max = max;
    const bar = el("div", { class: "timeline", role: "group", "aria-label": "Replay controls" });
    this.playBtn = el("button", { type: "button", class: "btn" }, "Play");
    const back = el("button", { type: "button", class: "btn" }, "◀ step");
    const fwd = el("button", { type: "button", class: "btn" }, "step ▶");
    const restart = el("button", { type: "button", class: "btn" }, "Restart");
    this.slider = el("input", { type: "range", min: "0", max: String(max), value: "0", "aria-label": "Environment step" });
    this.label = el("span", { class: "tl-label" }, "step 0");
    const speed = el("select", { "aria-label": "Playback speed" });
    for (const s of [0.5, 1, 2, 4, 8]) speed.append(el("option", { value: String(s) }, `${s}×`));
    speed.value = "1";
    bar.append(this.playBtn, back, fwd, restart, this.slider, this.label, speed);
    host.append(bar);
    this.playBtn.onclick = () => this.toggle();
    back.onclick = () => this.seek(this.t - 1);
    fwd.onclick = () => this.seek(this.t + 1);
    restart.onclick = () => { this.seek(0); };
    this.slider.oninput = () => this.seek(Number(this.slider.value));
    speed.onchange = () => { this.speed = Number(speed.value); };
    bar.addEventListener("keydown", (e) => {
      if ((e as KeyboardEvent).key === " " && (e.target as HTMLElement).tagName !== "BUTTON") { e.preventDefault(); this.toggle(); }
    });
  }

  setMax(m: number): void {
    this.max = m;
    this.slider.max = String(m);
  }

  seek(t: number): void {
    this.t = Math.max(0, Math.min(this.max, Math.round(t)));
    this.slider.value = String(this.t);
    this.label.textContent = `step ${this.t}`;
    for (const a of this.arenas) a.render(this.t);
    this.onTick(this.t);
  }

  toggle(): void {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? "Pause" : "Play";
    if (this.playing) {
      if (reducedMotion()) { /* reduced motion: step at a slow fixed rate instead of continuous animation */ }
      this.last = performance.now();
      requestAnimationFrame((n) => this.loop(n));
    }
  }

  private loop(now: number): void {
    if (!this.playing) return;
    const rate = reducedMotion() ? 2 : 20 * this.speed;  // env steps per second
    this.acc += ((now - this.last) / 1000) * rate;
    this.last = now;
    const whole = Math.floor(this.acc);
    if (whole > 0) {
      this.acc -= whole;
      this.seek(this.t + whole);
      if (this.t >= this.max) { this.playing = false; this.playBtn.textContent = "Play"; return; }
    }
    requestAnimationFrame((n) => this.loop(n));
  }
}
