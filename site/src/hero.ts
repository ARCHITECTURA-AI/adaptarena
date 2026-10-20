import { reducedMotion } from "./types";

/** Ambient background: slow drifting glow particles behind every page. Off under reduced motion. */
export function ambient(): void {
  const c = document.createElement("canvas");
  c.className = "ambient";
  c.setAttribute("aria-hidden", "true");
  document.body.prepend(c);
  const ctx = c.getContext("2d")!;
  const N = 46;
  const pts = Array.from({ length: N }, () => ({ x: Math.random(), y: Math.random(), r: 0.6 + Math.random() * 2.2, vx: (Math.random() - 0.5) * 0.00006, vy: -0.00004 - Math.random() * 0.00008, h: Math.random() }));
  const size = () => {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.round(window.innerWidth * d);
    c.height = Math.round(window.innerHeight * d);
    ctx.setTransform(d, 0, 0, d, 0, 0);
  };
  size();
  window.addEventListener("resize", size);
  const draw = (animate: boolean): void => {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    for (const p of pts) {
      if (animate) { p.x += p.vx * 16; p.y += p.vy * 16; if (p.y < -0.05) { p.y = 1.05; p.x = Math.random(); } }
      const x = p.x * window.innerWidth, y = p.y * window.innerHeight;
      const g = ctx.createRadialGradient(x, y, 0, x, y, p.r * 9);
      const hue = p.h > 0.5 ? "86,180,233" : "230,159,0";
      g.addColorStop(0, `rgba(${hue},.38)`);
      g.addColorStop(1, `rgba(${hue},0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(x, y, p.r * 9, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  if (reducedMotion()) { draw(false); return; }
  let last = 0;
  const loop = (now: number): void => {
    if (now - last > 33 && !document.hidden) { draw(true); last = now; }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/** Count a number up when it scrolls into view. */
export function countUp(el: HTMLElement): void {
  const target = Number(el.dataset.to ?? "0");
  const dec = Number(el.dataset.dec ?? "0");
  const fmt = (v: number): string => v.toFixed(dec);
  if (reducedMotion()) { el.textContent = fmt(target); return; }
  const run = (): void => {
    const t0 = performance.now();
    const step = (now: number): void => {
      const k = Math.min(1, (now - t0) / 1400);
      el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { run(); io.disconnect(); } }, { threshold: 0.4 });
  io.observe(el);
}

export function revealOnScroll(): void {
  const els = Array.from(document.querySelectorAll<HTMLElement>(".reveal"));
  if (reducedMotion() || !("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("in")); return; }
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
  els.forEach((e) => io.observe(e));
}

export function themeToggle(): void {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const apply = (t: string): void => {
    document.documentElement.setAttribute("data-theme", t);
    btn.setAttribute("aria-label", t === "dark" ? "Switch to light theme" : "Switch to dark theme");
    btn.textContent = t === "dark" ? "☀" : "☾";
  };
  let cur = "dark";
  try { cur = localStorage.getItem("aa-theme") ?? "dark"; } catch { /* ignore */ }
  apply(cur);
  btn.addEventListener("click", () => {
    cur = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("aa-theme", cur); } catch { /* ignore */ }
    apply(cur);
  });
}
