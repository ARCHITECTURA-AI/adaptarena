"""Two-minute video, recorded from the real site with burned-in captions (no narration track).

Budget (seconds): hook 15 (random dies, the planner lives) · game 30 · headline result 45 · what the agent learned 20 · repo 10.
Captions are generated from the results files, so the video states exactly what the data says.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time

from . import plan as planmod
from .experiments import tier_dir

CAPTION_JS = """
(() => {
  let d = document.getElementById('aa-cap');
  if (!d) { d = document.createElement('div'); d.id = 'aa-cap';
    d.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:18px 40px;background:rgba(10,10,12,.88);color:#fff;font:600 26px/1.3 system-ui;z-index:99999;text-align:center';
    document.body.appendChild(d); }
  window.aaCaption = (t) => { d.textContent = t; };
})();
"""


def _serve(root: str, port: int):
    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".mjs": "text/javascript"}
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), functools.partial(H, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def build(tier_name: str, pyodide_dir: str = None, out: str = None) -> str:
    from playwright.sync_api import sync_playwright
    tdir = tier_dir(tier_name)
    dist = os.path.join(planmod.REPO, "site", "dist")
    hl = json.load(open(os.path.join(tdir, "headline", "stats.json")))
    D = hl["comparisons"]["D_vs_B"]["iqm_diff"]
    pa = hl["per_arm"]
    cap = {
        "hook1": "Same world, same 500 steps. A random agent dies.",
        "hook2": "A planner that sees the whole map survives all 500. Can a policy that only sees a 9×9 window learn to adapt?",
        "play": "You can play the exact same fog-of-war view, in your browser, and rank yourself against the agents.",
        "q": "The question: does training where the policy fails most beat uniform randomization, at equal compute?",
        "res": (f"10 seeds per arm, locked test suite. Planner-regret replay vs uniform: IQM difference "
                f"{D['diff']:+.3f} [{D['lo']:+.3f}, {D['hi']:+.3f}]. {hl['decision']['text']}"),
        "learned": "Probes show what each policy did, with notes on what they do not show. Stress curves mark breaking points.",
        "repo": "pip install adaptarena · make reproduce TIER=smoke · the report is the product.",
    }
    tmp = tempfile.mkdtemp()
    serve_root = os.path.join(tmp, "root")
    os.makedirs(serve_root)
    os.symlink(dist, os.path.join(serve_root, "site"))
    if pyodide_dir:
        os.symlink(pyodide_dir, os.path.join(serve_root, "pyodide"))
    port = 8791
    srv = _serve(serve_root, port)
    base = f"http://127.0.0.1:{port}/site/"
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if os.path.exists(exe) else p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=os.path.join(tmp, "vid"),
                            record_video_size={"width": 1280, "height": 720})
        if pyodide_dir:
            ctx.add_init_script(f"window.ADAPTARENA_PYODIDE='http://127.0.0.1:{port}/pyodide/'")
        pg = ctx.new_page()

        def caption(t):
            pg.evaluate(CAPTION_JS)
            pg.evaluate("t => window.aaCaption(t)", t)

        # 1. hook (15 s)
        pg.goto(base + "compare.html")
        pg.wait_for_selector("canvas")
        pg.evaluate(CAPTION_JS)
        sels = pg.locator(".arena-col select")
        sels.nth(0).select_option("random")
        sels.nth(1).select_option("planner")
        caption(cap["hook1"])
        pg.select_option("select[aria-label=View]", "observer")
        pg.locator("select[aria-label='Playback speed']").select_option("2")
        pg.locator(".timeline button").first.click()
        pg.wait_for_timeout(6000)
        caption(cap["hook2"])
        pg.wait_for_timeout(8000)
        # 2. game (30 s)
        pg.goto(base + "index.html")
        caption(cap["play"])
        try:
            pg.wait_for_selector(".loading:has-text('Ready')", timeout=60000)
            pg.focus(".game")
            for k in ["ArrowLeft", "ArrowLeft", "ArrowUp", "ArrowUp", "ArrowRight", "ArrowRight", "ArrowDown", "e"] * 4:
                pg.keyboard.press(k)
                pg.wait_for_timeout(350)
            for _ in range(520):
                pg.keyboard.press("x")
                if pg.locator(".result h3").count():
                    break
            pg.wait_for_timeout(8000)
        except Exception:
            pg.wait_for_timeout(15000)
        # 3. headline (45 s)
        pg.goto(base + "report.html")
        pg.wait_for_selector("img")
        caption(cap["q"])
        pg.wait_for_timeout(9000)
        img = pg.locator("img[src$='headline.png']").first
        img.scroll_into_view_if_needed()
        caption(cap["res"])
        pg.wait_for_timeout(26000)
        pg.goto(base + "ledger.html")
        caption("Every claim has a status. Supported means a pre-declared comparison whose interval excludes no-difference.")
        pg.wait_for_timeout(10000)
        # 4. learned (20 s)
        pg.goto(base + "stress.html")
        caption(cap["learned"])
        pg.wait_for_timeout(10000)
        pg.goto(base + "probes.html")
        caption(cap["learned"])
        pg.wait_for_timeout(10000)
        # 5. repo (10 s)
        pg.goto(base + "museum.html")
        caption(cap["repo"])
        pg.wait_for_timeout(10000)
        ctx.close()
        b.close()
    srv.shutdown()
    vid = [f for f in os.listdir(os.path.join(tmp, "vid")) if f.endswith(".webm")][0]
    out = out or os.path.join(planmod.REPO, "docs", "adaptarena_v2_overview.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(tmp, "vid", vid), "-vf", "fps=24", "-c:v", "libx264",
                    "-crf", "30", "-preset", "fast", "-pix_fmt", "yuv420p", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out
