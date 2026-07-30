"""Regenerate the screenshots in README.md from the running report.

Start the report first, then:

    .venv\\Scripts\\python.exe -m streamlit run app.py
    .venv\\Scripts\\python.exe tools/screenshots.py docs/screenshots

Drives the Chrome already installed on the machine over the DevTools Protocol, speaking
it through the ``websockets`` package Streamlit already depends on - so this adds no
dependency and needs no headless browser install. The point is that the images in the
README can be rebuilt after a design change instead of quietly going stale, which is
what hand-taken screenshots always do.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222
URL = "http://localhost:8501"
OUT = Path(sys.argv[1])


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    async def send(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("id") == self.n:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})


async def shoot(cdp: CDP, path: Path, full: bool = True) -> None:
    params = {"format": "png"}
    if full:
        metrics = await cdp.send("Page.getLayoutMetrics")
        css = metrics["cssContentSize"]
        params["captureBeyondViewport"] = True
        params["clip"] = {
            "x": 0,
            "y": 0,
            "width": css["width"],
            "height": css["height"],
            "scale": 1,
        }
    result = await cdp.send("Page.captureScreenshot", **params)
    path.write_bytes(base64.b64decode(result["data"]))
    print(f"wrote {path}  ({path.stat().st_size // 1024} KB)")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    chrome = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            "--window-size=1440,1600",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            "--no-first-run",
            "--no-default-browser-check",
            # Deliberately outside the repository: Chrome writes a few thousand cache
            # files into its profile, and `git add -A` will happily commit every one.
            f"--user-data-dir={Path(tempfile.gettempdir()) / 'pl-shoot-profile'}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        target = None
        for _ in range(40):
            try:
                tabs = json.loads(urlopen(f"http://127.0.0.1:{PORT}/json").read())
                pages = [t for t in tabs if t["type"] == "page"]
                if pages:
                    target = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not target:
            raise SystemExit("chrome never came up")

        async with websockets.connect(target, max_size=200 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send(
                "Emulation.setDeviceMetricsOverride",
                width=1440,
                height=1600,
                deviceScaleFactor=2,
                mobile=False,
            )

            await cdp.send("Page.navigate", url=URL)
            # Streamlit renders client-side and streams the columns in, so waiting for
            # the first card catches a half-drawn grid. Wait for the count to stop
            # changing instead.
            previous, stable = -1, 0
            for _ in range(90):
                time.sleep(1)
                found = await cdp.send(
                    "Runtime.evaluate",
                    expression="document.querySelectorAll('.pl-card').length",
                    returnByValue=True,
                )
                count = found["result"].get("value", 0)
                stable = stable + 1 if count and count == previous else 0
                previous = count
                if stable >= 3:
                    print(f"{count} cards rendered and settled")
                    break
            else:
                raise SystemExit(f"grid never settled (last count {previous})")

            # Streamlit's own toolbar is host chrome, not the report - hide it so the
            # screenshot shows the page rather than the framework hosting it.
            await cdp.send(
                "Runtime.evaluate",
                expression="""
                (() => {
                  const css = document.createElement('style');
                  css.textContent = `
                    [data-testid="stToolbar"], header[data-testid="stHeader"],
                    #MainMenu, footer { display: none !important; }
                    [data-testid="stAppViewContainer"] { padding-top: 0 !important; }
                    .stAppDeployButton { display: none !important; }
                  `;
                  document.head.appendChild(css);
                  return 'hidden';
                })()
                """,
                returnByValue=True,
            )

            time.sleep(3)  # let the pixel font and inline SVG kits paint
            await shoot(cdp, OUT / "report.png")

            # The expected-XI overlay is a pure-CSS checkbox toggle, so opening it is a
            # matter of checking the box - the same thing the <label> does on click.
            opened = await cdp.send(
                "Runtime.evaluate",
                expression="""
                (() => {
                  const box = document.querySelector('.pl-modal-toggle');
                  if (!box) return 'no toggle';
                  box.checked = true;
                  box.dispatchEvent(new Event('change', {bubbles: true}));
                  const modal = box.parentElement.querySelector('.pl-modal');
                  return modal ? getComputedStyle(modal).display : 'no modal';
                })()
                """,
                returnByValue=True,
            )
            print("modal display:", opened["result"].get("value"))
            time.sleep(2)

            # Crop to the dialog itself. Its own backdrop dims the page behind it, which
            # is the effect worth showing, so a little margin stays in frame.
            box = await cdp.send(
                "Runtime.evaluate",
                expression="""
                (() => {
                  const open = '.pl-modal-toggle:checked ~ .pl-modal .pl-modal-box';
                  const m = document.querySelector(open)
                        || document.querySelector('.pl-modal-box');
                  if (!m) return null;
                  const r = m.getBoundingClientRect();
                  return JSON.stringify({x: r.x, y: r.y, width: r.width, height: r.height});
                })()
                """,
                returnByValue=True,
            )
            raw = box["result"].get("value")
            if raw:
                rect = json.loads(raw)
                pad = 28
                clip = {
                    "x": max(rect["x"] - pad, 0),
                    "y": max(rect["y"] - pad, 0),
                    "width": rect["width"] + pad * 2,
                    "height": rect["height"] + pad * 2,
                    "scale": 1,
                }
                result = await cdp.send(
                    "Page.captureScreenshot",
                    format="png",
                    captureBeyondViewport=True,
                    clip=clip,
                )
                path = OUT / "expected-xi.png"
                path.write_bytes(base64.b64decode(result["data"]))
                print(f"wrote {path}  ({path.stat().st_size // 1024} KB)  clip={clip}")
            else:
                print("no panel found, falling back to viewport")
                await shoot(cdp, OUT / "expected-xi.png", full=False)
    finally:
        chrome.terminate()


asyncio.run(main())
