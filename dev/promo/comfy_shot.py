"""out/workflow.png for the promo: the "Hy-Nomas start from image" workflow in ComfyUI, the Paris picture in
Load Image, at 2x. Needs the test server (dev/serve.sh) with the workflows in its user directory, ComfyUI's Python
(aiohttp) and google-chrome.

    python dev/promo/comfy_shot.py [--server http://127.0.0.1:8199] [--image paris_start.jpeg]
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
W, H, CDP = 1600, 1100, 9399

SETUP = """
(async () => {
  for (let i = 0; i < 200 && !(window.app?.graph); i++) await new Promise(r => setTimeout(r, 100));
  const app = window.app;
  for (const [k, v] of [["Comfy.Minimap.Visible", false], ["Comfy.Graph.CanvasInfo", false]])
    try { await app.ui.settings.setSettingValueAsync(k, v); } catch (e) {}
  const wf = await (await fetch("/api/userdata/workflows%2F" + encodeURIComponent("Hy-Nomas start from image.json"))).json();
  await app.loadGraphData(wf);
  for (const n of [...app.graph._nodes]) if (n.type === "MarkdownNote") app.graph.remove(n);
  const img = app.graph._nodes.find(n => n.type === "LoadImage");
  const w = img.widgets.find(w => w.name === "image");
  w.value = IMAGE; w.callback?.(IMAGE);
  await new Promise(r => setTimeout(r, 1500));
  const ns = app.graph._nodes; let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const n of ns) { const b = n.getBounding(); x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[0] + b[2]); y1 = Math.max(y1, b[1] + b[3]); }
  const c = app.canvas.canvas, dpr = window.devicePixelRatio || 1;
  const top = 130, side = 90;
  const s = Math.min((c.width / dpr - 2 * side) / (x1 - x0), (c.height / dpr - top - 40) / (y1 - y0));
  app.canvas.ds.scale = s;
  app.canvas.ds.offset = [-x0 + (c.width / dpr - s * (x1 - x0)) / 2 / s, -y0 + top / s];
  app.canvas.setDirty(true, true);
  await new Promise(r => setTimeout(r, 800));
  // this test server's frontend is older than its backend: drop the version warnings
  document.querySelectorAll('.p-toast-message, .p-toast').forEach(e => e.remove());
  await new Promise(r => setTimeout(r, 200));
  return [x0, y0, x1, y1, s];
})()
"""


async def main(server, image):
    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--hide-scrollbars", f"--remote-debugging-port={CDP}",
                               f"--user-data-dir={profile}", f"--window-size={W},{H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(100):
                try:
                    pages = await (await http.get(f"http://127.0.0.1:{CDP}/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.1)
            target = next(p for p in pages if p["type"] == "page")
            async with http.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                seq = 0

                async def call(method, **params):
                    nonlocal seq
                    seq += 1
                    await ws.send_json({"id": seq, "method": method, "params": params})
                    while True:
                        msg = json.loads((await ws.receive()).data)
                        if msg.get("id") == seq:
                            return msg.get("result", msg)
                await call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
                await call("Page.navigate", url=server)
                await asyncio.sleep(6)
                r = await call("Runtime.evaluate", expression=SETUP.replace("IMAGE", json.dumps(image)),
                               awaitPromise=True, returnByValue=True)
                print("graph bounds/scale:", r.get("result", {}).get("value"), r.get("exceptionDetails", ""))
                shot = await call("Page.captureScreenshot", format="png")
                (HERE / "out").mkdir(exist_ok=True)
                (HERE / "out" / "workflow.png").write_bytes(base64.b64decode(shot["data"]))
                print("wrote", HERE / "out" / "workflow.png")
    finally:
        chrome.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8199")
    ap.add_argument("--image", default="paris_start.jpeg")
    a = ap.parse_args()
    asyncio.run(main(a.server, a.image))
