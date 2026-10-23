"""The promo video for X and Reddit: 1920x1080, 60 fps, 68 s, with synthesized music and sound effects.

    python dev/promo/promo_video.py --stills 2,8,18,28,36,43,50,57,63    # check frames (out/still_*.png)
    python dev/promo/promo_video.py                                     # out/hy-nomas-promo.mp4

The picture is promo.html + promo.js rendered frame by frame by headless Chrome; every frame is a pure function of
its time, so the frames are split between several Chrome processes (--workers) and written as JPEGs, then encoded
with NVENC (h264_nvenc on the GPU given by --gpu, CUDA's PCI bus order; libx264 if NVENC is missing). The clips are
the sample videos (see CLIPS), pre-extracted as frames. The music is synthesized here (the same synth as
ComfyUI-SceneCut's promo, re-timed to these sections). Needs ComfyUI's Python (aiohttp, numpy, scipy),
google-chrome and ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD, OUT = HERE / "build", HERE / "out"
LAB = Path(os.environ.get("HY_NOMAS_LAB", Path.home() / "AI/hy-nomas-lab"))
VIDEOS = LAB / "server/output"
CLIPS = {                                                          # clip name -> rendered sample (server output)
    "paris": "hynomas/showcase_paris_720p_00001_.mp4",
    "beach": "hynomas/showcase_beach_720p_00001_.mp4",
    "orbit": "hynomas/paris_orbit_keys_00001_.mp4",
    "fox": "hynomas/showcase_fox_720p_00001_.mp4",
    "drive": "hy_nomas/drive_00007_.mp4",                          # drive_capture.py --seed 11 (she keeps her look)
}
START_IMAGE = LAB / "server/input/paris_start.jpeg"
FONTS = Path.home() / "Projects" / "music-visualizer" / "public" / "fonts"
PLEX = Path("/usr/share/fonts/truetype/ibm-plex")
FPS, W, H = 60, 1920, 1080
SR = 48000
PORT, CDP0 = 8195, 9340


# --------------------------------------------------------------------------------------------- #
# Page assets
# --------------------------------------------------------------------------------------------- #
def clip_path(name: str) -> Path:
    return VIDEOS / CLIPS[name]


def extract(src: Path, dst: Path, vf: str) -> int:
    if not (dst.exists() and any(dst.iterdir())):
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-q:v", "2", "-fps_mode", "passthrough",
                        "-start_number", "0", str(dst / "%04d.jpg")], check=True)
    return len(list(dst.glob("*.jpg")))


def build() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("promo.html", "promo.js"):
        shutil.copyfile(HERE / name, BUILD / name)
    (BUILD / "fonts").mkdir(exist_ok=True)
    shutil.copyfile(FONTS / "anton-latin-400-normal.woff2", BUILD / "fonts" / "anton.woff2")
    shutil.copyfile(FONTS / "oswald-latin-wght-normal.woff2", BUILD / "fonts" / "oswald.woff2")
    for f in ("IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold", "IBMPlexSans-Bold",
              "IBMPlexMono-Regular", "IBMPlexMono-Medium"):
        shutil.copyfile(PLEX / f"{f}.ttf", BUILD / "fonts" / f"{f}.ttf")
    clips = {}
    for name in CLIPS:
        src = clip_path(name)
        size = "1248:720" if name == "drive" else "1280:704"
        frames = extract(src, BUILD / "frames" / name, f"scale={size}:flags=lanczos")
        small = extract(src, BUILD / "small" / name, "select='not(mod(n\\,2))',scale=420:231:flags=lanczos")
        clips[name] = {"frames": frames, "small": small, "source": src.name}
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(START_IMAGE), "-vf", "scale=1120:-1:flags=lanczos", "-q:v", "2",
                    str(BUILD / "start.jpg")], check=True)
    shot = HERE / "out" / "workflow.png"
    if not shot.exists():
        sys.exit("make the workflow screenshot first: python dev/promo/comfy_shot.py")
    shutil.copyfile(shot, BUILD / "workflow.png")
    keys = json.loads((OUT / "drive_keys.json").read_text())
    typical = float(np.median(keys["wall"][1:])) if len(keys["wall"]) > 1 else keys["wall"][0]
    drive = {"turns": keys["turns"], "typical": round(typical, 1)}
    (BUILD / "data.json").write_text(json.dumps({"clips": clips, "drive": drive}, indent=1))
    print("clips:", {k: v["frames"] for k, v in clips.items()}, "drive turn", drive["typical"], "s")


def serve() -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(BUILD))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


# --------------------------------------------------------------------------------------------- #
# Rendering through the DevTools protocol
# --------------------------------------------------------------------------------------------- #
class Page:
    def __init__(self, ws):
        self.ws, self.seq = ws, 0

    async def call(self, method, **params):
        self.seq += 1
        await self.ws.send_json({"id": self.seq, "method": method, "params": params})
        while True:
            msg = json.loads((await self.ws.receive()).data)
            if msg.get("id") == self.seq:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]
            if msg.get("method") == "Runtime.exceptionThrown":
                print("page error:", msg["params"]["exceptionDetails"].get("exception", {}).get("description"), file=sys.stderr)
            if msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in ("error", "warning"):
                print("console:", [a.get("value", a.get("description")) for a in msg["params"]["args"]], file=sys.stderr)

    async def eval(self, expr, timeout=120000):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r))
        return r["result"].get("value")

    async def shot(self, fmt="jpeg") -> bytes:
        params = {"format": fmt, "clip": {"x": 0, "y": 0, "width": W, "height": H, "scale": 0.5}}
        if fmt == "jpeg":
            params["quality"] = 94
        r = await self.call("Page.captureScreenshot", **params)
        return base64.b64decode(r["data"])


async def with_page(fn, port: int):
    import aiohttp

    profile = tempfile.mkdtemp()
    # no --disable-gpu: headless Chrome composites on the Intel iGPU of this machine (the NVIDIA cards stay free)
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--hide-scrollbars", f"--remote-debugging-port={port}",
                               f"--user-data-dir={profile}", f"--window-size={W},{H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as http_:
            for _ in range(200):
                try:
                    pages = await (await http_.get(f"http://127.0.0.1:{port}/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.1)
            target = next(p for p in pages if p["type"] == "page")
            async with http_.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                page = Page(ws)
                await page.call("Runtime.enable")
                await page.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
                await page.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/promo.html")
                for _ in range(600):
                    if await page.eval("window.promoReady === true"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("the page did not get ready")
                return await fn(page)
    finally:
        chrome.terminate()
        shutil.rmtree(profile, ignore_errors=True)


async def render_stills(times: list[float]) -> None:
    async def go(page):
        for t in times:
            await page.eval(f"renderFrame({t})")
            (OUT / f"still_{t:05.2f}.png").write_bytes(await page.shot("png"))
            print(f"still {t}")
    await with_page(go, CDP0)


async def render_frames(workers: int) -> dict:
    frames_dir = OUT / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    meta_box = {}
    t0 = time.time()
    done = [0]

    async def worker(k):
        async def go(page):
            meta = await page.eval("window.PROMO")
            meta_box["meta"] = meta
            n = int(round(meta["duration"] * FPS))
            for f in range(k, n, workers):
                await page.eval(f"renderFrame({f / FPS})")
                (frames_dir / f"{f:05d}.jpg").write_bytes(await page.shot())
                done[0] += 1
                if k == 0 and (f // workers) % 30 == 0:
                    el = time.time() - t0
                    print(f"{done[0]}/{n} frames, {el:.0f}s elapsed, ~{el / max(1, done[0]) * (n - done[0]):.0f}s left",
                          flush=True)
        await with_page(go, CDP0 + k)

    await asyncio.gather(*(worker(k) for k in range(workers)))
    print(f"rendered {done[0]} frames in {time.time() - t0:.0f}s with {workers} browsers")
    return meta_box["meta"]


def encode(frames_dir: Path, audio: Path, out: Path, gpu: int) -> None:
    has_nvenc = "h264_nvenc" in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    common = ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.jpg"), "-i", str(audio),
              "-map", "0:v", "-map", "1:a"]
    if has_nvenc:
        video = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "16M",
                 "-maxrate", "28M", "-bufsize", "40M", "-profile:v", "high", "-bf", "3", "-spatial-aq", "1"]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    else:
        video = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high"]
        env = None
    t0 = time.time()
    subprocess.run(common + video + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                                     "-movflags", "+faststart", str(out)], check=True, env=env)
    print(f"encoded with {'h264_nvenc' if has_nvenc else 'libx264'} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------------------------- #
# Music and sound effects (ComfyUI-SceneCut's synth, re-timed)
# --------------------------------------------------------------------------------------------- #
BPM = 120
BEAT = 60 / BPM
NOTE = {"A2": 110.0, "B2": 123.47, "C3": 130.81, "D3": 146.83, "E3": 164.81, "F2": 87.31, "G2": 98.0, "F3": 174.61,
        "G3": 196.0, "A3": 220.0, "B3": 246.94, "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.0,
        "A4": 440.0, "B4": 493.88, "C5": 523.25, "D5": 587.33, "E5": 659.26, "G5": 783.99, "A5": 880.0}
# A minor: Am - F - C - G, two bars each
CHORDS = [("A2", ["A3", "C4", "E4"], ["A4", "C5", "E5", "C5"]), ("F2", ["F3", "A3", "C4"], ["A4", "C5", "F4", "C5"]),
          ("C3", ["G3", "C4", "E4"], ["G4", "C5", "E5", "C5"]), ("G2", ["G3", "B3", "D4"], ["G4", "B4", "D5", "B4"])]
DRUMS = (4.0, 60.0)                       # drums play here
FULL = ((14.0, 34.0), (46.0, 54.0))       # the busier sections
BREAK = (54.0, 60.0)                      # half-time kick
CRASHES = (4.0, 14.0, 34.0, 46.0, 60.0)
RISERS = (14.0, 46.0)


def lowpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "low", fs=SR, output="sos"), x)


def highpass(x, cutoff, order=2):
    from scipy.signal import butter, sosfilt

    return sosfilt(butter(order, cutoff, "high", fs=SR, output="sos"), x)


def saw(f, t, detune=0.0):
    ph = (f * (2 ** (detune / 1200))) * t
    return 2 * (ph - np.floor(ph + 0.5))


def add(buf, start, sig, gain=1.0, pan=0.0):
    i = int(round(start * SR))
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    j = min(buf.shape[1], i + len(sig))
    s = sig[: j - i] * gain
    buf[0, i:j] += s * np.sqrt(0.5 * (1 - pan))
    buf[1, i:j] += s * np.sqrt(0.5 * (1 + pan))


def env(n, a, d, sustain=0.0, r=0.0, hold=None):
    t = np.arange(n) / SR
    out = np.where(t < a, t / max(a, 1e-4), sustain + (1 - sustain) * np.exp(-(t - a) / max(d, 1e-4)))
    if r and hold is not None:
        out *= np.clip(1 - (t - hold) / r, 0, 1) ** 2 * (t > hold) + (t <= hold)
    return out


def kick():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.20)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t / 0.004) * 0.25
    return np.tanh(1.6 * (body + highpass(click, 1500)))


def hat(open_=False, seed=2):
    n = int((0.25 if open_ else 0.06) * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    return highpass(noise, 7000, 4) * np.exp(-t / (0.08 if open_ else 0.018))


def clap(seed=3):
    from scipy.signal import butter, sosfilt

    n = int(0.5 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    e = np.zeros(n)
    for k, dt in enumerate((0, 0.009, 0.018, 0.027)):
        i = int(dt * SR)
        e[i:] += np.exp(-(t[: n - i]) / (0.008 if k < 3 else 0.12))
    return sosfilt(butter(2, [900, 4200], "band", fs=SR, output="sos"), rng.standard_normal(n) * e) * 1.6


def crash(seed=4, length=2.4):
    n = int(length * SR)
    t = np.arange(n) / SR
    return highpass(np.random.default_rng(seed).standard_normal(n), 3500, 2) * np.exp(-t / 0.7) * 0.5


def riser(length=1.6, seed=5):
    n = int(length * SR)
    t = np.arange(n) / SR
    noise = np.random.default_rng(seed).standard_normal(n)
    ramp = (t / length) ** 2
    sweep = np.sin(2 * np.pi * np.cumsum(300 + 1400 * (t / length) ** 2) / SR) * 0.15
    return (highpass(noise, 1200) * 0.5 + sweep) * ramp


def pluck(f, length=0.32):
    n = int(length * SR)
    t = np.arange(n) / SR
    x = 0.6 * np.sign(np.sin(2 * np.pi * f * t)) * 0.4 + 0.6 * saw(f, t)
    return lowpass(x, min(9000, f * 6)) * np.exp(-t / 0.11)


def music(duration: float) -> np.ndarray:
    n = int(duration * SR) + SR
    mix = {k: np.zeros((2, n)) for k in ("pad", "bass", "drums", "arp", "fx")}
    bar = 4 * BEAT
    end = DRUMS[1]
    drums_on = lambda s: DRUMS[0] <= s < DRUMS[1]                          # noqa: E731
    full = lambda s: any(a <= s < b for a, b in FULL)                     # noqa: E731
    in_break = lambda s: BREAK[0] <= s < BREAK[1]                         # noqa: E731
    for b in range(0, int(duration / bar) + 1, 2):
        s0 = b * bar
        if s0 >= duration:
            break
        root, pad_notes, _ = CHORDS[(b // 2) % 4]
        length = 2 * bar
        m = int(length * SR) + int(0.6 * SR)
        tt = np.arange(m) / SR
        sig = sum(saw(NOTE[p], tt, d) for p in pad_notes for d in (-7, 0, 7)) / 9
        sig = lowpass(sig, 900 if s0 < 4 else 1800, 2) * env(m, 0.35, 10, 0.85, 0.5, length)
        if s0 >= end:
            sig *= np.clip(1 - tt / 7.5, 0, 1) ** 1.5
        add(mix["pad"], s0, sig, 0.8, 0.0)
        if drums_on(s0):
            for k in range(int(length / (BEAT / 2))):
                st = s0 + k * BEAT / 2
                if st >= end:
                    break
                bn = int(0.24 * SR)
                bt = np.arange(bn) / SR
                f = NOTE[root]
                bsig = (np.sin(2 * np.pi * f * bt) + 0.35 * saw(f, bt)) * np.exp(-bt / 0.16)
                add(mix["bass"], st, lowpass(np.tanh(1.4 * bsig), 900), 0.4 if k % 2 else 0.5)
    for k in range(int(duration / (BEAT / 4))):
        st = k * BEAT / 4
        if st >= end + 0.5:
            break
        b = int(st // bar)
        _, _, arp = CHORDS[(b // 2) % 4]
        f = NOTE[arp[k % 4]] * (2 if (k // 8) % 4 == 3 and full(st) else 1)
        level = 0.2 if st < 4 else (0.3 if not full(st) else 0.36)
        add(mix["arp"], st, pluck(f), level, -0.45 if k % 2 else 0.45)
    K, CL = kick(), clap()
    HH = hat()
    for k in range(int(duration / BEAT)):
        st = k * BEAT
        if not drums_on(st):
            continue
        if not in_break(st) or k % 2 == 0:
            add(mix["drums"], st, K, 0.62)
        add(mix["drums"], st + BEAT / 2, HH, 0.22, 0.25)
        if full(st):
            add(mix["drums"], st + BEAT / 4, hat(seed=7 + k % 5), 0.09, -0.3)
            add(mix["drums"], st + 3 * BEAT / 4, hat(seed=11 + k % 3), 0.09, -0.3)
            if k % 2 == 1:
                add(mix["drums"], st, CL, 0.45, 0.05)
    for s in CRASHES:
        add(mix["fx"], s, crash(seed=int(s)), 0.55)
    for s in RISERS:
        r = riser()
        add(mix["fx"], s - len(r) / SR, r, 0.35)
    duck = np.ones(n)
    for k in range(int(duration / BEAT)):
        st = k * BEAT
        if drums_on(st) and not (in_break(st) and k % 2):
            i = int(st * SR)
            m = min(n - i, int(0.35 * SR))
            duck[i:i + m] = np.minimum(duck[i:i + m], 1 - 0.4 * np.exp(-np.arange(m) / SR / 0.09))
    for key in ("pad", "bass"):
        mix[key] *= duck
    out = sum(mix.values())
    return out[:, :int(duration * SR)]


def sound_effects(events: list[dict], duration: float) -> np.ndarray:
    n = int(duration * SR)
    out = np.zeros((2, n))
    rng = np.random.default_rng(9)
    for e in events:
        t, kind = e["t"], e["kind"]
        if kind == "click":
            m = int(0.05 * SR)
            tt = np.arange(m) / SR
            sig = (np.sin(2 * np.pi * 2300 * tt) * np.exp(-tt / 0.006) + highpass(rng.standard_normal(m), 3000) * np.exp(-tt / 0.003) * 0.4)
            add(out, t, sig, 0.35)
        elif kind in ("blip", "blip2"):
            m = int(0.09 * SR)
            tt = np.arange(m) / SR
            f = 1760 if kind == "blip" else 2349
            add(out, t, np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.025), 0.12)
        elif kind == "success":
            for k, f in enumerate((1318.5, 1760.0)):
                m = int(0.7 * SR)
                tt = np.arange(m) / SR
                sig = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt / 0.22)
                add(out, t + k * 0.09, sig, 0.16)
        elif kind == "whoosh":
            m = int(0.55 * SR)
            tt = np.arange(m) / SR
            bell = np.sin(np.pi * tt / (m / SR)) ** 2
            add(out, t - 0.2, lowpass(rng.standard_normal(m), 2500) * bell, 0.10)
    return out


def write_audio(meta: dict, path: Path) -> None:
    import wave

    dur = meta["duration"]
    mixdown = 0.7 * music(dur) + sound_effects(meta["sfx"], dur)
    mixdown = np.tanh(mixdown * 1.1) / np.tanh(1.1)
    fade = int(1.2 * SR)
    mixdown[:, -fade:] *= np.linspace(1, 0, fade) ** 2
    mixdown /= max(1e-6, np.abs(mixdown).max()) / 0.9
    raw = path.with_suffix(".raw.wav")
    with wave.open(str(raw), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mixdown.T * 32767).astype("<i2").tobytes())
    # two-pass loudness normalisation to -14 LUFS (what X, Reddit and YouTube play at), true peak -1.5 dB
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(raw), "-af",
                          "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    j = json.loads(log[log.rindex("{"):log.rindex("}") + 1])
    af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", af, "-ar", str(SR), str(path)], check=True)
    raw.unlink()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stills", help="comma-separated times to render as PNG instead of the video")
    p.add_argument("--workers", type=int, default=8, help="Chrome processes rendering frames in parallel")
    p.add_argument("--gpu", type=int, default=1, help="NVENC device (nvidia-smi index)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build()
    server = serve()
    try:
        if a.stills:
            asyncio.run(render_stills([float(x) for x in a.stills.split(",")]))
            return
        meta = asyncio.run(render_frames(a.workers))
        (OUT / "meta.json").write_text(json.dumps(meta))
        write_audio(meta, OUT / "audio.wav")
        encode(OUT / "frames", OUT / "audio.wav", OUT / "hy-nomas-promo.mp4", a.gpu)
        print(f"wrote {OUT / 'hy-nomas-promo.mp4'}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
