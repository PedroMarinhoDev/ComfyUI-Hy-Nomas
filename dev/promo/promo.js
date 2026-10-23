// The promo video, as a page rendered one frame at a time: renderFrame(t) puts every element where it is at t
// seconds. Nothing depends on earlier frames, so any frame can be rendered on its own (the renderer splits the
// frames between several browsers). Video clips are pre-extracted JPEG frames (build/frames/<clip>/NNNN.jpg).

const $ = (id) => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeOut = (x) => 1 - Math.pow(1 - x, 3);
const lerp = (a, b, k) => a + (b - a) * k;
const pad = (n) => String(n).padStart(4, "0");
const pending = [];
function setImg(img, src) {
  if (img.dataset.src === src) return;
  img.dataset.src = src;
  img.src = src;
  pending.push(img.decode().catch(() => {}));
}
const raf = () => new Promise((r) => requestAnimationFrame(() => r()));
const XF = 0.4;
const vis = (t, a, b, first = false, last = false) =>
  Math.min(first ? 1 : seg(t, a - XF / 2, a + XF / 2), last ? 1 : 1 - seg(t, b - XF / 2, b + XF / 2));
// moves land on whole device pixels (the page renders at 2x), so text is never drawn between pixels differently
// from frame to frame
const px = (v) => Math.round(v * 2) / 2;
const fadeUp = (el, k, dy = 24) => { el.style.opacity = k; el.style.transform = `translateY(${px((1 - k) * dy)}px)`; };

const data = await (await fetch("data.json")).json();
const CLIPS = data.clips;
const VFPS = 24;
const frameOf = (clip, i) => `frames/${clip}/${pad(clamp(Math.floor(i), 0, CLIPS[clip].frames - 1))}.jpg`;
const smallOf = (clip, i) => `small/${clip}/${pad(Math.floor(i) % CLIPS[clip].small)}.jpg`;
const segOfFrame = (i) => (i < 1 ? 0 : Math.min(4, Math.floor((i - 1) / 120) + 1));   // 1 held frame, then 5 s segments
const mmss = (s) => `0:${String(Math.floor(s)).padStart(2, "0")}`;

// ---- timeline -------------------------------------------------------------------------------------------
const S = { title: [0, 4], image: [4, 14], chain: [14, 24], camera: [24, 34], drive: [34, 40], text: [40, 46],
            speed: [46, 54], comfy: [54, 60], outro: [60, 68] };
const DURATION = 68;
const sfx = [];
for (const k of Object.keys(S).slice(1)) sfx.push({ t: S[k][0] - 0.15, kind: "whoosh" });

// ---- mosaics --------------------------------------------------------------------------------------------
const MOS = ["paris", "beach", "orbit", "fox", "drive"];
function buildMosaic(el) {
  const rows = [];
  for (let r = 0; r < 4; r++) {
    const row = document.createElement("div");
    row.className = "mrow";
    const imgs = [];
    for (let c = 0; c < 6; c++) {
      const img = document.createElement("img");
      row.append(img);
      const clip = MOS[(r * 6 + c * 4 + r) % MOS.length];
      imgs.push({ img, clip, phase: (r * 37 + c * 71) % CLIPS[clip].small });
    }
    el.append(row);
    rows.push({ row, imgs, dir: r % 2 ? 1 : -1 });
  }
  return rows;
}
const mosaics = [buildMosaic($("mosaic1")), buildMosaic($("mosaic2"))];
function updateMosaic(rows, t) {
  for (const { row, imgs, dir } of rows) {
    row.style.transform = `translateX(${-335 + dir * (t * 26 - 60)}px)`;
    for (const m of imgs) setImg(m.img, smallOf(m.clip, m.phase + t * 12));     // the small frames are every 2nd frame
  }
}

// ---- 01 image --------------------------------------------------------------------------------------------
setImg($("i-photo-img"), "start.jpg");
const PARIS_TEXT = ["walks down to the river road", "keeps walking along the river", "stops at the wall to look at the bridge",
                    "the sun breaks through; she walks on"];
function sceneImage(t) {
  const [a] = S.image;
  fadeUp($("s1"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const p = easeOut(seg(t, a + 0.3, a + 1.0));
  $("i-photo").style.opacity = p;
  $("i-photo").style.transform = `translateX(${px((1 - p) * -80)}px)`;
  $("i-label").style.opacity = p;
  $("i-arrow").style.transform = `scaleX(${easeOut(seg(t, a + 1.0, a + 1.5))})`;
  $("i-arrow").style.opacity = seg(t, a + 1.0, a + 1.1);
  const s = easeOut(seg(t, a + 1.35, a + 2.0));
  $("i-screen").style.opacity = s;
  $("i-screen").style.transform = `scale(${lerp(0.94, 1, s)})`;
  const play = Math.max(0, t - (a + 2.0)) * VFPS;
  setImg($("i-video"), frameOf("paris", play));
  fadeUp($("i-chip1"), easeOut(seg(t, a + 2.2, a + 2.7)), 16);
  fadeUp($("i-chip2"), easeOut(seg(t, a + 2.5, a + 3.0)), 16);
  const k = segOfFrame(play);
  $("i-status").innerHTML = k === 0 ? "frame 1 is <b>your picture</b>" :
    `segment ${k} of 4 · <b>${PARIS_TEXT[k - 1]}</b>`;
  $("i-status").style.opacity = seg(t, a + 2.2, a + 2.6);
}

// ---- 02 chain ----------------------------------------------------------------------------------------------
const BEACH_TEXT = ["past the fire pit to the water", "turn right along the shore", "walk the waterline to the deck",
                    "golden light, out to sea"];
const tl = $("c-tl");
const chips = [];
for (let i = 0; i < 5; i++) {
  const d = document.createElement("div");
  d.className = "seg";
  d.style.width = i === 0 ? "180px" : "377px";
  d.innerHTML = `<div class="k">${i === 0 ? "IMAGE" : "SEGMENT " + i}</div><div class="x">${i === 0 ? "your picture" : BEACH_TEXT[i - 1]}</div><div class="fill"></div>`;
  tl.append(d);
  chips.push(d);
}
function sceneChain(t) {
  const [a] = S.chain;
  fadeUp($("s2"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.7));
  $("c-screen").style.opacity = s;
  $("c-screen").style.transform = `scale(${lerp(0.95, 1, s)})`;
  const f = Math.max(0, t - (a + 0.3)) * VFPS * 2;
  setImg($("c-video"), frameOf("beach", f));
  $("c-time").innerHTML = `${mmss(Math.min(f, 480) / VFPS)} / 0:20`;
  const k = segOfFrame(f);
  chips.forEach((c, i) => {
    c.classList.toggle("on", i === k);
    const prog = i < k ? 1 : i > k ? 0 : (i === 0 ? 1 : ((f - 1) - (i - 1) * 120) / 120);
    c.querySelector(".fill").style.width = `${clamp(prog) * 100}%`;
    fadeUp(c, easeOut(seg(t, a + 0.3 + i * 0.12, a + 0.8 + i * 0.12)), 18);
  });
}
for (let i = 1; i <= 4; i++) sfx.push({ t: S.chain[0] + 0.3 + (1 + (i - 1) * 120) / (VFPS * 2), kind: "blip" });

// ---- 03 camera ----------------------------------------------------------------------------------------------
const KEYLAYOUT = [["", "W", ""], ["A", "S", "D"], ["", "↑", ""], ["←", "↓", "→"]];
const KEYNAME = { W: "W", A: "A", S: "S", D: "D", Left: "←", Right: "→", Up: "↑", Down: "↓" };
function buildKeys(el) {
  const cells = {};
  for (const row of KEYLAYOUT) for (const k of row) {
    const d = document.createElement("div");
    if (k) { d.className = "key"; d.textContent = k; cells[k] = d; }
    el.append(d);
  }
  return cells;
}
const camKeys = buildKeys($("k-keys"));
const driveKeys = buildKeys($("d-keys"));
const setKeys = (cells, names) => { for (const [k, d] of Object.entries(cells)) d.classList.toggle("on", names.includes(k)); };
const parseKeys = (timeline) => timeline.split(/\s+/)[0].split("+").map((k) => KEYNAME[k] || k);
const ORBIT = ["D+Left 3", "D+Left 3", "D+Left 3", "S 4"];
const map = $("map"), mg = map.getContext("2d");
function drawMap(f) {
  // camera around the subject: three orbit segments of 60 degrees from behind her to in front, then backing away
  const W = 380, H = 340, cx = W / 2, cy = H / 2 + 10, R = 108;
  mg.setTransform(2, 0, 0, 2, 0, 0);
  mg.clearRect(0, 0, W, H);
  mg.fillStyle = "#151B23"; mg.strokeStyle = "#30363D"; mg.lineWidth = 1;
  mg.beginPath(); mg.roundRect(0.5, 0.5, W - 1, H - 1, 14); mg.fill(); mg.stroke();
  const prog = clamp((f - 1) / 360);                       // orbit progress over segments 1-3
  const back = clamp((f - 361) / 120);                     // segment 4
  const ang = (k) => Math.PI / 2 - k * Math.PI;            // screen y down: start below (behind her), end above
  const pos = (k, r) => [cx + Math.cos(ang(k)) * r, cy + Math.sin(ang(k)) * r];
  // tower and the street
  mg.fillStyle = "#9198A1"; mg.font = "600 19px Oswald"; mg.textAlign = "center";
  mg.fillText("EIFFEL TOWER ↑", cx, 24);
  mg.strokeStyle = "rgba(145,152,161,.35)"; mg.setLineDash([5, 6]);
  mg.beginPath(); mg.arc(cx, cy, R, 0, 2 * Math.PI); mg.stroke(); mg.setLineDash([]);
  // trail
  mg.strokeStyle = "#F2A93B"; mg.lineWidth = 4; mg.lineCap = "round";
  mg.beginPath();
  for (let i = 0; i <= 60; i++) {
    const [x, y] = pos(prog * i / 60, R);
    i ? mg.lineTo(x, y) : mg.moveTo(x, y);
  }
  const r2 = R + back * 70;
  const [bx, by] = pos(prog, r2);
  mg.lineTo(bx, by);
  mg.stroke();
  // subject
  mg.fillStyle = "#F0F6FC"; mg.beginPath(); mg.arc(cx, cy, 9, 0, 2 * Math.PI); mg.fill();
  mg.fillStyle = "#F0F6FC"; mg.font = "500 18px 'IBM Plex Sans'"; mg.fillText("her", cx + 30, cy + 6);
  // camera wedge looking at the subject
  const dir = Math.atan2(cy - by, cx - bx);
  mg.save(); mg.translate(bx, by); mg.rotate(dir);
  mg.fillStyle = "rgba(242,169,59,.22)";
  mg.beginPath(); mg.moveTo(0, 0); mg.lineTo(70, -34); mg.lineTo(70, 34); mg.closePath(); mg.fill();
  mg.fillStyle = "#F2A93B"; mg.beginPath(); mg.roundRect(-14, -10, 24, 20, 4); mg.fill();
  mg.restore();
}
function sceneCamera(t) {
  const [a] = S.camera;
  fadeUp($("s3"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.7));
  $("k-screen").style.opacity = s;
  $("k-screen").style.transform = `scale(${lerp(0.95, 1, s)})`;
  const f = Math.max(0, t - (a + 0.3)) * VFPS * 2;
  setImg($("k-video"), frameOf("orbit", f));
  $("k-time").innerHTML = `${mmss(Math.min(f, 480) / VFPS)} / 0:20`;
  const k = segOfFrame(f);
  setKeys(camKeys, k === 0 ? [] : parseKeys(ORBIT[k - 1]));
  drawMap(Math.min(f, 481));
  $("k-status").innerHTML = k === 0 ? "start: <b>your picture</b>" :
    `segment ${k}: Camera Keys <i>"${ORBIT[k - 1]}"</i>   ·   ${k < 4 ? "she stays put; the camera circles" : "she walks; the camera backs away"}`;
  for (const id of ["k-keys", "map", "k-status"]) $(id).style.opacity = seg(t, a + 0.4, a + 0.9);
}
for (let i = 1; i <= 4; i++) sfx.push({ t: S.camera[0] + 0.3 + (1 + (i - 1) * 120) / (VFPS * 2), kind: "blip2" });

// ---- 04 drive -----------------------------------------------------------------------------------------------
const DRIVE = data.drive;
function sceneDrive(t) {
  const [a] = S.drive;
  fadeUp($("s4"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.6));
  $("d-screen").style.opacity = s;
  $("d-screen").style.transform = `scale(${lerp(0.95, 1, s)})`;
  const f = Math.max(0, t - (a + 0.3)) * VFPS;
  setImg($("d-video"), frameOf("drive", f));
  const turn = f < 1 ? -1 : Math.min(DRIVE.turns.length - 1, Math.floor((f - 1) / 24));
  setKeys(driveKeys, turn < 0 ? [] : parseKeys(DRIVE.turns[turn]));
  $("d-turn").innerHTML = turn < 0 ? "<b>START</b> · your picture" : `<b>TURN ${turn + 1}</b> / ${DRIVE.turns.length}`;
  $("d-status").innerHTML = `Each <b>1 s turn</b> took <i>~${DRIVE.typical} s</i> on an RTX 4090 at 832×480: sample, remember, decode, stream to the node.<br><br>Not real time yet. Faster drive mode is on the roadmap.`;
  for (const id of ["d-keys", "d-status"]) $(id).style.opacity = seg(t, a + 0.4, a + 0.9);
}
DRIVE.turns.forEach((_, i) => sfx.push({ t: S.drive[0] + 0.3 + (1 + i * 24) / VFPS, kind: "click" }));

// ---- 05 text ------------------------------------------------------------------------------------------------
const CODE = [
  `<u>"perspective_view"</u>: <b>"third-person observer"</b>,`,
  `<u>"subject_id"</u>: <i>"&lt;|Subject1|&gt;"</i>,`,
  `<u>"segments"</u>: [`,
  ` { <u>"visual_description_long"</u>:`,
  `     <b>"A flagstone path crosses a luminous`,
  `      forest of tall dark trunks…"</b>,`,
  `   <u>"contained_actions"</u>: [`,
  `     <i>"&lt;Action1&gt;&lt;|Subject1|&gt; walks slowly`,
  `      along the stone path&lt;/Action1&gt;."</i> ] },`,
  ` { <u>"description_change"</u>: <b>"Compared to`,
  `     the previous segment, blue flowers open…"</b> }`,
];
function sceneText(t) {
  const [a] = S.text;
  fadeUp($("s5"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.6));
  $("x-screen").style.opacity = s;
  $("x-screen").style.transform = `scale(${lerp(0.95, 1, s)})`;
  const f = Math.max(0, t - (a + 0.3)) * VFPS * 2;
  setImg($("x-video"), frameOf("fox", f));
  const lines = Math.floor(seg(t, a + 0.8, a + 3.4) * CODE.length + 0.001);
  $("x-code").innerHTML = CODE.slice(0, Math.max(1, lines)).join("\n");
  fadeUp($("x-code"), easeOut(seg(t, a + 0.6, a + 1.1)), 30);
}

// ---- speed --------------------------------------------------------------------------------------------------
const STATS = [
  { v: 1.02, dec: 2, u: "s", d: "per <b>720p</b> sampling step, 8 steps per segment" },
  { v: 34, dec: 0, u: "s", d: "<b>10 s</b> of 832×480 video, prompt to mp4" },
  { v: 3.2, dec: 1, u: "min", pre: "~", d: "<b>20 s at 1280×704</b>, five chained segments" },
  { v: 12, dec: 0, u: "GB", d: "cards keep nearly full speed: <b>93 s</b> vs 89 s on 24 GB" },
];
const statEls = STATS.map((st) => {
  const d = document.createElement("div");
  d.className = "stat";
  d.innerHTML = `<div class="v"><span class="n"></span><small>${st.u}</small></div><div class="d">${st.d}</div>`;
  $("stats").append(d);
  return d;
});
const TECH = ["int8 block-scaled GEMMs (Triton)", "fused LayerNorm · QK-norm · RoPE", "FP8 decoder GEMMs", "int8 memory in RAM",
              "ComfyUI dynamic VRAM"];
const techBox = document.createElement("div");
techBox.className = "tech";
$("L-speed").append(techBox);
const techEls = TECH.map((x) => { const d = document.createElement("div"); d.className = "chip"; d.textContent = x; techBox.append(d); return d; });
function sceneSpeed(t) {
  const [a] = S.speed;
  fadeUp($("s6"), easeOut(seg(t, a + 0.05, a + 0.7)));
  STATS.forEach((st, i) => {
    const k = easeOut(seg(t, a + 0.3 + i * 0.3, a + 0.8 + i * 0.3));
    statEls[i].style.opacity = k;
    statEls[i].style.transform = `translateY(${px((1 - k) * 30)}px)`;
    const c = ease(seg(t, a + 0.4 + i * 0.3, a + 1.6 + i * 0.3));
    statEls[i].querySelector(".n").textContent = `${st.pre || ""}${(st.v * c).toFixed(st.dec)}`;
  });
  const MAXW = 1240;
  const b = ease(seg(t, a + 2.2, a + 3.6));
  $("bars").style.opacity = seg(t, a + 2.0, a + 2.4);
  $("b1").style.width = `${MAXW * b}px`;
  $("b2").style.width = `${MAXW * (1.02 / 2.29) * b}px`;
  $("b1v").style.left = `${260 + MAXW * b + 18}px`;
  $("b2v").style.left = `${260 + MAXW * (1.02 / 2.29) * b + 18}px`;
  $("b1v").textContent = `${(2.29 * b).toFixed(2)} s`;
  $("b2v").innerHTML = `${(1.02 * b).toFixed(2)} s  <span style="color:var(--accent)">${b > 0.98 ? "· 2.2× FASTER" : ""}</span>`;
  techEls.forEach((e, i) => fadeUp(e, easeOut(seg(t, a + 3.8 + i * 0.18, a + 4.3 + i * 0.18)), 14));
}
STATS.forEach((_, i) => sfx.push({ t: S.speed[0] + 0.3 + i * 0.3, kind: "blip" }));
sfx.push({ t: S.speed[0] + 3.6, kind: "success" });

// ---- comfy --------------------------------------------------------------------------------------------------
setImg($("w-img"), "workflow.png");
const FEATS = [
  ["Start From Image", "your picture as the first memory"],
  ["Camera recorder in the node", "fly a path with W/A/S/D"],
  ["Drive mode", "frames stream into the node"],
  ["Timed actions", "per-action attention masks"],
  ["Checked against Tencent's code", "block by block"],
  ["5 example workflows", "with read-me notes"],
];
const featEls = FEATS.map(([a, b]) => {
  const d = document.createElement("div");
  d.className = "feat";
  d.innerHTML = `<span class="dot"></span><span>${a}<br><small>${b}</small></span>`;
  $("w-feats").append(d);
  return d;
});
function sceneComfy(t) {
  const [a, b] = S.comfy;
  fadeUp($("s7"), easeOut(seg(t, a + 0.05, a + 0.7)));
  const s = easeOut(seg(t, a + 0.1, a + 0.8));
  $("w-shot").style.opacity = s;
  $("w-shot").style.transform = `scale(${lerp(0.96, 1.0, s)})`;
  $("w-img").style.transformOrigin = "56% 47%";
  $("w-img").style.transform = `scale(${lerp(1.3, 1.42, seg(t, a, b))})`;
  featEls.forEach((e, i) => fadeUp(e, easeOut(seg(t, a + 0.6 + i * 0.25, a + 1.1 + i * 0.25)), 16));
}
FEATS.forEach((_, i) => sfx.push({ t: S.comfy[0] + 0.6 + i * 0.25, kind: "blip2" }));

// ---- title and outro ---------------------------------------------------------------------------------------------
function sceneTitle(t) {
  updateMosaic(mosaics[0], t);
  fadeUp($("t-kicker"), easeOut(seg(t, 0.2, 0.8)), 16);
  const k = easeOut(seg(t, 0.4, 1.2));
  $("t-big").style.opacity = k;
  $("t-big").style.transform = `scale(${lerp(1.12, 1, k)})`;
  fadeUp($("t-tag"), easeOut(seg(t, 1.1, 1.7)), 16);
  fadeUp($("t-wip"), easeOut(seg(t, 1.7, 2.2)), 10);
}
function sceneOutro(t) {
  updateMosaic(mosaics[1], t);
}
sfx.push({ t: 0.4, kind: "whoosh" }, { t: 1.7, kind: "blip" });

// ---- frames -------------------------------------------------------------------------------------------------
const LAYERS = [["L-title", S.title, sceneTitle], ["L-image", S.image, sceneImage], ["L-chain", S.chain, sceneChain],
                ["L-camera", S.camera, sceneCamera], ["L-drive", S.drive, sceneDrive], ["L-text", S.text, sceneText],
                ["L-speed", S.speed, sceneSpeed], ["L-comfy", S.comfy, sceneComfy], ["L-outro", S.outro, sceneOutro]];
window.renderFrame = async (t) => {
  LAYERS.forEach(([id, [a, b], fn], i) => {
    const v = vis(t, a, b, i === 0, i === LAYERS.length - 1);
    $(id).style.opacity = v;
    $(id).style.display = v > 0 ? "" : "none";
    if (v > 0) fn(t);
  });
  // fade in from and out to black
  document.body.style.filter = `brightness(${Math.min(seg(t, 0, 0.5), 1 - seg(t, DURATION - 0.9, DURATION))})`;
  while (pending.length) await Promise.all(pending.splice(0));
  await raf();
  await raf();
};
window.PROMO = { duration: DURATION, sfx: sfx.sort((a, b) => a.t - b.t), sections: S };
await document.fonts.ready;
// the counting numbers get the width of their final value, right-aligned, so the unit beside them stays put
STATS.forEach((st, i) => {
  const n = statEls[i].querySelector(".n");
  n.textContent = `${st.pre || ""}${st.v.toFixed(st.dec)}`;
  n.style.width = `${Math.ceil(n.getBoundingClientRect().width) + 2}px`;
});
await renderFrame(0);
window.promoReady = true;
