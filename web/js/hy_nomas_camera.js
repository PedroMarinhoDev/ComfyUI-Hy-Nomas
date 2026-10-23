// Hy-Nomas camera recorder: a camera path drawn with the keys of Tencent's web demo, inside the
// "Hy-Nomas Camera Recording" node.
//
// The controls and the recording format follow apps/frontend/src/components/CameraPathDrawer.tsx: a three.js-style
// camera starting at (0, 1.5, 10) looking at (0, 1.5, 0), FOV 55, W/A/S/D move at 2.8 units/s (7 with Shift),
// arrows turn at 20 degrees/s, dragging turns, sampled at 24 fps into {frames: [{position, quaternion, fov}]}.
// The simulation runs in fixed 1/24 s steps, so a recording is the same whatever the screen's frame rate, and the
// same as the Camera Keys node gives for the same key timeline. The node's recording_json widget holds the result.
const { app } = window.comfyAPI.app

const FPS = 24
const STEP = 1 / FPS
const MOVE_SPEED = 2.8
const RUN_SPEED = 7
const TURN_SPEED = (20 * Math.PI) / 180
const LOOK = 0.004 * 0.6
const FOV = 55
const START = [0, 1.5, 10]
const MAX_SECONDS = 60
const PITCH_LIMIT = (85 * Math.PI) / 180
const CONTROL_KEYS = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']
const KEY_LABELS = { KeyW: 'W', KeyA: 'A', KeyS: 'S', KeyD: 'D', ArrowLeft: '←', ArrowUp: '↑', ArrowRight: '→', ArrowDown: '↓', Shift: '⇧' }

// --- camera math (three.js conventions: Euler 'YXZ', camera looks down -z) ---

function rotation(yaw, pitch) {
  // R = Ry(yaw) * Rx(pitch), row-major 3x3
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch)
  return [cy, sy * sp, sy * cp, 0, cp, -sp, -sy, cy * sp, cy * cp]
}

function quaternion(yaw, pitch) {
  // three.js Quaternion.setFromEuler(new Euler(pitch, yaw, 0, 'YXZ'))
  const c1 = Math.cos(pitch / 2), s1 = Math.sin(pitch / 2), c2 = Math.cos(yaw / 2), s2 = Math.sin(yaw / 2)
  return { x: s1 * c2, y: c1 * s2, z: -s1 * s2, w: c1 * c2 }
}

function eulerFromQuaternion(q) {
  // inverse of the above for a roll-free camera
  const n = Math.hypot(q.x, q.y, q.z, q.w) || 1
  const x = q.x / n, y = q.y / n, z = q.z / n, w = q.w / n
  const m12 = 2 * (y * z - w * x) // R[1][2] = -sin(pitch)
  const pitch = Math.asin(Math.max(-1, Math.min(1, -m12)))
  const yaw = Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y))
  return { yaw, pitch }
}

function forward(yaw, pitch) {
  const r = rotation(yaw, pitch)
  return [-r[2], -r[5], -r[8]]
}

class Pose {
  constructor() { this.reset() }
  reset() { this.pos = [...START]; this.yaw = 0; this.pitch = 0 }
  step(keys, dt) {
    const d = forward(this.yaw, this.pitch)
    let right = [-d[2], 0, d[0]] // cross(d, up)
    const len = Math.hypot(right[0], right[2])
    right = len > 1e-9 ? [right[0] / len, 0, right[2] / len] : [1, 0, 0]
    const speed = keys.has('Shift') ? RUN_SPEED : MOVE_SPEED
    const add = (v, s) => { for (let i = 0; i < 3; i++) this.pos[i] += v[i] * s }
    if (keys.has('KeyW')) add(d, speed * dt)
    if (keys.has('KeyS')) add(d, -speed * dt)
    if (keys.has('KeyA')) add(right, -speed * dt)
    if (keys.has('KeyD')) add(right, speed * dt)
    if (keys.has('ArrowLeft')) this.yaw += TURN_SPEED * dt
    if (keys.has('ArrowRight')) this.yaw -= TURN_SPEED * dt
    if (keys.has('ArrowUp')) this.pitch += TURN_SPEED * dt
    if (keys.has('ArrowDown')) this.pitch -= TURN_SPEED * dt
    this.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pitch))
  }
  frame() {
    const [x, y, z] = this.pos
    return { position: { x, y, z }, quaternion: quaternion(this.yaw, this.pitch), fov: FOV }
  }
  setFrame(f) {
    this.pos = [f.position.x, f.position.y, f.position.z]
    const e = eulerFromQuaternion(f.quaternion)
    this.yaw = e.yaw
    this.pitch = e.pitch
  }
}

function isStatic(a, b) {
  const dp = Math.hypot(a.position.x - b.position.x, a.position.y - b.position.y, a.position.z - b.position.z)
  const qa = a.quaternion, qb = b.quaternion
  const dot = Math.abs(qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w)
  return dp <= 1e-5 && 2 * Math.acos(Math.min(1, dot)) <= 1e-4
}

// trimTrailingStaticCameraFrames: drop the still tail after the last move, keep one frame at the final pose
function trimTail(frames) {
  let end = frames.length
  while (end > 1 && isStatic(frames[end - 2], frames[end - 1])) end -= 1
  return frames.slice(0, end)
}

// --- the scene: ground grid, a subject at the origin, a ring of posts ---

const POSTS = []
for (let i = 0; i < 12; i++) {
  const a = (i / 12) * Math.PI * 2
  POSTS.push([Math.cos(a) * 14, Math.sin(a) * 14, i % 3 === 0 ? 3.2 : 2.2])
}
for (let i = 0; i < 6; i++) POSTS.push([-24 + i * 9.5, -30, 4])

class View {
  constructor(canvas) { this.canvas = canvas; this.ctx = canvas.getContext('2d') }

  project(p, pose, w, h) {
    const r = rotation(pose.yaw, pose.pitch)
    const dx = p[0] - pose.pos[0], dy = p[1] - pose.pos[1], dz = p[2] - pose.pos[2]
    // camera space = R^T * (p - pos)
    return [r[0] * dx + r[3] * dy + r[6] * dz, r[1] * dx + r[4] * dy + r[7] * dz, r[2] * dx + r[5] * dy + r[8] * dz]
  }

  draw(pose, frames, recording) {
    const c = this.canvas, ctx = this.ctx, w = c.width, h = c.height
    if (!w || !h) return
    const f = 1 / Math.tan((FOV * Math.PI) / 360)
    const half = h / 2
    const toScreen = (q) => [w / 2 + (q[0] / -q[2]) * f * half, h / 2 - (q[1] / -q[2]) * f * half]
    const near = 0.05
    const line = (a, b, style, width) => {
      let qa = this.project(a, pose), qb = this.project(b, pose)
      if (qa[2] > -near && qb[2] > -near) return
      if (qa[2] > -near || qb[2] > -near) {
        const t = (-near - qa[2]) / (qb[2] - qa[2])
        const m = [qa[0] + t * (qb[0] - qa[0]), qa[1] + t * (qb[1] - qa[1]), -near]
        if (qa[2] > -near) qa = m
        else qb = m
      }
      const sa = toScreen(qa), sb = toScreen(qb)
      ctx.strokeStyle = style
      ctx.lineWidth = width
      ctx.beginPath()
      ctx.moveTo(sa[0], sa[1])
      ctx.lineTo(sb[0], sb[1])
      ctx.stroke()
    }
    // sky and ground split at the horizon
    const horizon = h / 2 + Math.tan(pose.pitch) * f * half
    const sky = ctx.createLinearGradient(0, 0, 0, Math.max(1, horizon))
    sky.addColorStop(0, '#20324a')
    sky.addColorStop(1, '#4d6d8f')
    ctx.fillStyle = '#1d2024'
    ctx.fillRect(0, 0, w, h)
    ctx.fillStyle = sky
    ctx.fillRect(0, 0, w, Math.max(0, Math.min(h, horizon)))
    // grid, fading with distance from the camera
    for (let i = -40; i <= 40; i += 2) {
      const fade = (v) => `rgba(150, 170, 190, ${Math.max(0.06, 0.45 - Math.abs(v) / 90)})`
      line([i, 0, -40], [i, 0, 40], fade(i - pose.pos[0]), i === 0 ? 1.5 : 1)
      line([-40, 0, i], [40, 0, i], fade(i - pose.pos[2]), i === 0 ? 1.5 : 1)
    }
    // posts and the subject
    const depth = (p) => -this.project(p, pose)[2]
    for (const [x, z, height] of POSTS) {
      const d = Math.max(0.5, depth([x, height / 2, z]))
      line([x, 0, z], [x, height, z], '#9aa7b4', Math.max(1, 30 / d))
    }
    const box = [[-0.6, 0, -0.6], [0.6, 0, -0.6], [0.6, 0, 0.6], [-0.6, 0, 0.6]]
    for (let i = 0; i < 4; i++) {
      const a = box[i], b = box[(i + 1) % 4]
      for (const y of [0, 1.2]) line([a[0], y, a[2]], [b[0], y, b[2]], '#e08a3c', 2)
      line(a, [a[0], 1.2, a[2]], '#e08a3c', 2)
    }
    // the recorded path on the ground
    for (let i = 1; i < frames.length; i++) {
      const a = frames[i - 1].position, b = frames[i].position
      line([a.x, 0.02, a.z], [b.x, 0.02, b.z], recording ? '#ff5a5a' : '#5ad1ff', 2.5)
    }
  }
}

function drawMap(canvas, pose, frames, recording) {
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height
  ctx.fillStyle = 'rgba(15, 18, 22, 0.82)'
  ctx.fillRect(0, 0, w, h)
  const xs = [0, pose.pos[0], ...frames.map((f) => f.position.x)]
  const zs = [0, pose.pos[2], ...frames.map((f) => f.position.z)]
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...zs) + Math.max(...zs)) / 2
  const span = Math.max(12, Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs)) * 1.25
  const s = Math.min(w, h) / span
  const map = (x, z) => [w / 2 + (x - cx) * s, h / 2 + (z - cz) * s]
  ctx.fillStyle = '#e08a3c'
  const o = map(0, 0)
  ctx.fillRect(o[0] - 3, o[1] - 3, 6, 6)
  if (frames.length > 1) {
    ctx.strokeStyle = recording ? '#ff5a5a' : '#5ad1ff'
    ctx.lineWidth = 2
    ctx.beginPath()
    frames.forEach((f, i) => {
      const p = map(f.position.x, f.position.z)
      if (i === 0) ctx.moveTo(p[0], p[1])
      else ctx.lineTo(p[0], p[1])
    })
    ctx.stroke()
  }
  const p = map(pose.pos[0], pose.pos[2])
  const d = forward(pose.yaw, 0)
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  ctx.moveTo(p[0] + d[0] * 9, p[1] + d[2] * 9)
  ctx.lineTo(p[0] - d[2] * 4 - d[0] * 3, p[1] + d[0] * 4 - d[2] * 3)
  ctx.lineTo(p[0] + d[2] * 4 - d[0] * 3, p[1] - d[0] * 4 - d[2] * 3)
  ctx.closePath()
  ctx.fill()
}

// --- the widget ---

const STYLE = `
.hynomas-cam { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; outline: none;
  font: 12px/1.3 system-ui, sans-serif; color: #d6dbe1; box-sizing: border-box; }
.hynomas-cam:focus-within .hynomas-view-wrap { box-shadow: 0 0 0 2px #5ad1ff; }
.hynomas-view-wrap { position: relative; flex: 1 1 auto; min-height: 120px; border-radius: 6px; overflow: hidden;
  background: #1d2024; cursor: grab; }
.hynomas-view-wrap canvas.view { width: 100%; height: 100%; display: block; }
.hynomas-view-wrap canvas.map { position: absolute; right: 6px; top: 6px; width: 96px; height: 96px;
  border-radius: 4px; }
.hynomas-hud { position: absolute; left: 8px; top: 6px; padding: 2px 6px; border-radius: 4px;
  background: rgba(0,0,0,0.45); font-variant-numeric: tabular-nums; }
.hynomas-hud.rec { color: #ff7070; }
.hynomas-keys { position: absolute; left: 8px; bottom: 6px; display: flex; gap: 3px; }
.hynomas-keys span { min-width: 18px; padding: 1px 4px; text-align: center; border-radius: 3px;
  background: rgba(0,0,0,0.45); color: #8a939c; }
.hynomas-keys span.on { background: #5ad1ff; color: #10161c; }
.hynomas-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.hynomas-bar button { padding: 3px 9px; border-radius: 4px; border: 1px solid #4a525b; background: #2b3036;
  color: #d6dbe1; cursor: pointer; font: inherit; }
.hynomas-bar button:hover { background: #363c43; }
.hynomas-bar button.rec { border-color: #b84a4a; }
.hynomas-bar button:disabled { opacity: 0.45; cursor: default; }
.hynomas-help { color: #8a939c; font-size: 11px; }
`

function injectStyle() {
  if (document.getElementById('hynomas-cam-style')) return
  const el = document.createElement('style')
  el.id = 'hynomas-cam-style'
  el.textContent = STYLE
  document.head.appendChild(el)
}

class Recorder {
  constructor(node, jsonWidget) {
    injectStyle()
    this.node = node
    this.jsonWidget = jsonWidget
    this.pose = new Pose()
    this.keys = new Set()
    this.frames = []
    this.state = 'idle' // idle | armed | recording | playing
    this.acc = 0
    this.last = 0
    this.raf = 0
    this.playIndex = 0

    const root = (this.root = document.createElement('div'))
    root.className = 'hynomas-cam'
    root.tabIndex = 0
    root.innerHTML = `
      <div class="hynomas-view-wrap">
        <canvas class="view"></canvas><canvas class="map" width="96" height="96"></canvas>
        <div class="hynomas-hud"></div>
        <div class="hynomas-keys">${['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowLeft', 'ArrowUp', 'ArrowDown', 'ArrowRight', 'Shift']
          .map((k) => `<span data-k="${k}">${KEY_LABELS[k]}</span>`).join('')}</div>
      </div>
      <div class="hynomas-bar">
        <button class="rec" data-a="record">● Record</button>
        <button data-a="stop">■ Stop</button>
        <button data-a="play">▶ Play</button>
        <button data-a="reset">↺ Clear</button>
        <button data-a="json">{ } JSON</button>
      </div>
      <div class="hynomas-help">Record, then hold W/A/S/D to move, arrows (or drag) to turn, Shift to run. Recording
        starts with the first key; pauses are kept. Stop when done. The path is stretched over the segment it steers;
        only its shape matters, not its scale.</div>`
    this.viewCanvas = root.querySelector('canvas.view')
    this.mapCanvas = root.querySelector('canvas.map')
    this.hud = root.querySelector('.hynomas-hud')
    this.view = new View(this.viewCanvas)
    this.buttons = Object.fromEntries([...root.querySelectorAll('button')].map((b) => [b.dataset.a, b]))
    for (const [action, button] of Object.entries(this.buttons)) {
      button.addEventListener('pointerdown', (e) => e.stopPropagation())
      button.addEventListener('click', (e) => { e.stopPropagation(); this[action]() })
    }

    root.addEventListener('keydown', (e) => this.onKey(e, true))
    root.addEventListener('keyup', (e) => this.onKey(e, false))
    root.addEventListener('blur', () => { this.keys.clear(); this.refreshKeys() })
    const wrap = root.querySelector('.hynomas-view-wrap')
    wrap.addEventListener('pointerdown', (e) => {
      e.stopPropagation()
      root.focus()
      if (this.state !== 'armed' && this.state !== 'recording') return
      this.drag = [e.clientX, e.clientY]
      wrap.setPointerCapture(e.pointerId)
    })
    wrap.addEventListener('pointermove', (e) => {
      if (!this.drag) return
      e.stopPropagation()
      const dx = e.clientX - this.drag[0], dy = e.clientY - this.drag[1]
      this.drag = [e.clientX, e.clientY]
      this.pose.yaw -= dx * LOOK
      this.pose.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, this.pose.pitch - dy * LOOK))
      if (this.state === 'armed') this.begin()
      this.render()
    })
    wrap.addEventListener('pointerup', (e) => { this.drag = null; wrap.releasePointerCapture?.(e.pointerId) })
    new ResizeObserver(() => this.resize()).observe(wrap)
    this.load()
  }

  // --- recording state ---

  record() {
    this.stopLoop()
    this.frames = []
    this.pose.reset()
    this.state = 'armed'
    this.root.focus()
    this.update()
  }

  begin() {
    this.state = 'recording'
    this.frames = [this.pose.frame()]
    this.acc = 0
    this.startLoop()
  }

  stop() {
    if (this.state === 'recording') {
      this.frames = trimTail(this.frames)
      this.save()
    }
    this.state = 'idle'
    this.keys.clear()
    this.stopLoop()
    if (this.frames.length) this.pose.setFrame(this.frames[this.frames.length - 1])
    this.update()
  }

  play() {
    if (this.frames.length < 2 || this.state === 'recording') return
    this.state = 'playing'
    this.playIndex = 0
    this.acc = 0
    this.startLoop()
  }

  reset() {
    this.stopLoop()
    this.frames = []
    this.pose.reset()
    this.state = 'idle'
    this.save()
    this.update()
  }

  json() {
    // show the raw recording (e.g. to paste a path exported from Tencent's demo); the node grows to fit it
    const show = this.jsonWidget.hidden
    this.jsonWidget.hidden = !show
    this.node.setSize([this.node.size[0], this.node.size[1] + (show ? 140 : -140)])
    if (!show) this.load()
    this.node.setDirtyCanvas(true, true)
  }

  onKey(e, down) {
    const code = e.code.startsWith('Shift') ? 'Shift' : e.code
    const control = CONTROL_KEYS.includes(code) || code === 'Shift'
    if (e.code === 'Escape' || (e.code === 'Enter' && down)) {
      if (this.state === 'recording' || this.state === 'armed') { this.stop(); e.preventDefault(); e.stopPropagation() }
      return
    }
    if (!control || (this.state !== 'armed' && this.state !== 'recording')) return
    e.preventDefault()
    e.stopPropagation()
    if (down) this.keys.add(code)
    else this.keys.delete(code)
    if (down && this.state === 'armed' && code !== 'Shift') this.begin()
    this.refreshKeys()
  }

  // --- loop ---

  startLoop() {
    if (this.raf) return
    this.last = performance.now()
    const tick = (now) => {
      const dt = Math.min((now - this.last) / 1000, 0.1)
      this.last = now
      this.acc += dt
      while (this.acc >= STEP) {
        this.acc -= STEP
        if (this.state === 'recording') {
          this.pose.step(this.keys, STEP)
          this.frames.push(this.pose.frame())
          if (this.frames.length > MAX_SECONDS * FPS) { this.stop(); return }
        } else if (this.state === 'playing') {
          this.pose.setFrame(this.frames[this.playIndex])
          this.playIndex += 1
          if (this.playIndex >= this.frames.length) { this.state = 'idle'; this.stopLoop(); this.update(); return }
        }
      }
      this.update()
      this.raf = requestAnimationFrame(tick)
    }
    this.raf = requestAnimationFrame(tick)
  }

  stopLoop() {
    if (this.raf) cancelAnimationFrame(this.raf)
    this.raf = 0
  }

  // --- drawing ---

  resize() {
    const r = this.viewCanvas.getBoundingClientRect()
    const scale = app.canvas?.ds?.scale ?? 1
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const w = Math.max(64, Math.round((r.width / scale) * dpr)), h = Math.max(36, Math.round((r.height / scale) * dpr))
    if (this.viewCanvas.width !== w || this.viewCanvas.height !== h) {
      this.viewCanvas.width = w
      this.viewCanvas.height = h
    }
    this.render()
  }

  render() {
    const recording = this.state === 'recording'
    this.view.draw(this.pose, this.frames, recording)
    drawMap(this.mapCanvas, this.pose, this.frames, recording)
  }

  refreshKeys() {
    for (const el of this.root.querySelectorAll('.hynomas-keys span')) el.classList.toggle('on', this.keys.has(el.dataset.k))
  }

  update() {
    const seconds = (n) => (Math.max(0, n - 1) / FPS).toFixed(1)
    let text
    if (this.state === 'armed') text = 'Press W/A/S/D or an arrow to start recording'
    else if (this.state === 'recording') text = `● REC ${seconds(this.frames.length)} s`
    else if (this.state === 'playing') text = `▶ ${seconds(this.playIndex + 1)} / ${seconds(this.frames.length)} s`
    else if (this.frames.length > 1) text = `Path: ${seconds(this.frames.length)} s, ${this.frames.length} frames`
    else text = 'No path yet: press Record'
    this.hud.textContent = text
    this.hud.classList.toggle('rec', this.state === 'recording')
    const busy = this.state === 'recording' || this.state === 'armed'
    this.buttons.record.disabled = busy
    this.buttons.stop.disabled = !busy && this.state !== 'playing'
    this.buttons.play.disabled = busy || this.frames.length < 2
    this.refreshKeys()
    this.render()
  }

  // --- the node's value ---

  save() {
    const value = JSON.stringify({ frames: this.frames, sampleRate: FPS })
    this.jsonWidget.value = value
    this.jsonWidget.callback?.(value)
    this.node.setDirtyCanvas(true, true)
  }

  load() {
    try {
      const data = JSON.parse(this.jsonWidget.value || '{}')
      this.frames = Array.isArray(data.frames) ? data.frames : []
    } catch {
      this.frames = []
    }
    this.pose.reset()
    if (this.frames.length) this.pose.setFrame(this.frames[this.frames.length - 1])
    this.state = 'idle'
    this.update()
  }

  destroy() { this.stopLoop() }
}

app.registerExtension({
  name: 'HyNomas.CameraRecorder',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'HyNomasCameraRecording') return
    const created = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments)
      const jsonWidget = this.widgets?.find((w) => w.name === 'recording_json')
      if (!jsonWidget) return result
      jsonWidget.hidden = true
      const recorder = new Recorder(this, jsonWidget)
      this.hyNomasRecorder = recorder
      this.addDOMWidget('camera_recorder', 'hynomas_camera', recorder.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 300,
      })
      this.resizable = true
      setTimeout(() => {
        if (this.size[0] < 460 || this.size[1] < 420) this.setSize([Math.max(460, this.size[0]), Math.max(420, this.size[1])])
      }, 0)
      return result
    }
    const configure = nodeType.prototype.onConfigure
    nodeType.prototype.onConfigure = function () {
      const result = configure?.apply(this, arguments)
      // a loaded workflow sets the widget values after creation: show its path
      setTimeout(() => this.hyNomasRecorder?.load(), 0)
      return result
    }
    const removed = nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved = function () {
      this.hyNomasRecorder?.destroy()
      return removed?.apply(this, arguments)
    }
  },
})
