// Hy-Nomas drive panel: explore a world turn by turn, steering with W/A/S/D and the arrows, inside the
// "Hy-Nomas Drive" node.
//
// Each turn is one queue of the drive node alone (partial_execution_targets), carrying a command: the keys held
// since the previous turn was sent (as a "KEYS SECONDS" timeline stretched over the turn), the camera mode and the
// texts. The node streams the turn's frames back over the websocket ("hynomas.drive" events) and the panel plays
// them at 24 fps; while running, the next turn is sent the moment a turn's frames arrive.
const { app } = window.comfyAPI.app
const { api } = window.comfyAPI.api

const FPS = 24
const KEY_CODES = {
  KeyW: 'W', KeyA: 'A', KeyS: 'S', KeyD: 'D',
  ArrowLeft: 'Left', ArrowRight: 'Right', ArrowUp: 'Up', ArrowDown: 'Down', Shift: 'Shift',
}
const CHIPS = [['KeyW', 'W'], ['KeyA', 'A'], ['KeyS', 'S'], ['KeyD', 'D'], ['ArrowLeft', '←'], ['ArrowUp', '↑'],
  ['ArrowDown', '↓'], ['ArrowRight', '→'], ['Shift', '⇧']]

const STYLE = `
.hynomas-drive { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; outline: none;
  font: 12px/1.35 system-ui, sans-serif; color: #d6dbe1; box-sizing: border-box; }
.hynomas-drive:focus-within .hynomas-screen { box-shadow: 0 0 0 2px #5ad1ff; }
.hynomas-screen { position: relative; flex: 1 1 auto; min-height: 140px; border-radius: 6px; overflow: hidden;
  background: #111417; cursor: pointer; }
.hynomas-screen canvas { width: 100%; height: 100%; display: block; object-fit: contain; }
.hynomas-overlay { position: absolute; left: 8px; top: 6px; right: 8px; display: flex; justify-content: space-between;
  gap: 8px; pointer-events: none; }
.hynomas-overlay span { padding: 2px 6px; border-radius: 4px; background: rgba(0,0,0,0.5);
  font-variant-numeric: tabular-nums; }
.hynomas-wait { position: absolute; left: 0; right: 0; bottom: 34px; text-align: center; pointer-events: none; }
.hynomas-wait span { padding: 3px 9px; border-radius: 10px; background: rgba(0,0,0,0.55); color: #ffd36a; }
.hynomas-drive-keys { position: absolute; left: 8px; bottom: 6px; display: flex; gap: 3px; pointer-events: none; }
.hynomas-drive-keys span { min-width: 18px; padding: 1px 4px; text-align: center; border-radius: 3px;
  background: rgba(0,0,0,0.5); color: #8a939c; }
.hynomas-drive-keys span.on { background: #5ad1ff; color: #10161c; }
.hynomas-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.hynomas-row button, .hynomas-row select { padding: 3px 9px; border-radius: 4px; border: 1px solid #4a525b;
  background: #2b3036; color: #d6dbe1; cursor: pointer; font: inherit; }
.hynomas-row button:hover { background: #363c43; }
.hynomas-row button.go { border-color: #3f8f5a; }
.hynomas-row button:disabled { opacity: 0.45; cursor: default; }
.hynomas-row input { flex: 1 1 140px; min-width: 100px; padding: 3px 6px; border-radius: 4px;
  border: 1px solid #4a525b; background: #1d2024; color: #d6dbe1; font: inherit; }
.hynomas-stats { color: #8a939c; font-size: 11px; font-variant-numeric: tabular-nums; }
.hynomas-stats a { color: #5ad1ff; }
`

function injectStyle() {
  if (document.getElementById('hynomas-drive-style')) return
  const el = document.createElement('style')
  el.id = 'hynomas-drive-style'
  el.textContent = STYLE
  document.head.appendChild(el)
}

function decodeFrame(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
  return createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }))
}

class DrivePanel {
  constructor(node, commandWidget) {
    injectStyle()
    this.node = node
    this.command = commandWidget
    this.session = crypto.randomUUID()
    this.retired = new Set()
    this.running = false
    this.busy = false
    this.started = false
    this.promptId = null
    this.frames = []          // decoded ImageBitmaps waiting to be shown
    this.current = null       // the frame on screen
    this.playClock = 0
    this.keys = new Set()
    this.keyLog = [{ t: performance.now(), keys: new Set() }]
    this.lastSubmit = performance.now()
    this.turns = 0
    this.videoSeconds = 0
    this.stage = ''
    this.raf = 0

    const root = (this.root = document.createElement('div'))
    root.className = 'hynomas-drive'
    root.tabIndex = 0
    root.innerHTML = `
      <div class="hynomas-screen">
        <canvas></canvas>
        <div class="hynomas-overlay"><span class="left"></span><span class="right"></span></div>
        <div class="hynomas-wait" hidden><span></span></div>
        <div class="hynomas-drive-keys">${CHIPS.map(([k, l]) => `<span data-k="${k}">${l}</span>`).join('')}</div>
      </div>
      <div class="hynomas-row">
        <button class="go" data-a="drive">▶ Drive</button>
        <button data-a="pause">⏸ Pause</button>
        <button data-a="restart">⟲ New world</button>
        <button data-a="save">⤓ Save mp4</button>
        <select data-a="camera" title="Steer: the keys you hold move the camera. Free: no camera input, the model frames the scene itself.">
          <option value="steer">Camera: steer</option><option value="free">Camera: free</option>
        </select>
      </div>
      <div class="hynomas-row">
        <input data-a="description" placeholder="Next turn: what changes in the scene (sent once)">
        <input data-a="action" placeholder="Subject action, e.g. turns around and sniffs the flowers (kept until cleared)">
      </div>
      <div class="hynomas-stats"></div>`
    this.canvas = root.querySelector('canvas')
    this.ctx = this.canvas.getContext('2d')
    this.left = root.querySelector('.hynomas-overlay .left')
    this.right = root.querySelector('.hynomas-overlay .right')
    this.wait = root.querySelector('.hynomas-wait')
    this.stats = root.querySelector('.hynomas-stats')
    this.description = root.querySelector('input[data-a="description"]')
    this.action = root.querySelector('input[data-a="action"]')
    this.cameraMode = root.querySelector('select[data-a="camera"]')
    this.buttons = Object.fromEntries([...root.querySelectorAll('button')].map((b) => [b.dataset.a, b]))
    for (const [name, button] of Object.entries(this.buttons)) {
      button.addEventListener('pointerdown', (e) => e.stopPropagation())
      button.addEventListener('click', (e) => { e.stopPropagation(); this[name]() })
    }
    for (const el of [this.description, this.action, this.cameraMode]) {
      el.addEventListener('pointerdown', (e) => e.stopPropagation())
      el.addEventListener('keydown', (e) => e.stopPropagation())
      el.addEventListener('keyup', (e) => e.stopPropagation())
    }
    root.querySelector('.hynomas-screen').addEventListener('pointerdown', (e) => { e.stopPropagation(); root.focus() })
    root.addEventListener('keydown', (e) => this.onKey(e, true))
    root.addEventListener('keyup', (e) => this.onKey(e, false))
    root.addEventListener('blur', () => { if (this.keys.size) { this.keys.clear(); this.logKeys() } })

    this.onEvent = (e) => this.onDrive(e.detail)
    this.onError = (e) => this.onFailure(e.detail, 'error')
    this.onInterrupted = (e) => this.onFailure(e.detail, 'interrupted')
    api.addEventListener('hynomas.drive', this.onEvent)
    api.addEventListener('execution_error', this.onError)
    api.addEventListener('execution_interrupted', this.onInterrupted)
    new ResizeObserver(() => this.resize()).observe(root.querySelector('.hynomas-screen'))
    this.update()
  }

  destroy() {
    if (this.started) this.resetServer(this.session)
    api.removeEventListener('hynomas.drive', this.onEvent)
    api.removeEventListener('execution_error', this.onError)
    api.removeEventListener('execution_interrupted', this.onInterrupted)
    cancelAnimationFrame(this.raf)
  }

  widget(name) { return this.node.widgets?.find((w) => w.name === name)?.value }

  // --- keys

  onKey(e, down) {
    const code = e.code.startsWith('Shift') ? 'Shift' : e.code
    if (!(code in KEY_CODES)) return
    e.preventDefault()
    e.stopPropagation()
    if (e.repeat) return
    if (down) this.keys.add(code)
    else this.keys.delete(code)
    this.logKeys()
  }

  logKeys() {
    this.keyLog.push({ t: performance.now(), keys: new Set(this.keys) })
    this.update()
  }

  // The keys held between the previous turn's sending and now, as a timeline stretched over one turn.
  takeTimeline(turnSeconds) {
    const now = performance.now()
    const from = this.lastSubmit
    this.lastSubmit = now
    const name = (keys) => (keys.size ? [...keys].map((k) => KEY_CODES[k]).join('+') : 'idle')
    let state = new Set()
    const spans = []
    let t = from
    for (const entry of this.keyLog) {
      if (entry.t <= from) { state = entry.keys; continue }
      spans.push([name(state), entry.t - t])
      state = entry.keys
      t = entry.t
    }
    spans.push([name(state), now - t])
    this.keyLog = [{ t: now, keys: new Set(this.keys) }]
    const total = spans.reduce((a, [, d]) => a + d, 0)
    if (total < 50) return `${name(this.keys)} ${turnSeconds}` // nothing to stretch: the keys held right now
    const merged = []
    for (const [keys, d] of spans) {
      if (merged.length && merged[merged.length - 1][0] === keys) merged[merged.length - 1][1] += d
      else merged.push([keys, d])
    }
    return merged.filter(([, d]) => d > 0).map(([keys, d]) => `${keys} ${((d / total) * turnSeconds).toFixed(3)}`).join(', ')
  }

  // --- commands

  async send(command) {
    command.session = command.session ?? this.session
    command.nonce = Math.random().toString(36).slice(2)
    this.command.value = JSON.stringify(command)
    let prompt
    try {
      prompt = await app.graphToPrompt()
    } finally {
      this.command.value = ''
    }
    const body = { prompt: prompt.output, client_id: api.clientId, partial_execution_targets: [String(this.node.id)] }
    const res = await api.fetchApi('/prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) })
    const data = await res.json()
    if (!res.ok || data.error) {
      const nodeErrors = Object.values(data.node_errors || {}).flatMap((n) => n.errors.map((e) => `${n.class_type}: ${e.message} ${e.details}`))
      throw new Error([data.error?.message, ...nodeErrors].filter(Boolean).join('; '))
    }
    this.promptId = data.prompt_id
  }

  async submitTurn() {
    if (this.busy) return
    this.busy = true
    this.stage = 'queued'
    const turnSeconds = Number(this.widget('turn_seconds')) || 1
    const timeline = this.takeTimeline(turnSeconds)
    const command = {
      action: 'turn', fresh: !this.started, timeline, camera: this.cameraMode.value,
      description: this.description.value.trim(), subject_action: this.action.value.trim(),
    }
    this.description.value = ''
    this.lastTimeline = timeline
    this.started = true
    try {
      await this.send(command)
    } catch (err) {
      this.busy = false
      this.running = false
      this.error = String(err.message || err)
    }
    this.update()
  }

  drive() {
    this.error = null
    this.running = true
    this.root.focus()
    if (!this.started) this.lastSubmit = performance.now()
    this.submitTurn()
    this.startLoop()
  }

  pause() {
    this.running = false
    this.update()
  }

  restart() {
    this.running = false
    this.started = false
    this.frames = []
    this.current = null
    this.turns = 0
    this.videoSeconds = 0
    this.saved = null
    this.error = null
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.resetServer(this.session)
    this.retired.add(this.session)
    this.session = crypto.randomUUID()
    this.update()
  }

  async save() {
    if (this.busy || !this.turns) return
    this.busy = true
    this.stage = 'saving'
    try {
      await this.send({ action: 'save' })
    } catch (err) {
      this.busy = false
      this.error = String(err.message || err)
    }
    this.update()
  }

  // --- server events

  async onDrive(d) {
    if (d && this.retired.has(d.session) && this.busy && (d.type === 'frames' || d.type === 'saved')) {
      this.busy = false // the turn of a world that was replaced finished
      if (this.running) this.submitTurn()
      return
    }
    if (!d || d.session !== this.session) return
    if (d.type === 'progress') {
      this.stage = d.stage === 'sampling' ? `sampling ${d.step}/${d.steps}` : d.stage
      this.update()
      return
    }
    if (d.type === 'saved') {
      this.busy = false
      this.saved = d.file
      this.update()
      return
    }
    if (d.type !== 'frames') return
    const bitmaps = await Promise.all(d.frames.map(decodeFrame))
    this.frames.push(...bitmaps)
    this.turns = d.turn + 1
    this.videoSeconds = d.seconds
    this.lastTiming = d.timing
    this.archive = d.archive_gib
    this.busy = false
    this.stage = ''
    this.startLoop()
    if (this.running) this.submitTurn()
    this.update()
  }

  // free a world's memory on the server; a turn still running for it finishes and is ignored
  resetServer(session) {
    api.fetchApi('/hynomas/drive/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session }) }).catch(() => {})
  }

  onFailure(d, kind) {
    if (!d || d.prompt_id !== this.promptId) return
    this.busy = false
    this.running = false
    this.error = kind === 'error' ? `${d.exception_type || 'Error'}: ${d.exception_message || ''}` : 'interrupted'
    this.update()
  }

  // --- playback

  startLoop() {
    if (this.raf) return
    let last = performance.now()
    const tick = (now) => {
      this.playClock += now - last
      last = now
      const step = 1000 / FPS
      while (this.playClock >= step && this.frames.length) {
        this.playClock -= step
        this.current?.close?.()
        this.current = this.frames.shift()
      }
      if (!this.frames.length) this.playClock = Math.min(this.playClock, step)
      this.draw()
      this.raf = (this.running || this.busy || this.frames.length) ? requestAnimationFrame(tick) : 0
      if (!this.raf) this.update()
    }
    this.raf = requestAnimationFrame(tick)
  }

  resize() {
    const r = this.canvas.getBoundingClientRect()
    const scale = app.canvas?.ds?.scale ?? 1
    const w = Math.max(64, Math.round(r.width / scale)), h = Math.max(36, Math.round(r.height / scale))
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    this.draw()
  }

  draw() {
    const { ctx, canvas } = this
    ctx.fillStyle = '#111417'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    if (!this.current) return
    const s = Math.min(canvas.width / this.current.width, canvas.height / this.current.height)
    const w = this.current.width * s, h = this.current.height * s
    ctx.drawImage(this.current, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h)
  }

  update() {
    for (const el of this.root.querySelectorAll('.hynomas-drive-keys span')) el.classList.toggle('on', this.keys.has(el.dataset.k))
    this.left.textContent = this.started ? `turn ${this.turns} · ${this.videoSeconds.toFixed(1)} s` : 'click ▶ Drive, then steer'
    this.right.textContent = this.running ? '● driving' : (this.busy ? 'finishing turn' : 'paused')
    const waiting = this.busy && !this.frames.length
    this.wait.hidden = !waiting
    if (waiting) this.wait.firstElementChild.textContent = `generating · ${this.stage}`
    this.buttons.drive.disabled = this.running
    this.buttons.pause.disabled = !this.running
    this.buttons.save.disabled = this.busy || !this.turns
    const parts = []
    if (this.error) parts.push(`<span style="color:#ff7070">${this.error}</span>`)
    if (this.lastTiming) {
      const t = this.lastTiming, sec = Number(this.widget('turn_seconds')) || 1
      parts.push(`last turn ${t.total}s for ${sec}s of video (sample ${t.sampling} · memory ${t.memory} · decode ${t.decoding})`)
    }
    if (this.lastTimeline) parts.push(`keys sent: ${this.lastTimeline}`)
    if (this.archive) parts.push(`world memory ${this.archive} GiB`)
    if (this.saved) {
      const url = api.apiURL(`/view?filename=${encodeURIComponent(this.saved.filename)}&subfolder=${encodeURIComponent(this.saved.subfolder)}&type=output`)
      parts.push(`saved <a href="${url}" target="_blank">${this.saved.filename}</a>`)
    }
    this.stats.innerHTML = parts.join(' · ')
  }
}

app.registerExtension({
  name: 'HyNomas.Drive',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'HyNomasDrive') return
    const created = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments)
      const command = this.widgets?.find((w) => w.name === 'command')
      if (!command) return result
      command.hidden = true
      const panel = new DrivePanel(this, command)
      this.hyNomasDrive = panel
      this.addDOMWidget('drive_panel', 'hynomas_drive', panel.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 380,
      })
      this.resizable = true
      setTimeout(() => {
        if (this.size[0] < 560 || this.size[1] < 720) this.setSize([Math.max(560, this.size[0]), Math.max(720, this.size[1])])
      }, 0)
      return result
    }
    const removed = nodeType.prototype.onRemoved
    nodeType.prototype.onRemoved = function () {
      this.hyNomasDrive?.destroy()
      return removed?.apply(this, arguments)
    }
  },
})
