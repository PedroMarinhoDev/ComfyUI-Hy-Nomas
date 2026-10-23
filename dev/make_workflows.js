// Builds the example workflows in the ComfyUI frontend (run in the page: the browser pane's javascript tool),
// lays them out, checks them with graphToPrompt and saves them to the user's workflows (POST /api/userdata).
// The saved JSON is then copied into the pack's workflows/ folder.
(async () => {
  const LG = window.LiteGraph
  const app = window.app ?? window.comfyAPI?.app?.app
  const W = 1280, H = 704

  const DIT = 'hy_nomas_distil_int8.safetensors'
  const DEC = 'hy_nomas_pixelumm_fp8.safetensors'
  const TE = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'

  const README = (title, extra) => `# ${title}

Tencent's **Hy-Nomas** world model: a video generated segment by segment, each segment continuing the world
from the cached memory of the earlier ones. Nodes: **ComfyUI-HyNomas**.

**Files** (put them in these folders, rename freely):
- models/diffusion_models: the INT8 DiT, DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors from [tencent/Hy-Nomas](https://huggingface.co/tencent/Hy-Nomas)
- models/vae: the FP8 PixelUMM decoder, VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors from [tencent/Hy-Nomas](https://huggingface.co/tencent/Hy-Nomas)
- models/text_encoders: qwen_2.5_vl_7b_fp8_scaled.safetensors from [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) (Load CLIP type **qwen_image**)

**Settings** (the distilled model's contract): KSampler **euler / simple / 8 steps / cfg 1**. Width and height
multiples of 32: 1280x704 (Tencent's 720p) or 832x480. 24 fps; the segment lengths come from the
start_time/end_time of the World Prompt's segments.

**Prompt format**: write the world the way the model was trained (the example below; see *Writing a world* in the pack README): the subject is \`<|Subject1|>\`, actions are \`<Action1><|Subject1|> ...</Action1>.\`, later segments start "Compared to the previous segment, ". Loosely written follow-up segments are mostly ignored.

${extra}

The PixelUMM decoder weights are under NVIDIA's One-Way Noncommercial license (research/evaluation only).`

  function add(type, pos, widgets = {}, title) {
    const node = LG.createNode(type)
    if (!node) throw new Error('unknown node ' + type)
    app.graph.add(node)
    node.pos = pos
    if (title) node.title = title
    for (const [name, value] of Object.entries(widgets)) {
      const w = node.widgets?.find((x) => x.name === name)
      if (!w) throw new Error(`${type}: no widget ${name}`)
      w.value = value
    }
    node.setSize(node.computeSize())
    return node
  }
  function link(a, out, b, input) {
    const oi = typeof out === 'number' ? out : a.outputs.findIndex((o) => o.name === out)
    const ii = b.inputs.findIndex((i) => i.name === input)
    if (oi < 0 || ii < 0) throw new Error(`no slot ${out} -> ${input} (${a.type} -> ${b.type})`)
    a.connect(oi, b, ii)
  }
  function note(text, pos, size) {
    const n = add('MarkdownNote', pos)
    n.widgets[0].value = text
    n.size = size
    return n
  }
  // place columns of nodes left to right, each column stacked top to bottom, from measured sizes
  function layout(columns, x0 = 0, gap = 50) {
    let x = x0
    for (const column of columns) {
      let y = 0
      let width = 0
      for (const n of column) {
        n.pos = [x, y + LG.NODE_TITLE_HEIGHT]
        y += n.size[1] + LG.NODE_TITLE_HEIGHT + gap
        width = Math.max(width, n.size[0])
      }
      x += width + gap
    }
  }
  async function save(name) {
    const prompt = await app.graphToPrompt()
    const errors = Object.values(prompt.output).filter((n) => !n.class_type)
    if (errors.length) throw new Error('graphToPrompt failed for ' + name)
    const data = JSON.stringify(app.graph.serialize())
    const resp = await fetch(`/api/userdata/workflows%2F${encodeURIComponent(name)}.json?overwrite=true`,
      { method: 'POST', body: data, headers: { 'Content-Type': 'application/json' } })
    if (!resp.ok) throw new Error('save failed ' + resp.status)
    return Object.keys(prompt.output).length
  }

  const worldJson = (await (await fetch('/api/object_info/HyNomasWorldPrompt')).json())
    .HyNomasWorldPrompt.input.required.world_json[1].default
  const out = {}

  // 1) per segment: Segment -> KSampler -> Extend World, twice, then decode
  app.graph.clear()
  note(README('Hy-Nomas: a world, one segment at a time', `**How it flows:** each **Segment** gets the world so far,
**KSampler** samples it, **Extend World** adds it to the world (one more forward pass that caches its memory).
Add segments to the World Prompt JSON and copy the Segment / KSampler / Extend World trio for each,
increasing segment_index. The **Decode** node turns the whole world into one clip.`), [-700, 0], [640, 560])
  const dit = add('HyNomasLoadDiT', [0, 0], { dit_name: DIT, matmul: 'int8' })
  const clip = add('CLIPLoader', [0, 160], { clip_name: TE, type: 'qwen_image' })
  const prompt = add('HyNomasWorldPrompt', [0, 330], { world_json: worldJson })
  prompt.size = [420, 520]
  let world = null
  const xs = [520, 1300]
  for (let i = 0; i < 2; i++) {
    const x = xs[i]
    const seg = add('HyNomasSegment', [x, 0], { segment_index: i, width: W, height: H }, `Segment ${i}`)
    link(clip, 0, seg, 'clip')
    link(prompt, 0, seg, 'script')
    if (world) link(world, 0, seg, 'world')
    const ks = add('KSampler', [x, 330], { seed: 42 + i, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple' })
    link(dit, 0, ks, 'model'); link(seg, 0, ks, 'positive'); link(seg, 1, ks, 'negative'); link(seg, 2, ks, 'latent_image')
    const ext = add('HyNomasExtendWorld', [x + 380, 0], { seed: i }, `Extend World ${i}`)
    link(dit, 0, ext, 'model'); link(seg, 0, ext, 'positive'); link(ks, 0, ext, 'samples')
    world = ext
  }
  const dec = add('HyNomasLoadDecoder', [2100, 0], { decoder_name: DEC, matmul: 'fp8' })
  const decode = add('HyNomasDecode', [2100, 150], {})
  const cols1 = [[dit, clip, prompt]]
  link(dec, 0, decode, 'decoder'); link(world, 0, decode, 'world'); link(prompt, 0, decode, 'script')
  const video = add('CreateVideo', [2100, 330], {})
  link(decode, 0, video, 'images'); link(decode, 1, video, 'fps')
  const savev = add('SaveVideo', [2100, 470], { filename_prefix: 'hy_nomas/world' })
  link(video, 0, savev, 'video')
  const segNodes = app.graph._nodes.filter((n) => n.type === 'HyNomasSegment')
  const ksNodes = app.graph._nodes.filter((n) => n.type === 'KSampler')
  const extNodes = app.graph._nodes.filter((n) => n.type === 'HyNomasExtendWorld')
  for (let i = 0; i < segNodes.length; i++) cols1.push([segNodes[i], ksNodes[i]], [extNodes[i]])
  cols1.push([dec, decode, video, savev])
  layout(cols1)
  out.segments = await save('Hy-Nomas world (per segment)')

  // 2) one node for the whole script
  app.graph.clear()
  note(README('Hy-Nomas: a world in one node', `**Generate World** runs every segment of the World Prompt in order
(Segment, sample, Extend World), with the KSampler settings on the node. Same result as the per-segment
workflow, less to wire; connect a **world** input to continue an earlier world.`), [-700, 0], [640, 520])
  const dit2 = add('HyNomasLoadDiT', [0, 0], { dit_name: DIT, matmul: 'int8' })
  const clip2 = add('CLIPLoader', [0, 160], { clip_name: TE, type: 'qwen_image' })
  const prompt2 = add('HyNomasWorldPrompt', [0, 330], { world_json: worldJson })
  prompt2.size = [420, 520]
  const gen = add('HyNomasGenerateWorld', [520, 0], { width: W, height: H, seed: 42, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple' })
  link(dit2, 0, gen, 'model'); link(clip2, 0, gen, 'clip'); link(prompt2, 0, gen, 'script')
  const dec2 = add('HyNomasLoadDecoder', [960, 0], { decoder_name: DEC, matmul: 'fp8' })
  const decode2 = add('HyNomasDecode', [960, 150], {})
  link(dec2, 0, decode2, 'decoder'); link(gen, 0, decode2, 'world'); link(prompt2, 0, decode2, 'script')
  const video2 = add('CreateVideo', [960, 330], {})
  link(decode2, 0, video2, 'images'); link(decode2, 1, video2, 'fps')
  const save2 = add('SaveVideo', [960, 470], { filename_prefix: 'hy_nomas/world' })
  link(video2, 0, save2, 'video')
  layout([[dit2, clip2, prompt2], [gen], [dec2, decode2, video2, save2]])
  out.one_node = await save('Hy-Nomas world (one node)')

  // 3) camera: a key timeline steering the second segment
  app.graph.clear()
  note(README('Hy-Nomas: camera control', `**Camera Recording** (segment 0): fly the camera inside the node. Press
**Record**, then hold W/A/S/D to move, the arrow keys (or drag the view) to turn, Shift to run; **Stop** when done,
**Play** to watch it. These are the controls of Tencent's web demo. The path is stretched over the segment and
only its shape matters. A path recorded in the demo can be pasted under **{ } JSON**.

**Camera Keys** (segment 1): the same controls written as a timeline, "KEYS SECONDS" steps such as
"Left 2.5, W 2.5". Leave a Segment's camera unconnected for text-only control. rotation_scale 2.5 is Tencent's
default gain.`), [-700, 0], [640, 620])
  const dit3 = add('HyNomasLoadDiT', [0, 0], { dit_name: DIT, matmul: 'int8' })
  const clip3 = add('CLIPLoader', [0, 160], { clip_name: TE, type: 'qwen_image' })
  const prompt3 = add('HyNomasWorldPrompt', [0, 330], { world_json: worldJson })
  prompt3.size = [420, 520]
  const camA = add('HyNomasCameraRecording', [520, 560], {}, 'Camera: draw it (Record, then W/A/S/D)')
  {
    // a recorded example path: walk 2.5 s, then walk while turning right for 2.5 s (fixed 24 fps steps)
    const r = camA.hyNomasRecorder
    r.record()
    r.begin()
    for (const [keys, seconds] of [[['KeyW'], 2.5], [['KeyW', 'ArrowRight'], 2.5]]) {
      for (let i = 0; i < Math.round(seconds * 24); i++) {
        r.pose.step(new Set(keys), 1 / 24)
        r.frames.push(r.pose.frame())
      }
    }
    r.stop()
    camA.setSize([480, 440])
  }
  const camB = add('HyNomasCameraKeys', [1300, 560], { timeline: 'Left 2.5, W 2.5' }, 'Camera: turn left, then forward')
  let world3 = null
  for (let i = 0; i < 2; i++) {
    const x = xs[i]
    const seg = add('HyNomasSegment', [x, 0], { segment_index: i, width: W, height: H }, `Segment ${i}`)
    link(clip3, 0, seg, 'clip'); link(prompt3, 0, seg, 'script')
    link(i === 0 ? camA : camB, 0, seg, 'camera')
    if (world3) link(world3, 0, seg, 'world')
    const ks = add('KSampler', [x, 330], { seed: 42 + i, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple' })
    link(dit3, 0, ks, 'model'); link(seg, 0, ks, 'positive'); link(seg, 1, ks, 'negative'); link(seg, 2, ks, 'latent_image')
    const ext = add('HyNomasExtendWorld', [x + 380, 0], { seed: i }, `Extend World ${i}`)
    link(dit3, 0, ext, 'model'); link(seg, 0, ext, 'positive'); link(ks, 0, ext, 'samples')
    world3 = ext
  }
  const dec3 = add('HyNomasLoadDecoder', [2100, 0], { decoder_name: DEC, matmul: 'fp8' })
  const decode3 = add('HyNomasDecode', [2100, 150], {})
  link(dec3, 0, decode3, 'decoder'); link(world3, 0, decode3, 'world'); link(prompt3, 0, decode3, 'script')
  const video3 = add('CreateVideo', [2100, 330], {})
  link(decode3, 0, video3, 'images'); link(decode3, 1, video3, 'fps')
  const save3 = add('SaveVideo', [2100, 470], { filename_prefix: 'hy_nomas/camera' })
  link(video3, 0, save3, 'video')
  {
    const segs = app.graph._nodes.filter((n) => n.type === 'HyNomasSegment')
    const kss = app.graph._nodes.filter((n) => n.type === 'KSampler')
    const exts = app.graph._nodes.filter((n) => n.type === 'HyNomasExtendWorld')
    const cams = [camA, camB]
    const cols = [[dit3, clip3, prompt3]]
    for (let i = 0; i < segs.length; i++) cols.push([cams[i], segs[i], kss[i]], [exts[i]])
    cols.push([dec3, decode3, video3, save3])
    layout(cols)
  }
  out.camera = await save('Hy-Nomas camera control')
  // 4) drive: explore the world interactively
  app.graph.clear()
  note(README('Hy-Nomas: drive (interactive)', `**Drive** generates the world turn by turn while you steer it, like
Tencent's web demo. Click **▶ Drive**, then click the video and hold **W/A/S/D** (move), the **arrow keys**
(look), **Shift** (run). Each turn continues the world with the keys you held since the previous turn was sent.
Type in **Next turn** to change the scene once (e.g. "glowing blue flowers open beside the path"), or a
**Subject action** to keep the subject doing something (e.g. "turns around and sniffs the flowers"; it is sent
as \`<Action1><|Subject1|> turns around and sniffs the flowers</Action1>.\`). **⤓ Save mp4** decodes the whole drive properly and saves it. No need to press Run.

It is not real time: a 1 s turn takes about 3.5 s at 832x480 on an RTX 4090 (sampling, memory, decoding), so the
video plays a second, then waits for the next one. Lower width/height for faster turns; turn_seconds 2-3 is
more efficient but answers the keys later. The world's memory grows ~0.5 GB of RAM per second at 832x480.

To drive from a picture, connect **Hy-Nomas Load Encoder** to encoder and a **Load Image** to start_image: the
picture becomes the world's first segment (described by world_json's first segment) and the first turn moves
on from it.`),
  [-700, 0], [640, 600])
  const dit4 = add('HyNomasLoadDiT', [0, 0], { dit_name: DIT, matmul: 'int8' })
  const clip4 = add('CLIPLoader', [0, 160], { clip_name: TE, type: 'qwen_image' })
  const dec4 = add('HyNomasLoadDecoder', [0, 330], { decoder_name: DEC, matmul: 'fp8' })
  const drive = add('HyNomasDrive', [440, 0], { width: 832, height: 480, turn_seconds: 1.0 })
  link(dit4, 0, drive, 'model'); link(clip4, 0, drive, 'clip'); link(dec4, 0, drive, 'decoder')
  layout([[dit4, clip4, dec4], [drive]])
  drive.setSize([620, 960])
  out.drive = await save('Hy-Nomas drive (interactive)')

  // 5) start from an image: the picture becomes segment 0, Generate World continues from it
  app.graph.clear()
  note(README('Hy-Nomas: start from an image', `**Start From Image** puts a picture at the start of the world: it is
encoded as a short still clip (Load Encoder reads the encoder from the PixelUMM file) and becomes segment 0, the
world's memory of what came before. **Generate World** then continues from segment 1. Hy-Nomas has no image
input, so this is a workaround, but the look of the picture carries over well.

**Segment 0 of the World Prompt describes the picture** (scene, style, the subject and the perspective); its
actions and times are not used. Segment 1 onward say what happens next. Pick your image in **Load Image**; it is
scaled to cover width x height and center-cropped. Decode keeps one frame of the held image (start_image).

The **Drive** node takes the same encoder and image (start_image) to steer a world that starts from your picture.`),
  [-700, 0], [640, 640])
  const paris = await (await fetch('/api/view?filename=paris_walk.json&type=input')).text()
  const dit5 = add('HyNomasLoadDiT', [0, 0], { dit_name: DIT, matmul: 'int8' })
  const clip5 = add('CLIPLoader', [0, 160], { clip_name: TE, type: 'qwen_image' })
  const enc5 = add('HyNomasLoadEncoder', [0, 330], { encoder_name: DEC })
  const prompt5 = add('HyNomasWorldPrompt', [0, 450], { world_json: paris })
  prompt5.size = [420, 520]
  const img5 = add('LoadImage', [520, 0], {})
  const start5 = add('HyNomasStartFromImage', [520, 400], { width: W, height: H, still_frames: 3, seed: 0 })
  link(dit5, 0, start5, 'model'); link(clip5, 0, start5, 'clip'); link(enc5, 0, start5, 'encoder')
  link(prompt5, 0, start5, 'script'); link(img5, 0, start5, 'image')
  const gen5 = add('HyNomasGenerateWorld', [960, 0], { width: W, height: H, seed: 42, steps: 8, cfg: 1.0, sampler_name: 'euler', scheduler: 'simple' })
  link(dit5, 0, gen5, 'model'); link(clip5, 0, gen5, 'clip'); link(prompt5, 0, gen5, 'script'); link(start5, 0, gen5, 'world')
  const dec5 = add('HyNomasLoadDecoder', [1400, 0], { decoder_name: DEC, matmul: 'fp8' })
  const decode5 = add('HyNomasDecode', [1400, 150], {})
  link(dec5, 0, decode5, 'decoder'); link(gen5, 0, decode5, 'world')
  const video5 = add('CreateVideo', [1400, 330], {})
  link(decode5, 0, video5, 'images'); link(decode5, 1, video5, 'fps')
  const save5 = add('SaveVideo', [1400, 470], { filename_prefix: 'hy_nomas/from_image' })
  link(video5, 0, save5, 'video')
  layout([[dit5, clip5, enc5, prompt5], [img5, start5], [gen5], [dec5, decode5, video5, save5]])
  out.image = await save('Hy-Nomas start from image')

  // check overlaps of the last graph and report them
  const ns = app.graph._nodes, overlaps = []
  for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) {
    const a = ns[i].getBounding(), b = ns[j].getBounding()
    if (a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]) overlaps.push(ns[i].title + ' x ' + ns[j].title)
  }
  out.overlaps = overlaps
  return out
})()
