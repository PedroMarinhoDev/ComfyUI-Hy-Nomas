"""Camera control: a key timeline -> the camera tokens one segment's video attends to.

Hy-Nomas takes the camera as a per-frame pose sequence (`extrinsics_attn`). Tencent's web demo records it from a
three.js camera driven by W/A/S/D (move, 2.8 units/s, 7 with Shift) and the arrow keys (turn, 20 degrees/s),
sampled at 24 fps from (0, 1.5, 10) looking at (0, 1.5, 0) with a 55 degree FOV
(`apps/frontend/src/components/CameraPathDrawer.tsx`). `simulate_keys` replays a key timeline the same way.

A recorded turn becomes model input as in `demo_precomputed.prepare_turn` + `bidirectional.prepare_camera_inputs`
(vendored below, Apache-2.0): OpenGL -> OpenCV axes, poses relative to the first frame, resampled to
((frames - 1) // 4) * 4 + 1 poses, normalized framewise with translations divided by their 99th-percentile length,
then the inference gains (translation 1.0, rotation 2.5 in `inference.yaml`). The DiT then pads three copies of
the first pose in front and turns every four poses into one token per latent frame (`model.camera_tokens`).

Translations are normalized away, so the camera's speed sets the shape of the path, not its scale.
"""
import json
import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

FPS = 24
MOVE_SPEED = 2.8
SHIFT_SPEED = 7.0
TURN_SPEED = math.radians(20)
START_POSITION = (0.0, 1.5, 10.0)
FOV = 55.0
NORMALIZATION = dict(normalize_relative_to_first_frame=True, normalize_framewise=True, normalize_displacement=True,
                     normalize_displacement_per_video=True)
KEY_NAMES = {"w": "KeyW", "a": "KeyA", "s": "KeyS", "d": "KeyD", "left": "ArrowLeft", "right": "ArrowRight",
             "up": "ArrowUp", "down": "ArrowDown", "shift": "Shift", "idle": None, "still": None, "none": None}


# ---------------------------------------------------------------------------------------------------------------
# key timeline -> recorded frames


def parse_timeline(text):
    """"W 2, W+Left 1.5, idle 0.5" -> [(set of keys, seconds), ...]. Newlines work as separators too."""
    steps = []
    for chunk in text.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if len(parts) != 2:
            raise ValueError(f"camera step {chunk!r}: expected 'KEYS SECONDS', e.g. 'W+Left 1.5'")
        keys = set()
        for name in parts[0].lower().split("+"):
            if name not in KEY_NAMES:
                raise ValueError(f"unknown camera key {name!r}; use W, A, S, D, Left, Right, Up, Down, Shift, idle")
            if KEY_NAMES[name]:
                keys.add(KEY_NAMES[name])
        steps.append((keys, float(parts[1])))
    if not steps:
        raise ValueError("empty camera timeline")
    return steps


def simulate_keys(steps, fps=FPS):
    """Replay the demo's controls; returns its recording format {"frames": [{position, quaternion, fov}]}."""
    yaw, pitch = 0.0, 0.0                      # three.js Euler 'YXZ' after lookAt(0, 1.5, 0): identity
    position = np.array(START_POSITION, dtype=np.float64)
    dt = 1.0 / fps

    def frame():
        q = Rotation.from_euler("YXZ", [yaw, pitch, 0.0]).as_quat()     # x, y, z, w
        return {"position": dict(zip("xyz", position.tolist())), "quaternion": dict(zip("xyzw", q.tolist())),
                "fov": FOV}

    frames = [frame()]
    for keys, seconds in steps:
        for _ in range(max(0, int(round(seconds * fps)))):
            rot = Rotation.from_euler("YXZ", [yaw, pitch, 0.0]).as_matrix()
            direction = rot @ np.array([0.0, 0.0, -1.0])
            right = np.cross(direction, np.array([0.0, 1.0, 0.0]))
            norm = np.linalg.norm(right)
            right = right / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
            speed = SHIFT_SPEED if "Shift" in keys else MOVE_SPEED
            if "KeyW" in keys:
                position = position + direction * speed * dt
            if "KeyS" in keys:
                position = position - direction * speed * dt
            if "KeyA" in keys:
                position = position - right * speed * dt
            if "KeyD" in keys:
                position = position + right * speed * dt
            if "ArrowLeft" in keys:
                yaw += TURN_SPEED * dt
            if "ArrowRight" in keys:
                yaw -= TURN_SPEED * dt
            if "ArrowUp" in keys:
                pitch += TURN_SPEED * dt
            if "ArrowDown" in keys:
                pitch -= TURN_SPEED * dt
            frames.append(frame())
    return {"frames": frames}


# ---------------------------------------------------------------------------------------------------------------
# vendored from hymm/models/camera/geometry/{demo_precomputed,rotation_scale}.py and discretize_poses.py


def normalize_poses(poses, segment_boundaries, normalize_relative_to_first_frame=True, normalize_framewise=True,
                    normalize_displacement=True, normalize_displacement_per_video=False):
    if not normalize_relative_to_first_frame:
        return poses.copy()
    out = np.zeros_like(poses)
    identity = np.eye(4, dtype=poses.dtype)

    def scale_translations(translations):
        raw = float(np.quantile(np.linalg.norm(translations, axis=1), 0.99)) if len(translations) else 0.0
        if not np.isfinite(raw) or raw < 0:
            raise ValueError("Invalid displacement scale")
        return max(raw, 1e-6)

    for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
        if end <= start:
            continue
        relative = identity @ np.linalg.inv(poses[start])
        out[start] = identity
        if end - start > 1:
            out[start + 1:end] = relative @ poses[start + 1:end]
        if normalize_framewise and end - start > 1:
            integral = out[start:end]
            inverse = np.zeros_like(integral[:-1])
            rotation_inverse = np.swapaxes(integral[:-1, :3, :3], -1, -2)
            translation_inverse = -np.matmul(rotation_inverse, integral[:-1, :3, 3:4])
            inverse[:, :3, :3] = rotation_inverse
            inverse[:, :3, 3:4] = translation_inverse
            inverse[:, 3, 3] = 1.0
            out[start + 1:end] = np.matmul(inverse, integral[1:])
            if normalize_displacement and not normalize_displacement_per_video:
                out[start:end, :3, 3] /= scale_translations(out[start + 1:end, :3, 3])
    if normalize_framewise and normalize_displacement and normalize_displacement_per_video and len(out) > 1:
        chunks = [out[start + 1:end, :3, 3] for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:])
                  if end - start > 1]
        translations = np.concatenate(chunks) if chunks else out[:0, :3, 3]
        out[:, :3, 3] /= scale_translations(translations)
    return out


def resample(c2ws, intrinsics, target):
    n = len(c2ws)
    if n == target:
        return c2ws.copy(), intrinsics.copy()
    if n == 1:
        return np.repeat(c2ws, target, axis=0), np.repeat(intrinsics, target, axis=0)
    src, dst = np.linspace(0, 1, n), np.linspace(0, 1, target)
    poses = np.repeat(np.eye(4)[None], target, axis=0)
    poses[:, :3, :3] = Slerp(src, Rotation.from_matrix(c2ws[:, :3, :3]))(dst).as_matrix()
    poses[:, :3, 3] = np.stack([np.interp(dst, src, c2ws[:, axis, 3]) for axis in range(3)], axis=1)
    flat = intrinsics.reshape(n, -1)
    ks = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return poses.astype(c2ws.dtype), ks.reshape(target, 3, 3).astype(intrinsics.dtype)


def scale_framewise_rotations(poses, scale=1.0):
    scale = float(scale)
    if scale == 1.0:
        return poses
    rotvecs = Rotation.from_matrix(poses[:, :3, :3].astype(np.float64)).as_rotvec()
    scaled = rotvecs * scale
    if float(np.linalg.norm(scaled, axis=1).max()) >= np.pi - 1e-7:
        raise ValueError("the rotation gain makes one frame turn 180 degrees or more; lower rotation_scale")
    result = poses.copy()
    result[:, :3, :3] = Rotation.from_rotvec(scaled).as_matrix()
    return result


def recording_to_poses(camera):
    """The demo's recorded frames -> (c2ws relative to the first frame, intrinsics), as `prepare_turn` reads them
    (`discretize_poses` keeps the relative c2ws and the raw intrinsics; its action labels are unused here)."""
    camera = json.loads(camera) if isinstance(camera, str) else camera
    c2ws, ks = [], []
    for frame in camera["frames"]:
        pos = np.array([frame["position"][a] for a in "xyz"], dtype=np.float64)
        quat = np.array([frame["quaternion"][a] for a in "xyzw"], dtype=np.float64)
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = pos
        focal = 0.5 / np.tan(np.radians(float(frame.get("fov", FOV))) / 2)
        c2ws.append(pose)
        ks.append(np.array([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1.0]]))
    c2ws, ks = np.stack(c2ws), np.stack(ks)
    rel = np.linalg.inv(c2ws[0])
    relative = np.zeros_like(c2ws)
    relative[0] = np.eye(4)
    relative[1:] = rel @ c2ws[1:]
    return relative, ks


def turn_extrinsics(camera, num_frames, translation_scale=1.0, rotation_scale=2.5):
    """One segment's recorded camera -> normalized poses [((num_frames - 1) // 4) * 4 + 1, 4, 4] float32, with the
    inference gains applied (`prepare_turn` + `prepare_camera_inputs.apply_camera_gains`)."""
    count = (int(num_frames) - 1) // 4 * 4 + 1
    poses, ks = recording_to_poses(camera)
    if len(poses) != count:
        poses, ks = resample(poses, ks, count)
    poses = normalize_poses(poses.astype(np.float64), [0, count], **NORMALIZATION).astype(np.float32)
    if translation_scale != 1.0:
        poses = poses.copy()
        poses[:, :3, 3] *= translation_scale
    poses = scale_framewise_rotations(poses, rotation_scale)
    if not np.isfinite(poses).all() or np.abs(poses).max() > 1e6:
        raise ValueError("invalid normalized camera")
    return poses


class CameraPath:
    """A camera for one segment: the demo's recording (from a key timeline or loaded), and the gains."""

    def __init__(self, recording, translation_scale=1.0, rotation_scale=2.5, description=""):
        self.recording = recording
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.description = description

    def tensor(self, latent_frames):
        """[1, 4 * latent_frames, 4, 4] poses for a segment of `latent_frames`: the turn's poses for its
        4 * latent_frames video frames, with the three front copies of the first pose `_preprocess_camera` adds."""
        num_frames = 4 * latent_frames
        poses = turn_extrinsics(self.recording, num_frames, self.translation_scale, self.rotation_scale)
        poses = np.concatenate([np.repeat(poses[:1], 3, axis=0), poses], axis=0)
        if len(poses) != 4 * latent_frames:
            raise RuntimeError(f"camera poses {len(poses)} != 4 x {latent_frames} latent frames")
        return torch.from_numpy(poses).unsqueeze(0)

    def segment(self, script, index):
        return self.tensor(script.segment_frames(index))
