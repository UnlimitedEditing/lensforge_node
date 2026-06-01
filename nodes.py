"""
lensforge_node/nodes.py — ComfyUI node for LensForge camera codes.

Strips a [LF1:...] token from the negative prompt, decodes the camera
keyframes, and outputs CAMERACTRL_POSES for WanVideoFunCameraEmbeds.

Wiring in the Graydient workflow:
  negative_prompt (STRING field)
       ↓
  [LensForgeCameraNode]
       ├─ camera_poses (CAMERACTRL_POSES) → [WanVideoFunCameraEmbeds] → WANVIDIMAGE_EMBEDS → [WanVideoSampler]
       └─ negative_text (STRING)          → [WanVideoTextEncode (negative)]

Fields config (Graydient → Fields tab):
  negative_prompt  → LensForgeCameraNode, widget "negative_prompt"
  positive_prompt  → WanVideoTextEncode (positive), widget "text"
  seed             → WanVideoSampler, widget "seed"
"""

import json
import math
import re
import numpy as np

# ── Encoding tables (must match app.js) ──────────────────────────────────────

_B36 = '0123456789abcdefghijklmnopqrstuvwxyz'

RESOLUTIONS = ['848x480', '1280x720', '640x480', '480x848', '720x1280']
FPS_OPTIONS = [8, 12, 16, 24, 25, 30]
MOTION_TYPES = [
    'static',
    'pan_left',  'pan_right',
    'tilt_up',   'tilt_down',
    'zoom_in',   'zoom_out',
    'dolly_fwd', 'dolly_bwd',
    'crane_up',  'crane_down',
    'orbit_cw',  'orbit_ccw',
]

def _enc(n, width):
    s = ''
    for _ in range(width):
        s = _B36[n % 36] + s
        n //= 36
    return s

def _dec(s):
    return int(s, 36)

def decode(code):
    code = code.strip()
    if code.upper().startswith('LF1:'):
        code = code[4:]
    if len(code) < 4:
        raise ValueError(f'LensForge code too short: {repr(code)}')
    frames  = _dec(code[0:2])
    fps_idx = _dec(code[2])
    res_idx = _dec(code[3])
    fps        = FPS_OPTIONS[fps_idx]  if fps_idx < len(FPS_OPTIONS)  else 16
    resolution = RESOLUTIONS[res_idx]  if res_idx < len(RESOLUTIONS)  else '848x480'
    keyframes = []
    pos = 4
    while pos + 6 <= len(code):
        start  = _dec(code[pos:pos+2])
        end    = _dec(code[pos+2:pos+4])
        m_idx  = _dec(code[pos+4])
        s_enc  = _dec(code[pos+5])
        motion = MOTION_TYPES[m_idx] if m_idx < len(MOTION_TYPES) else 'static'
        speed  = s_enc / 35.0
        keyframes.append({'start': start, 'end': end, 'motion': motion, 'speed': speed})
        pos += 6
    return {'frames': frames, 'fps': fps, 'resolution': resolution, 'keyframes': keyframes}

def strip_lf_token(text):
    match = re.search(r'\[LF1:[0-9a-z]+\]', text, re.IGNORECASE)
    if not match:
        return text, None
    raw     = match.group(0)[1:-1]
    cleaned = text[:match.start()] + text[match.end():]
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip().strip(',').strip()
    return cleaned, raw

# ── Camera trajectory math ────────────────────────────────────────────────────

_MAX_PAN   = math.radians(30)
_MAX_TILT  = math.radians(20)
_MAX_DOLLY = 0.8
_MAX_CRANE = 0.4
_MAX_ORBIT = math.pi * 2

def _I():
    return np.eye(4, dtype=np.float32)

def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0,0]=c; m[0,2]=s; m[2,0]=-s; m[2,2]=c
    return m

def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1,1]=c; m[1,2]=-s; m[2,1]=s; m[2,2]=c
    return m

def _T(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0,3]=x; m[1,3]=y; m[2,3]=z
    return m

def _ease(t):
    t = max(0., min(1., t))
    return t * t * (3 - 2 * t)

def _c2w(motion, speed, t):
    t = _ease(t); s = speed
    if   motion == 'pan_left':   return _Ry(-s * _MAX_PAN  * t)
    elif motion == 'pan_right':  return _Ry( s * _MAX_PAN  * t)
    elif motion == 'tilt_up':    return _Rx(-s * _MAX_TILT * t)
    elif motion == 'tilt_down':  return _Rx( s * _MAX_TILT * t)
    elif motion in ('zoom_in',  'dolly_fwd'): return _T(0, 0, -s * _MAX_DOLLY * t)
    elif motion in ('zoom_out', 'dolly_bwd'): return _T(0, 0,  s * _MAX_DOLLY * t)
    elif motion == 'crane_up':   return _T(0,  s * _MAX_CRANE * t, 0)
    elif motion == 'crane_down': return _T(0, -s * _MAX_CRANE * t, 0)
    elif motion in ('orbit_cw', 'orbit_ccw'):
        d = 1. if motion == 'orbit_cw' else -1.
        a = d * s * _MAX_ORBIT * t
        r = 1.5
        tr = _T(r * math.sin(a), 0, r * (math.cos(a) - 1))
        return tr @ _Ry(-a)
    return _I()


def generate_cameractrl_poses(keyframes, total_frames, fx=1.0, fy=1.0, cx=0.5, cy=0.5):
    """
    Generate poses in CameraCtrl format consumed by WanVideoFunCameraEmbeds.

    Each pose entry: [frame_idx, fx, fy, cx, cy, 0, 0, *w2c_3x4_flat]
    - Indices 1-4: normalised camera intrinsics (fx≈1 gives ~53° FoV)
    - Indices 7-18: world-to-camera matrix, row-major 3×4

    fx/fy default 1.0 → focal_px = 1.0 × width (reasonable cinematic FoV).
    cx/cy default 0.5 → principal point at image centre.
    """
    kfs = sorted(keyframes, key=lambda k: k['start'])
    poses = []

    for frame in range(total_frames):
        kf = next((k for k in kfs if k['start'] <= frame < k['end']), None)
        if kf is None:
            c2w = _I()
        else:
            span = max(1, kf['end'] - kf['start'])
            c2w  = _c2w(kf['motion'], kf['speed'], (frame - kf['start']) / span)

        w2c     = np.linalg.inv(c2w)           # 4×4 world-to-camera
        w2c_3x4 = w2c[:3, :].flatten().tolist() # 12 values, row-major

        pose = [float(frame), fx, fy, cx, cy, 0.0, 0.0] + w2c_3x4
        poses.append(pose)

    return poses   # list of 19-element lists — CAMERACTRL_POSES type


# ── ComfyUI node ──────────────────────────────────────────────────────────────

class LensForgeCameraNode:
    CATEGORY     = 'LensForge'
    FUNCTION     = 'execute'
    RETURN_TYPES  = ('CAMERACTRL_POSES', 'STRING')
    RETURN_NAMES  = ('camera_poses',     'negative_text')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'negative_prompt': ('STRING', {'multiline': True,  'default': ''}),
                'num_frames':      ('INT',    {'default': 81,  'min': 9,  'max': 257}),
                'width':           ('INT',    {'default': 848, 'min': 64, 'max': 2048}),
                'height':          ('INT',    {'default': 480, 'min': 64, 'max': 2048}),
            }
        }

    def execute(self, negative_prompt, num_frames, width, height):
        cleaned, raw_code = strip_lf_token(negative_prompt)

        if raw_code:
            try:
                scene        = decode(raw_code)
                keyframes    = scene['keyframes']
                total_frames = scene.get('frames', num_frames)
            except Exception as e:
                print(f'[LensForgeCameraNode] decode error: {e} — falling back to static')
                keyframes    = []
                total_frames = num_frames
        else:
            print('[LensForgeCameraNode] no LF1 token found — static camera')
            keyframes    = []
            total_frames = num_frames

        if not keyframes:
            keyframes = [{'start': 0, 'end': total_frames, 'motion': 'static', 'speed': 0.0}]

        poses = generate_cameractrl_poses(keyframes, total_frames)
        return (poses, cleaned)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    'LensForgeCameraNode': LensForgeCameraNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LensForgeCameraNode': 'LensForge Camera (code → poses)',
}
