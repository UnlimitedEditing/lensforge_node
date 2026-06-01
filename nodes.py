"""
lensforge_node/nodes.py — ComfyUI node for LensForge camera codes.

The LensForge UI generates a compact base36 code representing a camera
trajectory, wrapped in brackets: [LF1:29g0001412142a14512a]

The user pastes this anywhere in their negative prompt alongside normal
negative terms, e.g.:
    "blurry, low quality [LF1:29g0001412142a14512a]"

This node:
  1. Strips the [LF1:...] token from the negative prompt string
  2. Decodes the camera keyframes from it
  3. Generates per-frame 4×4 camera-to-world pose matrices
  4. Returns the poses (for WanFunControlCameraEmbed) and the cleaned
     negative prompt (for WanVideoTextEncode)

Wiring in the Graydient workflow:
  negative_prompt (STRING field)
       ↓
  [LensForgeCameraNode]
       ├─ camera_poses  → [WanFunControlCameraEmbed] → WANVIDIMAGE_EMBEDS → [WanVideoSampler]
       └─ negative_text → [WanVideoTextEncode (negative)] → WANVIDEOTEXTEMBEDS → [WanVideoSampler]

Fields config (Graydient → Fields tab):
  negative_prompt  → LensForgeCameraNode, widget "negative_prompt"
  positive_prompt  → WanVideoTextEncode (positive), widget "text"
  seed             → WanVideoSampler, widget "seed"

⚠  CAMERA_POSES output type: verify against WanFunControlCameraEmbed's
   input socket type string. Run inspect_workflow.py on a working Fun-Camera
   graph and look for the camera_poses input type. Update CAMERA_POSES_TYPE
   below if it differs.
"""

import json
import math
import re

# ── Encoding tables ───────────────────────────────────────────────────────────

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

# ── Base36 helpers ─────────────────────────────────────────────────────────────

def _enc(n: int, width: int) -> str:
    s = ''
    for _ in range(width):
        s = _B36[n % 36] + s
        n //= 36
    return s

def _dec(s: str) -> int:
    return int(s, 36)

# ── Encode / decode ────────────────────────────────────────────────────────────

def encode(frames: int, fps: int, resolution: str, keyframes: list) -> str:
    """
    Encode a camera shot as a compact alphanumeric string.
    Format: LF1:<FF><P><R>[<SS><EE><M><S>...]
      FF = total frames   (2 base36 chars, max 1295)
      P  = fps index      (1 char, index into FPS_OPTIONS)
      R  = resolution idx (1 char, index into RESOLUTIONS)
      per keyframe:
        SS = start frame  (2 chars)
        EE = end frame    (2 chars)
        M  = motion type  (1 char, index into MOTION_TYPES)
        S  = speed×35     (1 char, 0–35)
    """
    fps_idx = FPS_OPTIONS.index(fps) if fps in FPS_OPTIONS else 2
    res_idx = RESOLUTIONS.index(resolution) if resolution in RESOLUTIONS else 0
    code = 'LF1:' + _enc(frames, 2) + _enc(fps_idx, 1) + _enc(res_idx, 1)
    for kf in keyframes:
        m_idx = MOTION_TYPES.index(kf['motion']) if kf['motion'] in MOTION_TYPES else 0
        s_enc = round(kf['speed'] * 35)
        code += _enc(kf['start'], 2) + _enc(kf['end'], 2) + _enc(m_idx, 1) + _enc(s_enc, 1)
    return code


def decode(code: str) -> dict:
    """
    Decode an LF1 code string back into a scene description dict.
    Accepts the raw code with or without the LF1: prefix.
    Returns: { frames, fps, resolution, keyframes }
    """
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
        start    = _dec(code[pos:pos+2])
        end      = _dec(code[pos+2:pos+4])
        m_idx    = _dec(code[pos+4])
        s_enc    = _dec(code[pos+5])
        motion   = MOTION_TYPES[m_idx] if m_idx < len(MOTION_TYPES) else 'static'
        speed    = s_enc / 35.0
        keyframes.append({'start': start, 'end': end, 'motion': motion, 'speed': speed})
        pos += 6

    return {'frames': frames, 'fps': fps, 'resolution': resolution, 'keyframes': keyframes}


def strip_lf_token(text: str) -> tuple[str, str | None]:
    """
    Find and remove the first [LF1:...] token from `text`.
    Returns (cleaned_text, raw_code_without_brackets) or (text, None) if not found.
    """
    match = re.search(r'\[LF1:[0-9a-z]+\]', text, re.IGNORECASE)
    if not match:
        return text, None
    raw = match.group(0)[1:-1]   # strip [ ]
    cleaned = text[:match.start()] + text[match.end():]
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip().strip(',').strip()
    return cleaned, raw


# ── Camera trajectory math ────────────────────────────────────────────────────

_MAX_PAN   = math.radians(30)
_MAX_TILT  = math.radians(20)
_MAX_DOLLY = 0.8
_MAX_CRANE = 0.4
_MAX_ORBIT = math.pi * 2

def _I():  return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
def _Ry(a):
    c,s = math.cos(a),math.sin(a)
    return [[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]]
def _Rx(a):
    c,s = math.cos(a),math.sin(a)
    return [[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]]
def _T(x,y,z): return [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]
def _mul(a,b):
    r=[[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4): r[i][j]+=a[i][k]*b[k][j]
    return r
def _ease(t): t=max(0.,min(1.,t)); return t*t*(3-2*t)

def _pose(motion, speed, t):
    t = _ease(t); s = speed
    if   motion == 'pan_left':   return _Ry(-s*_MAX_PAN*t)
    elif motion == 'pan_right':  return _Ry( s*_MAX_PAN*t)
    elif motion == 'tilt_up':    return _Rx(-s*_MAX_TILT*t)
    elif motion == 'tilt_down':  return _Rx( s*_MAX_TILT*t)
    elif motion in ('zoom_in',  'dolly_fwd'): return _T(0,0,-s*_MAX_DOLLY*t)
    elif motion in ('zoom_out', 'dolly_bwd'): return _T(0,0, s*_MAX_DOLLY*t)
    elif motion == 'crane_up':   return _T(0, s*_MAX_CRANE*t,0)
    elif motion == 'crane_down': return _T(0,-s*_MAX_CRANE*t,0)
    elif motion in ('orbit_cw','orbit_ccw'):
        d=1. if motion=='orbit_cw' else -1.
        a=d*s*_MAX_ORBIT*t
        return _mul(_T(1.5*math.sin(a),0,1.5*(math.cos(a)-1)),_Ry(-a))
    return _I()

def generate_poses(keyframes, total_frames):
    kfs = sorted(keyframes, key=lambda k: k['start'])
    poses = []
    for f in range(total_frames):
        kf = next((k for k in kfs if k['start'] <= f < k['end']), None)
        if kf is None:
            poses.append(_I())
        else:
            span = max(1, kf['end'] - kf['start'])
            poses.append(_pose(kf['motion'], kf['speed'], (f - kf['start']) / span))
    return poses


# ── ComfyUI node ──────────────────────────────────────────────────────────────

CAMERA_POSES_TYPE = 'CAMERA_POSES'


class LensForgeCameraNode:
    CATEGORY     = 'LensForge'
    FUNCTION     = 'execute'
    RETURN_TYPES  = (CAMERA_POSES_TYPE, 'STRING')
    RETURN_NAMES  = ('camera_poses',    'negative_text')

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

    def execute(self, negative_prompt: str, num_frames: int, width: int, height: int):
        import torch

        cleaned, raw_code = strip_lf_token(negative_prompt)

        if raw_code:
            try:
                scene = decode(raw_code)
                keyframes   = scene['keyframes']
                total_frames = scene.get('frames', num_frames)
            except Exception as e:
                print(f'[LensForgeCameraNode] decode error: {e} — falling back to static')
                keyframes    = []
                total_frames = num_frames
        else:
            print('[LensForgeCameraNode] no LF1 token found — using static camera')
            keyframes    = []
            total_frames = num_frames

        if not keyframes:
            keyframes = [{'start': 0, 'end': total_frames, 'motion': 'static', 'speed': 0.0}]

        poses  = generate_poses(keyframes, total_frames)
        tensor = torch.tensor(poses, dtype=torch.float32)   # (T, 4, 4)

        return (tensor, cleaned)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    'LensForgeCameraNode': LensForgeCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LensForgeCameraNode': 'LensForge Camera (code → poses)',
}
