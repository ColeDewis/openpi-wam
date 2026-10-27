"""LIBERO simulator actor/learner for RLPD and HG-Dagger.

`wam/inference/main.py` is the real-WAM actor. This module is the simulator
analogue: it steps LIBERO, records the same transition schema, and trains the
same SAC / BC agents from `serl_launcher`. Everything runs in one process on
the GPU box, so the learner and actor share the replay buffers directly.

Control is a PS4 (DualShock 4) controller, read by the browser Gamepad API on
the stream page and POSTed to this process (so it works headless / over an SSH
tunnel; the controller is plugged into / paired with the machine running the
browser):

    OPTIONS (top right)   start the episode (when idle) / mark SUCCESS and flush
    SHARE   (top left)    mark FAILURE and flush
    left stick            x (up/down) and y (left/right) translation
    right stick           pitch (up/down) and yaw (left/right)
    triangle / cross      +z / -z
    square / circle       roll left / roll right
    L2 / R2 (triggers)    gripper open / close (release = hold)
    PS (center button)    hand control back to the policy

Intervention is latched: the first stick/trigger/button input takes control and
the policy is not queried at all (when the pad is idle the robot holds still and
the gripper holds) until PS is pressed, ``--handback-s`` of idle input elapses
(if set), or the episode ends. Frames where the human actually commands
something are marked ``intervened=True`` and enter the demo buffer; idle hold
frames do not.

The gripper is discrete, as in the hil-serl paper: -1 open, 0 hold, +1 close.
RLPD learns it with a DQN "grasp critic" (``SACAgentHybridSingleArm``) and a
small penalty on every toggle of the commanded gripper state (``--grasp-penalty``).
HG-Dagger's BC regresses a persistent open/closed label instead (a DQN needs rewards).

Agent actions are normalized to [-1, 1]; ``--max-step-m`` / ``--max-step-rad``
set the physical per-step motion that +-1 maps to; the sim passes them to
``ActionAdapter`` as ``scales`` (the adapter is the only place that knows units).

The observation is the LIBERO image pair + end-effector proprio, and the agent
action is the normalized 7-dim LIBERO delta-pose command. `ActionAdapter`
isolates the normalized->robot-units mapping, so a future wrench / joint-torque
controller can be added without touching the agents or the replay format.

Run with the helper script:
    uv run scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 --display
    uv run scripts/train_hil_libero.py --algorithm hgdagger \
        --suite libero_goal --task-id 1 --display

``--display`` starts a headless-safe MJPEG stream plus a browser controller
endpoint at ``:8080``; open the printed URL (or an SSH tunnel to it) in Chrome,
focus the tab and press any controller button. No VNC/OpenCV GUI is required.
"""

# The MuJoCo backend and third_party/libero must be configured before importing
# libero/mujoco, so this module intentionally has imports after that setup.
# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
from http import server as http_server
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Literal

import numpy as np

# MuJoCo renders offscreen (EGL is the most reliable backend on a headless box).
# Export MUJOCO_GL=glx to opt into native GLX.
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LIBERO_ROOT = _REPO_ROOT / "third_party/libero"
if str(_LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIBERO_ROOT))

from flax.training import checkpoints
import imageio.v2 as imageio
import jax
from openpi_client import image_tools

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.hil import encoders as _encoders
from openpi.training.hil import learner as _learner
from openpi.training.hil.action_adapter import ActionAdapter
from openpi.training.hil.transitions import make_transition
from openpi.training.rlt import envs as _envs

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Full-scale per-step output of robosuite's default OSC_POSE controller (what LIBERO
# uses): an env action of +-1 means 0.05 m / 0.5 rad per step. Verify against
# ``robot.controller.output_max`` if you changed the controller config.
_OSC_MAX_M = 0.05
_OSC_MAX_RAD = 0.5
_WHITE, _GREEN, _ORANGE = (255, 255, 255), (80, 255, 80), (255, 160, 0)


def _discretize_gripper(g: float) -> float:
    """Snap a gripper command to {-1 open, 0 hold, +1 close} (hil-serl's +-0.5 thresholds).

    robosuite's Panda gripper integrates ``sign(action)`` each step, so 0 holds the current width.
    """
    return 0.0 if abs(g) < 0.5 else (1.0 if g > 0 else -1.0)


# ════════════════════════════════════════════════════════════════════════════
# LIBERO observation
# ════════════════════════════════════════════════════════════════════════════


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite quaternion (x, y, z, w) -> axis-angle; copied from LIBERO."""
    quat = np.asarray(quat, np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


class LiberoObsAdapter:
    """Raw LIBERO obs -> the encoder's ``{front_image, wrist_image, state}`` dict.

    The images are rotated 180 degrees to match LIBERO's training/eval
    convention, resized with padding to the encoder resolution, and kept as
    uint8. The state is the same 8-dim end-effector state used by the LIBERO
    VLA bridge; ``include_velocity`` appends the 9 joint+gripper velocities
    (the RLT convention) and gives a 17-dim state.
    """

    def __init__(self, image_size: tuple[int, int] = (128, 128), *, include_velocity: bool = False):
        self.image_size = tuple(image_size)
        self.include_velocity = include_velocity

    @property
    def proprio_dim(self) -> int:
        return 8 + (9 if self.include_velocity else 0)

    def _image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
        image = image_tools.resize_with_pad(image, self.image_size[0], self.image_size[1])
        return image_tools.convert_to_uint8(image)

    def state(self, obs: dict) -> np.ndarray:
        parts = [
            np.asarray(obs["robot0_eef_pos"], np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], np.float32),
        ]
        if self.include_velocity:
            parts.append(np.asarray(obs["robot0_joint_vel"], np.float32))
            parts.append(np.asarray(obs["robot0_gripper_qvel"], np.float32))
        return np.concatenate(parts).astype(np.float32)

    def encode(self, obs: dict) -> dict:
        front = self._image(obs["agentview_image"])
        wrist = self._image(obs["robot0_eye_in_hand_image"])
        # (T=1, H, W, C) and (T=1, D): the upstream agents stack a time axis
        # even when T=1 (the WAM actor uses the same convention).
        return {
            "front_image": front[None],
            "wrist_image": wrist[None],
            "state": self.state(obs)[None],
        }

    def display(self, obs: dict) -> np.ndarray:
        """RGB frame with the front and wrist cameras side by side."""
        front = np.asarray(obs["agentview_image"])[::-1, ::-1]
        wrist = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
        front = image_tools.convert_to_uint8(front)
        wrist = image_tools.convert_to_uint8(wrist)
        return np.concatenate([front, wrist], axis=1)


# ════════════════════════════════════════════════════════════════════════════
# Remote display + gamepad
# ════════════════════════════════════════════════════════════════════════════

_PAD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>HIL LIBERO</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 html, body { height:100%; margin:0; }
 body { background:#111; color:#eee; font-family: ui-monospace, monospace; display:flex; flex-direction:column; }
 #view { position:relative; flex:1 1 0; min-height:0; margin:8px 8px 0 8px; }
 #view img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
             background:#000; border:1px solid #444; box-sizing:border-box; }
 #help { flex:0 0 auto; padding:8px 16px 10px 16px; font-size:13px; line-height:1.5; }
 #pad { color:#8f8; margin-bottom:4px; }
 #pad.bad { color:#fa6; }
 #grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:0 24px; }
 b { color:#fff; }
 .dim { color:#999; }
</style></head>
<body>
<div id="view"><img src="/stream.mjpg" alt="LIBERO stream"></div>
<div id="help">
 <div id="pad" class="bad">controller: none &mdash; focus this tab and press any button on the pad</div>
 <div id="grid">
  <div><b>OPTIONS</b> start episode / mark <b>SUCCESS</b></div>
  <div><b>SHARE</b> mark <b>FAILURE</b></div>
  <div><b>Left stick</b> x (up/down), y (left/right)</div>
  <div><b>Right stick</b> pitch (up/down), yaw (left/right)</div>
  <div><b>&#9651; / &#10005;</b> z up / z down</div>
  <div><b>&#9633; / &#9675;</b> roll left / roll right</div>
  <div><b>L2 / R2</b> gripper open / close (release = hold)</div>
  <div><b>PS</b> hand control back to the policy</div>
  <div class="dim">first stick/trigger/button input = human takeover; the policy stays off until PS</div>
 </div>
</div>
<script>
const NBTN = 18;
let busy = false, padId = "", padBad = false;
function neutral() { return {axes:[0,0,0,0], buttons:Array(NBTN).fill(false)}; }
function readPad() {
  // Never report a stale "held" state: an unfocused tab sends neutral.
  if (!document.hasFocus() || document.hidden) return neutral();
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  const g = [...pads].find(p => p && p.connected);
  if (!g) { padId = ""; return neutral(); }
  padId = g.id;
  padBad = g.mapping !== "standard";
  return {
    axes: [0,1,2,3].map(i => +((g.axes[i] || 0).toFixed(3))),
    buttons: Array.from({length: NBTN}, (_, i) => !!(g.buttons[i] && g.buttons[i].pressed)),
  };
}
function tick() {
  const s = readPad();
  const el = document.getElementById("pad");
  if (!padId) {
    el.className = "bad";
    el.textContent = document.hasFocus()
      ? "controller: none \\u2014 press any button on the pad (Chrome only shows it after a press)"
      : "controller: tab not focused \\u2014 click this page";
  } else {
    el.className = padBad ? "bad" : "";
    const held = s.buttons.map((p, i) => p ? i : -1).filter(i => i >= 0).join(",");
    el.textContent = "controller: " + padId + (padBad ? "  [non-standard mapping: button indices will be wrong, use Chrome]" : "")
      + "   pressed: [" + held + "]";
  }
  if (busy) return;
  busy = true;
  fetch("/pad", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(s)})
    .catch(() => {}).finally(() => { busy = false; });
}
setInterval(tick, 33);
</script>
</body></html>
"""


class MJPEGServer:
    """Serve the latest frame as MJPEG and accept browser gamepad state.

    The GPU box only needs a headless OpenCV build: frames are JPEG-encoded
    with ``cv2.imencode`` and streamed over HTTP. Open the printed URL from the
    laptop (or an SSH tunnel); the page polls the browser Gamepad API and POSTs
    the pad state to ``/pad`` on the same process.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, on_pad: Callable[[dict], None] | None = None):
        self.host = host
        self.port = port
        self.on_pad = on_pad
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._server: http_server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._cv2 = None

    def start(self) -> bool:
        server = self

        class Handler(http_server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = _PAD_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    while not server.stop_event.is_set():
                        jpeg = server.latest()
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode()
                                + b"\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        time.sleep(0.05)
                elif self.path == "/healthz":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK\n")
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                if self.path != "/pad":
                    self.send_error(404)
                    return
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    data = json.loads(self.rfile.read(n) or b"{}")
                    if server.on_pad is not None:
                        server.on_pad(data)
                except Exception as exc:
                    print(f"[pad] bad request: {exc!r}")
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        try:
            self._server = http_server.ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            print(f"[stream] could not bind {self.host}:{self.port} ({exc}); live view disabled")
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(
            f"[stream] live view + gamepad: http://<gb10-ip>:{self.port}/ "
            f"(or: ssh -L {self.port}:localhost:{self.port} <user>@<gb10>); open in Chrome"
        )
        return True

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def update(self, frame: np.ndarray, label: str = "", color: tuple[int, int, int] = (255, 255, 255)) -> None:
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        cv2 = self._cv2
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        if label:
            cv2.putText(bgr, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color[::-1], 1, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = buffer.tobytes()

    def close(self) -> None:
        self.stop_event.set()
        if self._server is not None:
            with contextlib.suppress(Exception):
                self._server.shutdown()
                self._server.server_close()


# ════════════════════════════════════════════════════════════════════════════
# Gamepad operator
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class PadEvent:
    options: bool = False  # OPTIONS: start (idle) / mark success (recording)
    share: bool = False  # SHARE: mark failure


class GamepadOperator:
    """PS4 controller state, fed by the browser page via ``apply_pad``.

    Indices are the W3C "standard" gamepad mapping, which Chrome applies to the
    DualShock 4. The HTTP thread calls ``apply_pad``; the control loop only
    reads the shared state, so slow model steps never block input handling.

    Intervention is *latched*: the first stick/trigger/button input takes control
    and the policy stays off until PS is pressed (or ``handback_s`` of idle input
    elapses, if > 0) or ``reset()`` is called at episode start. While latched and
    idle the operator commands zero motion and a *hold* gripper. Stale pad data
    (tab hidden, connection lost) keeps the latch but commands nothing, so a
    network hiccup never hands the robot back to the policy.

    The gripper is discrete like hil-serl's spacemouse buttons: L2 = open (-1),
    R2 = close (+1), neither = hold (0).
    """

    CROSS, CIRCLE, SQUARE, TRIANGLE = 0, 1, 2, 3
    L2, R2 = 6, 7
    SHARE, OPTIONS = 8, 9
    PS = 16  # if Chrome never reports 16 (OS/Steam can swallow it), use 17 (touchpad click)
    N_BUTTONS = 18

    def __init__(
        self, takeover_step: float = 0.5, deadzone: float = 0.15, stale_s: float = 0.3, handback_s: float = 0.0
    ):
        self.takeover_step = float(takeover_step)
        self.deadzone = float(deadzone)
        self.stale_s = float(stale_s)
        self.handback_s = float(handback_s)
        self._lock = threading.Lock()
        self._axes = np.zeros(4, np.float32)  # LX, LY, RX, RY (stick up = negative)
        self._buttons = np.zeros(self.N_BUTTONS, bool)
        self._last_msg = float("-inf")
        self._event = PadEvent()
        self._latched = False
        self._release = False  # PS pressed, consumed by takeover()
        self._wait_idle = False  # after hand-back, ignore input until the pad is idle once
        self._last_input = float("-inf")
        self.enabled = False

    def apply_pad(self, state: dict) -> None:
        axes = np.zeros(4, np.float32)
        a = np.nan_to_num(np.asarray(state.get("axes", []), np.float32))[:4]
        axes[: len(a)] = a
        buttons = np.zeros(self.N_BUTTONS, bool)
        b = np.asarray(state.get("buttons", []), bool)[: self.N_BUTTONS]
        buttons[: len(b)] = b
        with self._lock:
            rising = buttons & ~self._buttons
            if rising[self.OPTIONS]:
                self._event.options = True
            if rising[self.SHARE]:
                self._event.share = True
            if rising[self.PS]:
                self._release = True
            self._axes, self._buttons = axes, buttons
            self._last_msg = time.monotonic()

    def poll(self) -> PadEvent:
        with self._lock:
            event, self._event = self._event, PadEvent()
        return event

    def reset(self) -> None:
        """Drop latched presses and the intervention latch (episode start)."""
        with self._lock:
            self._event = PadEvent()
            self._latched = False
            self._release = False
            self._wait_idle = False

    def _shape(self, v: np.ndarray) -> np.ndarray:
        mag = np.abs(v)
        dz = self.deadzone
        return np.where(mag < dz, 0.0, np.sign(v) * (mag - dz) / (1.0 - dz)).astype(np.float32)

    def takeover(self) -> tuple[bool, bool, np.ndarray]:
        """``(active, moving, cmd)``.

        ``active``: the human owns the robot, so do not query the policy.
        ``moving``: the human is commanding something this frame (demo-worthy).
        ``cmd``: ``[x, y, z, roll, pitch, yaw, gripper]``; the first six are scaled to
        +-takeover_step, the gripper is -1 (L2 open) / 0 (hold) / +1 (R2 close).
        """
        zeros = np.zeros(7, np.float32)
        now = time.monotonic()
        with self._lock:
            fresh = (now - self._last_msg) < self.stale_s
            axes, b = self._axes.copy(), self._buttons.copy()
            if not fresh:
                return self._latched, False, zeros
            lx, ly, rx, ry = self._shape(axes)
            # Flip any sign here if a direction feels backwards in the sim.
            cmd = np.array(
                [
                    -ly,  # left stick up    -> +x
                    lx,  # left stick right -> +y
                    float(b[self.TRIANGLE]) - float(b[self.CROSS]),  # triangle +z, cross -z
                    float(b[self.CIRCLE]) - float(b[self.SQUARE]),  # circle roll right, square roll left
                    -ry,  # right stick up   -> +pitch
                    rx,  # right stick right -> +yaw
                ],
                np.float32,
            )
            grip = float(b[self.R2]) - float(b[self.L2])  # R2 close, L2 open, neither/both = hold
            moving = bool(np.any(cmd != 0.0) or grip != 0.0)
            if self._release:
                self._release = False
                if self._latched:
                    self._latched = False
                    self._wait_idle = True
            if self._wait_idle:
                if moving:
                    return False, False, zeros  # still holding input after hand-back: policy runs
                self._wait_idle = False
            if moving:
                self._latched = True
                self._last_input = now
            elif self._latched and self.handback_s > 0 and now - self._last_input > self.handback_s:
                self._latched = False
            full = np.concatenate([cmd * self.takeover_step, [grip]]).astype(np.float32)
            return self._latched, moving and self._latched, full

    def human_action(self, cmd: np.ndarray) -> np.ndarray:
        """Full 7-dim action from the pad alone: the policy contributes nothing."""
        return np.clip(np.asarray(cmd, np.float32), -1.0, 1.0)

    def close(self) -> None:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class LiberoEnvConfig:
    """The subset of ``rlt.envs.EnvWorker``'s config that the pool reads."""

    suite: str = "libero_goal"
    task_id: int = 0
    total_num_envs: int = 1
    seed: int = 42
    camera_size: int = 128
    action_chunk: int = 1
    action_env_dim: int = 7
    group_size: int = 1

    @property
    def task_ids(self) -> list[int]:
        return [self.task_id]


@dataclasses.dataclass
class SimConfig:
    """CLI config for the LIBERO RLPD/HG-Dagger loop."""

    algorithm: Literal["rlpd", "hgdagger"] = "rlpd"
    suite: str = "libero_goal"
    task_id: int = 0
    total_num_envs: int = 1
    camera_size: int = 128
    image_size: tuple[int, int] = (128, 128)
    include_velocity: bool = False
    proprio_dim: int = 0  # 0 -> 8, or 17 with velocities

    max_episode_steps: int = 0  # 0 -> the suite's standard limit
    max_episodes: int = 10
    max_env_steps: int = 0  # 0 -> max_episodes * max_episode_steps
    control_hz: float = 20.0
    auto_start: bool = False

    seed: int = 42
    action_dim: int = 7
    encoder: _encoders.EncoderConfig = dataclasses.field(
        default_factory=lambda: _encoders.EncoderConfig(kind="hilserl", proprio_dim=8, image_size=(128, 128))
    )
    action_kind: Literal["delta_pose", "wrench", "joint_torque"] = "delta_pose"

    checkpoint_dir: str = "checkpoints/hil/libero"
    resume: bool = False
    keep: int = 20
    checkpoint_period: int = 1_000
    batch_size: int = 256
    cta_ratio: int = 5
    utd: float = 1.0
    max_updates_per_episode: int = 200
    training_starts: int = 500
    pretrain_steps: int = 5_000
    replay_buffer_capacity: int = 100_000
    log_interval: int = 10

    eval_interval: int = 5  # episodes
    eval_episodes: int = 5
    eval_only: bool = False
    video_dir: str = "videos/hil_libero"

    display: bool = True
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    takeover_step: float = 0.5  # fraction of the normalized action range at full stick
    handback_s: float = 0.0  # >0: hand back to the policy after this many idle seconds (0 = only PS)
    # Reward shaping for the gripper DQN: added whenever the commanded gripper state toggles
    # (hil-serl's GripperPenaltyWrapper, penalty=-0.02). Negative = a cost.
    grasp_penalty: float = -0.02

    # Physical per-step motion that a normalized action of +-1 maps to. The agent and the
    # replay buffer always see normalized actions; the scaling lives in ActionAdapter.scales.
    max_step_m: float = 0.02
    max_step_rad: float = 0.05

    def __post_init__(self):
        if self.suite not in _MAX_STEPS:
            raise ValueError(f"unknown LIBERO suite {self.suite!r}; expected one of {tuple(_MAX_STEPS)}")
        if self.encoder.kind != "hilserl":
            raise ValueError(
                "the LIBERO sim currently supports encoder.kind='hilserl' (images + proprio). "
                "The WAM rltoken/timm encoders are bound to WAM low_dim fields."
            )
        self.image_size = tuple(self.image_size)
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be (height, width), got {self.image_size}")
        self.encoder.image_size = self.image_size
        self.encoder.proprio_dim = int(self.proprio_dim) or (8 + (9 if self.include_velocity else 0))
        self.encoder.image_keys = ("front_image", "wrist_image")
        self.max_episode_steps = int(self.max_episode_steps) or _MAX_STEPS[self.suite]
        self.max_env_steps = int(self.max_env_steps) or self.max_episodes * self.max_episode_steps
        if self.action_kind != "delta_pose":
            raise ValueError(
                "LIBERO expects delta_pose actions; the wrench/joint_torque modes are for the "
                "WAM actor. The ActionAdapter extension point is ready for them."
            )
        if not (0.0 < self.max_step_m <= _OSC_MAX_M and 0.0 < self.max_step_rad <= _OSC_MAX_RAD):
            raise ValueError(
                f"max_step_m must be in (0, {_OSC_MAX_M}] and max_step_rad in (0, {_OSC_MAX_RAD}] "
                "(the OSC controller's full scale)"
            )
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        if self.eval_only and checkpoints.latest_checkpoint(self.checkpoint_dir) is None:
            raise FileNotFoundError(f"--eval-only needs a checkpoint under {self.checkpoint_dir}")


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════


class HilLiberoTrainer:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.obs_adapter = LiberoObsAdapter(cfg.image_size, include_velocity=cfg.include_velocity)
        # delta_pose is the identity by default (full OSC scale: 0.05 m / 0.5 rad per step).
        # Shrink it so a normalized +-1 means max_step_m / max_step_rad per step.
        scales = (
            *([cfg.max_step_m / _OSC_MAX_M] * 3),
            *([cfg.max_step_rad / _OSC_MAX_RAD] * 3),
            1.0,
        )
        self.act_adapter = ActionAdapter(kind=cfg.action_kind, scales=tuple(float(x) for x in scales))
        self.operator = GamepadOperator(cfg.takeover_step, handback_s=cfg.handback_s)
        print(
            f"[action] normalized +-1 -> {cfg.max_step_m} m / {cfg.max_step_rad} rad per step "
            f"(adapter scales {self.act_adapter.scale_vector.round(3).tolist()})"
        )
        self._stream: MJPEGServer | None = None
        if cfg.display:
            self._stream = MJPEGServer(cfg.stream_host, cfg.stream_port, self.operator.apply_pad)
            if self._stream.start():
                # The browser page is the only gamepad source.
                self.operator.enabled = True
            else:
                self._stream = None
        self._rng = jax.random.PRNGKey(cfg.seed)

        env_cfg = LiberoEnvConfig(
            suite=cfg.suite,
            task_id=cfg.task_id,
            total_num_envs=cfg.total_num_envs,
            seed=cfg.seed,
            camera_size=cfg.camera_size,
        )
        self.env_worker = _envs.EnvWorker(
            env_cfg,
            PaligemmaTokenizer(max_len=48),
            action_dim=cfg.action_dim,
            state_norm=None,
            action_norm=None,
        )

        try:
            from libero.libero import benchmark

            task = benchmark.get_benchmark(cfg.suite)().get_task(cfg.task_id)
            print(f"[task] {cfg.suite}[{cfg.task_id}]: {task.language}")
        except Exception as exc:
            print(f"[task] could not resolve language: {exc}")

        self.agent = _encoders.make_encoder_agent(
            "bc" if cfg.algorithm == "hgdagger" else "sac",
            cfg.encoder,
            cfg.seed,
            cfg.action_dim,
            reward_bias=0.0,
            target_entropy=None,
            discount=0.97,
            soft_target_update_rate=0.005,
        )
        _learner.save_run_config(cfg)
        latest = checkpoints.latest_checkpoint(cfg.checkpoint_dir)
        if latest is not None and (cfg.resume or cfg.eval_only):
            state = checkpoints.restore_checkpoint(latest, target=self.agent.state)
            self.agent = self.agent.replace(state=state)
            self._grad_step = int(Path(latest).name.rsplit("_", 1)[-1])
            print(f"[ckpt] restored {latest} (step {self._grad_step})")
        else:
            self._grad_step = 0
            if latest is None:
                _learner.save_agent_checkpoint(cfg, self.agent, 0)

        self.online = (
            _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity) if cfg.algorithm == "rlpd" else None
        )
        self.demo = _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity)
        self._episodes = 0
        self._env_steps = 0
        self._bc_steps = 0
        self._eval_log = Path(cfg.checkpoint_dir) / "eval_log.jsonl"
        self._last_hint = 0.0

    # ── episode loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            if self.cfg.eval_only:
                self.evaluate(self.cfg.eval_episodes, iteration=0, tag="eval_only")
                return
            while self._episodes < self.cfg.max_episodes and self._env_steps < self.cfg.max_env_steps:
                self._run_episode()
                self._episodes += 1
                if self.cfg.eval_interval > 0 and self._episodes % self.cfg.eval_interval == 0:
                    self.evaluate(self.cfg.eval_episodes, iteration=self._episodes, tag=f"ep{self._episodes}")
            _learner.save_agent_checkpoint(self.cfg, self.agent, self._grad_step)
            print(f"[done] episodes={self._episodes} env_steps={self._env_steps} updates={self._grad_step}")
        finally:
            self.close()

    def _run_episode(self) -> None:
        pool = self.env_worker.make_pool(train=True, task=self.cfg.task_id, n_active=self.cfg.total_num_envs)
        obs_enc = [self.obs_adapter.encode(s.obs) for s in pool]
        episodes: list[list[dict]] = [[] for _ in pool]
        manual_success = manual_failure = False
        grip = [-1.0] * len(pool)  # commanded gripper state per env (LIBERO resets open)

        # Drop button presses made while the previous episode was training.
        self.operator.reset()
        self._show(pool[0].obs, "IDLE: press OPTIONS", _WHITE)
        if not (self.cfg.auto_start or not self.operator.enabled):
            print("\n[episode] Press OPTIONS to start.")
            while not self.operator.poll().options:
                time.sleep(0.02)
            print("[episode] recording: OPTIONS = success, SHARE = failure")

        while True:
            event = self.operator.poll()
            if event.options:
                manual_success = True
                break
            if event.share:
                manual_failure = True
                break

            step_start = time.monotonic()
            active, moving, cmd = self.operator.takeover()
            actions_norm = [None] * len(pool)
            intervened = [False] * len(pool)
            penalties = [0.0] * len(pool)
            for j, state in enumerate(pool):
                if state.done:
                    continue
                # The gamepad drives the first env; additional envs are
                # headless rollout workers.
                if active and j == 0:
                    # The human owns the robot: the policy is not queried at all.
                    action = self.operator.human_action(cmd)
                    intervened[j] = moving  # idle hold frames are not demos
                else:
                    action = np.array(
                        self.agent.sample_actions(obs_enc[j], seed=self._next_key(), argmax=False),
                        np.float32,
                    )
                action[6] = _discretize_gripper(float(action[6]))  # {-1 open, 0 hold, +1 close}
                g = float(action[6])
                if g != 0.0 and g != grip[j]:  # toggles the commanded gripper state
                    penalties[j] = self.cfg.grasp_penalty
                if g != 0.0:
                    grip[j] = g
                if self.cfg.algorithm == "hgdagger":
                    action[6] = grip[j]  # BC regresses a persistent open/closed label, not events
                actions_norm[j] = action
                state.env.send_step(self.act_adapter.to_env(action))

            for j, state in enumerate(pool):
                if state.done:
                    continue
                obs, done, info = state.env.recv_step()
                state.obs = obs
                state.steps += 1
                reward = info.get("reward") if info else None
                success = bool(done and (reward is None or reward > 0))
                if success:
                    state.success = True
                    state.done = True
                elif done or state.steps >= self.cfg.max_episode_steps:
                    state.done = True

                next_enc = self.obs_adapter.encode(obs)
                episodes[j].append(
                    {
                        "observations": obs_enc[j],
                        "actions": actions_norm[j],
                        "next_observations": next_enc,
                        "intervened": intervened[j],
                        "grasp_penalty": penalties[j],
                    }
                )
                obs_enc[j] = next_enc

            self._env_steps += len(pool)
            self._show(pool[0].obs, "HUMAN" if active else "POLICY", _ORANGE if active else _GREEN)
            if all(state.done for state in pool):
                break
            time.sleep(max(0.0, 1.0 / max(self.cfg.control_hz, 1e-6) - (time.monotonic() - step_start)))

        if manual_failure:
            success = False
        elif manual_success:
            success = True
        else:
            success = all(state.success for state in pool)
        for j, state in enumerate(pool):
            episode_success = success if (manual_success or manual_failure) else state.success
            self._flush(episodes[j], success=bool(episode_success))

    # ── buffering / training ────────────────────────────────────────────────

    def _flush(self, episode: list[dict], *, success: bool) -> None:
        if not episode:
            return
        transitions = []
        for i, step in enumerate(episode):
            last = i == len(episode) - 1
            transitions.append(
                make_transition(
                    step["observations"],
                    step["actions"],
                    step["next_observations"],
                    reward=1.0 if (success and last) else 0.0,
                    done=last,
                    intervened=step["intervened"],
                    grasp_penalty=step["grasp_penalty"],
                )
            )
        new_online = 0
        if self.online is not None:
            for transition in transitions:
                self.online.insert(transition)
            new_online = len(transitions)
        demo = [transition for transition in transitions if transition["intervened"]]
        for transition in demo:
            self.demo.insert(transition)

        print(
            f"[episode {self._episodes}] success={success} steps={len(transitions)} "
            f"interventions={len(demo)} online={len(self.online) if self.online is not None else 0} "
            f"demo={len(self.demo)}"
        )
        if self.cfg.algorithm == "hgdagger" and len(demo) == 0:
            now = time.monotonic()
            if now - self._last_hint > 5.0:
                print("[hgdagger] no interventions yet — move a stick during rollout to create BC data.")
                self._last_hint = now
        self._train(new_online, len(demo))

    def _train(self, new_online: int, new_demo: int) -> None:
        if self.cfg.algorithm == "rlpd":
            if self.online is None or len(self.online) < self.cfg.training_starts:
                return
            n_updates = min(
                max(1, round(self.cfg.utd * max(new_online, 1))),
                self.cfg.max_updates_per_episode,
            )
            info = {}
            for g in range(n_updates):
                batch = _learner.mixed_batch(self.online, self.demo, self.cfg.batch_size)
                if g % max(self.cfg.cta_ratio, 1) == max(self.cfg.cta_ratio, 1) - 1:
                    self.agent, info = self.agent.update(batch, networks_to_update=_learner.FULL_UPDATE)
                else:
                    self.agent, _ = self.agent.update(batch, networks_to_update=_learner.CRITIC_ONLY)
                self._after_update()
        else:
            if len(self.demo) == 0:
                return
            if self._bc_steps < self.cfg.pretrain_steps:
                n_updates = min(
                    self.cfg.pretrain_steps - self._bc_steps,
                    max(1, round(self.cfg.utd * max(new_demo, 1))),
                )
                phase = "bc_pretrain"
            else:
                n_updates = min(
                    max(1, round(self.cfg.utd * max(new_demo, 1))),
                    self.cfg.max_updates_per_episode,
                )
                phase = "bc"
            info = {}
            for _ in range(n_updates):
                self.agent, info = self.agent.update(_learner.sample_buffer(self.demo, self.cfg.batch_size))
                self._bc_steps += 1
                self._after_update()
            print(f"[{phase}] updates={n_updates} total_bc={self._bc_steps}")

        if info and self._grad_step % self.cfg.log_interval == 0:
            flat = _learner.flatten_info(info)
            print(f"[update {self._grad_step}] " + " ".join(f"{k}={float(v):.4f}" for k, v in sorted(flat.items())))

    def _after_update(self) -> None:
        self._grad_step += 1
        if self.cfg.checkpoint_period > 0 and self._grad_step % self.cfg.checkpoint_period == 0:
            _learner.save_agent_checkpoint(self.cfg, self.agent, self._grad_step)

    # ── eval ────────────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int, *, iteration: int, tag: str) -> dict:
        n_episodes = max(1, int(n_episodes))
        successes: list[bool] = []
        trial_offset = 0
        while len(successes) < n_episodes:
            m = min(self.cfg.total_num_envs, n_episodes - len(successes))
            pool = self.env_worker.make_pool(train=False, task=self.cfg.task_id, n_active=m, trial_offset=trial_offset)
            trial_offset += m
            obs_enc = [self.obs_adapter.encode(state.obs) for state in pool]
            frames: list[list[np.ndarray]] = [[self.obs_adapter.display(state.obs)] for state in pool]
            grip = [-1.0] * len(pool)
            while not all(state.done for state in pool):
                for j, state in enumerate(pool):
                    if state.done:
                        continue
                    action = np.array(
                        self.agent.sample_actions(obs_enc[j], seed=self._next_key(), argmax=True),
                        np.float32,
                    )
                    g = _discretize_gripper(float(action[6]))
                    if g != 0.0:
                        grip[j] = g
                    action[6] = grip[j] if self.cfg.algorithm == "hgdagger" else g
                    state.env.send_step(self.act_adapter.to_env(action))
                for j, state in enumerate(pool):
                    if state.done:
                        continue
                    obs, done, info = state.env.recv_step()
                    state.obs = obs
                    state.steps += 1
                    reward = info.get("reward") if info else None
                    success = bool(done and (reward is None or reward > 0))
                    if success:
                        state.success = True
                        state.done = True
                    elif done or state.steps >= self.cfg.max_episode_steps:
                        state.done = True
                    obs_enc[j] = self.obs_adapter.encode(obs)
                    frames[j].append(self.obs_adapter.display(obs))
                self._show(pool[0].obs, "EVAL (policy, argmax)", _WHITE)
            for j, state in enumerate(pool):
                successes.append(bool(state.success))
                self._save_video(frames[j], tag=f"{tag}_s{j}", success=bool(state.success))
        rate = float(np.mean(successes)) if successes else 0.0
        result = {"tag": tag, "iteration": iteration, "success_rate": rate, "n": len(successes)}
        print(f"[eval {tag}] success={rate:.2f} n={len(successes)}")
        with self._eval_log.open("a") as f:
            f.write(json.dumps(result) + "\n")
        return result

    # ── plumbing ────────────────────────────────────────────────────────────

    def _next_key(self):
        self._rng, key = jax.random.split(self._rng)
        return key

    def _show(self, obs: dict, label: str = "", color: tuple[int, int, int] = _WHITE) -> None:
        if self._stream is not None:
            self._stream.update(self.obs_adapter.display(obs), label, color)

    def _save_video(self, frames: list[np.ndarray], *, tag: str, success: bool) -> None:
        if not frames:
            return
        suffix = "success" if success else "failure"
        path = Path(self.cfg.video_dir) / f"{tag}_{suffix}.mp4"
        try:
            imageio.mimsave(path, frames, fps=max(self.cfg.control_hz, 1.0))
        except Exception as exc:
            print(f"[video] failed to write {path}: {exc}")

    def close(self) -> None:
        self.operator.close()
        self.env_worker.close()
        if self._stream is not None:
            self._stream.close()
