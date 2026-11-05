"""LIBERO actor/learner for RLPD and HG-Dagger.

The actor steps LIBERO in the main thread; the learner runs in the background
and publishes weights every ``steps_per_update`` steps. Human interventions
come from the browser gamepad page.

Record:
    cd /home/serg/projects/openpi-wam
    uv run --extra libero /home/serg/projects/openpi-wam/scripts/train_hil_libero.py \
        --record-demos 20 --bottleneck-label --suite libero_goal --task-id 1 \
        --demo-path /home/serg/projects/openpi-wam/demos/dmp_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_demos \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/goal_t1_demos

Train:
    cd /home/serg/projects/openpi-wam
    uv run --extra libero /home/serg/projects/openpi-wam/scripts/train_hil_libero.py \
        --algorithm rlpd --suite libero_goal --task-id 1 --base-policy dmp \
        --demo-path /home/serg/projects/openpi-wam/demos/dmp_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd_dmp_bottleneck_lagrange \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1_dmp_bottleneck_lagrange

Controls: OPTIONS/SHARE mark success/failure, D-pad up/down selects a
bottleneck, D-pad left/right marks sections while recording and skips eval
otherwise, L1/R1 skip eval, PS hands control back to the policy.
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
from openpi.training.hil.dmp import DmpPolicy
from openpi.training.hil.dmp import combine_base_residual
from openpi.training.hil.dmp import residual_target
from openpi.training.hil.transitions import episode_stats
from openpi.training.hil.transitions import load_demo
from openpi.training.hil.transitions import make_transition
from openpi.training.hil.transitions import save_demo
from openpi.training.hil.transitions import save_transitions
from openpi.training.rlt import envs as _envs

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    # "libero_goal": 30000,
    "libero_10": 520,
    "libero_90": 400,
}

# Full-scale per-step output of robosuite's default OSC_POSE controller.
_OSC_MAX_M = 0.05
_OSC_MAX_RAD = 0.5
_WHITE, _GREEN, _ORANGE = (255, 255, 255), (80, 255, 80), (255, 160, 0)


def _discretize_gripper(g: float) -> float:
    """Snap a gripper command to {-1 open, 0 hold, +1 close} (hil-serl's thresholds)."""
    return 0.0 if abs(g) < 0.5 else (1.0 if g > 0 else -1.0)


# ---------------------------------------------------------------------------
# LIBERO observation
# ---------------------------------------------------------------------------


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

    When a base policy is active, ``base_action_dim`` extra values are appended
    to the proprio vector.  They are the normalized delta-pose action proposed
    by the DMP at this step, so the RL agent can learn a residual on top of it.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        *,
        include_velocity: bool = False,
        base_action_dim: int = 0,
    ):
        self.image_size = tuple(image_size)
        self.include_velocity = include_velocity
        self.base_action_dim = int(base_action_dim)

    @property
    def proprio_dim(self) -> int:
        return 8 + (9 if self.include_velocity else 0) + self.base_action_dim

    def _image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
        image = image_tools.resize_with_pad(image, self.image_size[0], self.image_size[1])
        return image_tools.convert_to_uint8(image)

    def state(self, obs: dict, base_action: np.ndarray | None = None) -> np.ndarray:
        parts = [
            np.asarray(obs["robot0_eef_pos"], np.float32),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], np.float32),
        ]
        if self.include_velocity:
            parts.append(np.asarray(obs["robot0_joint_vel"], np.float32))
            parts.append(np.asarray(obs["robot0_gripper_qvel"], np.float32))
        if self.base_action_dim:
            if base_action is None:
                base_action = np.zeros(self.base_action_dim, np.float32)
            base_action = np.asarray(base_action, np.float32).reshape(-1)
            if base_action.shape[0] != self.base_action_dim:
                raise ValueError(f"base_action must have {self.base_action_dim} entries, got {base_action.shape[0]}")
            parts.append(base_action)
        return np.concatenate(parts).astype(np.float32)

    def pose(self, obs: dict, *, rotation: bool) -> np.ndarray:
        """End-effector pose used by the DMP: xyz [+ axis-angle rotation]."""
        state = self.state(obs)
        return state[:6] if rotation else state[:3]

    def encode(self, obs: dict, base_action: np.ndarray | None = None) -> dict:
        front = self._image(obs["agentview_image"])
        wrist = self._image(obs["robot0_eye_in_hand_image"])
        # (T=1, H, W, C) and (T=1, D): the agents stack a time axis even when T=1.
        return {
            "front_image": front[None],
            "wrist_image": wrist[None],
            "state": self.state(obs, base_action)[None],
        }

    def display(self, obs: dict) -> np.ndarray:
        front = image_tools.convert_to_uint8(np.asarray(obs["agentview_image"])[::-1, ::-1])
        wrist = image_tools.convert_to_uint8(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])
        return np.concatenate([front, wrist], axis=1)


# ---------------------------------------------------------------------------
# Remote display + gamepad
# ---------------------------------------------------------------------------

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
 #status { color:#8cf; margin-bottom:4px; }
 #grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:0 24px; }
 b { color:#fff; }
 .dim { color:#999; }
</style></head>
<body>
<div id="view"><img src="/stream.mjpg" alt="LIBERO stream"></div>
<div id="help">
 <div id="pad" class="bad">controller: none - focus this tab and press any button on the pad</div>
 <div id="status">task: loading... | online episodes: 0 | demo episodes: 0 | collected episodes: 0 | training: 00:00:00 | bottleneck: off</div>
 <div id="grid">
  <div><b>OPTIONS</b> start episode / mark <b>SUCCESS</b></div>
  <div><b>SHARE</b> mark <b>FAILURE</b></div>
  <div><b>Left stick</b> x (up/down), y (left/right)</div>
  <div><b>Right stick</b> pitch / yaw <span class="dim">(only with --rotation)</span></div>
  <div><b>&#9651; / &#10005;</b> z up / z down</div>
  <div><b>&#9633; / &#9675;</b> roll left / right <span class="dim">(only with --rotation)</span></div>
  <div><b>L2 / R2</b> gripper open / close (release = hold)</div>
  <div><b>D-pad &#9664; / &#9654;</b> bottleneck start / end <span class="dim">(recording with --bottleneck-label)</span> or skip eval <span class="dim">(otherwise)</span></div>
  <div><b>D-pad &#9650; / &#9660;</b> select bottleneck <span class="dim">(training B1..BN; eval B0..BN)</span></div>
  <div><b>L1 / R1</b> skip current eval episode</div>
  <div><b>PS</b> hand control back to the policy</div>
  <div class="dim">first stick/trigger/button input = human takeover; the policy stays off until PS</div>
 </div>
</div>
<script>
const NBTN = 18;
let busy = false, padId = "", padBad = false;
function neutral() { return {axes:[0,0,0,0], buttons:Array(NBTN).fill(false)}; }
function readPad() {
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
      ? "controller: none - press any button on the pad (Chrome only shows it after a press)"
      : "controller: tab not focused - click this page";
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
async function pollStatus() {
  try {
    const s = await (await fetch("/status")).json();
    document.getElementById("status").textContent =
      "task: " + s.task + " | online episodes: " + s.online_episodes + " | demo episodes: " + s.demo_episodes +
      " | collected episodes: " + s.collected_episodes +
      " | training: " + s.training + " | " + s.mode + " | learner step " + s.learner_step +
      " | bottleneck: " + s.bottleneck;
  } catch (e) {}
}
setInterval(tick, 33);
setInterval(pollStatus, 1000);
</script>
</body></html>
"""


class MJPEGServer:
    """Serve the latest frame as MJPEG, accept gamepad state, and expose status."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        on_pad: Callable[[dict], None] | None = None,
        on_status: Callable[[], dict] | None = None,
    ):
        self.host = host
        self.port = port
        self.on_pad = on_pad
        self.on_status = on_status
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
                elif self.path == "/status":
                    body = json.dumps(server.status()).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
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

    def status(self) -> dict:
        return self.on_status() if self.on_status is not None else {}

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
            cv2.putText(bgr, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color[::-1], 1, cv2.LINE_AA)
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


@dataclasses.dataclass
class PadEvent:
    options: bool = False  # OPTIONS: start (idle) / mark success (recording or bottleneck training)
    share: bool = False  # SHARE: mark failure
    skip: bool = False  # L1/R1 (and D-pad left/right outside bottleneck-label mode): skip eval
    bottleneck_start: bool = False  # D-pad left: start a bottleneck section while recording
    bottleneck_end: bool = False  # D-pad right: end a bottleneck section while recording
    prev_bottleneck: bool = False  # D-pad up: previous bottleneck state (B0 = full task)
    next_bottleneck: bool = False  # D-pad down: next bottleneck state


class GamepadOperator:
    """PS4 controller state, fed by the browser page via ``apply_pad``.

    Intervention is latched: the first stick/trigger/button input takes control
    and the policy stays off until PS is pressed (or ``handback_s`` of idle input
    elapses, if > 0) or ``reset()`` is called at episode start.
    """

    CROSS, CIRCLE, SQUARE, TRIANGLE = 0, 1, 2, 3
    L1, R1 = 4, 5
    L2, R2 = 6, 7
    SHARE, OPTIONS = 8, 9
    DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = 12, 13, 14, 15
    PS = 16
    N_BUTTONS = 18

    def __init__(
        self,
        takeover_step: float = 0.5,
        deadzone: float = 0.15,
        stale_s: float = 0.3,
        handback_s: float = 0.0,
        *,
        rotation: bool = True,
        bottleneck_label: bool = False,
    ):
        self.rotation = bool(rotation)
        self.bottleneck_label = bool(bottleneck_label)
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
        self._release = False
        self._wait_idle = False
        self._last_input = float("-inf")

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
            if rising[self.DPAD_LEFT]:
                self._event.bottleneck_start = self.bottleneck_label
                self._event.skip = not self.bottleneck_label
            if rising[self.DPAD_RIGHT]:
                self._event.bottleneck_end = self.bottleneck_label
                self._event.skip = not self.bottleneck_label
            if rising[[self.L1, self.R1]].any():
                self._event.skip = True
            if rising[self.DPAD_UP]:
                self._event.prev_bottleneck = True
            if rising[self.DPAD_DOWN]:
                self._event.next_bottleneck = True
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

    def takeover(self) -> tuple[bool, np.ndarray]:
        """``(intervened, cmd)``.

        ``intervened``: the human owns the robot from the first input until
        handback (PS release or ``handback_s`` without input).
        ``cmd``: ``[x, y, z, roll, pitch, yaw, gripper]``.
        """
        zeros = np.zeros(7, np.float32)
        now = time.monotonic()
        with self._lock:
            if now - self._last_msg >= self.stale_s:
                return self._latched, zeros
            lx, ly, rx, ry = self._shape(self._axes.copy())
            b = self._buttons.copy()
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
            if not self.rotation:
                cmd[3:6] = 0.0
            grip = float(b[self.R2]) - float(b[self.L2])  # R2 close, L2 open, neither/both = hold
            moving = bool(np.any(cmd != 0.0) or grip != 0.0)

            if self._release:
                self._release = False
                if self._latched:
                    self._latched = False
                    self._wait_idle = True
            if self._wait_idle:
                if moving:
                    return False, zeros  # still holding input after hand-back: policy runs
                self._wait_idle = False
            if moving:
                self._latched = True
                self._last_input = now
            elif self._latched and self.handback_s > 0 and now - self._last_input > self.handback_s:
                self._latched = False

            full = np.concatenate([cmd * self.takeover_step, [grip]]).astype(np.float32)
            return self._latched, full


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LiberoEnvConfig:
    """The subset of ``rlt.envs.EnvWorker``'s config that the pool reads."""

    suite: str = "libero_goal"
    task_id: int = 0
    seed: int = 42
    camera_size: int = 128
    total_num_envs: int = 1
    action_chunk: int = 1
    action_env_dim: int = 7
    group_size: int = 1
    # Start-state perturbations (LIBERO init-state access).  EEF gets +-xyz;
    # objects get +-xy only so their z stays on the table.  All movable objects
    # are attempted except non-primary ones inside a container; primary
    # obj_of_interest objects are always perturbed.
    perturb_objects: bool = True
    perturb_eef_xyz_m: float = 0.02
    perturb_object_xy_m: float = 0.02
    perturb_all_objects: bool = True
    perturb_after_success_rate: float = 0.0

    @property
    def task_ids(self) -> list[int]:
        return [self.task_id]


@dataclasses.dataclass
class SimConfig:
    """CLI config for the LIBERO RLPD/HG-Dagger loop."""

    algorithm: Literal["rlpd", "hgdagger"] = "rlpd"
    target_intervention_rate: float = 0.05
    """Constrain human-takeover steps to this fraction of episode steps."""
    intervention_window: int = 20
    """Number of recent episodes used for the moving-average rate."""
    suite: str = "libero_goal"
    task_id: int = 0
    seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    include_velocity: bool = False
    rotation: bool = False

    max_episodes: int = 2000
    max_episode_steps: int = 0  # 0 -> the suite's standard limit
    max_env_steps: int = 0  # 0 -> max_episodes * max_episode_steps
    control_hz: float = 20.0
    auto_start: bool = False

    record_demos: int = 0
    """Pure teleop: save N successful demos to the first --demo-path and exit."""
    demo_path: tuple[str, ...] = ()
    """Demo source for training; the first entry receives --record-demos."""
    bottleneck_label: bool = False
    """During teleop, D-pad left/right marks bottleneck start/end.

    The full simulator state and gripper command are saved at each section
    start.  A demo with no marks is treated as one full-task section.
    """

    base_policy: Literal["none", "dmp"] = "none"
    dmp_weights_per_dim: int = 20
    dmp_regularization: float = 0.0
    dmp_gripper: bool = True
    """Model the demonstrated gripper command as an extra DMP dimension."""

    residual_max: float = 1.0
    """Maximum residual magnitude used by the base/residual blend."""

    # Start-state perturbations from the LIBERO init states.  The EEF start is
    # perturbed +-xyz; movable objects are perturbed +-xy only, keeping z fixed
    # so they stay on the table.  All movable objects are attempted except
    # non-primary ones inside a container; primary ``obj_of_interest`` objects
    # (the bowl in task 1) are always perturbed.
    perturb_objects: bool = True
    perturb_eef_xyz_m: float = 0.02
    perturb_object_xy_m: float = 0.02
    perturb_all_objects: bool = True
    perturb_after_success_rate: float = 0.0
    """0 = perturb from the first episode.  >0 = start unperturbed and enable
    perturbations after an eval reports a success rate above this threshold
    (the >50% curriculum switch)."""

    checkpoint_dir: str = "checkpoints/hil/libero"
    keep: int = 20
    checkpoint_period: int = 1_000
    batch_size: int = 256
    cta_ratio: int = 5
    discount: float = 0.985
    steps_per_update: int = 50
    training_starts: int = 20
    pretrain_steps: int = 2_000
    replay_buffer_capacity: int = 100_000
    log_interval: int = 100

    eval_after_episodes: int = 50
    eval_interval: int = 10
    eval_episodes: int = 5
    eval_only: bool = False
    video_dir: str = "videos/hil_libero"

    display: bool = True
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    takeover_step: float = 0.5
    handback_s: float = 0.0
    grasp_penalty: float = -0.02
    max_step_m: float = 0.02
    max_step_rad: float = 0.05

    def __post_init__(self):
        if self.suite not in _MAX_STEPS:
            raise ValueError(f"unknown LIBERO suite {self.suite!r}; expected one of {tuple(_MAX_STEPS)}")
        if not 0.0 <= self.target_intervention_rate <= 1.0:
            raise ValueError(
                "target_intervention_rate must be in [0, 1], "
                f"got {self.target_intervention_rate}"
            )
        if self.intervention_window <= 0:
            raise ValueError(
                f"intervention_window must be positive, got {self.intervention_window}"
            )
        self.image_size = tuple(self.image_size)
        self.demo_path = tuple(self.demo_path)
        self.max_episode_steps = int(self.max_episode_steps) or _MAX_STEPS[self.suite]
        self.max_env_steps = int(self.max_env_steps) or self.max_episodes * self.max_episode_steps
        if not (0.0 < self.max_step_m <= _OSC_MAX_M and 0.0 < self.max_step_rad <= _OSC_MAX_RAD):
            raise ValueError(
                f"max_step_m must be in (0, {_OSC_MAX_M}] and max_step_rad in (0, {_OSC_MAX_RAD}] "
                "(the OSC controller's full scale)"
            )
        if self.base_policy == "dmp" and self.record_demos == 0 and not self.demo_path:
            raise ValueError("--base-policy dmp needs --demo-path")
        if self.residual_max <= 0.0:
            raise ValueError("residual_max must be positive")
        if self.record_demos > 0 and not self.display:
            raise ValueError("--record-demos needs --display")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)
        if self.eval_only and checkpoints.latest_checkpoint(self.checkpoint_dir) is None:
            raise FileNotFoundError(f"--eval-only needs a checkpoint under {self.checkpoint_dir}")
        self.encoder = _encoders.EncoderConfig(
            kind="hilserl",
            proprio_dim=self.proprio_dim,
            base_action_dim=self.base_action_dim,
            image_size=self.image_size,
        )

    @property
    def base_action_dim(self) -> int:
        """Width of the DMP action appended to proprio (0 when disabled)."""
        return self.action_dim if self.base_policy == "dmp" else 0

    @property
    def proprio_dim(self) -> int:
        return 8 + (9 if self.include_velocity else 0) + self.base_action_dim

    @property
    def action_dim(self) -> int:
        return 7 if self.rotation else 4

    @property
    def demo_output_dir(self) -> Path:
        """Directory used by ``--record-demos`` (first ``--demo-path`` entry)."""
        return Path(self.demo_path[0]) if self.demo_path else Path("demos/hil_libero")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Bottleneck:
    """A contiguous slice of the first demo, in transition indices."""

    start: int
    end: int
    start_state: np.ndarray
    start_grip: float

    @property
    def n_steps(self) -> int:
        return self.end - self.start


@dataclasses.dataclass
class EpisodeResult:
    transitions: list[dict]
    success: bool
    frames: list[np.ndarray]
    skipped: bool = False


class HilLiberoTrainer:
    """Actor in the main thread, learner in a background thread."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.teleop_only = cfg.record_demos > 0

        self.bottlenecks: list[Bottleneck] = []
        self.bottleneck_mode = False
        self.train_selected: int | None = None
        self.eval_selected: int | None = None
        self._episode_train_idx: int | None = None
        self._episode_eval_idx: int | None = None
        self._episode_is_train = True
        self._active_section_idx: int | None = None
        self._recording_sections: list[dict] = []
        self._recording_open: dict | None = None

        self.full_dmp: DmpPolicy | None = None
        self.section_dmps: list[DmpPolicy] = []
        if not self.teleop_only and cfg.base_policy == "dmp":
            self._load_bottleneck_specs()
        elif cfg.base_policy == "dmp":
            print("[dmp] --record-demos is pure teleop; --base-policy is ignored while recording.")

        base_action_dim = 0 if self.teleop_only else cfg.base_action_dim
        self.obs_adapter = LiberoObsAdapter(
            cfg.image_size,
            include_velocity=cfg.include_velocity,
            base_action_dim=base_action_dim,
        )
        scales = (
            *([cfg.max_step_m / _OSC_MAX_M] * 3),
            *([cfg.max_step_rad / _OSC_MAX_RAD] * 3),
            1.0,
        )
        self.act_adapter = ActionAdapter(kind="delta_pose", scales=tuple(float(x) for x in scales))
        self.operator = GamepadOperator(
            cfg.takeover_step,
            handback_s=cfg.handback_s,
            rotation=cfg.rotation,
            bottleneck_label=cfg.bottleneck_label and self.teleop_only,
        )
        base_note = ""
        if cfg.base_policy == "dmp" and not self.teleop_only:
            base_note = (
                f" | base policy: DMP ({base_action_dim}-dim base action appended to proprio, "
                f"gripper={'modeled' if cfg.dmp_gripper else 'residual-only'})"
            )
            if self.bottleneck_mode:
                base_note += f" | bottleneck mode: {len(self.bottlenecks)} section(s)"
        print(
            f"[action] {cfg.action_dim}-dim agent action ({'6-DoF' if cfg.rotation else 'xyz'} + gripper); "
            f"normalized +-1 -> {cfg.max_step_m} m / {cfg.max_step_rad} rad per step{base_note}"
        )

        self.agent = self.actor_agent = None
        self.online = self.demo = None
        self._hud = ""
        self._teleop_base_hud = ""
        self._mode = "IDLE"
        self._grad_step = 0
        self._updates = 0
        self._episodes = 0
        self._collected_episodes = 0
        self._env_steps = 0
        self._online_episodes = 0
        self._demo_episodes = 0
        self._training_seconds = 0.0
        self._session_start: float | None = None
        self._demo_files_seen: set[str] = set()
        self.intervention_tracker = _learner.InterventionRateTracker(
            cfg.intervention_window
        )
        self._stop = threading.Event()
        self._learner_thread: threading.Thread | None = None
        self._learner_error: BaseException | None = None
        self._eval_log = Path(cfg.checkpoint_dir) / "eval_log.jsonl"
        self._last_hint = 0.0
        self._stats_path = Path(cfg.checkpoint_dir) / "training_stats.json"
        self._load_stats()
        # The browser status endpoint can be polled before the env worker has
        # resolved the task language; keep a safe placeholder until then.
        self.task_name = "loading..."

        self._stream: MJPEGServer | None = None
        if cfg.display:
            self._stream = MJPEGServer(
                cfg.stream_host,
                cfg.stream_port,
                on_pad=self.operator.apply_pad,
                on_status=self._status,
            )
            if not self._stream.start():
                self._stream = None

        self._rng = jax.random.PRNGKey(cfg.seed)
        env_cfg = LiberoEnvConfig(
            suite=cfg.suite,
            task_id=cfg.task_id,
            seed=cfg.seed,
            camera_size=cfg.image_size[0],
            perturb_objects=cfg.perturb_objects,
            perturb_eef_xyz_m=cfg.perturb_eef_xyz_m,
            perturb_object_xy_m=cfg.perturb_object_xy_m,
            perturb_all_objects=cfg.perturb_all_objects,
            perturb_after_success_rate=cfg.perturb_after_success_rate,
        )
        self.env_worker = _envs.EnvWorker(
            env_cfg,
            PaligemmaTokenizer(max_len=48),
            action_dim=7,
            state_norm=None,
            action_norm=None,
        )
        self.task_name = self.env_worker.task_language(cfg.task_id)
        print(f"[task] {cfg.suite} task {cfg.task_id}: {self.task_name}")
        if self.teleop_only:
            return

        self._check_checkpoint_config()
        self.agent = _encoders.make_encoder_agent(
            "bc" if cfg.algorithm == "hgdagger" else "sac",
            cfg.encoder,
            cfg.seed,
            cfg.action_dim,
            reward_bias=0.0,
            target_entropy=None,
            discount=cfg.discount,
            soft_target_update_rate=0.005,
            target_intervention_rate=cfg.target_intervention_rate,
        )
        _learner.save_run_config(cfg)
        latest = checkpoints.latest_checkpoint(cfg.checkpoint_dir)
        if latest is not None:
            state = checkpoints.restore_checkpoint(latest, target=self.agent.state)
            self.agent = self.agent.replace(state=state)
            self._grad_step = int(Path(latest).name.rsplit("_", 1)[-1])
            print(f"[ckpt] restored {latest} (learner step {self._grad_step})")
        else:
            _learner.save_agent_checkpoint(cfg, self.agent, 0)
        self.actor_agent = self.agent

        self.online = (
            _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim)
            if cfg.algorithm == "rlpd"
            else None
        )
        self.demo = _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim)
        self._load_demos()
        if cfg.algorithm == "rlpd" and len(self.demo) == 0 and not cfg.eval_only:
            print("[demo] no demos loaded; RLPD will train online-only. Pass --demo-path to use demos.")

    # -- stats -------------------------------------------------------------

    def _load_stats(self) -> None:
        try:
            stats = json.loads(self._stats_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            stats = {}
        self._online_episodes = int(stats.get("online_episodes", 0))
        self._demo_episodes = int(stats.get("demo_episodes", 0))
        self._collected_episodes = int(stats.get("collected_episodes", 0))
        self._training_seconds = float(stats.get("training_seconds", 0.0))
        self._demo_files_seen = set(stats.get("demo_files", ()))
        self._intervention_steps = int(stats.get("intervention_steps", 0))

    def _elapsed_seconds(self) -> float:
        if self._session_start is None:
            return self._training_seconds
        return self._training_seconds + time.monotonic() - self._session_start

    def _elapsed(self) -> str:
        seconds = int(self._elapsed_seconds())
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"

    def _save_stats(self) -> None:
        if self.teleop_only:
            return
        payload = {
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "intervention_steps": self._intervention_steps,
            "target_intervention_rate": self.cfg.target_intervention_rate,
            "intervention_rate": self.intervention_tracker.rate,
            "intervention_episodes": self.intervention_tracker.num_episodes,
            "training_seconds": round(self._elapsed_seconds(), 1),
            "demo_files": sorted(self._demo_files_seen),
        }
        self._stats_path.write_text(json.dumps(payload, indent=2))

    def _bottleneck_status(self) -> str:
        if self.teleop_only:
            if not self.cfg.bottleneck_label:
                return "off"
            n = len(self._recording_sections)
            if self._recording_open is not None:
                return f"recording: {n} marked, B{n + 1} open"
            return f"recording: {n} marked section(s)"
        if not self.bottleneck_mode:
            return "off"
        selected = self.train_selected if self._episode_is_train else self.eval_selected
        if self._mode == "IDLE":
            return self._selection_label(selected)
        idx = self._episode_train_idx if self._episode_is_train else self._episode_eval_idx
        current = self._selection_label(idx)
        if selected != idx:
            current += f" | next: {self._selection_label(selected)}"
        return current

    def _selection_label(self, idx: int | None) -> str:
        if idx is None:
            return "B0 full task"
        return f"B{idx + 1}/{len(self.bottlenecks)}"

    def _status(self) -> dict:
        return {
            "task": self.task_name,
            "target_intervention_rate": self.cfg.target_intervention_rate,
            "intervention_rate": round(self.intervention_tracker.rate, 4),
            "intervention_episodes": self.intervention_tracker.num_episodes,
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "intervention_steps": self._intervention_steps,
            "training": self._elapsed(),
            "mode": self._mode,
            "learner_step": self._grad_step,
            "bottleneck": self._bottleneck_status(),
        }

    # -- base policy -------------------------------------------------------

    def _check_checkpoint_config(self) -> None:
        """Refuse to restore a checkpoint trained with a different configuration."""
        config_path = Path(self.cfg.checkpoint_dir) / "config.json"
        if not config_path.exists():
            return
        old = json.loads(config_path.read_text())
        old_encoder = old.get("encoder", {})
        old_kind = old_encoder.get("kind")
        old_proprio = old_encoder.get("proprio_dim")
        old_base_dim = old_encoder.get("base_action_dim")
        old_action = old.get("action_dim")
        old_base = old.get("base_policy", "none")
        old_gripper = old.get("dmp_gripper")
        old_residual_max = old.get("residual_max")
        old_target = old.get("target_intervention_rate")
        # RLPD checkpoints predate the intervention-cost networks and cannot be
        # restored. A different target rate is fine: the multiplier adapts.
        # HG-Dagger uses the unchanged BC agent, so it can still restore.
        predates_intervention = old_target is None and self.cfg.algorithm == "rlpd"
        if (
            (old_kind is not None and old_kind != self.cfg.encoder.kind)
            or (old_proprio is not None and int(old_proprio) != self.cfg.encoder.proprio_dim)
            or (old_base_dim is not None and int(old_base_dim) != self.cfg.encoder.base_action_dim)
            or (old_action is not None and int(old_action) != self.cfg.action_dim)
            or old_base != self.cfg.base_policy
            or (old_gripper is not None and bool(old_gripper) != self.cfg.dmp_gripper)
            or (old_residual_max is not None and abs(float(old_residual_max) - self.cfg.residual_max) > 1e-9)
            or predates_intervention
        ):
            raise ValueError(
                f"{config_path} was trained with a different observation/action/intervention configuration "
                "(or predates the intervention-cost networks); use a fresh --checkpoint-dir"
            )

    def _pose_from_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, np.float32).reshape(-1)
        return state[:6] if self.cfg.rotation else state[:3]

    def _pose(self, obs: dict) -> np.ndarray:
        return self._pose_from_state(self.obs_adapter.state(obs))

    def _pose_with_grip(self, obs: dict, grip: float) -> np.ndarray:
        pose = self._pose(obs)
        if self.cfg.dmp_gripper:
            pose = np.concatenate([pose, np.asarray([grip], np.float32)])
        return pose

    def _trajectory_for_slice(
        self,
        transitions: list[dict],
        start: int,
        end: int,
        start_grip: float,
    ) -> np.ndarray:
        actions = [np.asarray(t["actions"], np.float32) for t in transitions[start:end]]
        states = [np.asarray(t["observations"]["state"]) for t in transitions[start:end]]
        boundary = transitions[end]["observations"]["state"] if end < len(transitions) else transitions[-1]["next_observations"]["state"]
        states.append(boundary)
        poses = np.stack([self._pose_from_state(state) for state in states])
        if self.cfg.dmp_gripper:
            grips = np.concatenate([[float(start_grip)], [float(action[-1]) for action in actions]])
            poses = np.column_stack([poses, grips])
        return poses

    def _make_dmp(
        self,
        transitions: list[dict],
        start: int,
        end: int,
        start_grip: float,
    ) -> DmpPolicy:
        return DmpPolicy(
            self._trajectory_for_slice(transitions, start, end, start_grip),
            dt=1.0 / self.cfg.control_hz,
            rotation=self.cfg.rotation,
            gripper=self.cfg.dmp_gripper,
            action_dim=self.cfg.action_dim,
            max_step_m=self.cfg.max_step_m,
            max_step_rad=self.cfg.max_step_rad,
            n_weights_per_dim=self.cfg.dmp_weights_per_dim,
            regularization_coefficient=self.cfg.dmp_regularization,
        )

    def _validate_demo_dims(self, transitions: list[dict], source: str) -> None:
        dim = int(np.asarray(transitions[0]["actions"]).shape[-1])
        if dim != self.cfg.action_dim:
            raise ValueError(f"{source}: demo actions are {dim}-dim, expected {self.cfg.action_dim}")
        expected_state_dim = 8 + (9 if self.cfg.include_velocity else 0)
        demo_state_dim = int(np.asarray(transitions[0]["observations"]["state"]).shape[-1])
        if demo_state_dim != expected_state_dim:
            raise ValueError(f"{source}: demo states are {demo_state_dim}-dim, expected {expected_state_dim}")

    def _load_bottleneck_specs(self) -> None:
        files = self._demo_files()
        if not files:
            raise FileNotFoundError("--base-policy dmp needs at least one demo pickle")
        source = files[0]
        for path in files:
            if "bottlenecks" in load_demo(path).metadata:
                source = path
                break
        demo = load_demo(source)
        transitions = demo.transitions
        if not transitions:
            raise ValueError(f"{source}: demo is empty")
        self._validate_demo_dims(transitions, str(source))

        meta_task = demo.metadata.get("task_id")
        if meta_task is not None and int(meta_task) != self.cfg.task_id:
            raise ValueError(f"{source}: recorded task {meta_task}, training task {self.cfg.task_id}")
        meta_suite = demo.metadata.get("suite")
        if meta_suite is not None and meta_suite != self.cfg.suite:
            raise ValueError(f"{source}: recorded suite {meta_suite}, training suite {self.cfg.suite}")
        meta_rotation = demo.metadata.get("rotation")
        if meta_rotation is not None and bool(meta_rotation) != self.cfg.rotation:
            raise ValueError(f"{source}: recorded rotation={meta_rotation}, training rotation={self.cfg.rotation}")

        self.bottleneck_mode = "bottlenecks" in demo.metadata
        self.bottlenecks = []
        for i, section in enumerate(demo.bottlenecks):
            start = int(section["start"])
            end = int(section["end"])
            start_state = section.get("start_state")
            if not (0 <= start < end <= len(transitions)) or end - start < 2 or start_state is None:
                raise ValueError(f"{source}: invalid bottleneck section {i}: [{start}, {end})")
            self.bottlenecks.append(
                Bottleneck(
                    start=start,
                    end=end,
                    start_state=np.asarray(start_state, np.float64).reshape(-1),
                    start_grip=float(section.get("start_grip", -1.0)),
                )
            )

        self.full_dmp = self._make_dmp(transitions, 0, len(transitions), -1.0)
        self.section_dmps = [
            self._make_dmp(transitions, bottleneck.start, bottleneck.end, bottleneck.start_grip)
            for bottleneck in self.bottlenecks
        ]
        self.train_selected = 0 if self.bottlenecks else None
        self.eval_selected = None
        if self.bottleneck_mode and not self.bottlenecks:
            print("[bottleneck] no marked sections; the full task is the bottleneck window")
        print(
            f"[dmp] fit full-task DMP to {len(transitions)} demo steps and "
            f"{len(self.bottlenecks)} section DMP(s) from {source}"
        )

    def _window_at_step(self, step: int) -> int | None:
        for idx, bottleneck in enumerate(self.bottlenecks):
            if bottleneck.start <= step < bottleneck.end:
                return idx
        return None

    def _base_action_for_step(
        self,
        obs: dict,
        grip: float,
        bottleneck_idx: int | None,
        step_idx: int,
    ) -> tuple[np.ndarray | None, bool]:
        if self.full_dmp is None:
            return None, True
        pose = self._pose_with_grip(obs, grip)
        if bottleneck_idx is not None:
            return self.section_dmps[bottleneck_idx].action(pose), True

        full_action = self.full_dmp.action(pose)
        if not self.bottlenecks:
            return full_action, True

        window = self._window_at_step(step_idx)
        if window is None:
            self._active_section_idx = None
            return full_action, False
        if self._active_section_idx != window:
            self.section_dmps[window].reset(pose)
            self._active_section_idx = window
        return self.section_dmps[window].action(pose), True

    def _reset_episode(
        self,
        bottleneck_idx: int | None,
        *,
        train: bool,
        trial_offset: int = 0,
    ) -> tuple[object, float, np.ndarray | None, bool, dict]:
        if bottleneck_idx is None:
            init_state = None
            grip = -1.0
        else:
            bottleneck = self.bottlenecks[bottleneck_idx]
            init_state = bottleneck.start_state
            grip = bottleneck.start_grip

        state = self.env_worker.make_pool(
            train=train,
            task=self.cfg.task_id,
            trial_offset=trial_offset,
            n_active=1,
            init_state_override=init_state,
            reset_grip=grip,
        )[0]
        if self.full_dmp is not None:
            pose = self._pose_with_grip(state.obs, grip)
            if bottleneck_idx is not None:
                self.section_dmps[bottleneck_idx].reset(pose)
            else:
                self.full_dmp.reset(pose)
                self._active_section_idx = None
        base_action, residual_active = self._base_action_for_step(state.obs, grip, bottleneck_idx, 0)
        obs_enc = self.obs_adapter.encode(state.obs, base_action=base_action)
        return state, grip, base_action, residual_active, obs_enc

    def _append_base_action(self, obs: dict, base_action: np.ndarray | None) -> dict:
        if base_action is None:
            return obs
        obs = {**obs}
        state = np.asarray(obs["state"], np.float32)
        base = np.asarray(base_action, np.float32).reshape(-1)
        base = np.broadcast_to(base, (*state.shape[:-1], base.size))
        obs["state"] = np.concatenate([state, base], axis=-1)
        return obs

    def _prepare_segment(
        self,
        transitions: list[dict],
        policy: DmpPolicy,
        start_grip: float,
    ) -> list[dict]:
        if not transitions:
            return []
        transitions = self._finalize_demo_actions(transitions, start_grip=start_grip)
        actions = [np.asarray(t["actions"], np.float32) for t in transitions]
        states = [np.asarray(t["observations"]["state"]) for t in transitions]
        states.append(np.asarray(transitions[-1]["next_observations"]["state"]))
        poses = np.stack([self._pose_from_state(state) for state in states])
        if self.cfg.dmp_gripper:
            grips = np.concatenate([[float(start_grip)], [float(action[-1]) for action in actions]])
            poses = np.column_stack([poses, grips])
        policy.reset(poses[0])
        base_actions = np.stack([policy.action(pose) for pose in poses])

        prepared = []
        for i, transition in enumerate(transitions):
            action = np.asarray(transition["actions"], np.float32)
            prepared.append(
                {
                    **transition,
                    "observations": self._append_base_action(transition["observations"], base_actions[i]),
                    "next_observations": self._append_base_action(transition["next_observations"], base_actions[i + 1]),
                    "actions": residual_target(action, base_actions[i]),
                }
            )
        return self._mark_terminal_success(prepared)

    def _prepare_demo(self, transitions: list[dict]) -> list[dict]:
        return self._finalize_demo_actions(transitions)

    def _mark_terminal_success(self, transitions: list[dict]) -> list[dict]:
        if transitions:
            transitions[-1] = {
                **transitions[-1],
                "rewards": np.float32(1.0),
                "masks": np.float32(0.0),
                "dones": True,
            }
        return transitions

    def _finalize_demo_actions(self, transitions: list[dict], start_grip: float = -1.0) -> list[dict]:
        if self.cfg.algorithm != "hgdagger":
            return transitions
        out, state = [], float(start_grip)
        for transition in transitions:
            action = np.array(transition["actions"], np.float32)
            if action[-1] != 0.0:
                state = float(np.sign(action[-1]))
            action[-1] = state
            out.append({**transition, "actions": action})
            if transition["dones"]:
                state = -1.0
        return out

    # -- demos -------------------------------------------------------------

    def _demo_files(self) -> list[Path]:
        files: list[Path] = []
        for path in map(Path, self.cfg.demo_path):
            if path.is_dir():
                files += sorted(path.glob("*.pkl"))
            elif path.is_file():
                files.append(path)
            else:
                raise FileNotFoundError(f"--demo-path {path} does not exist")
        return files

    def _load_demos(self) -> None:
        files = self._demo_files()
        total = 0
        segments = 0
        for path in files:
            demo = load_demo(path)
            if not demo.transitions:
                continue
            # Demos recorded before the intervention-cost work may carry the
            # old ``active`` field; the buffer only accepts ``intervened``.
            for transition in demo.transitions:
                transition.pop("active", None)
                transition.setdefault("grasp_penalty", 0.0)
            self._validate_demo_dims(demo.transitions, str(path))

            prepared_segments: list[list[dict]] = []
            if self.cfg.base_policy == "dmp":
                if self.bottleneck_mode and self.bottlenecks:
                    if len(demo.bottlenecks) < len(self.bottlenecks):
                        print(f"[demo] {path}: expected {len(self.bottlenecks)} sections; skipping")
                        continue
                    for i, bottleneck in enumerate(self.bottlenecks):
                        section = demo.bottlenecks[i]
                        start = int(section["start"])
                        end = int(section["end"])
                        if not (0 <= start < end <= len(demo.transitions)) or end - start < 2:
                            raise ValueError(f"{path}: invalid bottleneck section {i}: [{start}, {end})")
                        start_grip = float(section.get("start_grip", bottleneck.start_grip))
                        prepared_segments.append(
                            self._prepare_segment(demo.transitions[start:end], self.section_dmps[i], start_grip)
                        )
                else:
                    prepared_segments.append(self._prepare_segment(demo.transitions, self.full_dmp, -1.0))
            else:
                prepared_segments.append(self._prepare_demo(demo.transitions))

            for prepared in prepared_segments:
                if not prepared:
                    continue
                self.demo.force_episode_start()
                for transition in prepared:
                    self.demo.insert(transition)
                total += len(prepared)
                segments += 1

            key = str(path)
            if key not in self._demo_files_seen:
                self._demo_files_seen.add(key)
                self._demo_episodes += 1

        if files:
            print(
                f"[demo] loaded {total} transitions ({segments} segment(s)) "
                f"from {len(files)} file(s); demo buffer size = {len(self.demo)}"
            )
        self._save_stats()

    # -- teleop ------------------------------------------------------------

    def record_demos(self) -> None:
        cfg = self.cfg
        out = cfg.demo_output_dir
        out.mkdir(parents=True, exist_ok=True)
        saved = 0
        attempts = 0
        while saved < cfg.record_demos and self._env_steps < cfg.max_env_steps:
            attempts += 1
            self._teleop_base_hud = f"TELEOP demo {saved + 1}/{cfg.record_demos}"
            episode, success, sections = self._record_episode()
            if not success:
                print(f"[teleop] attempt {attempts}: not successful; discard")
                continue
            first_intervened = next((i for i, step in enumerate(episode) if step["intervened"]), None)
            if first_intervened is None:
                print(f"[teleop] attempt {attempts}: no intervention; discard")
                continue
            first = 0 if cfg.bottleneck_label else first_intervened
            # Teleop demos keep the sparse success reward; precollected demos
            # are untouched too.
            transitions = self._make_transitions(
                episode[first:],
                success=True,
            )
            adjusted = [
                {
                    "start": section["start"] - first,
                    "end": section["end"] - first,
                    "start_state": section["start_state"],
                    "end_state": section["end_state"],
                    "start_grip": section["start_grip"],
                }
                for section in sections
            ]
            if cfg.bottleneck_label:
                path = save_demo(
                    out,
                    transitions,
                    bottlenecks=adjusted,
                    metadata={
                        "task_id": int(cfg.task_id),
                        "suite": cfg.suite,
                        "rotation": bool(cfg.rotation),
                    },
                )
            else:
                path = save_transitions(out, transitions)
            saved += 1
            print(
                f"[teleop] saved demo {saved}/{cfg.record_demos}: "
                f"{len(transitions)} steps, {len(adjusted)} bottleneck section(s) -> {path}"
            )
        print(f"[teleop] done: {saved} demos in {out}. Train with: --demo-path {out}")

    def _record_episode(self) -> tuple[list[dict], bool, list[dict]]:
        cfg = self.cfg
        state = self.env_worker.make_pool(train=True, task=cfg.task_id, n_active=1)[0]
        grip = -1.0
        obs_enc = self.obs_adapter.encode(state.obs)
        episode: list[dict] = []
        sections: list[dict] = []
        open_section = None
        self._recording_sections = sections
        self._recording_open = None
        self.operator.reset()
        self._mode = "TELEOP"
        self._show(state.obs, "IDLE: press OPTIONS", _WHITE)
        if not (cfg.auto_start or self._stream is None):
            print("[teleop] press OPTIONS to start")
            while True:
                event = self.operator.poll()
                if event.options:
                    break
                time.sleep(0.02)

        success = False
        while self._env_steps < cfg.max_env_steps:
            event = self.operator.poll()
            if event.options:
                success = True
                break
            if event.share:
                success = False
                break
            if cfg.bottleneck_label:
                if event.bottleneck_start and open_section is None:
                    open_section = {
                        "start": len(episode),
                        "start_state": np.asarray(state.env.get_state(), np.float64),
                        "start_grip": float(grip),
                    }
                    self._recording_open = open_section
                    print(f"[teleop] bottleneck B{len(sections) + 1} started at step {len(episode)}")
                if event.bottleneck_end and open_section is not None:
                    end = len(episode)
                    if end - open_section["start"] >= 2:
                        open_section["end"] = end
                        open_section["end_state"] = np.asarray(state.env.get_state(), np.float64)
                        sections.append(open_section)
                        print(f"[teleop] bottleneck B{len(sections)} ended at step {end}")
                    else:
                        print("[teleop] bottleneck too short; ignored")
                    open_section = None
                    self._recording_open = None

            intervened, command = self.operator.takeover()
            final_action = command if cfg.rotation else command[[0, 1, 2, 6]]
            final_action = np.asarray(final_action, np.float32)
            final_action[-1] = _discretize_gripper(float(final_action[-1]))
            gripper = float(final_action[-1])
            penalty = cfg.grasp_penalty if gripper not in (0.0, grip) else 0.0
            if gripper != 0.0:
                grip = gripper

            state.env.send_step(self._env_action(final_action, grip))
            obs, done, info = state.env.recv_step()
            state.obs = obs
            state.steps += 1
            self._env_steps += 1
            reward = info.get("reward") if info else None
            success = bool(done and (reward is None or reward > 0))
            if success:
                state.success = True
                state.done = True
            elif done or state.steps >= cfg.max_episode_steps:
                state.done = True

            next_enc = self.obs_adapter.encode(obs)
            episode.append(
                {
                    "observations": obs_enc,
                    "actions": final_action,
                    "next_observations": next_enc,
                    "intervened": intervened,
                    "grasp_penalty": penalty,
                }
            )
            obs_enc = next_enc
            self._show(state.obs, self._teleop_base_hud, _ORANGE if intervened else _GREEN)
            if state.done:
                break
            time.sleep(1.0 / max(cfg.control_hz, 1e-6))

        if cfg.bottleneck_label and open_section is not None:
            end = len(episode)
            if end - open_section["start"] >= 2:
                open_section["end"] = end
                open_section["end_state"] = np.asarray(state.env.get_state(), np.float64)
                sections.append(open_section)
            self._recording_open = None
        return episode, success, sections

    def _make_transitions(
        self,
        steps: list[dict],
        *,
        success: bool,
    ) -> list[dict]:
        out = []
        for i, step in enumerate(steps):
            last = i == len(steps) - 1
            # Sparse success reward; intervention cost is handled by the
            # Lagrange multiplier in the learner.
            reward = 1.0 if success and last else 0.0
            transition = make_transition(
                step["observations"],
                step["actions"],
                step["next_observations"],
                reward=reward,
                done=last,
                intervened=step["intervened"],
                grasp_penalty=step["grasp_penalty"],
            )
            out.append(transition)
        return out

    # -- episode loop ------------------------------------------------------

    def run(self) -> None:
        self._training_seconds = self._elapsed_seconds()
        self._session_start = time.monotonic()
        try:
            if self.teleop_only:
                self.record_demos()
                return
            if self.cfg.eval_only:
                result = self.evaluate(self.cfg.eval_episodes, iteration=0, tag="eval_only")
                self._maybe_enable_perturbation(result)
                return
            self._start_learner()
            while self._episodes < self.cfg.max_episodes and self._env_steps < self.cfg.max_env_steps:
                result = self._train_episode()
                self._episodes += 1
                self._collected_episodes += 1
                self._flush(result.transitions, success=result.success)
                if self._should_eval():
                    result = self.evaluate(
                        self.cfg.eval_episodes,
                        iteration=self._collected_episodes,
                        tag=f"ep{self._collected_episodes}",
                    )
                    self._maybe_enable_perturbation(result)
            self._stop_learner()
            _learner.save_agent_checkpoint(self.cfg, self.agent, self._grad_step)
            print(
                f"[done] episodes={self._episodes} env_steps={self._env_steps} "
                f"learner_steps={self._grad_step} updates={self._updates}"
            )
            print(
                f"[done] online_episodes={self._online_episodes} demo_episodes={self._demo_episodes} "
                f"collected={self._collected_episodes} training={self._elapsed()}"
            )
        finally:
            self.close()

    def _train_episode(self) -> EpisodeResult:
        idx = self.train_selected if self.bottlenecks else None
        print(f"[train] episode {self._episodes + 1}: {self._selection_label(idx)}")
        return self._run_episode(idx, train=True)

    def _eval_episode(self, trial_offset: int) -> EpisodeResult:
        idx = self.eval_selected if self.bottleneck_mode else None
        return self._run_episode(idx, train=False, trial_offset=trial_offset)

    def _run_episode(
        self,
        bottleneck_idx: int | None,
        *,
        train: bool,
        trial_offset: int = 0,
    ) -> EpisodeResult:
        cfg = self.cfg
        bottleneck = self.bottlenecks[bottleneck_idx] if bottleneck_idx is not None else None
        self._episode_is_train = train
        if train:
            self._episode_train_idx = bottleneck_idx
        else:
            self._episode_eval_idx = bottleneck_idx

        state, grip, base_action, residual_active, obs_enc = self._reset_episode(
            bottleneck_idx,
            train=train,
            trial_offset=trial_offset,
        )
        transitions: list[dict] = []
        frames = [] if train else [self.obs_adapter.display(state.obs)]
        self.operator.reset()
        self._mode = "TRAIN" if train else "EVAL"
        label = f"{self._selection_label(bottleneck_idx)} | OPTIONS"
        self._show(state.obs, f"IDLE: {label}" if self._stream is not None else label, _WHITE)

        skipped = False
        if not (cfg.auto_start or self._stream is None):
            print("[episode] press OPTIONS to start")
            while True:
                event = self.operator.poll()
                if event.options:
                    break
                if not train and event.skip:
                    skipped = True
                    break
                self._handle_selection(event, training=train)
                self._check_learner()
                time.sleep(0.02)
            if skipped:
                return EpisodeResult(transitions=[], success=False, frames=frames, skipped=True)

        manual_mode = self.bottleneck_mode and (train or bottleneck is not None)
        manual_success = False
        manual_failure = False
        marking = False
        marking_start = 0.0
        mark_timeout_s = max(10.0, cfg.max_episode_steps / max(cfg.control_hz, 1e-6))
        max_steps = bottleneck.n_steps if bottleneck is not None else cfg.max_episode_steps
        step_idx = 0
        success = False

        while self._env_steps < cfg.max_env_steps:
            self._check_learner()
            event = self.operator.poll()
            if not train and event.skip:
                skipped = True
                break
            if event.options:
                manual_success = True
                break
            if event.share:
                manual_failure = True
                break
            self._handle_selection(event, training=train)

            if marking:
                if time.monotonic() - marking_start > mark_timeout_s:
                    print("[episode] marking timeout -> failure")
                    break
                time.sleep(0.02)
                continue

            if step_idx >= max_steps:
                if manual_mode:
                    marking = True
                    marking_start = time.monotonic()
                    print("[episode] bottleneck finished; OPTIONS=success SHARE=failure")
                    self._show(state.obs, "BOTTLENECK DONE: OPTIONS=success, SHARE=failure", _WHITE)
                    continue
                break

            step_start = time.monotonic()
            if train:
                intervened, command = self.operator.takeover()
            else:
                intervened, command = False, None

            if intervened:
                final_action = command if cfg.rotation else command[[0, 1, 2, 6]]
                final_action = np.asarray(final_action, np.float32)
            elif not residual_active:
                final_action = np.asarray(base_action, np.float32)
            else:
                agent = self.actor_agent if train else self.agent
                residual = np.array(agent.sample_actions(obs_enc, seed=self._next_key(), argmax=not train), np.float32)
                final_action = (
                    residual
                    if base_action is None
                    else combine_base_residual(base_action, residual, max_residual=cfg.residual_max)
                )

            final_action = np.asarray(final_action, np.float32)
            final_action[-1] = _discretize_gripper(float(final_action[-1]))
            gripper = float(final_action[-1])
            penalty = cfg.grasp_penalty if gripper not in (0.0, grip) else 0.0
            if gripper != 0.0:
                grip = gripper
            if cfg.algorithm == "hgdagger" and (intervened or residual_active):
                final_action[-1] = grip
            residual_action = final_action.copy() if base_action is None else residual_target(final_action, base_action)

            state.env.send_step(self._env_action(final_action, grip))
            obs, done, info = state.env.recv_step()
            state.obs = obs
            state.steps += 1
            self._env_steps += 1
            reward = info.get("reward") if info else None
            success = bool(done and (reward is None or reward > 0))
            if success:
                state.success = True
                state.done = True
            elif done or state.steps >= cfg.max_episode_steps:
                state.done = True

            next_base_action, next_residual_active = self._base_action_for_step(
                obs,
                grip,
                bottleneck_idx,
                step_idx + 1,
            )
            next_enc = self.obs_adapter.encode(obs, base_action=next_base_action)
            if train and residual_active:
                transitions.append(
                    {
                        "observations": obs_enc,
                        "actions": residual_action,
                        "next_observations": next_enc,
                        "intervened": intervened,
                        "grasp_penalty": penalty,
                    }
                )
            obs_enc, base_action, residual_active = next_enc, next_base_action, next_residual_active
            step_idx += 1

            self._mode = "HUMAN" if intervened else ("TRAIN" if train else "EVAL")
            self._show(
                state.obs,
                self._label(intervened=intervened),
                _ORANGE if intervened else _GREEN,
            )
            if not train:
                frames.append(self.obs_adapter.display(obs))

            if state.done:
                if manual_mode:
                    marking = True
                    marking_start = time.monotonic()
                    print("[episode] env ended; OPTIONS=success SHARE=failure")
                    self._show(state.obs, "BOTTLENECK DONE: OPTIONS=success, SHARE=failure", _WHITE)
                else:
                    break
            sleep_s = 1.0 / max(cfg.control_hz, 1e-6) - (time.monotonic() - step_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        if manual_failure:
            success = False
        elif manual_success:
            success = True
        elif manual_mode:
            success = False
        else:
            success = state.success

        return EpisodeResult(
            self._make_transitions(transitions, success=success),
            bool(success),
            frames,
            skipped,
        )

    def _flush(self, transitions: list[dict], *, success: bool) -> None:
        if not transitions:
            return
        if self.online is not None:
            for transition in transitions:
                self.online.insert(transition)
            self._online_episodes += 1

        if self.bottleneck_mode:
            demo_segments = [transitions]
        else:
            demo_segments = []
            segment: list[dict] = []
            for transition in transitions:
                if transition["intervened"]:
                    segment.append(transition)
                elif segment:
                    demo_segments.append(segment)
                    segment = []
            if segment:
                demo_segments.append(segment)

        demo_steps = 0
        for segment in demo_segments:
            if hasattr(self.demo, "force_episode_start"):
                self.demo.force_episode_start()
            for transition in segment:
                self.demo.insert(transition)
            demo_steps += len(segment)
        if demo_steps:
            self._demo_episodes += 1
        intervened_steps, total_steps = episode_stats(transitions)
        self.intervention_tracker.add_episode(intervened_steps, total_steps)
        intervention_segments = 0
        prev_intervened = False
        for transition in transitions:
            intervened = bool(transition["intervened"])
            if intervened and not prev_intervened:
                intervention_segments += 1
            prev_intervened = intervened
        self._intervention_steps += intervened_steps
        self._save_stats()
        print(
            f"[flush] online={len(transitions)} demo={demo_steps} "
            f"intervened={intervened_steps} rate={intervened_steps / total_steps:.3f} "
            f"avg_rate={self.intervention_tracker.rate:.3f} "
            f"segments={intervention_segments} success={success} "
            f"online_episodes={self._online_episodes} demo_episodes={self._demo_episodes}"
        )

    def _handle_selection(self, event: PadEvent, *, training: bool) -> bool:
        if not self.bottleneck_mode or not (event.prev_bottleneck or event.next_bottleneck):
            return False
        if not self.bottlenecks:
            return False
        delta = -1 if event.prev_bottleneck else 1
        if training:
            n = len(self.bottlenecks)
            current = self.train_selected if self.train_selected is not None else 0
            self.train_selected = (current + delta) % n
            print(f"[bottleneck] training {self._selection_label(self.train_selected)}")
        else:
            options = [None, *range(len(self.bottlenecks))]
            current = self.eval_selected
            pos = (options.index(current) + delta) % len(options)
            self.eval_selected = options[pos]
            print(f"[bottleneck] eval {self._selection_label(self.eval_selected)}")
        return True

    def _should_eval(self) -> bool:
        return (
            self.cfg.eval_interval > 0
            and self._collected_episodes >= self.cfg.eval_after_episodes
            and self._collected_episodes % self.cfg.eval_interval == 0
        )

    def _maybe_enable_perturbation(self, result: dict) -> None:
        threshold = float(self.cfg.perturb_after_success_rate)
        if (
            threshold > 0.0
            and bool(self.cfg.perturb_objects)
            and not self.env_worker.perturb_enabled
            and float(result.get("success_rate", 0.0)) > threshold
        ):
            self.env_worker.set_perturb(enabled=True)
            print(
                f"[perturb] success={result['success_rate']:.2f} > {threshold:.2f}; "
                "enabling EEF/object start-state perturbations"
            )

    def _env_action(self, action: np.ndarray, grip: float) -> np.ndarray:
        action = np.asarray(action, np.float32)
        if action.shape[-1] == 4:
            action = np.array([action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]], np.float32)
        env_action = self.act_adapter.to_env(action)
        env_action[6] = grip
        return env_action

    def _label(self, *, intervened: bool) -> str:
        mode = "HUMAN" if intervened else "POLICY"
        bottleneck = ""
        if self.bottleneck_mode:
            idx = self._episode_train_idx if self._episode_is_train else self._episode_eval_idx
            selected = self.train_selected if self._episode_is_train else self.eval_selected
            bottleneck = f" | {self._selection_label(idx)}"
            if selected != idx:
                bottleneck += f" (next {self._selection_label(selected)})"
        return (
            f"O{self._online_episodes} D{self._demo_episodes} E{self._collected_episodes} "
            f"{mode} | {bottleneck}"
        )

    # -- learner -----------------------------------------------------------

    def _start_learner(self) -> None:
        self._learner_thread = threading.Thread(target=self._learner_loop, name="learner", daemon=True)
        self._learner_thread.start()

    def _stop_learner(self) -> None:
        self._stop.set()
        if self._learner_thread is not None:
            self._learner_thread.join(timeout=60)

    def _check_learner(self) -> None:
        if self._learner_error is not None:
            raise RuntimeError("the learner thread crashed") from self._learner_error

    def _wait_for_data(self) -> bool:
        cfg = self.cfg
        buffer, need = (self.online, cfg.training_starts) if cfg.algorithm == "rlpd" else (self.demo, 1)
        while not self._stop.is_set() and len(buffer) < need:
            time.sleep(0.5)
        return not self._stop.is_set()

    def _learner_loop(self) -> None:
        cfg = self.cfg
        try:
            if not self._wait_for_data():
                return
            agent = self.agent
            if cfg.algorithm == "hgdagger" and self._grad_step == 0 and cfg.pretrain_steps > 0:
                print(f"[learner] BC pretraining: {cfg.pretrain_steps} steps on {len(self.demo)} demo transitions")
                for _ in range(cfg.pretrain_steps):
                    if self._stop.is_set():
                        break
                    agent, _ = agent.update(_learner.sample_buffer(self.demo, cfg.batch_size))
                    self._grad_step += 1
                    self._updates += 1
                    self.agent = agent
            self.actor_agent = jax.block_until_ready(agent)
            print("[learner] published the initial network to the actor; learning continuously")

            step = self._grad_step
            log_time, log_step = time.monotonic(), step
            while not self._stop.is_set():
                if cfg.algorithm == "rlpd":
                    for _ in range(max(cfg.cta_ratio - 1, 0)):
                        batch = _learner.mixed_batch(self.online, self.demo, cfg.batch_size)
                        agent, _ = agent.update(batch, networks_to_update=_learner.CRITIC_ONLY)
                    batch = _learner.mixed_batch(self.online, self.demo, cfg.batch_size)
                    # Before the first episode completes, pass the target so
                    # the multiplier does not decay for no reason.
                    intervention_rate = (
                        self.intervention_tracker.rate
                        if self.intervention_tracker.num_episodes > 0
                        else cfg.target_intervention_rate
                    )
                    agent, info = agent.update(
                        batch,
                        networks_to_update=_learner.FULL_UPDATE,
                        intervention_rate=intervention_rate,
                    )
                    n_updates = max(cfg.cta_ratio, 1)
                else:
                    agent, info = agent.update(_learner.sample_buffer(self.demo, cfg.batch_size))
                    n_updates = 1

                step += n_updates
                self._updates += n_updates
                self.agent, self._grad_step = agent, step
                if cfg.steps_per_update > 0 and step % cfg.steps_per_update == 0:
                    self.actor_agent = jax.block_until_ready(agent)
                if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                    flat = _learner.flatten_info(info)
                    flat["intervention_episodes"] = self.intervention_tracker.num_episodes
                    steps_per_s = (step - log_step) / max(time.monotonic() - log_time, 1e-6)
                    log_time, log_step = time.monotonic(), step
                    print(
                        f"[learner step {step}] {steps_per_s:.1f} steps/s updates/env_step="
                        f"{self._updates / max(self._env_steps, 1):.2f} | "
                        + " ".join(f"{k}={float(v):.4f}" for k, v in sorted(flat.items()))
                    )
                if cfg.checkpoint_period > 0 and step % cfg.checkpoint_period == 0:
                    _learner.save_agent_checkpoint(cfg, agent, step)
        except BaseException as exc:
            import traceback

            traceback.print_exc()
            self._learner_error = exc

    # -- eval --------------------------------------------------------------

    def evaluate(self, n_episodes: int, *, iteration: int, tag: str) -> dict:
        n_attempts = max(1, int(n_episodes))
        successes: list[bool] = []
        skipped = 0
        self._mode = "EVAL"
        self._episode_is_train = False
        print(
            f"[eval {tag}] task {self.cfg.task_id}: {self.task_name} | "
            f"{n_attempts} episodes | {self._selection_label(self.eval_selected)}"
        )
        for i in range(n_attempts):
            result = self._eval_episode(trial_offset=i)
            self._save_video(result.frames, tag=f"{tag}_s{i}", success=result.success, skipped=result.skipped)
            if result.skipped:
                skipped += 1
                print(f"[eval {tag}] episode {i + 1}/{n_attempts}: skipped")
            else:
                successes.append(result.success)
                print(f"[eval {tag}] episode {i + 1}/{n_attempts}: success={result.success}")

        success_rate = float(np.mean(successes)) if successes else 0.0
        n_success = int(sum(successes))
        result = {
            "tag": tag,
            "iteration": int(iteration),
            "episodes": n_attempts,
            "scored": len(successes),
            "successes": n_success,
            "success_rate": success_rate,
            "skipped": skipped,
            "bottleneck": self._selection_label(self.eval_selected) if self.bottleneck_mode else None,
        }
        print(
            f"[eval {tag}] success_rate={success_rate:.2f} "
            f"({n_success}/{len(successes)}) skipped={skipped}"
        )
        with self._eval_log.open("a") as f:
            f.write(json.dumps(result) + "\n")
        self._mode = "IDLE"
        return result

    # -- display -----------------------------------------------------------

    def _next_key(self):
        self._rng, key = jax.random.split(self._rng)
        return key

    def _show(self, obs: dict, label: str = "", color: tuple[int, int, int] = _WHITE) -> None:
        if self._stream is not None:
            self._stream.update(self.obs_adapter.display(obs), label, color)

    def _save_video(
        self,
        frames: list[np.ndarray],
        *,
        tag: str,
        success: bool,
        skipped: bool = False,
    ) -> None:
        if not frames:
            return
        suffix = "skipped" if skipped else ("success" if success else "failure")
        path = Path(self.cfg.video_dir) / f"{tag}_{suffix}.mp4"
        try:
            imageio.mimsave(path, frames, fps=max(self.cfg.control_hz, 1.0))
        except Exception as exc:
            print(f"[video] failed to write {path}: {exc}")

    def close(self) -> None:
        self._stop.set()
        if self._learner_thread is not None:
            self._learner_thread.join(timeout=10)
        self.env_worker.close()
        if self._stream is not None:
            self._stream.close()
        self._save_stats()
