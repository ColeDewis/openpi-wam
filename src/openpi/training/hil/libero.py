"""LIBERO actor/learner for RLPD and HG-Dagger.

The actor runs in the main thread and steps LIBERO. The learner runs in a
background thread and publishes weights every ``steps_per_update`` steps, as in
hil-serl. Human interventions come from the browser gamepad page served by the
optional MJPEG stream.

Record demos (pure teleop, one pickle per successful episode):

    uv run --extra libero scripts/train_hil_libero.py --record-demos 20 \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 --suite libero_goal --task-id 1

Train (resumes from the latest checkpoint in ``--checkpoint-dir``):

    uv run --extra libero scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1

Train a residual policy on top of a DMP base policy. The DMP is fit to the
first trajectory found under ``--demo-path``; the RL agent sees the DMP's next
delta-pose action in proprio and learns only the residual:

    uv run --extra libero scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 \
        --base-policy dmp \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd_dmp \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1_dmp

Controls: OPTIONS starts an episode / marks success, SHARE marks failure,
left stick moves x/y, triangle/cross moves z, L2/R2 opens/closes the gripper,
and PS hands control back to the policy. D-pad left/right (or L1/R1) skips the
current eval episode. The right stick and square/circle control rotation only
with ``--rotation``.
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
import pickle as pkl
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
from openpi.training.hil.dmp import DmpBasePolicy
from openpi.training.hil.dmp import combine_base_residual
from openpi.training.hil.dmp import residual_from_final
from openpi.training.hil.transitions import make_transition
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
 <div id="status">task: loading... | online episodes: 0 | demo episodes: 0 | collected episodes: 0 | training: 00:00:00</div>
 <div id="grid">
  <div><b>OPTIONS</b> start episode / mark <b>SUCCESS</b></div>
  <div><b>SHARE</b> mark <b>FAILURE</b></div>
  <div><b>Left stick</b> x (up/down), y (left/right)</div>
  <div><b>Right stick</b> pitch / yaw <span class="dim">(only with --rotation)</span></div>
  <div><b>&#9651; / &#10005;</b> z up / z down</div>
  <div><b>&#9633; / &#9675;</b> roll left / right <span class="dim">(only with --rotation)</span></div>
  <div><b>L2 / R2</b> gripper open / close (release = hold)</div>
  <div><b>D-pad &#9664; / &#9654;</b> (or L1 / R1) skip current eval episode</div>
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
      " | training: " + s.training + " | " + s.mode + " | learner step " + s.learner_step;
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
    options: bool = False  # OPTIONS: start (idle) / mark success (recording)
    share: bool = False  # SHARE: mark failure
    skip: bool = False  # D-pad left/right or L1/R1: skip the current eval episode


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
    DPAD_LEFT, DPAD_RIGHT = 14, 15
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
    ):
        self.rotation = bool(rotation)
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
            if rising[[self.DPAD_LEFT, self.DPAD_RIGHT, self.L1, self.R1]].any():
                self._event.skip = True
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
        ``cmd``: ``[x, y, z, roll, pitch, yaw, gripper]``.
        """
        zeros = np.zeros(7, np.float32)
        now = time.monotonic()
        with self._lock:
            if now - self._last_msg >= self.stale_s:
                return self._latched, False, zeros
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
                    return False, False, zeros  # still holding input after hand-back: policy runs
                self._wait_idle = False
            if moving:
                self._latched = True
                self._last_input = now
            elif self._latched and self.handback_s > 0 and now - self._last_input > self.handback_s:
                self._latched = False

            full = np.concatenate([cmd * self.takeover_step, [grip]]).astype(np.float32)
            return self._latched, moving, full


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

    # Pure teleop: save N successful demos to the first ``demo_path`` and exit.
    # The same ``demo_path`` is also the source of training demos, so there is
    # only one name for the directory.
    record_demos: int = 0
    demo_path: tuple[str, ...] = ()

    # Base policy that the RL agent learns a residual on top of.
    # ``dmp`` fits a dynamic movement primitive to the first trajectory found
    # under ``demo_path`` and appends its next delta-pose action to proprio.
    base_policy: Literal["none", "dmp"] = "none"
    dmp_weights_per_dim: int = 20
    dmp_regularization: float = 0.0
    dmp_gripper: bool = True
    """Include the demonstrated gripper command as a DMP dimension.  Its base
    action is the absolute normalized gripper target; the RL residual is added
    on top with the same residual-driven weighting as the arm dims."""

    # Base/residual composition.  The base action is scaled by
    # ``1 - |residual| / residual_blend_scale`` per dimension (clipped to
    # ``residual_min_base_weight``), then the residual is added.  A small
    # residual therefore leaves the base in charge; a large residual suppresses
    # it.  This is the sim stand-in for the real robot's ``policy_torque_scale``.
    residual_blend_scale: float = 1.0
    residual_min_base_weight: float = 0.0

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
        self.image_size = tuple(self.image_size)
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be (height, width), got {self.image_size}")
        self.demo_path = tuple(self.demo_path)
        self.max_episode_steps = int(self.max_episode_steps) or _MAX_STEPS[self.suite]
        self.max_env_steps = int(self.max_env_steps) or self.max_episodes * self.max_episode_steps
        if not (0.0 < self.max_step_m <= _OSC_MAX_M and 0.0 < self.max_step_rad <= _OSC_MAX_RAD):
            raise ValueError(
                f"max_step_m must be in (0, {_OSC_MAX_M}] and max_step_rad in (0, {_OSC_MAX_RAD}] "
                "(the OSC controller's full scale)"
            )
        if self.base_policy not in ("none", "dmp"):
            raise ValueError(f"base_policy must be 'none' or 'dmp', got {self.base_policy!r}")
        if self.base_policy == "dmp":
            if self.record_demos == 0 and not self.demo_path:
                raise ValueError("--base-policy dmp needs --demo-path with at least one demo pickle")
            if self.dmp_weights_per_dim <= 1:
                raise ValueError("dmp_weights_per_dim must be > 1")
            if self.dmp_regularization < 0.0:
                raise ValueError("dmp_regularization must be >= 0")
        if self.residual_blend_scale <= 0.0:
            raise ValueError("residual_blend_scale must be positive")
        if not (0.0 <= self.residual_min_base_weight <= 1.0):
            raise ValueError("residual_min_base_weight must be in [0, 1]")
        if self.perturb_eef_xyz_m < 0.0 or self.perturb_object_xy_m < 0.0:
            raise ValueError("perturbation magnitudes must be >= 0")
        if not (0.0 <= self.perturb_after_success_rate <= 1.0):
            raise ValueError("perturb_after_success_rate must be in [0, 1]")
        if self.record_demos > 0 and not self.display:
            raise ValueError("--record-demos needs --display (the browser gamepad page)")
        if self.eval_after_episodes < 0 or self.eval_interval < 0:
            raise ValueError("eval_after_episodes and eval_interval must be >= 0")
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


class HilLiberoTrainer:
    """Actor in the main thread, learner in a background thread."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.teleop_only = cfg.record_demos > 0

        # The DMP is fit before the observation adapter is built because its
        # action width is part of the proprio vector.
        self.base_policy: DmpBasePolicy | None = None
        if not self.teleop_only and cfg.base_policy == "dmp":
            self.base_policy = self._make_dmp_base_policy()
        elif cfg.base_policy == "dmp":
            print("[dmp] --record-demos is pure teleop; --base-policy is ignored while recording.")

        base_action_dim = cfg.base_action_dim if self.base_policy is not None else 0
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
        self.operator = GamepadOperator(cfg.takeover_step, handback_s=cfg.handback_s, rotation=cfg.rotation)
        base_note = (
            f" | base policy: DMP ({base_action_dim}-dim base action appended to proprio, "
            f"gripper={'modeled' if cfg.dmp_gripper else 'residual-only'})"
            if self.base_policy is not None
            else ""
        )
        print(
            f"[action] {cfg.action_dim}-dim agent action ({'6-DoF' if cfg.rotation else 'xyz'} + gripper); "
            f"normalized +-1 -> {cfg.max_step_m} m / {cfg.max_step_rad} rad per step{base_note}"
        )

        self.agent = self.actor_agent = None
        self.online = self.demo = None
        self._hud = ""
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
            "training_seconds": round(self._elapsed_seconds(), 1),
            "demo_files": sorted(self._demo_files_seen),
        }
        self._stats_path.write_text(json.dumps(payload, indent=2))

    def _status(self) -> dict:
        return {
            "task": self.task_name,
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "training": self._elapsed(),
            "mode": self._mode,
            "learner_step": self._grad_step,
        }

    # -- base policy -------------------------------------------------------

    def _check_checkpoint_config(self) -> None:
        """Refuse to restore a checkpoint trained with a different obs/action space."""
        config_path = Path(self.cfg.checkpoint_dir) / "config.json"
        if not config_path.exists():
            return
        try:
            old = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        old_encoder = old.get("encoder", {})
        old_kind = old_encoder.get("kind")
        old_proprio = old_encoder.get("proprio_dim")
        old_base_dim = old_encoder.get("base_action_dim")
        old_action = old.get("action_dim")
        old_base = old.get("base_policy", "none")
        old_gripper = old.get("dmp_gripper")
        old_blend = old.get("residual_blend_scale")
        old_min_weight = old.get("residual_min_base_weight")
        if (
            (old_kind is not None and old_kind != self.cfg.encoder.kind)
            or (old_proprio is not None and int(old_proprio) != self.cfg.encoder.proprio_dim)
            or (old_base_dim is not None and int(old_base_dim) != self.cfg.encoder.base_action_dim)
            or (old_action is not None and int(old_action) != self.cfg.action_dim)
            or old_base != self.cfg.base_policy
            or (old_gripper is not None and bool(old_gripper) != self.cfg.dmp_gripper)
            or (old_blend is not None and abs(float(old_blend) - self.cfg.residual_blend_scale) > 1e-9)
            or (
                old_min_weight is not None
                and abs(float(old_min_weight) - self.cfg.residual_min_base_weight) > 1e-9
            )
        ):
            raise ValueError(
                f"{config_path} was trained with encoder={old_kind}, proprio_dim={old_proprio}, "
                f"base_action_dim={old_base_dim}, action_dim={old_action}, base_policy={old_base!r}, "
                f"dmp_gripper={old_gripper}, residual_blend_scale={old_blend}, "
                f"residual_min_base_weight={old_min_weight}; "
                f"this run uses encoder={self.cfg.encoder.kind}, "
                f"proprio_dim={self.cfg.encoder.proprio_dim}, "
                f"base_action_dim={self.cfg.encoder.base_action_dim}, "
                f"action_dim={self.cfg.action_dim}, base_policy={self.cfg.base_policy!r}, "
                f"dmp_gripper={self.cfg.dmp_gripper}, "
                f"residual_blend_scale={self.cfg.residual_blend_scale}, "
                f"residual_min_base_weight={self.cfg.residual_min_base_weight}. "
                "Use a fresh --checkpoint-dir for the new observation/action space."
            )

    def _pose_from_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, np.float32).reshape(-1)
        return state[:6] if self.cfg.rotation else state[:3]

    def _pose(self, obs: dict) -> np.ndarray:
        return self._pose_from_state(self.obs_adapter.state(obs))

    def _pose_with_grip(self, obs: dict, grip: float) -> np.ndarray:
        """DMP pose plus the current commanded gripper state, when modeled."""
        pose = self._pose(obs)
        if self.cfg.dmp_gripper:
            pose = np.concatenate([pose, np.asarray([grip], np.float32)])
        return pose

    def _make_dmp_base_policy(self) -> DmpBasePolicy:
        files = self._demo_files()
        if not files:
            raise FileNotFoundError("--base-policy dmp needs at least one demo pickle under --demo-path")
        source = files[0]
        with source.open("rb") as f:
            transitions = pkl.load(f)
        if not transitions:
            raise ValueError(f"--base-policy dmp: {source} contains no transitions")

        actions = [np.asarray(t["actions"], np.float32) for t in transitions]
        states = [np.asarray(t["observations"]["state"]) for t in transitions]
        states.append(np.asarray(transitions[-1]["next_observations"]["state"]))
        poses = np.stack([self._pose_from_state(s) for s in states])
        if self.cfg.dmp_gripper:
            # State i sees the gripper command applied at i-1; state 0 starts open
            # (LIBERO resets with gripper=-1).  This aligns each DMP target with
            # the next demonstrated action's gripper entry.
            grip_states = np.concatenate([[-1.0], [float(a[-1]) for a in actions]])
            poses = np.concatenate([poses, grip_states[:, None]], axis=1)
        policy = DmpBasePolicy(
            poses,
            dt=1.0 / max(self.cfg.control_hz, 1e-6),
            rotation=self.cfg.rotation,
            gripper=self.cfg.dmp_gripper,
            action_dim=self.cfg.action_dim,
            max_step_m=self.cfg.max_step_m,
            max_step_rad=self.cfg.max_step_rad,
            n_weights_per_dim=self.cfg.dmp_weights_per_dim,
            regularization_coefficient=self.cfg.dmp_regularization,
        )
        print(
            f"[dmp] fit base policy to first demo {source} "
            f"({len(transitions)} transitions, {len(poses)} pose samples, "
            f"pose_dim={policy.pose_dim}, gripper={policy.gripper}, "
            f"weights/dim={self.cfg.dmp_weights_per_dim})"
        )
        return policy

    def _base_action(self, obs: dict, grip: float) -> np.ndarray | None:
        if self.base_policy is None:
            return None
        return self.base_policy.action(self._pose_with_grip(obs, grip))

    def _append_base_action(self, obs: dict, base_action: np.ndarray) -> dict:
        out = dict(obs)
        state = np.asarray(out["state"], np.float32).reshape(-1)
        out["state"] = np.concatenate([state, np.asarray(base_action, np.float32)])[None]
        return out

    def _add_base_actions(self, transitions: list[dict]) -> list[dict]:
        """Expand demo observations with DMP actions and convert actions to residuals."""
        actions = [np.asarray(t["actions"], np.float32) for t in transitions]
        states = [np.asarray(t["observations"]["state"]) for t in transitions]
        states.append(np.asarray(transitions[-1]["next_observations"]["state"]))
        poses = np.stack([self._pose_from_state(s) for s in states])
        if self.cfg.dmp_gripper:
            grip_states = np.concatenate([[-1.0], [float(a[-1]) for a in actions]])
            poses = np.concatenate([poses, grip_states[:, None]], axis=1)
        self.base_policy.reset(poses[0])
        base_actions = np.stack([self.base_policy.action(pose) for pose in poses])

        out = []
        for i, transition in enumerate(transitions):
            action = np.asarray(transition["actions"], np.float32).copy()
            residual = residual_from_final(
                action,
                base_actions[i],
                residual_scale=self.cfg.residual_blend_scale,
                min_base_weight=self.cfg.residual_min_base_weight,
            )
            out.append(
                {
                    **transition,
                    "observations": self._append_base_action(transition["observations"], base_actions[i]),
                    "next_observations": self._append_base_action(transition["next_observations"], base_actions[i + 1]),
                    "actions": residual,
                }
            )
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

    def _prepare_demo(self, transitions: list[dict], source: str) -> list[dict]:
        if not transitions:
            return []
        dim = int(np.asarray(transitions[0]["actions"]).shape[-1])
        if dim != self.cfg.action_dim:
            raise ValueError(
                f"{source}: demo actions are {dim}-dim but this run uses {self.cfg.action_dim} "
                f"(--rotation={self.cfg.rotation}). Record and train with the same --rotation."
            )
        expected_state_dim = self.obs_adapter.proprio_dim - self.obs_adapter.base_action_dim
        demo_state_dim = int(np.asarray(transitions[0]["observations"]["state"]).shape[-1])
        if demo_state_dim != expected_state_dim:
            raise ValueError(
                f"{source}: demo states are {demo_state_dim}-dim but this run uses "
                f"{expected_state_dim} (--include-velocity={self.cfg.include_velocity}). "
                "Record and train with the same --include-velocity."
            )

        # With a base policy the stored action is the residual the RL agent
        # must learn: residual = demonstrated_action - dmp_action.
        if self.base_policy is not None:
            transitions = self._add_base_actions(transitions)

        if self.cfg.algorithm != "hgdagger":
            return transitions
        out, state = [], -1.0  # LIBERO resets with the gripper open
        for transition in transitions:
            action = np.array(transition["actions"], np.float32)
            if action[-1] != 0.0:
                state = float(np.sign(action[-1]))
            action[-1] = state
            out.append({**transition, "actions": action})
            if transition["dones"]:
                state = -1.0
        return out

    def _load_demos(self) -> None:
        files = self._demo_files()
        total = 0
        for path in files:
            with path.open("rb") as f:
                transitions = self._prepare_demo(pkl.load(f), str(path))
            if not transitions:
                continue
            self.demo.force_episode_start()
            for transition in transitions:
                self.demo.insert(transition)
            total += len(transitions)
            key = str(path)
            if key not in self._demo_files_seen:
                self._demo_files_seen.add(key)
                self._demo_episodes += 1
        if files:
            print(f"[demo] loaded {total} transitions from {len(files)} file(s); demo buffer size = {len(self.demo)}")
        self._save_stats()

    def record_demos(self) -> None:
        """Pure teleop: save successful episodes, one pickle each."""
        if self._stream is None:
            raise RuntimeError("--record-demos needs the browser gamepad page (the stream failed to start)")
        out = self.cfg.demo_output_dir
        saved = attempts = 0
        print(
            f"[teleop] task {self.cfg.task_id}: {self.task_name}\n"
            f"[teleop] recording {self.cfg.record_demos} successful demos -> {out}\n"
            "[teleop] OPTIONS starts an attempt; the env ends it on success (or OPTIONS marks success); "
            "SHARE or a timeout discards it."
        )
        while saved < self.cfg.record_demos:
            self._hud = f"TELEOP {saved}/{self.cfg.record_demos}"
            episode, success = self._run_episode(teleop_only=True)
            attempts += 1
            first = next((i for i, step in enumerate(episode) if step["intervened"]), None)
            if not success or first is None:
                print(f"[teleop] discarded attempt {attempts} (success={success}, steps={len(episode)})")
                continue
            transitions = self._make_transitions(episode[first:], success=True)
            path = save_transitions(out, transitions)
            saved += 1
            print(f"[teleop] saved demo {saved}/{self.cfg.record_demos}: {len(transitions)} steps -> {path}")
        print(f"[teleop] done: {saved} demos in {out}. Train with: --demo-path {out}")

    # -- episode loop ------------------------------------------------------

    def run(self) -> None:
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
                episode, success = self._run_episode()
                self._episodes += 1
                self._collected_episodes += 1
                self._flush(episode, success=success)
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
        """Agent action -> the 7-dim LIBERO action.

        The env receives the persistent commanded gripper state (+-1), not the
        agent's {-1, 0, +1} event. A held command lets a tap complete a full
        open/close, like the real Franka gripper.
        """
        action = np.asarray(action, np.float32)
        if action.shape[-1] == 4:
            action = np.array([action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]], np.float32)
        env_action = self.act_adapter.to_env(action)
        env_action[6] = grip
        return env_action

    def _label(self, *, active: bool) -> str:
        mode = "HUMAN" if active else "POLICY"
        return f"O{self._online_episodes} D{self._demo_episodes} E{self._collected_episodes} {self._elapsed()} | {mode} L{self._grad_step}"

    def _run_episode(self, *, teleop_only: bool = False) -> tuple[list[dict], bool]:
        cfg = self.cfg
        state = self.env_worker.make_pool(train=True, task=cfg.task_id, n_active=1)[0]
        episode: list[dict] = []
        grip = -1.0  # commanded gripper state (LIBERO resets open)
        manual_success = manual_failure = False

        self.operator.reset()
        self._mode = "IDLE"
        self._show(state.obs, "IDLE: press OPTIONS", _WHITE)

        # Start the DMP from the actual episode pose and encode its first base
        # action into the observation the agent sees.
        base_action = None
        if self.base_policy is not None:
            self.base_policy.reset(self._pose_with_grip(state.obs, grip))
            base_action = self._base_action(state.obs, grip)
        obs_enc = self.obs_adapter.encode(state.obs, base_action=base_action)

        print(f"\n[episode] task {cfg.task_id}: {self.task_name}")
        if not (cfg.auto_start or self._stream is None):
            print("[episode] Press OPTIONS to start.")
            while not self.operator.poll().options:
                self._check_learner()
                time.sleep(0.02)
            print("[episode] recording: OPTIONS = success, SHARE = failure")
        if not teleop_only and self._session_start is None:
            self._session_start = time.monotonic()

        while True:
            self._check_learner()
            event = self.operator.poll()
            if event.options:
                manual_success = True
                break
            if event.share:
                manual_failure = True
                break
            # if self._env_steps % 50 == 0:
            #     print("COINSDFOIN")
            #     print(self._env_steps)

            if self._env_steps >= cfg.max_env_steps:
                break

            step_start = time.monotonic()
            active, moving, cmd = self.operator.takeover()
            if teleop_only:
                active = True

            if active:
                final_action = cmd if cfg.rotation else cmd[[0, 1, 2, 6]]
                intervened = moving  # idle hold frames are not demos
            else:
                residual = np.array(
                    self.actor_agent.sample_actions(obs_enc, seed=self._next_key(), argmax=False),
                    np.float32,
                )
                final_action = (
                    residual
                    if base_action is None
                    else combine_base_residual(
                        base_action,
                        residual,
                        residual_scale=cfg.residual_blend_scale,
                        min_base_weight=cfg.residual_min_base_weight,
                    )
                )
                intervened = False

            final_action = np.asarray(final_action, np.float32)
            final_action[-1] = _discretize_gripper(float(final_action[-1]))
            g = float(final_action[-1])
            penalty = cfg.grasp_penalty if g not in (0.0, grip) else 0.0
            if g != 0.0:
                grip = g
            if cfg.algorithm == "hgdagger" and not teleop_only:
                final_action[-1] = grip  # BC regresses a persistent open/closed label

            # The buffer stores the residual the RL agent must learn. With a
            # base policy the executed action is the weighted base+residual
            # composition; invert it so the stored residual reproduces the
            # demonstrated/executed action under the same weighting.
            if base_action is not None:
                residual_action = residual_from_final(
                    final_action,
                    base_action,
                    residual_scale=cfg.residual_blend_scale,
                    min_base_weight=cfg.residual_min_base_weight,
                )
            else:
                residual_action = final_action.copy()

            state.env.send_step(self._env_action(final_action, grip))
            obs, done, info = state.env.recv_step()
            state.obs = obs
            state.steps += 1
            reward = info.get("reward") if info else None
            success = bool(done and (reward is None or reward > 0))
            if success:
                state.success = True
                state.done = True
            elif done or state.steps >= cfg.max_episode_steps:
                state.done = True

            next_base_action = self._base_action(obs, grip) if self.base_policy is not None else None
            next_enc = self.obs_adapter.encode(obs, base_action=next_base_action)
            episode.append(
                {
                    "observations": obs_enc,
                    "actions": residual_action,
                    "next_observations": next_enc,
                    "intervened": intervened,
                    "grasp_penalty": penalty,
                }
            )
            obs_enc, base_action = next_enc, next_base_action
            self._env_steps += 1
            self._mode = "HUMAN" if active else "POLICY"
            self._show(state.obs, self._hud or self._label(active=active), _ORANGE if active else _GREEN)
            if state.done:
                break
            time.sleep(max(0.0, 1.0 / max(cfg.control_hz, 1e-6) - (time.monotonic() - step_start)))

        if manual_failure:
            success = False
        elif manual_success:
            success = True
        else:
            success = state.success
        return episode, bool(success)

    # -- buffering ---------------------------------------------------------

    def _make_transitions(self, episode: list[dict], *, success: bool) -> list[dict]:
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
        return transitions

    def _flush(self, episode: list[dict], *, success: bool) -> None:
        if not episode:
            return
        transitions = self._make_transitions(episode, success=success)
        if self.online is not None:
            for transition in transitions:
                self.online.insert(transition)
            self._online_episodes += 1

        # Demo pickles/segments are non-contiguous: force a fresh stacked-observation
        # window at the start of every intervention run.
        in_segment = False
        interventions = 0
        for transition in transitions:
            if not transition["intervened"]:
                in_segment = False
                continue
            if not in_segment:
                self.demo.force_episode_start()
                in_segment = True
            self.demo.insert(transition)
            interventions += 1
        if interventions:
            self._demo_episodes += 1
        self._save_stats()

        print(
            f"[episode {self._episodes}] success={success} steps={len(transitions)} "
            f"interventions={interventions} online={len(self.online) if self.online is not None else 0} "
            f"demo={len(self.demo)} | online_eps={self._online_episodes} demo_eps={self._demo_episodes} "
            f"collected={self._collected_episodes} training={self._elapsed()} learner_step={self._grad_step}"
        )
        if self.cfg.algorithm == "hgdagger" and interventions == 0 and len(self.demo) == 0:
            now = time.monotonic()
            if now - self._last_hint > 5.0:
                print("[hgdagger] no interventions yet - move a stick during rollout to create BC data.")
                self._last_hint = now

    # -- learner thread ----------------------------------------------------

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
                    agent, info = agent.update(batch, networks_to_update=_learner.FULL_UPDATE)
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
                    rate = (step - log_step) / max(time.monotonic() - log_time, 1e-6)
                    log_time, log_step = time.monotonic(), step
                    print(
                        f"[learner step {step}] {rate:.1f} steps/s updates/env_step="
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
        agent = self.agent
        successes: list[bool] = []
        skipped = 0
        n_attempts = max(1, int(n_episodes))
        self._mode = "EVAL"
        print(
            f"[eval {tag}] task {self.cfg.task_id}: {self.task_name} | {n_attempts} episodes "
            "(D-pad left/right or L1/R1 skips the current episode)"
        )
        for i in range(n_attempts):
            state = self.env_worker.make_pool(train=False, task=self.cfg.task_id, n_active=1, trial_offset=i)[0]

            # Clear stale pad events and re-anchor the DMP for this episode.
            self.operator.reset()
            grip = -1.0
            base_action = None
            if self.base_policy is not None:
                self.base_policy.reset(self._pose_with_grip(state.obs, grip))
                base_action = self._base_action(state.obs, grip)
            obs_enc = self.obs_adapter.encode(state.obs, base_action=base_action)
            frames = [self.obs_adapter.display(state.obs)]
            skipped_this = False

            while not state.done:
                self._check_learner()
                event = self.operator.poll()
                if event.skip:
                    skipped_this = True
                    break

                residual = np.array(agent.sample_actions(obs_enc, seed=self._next_key(), argmax=True), np.float32)
                action = (
                    residual
                    if base_action is None
                    else combine_base_residual(
                        base_action,
                        residual,
                        residual_scale=self.cfg.residual_blend_scale,
                        min_base_weight=self.cfg.residual_min_base_weight,
                    )
                )
                g = _discretize_gripper(float(action[-1]))
                if g != 0.0:
                    grip = g
                action[-1] = grip if self.cfg.algorithm == "hgdagger" else g
                state.env.send_step(self._env_action(action, grip))
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

                next_base_action = self._base_action(obs, grip) if self.base_policy is not None else None
                obs_enc = self.obs_adapter.encode(obs, base_action=next_base_action)
                base_action = next_base_action
                frames.append(self.obs_adapter.display(obs))
                self._show(state.obs, f"EVAL L{self._grad_step} | {self.task_name}", _WHITE)

            if skipped_this:
                skipped += 1
                self._show(state.obs, f"EVAL SKIPPED L{self._grad_step}", _ORANGE)
                print(f"[eval {tag}] episode {i + 1}/{n_attempts} skipped by operator")
                self._save_video(frames, tag=f"{tag}_s{i}", success=False, skipped=True)
            else:
                successes.append(bool(state.success))
                self._save_video(frames, tag=f"{tag}_s{i}", success=bool(state.success))

        rate = float(np.mean(successes)) if successes else 0.0
        result = {
            "tag": tag,
            "iteration": iteration,
            "success_rate": rate,
            "n": len(successes),
            "skipped": skipped,
            "attempted": n_attempts,
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "training_seconds": round(self._elapsed_seconds(), 1),
        }
        print(f"[eval {tag}] success={rate:.2f} n={len(successes)} skipped={skipped}")
        with self._eval_log.open("a") as f:
            f.write(json.dumps(result) + "\n")
        self._mode = "IDLE"
        return result

    # -- plumbing ----------------------------------------------------------

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
