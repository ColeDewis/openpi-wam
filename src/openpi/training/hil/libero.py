"""LIBERO actor/learner for RLPD and HG-Dagger.

The actor runs in the main thread and steps LIBERO. The learner runs in a
background thread and publishes weights every ``steps_per_update`` steps, as in
hil-serl. Human interventions come from the browser gamepad page served by the
optional MJPEG stream.

Record demos (pure teleop, one pickle per successful episode):

    uv run --extra libero scripts/train_hil_libero.py --record-demos 20 \
        --demo-dir /home/serg/projects/openpi-wam/demos/goal_t1 --suite libero_goal --task-id 1

Train (resumes from the latest checkpoint in ``--checkpoint-dir``):

    uv run --extra libero scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1

Controls: OPTIONS starts an episode / marks success, SHARE marks failure,
left stick moves x/y, triangle/cross moves z, L2/R2 opens/closes the gripper,
and PS hands control back to the policy. The right stick and square/circle
control rotation only with ``--rotation``.
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
    """Raw LIBERO obs -> the encoder's ``{front_image, wrist_image, state}`` dict."""

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
        # (T=1, H, W, C) and (T=1, D): the agents stack a time axis even when T=1.
        return {
            "front_image": front[None],
            "wrist_image": wrist[None],
            "state": self.state(obs)[None],
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
 <div id="status">online episodes: 0 | demo episodes: 0 | collected episodes: 0 | training: 00:00:00</div>
 <div id="grid">
  <div><b>OPTIONS</b> start episode / mark <b>SUCCESS</b></div>
  <div><b>SHARE</b> mark <b>FAILURE</b></div>
  <div><b>Left stick</b> x (up/down), y (left/right)</div>
  <div><b>Right stick</b> pitch / yaw <span class="dim">(only with --rotation)</span></div>
  <div><b>&#9651; / &#10005;</b> z up / z down</div>
  <div><b>&#9633; / &#9675;</b> roll left / right <span class="dim">(only with --rotation)</span></div>
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
      "online episodes: " + s.online_episodes + " | demo episodes: " + s.demo_episodes +
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


class GamepadOperator:
    """PS4 controller state, fed by the browser page via ``apply_pad``.

    Intervention is latched: the first stick/trigger/button input takes control
    and the policy stays off until PS is pressed (or ``handback_s`` of idle input
    elapses, if > 0) or ``reset()`` is called at episode start.
    """

    CROSS, CIRCLE, SQUARE, TRIANGLE = 0, 1, 2, 3
    L2, R2 = 6, 7
    SHARE, OPTIONS = 8, 9
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

    # Pure teleop: save N successful demos to ``demo_dir`` and exit.
    record_demos: int = 0
    demo_dir: str = "demos/hil_libero"
    demo_path: tuple[str, ...] = ()

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
            image_size=self.image_size,
        )

    @property
    def proprio_dim(self) -> int:
        return 8 + (9 if self.include_velocity else 0)

    @property
    def action_dim(self) -> int:
        return 7 if self.rotation else 4


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class HilLiberoTrainer:
    """Actor in the main thread, learner in a background thread."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.teleop_only = cfg.record_demos > 0
        self.obs_adapter = LiberoObsAdapter(cfg.image_size, include_velocity=cfg.include_velocity)
        scales = (
            *([cfg.max_step_m / _OSC_MAX_M] * 3),
            *([cfg.max_step_rad / _OSC_MAX_RAD] * 3),
            1.0,
        )
        self.act_adapter = ActionAdapter(kind="delta_pose", scales=tuple(float(x) for x in scales))
        self.operator = GamepadOperator(cfg.takeover_step, handback_s=cfg.handback_s, rotation=cfg.rotation)
        print(
            f"[action] {cfg.action_dim}-dim agent action ({'6-DoF' if cfg.rotation else 'xyz'} + gripper); "
            f"normalized +-1 -> {cfg.max_step_m} m / {cfg.max_step_rad} rad per step"
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
        )
        self.env_worker = _envs.EnvWorker(
            env_cfg,
            PaligemmaTokenizer(max_len=48),
            action_dim=7,
            state_norm=None,
            action_norm=None,
        )
        if self.teleop_only:
            return

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
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "training": self._elapsed(),
            "mode": self._mode,
            "learner_step": self._grad_step,
        }

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
        out = Path(self.cfg.demo_dir)
        saved = attempts = 0
        print(
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
                self.evaluate(self.cfg.eval_episodes, iteration=0, tag="eval_only")
                return
            self._start_learner()
            while self._episodes < self.cfg.max_episodes and self._env_steps < self.cfg.max_env_steps:
                episode, success = self._run_episode()
                self._episodes += 1
                self._collected_episodes += 1
                self._flush(episode, success=success)
                if self._should_eval():
                    self.evaluate(
                        self.cfg.eval_episodes,
                        iteration=self._collected_episodes,
                        tag=f"ep{self._collected_episodes}",
                    )
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
        obs_enc = self.obs_adapter.encode(state.obs)
        episode: list[dict] = []
        grip = -1.0  # commanded gripper state (LIBERO resets open)
        manual_success = manual_failure = False

        self.operator.reset()
        self._mode = "IDLE"
        self._show(state.obs, "IDLE: press OPTIONS", _WHITE)
        if not (cfg.auto_start or self._stream is None):
            print("\n[episode] Press OPTIONS to start.")
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
                action = cmd if cfg.rotation else cmd[[0, 1, 2, 6]]
                intervened = moving  # idle hold frames are not demos
            else:
                action = np.array(
                    self.actor_agent.sample_actions(obs_enc, seed=self._next_key(), argmax=False),
                    np.float32,
                )
                intervened = False

            action[-1] = _discretize_gripper(float(action[-1]))
            g = float(action[-1])
            penalty = cfg.grasp_penalty if g not in (0.0, grip) else 0.0
            if g != 0.0:
                grip = g
            if cfg.algorithm == "hgdagger" and not teleop_only:
                action[-1] = grip  # BC regresses a persistent open/closed label

            state.env.send_step(self._env_action(action, grip))
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

            next_enc = self.obs_adapter.encode(obs)
            episode.append(
                {
                    "observations": obs_enc,
                    "actions": action,
                    "next_observations": next_enc,
                    "intervened": intervened,
                    "grasp_penalty": penalty,
                }
            )
            obs_enc = next_enc
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
        self._mode = "EVAL"
        for i in range(max(1, int(n_episodes))):
            state = self.env_worker.make_pool(
                train=False, task=self.cfg.task_id, n_active=1, trial_offset=i
            )[0]
            obs_enc = self.obs_adapter.encode(state.obs)
            frames = [self.obs_adapter.display(state.obs)]
            grip = -1.0
            while not state.done:
                self._check_learner()
                action = np.array(agent.sample_actions(obs_enc, seed=self._next_key(), argmax=True), np.float32)
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
                obs_enc = self.obs_adapter.encode(obs)
                frames.append(self.obs_adapter.display(obs))
                self._show(state.obs, f"EVAL L{self._grad_step}", _WHITE)
            successes.append(bool(state.success))
            self._save_video(frames, tag=f"{tag}_s{i}", success=bool(state.success))

        rate = float(np.mean(successes)) if successes else 0.0
        result = {
            "tag": tag,
            "iteration": iteration,
            "success_rate": rate,
            "n": len(successes),
            "online_episodes": self._online_episodes,
            "demo_episodes": self._demo_episodes,
            "collected_episodes": self._collected_episodes,
            "training_seconds": round(self._elapsed_seconds(), 1),
        }
        print(f"[eval {tag}] success={rate:.2f} n={len(successes)}")
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
        self._stop.set()
        if self._learner_thread is not None:
            self._learner_thread.join(timeout=10)
        self.env_worker.close()
        if self._stream is not None:
            self._stream.close()
        self._save_stats()
