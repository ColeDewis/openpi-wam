import argparse
import collections
import os
import signal
import struct
import time

import cv2
import numpy as np
from termcolor import cprint

from openpi.training.hil.transitions import TransitionRecorder
from wam.flir.multi_flir_manager import MultiFLIRManager
from wam.inference.hdf5_recorder import HDF5Recorder
from wam.inference.hil_encoder import discrete_gripper
from wam.inference.hil_policy import HilPolicy
from wam.inference.openpi_policy import OpenPIPolicy
from wam.inference.recorder_receiver import RecorderReceiver
from wam.inference.udp_handler import TeleopUDPHandler

np.set_printoptions(precision=4, suppress=True)

# ══════════════════════════════════════════════════════════════════════════════
# Editable knobs (kept at the top by request). Everything here is in
# SCREAMING_SNAKE_CASE so it is easy to find and tune on the robot.
# ══════════════════════════════════════════════════════════════════════════════

# ── intervention detection (from the UDP packet) ─────────────────────────────
# The controller reports a per-joint policy torque scale. A human takeover is
# any scale above this threshold; tune it on the robot.
INTERVENTION_POLICY_TORQUE_SCALE_THRESHOLD = 0.5

# ── RL action normalization ──────────────────────────────────────────────────
# The RL action is 7 delta joint positions + a discrete gripper. The SAC / BC
# agents work in [-1, 1], so each channel is divided by these scales before
# being stored in a transition and multiplied back before sending.
ACTION_JP_SCALE = 0.05  # rad per normalized step, matches the simulator
ACTION_GRIPPER_SCALE = 1.0
ACTION_SCALE = np.array([ACTION_JP_SCALE] * 7 + [ACTION_GRIPPER_SCALE], np.float32)
ACTION_DIM = len(ACTION_SCALE)

# ── collection / deployment ──────────────────────────────────────────────────
HIL_DATA_DIR = "./interventions"  # online/ and demo/ transition pickles
HIL_POLICY_RELOAD_INTERVAL_S = 30.0  # pick up new learner checkpoints
MAX_STATE_IMAGE_DT_NS = 100_000_000  # 100 ms state/image alignment tolerance


def precise_sleep(dt: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Sleep most of ``dt``, then spin for the remainder to reduce jitter."""
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Wait until ``t_end`` using a hybrid sleep/spin."""
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def preprocess_image(img_bgr: np.ndarray, out_size=(128, 128)) -> np.ndarray:
    """Resize with padding, matching the simulator's ``resize_with_pad``.

    OpenCV on the CPU is faster than a torch GPU resize for these small frames:
    it avoids the host-to-device copy, kernel launch, and device-to-host copy.
    """
    h, w = img_bgr.shape[:2]
    target_h, target_w = out_size
    ratio = max(w / target_w, h / target_h)
    new_h = int(h / ratio)
    new_w = int(w / ratio)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    return canvas


class PiZeroTeleop:
    def __init__(
        self,
        remote_ip: str,
        send_port: int,
        recv_port: int,
        dof: int,
        wrist_cam_serial: str,
        front_cam_serial: str,
        checkpoint_path: str,
        model_config: str,
        action_horizon: int = 5,
        loop_hz: int = 30,
        display_scale: int = 3,
        debug: bool = False,
        offline: bool = False,
        infer: bool = False,
        record: bool = False,
        policy: str = "openpi",
        hil_checkpoint: str = "",
        hil_data_dir: str = HIL_DATA_DIR,
    ):

        # System config
        self.debug = debug
        self.offline = offline
        self.display_scale = display_scale
        self.action_horizon = action_horizon
        self.loop_hz = loop_hz
        self.dt = 1.0 / self.loop_hz
        self.policy_name = policy
        self.image_size = (224, 224)
        self.action_size = 7 if policy == "openpi" else ACTION_DIM
        # The hil policies are deployed from a trained checkpoint, which implies
        # inference even when --infer is omitted.
        self.doing_inference = infer or policy != "openpi"
        self.doing_recording = record

        # UDP Configuration
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.DOF = dof

        # even though we listen to the follower we actually also get leader data, just dont do anything with it for now
        # also, leader data is just 0's if we dont start the leader
        self.follower_listener = RecorderReceiver(self.remote_ip, self.recv_port, self.DOF)
        self.follower_sender = TeleopUDPHandler(
            self.remote_ip,
            self.send_port,
            DOF=self.DOF,
            action_size=self.action_size,
        )

        # FLIR setup
        camera_configs = {
            "wrist_image": {"serial": wrist_cam_serial},
            "front_image": {"serial": front_cam_serial},
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()


        # Policy Configuration
        self.policy = None
        if self.doing_inference:
            if policy == "openpi":
                self.policy = OpenPIPolicy(checkpoint_path, model_config, "libero", debug=self.debug, dof=self.DOF)
            elif policy in ("hilserl", "rltoken", "timm"):
                self.policy = HilPolicy(hil_checkpoint, ACTION_SCALE)
                if self.policy.encoder_config.kind != policy:
                    raise ValueError(
                        f"--policy {policy} but {hil_checkpoint}/config.json was trained with "
                        f"encoder kind {self.policy.encoder_config.kind!r}"
                    )
                self.image_size = self.policy.image_size
                cprint(
                    f"[HIL] {self.policy.algorithm} policy loaded from {hil_checkpoint} "
                    f"(encoder={policy}, image_size={self.image_size})",
                    "green",
                )
            else:
                raise ValueError(f"unknown policy {policy!r}")

        # Online transition collection (RLPD: online+demo, HG-Dagger: demo only).
        # The recorder reads the algorithm from the loaded policy checkpoint.
        self.transition_recorder = None
        if self.policy is not None and policy != "openpi":
            self.transition_recorder = TransitionRecorder(hil_data_dir, self.policy.algorithm)
            cprint(
                f"[HIL] recording transitions to {hil_data_dir} "
                f"({'online+demo' if self.policy.algorithm == 'rlpd' else 'demo only'})",
                "cyan",
            )
        self._prev_state_dict = None
        self._last_policy_reload = time.time()
        self._state_buffer = collections.deque(maxlen=200)
        self._last_images = {}
        self._last_image_ts = {}

        # Set up recorder
        self.loop_state = "IDLE"  # States: IDLE, RECORDING, PENDING
        self.recorder = HDF5Recorder(save_dir="./dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()

        cprint("Initialization complete. Running teleop loop...", "green")
        cprint(
            "Joystick Controls: OPTIONS ('o') Start/Pause/Save | SHARE ('x') Discard",
            "cyan",
        )

    def _init_joystick(self):
        """Initializes the joystick file descriptor in non-blocking mode."""
        try:
            self.joy_fd = os.open("/dev/input/js0", os.O_RDONLY | os.O_NONBLOCK)
            cprint("[SYSTEM] Successfully opened joystick /dev/input/js0", "green")
        except OSError:
            cprint("[SYSTEM] Could not open joystick /dev/input/js0. Continuing without joystick.", "yellow")

    def _poll_joystick(self):
        """Reads non-blocking events from the joystick."""
        if self.joy_fd is None:
            return

        try:
            while True:
                event_data = os.read(self.joy_fd, 8)
                if not event_data:
                    break

                time_msec, value, ev_type, number = struct.unpack("IhBB", event_data)

                # Remove the init event flag (0x80)
                ev_type &= ~0x80

                # ev_type == 1 means button event, value == 1 means pressed (down)
                if ev_type == 0x01 and value == 1:
                    if number == 1:  # OPTIONS ('o')
                        self._handle_start_save_action()
                    elif number == 0:  # SHARE ('x')
                        self._handle_discard_action()

        except BlockingIOError:
            pass
        except Exception as e:
            cprint(f"[SYSTEM] Error reading joystick: {e}", "red")

    def _handle_start_save_action(self):
        """OPTIONS: start, pause, or save the episode."""
        if self.loop_state == "IDLE":
            self.loop_state = "RECORDING"
            cprint("\n[RECORDER] EPISODE STARTED", "red", attrs=["bold"])
        elif self.loop_state == "RECORDING":
            self.loop_state = "PENDING"
            cprint("\n[RECORDER] EPISODE PAUSED", "yellow")
            cprint("Press OPTIONS ('o') to Save, SHARE ('x') to Discard.", "cyan")
        elif self.loop_state == "PENDING":
            self._save_episode()

    def _handle_discard_action(self):
        """SHARE: discard the episode."""
        if self.loop_state in ["RECORDING", "PENDING"]:
            self.recorder.clear()
            # Failed episodes are still valuable off-policy data: RLPD trains
            # on them with reward 0 and HG-Dagger keeps any human corrections
            # inside them. Only the HDF5 demo is discarded.
            if self.transition_recorder is not None:
                intervened_steps, total_steps = self.transition_recorder.stats()
                self.transition_recorder.flush(success=False)
                rate = intervened_steps / max(total_steps, 1)
                cprint(
                    f"[RECORDER] intervention rate: {rate:.1%} "
                    f"({intervened_steps}/{total_steps} steps)",
                    "cyan",
                )
            self._prev_state_dict = None
            self.loop_state = "IDLE"
            cprint(
                "\n[RECORDER] Episode discarded. Press OPTIONS ('o') to start.",
                "red",
            )

    def _save_episode(self):
        """Handles packaging and saving the episode to disk."""
        ep_name = f"episode_{int(time.time())}_{self.episode_counter}"
        metadata = {"action_horizon": self.action_horizon, "dof": self.DOF}
        self.recorder.save_episode(ep_name, metadata)
        if self.transition_recorder is not None:
            intervened_steps, total_steps = self.transition_recorder.stats()
            self.transition_recorder.flush(success=True)
            rate = intervened_steps / max(total_steps, 1)
            cprint(
                f"[RECORDER] intervention rate: {rate:.1%} "
                f"({intervened_steps}/{total_steps} steps)",
                "cyan",
            )
        self._prev_state_dict = None
        self.episode_counter += 1
        self.loop_state = "IDLE"
        cprint(
            "[RECORDER] Ready for next episode. Press OPTIONS ('o') to start.",
            "cyan",
        )

    def _detect_intervention(self, state_dict):
        """Whether the policy torque scale indicates a human takeover."""
        scale = np.asarray(
            state_dict.get("policy_torque_scale", np.zeros(self.DOF)),
            np.float32,
        )
        return bool(
            np.max(np.abs(scale)) > INTERVENTION_POLICY_TORQUE_SCALE_THRESHOLD
        )

    def _intervention_action(self, state_dict):
        """Delta joint position equivalent to the human's realized correction."""
        if self._prev_state_dict is None:
            return np.zeros(ACTION_DIM, np.float32)
        prev_jp = np.asarray(self._prev_state_dict["follower_jp"], np.float32)
        jp = np.asarray(state_dict["follower_jp"], np.float32)
        delta_jp = jp - prev_jp
        gripper = discrete_gripper(float(state_dict.get("gripper_pos", 0.0)))
        return np.concatenate([delta_jp, [gripper]]).astype(np.float32)

    def _maybe_reload_policy(self):
        """Pick up a newer learner checkpoint without restarting the actor."""
        if self.policy is None or self.policy_name == "openpi":
            return
        if time.time() - self._last_policy_reload < HIL_POLICY_RELOAD_INTERVAL_S:
            return
        self._last_policy_reload = time.time()
        try:
            self.policy.agent = self.policy.reload()
            cprint("[HIL] policy checkpoint reloaded", "green")
        except Exception as e:
            cprint(f"[HIL] policy reload skipped: {e}", "yellow")

    def _read_images(self):
        new_frames = self.camera_manager.read_all()
        for name, (img_bgr, timestamp_ns) in new_frames.items():
            proc_img = preprocess_image(img_bgr, out_size=self.image_size)
            self._last_images[name] = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
            self._last_image_ts[name] = timestamp_ns

        if len(self._last_images) < len(self.camera_manager.streams):
            return False, None, None

        image_ts = max(self._last_image_ts.values())
        return True, dict(self._last_images), image_ts

    def _read_state(self):
        if self.offline:
            state_dict = {
                "follower_jp": np.array([0, 0.2, 1.5, -1.5, 0, 0, 0], np.float32),
                "gripper_pos": 0.0,
                "policy_torque_scale": np.zeros(self.DOF, np.float32),
                "timestamp_ns": time.time_ns(),
            }
            packets = [state_dict]
        else:
            packets = self.follower_listener.receive_all_new()

        for state_dict in packets:
            self._state_buffer.append(state_dict)
        return bool(packets) or bool(self._state_buffer)

    def _match_state(self, image_ts):
        """Pick the low-dim packet closest to the camera timestamp."""
        if not self._state_buffer:
            return None
        state_dict = min(
            self._state_buffer,
            key=lambda state: abs(int(state["timestamp_ns"]) - image_ts),
        )
        if abs(int(state_dict["timestamp_ns"]) - image_ts) > MAX_STATE_IMAGE_DT_NS:
            return None
        return state_dict

    def _update_displays(self, front_image, wrist_image):
        return cv2.waitKey(1) & 0xFF != ord("q")

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.camera_manager.stop_all()
        self.follower_sender.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass
        try:
            # Explicitly target the named windows
            cv2.destroyWindow("Wrist")
            cv2.destroyWindow("Front")
            cv2.waitKey(1)
        except Exception:
            pass
        time.sleep(2)

        os._exit(0)

    def run(self):
        try:
            self.system_running = True

            def force_shutdown(signum, frame):
                cprint(
                    "\n[SYSTEM] Ctrl+C detected! Forcing main loop shutdown...", "red"
                )
                self.system_running = False

            signal.signal(signal.SIGINT, force_shutdown)

            waiting_counter = 0
            t_start = time.monotonic()
            iter_idx = 0
            while self.system_running:
                t_cycle_end = t_start + (iter_idx + 1) * self.dt

                self._poll_joystick()

                # Read images and drain every new low-dim packet.
                img_status, image_dict, image_ts = self._read_images()
                state_status = self._read_state()

                if not state_status:
                    if waiting_counter == 10:
                        cprint("waiting for messages", "yellow")
                        waiting_counter = 0
                    waiting_counter += 1
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    continue

                if self.debug:
                    cv2.waitKey(1)

                if not img_status:
                    cprint("State or images not received yet, waiting...", "red")
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    continue

                state_dict = self._match_state(image_ts)
                if state_dict is None:
                    cprint("state/image timestamps not aligned, waiting...", "red")
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                    continue

                obs = image_dict.copy()
                obs["low_dim"] = state_dict

                # Update display windows if debugging
                if self.debug and not self._update_displays(
                    obs["front_image"], obs["wrist_image"]
                ):
                    cprint("Quitting...", "red")
                    break

                # Recording
                if self.doing_recording and self.loop_state == "RECORDING":
                    self.recorder.add_step(obs)

                # Infer + send to WAM
                if self.doing_inference and self.loop_state == "RECORDING":
                    if self.policy_name == "openpi":
                        action_chunk = self.policy.infer(obs)
                        self.follower_sender.send_action_chunk(action_chunk)
                    else:
                        self._maybe_reload_policy()
                        # The packet is processed BEFORE the policy acts so the
                        # same state produces both the intervention label and
                        # the encoder observation for that transition.
                        intervened = self._detect_intervention(state_dict)
                        action_delta_jp, action_norm, obs_enc = self.policy.act(obs)
                        if intervened:
                            # Human takeover: record the realized correction but
                            # do NOT send it back; the operator is in control.
                            action_delta_jp = self._intervention_action(state_dict)
                            action_norm = np.clip(
                                action_delta_jp / ACTION_SCALE, -1.0, 1.0
                            )
                        else:
                            self.follower_sender.send_action_chunk(
                                np.tile(
                                    action_delta_jp[None],
                                    (self.action_horizon, 1),
                                )
                            )
                        if self.transition_recorder is not None:
                            self.transition_recorder.add_step(
                                obs_enc,
                                action_norm,
                                intervened=intervened,
                            )

                # Keep the last packet for the next step's intervention delta.
                self._prev_state_dict = state_dict

                precise_wait(t_cycle_end)
                iter_idx += 1

        except Exception:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PiZero Teleop Node")

    # Network config
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Remote UDP IP")
    parser.add_argument("--send_port", type=int, default=6666, help="UDP Send Port")
    parser.add_argument("--recv_port", type=int, default=6554, help="UDP Receive Port")

    # Robot config
    parser.add_argument("--dof", type=int, default=7, help="Degrees of Freedom")

    # Camera config
    parser.add_argument(
        "--wrist_cam",
        type=str,
        default="18475182",
        help="Serial number for wrist camera",
    )
    parser.add_argument(
        "--front_cam",
        type=str,
        default="18475176",
        help="Serial number for front camera",
    )

    # Model config
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/10tb/serg/openpi-wam/checkpoints/haptic_wam/haptic_wam_pi05_freeze_long/20000",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="haptic_wam_pi05", help="Policy config name. see config.py"
    )

    # System config
    parser.add_argument( # does not really work
        "--debug",
        action="store_true",
        help="Display OpenCV windows and extra print statements",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run without UDP robot connection (mocks hardware state)",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference if true",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record data if true",
    )

    # Policy selection. `openpi` is the existing VLA path and the default;
    # the other three are RLPD / HG-Dagger trained agents whose encoder is
    # recorded in the checkpoint's config.json.
    parser.add_argument(
        "--policy",
        type=str,
        default="openpi",
        choices=["openpi", "hilserl", "rltoken", "timm"],
        help="openpi = VLA; hilserl/rltoken/timm = RLPD/HG-Dagger encoder variant",
    )
    parser.add_argument(
        "--hil_checkpoint",
        type=str,
        default="",
        help="RLPD/HG-Dagger run directory (contains config.json + checkpoint_<step>)",
    )
    parser.add_argument(
        "--hil_data_dir",
        type=str,
        default=HIL_DATA_DIR,
        help="Where online/ and demo/ transition pickles are written",
    )

    args = parser.parse_args()

    teleop = PiZeroTeleop(
        remote_ip=args.ip,
        send_port=args.send_port,
        recv_port=args.recv_port,
        dof=args.dof,
        wrist_cam_serial=args.wrist_cam,
        front_cam_serial=args.front_cam,
        checkpoint_path=args.checkpoint,
        model_config=args.config,
        debug=args.debug,
        offline=args.offline,
        infer=args.infer,
        record=args.record,
        policy=args.policy,
        hil_checkpoint=args.hil_checkpoint,
        hil_data_dir=args.hil_data_dir,
    )

    teleop.run()
