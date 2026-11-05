import argparse
import os
import signal
import struct
import time

import cv2
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation
from termcolor import cprint

from openpi.training.hil.transitions import TransitionRecorder
from wam.flir.multi_flir_manager import MultiFLIRManager
from wam.inference.hdf5_recorder import HDF5Recorder
from wam.inference.hil_policy import HilPolicy
from wam.inference.openpi_policy import KP_POS
from wam.inference.openpi_policy import KP_ROT
from wam.inference.openpi_policy import WAM_FORCE_LIMIT
from wam.inference.openpi_policy import WAM_TORQUE_LIMIT
from wam.inference.openpi_policy import OpenPIPolicy
from wam.inference.openpi_policy import _rotvec_delta_to_rot_error
from wam.inference.recorder_receiver import RecorderReceiver
from wam.inference.udp_handler import TeleopUDPHandler

np.set_printoptions(precision=4, suppress=True)

# ══════════════════════════════════════════════════════════════════════════════
# Editable knobs (kept at the top by request). Everything here is in
# SCREAMING_SNAKE_CASE so it is easy to find and tune on the robot.
# ══════════════════════════════════════════════════════════════════════════════

# ── intervention detection (from the UDP packet) ─────────────────────────────
# A human takeover is inferred from the follower's measured external torque
# (the operator pushes the arm) or from leader-arm joint velocity (the operator
# moves the leader). Either signal, held for INTERVENTION_MIN_STEPS packets,
# marks the step as intervened. All values are on the raw packet units.
INTERVENTION_FOLLOWER_EXT_TORQUE_THRESHOLD = 1.0  # N·m
INTERVENTION_LEADER_JOINT_VEL_THRESHOLD = 0.1  # rad/s
INTERVENTION_MIN_STEPS = 1  # consecutive packets above threshold
INTERVENTION_ACTION_DAMPING = 1.0  # scale on the derived human wrench

# ── RL action normalization ──────────────────────────────────────────────────
# The UDP action is a 7-dim wrench [fx, fy, fz, tx, ty, tz, gripper]. The SAC /
# BC agents work in [-1, 1], so each channel is divided by these scales before
# being stored in a transition and multiplied back before sending.
ACTION_FORCE_SCALE = 5.0  # matches WAM_FORCE_LIMIT
ACTION_TORQUE_SCALE = 5.0  # matches WAM_TORQUE_LIMIT
ACTION_GRIPPER_SCALE = 1.0
ACTION_SCALE = np.array(
    [ACTION_FORCE_SCALE] * 3 + [ACTION_TORQUE_SCALE] * 3 + [ACTION_GRIPPER_SCALE],
    np.float32,
)

# ── collection / deployment ──────────────────────────────────────────────────
HIL_DATA_DIR = "./interventions"  # online/ and demo/ transition pickles
HIL_POLICY_RELOAD_INTERVAL_S = 30.0  # pick up new learner checkpoints


def preprocess_image(
    img_bgr: np.ndarray, crop_scale: float = 0.9, out_size=(224, 224)
) -> np.ndarray:
    """Center-crop by area 'crop_scale' and resize to out_size using OpenCV. Keeps image in BGR."""
    H, W = img_bgr.shape[:2]
    s = float(crop_scale) ** 0.5
    crop_h, crop_w = int(round(H * s)), int(round(W * s))

    y0 = max((H - crop_h) // 2, 0)
    x0 = max((W - crop_w) // 2, 0)

    img_cropped = img_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    img_resized = cv2.resize(img_cropped, out_size, interpolation=cv2.INTER_AREA)

    return img_resized


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
        self.send_interval = 0.5 # send interpolated points for 3 seconds
        self.last_send_time = 0.0
        self.policy_name = policy
        # The hil policies are deployed from a trained checkpoint, which implies
        # inference even when --infer is omitted.
        self.doing_inference = infer or policy != "openpi"
        self.doing_recording = record

        # UDP Configuration
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port # not used
        self.DOF = dof

        # even though we listen to the follower we actually also get leader data, just dont do anything with it for now
        # also, leader data is just 0's if we dont start the leader
        self.follower_listener = RecorderReceiver(self.remote_ip, self.recv_port, self.DOF)
        self.follower_sender = TeleopUDPHandler(self.remote_ip, self.send_port, DOF=self.DOF)

        # FLIR setup
        camera_configs = {
            "wrist_image": wrist_cam_serial,
            "front_image": front_cam_serial,
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
                cprint(
                    f"[HIL] {self.policy.algorithm} policy loaded from {hil_checkpoint} "
                    f"(encoder={policy})",
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
        self._intervention_streak = 0
        self._last_policy_reload = time.time()

        # Set up recorder
        self.loop_state = "IDLE"  # States: IDLE, RECORDING, PENDING
        self.recorder = HDF5Recorder(save_dir="./dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        # Set up Joystick
        self.joy_fd = None
        self._init_joystick()

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Controls: [R] Start/Stop | [S] Save | [D] Discard", "cyan")
        cprint("Joystick Controls: [o] Start/Stop/Save  | [x] Discard", "cyan")

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

                time_msec, value, ev_type, number = struct.unpack('IhBB', event_data)

                # Remove the init event flag (0x80)
                ev_type &= ~0x80

                # ev_type == 1 means button event, value == 1 means pressed (down)
                if ev_type == 0x01 and value == 1:
                    if number == 1:  # 'o' button
                        self._handle_start_save_action()
                    elif number == 0:  # 'x' button
                        self._handle_discard_action()

        except BlockingIOError:
            pass
        except Exception as e:
            cprint(f"[SYSTEM] Error reading joystick: {e}", "red")


    def _on_key_press(self, key):
        """Asynchronous callback for keyboard events."""
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()

                if k == "r":
                    if self.loop_state == "IDLE" or self.loop_state == "RECORDING":
                        self._handle_start_save_action()
                elif k == "s":
                    if self.loop_state == "PENDING":
                        self._save_episode()
                elif k == "d":
                    self._handle_discard_action()
        except Exception:
            pass

    def _handle_start_save_action(self):
        """Unified logic for 'r' / 's' on keyboard and 'o' on joystick."""
        if self.loop_state == "IDLE":
            self.loop_state = "RECORDING"
            cprint("\n[RECORDER] 🔴 EPISODE STARTED", "red", attrs=["bold"])
        elif self.loop_state == "RECORDING":
            self.loop_state = "PENDING"
            cprint("\n[RECORDER] ⏸ EPISODE PAUSED", "yellow")
            cprint("Press [S] or 'o' to Save, [D] or 'x' to Discard.", "cyan")
        elif self.loop_state == "PENDING":
            self._save_episode()

    def _handle_discard_action(self):
        """Unified logic for 'd' on keyboard and 'x' on joystick."""
        if self.loop_state in ["RECORDING", "PENDING"]:
            self.recorder.clear()
            # Failed episodes are still valuable off-policy data: RLPD trains
            # on them with reward 0 and HG-Dagger keeps any human corrections
            # inside them. Only the HDF5 demo is discarded.
            if self.transition_recorder is not None:
                active_steps, total_steps = self.transition_recorder.stats()
                self.transition_recorder.flush(success=False)
                rate = active_steps / max(total_steps, 1)
                cprint(
                    f"[RECORDER] intervention rate: {rate:.1%} "
                    f"({active_steps}/{total_steps} steps)",
                    "cyan",
                )
            self._prev_state_dict = None
            self._intervention_streak = 0
            self.loop_state = "IDLE"
            cprint("\n[RECORDER] 🗑 Episode discarded. Press [R] or 'o' to start a new one.", "red")

    def _save_episode(self):
        """Handles packaging and saving the episode to disk."""
        ep_name = f"episode_{int(time.time())}_{self.episode_counter}"
       
        metadata = {
            "action_horizon": self.action_horizon,
            "dof": self.DOF
        }
        
        self.recorder.save_episode(ep_name, metadata)
        if self.transition_recorder is not None:
            active_steps, total_steps = self.transition_recorder.stats()
            self.transition_recorder.flush(success=True)
            rate = active_steps / max(total_steps, 1)
            cprint(
                f"[RECORDER] intervention rate: {rate:.1%} "
                f"({active_steps}/{total_steps} steps)",
                "cyan",
            )
        self._prev_state_dict = None
        self._intervention_streak = 0
        self.episode_counter += 1
        self.loop_state = "IDLE"
        cprint("[RECORDER] Ready for next episode. Press [R] or 'o' to start.", "cyan")

    def _detect_intervention(self, state_dict):
        """Whether this packet represents a human takeover.

        Uses only fields the existing RecorderReceiver packet already carries:
        the follower's measured external torque and the leader's joint velocity.
        The packet format is unchanged; this is pure post-processing.
        """
        torque = np.asarray(state_dict.get("follower_ext_torque", np.zeros(self.DOF)), np.float32)
        leader_jv = np.asarray(state_dict.get("leader_jv", np.zeros(self.DOF)), np.float32)
        active = (
            np.linalg.norm(torque) > INTERVENTION_FOLLOWER_EXT_TORQUE_THRESHOLD
            or np.linalg.norm(leader_jv) > INTERVENTION_LEADER_JOINT_VEL_THRESHOLD
        )
        self._intervention_streak = self._intervention_streak + 1 if active else 0
        return self._intervention_streak >= INTERVENTION_MIN_STEPS

    def _intervention_action(self, state_dict):
        """EE wrench equivalent to the human's realized correction.

        The human's commanded action is not transmitted as a wrench, but the
        follower's realized motion between two packets is. Convert that motion
        with the same proportional gains OpenPIPolicy uses in the other
        direction (``delta_to_wrench``), so the recorded action lives in the
        exact space the RL agent outputs.
        """
        if self._prev_state_dict is None:
            return np.zeros(7, np.float32)
        prev = self._prev_state_dict
        dpos = np.asarray(state_dict["follower_cart_pos"], np.float32) - np.asarray(
            prev["follower_cart_pos"], np.float32
        )
        force = np.clip(KP_POS * dpos, -WAM_FORCE_LIMIT, WAM_FORCE_LIMIT)
        # Quaternion delta prev -> current, expressed as the same rotation
        # error vector used by delta_to_wrench.
        q_prev = Rotation.from_quat(np.roll(np.asarray(prev["follower_cart_rot"], np.float32), -1))
        q_cur = Rotation.from_quat(np.roll(np.asarray(state_dict["follower_cart_rot"], np.float32), -1))
        d_rotvec = (q_cur * q_prev.inv()).as_rotvec()
        torque = np.clip(
            KP_ROT * _rotvec_delta_to_rot_error(d_rotvec), -WAM_TORQUE_LIMIT, WAM_TORQUE_LIMIT
        )
        gripper = float(state_dict.get("gripper_pos", 0.0))
        return INTERVENTION_ACTION_DAMPING * np.concatenate(
            [force, torque, [np.clip(gripper, -1.0, 1.0)]]
        ).astype(np.float32)

    def _maybe_reload_policy(self):
        """Pick up a newer learner checkpoint without restarting the actor."""
        if self.policy is None or self.policy_name == "openpi":
            return
        if time.time() - self._last_policy_reload < HIL_POLICY_RELOAD_INTERVAL_S:
            return
        self._last_policy_reload = time.time()
        try:
            self.policy = self.policy.reload()
            cprint("[HIL] policy checkpoint reloaded", "green")
        except Exception as e:
            cprint(f"[HIL] policy reload skipped: {e}", "yellow")

    def _read_images(self):
        status, raw_frames = self.camera_manager.read_all()
        if not status:
            return False, None

        # TODO: we really shouldn't be doing this here, it would make more sense
        # for the policy to preprocess itself.
        processed_frames = {}
        for name, img_bgr in raw_frames.items():
            proc_img = preprocess_image(img_bgr)
            processed_frames[name] = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

        return True, processed_frames

    def _read_state(self):
        if self.offline:
            jp_state = np.array([0, 0.2, 1.5, -1.5, 0, 0, 0])
            state_dict = {"follower_jp": jp_state}
            status = True
        else:
            state_dict = self.follower_listener.receive_latest_data()
            status = state_dict is not None

        if self.debug:
            # cprint(f"Robot State [jp]: {np.round(state_dict['follower_state']['jp'], 3)}", "yellow")
            # cprint(f"Robot State [jv]: {np.round(state_dict['follower_state']['jv'], 3)}", "yellow")
            # cprint(f"Robot State [ext_torque]: {np.round(state_dict['follower_state']['ext_torque'], 3)}", "yellow")
            # cprint(f"Robot State [meas_torque]: {np.round(state_dict['follower_state']['meas_torque'], 3)}", "yellow")
            # cprint(f"Robot State [cart_pos]: {np.round(state_dict['follower_state']['cart_pos'], 3)}", "yellow")
            # cprint(f"Robot State [cart_rot]: {np.round(state_dict['follower_state']['cart_rot'], 3)}", "yellow")
            # cprint(f"Robot State [gripper]: {np.round(state_dict['follower_state']['gripper'], 3)}", "yellow")
            # cprint(f"Robot State [timestamp_ns]: {state_dict['follower_state']['timestamp_ns']}", "yellow")
            pass

        return status, state_dict

    def _update_displays(self, front_image, wrist_image):
        debug_front = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
        debug_wrist = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
        try:
            pass
            # cv2.imshow(
            #     "Wrist",
            #     cv2.resize(
            #         debug_wrist, (224 * self.display_scale, 224 * self.display_scale)
            #     ),
            # )
            # cv2.imshow(
            #     "Front",
            #     cv2.resize(
            #         debug_front, (224 * self.display_scale, 224 * self.display_scale)
            #     ),
            # )
        except Exception as e:
            print(e)

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

            loop_delay = 1 / self.loop_hz
            waiting_counter = 0
            while self.system_running:
                loop_start_time = time.time()

                self._poll_joystick()

                # Read images
                img_status, image_dict = self._read_images()
                state_status, state_dict = self._read_state()

                if not state_status:
                    if waiting_counter == 10:
                        cprint("waiting for messages", "yellow")
                        waiting_counter = 0
                    waiting_counter += 1
                    continue

                if self.debug:
                    cv2.waitKey(1)

                if not img_status or not state_status:
                    cprint("State or images not received yet, waiting...", "red")
                    time.sleep(loop_delay)
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
                # if (loop_start_time - self.last_send_time) >= self.send_interval:
                if self.doing_inference and self.loop_state == "RECORDING":
                    if self.policy_name == "openpi":
                        action_chunk = self.policy.infer(obs)
                        self.follower_sender.send_action_chunk(action_chunk)
                    else:
                        self._maybe_reload_policy()
                        # The packet is processed BEFORE the policy acts so the
                        # same state produces both the intervention label and
                        # the encoder observation for that transition. The
                        # detection is policy-type agnostic; the loaded encoder
                        # decides only what `obs_enc` contains.
                        intervened = self._detect_intervention(state_dict)
                        action_wrench, action_norm, obs_enc = self.policy.act(obs)
                        if intervened:
                            # Human takeover: record the human correction (the
                            # DAgger target) but do NOT command it back — the
                            # operator is physically in control, and echoing
                            # their realized motion as a wrench would push in
                            # the same direction. The robot side holds/teleops
                            # while the takeover is active.
                            action_wrench = self._intervention_action(state_dict)
                            action_norm = np.clip(action_wrench / ACTION_SCALE, -1.0, 1.0)
                        else:
                            self.follower_sender.send_action_chunk(
                                np.tile(action_wrench[None], (self.action_horizon, 1))
                            )
                        if self.transition_recorder is not None:
                            self.transition_recorder.add_step(obs_enc, action_norm, intervened=intervened)

                # Keep the last packet for the next step's intervention delta.
                self._prev_state_dict = state_dict

                # sleep, accounting for the time the loop already took to try and keep the desired frequency
                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0.0, loop_delay - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
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
