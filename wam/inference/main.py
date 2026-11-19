"""Run one base policy and one residual policy on the real robot.

Example:
    uv run python -m wam.inference.main \
        --base_policy dmp \
        --base_trajectory /path/to/episode.hdf5 \
        --res_policy hilserl \
        --res_checkpoint /path/to/checkpoint_dir \
        --res_data_dir ./interventions
"""

import argparse
import collections
import os
import signal
import struct
import time

import cv2
import numpy as np
from termcolor import cprint

from wam.flir.multi_flir_manager import MultiFLIRManager
from wam.inference.hdf5_recorder import HDF5Recorder
from wam.inference.hil_policy import HilPolicy
from wam.inference.openpi_policy import OpenPIPolicy
from wam.inference.policy import PolicyTemplate
from wam.inference.recorder_receiver import RecorderReceiver
from wam.inference.udp_handler import TeleopUDPHandler
from wam.inference.inference_utils import precise_wait, preprocess_image

MAX_STATE_IMAGE_DT_NS = 100_000_000  # 100 ms state/image alignment tolerance


class PiZeroTeleop:
    def __init__(
        self,
        remote_ip: str,
        recv_port: int,
        dof: int,
        wrist_cam_serial: str,
        front_cam_serial: str,
        base_policy: PolicyTemplate | None,
        base_ports: tuple[int, int],
        res_policy: PolicyTemplate | None,
        res_ports: tuple[int, int],
        loop_hz: int = 10,
        display_scale: int = 3,
        debug: bool = False,
        offline: bool = False,
    ):
        self.debug = debug
        self.offline = offline
        self.display_scale = display_scale
        self.loop_hz = loop_hz
        self.dt = 1.0 / self.loop_hz
        self.DOF = dof
        self.remote_ip = remote_ip

        self.base_policy = base_policy
        self.res_policy = res_policy
        self.policies = [p for p in (base_policy, res_policy) if p is not None]
        self.image_size = next(
            (p.image_size for p in self.policies if p.needs_images),
            (128, 128),
        )

        camera_configs = {
            "wrist_image": {"serial": wrist_cam_serial},
            "front_image": {"serial": front_cam_serial},
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()

        # We listen to the follower but also receive leader data and policy
        # targets; the base policy's ``policy_jp`` reaches the residual encoder.
        self.follower_listener = RecorderReceiver(remote_ip, recv_port, dof)

        self.base_leader_sender = self._make_sender(base_policy, base_ports[0])
        self.base_follower_sender = self._make_sender(base_policy, base_ports[1])
        self.res_leader_sender = self._make_sender(res_policy, res_ports[0])
        self.res_follower_sender = self._make_sender(res_policy, res_ports[1])

        self._state_buffer = collections.deque(maxlen=200)
        self._last_images = {}
        self._last_image_ts = {}

        self.loop_state = "IDLE"
        self.recorder = HDF5Recorder(save_dir="./dataset")
        self.episode_counter = self.recorder.get_next_episode_index()

        self.joy_fd = None
        self._init_joystick()

        cprint("Initialization complete. Running teleop loop...", "green")
        cprint(
            "Joystick Controls: OPTIONS ('o') start | SHARE ('x') success | DOWN fail",
            "cyan",
        )

    def _make_sender(self, policy: PolicyTemplate | None, port: int):
        if policy is None:
            return None
        return TeleopUDPHandler(
            self.remote_ip,
            port,
            horizon=policy.horizon,
            DOF=self.DOF,
            action_size=policy.action_size,
        )

    def _init_joystick(self):
        """Initializes the joystick file descriptor in non-blocking mode."""
        try:
            self.joy_fd = os.open("/dev/input/js0", os.O_RDONLY | os.O_NONBLOCK)
            cprint("[SYSTEM] Successfully opened joystick /dev/input/js0", "green")
        except OSError:
            cprint(
                "[SYSTEM] Could not open joystick /dev/input/js0. Continuing without joystick.",
                "yellow",
            )

    def _poll_joystick(self):
        """Reads non-blocking events from the joystick."""
        if self.joy_fd is None:
            return

        try:
            while True:
                event_data = os.read(self.joy_fd, 8)
                if not event_data:
                    break

                _, value, ev_type, number = struct.unpack("IhBB", event_data)
                ev_type &= ~0x80  # remove the init event flag

                if ev_type == 0x01 and value == 1:
                    if number == 1:  # OPTIONS ('o')
                        self._handle_start()
                    elif number == 0:  # SHARE ('x')
                        self._handle_success()
                    elif number == 9:  # D-pad down
                        self._handle_fail()

        except BlockingIOError:
            pass
        except Exception as e:
            cprint(f"[SYSTEM] Error reading joystick: {e}", "red")

    def _handle_start(self):
        """OPTIONS: begin recording a new episode."""
        if self.loop_state != "IDLE":
            return
        self.recorder.clear()
        self.loop_state = "RECORDING"
        for policy in self.policies:
            policy.start()
        cprint("\n[RECORDER] EPISODE STARTED", "red", attrs=["bold"])

    def _handle_success(self):
        """SHARE ('x'): save the episode and mark it successful."""
        self._finish_episode(success=True)

    def _handle_fail(self):
        """D-pad down: discard the HDF5 episode and mark it failed."""
        self._finish_episode(success=False)

    def _finish_episode(self, success: bool):
        if self.loop_state != "RECORDING":
            return

        if success:
            ep_name = f"episode_{int(time.time())}_{self.episode_counter}"
            self.recorder.save_episode(
                ep_name,
                {"dof": self.DOF},
            )
            self.episode_counter += 1
            cprint(f"[RECORDER] Saved {ep_name}", "green")
        else:
            self.recorder.clear()
            cprint("[RECORDER] Episode discarded", "red")

        for policy in self.policies:
            policy.finish(success)
        self.loop_state = "IDLE"
        cprint("Press OPTIONS ('o') to start a new episode.", "cyan")

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
                "follower_jv": np.zeros(self.DOF, np.float32),
                "leader_jp": np.zeros(self.DOF, np.float32),
                "policy_jp": np.zeros(self.DOF, np.float32),
                "policy_torque_scale": np.zeros(self.DOF, np.float32),
                "gripper_pos": 0.0,
                "gripper_vel": 0.0,
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

    def _infer_and_send(self, obs: dict):
        for policy, leader, follower in (
            (self.base_policy, self.base_leader_sender, self.base_follower_sender),
            (self.res_policy, self.res_leader_sender, self.res_follower_sender),
        ):
            if policy is None:
                continue
            policy.pre_infer(obs)
            action_chunk = policy.infer(obs)
            policy.post_infer(obs)
            if action_chunk is not None:
                leader.send_action_chunk(action_chunk)
                follower.send_action_chunk(action_chunk)

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.camera_manager.stop_all()
        for sender in (
            self.base_leader_sender,
            self.base_follower_sender,
            self.res_leader_sender,
            self.res_follower_sender,
        ):
            if sender is not None:
                sender.close()
        self.follower_listener.close()
        for policy in self.policies:
            policy.close()
        if self.joy_fd is not None:
            try:
                os.close(self.joy_fd)
            except Exception:
                pass
        try:
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

                if self.debug and not self._update_displays(
                    obs["front_image"], obs["wrist_image"]
                ):
                    cprint("Quitting...", "red")
                    break

                if self.loop_state == "RECORDING":
                    self.recorder.add_step(obs)
                    self._infer_and_send(obs)

                precise_wait(t_cycle_end)
                iter_idx += 1

        except Exception:
            import traceback

            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
            self.shutdown()


def build_policy(
    kind: str,
    *,
    checkpoint: str,
    model_config: str,
    data_dir: str,
    trajectory: str,
    dof: int,
    loop_hz: int,
    action_horizon: int,
    base_horizon: int,
    use_gripper: bool,
    debug: bool,
) -> PolicyTemplate | None:
    if kind == "none":
        return None
    if kind == "dmp":
        from wam.inference.dmp_policy import DMPPolicy

        return DMPPolicy(
            trajectory,
            dt=1.0 / loop_hz,
            dof=dof,
            use_gripper=use_gripper,
            horizon=base_horizon,
        )
    if kind == "openpi":
        return OpenPIPolicy(
            checkpoint,
            model_config,
            "libero",
            debug=debug,
            dof=dof,
            action_horizon=action_horizon,
        )

    policy = HilPolicy(checkpoint, data_dir, dof=dof)
    if policy.encoder_config.kind != kind:
        raise ValueError(
            f"policy {kind} but {checkpoint}/config.json was trained with "
            f"encoder kind {policy.encoder_config.kind!r}"
        )
    cprint(
        f"[HIL] {kind} loaded from {checkpoint} (image_size={policy.image_size})",
        "green",
    )
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run base + residual policies")

    # Network config
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Remote UDP IP")
    parser.add_argument("--recv_port", type=int, default=6554, help="UDP receive port")
    parser.add_argument("--dof", type=int, default=7, help="Degrees of freedom")

    # Base policy
    parser.add_argument(
        "--base_policy",
        choices=["none", "dmp", "openpi", "hilserl", "rltoken", "timm"],
        default="dmp",
    )
    parser.add_argument("--base_trajectory", type=str, default="")
    parser.add_argument("--base_checkpoint", type=str, default="")
    parser.add_argument("--base_config", type=str, default="haptic_wam_pi05")
    parser.add_argument("--base_data_dir", type=str, default="./interventions")
    parser.add_argument("--base_use_gripper", action="store_true")
    parser.add_argument("--base_horizon", type=int, default=8)
    parser.add_argument("--base_leader_port", type=int, default=10000)
    parser.add_argument("--base_follower_port", type=int, default=20000)

    # Residual policy
    parser.add_argument(
        "--res_policy",
        choices=["none", "dmp", "openpi", "hilserl", "rltoken", "timm"],
        default="hilserl",
    )
    parser.add_argument("--res_checkpoint", type=str, default="")
    parser.add_argument("--res_config", type=str, default="haptic_wam_pi05")
    parser.add_argument("--res_data_dir", type=str, default="./interventions")
    parser.add_argument("--res_trajectory", type=str, default="")
    parser.add_argument("--res_use_gripper", action="store_true")
    parser.add_argument("--res_horizon", type=int, default=8)
    parser.add_argument("--res_leader_port", type=int, default=10001)
    parser.add_argument("--res_follower_port", type=int, default=20001)

    # Camera config
    parser.add_argument("--wrist_cam", type=str, default="18475182")
    parser.add_argument("--front_cam", type=str, default="18475176")

    # System config
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--loop_hz", type=int, default=10)
    parser.add_argument("--action_horizon", type=int, default=5)

    args = parser.parse_args()

    base_policy = build_policy(
        args.base_policy,
        checkpoint=args.base_checkpoint,
        model_config=args.base_config,
        data_dir=args.base_data_dir,
        trajectory=args.base_trajectory,
        dof=args.dof,
        loop_hz=args.loop_hz,
        action_horizon=args.action_horizon,
        base_horizon=args.base_horizon,
        use_gripper=args.base_use_gripper,
        debug=args.debug,
    )
    res_policy = build_policy(
        args.res_policy,
        checkpoint=args.res_checkpoint,
        model_config=args.res_config,
        data_dir=args.res_data_dir,
        trajectory=args.res_trajectory,
        dof=args.dof,
        loop_hz=args.loop_hz,
        action_horizon=args.action_horizon,
        base_horizon=args.res_horizon,
        use_gripper=args.res_use_gripper,
        debug=args.debug,
    )

    teleop = PiZeroTeleop(
        remote_ip=args.ip,
        recv_port=args.recv_port,
        dof=args.dof,
        wrist_cam_serial=args.wrist_cam,
        front_cam_serial=args.front_cam,
        base_policy=base_policy,
        base_ports=(args.base_leader_port, args.base_follower_port),
        res_policy=res_policy,
        res_ports=(args.res_leader_port, args.res_follower_port),
        loop_hz=args.loop_hz,
        debug=args.debug,
        offline=args.offline,
    )

    teleop.run()
