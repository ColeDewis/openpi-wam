import argparse
import signal
import time
from pynput import keyboard
import cv2
import numpy as np
from termcolor import cprint
from wam.inference.openpi_policy import OpenPIPolicy
from wam.inference.interpolating_streamer import InterpolatingStreamer
from wam.inference.hdf5_recorder import HDF5Recorder
from wam.inference.wam_manager import WAMManager
from wam.flir.multi_flir_manager import MultiFLIRManager

np.set_printoptions(precision=4, suppress=True)


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
        control_hz: int = 10,
        action_horizon: int = 5,
        loop_hz: int = 30,
        display_scale: int = 3,
        debug: bool = False,
        offline: bool = False,
        infer: bool = False,
        record: bool = False,
    ):

        # System config
        self.debug = debug
        self.offline = offline
        self.display_scale = display_scale
        self.action_horizon = action_horizon
        self.control_hz = control_hz
        self.loop_hz = loop_hz
        self.save_action_step_size = max(1, int(round(self.loop_hz / self.control_hz)))
        self.send_interval = self.action_horizon / self.control_hz
        self.last_send_time = 0.0
        self.doing_inference = infer
        self.doing_recording = record

        # UDP Configuration
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.DOF = dof
        self.wam_manager = WAMManager(
            self.remote_ip,
            send_port=self.send_port,
            follower_recv_port=5557,
            leader_recv_port=5558,
            dof=self.DOF,
        )
        self.udp_stream = InterpolatingStreamer(
            self.wam_manager.follower_sender,
            self.DOF,
            self.send_interval,
            stream_hz=100,
            action_horizon=self.action_horizon,
        )

        # FLIR setup
        camera_configs = {
            "wrist_image": wrist_cam_serial,
            # "front_image": front_cam_serial,
        }
        self.camera_manager = MultiFLIRManager(camera_configs)
        self.camera_manager.start_all()

        # pi0 Model Configuration
        if self.doing_inference:
            self.policy = OpenPIPolicy(checkpoint_path, model_config, debug=self.debug)

        # Set up recorder
        self.loop_state = "IDLE"  # States: IDLE, RECORDING, PENDING
        self.episode_counter = 0
        self.recorder = HDF5Recorder(save_dir="./dataset")

        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Controls: [R] Start/Stop | [S] Save | [D] Discard", "cyan")

    def _on_key_press(self, key):
        """Asynchronous callback for keyboard events."""
        try:
            if hasattr(key, "char") and key.char is not None:
                k = key.char.lower()

                if self.loop_state == "IDLE" and k == "r":
                    self.loop_state = "RECORDING"
                    cprint("\n[RECORDER] 🔴 EPISODE STARTED", "red", attrs=["bold"])

                elif self.loop_state == "RECORDING" and k == "r":
                    self.loop_state = "PENDING"
                    cprint("\n[RECORDER] ⏸ EPISODE PAUSED", "yellow")
                    cprint("Press [S] to Save or [D] to Discard.", "cyan")

                elif self.loop_state == "PENDING":
                    if k == "s":
                        ep_name = f"episode_{int(time.time())}_{self.episode_counter}"
                        
                        # Package the metadata for future you
                        metadata = {
                            "loop_hz": self.loop_hz,
                            "control_hz": self.control_hz,
                            "action_step_size": self.save_action_step_size,
                            "action_horizon": self.action_horizon,
                            "dof": self.DOF
                        }
                        
                        self.recorder.save_episode(ep_name, metadata)
                        self.episode_counter += 1
                        self.recording_state = "IDLE"
                        cprint(
                            "[RECORDER] Ready for next episode. Press [R] to start.",
                            "cyan",
                        )
                    elif k == "d":
                        self.recorder.clear()
                        self.loop_state = "IDLE"
                        cprint(
                            "\n[RECORDER] 🗑 Episode discarded. Press [R] to start a new one.",
                            "red",
                        )

        except Exception:
            pass

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
            state_dict = {"follower_state": {"jp": jp_state}}
            status = True
        else:
            state_dict = self.wam_manager.get_latest_states()
            status = state_dict["follower_state"].get("jp") is not None

        if self.debug:
            cprint(f"Robot State: {np.round(state_dict['follower_state']['jp'], 3)}", "yellow")

        return status, state_dict

    def _update_displays(self, front_image, wrist_image):
        debug_front = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
        debug_wrist = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(
            "Wrist",
            cv2.resize(
                debug_wrist, (224 * self.display_scale, 224 * self.display_scale)
            ),
        )
        cv2.imshow(
            "Front",
            cv2.resize(
                debug_front, (224 * self.display_scale, 224 * self.display_scale)
            ),
        )

        return cv2.waitKey(1) & 0xFF != ord("q")

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.udp_stream.running = False
        self.camera_manager.stop_all()
        self.udp_stream.stop()
        try:
            # Explicitly target the named windows
            cv2.destroyWindow("Wrist")
            cv2.destroyWindow("Front")
            cv2.waitKey(1)
        except Exception:
            pass
        time.sleep(2)
        import os

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
            while self.system_running:
                loop_start_time = time.time()

                # Read images
                img_status, image_dict = self._read_images()
                state_status, state_dict = self._read_state()
                if not img_status or not state_status:
                    cprint("State or images not received yet, waiting...", "red")
                    time.sleep(loop_delay)
                    continue
                obs = image_dict.copy()
                if state_dict.get("leader_state") is not None:
                    obs["leader_state"] = state_dict["leader_state"]
                obs["follower_state"] = state_dict["follower_state"]

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
                if (loop_start_time - self.last_send_time) >= self.send_interval:
                    if self.doing_inference and self.loop_state == "RECORDING":
                        action_chunk = self.policy.infer(obs)
                        self.udp_stream.update_chunk(action_chunk)

                    self.last_send_time = loop_start_time

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
    parser.add_argument("--send_port", type=int, default=5556, help="UDP Send Port")
    parser.add_argument("--recv_port", type=int, default=5557, help="UDP Receive Port")

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
        default="gs://openpi-assets/checkpoints/pi05_base",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", type=str, default="pi05_droid", help="Policy config name"
    )

    # System config
    parser.add_argument(
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
        record=args.record
    )

    teleop.run()
