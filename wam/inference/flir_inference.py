import argparse
import dataclasses
import shutil
import signal
import time
from pathlib import Path
from pynput import keyboard
import cv2
import numpy as np
from harvesters.core import Harvester
from termcolor import cprint
from wam.inference.udp_handler import TeleopUDPHandler
from wam.inference.interpolating_streamer import InterpolatingStreamer
from wam.inference.hdf5_recorder import HDF5Recorder
from wam.flir.flir_streamer import FLIRStream, open_flir

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

np.set_printoptions(precision=4, suppress=True)

# NOTE a bit lower than actual for safety
WAM_MIN_LIMITS = np.array([-2.5, -1.9, -2.6, -0.7, -4.5, -1.4, -2.9])
WAM_MAX_LIMITS = np.array([ 2.5,  1.9,  2.6,  2.9,  1.1,  1.4,  2.9])

def preprocess_image(img_bgr: np.ndarray, crop_scale: float = 0.9, out_size=(224, 224)) -> np.ndarray:
    """Center-crop by area 'crop_scale' and resize to out_size using OpenCV. Keeps image in BGR."""
    H, W = img_bgr.shape[:2]
    s = float(crop_scale) ** 0.5
    crop_h, crop_w = int(round(H * s)), int(round(W * s))
    
    y0 = max((H - crop_h) // 2, 0)
    x0 = max((W - crop_w) // 2, 0)
    
    img_cropped = img_bgr[y0:y0 + crop_h, x0:x0 + crop_w]
    img_resized = cv2.resize(img_cropped, out_size, interpolation=cv2.INTER_AREA)
    
    return img_resized

class PiZeroTeleop:
    def __init__(self, 
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
                 mode: str = 'record'):
        
        # System config
        self.debug = debug
        self.offline = offline
        self.display_scale = display_scale
        self.action_horizon = action_horizon
        self.control_hz = control_hz
        self.save_action_step_size = max(1, int(round(self.loop_hz / self.control_hz)))
        self.send_interval = self.action_horizon / self.control_hz
        self.loop_hz = loop_hz
        self.last_send_time = 0.0
        self.mode = mode

        # UDP Configuration
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.DOF = dof
        self.udp_handler = TeleopUDPHandler(self.remote_ip, self.send_port, self.recv_port, DOF=self.DOF)
        self.udp_stream = InterpolatingStreamer(self.udp_handler, self.DOF, self.send_interval, stream_hz=100, action_horizon=self.action_horizon)

        # Camera Configuration
        cprint("Initializing Harvesters and FLIR Cameras...", "green")
        self.harvester = Harvester()
        self.harvester.add_file("/opt/spinnaker/lib/spinnaker-gentl/Spinnaker_GenTL.cti")
        self.harvester.update()
        
        wrist_cam = open_flir(self.harvester, wrist_cam_serial)
        scene_cam = open_flir(self.harvester, front_cam_serial)
        
        self.wrist_stream = FLIRStream(wrist_cam)
        self.scene_stream = FLIRStream(scene_cam)
        self.wrist_stream.start()
        self.scene_stream.start()

        # pi0 Model Configuration
        if "infer" in self.mode:
            cprint(f"Loading policy with config: {model_config} from {checkpoint_path}...", "green")
            self.checkpoint_path = checkpoint_path
            self.config = _config.get_config(model_config)
            wam_assets = _config.AssetsConfig(assets_dir="/home/coled/openpi/wam/config", asset_id="wam")
            new_data_cfg = dataclasses.replace(self.config.data, assets=wam_assets)
            self.config = dataclasses.replace(self.config, data=new_data_cfg)
            checkpoint_dir = download.maybe_download(self.checkpoint_path)

            # openpi requires file to be present in the assets directory, so copy it there
            local_norm_path = Path("/home/coled/openpi/wam/config/wam/norm_stats.json")
            cached_assets_dir = Path(checkpoint_dir) / "assets" / "wam"
            cached_assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_norm_path, cached_assets_dir / "norm_stats.json")
            
            # Create a trained policy
            self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)
            
            # Do one initial inference on the policy to make sure it is fully loaded
            if "droid" in model_config:
                example = {
                    "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                    "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                    "observation/joint_position": np.random.rand(7),
                    "observation/gripper_position": np.random.rand(1),
                    "prompt": "do something",
                }
                self.policy.infer(example)
            
        # Set up recorder
        self.recording_state = "IDLE"  # States: IDLE, RECORDING, PENDING
        self.episode_counter = 0
        self.recorder = HDF5Recorder(save_dir="./dataset")
        
        self.kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self.kb_listener.start()
        cprint("Initialization complete. Running teleop loop...", "green")
        cprint("Recording controls: [R] Start/Stop | [S] Save | [D] Discard", "cyan")
            

    def _on_key_press(self, key):
        """Asynchronous callback for keyboard events."""
        try:
            if hasattr(key, 'char') and key.char is not None:
                k = key.char.lower()
                
                if self.recording_state == "IDLE" and k == 'r':
                    self.recording_state = "RECORDING"
                    cprint("\n[RECORDER] 🔴 RECORDING STARTED", "red", attrs=["bold"])
                    
                elif self.recording_state == "RECORDING" and k == 'r':
                    self.recording_state = "PENDING"
                    cprint("\n[RECORDER] ⏸ RECORDING PAUSED", "yellow")
                    cprint("Press [S] to Save or [D] to Discard.", "cyan")
                    
                elif self.recording_state == "PENDING":
                    if k == 's':
                        ep_name = f"episode_{int(time.time())}_{self.episode_counter}"
                        self.episode_counter += 1
                        self.recording_state = "IDLE"
                        self.recorder.save_episode(ep_name, action_step_size=self.save_action_step_size)
                        cprint("[RECORDER] Ready for next episode. Press [R] to start.", "cyan")
                    elif k == 'd':
                        self.recorder.clear()
                        self.recording_state = "IDLE"
                        cprint("\n[RECORDER] 🗑 Episode discarded. Press [R] to start a new one.", "red")
                        
        except Exception:
            pass

    def _read_images(self):
        frame_wrist = self.wrist_stream.read()
        frame_front = self.scene_stream.read()
        
        if frame_wrist is None or frame_front is None:
            return False, frame_front, frame_wrist

        proc_wrist = preprocess_image(frame_wrist) 
        proc_front = preprocess_image(frame_front)
        wrist_image = cv2.cvtColor(proc_wrist, cv2.COLOR_BGR2RGB)
        front_image = cv2.cvtColor(proc_front, cv2.COLOR_BGR2RGB)
        
        return True, front_image, wrist_image

    def _read_state(self):
        if self.offline:
            jp_state = np.array([0, 0.2, 1.5, -1.5, 0, 0, 0])
        else:
            robot_state = self.udp_handler.receive_data()
            if robot_state is None:
                return False, robot_state
            jp_state = np.array(robot_state["jp"], dtype=np.float32)
            
        if self.debug:
            cprint(f"Robot State: {np.round(jp_state, 3)}", "yellow")
            
        return True, jp_state

    def _model_inference(self, front_image, wrist_image, robot_state):
        example = {
            "observation/exterior_image_1_left": front_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": robot_state,
            "observation/gripper_position": np.zeros(1, dtype=np.float32), # TODO figure out gripper stuff
            "prompt": "touch the red cup",
        }
        
        action_chunk = self.policy.infer(example)["actions"]
        
        # add joints to convert to absolute
        robot_state = np.concatenate([robot_state, np.zeros(1)]) # add dummy gripper state for now, fix later
        action_chunk = action_chunk + robot_state
        
        # clip by joint limits
        action_chunk[:, :7] = np.clip(action_chunk[:, :7], WAM_MIN_LIMITS, WAM_MAX_LIMITS)
        
        return action_chunk

    def _update_displays(self, front_image, wrist_image):
        debug_front = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
        debug_wrist = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Wrist", cv2.resize(debug_wrist, (224 * self.display_scale, 224 * self.display_scale)))
        cv2.imshow("Front", cv2.resize(debug_front, (224 * self.display_scale, 224 * self.display_scale)))
        
        return cv2.waitKey(1) & 0xFF != ord('q')

    def shutdown(self):
        cprint("Cleaning up streams and windows...", "red")
        self.udp_stream.running = False
        self.wrist_stream.running = False
        self.scene_stream.running = False
        
        self.wrist_stream.stop()
        self.scene_stream.stop()
        self.udp_stream.stop()
        self.harvester.reset()
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
                cprint("\n[SYSTEM] Ctrl+C detected! Forcing main loop shutdown...", "red")
                self.system_running = False

            signal.signal(signal.SIGINT, force_shutdown)
            
            loop_delay = 1 / self.loop_hz
            while self.system_running:
                loop_start_time = time.time()
                
                # Read images
                img_status, front_image, wrist_image = self._read_images()
                state_status, jp_state = self._read_state()
                if not img_status or not state_status: 
                    cprint("State or images not received yet, waiting...", "red")
                    time.sleep(loop_delay)
                    continue

                # Update display windows if debugging
                if self.debug and not self._update_displays(front_image, wrist_image):
                    cprint("Quitting...", "red")
                    break
                
                # Recording
                if "record" in self.mode and self.recording_state == "RECORDING":
                    self.recorder.add_step(wrist_image, front_image, jp_state)
                
                # Infer + send to WAM
                if (loop_start_time - self.last_send_time) >= self.send_interval:
                    if "infer" in self.mode:
                        action_chunk = self._model_inference(front_image, wrist_image, jp_state)

                        if not self.udp_stream.running:
                            self.udp_stream.start()
                        self.udp_stream.update_chunk(action_chunk)
                    
                        if self.debug:
                            cprint("--- New chunk ---", "blue")
                            for act in action_chunk:
                                act = act[:self.DOF]
                                cprint(f"Action from chunk: {np.round(act, 3)}", "cyan")
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
    parser.add_argument("--wrist_cam", type=str, default="18475182", help="Serial number for wrist camera")
    parser.add_argument("--front_cam", type=str, default="18475176", help="Serial number for front camera")
    
    # Model config
    parser.add_argument("--checkpoint", type=str, default="gs://openpi-assets/checkpoints/pi05_base", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="pi05_droid", help="Policy config name")
    
    # System config
    parser.add_argument("--debug", action="store_true", help="Display OpenCV windows and extra print statements")
    parser.add_argument("--offline", action="store_true", help="Run without UDP robot connection (mocks hardware state)")
    parser.add_argument("--mode", type=str, default="record", help="Mode, should be (record, infer, infer_record)", choices=("record", "infer", "infer_record"))
    
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
        mode=args.mode
    )
    
    teleop.run()