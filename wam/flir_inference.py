import argparse
import dataclasses
import shutil
import signal
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from genicam.gentl import TimeoutException
from harvesters.core import Harvester
from termcolor import cprint
from udp_handler import TeleopUDPHandler

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


def buffer_to_image(buffer):
    component = buffer.payload.components[0]
    width = component.width
    height = component.height
    data_format = component.data_format
    
    raw_data = component.data.copy()

    if "Mono" in data_format:
        return raw_data.reshape((height, width))

    elif "RGB" in data_format or "BGR" in data_format:
        channels = 4 if "a" in data_format.lower() else 3
        img_nd = raw_data.reshape((height, width, channels))
        if "RGB" in data_format:
            img_nd = img_nd[:, :, :3][:, :, ::-1]
        return img_nd

    elif "Bayer" in data_format:
        img_1d = raw_data.reshape((height, width))
        if "BayerRG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)
        elif "BayerBG" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerRG2BGR)
        elif "BayerGB" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGR2BGR)
        elif "BayerGR" in data_format:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerGB2BGR)
        else:
            return cv2.cvtColor(img_1d, cv2.COLOR_BayerBG2BGR)

    else:
        raise ValueError(f"Can't convert {data_format}")


class FLIRStream:
    def __init__(self, device):
        self.device = device
        self.latest_image = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.device.start()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                with self.device.fetch(timeout=0.5) as buffer:
                    if len(buffer.payload.components) == 0:
                        continue
                    
                    img = buffer_to_image(buffer)
                    with self.lock:
                        self.latest_image = img
            except TimeoutException:
                # Silently ignore dropped frames and try again
                continue
            except Exception as e:
                # If the main thread destroys the GenTL device during shutdown,
                # catch the ClosedException and exit the loop gracefully.
                if not self.running:
                    break
                else:
                    cprint(f"Unexpected FLIR stream error: {e}", "red")
                    time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.latest_image is not None:
                return self.latest_image.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.device.stop()
        except Exception:
            pass


def open_flir(harvester: Harvester, serial_num: str = None):
    if serial_num:
        device = harvester.create({'serial_number': str(serial_num)})
    else:
        device = harvester.create()
    return device


class InterpolatingStreamer:
    def __init__(self, udp_handler: TeleopUDPHandler, dof: int, send_interval: float, stream_hz: int = 100, action_horizon: int = 5):
        self.udp_handler = udp_handler
        self.dof = dof
        
        # Timing parameters
        self.send_interval = send_interval
        self.stream_hz = stream_hz
        self.stream_dt = 1.0 / stream_hz
        self.action_horizon = action_horizon
        
        # Thread-safe data structures
        self.waypoint_queue = deque() 
        self.last_sent_joints = np.zeros(dof)
        self.queue_lock = threading.Lock() # Prevents popping while overwriting
        
        self.running = False
        self.thread = None

    def update_chunk(self, raw_action_chunk: np.ndarray):
        """Called by the main thread. Instantly interpolates and queues the new trajectory."""
        total_interpolated_points = int(self.send_interval * self.stream_hz)
        
        # TODO: handle gripper later...
        raw_waypoints = raw_action_chunk[:self.action_horizon, :self.dof]

        # Stretch the original N waypoints into M high-frequency waypoints
        original_time = np.linspace(0, 1, self.action_horizon)
        new_time = np.linspace(0, 1, total_interpolated_points)
        
        high_freq_waypoints = np.zeros((total_interpolated_points, self.dof))
        for j in range(self.dof):
            high_freq_waypoints[:, j] = np.interp(new_time, original_time, raw_waypoints[:, j])

        # Safely overwrite the queue so the streaming loop seamlessly transitions
        with self.queue_lock:
            self.waypoint_queue.clear()
            self.waypoint_queue.extend(high_freq_waypoints)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _stream_loop(self):
        while self.running:
            loop_start = time.time()
            
            # 1. Safely pop from the left of the queue, or hold position if empty
            with self.queue_lock:
                if len(self.waypoint_queue) > 0:
                    target_joints = self.waypoint_queue.popleft()
                    self.last_sent_joints = target_joints
                else:
                    target_joints = self.last_sent_joints

            # 2. Send UDP command
            self.udp_handler.send_data(target_joints, [0] * self.dof, [0] * self.dof)
            
            # 3. Sleep to maintain exact stream_hz
            sleep_time = self.stream_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

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
                 send_interval: float = 0.2,
                 display_scale: int = 3,
                 debug: bool = False,
                 offline: bool = False):
        
        # System config
        self.debug = debug
        self.offline = offline
        self.display_scale = display_scale
        self.send_interval = send_interval
        self.last_send_time = 0.0

        # UDP Configuration
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.DOF = dof
        self.udp_handler = TeleopUDPHandler(self.remote_ip, self.send_port, self.recv_port, DOF=self.DOF)
        self.udp_stream = InterpolatingStreamer(self.udp_handler, self.DOF, self.send_interval, stream_hz=100, action_horizon=5)

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
        
        cprint("Initialization complete. Running teleop loop...", "green")

    def run(self):
        try:
            self.system_running = True
            def force_shutdown(signum, frame):
                cprint("\n[SYSTEM] Ctrl+C detected! Forcing main loop shutdown...", "red")
                self.system_running = False

            signal.signal(signal.SIGINT, force_shutdown)
            
            while self.system_running:
                frame_wrist = self.wrist_stream.read()
                frame_front = self.scene_stream.read()
                
                if frame_wrist is None or frame_front is None:
                    continue

                proc_wrist = preprocess_image(frame_wrist) 
                proc_front = preprocess_image(frame_front)

                if self.debug:
                    cv2.imshow("Wrist", cv2.resize(proc_wrist, (224 * self.display_scale, 224 * self.display_scale)))
                    cv2.imshow("Front", cv2.resize(proc_front, (224 * self.display_scale, 224 * self.display_scale)))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cprint("Quitting...", "red")
                        break
                
                now = time.time()
                if now - self.last_send_time >= self.send_interval:
                    wrist_image = cv2.cvtColor(proc_wrist, cv2.COLOR_BGR2RGB)
                    front_image = cv2.cvtColor(proc_front, cv2.COLOR_BGR2RGB)
                    
                    if self.offline:
                        jp_state = np.array([0, 0.2, 1.5, -1.5, 0, 0, 0])
                    else:
                        robot_state = self.udp_handler.receive_data()
                        if robot_state is None:
                            cprint("No robot state received yet, waiting...", "red")
                            self.last_send_time = now
                            continue
                        jp_state = np.array(robot_state["jp"], dtype=np.float32)
                    cprint(f"Robot State: {np.round(jp_state, 3)}", "yellow")
                        
                    example = {
                        "observation/exterior_image_1_left": front_image,
                        "observation/wrist_image_left": wrist_image,
                        "observation/joint_position": jp_state,
                        "observation/gripper_position": np.zeros(1, dtype=np.float32), # TODO figure out gripper stuff
                        "prompt": "touch the red cup",
                    }
                    
                    action_chunk = self.policy.infer(example)["actions"]
                    
                    # add joints to convert to absolute
                    jp_state = np.concatenate([jp_state, np.zeros(1)]) # add dummy gripper state for now, fix later
                    action_chunk = action_chunk + jp_state
                    
                    # clip by joint limits
                    action_chunk[:, :7] = np.clip(action_chunk[:, :7], WAM_MIN_LIMITS, WAM_MAX_LIMITS)

                    if not self.udp_stream.running:
                        self.udp_stream.start()
                    self.udp_stream.update_chunk(action_chunk)
                    if self.debug:
                        # debug printing for the action chunk
                        cprint("--- New chunk ---", "blue")
                        for act in action_chunk:
                            act = act[:self.DOF]
                            cprint(f"Action from chunk: {np.round(act, 3)}", "cyan")

                    self.last_send_time = now
                
                else:
                    # If not debugging/rendering, prevent the CPU from spinning at 100% 
                    # while waiting for the next send interval.
                    if not self.debug:
                        time.sleep(0.005)

        except Exception as e:
            import traceback
            print("CRASHED WITH ERROR:")
            traceback.print_exc()

        finally:
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
        offline=args.offline
    )
    
    teleop.run()