import numpy as np
from collections import deque
import threading
import time
from wam.inference.udp_handler import TeleopUDPHandler

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