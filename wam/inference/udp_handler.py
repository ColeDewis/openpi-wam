import socket
import struct
import time
import numpy as np


class TeleopUDPHandler:
    def __init__(self, remote_ip, send_port, DOF=7):
        """
        :param remote_ip: IP address of the target (e.g., '127.0.0.1')
        :param send_port: The port the target is listening on.
        :param dof: Degrees of freedom (default 7).
        """
        self.dof = DOF
        self.remote_ip = remote_ip
        self.send_port = send_port
        
        # jp (DOF doubles) + jv (DOF doubles) + ext_torque (DOF doubles) +
        # meas_torque (DOF doubles) + gripper (1 double) + timestamp (uint64_t)
        num_doubles = 7
        self.fmt = f"<{num_doubles}dQ"
        self.packet_size = struct.calcsize(self.fmt)

        # 1. Sender Socket (Always created)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"UDP [inference send]: {remote_ip}:{send_port}.")

    def send_data(self, cart_pos=None, cart_rot=None, gripper_cmd=0.0):
        """
        Sends joint data to the remote target.
        """
        if cart_pos is None:
            cart_pos = [0.0] * 3
        if cart_rot is None:
            cart_rot = [0.0] * 3
 
        now_ns = time.time_ns()
        payload = list(cart_pos) + list(cart_rot) + [float(gripper_cmd)]
 
        try:
            data_bytes = struct.pack(self.fmt, *payload, now_ns)
            self.sock_send.sendto(data_bytes, (self.remote_ip, self.send_port))
        except Exception as e:
            print(f"Send Error: {e}")


    def close(self):
        if self.sock_send: self.sock_send.close()

