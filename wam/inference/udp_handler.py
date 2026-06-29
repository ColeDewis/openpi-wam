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
        num_doubles = (4 * self.dof) + 3 + 4 + 1
        self.fmt = f"<{num_doubles}dQ"
        self.packet_size = struct.calcsize(self.fmt)

        # 1. Sender Socket (Always created)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"UDP [inference send]: {remote_ip}:{send_port}.")

    def send_data(self, jp=None, jv=None, ext_torque=None, meas_torque=None, cart_pos=None, cart_rot=None, gripper=0.0):
        """
        Sends joint data to the remote target.
        """
        if jp is None:
            jp = [0.0] * self.dof
        if jv is None:
            jv = [0.0] * self.dof
        if ext_torque is None:
            ext_torque = [0.0] * self.dof
        if meas_torque is None:
            meas_torque = [0.0] * self.dof
        if cart_pos is None:
            cart_pos = [0.0] * 3
        if cart_rot is None:
            cart_rot = [0.0] * 4 # we will be sending euler but for code simplicity we keep 4 spaces
 
        if any(len(v) != self.dof for v in [jp, jv, ext_torque, meas_torque]):
            print(f"Error: jp, jv, ext_torque, meas_torque must all be length {self.dof}")
            return
 
        now_ns = time.time_ns()
        payload = list(jp) + list(jv) + list(ext_torque) + list(meas_torque) + list(cart_pos) + list(cart_rot) + [float(gripper)]
 
        try:
            data_bytes = struct.pack(self.fmt, *payload, now_ns)
            self.sock_send.sendto(data_bytes, (self.remote_ip, self.send_port))
        except Exception as e:
            print(f"Send Error: {e}")


    def close(self):
        if self.sock_send: self.sock_send.close()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":    
    dof = 4
    udp = TeleopUDPHandler(remote_ip="127.0.0.1", send_port=5556, DOF=dof)

    try:
        t = 0
        while True:
            my_jp = [0.0] * dof
            my_jp[0] = 0.5 # Just move joint 1
            
            my_jv = [0.0] * dof
            my_tau = [0.0] * dof
            
            # 2. Send (No receive needed)
            udp.send_data(my_jp, my_jv, my_tau)
            
            time.sleep(0.002) # 500Hz
            t += 1

    except KeyboardInterrupt:
        print("\nStopping...")
        udp.close()
