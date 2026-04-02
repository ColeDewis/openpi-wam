import socket
import struct
import time
import numpy as np


class TeleopUDPHandler:
    def __init__(self, remote_ip, send_port, recv_port=None, DOF=7):
        """
        :param remote_ip: IP address of the target (e.g., '127.0.0.1')
        :param send_port: The port the target is listening on.
        :param recv_port: (Optional) The port THIS Python code listens on. 
                          If None, this instance is SEND-ONLY.
        :param dof: Degrees of freedom (default 7).
        """
        self.dof = DOF
        self.remote_ip = remote_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.is_listening = (recv_port is not None)
        
        # Structure format: 3 vectors of doubles
        # 'd' = double (8 bytes) -> 21 doubles for 7-DOF
        self.fmt = f'{self.dof * 3}d'
        self.packet_size = struct.calcsize(self.fmt)

        # 1. Sender Socket (Always created)
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 2. Receiver Socket (Only created if recv_port is specified)
        self.sock_recv = None
        if self.is_listening:
            self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Bind allows us to listen on this port
                self.sock_recv.bind(("0.0.0.0", self.recv_port))
                self.sock_recv.setblocking(False) # Non-blocking read
                print(f"UDP [Recv+Send]: Sending to {remote_ip}:{send_port}, Listening on {recv_port}")
            except OSError as e:
                print(f"Error binding to port {recv_port}: {e}")
                self.is_listening = False # Fallback to send-only on error
        else:
            print(f"UDP [Send-Only]: Sending to {remote_ip}:{send_port}, No listening.")

    def send_data(self, jp, jv, torque):
        """
        Sends joint data to the remote target.
        """
        if len(jp) != self.dof or len(jv) != self.dof or len(torque) != self.dof:
            print(f"Error: All inputs must be lists of size {self.dof}")
            return

        payload = np.concatenate([jp, jv, torque])
        
        try:
            data_bytes = struct.pack(self.fmt, *payload)
            self.sock_send.sendto(data_bytes, (self.remote_ip, self.send_port))
        except Exception as e:
            print(f"Send Error: {e}")

    def receive_data(self):
        """
        Checks for new data. Returns None if configured as Send-Only.
        """
        if not self.is_listening:
            return None

        try:
            data, _ = self.sock_recv.recvfrom(1024)
            
            if len(data) != self.packet_size:
                return None
            
            unpacked = struct.unpack(self.fmt, data)
            
            return {
                'jp': list(unpacked[0 : self.dof]),
                'jv': list(unpacked[self.dof : self.dof*2]),
                'torque': list(unpacked[self.dof*2 : self.dof*3])
            }
            
        except BlockingIOError:
            return None # No data waiting
        except Exception as e:
            print(f"Receive Error: {e}")
            return None

    def close(self):
        if self.sock_send: self.sock_send.close()
        if self.sock_recv: self.sock_recv.close()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":    
    dof = 4
    udp = TeleopUDPHandler(remote_ip="127.0.0.1", send_port=5556, recv_port=None, DOF=dof)

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