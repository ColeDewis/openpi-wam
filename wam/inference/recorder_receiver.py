import socket
import struct
import time


class RecorderReceiver:
    def __init__(self, ip, recv_port, dof=7):
        """
        :param recv_port: The port THIS Python code listens on.
        :param dof: Degrees of freedom (default 7).
        """
        self.dof = dof
        self.recv_port = recv_port

        # Structure format matches C++ RecorderPayload
        # '<'  = Little-endian (Standard memory layout)
        # 4 arrays of doubles (jp, jv, ext_tau, meas_tau) + 1 double (gripper) = (4 * dof) + 1
        # 'Q'  = uint64_t (timestamp)
        num_doubles = (4 * self.dof) + 1
        self.fmt = f"<{num_doubles}dQ"
        self.packet_size = struct.calcsize(self.fmt)

        # Receiver Socket
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock_recv.bind((ip, self.recv_port))
            self.sock_recv.setblocking(False)
            print(
                f"UDP [Recorder]: Listening on port {self.recv_port} for {self.packet_size}-byte packets"
            )
        except OSError as e:
            print(f"Error binding to port {self.recv_port}: {e}")
            self.sock_recv = None

    def receive_latest_data(self):
        """
        Drains the OS UDP buffer and returns ONLY the most recent packet.
        This ensures Python doesn't fall behind the high-frequency C++ control loop.
        """
        if not self.sock_recv:
            return None

        latest_data = None

        # Keep reading until the buffer throws a BlockingIOError (meaning it's empty)
        while True:
            try:
                data, _ = self.sock_recv.recvfrom(1024)
                if len(data) == self.packet_size:
                    latest_data = data
            except BlockingIOError:
                break  # Buffer is empty, we have the freshest packet
            except Exception as e:
                print(f"Receive Error: {e}")
                break

        # If we didn't get any valid data this loop, return None
        if latest_data is None:
            return None

        # Unpack the freshest packet
        unpacked = struct.unpack(self.fmt, latest_data)

        # Calculate slicing indices
        idx_jp = 0
        idx_jv = self.dof
        idx_ext_tau = self.dof * 2
        idx_meas_tau = self.dof * 3
        idx_gripper = self.dof * 4
        idx_timestamp = self.dof * 4 + 1

        return {
            "jp": list(unpacked[idx_jp:idx_jv]),
            "jv": list(unpacked[idx_jv:idx_ext_tau]),
            "ext_torque": list(unpacked[idx_ext_tau:idx_meas_tau]),
            "meas_torque": list(unpacked[idx_meas_tau:idx_gripper]),
            "gripper": unpacked[idx_gripper],
            "timestamp_us": unpacked[idx_timestamp],
        }

    def close(self):
        if self.sock_recv:
            self.sock_recv.close()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    dof = 7
    # E.g., Port 8001 for Leader, 8002 for Follower
    follower_udp = RecorderReceiver(recv_port=8002, dof=dof)

    try:
        print("Waiting for data... (Press Ctrl+C to stop)")
        while True:
            # This simulates a slow Python loop (e.g., waiting on a 30Hz camera frame)
            time.sleep(1.0 / 30.0)

            data = follower_udp.receive_latest_data()

            if data:
                print(
                    f"[{data['timestamp_us']}] Gripper: {data['gripper']:.3f} | J1 Pos: {data['jp'][0]:.3f}"
                )

    except KeyboardInterrupt:
        print("\nStopping...")
        follower_udp.close()
