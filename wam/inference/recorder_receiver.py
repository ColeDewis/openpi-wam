"""UDP receiver for the WAM recorder/controller packet.

Two packet layouts are accepted:

1. The original ``RecorderPayload`` (6 arrays of doubles + gripper + one
   timestamp).  It exposes ``follower_jp``, ``follower_jv``,
   ``follower_ext_torque``, ``leader_jp``, ``leader_jv``, ``leader_ext_torque``,
   the follower cartesian pose, the gripper state, and ``timestamp_ns``.
2. The extended payload used by the base-policy + residual-policy setup.  It
   carries 17 DOF-length arrays (including ``policy_jp`` / ``res_policy_jp``),
   follower/leader cartesian poses and quaternions, gripper state, and two
   chunk-end timestamps.

The extended layout is parsed into the same keys as before where possible, so
``wam/inference/main.py`` keeps working unchanged.  The extra fields are added
verbatim so callers can use the base policy's ``policy_jp`` directly.
"""

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

        # Original RecorderPayload:
        #   follower_jp, follower_jv, follower_ext_torque,
        #   leader_jp, leader_jv, leader_ext_torque,
        #   follower_cart_pos(3), follower_cart_rot(4),
        #   gripper_pos, gripper_vel, gripper_torque, timestamp_ns
        num_doubles = (6 * self.dof) + 10
        self.fmt = f"<{num_doubles}dQ"
        self.packet_size = struct.calcsize(self.fmt)

        # Extended payload (base policy + residual policy):
        #   17 DOF-length arrays:
        #     follower_jp, follower_jv, follower_dyngravcomp_torque,
        #     environment_torque, filtered_environment_torque,
        #     leader_jp, leader_jv, leader_dyngravcomp_torque,
        #     human_torque, filtered_human_torque,
        #     policy_jp, policy_jt, policy_torque_scale,
        #     res_policy_jp, res_policy_jt,
        #     ref_policy_torque, ref_torque_policy_jt
        #   follower(cart_pos 3 + quat 4) + leader(cart_pos 3 + quat 4)
        #   gripper_pos, gripper_vel, gripper_torque
        #   'Q' = time_to_chunk_end (ns)
        #   'Q' = res_time_to_chunk_end (ns)
        ext_num_doubles = (17 * self.dof) + 17
        self.ext_fmt = f"<{ext_num_doubles}dQQ"
        self.ext_packet_size = struct.calcsize(self.ext_fmt)

        # Receiver Socket
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock_recv.bind((ip, self.recv_port))
            self.sock_recv.setblocking(False)  # noqa: FBT003
            print(
                f"UDP [Recorder]: Listening on port {self.recv_port} for "
                f"{self.packet_size}-byte (legacy) or {self.ext_packet_size}-byte "
                "(extended) packets"
            )
        except OSError as e:
            print(f"Error binding to port {self.recv_port}: {e}")
            self.sock_recv = None

    def receive_all_new(self):
        """Drain the OS UDP buffer and return every valid packet, in order."""
        if not self.sock_recv:
            return []

        packets = []
        max_size = max(self.packet_size, self.ext_packet_size) + 64
        while True:
            try:
                data, _ = self.sock_recv.recvfrom(max_size)
            except BlockingIOError:
                break  # Buffer is empty
            except Exception as e:
                print(f"Receive Error: {e}")
                break
            if len(data) == self.packet_size:
                packets.append(self._unpack_legacy(data))
            elif len(data) == self.ext_packet_size:
                packets.append(self._unpack_extended(data))
            else:
                print(
                    f"⚠️ Dropping packet! Expected {self.packet_size} (legacy) or "
                    f"{self.ext_packet_size} (extended) bytes, got {len(data)} bytes."
                )
        return packets

    def receive_latest_data(self):
        """
        Drain the OS UDP buffer and return ONLY the most recent packet.

        This ensures Python doesn't fall behind the high-frequency C++ control
        loop.  ``receive_all_new`` is available when every packet is needed.
        """
        packets = self.receive_all_new()
        return packets[-1] if packets else None

    def _unpack_legacy(self, data):
        unpacked = struct.unpack(self.fmt, data)

        idx_follower_jp = 0
        idx_follower_jv = self.dof
        idx_follower_ext_tau = self.dof * 2
        idx_leader_jp = self.dof * 3
        idx_leader_jv = self.dof * 4
        idx_leader_ext_tau = self.dof * 5
        idx_follower_cart_pos = self.dof * 6
        idx_follower_cart_rot = self.dof * 6 + 3
        idx_gripper_pos = self.dof * 6 + 7
        idx_gripper_vel = self.dof * 6 + 8
        idx_gripper_torque = self.dof * 6 + 9
        idx_timestamp = self.dof * 6 + 10

        return {
            "follower_jp": list(unpacked[idx_follower_jp:idx_follower_jv]),
            "follower_jv": list(unpacked[idx_follower_jv:idx_follower_ext_tau]),
            "follower_ext_torque": list(unpacked[idx_follower_ext_tau:idx_leader_jp]),
            "leader_jp": list(unpacked[idx_leader_jp:idx_leader_jv]),
            "leader_jv": list(unpacked[idx_leader_jv:idx_leader_ext_tau]),
            "leader_ext_torque": list(unpacked[idx_leader_ext_tau:idx_follower_cart_pos]),
            "follower_cart_pos": list(unpacked[idx_follower_cart_pos:idx_follower_cart_rot]),
            "follower_cart_rot": list(unpacked[idx_follower_cart_rot:idx_gripper_pos]),
            "gripper_pos": unpacked[idx_gripper_pos],
            "gripper_vel": unpacked[idx_gripper_vel],
            "gripper_torque": unpacked[idx_gripper_torque],
            "timestamp_ns": unpacked[idx_timestamp],
        }

    def _unpack_extended(self, data):
        unpacked = struct.unpack(self.ext_fmt, data)
        recv_ts_ns = time.time_ns()  # capture immediately on arrival
        dof = self.dof

        idx_follower_jp = 0
        idx_follower_jv = dof
        idx_follower_dyngravcomp_torque = dof * 2
        idx_environment_torque = dof * 3
        idx_filtered_environment_torque = dof * 4
        idx_leader_jp = dof * 5
        idx_leader_jv = dof * 6
        idx_leader_dyngravcomp_torque = dof * 7
        idx_human_torque = dof * 8
        idx_filtered_human_torque = dof * 9
        idx_policy_jp = dof * 10
        idx_policy_jt = dof * 11
        idx_policy_torque_scale = dof * 12
        idx_res_policy_jp = dof * 13
        idx_res_policy_jt = dof * 14
        idx_ref_policy_torque = dof * 15
        idx_ref_torque_policy_jt = dof * 16
        idx_follower_cart_pos = dof * 17
        idx_follower_quat = dof * 17 + 3
        idx_leader_cart_pos = dof * 17 + 7
        idx_leader_quat = dof * 17 + 10
        idx_gripper_pos = dof * 17 + 14
        idx_gripper_vel = dof * 17 + 15
        idx_gripper_torque = dof * 17 + 16
        idx_time_to_chunk_end = dof * 17 + 17
        idx_res_time_to_chunk_end = dof * 17 + 18

        follower_quat_wxyz = list(unpacked[idx_follower_quat:idx_leader_cart_pos])
        leader_quat_wxyz = list(unpacked[idx_leader_quat:idx_gripper_pos])
        filtered_environment_torque = list(unpacked[idx_filtered_environment_torque:idx_leader_jp])
        filtered_human_torque = list(unpacked[idx_filtered_human_torque:idx_policy_jp])

        return {
            "follower_jp": list(unpacked[idx_follower_jp:idx_follower_jv]),
            "follower_jv": list(unpacked[idx_follower_jv:idx_follower_dyngravcomp_torque]),
            "follower_dyngravcomp_torque": list(unpacked[idx_follower_dyngravcomp_torque:idx_environment_torque]),
            "environment_torque": list(unpacked[idx_environment_torque:idx_filtered_environment_torque]),
            "filtered_environment_torque": filtered_environment_torque,
            "leader_jp": list(unpacked[idx_leader_jp:idx_leader_jv]),
            "leader_jv": list(unpacked[idx_leader_jv:idx_leader_dyngravcomp_torque]),
            "leader_dyngravcomp_torque": list(unpacked[idx_leader_dyngravcomp_torque:idx_human_torque]),
            "human_torque": list(unpacked[idx_human_torque:idx_filtered_human_torque]),
            "filtered_human_torque": filtered_human_torque,
            "policy_jp": list(unpacked[idx_policy_jp:idx_policy_jt]),
            "policy_jt": list(unpacked[idx_policy_jt:idx_policy_torque_scale]),
            "policy_torque_scale": list(unpacked[idx_policy_torque_scale:idx_res_policy_jp]),
            "res_policy_jp": list(unpacked[idx_res_policy_jp:idx_res_policy_jt]),
            "res_policy_jt": list(unpacked[idx_res_policy_jt:idx_ref_policy_torque]),
            "ref_policy_torque": list(unpacked[idx_ref_policy_torque:idx_ref_torque_policy_jt]),
            "ref_torque_policy_jt": list(unpacked[idx_ref_torque_policy_jt:idx_follower_cart_pos]),
            "follower_cart_pos": list(unpacked[idx_follower_cart_pos:idx_follower_quat]),
            "follower_quat_wxyz": follower_quat_wxyz,
            "leader_cart_pos": list(unpacked[idx_leader_cart_pos:idx_leader_quat]),
            "leader_quat_wxyz": leader_quat_wxyz,
            "gripper_pos": unpacked[idx_gripper_pos],
            "gripper_vel": unpacked[idx_gripper_vel],
            "gripper_torque": unpacked[idx_gripper_torque],
            "time_to_chunk_end_ns": unpacked[idx_time_to_chunk_end],
            "res_time_to_chunk_end_ns": unpacked[idx_res_time_to_chunk_end],
            "timestamp_ns": recv_ts_ns,
            # Compatibility aliases for the original consumer in main.py.
            "follower_ext_torque": filtered_environment_torque,
            "leader_ext_torque": filtered_human_torque,
            "follower_cart_rot": follower_quat_wxyz,
            "leader_cart_rot": leader_quat_wxyz,
        }

    def close(self):
        if self.sock_recv:
            self.sock_recv.close()
