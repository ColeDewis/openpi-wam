from wam.inference.udp_handler import TeleopUDPHandler
from wam.inference.recorder_receiver import RecorderReceiver


class WAMManager:
    def __init__(
        self,
        follower_ip: str,
        send_port: int,
        follower_recv_port: int,
        leader_recv_port: int,
        dof: int = 7,
    ):
        # 1. Inputs (Proprioception)
        self.follower_listener = RecorderReceiver(follower_ip, follower_recv_port, dof)
        self.leader_listener = RecorderReceiver(follower_ip, leader_recv_port, dof)

        # 2. Outputs (Actuation)
        self.follower_sender = TeleopUDPHandler(follower_ip, send_port, DOF=dof)

    def get_latest_states(self):
        """Returns the state of both robots for the Orchestrator."""
        return {
            "follower_state": self.follower_listener.receive_latest_data(),
            "leader_state": self.leader_listener.receive_latest_data(),
        }

    def send_action(self, jp, jv, torque):
        """Passes the action chunk to the sender."""
        self.follower_sender.send_data(jp, jv, torque)
