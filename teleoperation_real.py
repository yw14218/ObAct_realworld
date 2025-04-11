from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from threading import Event
from filter import OneEuroFilter
import socket
import queue
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import ik_solver, compose_homogeneous_matrix_euler

ready_to_receive = Event()
ready_to_execute = Event()
ready_to_receive.set()
previous_guess = None
GRIPPER_CLOSED = False

class UDPServer:
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._ip, self._port))
        self._socket.setblocking(False)
        self._running = False

        # queue for storing received data
        self.cmd_buffer = queue.Queue(maxsize=1)
        self.latest_cmd = None
        print(f"UDP server started at {self._ip}:{self._port}")
    
    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()
    
    def stop(self):
        self._running = False
        # kill the thread
        self.thread.join()
        self._socket.close()
    
    def _listen(self):
        cnt = 0
        while self._running:
            received_data = None
            try:
                # print("Waiting for event...")
                ready_to_receive.wait()
                while True:
                    # print("Waiting for data...")
                    data, addr = self._socket.recvfrom(2048)
                    received_data = np.frombuffer(data, dtype=np.float32)
                    cnt += 1
                    # print(f"Received data: {received_data} from {addr}")
                    pass
                    # self.cmd_buffer.put(received_data)
            except BlockingIOError:
                # print("No data received")
                pass
            if received_data is not None:
                self.latest_cmd = received_data
                print(received_data)
                ready_to_receive.clear()
                ready_to_execute.set()
            else:
                # print("No data received")
                pass


def move_arm(bot, goal_pos, goal_rot, gripper_status):
    """Move the arm to a specified position."""
    global GRIPPER_CLOSED

    ee_pose = np.eye(4)
    ee_pose = compose_homogeneous_matrix_euler(goal_pos, goal_rot)
    qpos = ik_solver.ik(ee_pose, qinit=bot.arm.get_joint_positions())

    if qpos is not None:
        bot.arm.set_joint_positions(qpos, moving_time=0.1, accel_time=0.05, blocking=False)
    else:
        print("IK failed.")
    # Move the arm using previous guess if available
    # solution, success = bot.arm.set_ee_pose_components(
    #     *goal_pos, *goal_rot, blocking=False, moving_time=0.2, accel_time=0.1,
    #     custom_guess=bot.arm.get_joint_positions(),
    # )

    if gripper_status > 0.1 and GRIPPER_CLOSED == False:
        bot.gripper.grasp()
        GRIPPER_CLOSED = True
    elif gripper_status <= 0.1 and GRIPPER_CLOSED == True:
        bot.gripper.release()
        GRIPPER_CLOSED = False

def main():
    bot_1 = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='arm_1',
        group_name='arm',
        gripper_name='gripper',
        moving_time=5,
        accel_time=0.3
    )

    robot_startup()

    # bot_1.arm.go_to_sleep_pose(moving_time=2, accel_time=0.3)
    # raise

    # # start a udp server
    udp_server = UDPServer(ip="127.0.0.1", port=8006)
    udp_server.start()
    # bot_1.gripper.release()

    # input("Press Enter to grasp...")
    bot_1.gripper.release()
    # bot_1.gripper.set_pressure(1)
    bot_1.arm.set_joint_positions([0, -0.72, 0.59, 0, 1.02, 0], moving_time=2, accel_time=0.3)

    bot_1.arm.moving_time=0.2


    # Filter for smoothing the gripper commands
    filter = OneEuroFilter(min_cutoff=0.01, beta=10.0)
    filter_rot = OneEuroFilter(min_cutoff=0.01, beta=10.0)
    current_gripper_status = 0
    initial_gripper_pose = bot_1.arm.get_ee_pose().copy()
    goal_pos = initial_gripper_pose[:3,3].copy()
    goal_rot = R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()

    while True:
        if ready_to_execute.is_set():
            if udp_server.latest_cmd is not None:
                # cmd = udp_server.cmd_buffer.get()
                cmd = udp_server.latest_cmd
                goal_pos = cmd[:3] + initial_gripper_pose[:3,3].copy()
                gripper_status = cmd[3]
                current_gripper_status = min(2 * gripper_status, 1)
                gripper_rot = (cmd[4]) / 180 * np.pi
                goal_rot = np.array([0, 0, gripper_rot]) + R.from_matrix(initial_gripper_pose[:3,:3]).as_euler("xyz").copy()
                t = time.time()
                goal_pos = filter(t, goal_pos)
                goal_rot = filter_rot(t, goal_rot)
                move_arm(bot_1, goal_pos, goal_rot, current_gripper_status)
                udp_server.latest_cmd = None
                ready_to_receive.set()
                ready_to_execute.clear()
        else:
            t = time.time()
            goal_pos = filter(t, goal_pos)
            move_arm(bot_1, goal_pos, goal_rot, current_gripper_status)
  
    robot_shutdown()

if __name__ == '__main__':
    main()