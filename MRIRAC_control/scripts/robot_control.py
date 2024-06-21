#!/usr/bin/env python

import numpy as np
import copy
import tiktoken
import re
import argparse
import os
import json
from task_decomposition import ChatGPT
from camera_calibration import Camera, ARUCO_DICT
from scipy.spatial.transform import Rotation
import rospy
import moveit_commander
import actionlib
from moveit_commander import MoveGroupCommander
from moveit_commander import MoveItCommanderException
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    JointConstraint,
    BoundingVolume,
)
from geometry_msgs.msg import Pose, Vector3
from shape_msgs.msg import SolidPrimitive
from franka_gripper.msg import MoveGoal, MoveAction, GraspGoal, GraspAction, HomingGoal, HomingAction

# Initialize encoding and load credentials
enc = tiktoken.get_encoding("cl100k_base")
with open('secrets.json') as f:
    credentials = json.load(f)

# Define directory paths
dir_system = './system'
dir_prompt = './prompt'
dir_query = './query'
prompt_load_order = ['prompt_role',
                     'input_environment',
                     'prompt_function',
                     'prompt_environment',
                     'prompt_output_format',
                     'prompt_example']

# Define robot and camera parameters
ROBOT = "fr3"
CAMERA = "right"
CAM_IP = "192.168.0.200"
GRASP_WIDTH = 0.5
Z_ABOVE_BOARD = 0.1
Z_TO_PIECE = 0.1

# Predefined joint configurations
LOW_CAM_JOINTS = [0.014270682464896165, -0.9818395060681135, 0.048710415021898, -2.569057076198659,
                  -0.009787638639837783, 1.6403438004351425, 0.8450968248276278, 0.0407734178006649, 0.0407734178006649]
GIVE_JOINTS = [-0.2212685437466231, -0.5117288414050377, 0.1975034868122401, -2.777970915259786, 0.23110102446657219,
               2.795546833871574, 0.5554736904880095]
HIGH_CAM_JOINTS = [-0.10316570739566021, 0.021436616563050445, -0.22314194236226542, -2.5075683853755493,
                   -0.13751088517681878, 2.5268053995945454, -1.0160112798778322, 0.03986130654811859,
                   0.03986130654811859]

Z_DROP = 0.05
X_OFFSET = 0.0
Y_OFFSET = 0.0

GOALWIDTH = 0.005
GOALSPEED = 1
GOALFORCE = 10
board_height = 0.01

class RobotControl(object):
    robot_acc = 0.2
    robot_vel = 0.2

    def __init__(self, cam=True, camera_index=4):
        """
        Initialize the RobotControl object.

        Args:
            cam (bool): Flag to indicate if camera is used.
            camera_index (int): Index of the camera to use.
        """
        self.execution = True
        rospy.on_shutdown(self.shutdown)
        self.commander = MoveGroupCommander(ROBOT + "_arm")
        self.commander.allow_replanning(True)
        self.commander.set_planning_time(10)
        self.commander.set_planner_id("RRTstar")

        # Initialize gripper clients
        self.move_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        self.gripper_is_open = True
        self.ignore_grasp_error = True

        # Predefined configurations
        self._R_flange2zed = [45, 2.5, 0]
        if CAMERA == "left":
            self._T_flange2zed = [-0.06, -0.06, 0.02]
        elif CAMERA == "right":
            self._T_flange2zed = [0.06, -0.06, 0.02]

        # Initialize positions and configurations
        self._zed_position_world = np.zeros(3)
        self._markers_world = np.zeros((1, 3))
        self.board_grid = dict()
        self.board_corners = np.zeros((4, 3))
        self.cam_on = False
        if cam:
            self.cam_on = True
            self.camera = Camera(camera_index=camera_index, name="1", marker_size=0.04)

        # Set ROS parameters
        rospy.set_param('x_offset', X_OFFSET)
        rospy.set_param('y_offset', Y_OFFSET)

        self.constraints = None
        self.init_constraints()

    def update_camera_index(self, camera_index):
        """Update the camera index."""
        if self.cam_on:
            self.camera.camera_index = camera_index

    def shutdown(self):
        """Shutdown the RobotControl node."""
        rospy.loginfo("Stopping the RobotControl node")
        if hasattr(self, "cam_on") and self.cam_on:
            self.camera.close()
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def homing(self):
        """Homing the gripper."""
        homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        homing_client.wait_for_server()
        homing_client.send_goal(HomingGoal())
        homing_client.wait_for_result(rospy.Duration.from_sec(5.0))

    def update_world_positions(self):
        """Update world positions of markers based on camera detection."""
        current_pose = self.commander.get_current_pose().pose

        P = current_pose.position
        Q = current_pose.orientation

        R_arm = Rotation.from_quat([Q.x, Q.y, Q.z, Q.w])
        R_zed = Rotation.from_euler("ZXY", self._R_flange2zed, degrees=True)

        if self.cam_on and len(self.camera.detected_markers) > 0:
            self._markers_world = {}
            rospy.loginfo("Using detected markers")
            for marker_id, marker_prop in self.camera.detected_markers.items():
                marker = marker_prop["pos2camera"]
                zed2flange = R_zed.apply(marker) + R_zed.apply(self._T_flange2zed)
                flange2base = R_arm.apply(zed2flange) + np.array([P.x, P.y, P.z])

                # Calculating offsets
                flange2base[0] += 0.12
                flange2base[1] -= 0.02
                flange2base[2] -= 0.18

                self._markers_world[marker_id] = flange2base
                rospy.logdebug(f"Marker {marker_id} world position: {flange2base}")
        else:
            raise Exception("No markers available when updating!")

        transformed_markers = self._markers_world

        return transformed_markers

    def marker_update(self, side=CAMERA):
        """Update markers detected by the camera."""
        if self.cam_on:
            current_frame = self.camera.get_img(side)
            corners, ids = self.camera.detect_markers(frame=current_frame, ARUCO_DICT=ARUCO_DICT, show=True,
                                                      save_image=True)
            attempt = 0
            while ids is None:
                rospy.sleep(0.05)
                corners, ids = self.camera.detect_markers(frame=self.camera.get_img(side), ARUCO_DICT=ARUCO_DICT,
                                                          save_image=True)
                attempt += 1
                rospy.sleep(0.05)
                if attempt > 20:
                    raise ValueError(f"Expected 1 markers but found markers {ids} in the current image.")
            self.camera.locate_markers(corners, ids)

    def all_update(self):
        """Update markers and world positions."""
        self.marker_update()
        self.update_world_positions()

    def workspace_constraint(self):
        """Define workspace constraints for the robot arm."""
        box_constraint = PositionConstraint()
        box_constraint.header = self.commander.get_current_pose().header
        box_constraint.link_name = ROBOT + "_link8"
        box_constraint.target_point_offset = Vector3(0, 0, 0.15)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.2, 1.0, 1.0]

        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0
        pose.position.z = 0.5
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1

        bounds = BoundingVolume()
        bounds.primitives = [box]
        bounds.primitive_poses = [pose]
        box_constraint.constraint_region = bounds
        box_constraint.weight = 1

        self.constraints.position_constraints.append(box_constraint)

    def upright_constraints(self):
        """Define upright constraints for the robot arm."""
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = self.commander.get_current_pose().header
        orientation_constraint.link_name = ROBOT + "_link8"
        orientation_constraint.orientation.x = 0.9238795
        orientation_constraint.orientation.y = -0.3826834
        orientation_constraint.orientation.z = 0
        orientation_constraint.orientation.w = 0
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1

        self.constraints.orientation_constraints.append(orientation_constraint)

        joint_constraints = [
            JointConstraint(
                joint_name=f"{ROBOT}_joint1", position=0, tolerance_above=0.785398, tolerance_below=0.785398, weight=0.5
            ),
            JointConstraint(
                joint_name=f"{ROBOT}_joint7",
                position=0.785398,
                tolerance_above=0.785398,
                tolerance_below=0.785398,
                weight=0.5,
            ),
        ]

        self.constraints.joint_constraints.extend(joint_constraints)

    def start_camera_for_short_duration(self):
        """Run the camera for a short duration and detect markers."""
        rospy.loginfo("Starting camera for a short duration...")
        self.move_camera_state()
        start_time = rospy.Time.now()
        duration = rospy.Duration(1)
        while rospy.Time.now() - start_time < duration:
            self.marker_update()
            rospy.sleep(0.1)
        rospy.loginfo("Completed camera run for a short duration.")

    def init_constraints(self, workspace=True, upright=True):
        """Initialize constraints for the robot arm."""
        self.commander.clear_path_constraints()
        self.constraints = Constraints()
        self.constraints.name = "robot constraints"
        if workspace:
            self.workspace_constraint()
        if upright:
            self.upright_constraints()

        self.commander.set_path_constraints(self.constraints)

    def move_ready_state(self, acc=None, vel=None):
        """Move the robot to the ready state."""
        if acc is None:
            acc = RobotControl.robot_acc
        if vel is None:
            vel = RobotControl.robot_vel
        self.move_gripper("open")
        self.commander.set_named_target("ready")
        self.commander.set_max_acceleration_scaling_factor(acc)
        self.commander.set_max_velocity_scaling_factor(vel)
        self.commander.go(wait=True)

    def move_camera_state(self, acc=None, vel=None, joint_position=LOW_CAM_JOINTS):
        """Move the robot to the camera state."""
        if acc is None:
            acc = RobotControl.robot_acc
        if vel is None:
            vel = RobotControl.robot_vel
        joint_goal = joint_position[0:7]
        self.commander.set_max_acceleration_scaling_factor(acc)
        self.commander.set_max_velocity_scaling_factor(vel)
        self.commander.go(joint_goal, wait=True)

    def move_camera_state_low(self, acc=None, vel=None):
        """Move the robot to the low camera state."""
        self.move_camera_state(acc, vel, joint_position=LOW_CAM_JOINTS)

    def move_camera_state_place(self, acc=None, vel=None):
        """Move the robot to the place camera state."""
        self.move_camera_state(acc, vel, joint_position=HIGH_CAM_JOINTS)

    def execute_path(self, waypoints, acc=None, vel=None):
        """Execute a path based on the provided waypoints."""
        acc = acc if acc is not None else RobotControl.robot_acc
        vel = vel if vel is not None else RobotControl.robot_vel

        self.commander.set_max_acceleration_scaling_factor(acc)
        self.commander.set_max_velocity_scaling_factor(vel)

        jump_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        max_attempts = 10
        fraction = 0.0

        for jump_threshold in jump_thresholds:
            plan, fraction = self.commander.compute_cartesian_path(
                waypoints, 0.01, jump_threshold, avoid_collisions=True)
            rospy.loginfo(f"Computed Cartesian path with fraction: {fraction}, jump_threshold: {jump_threshold}")
            if fraction >= 1.0:
                break

        attempts = 0
        while fraction < 1.0 and attempts < max_attempts:
            plan, fraction = self.commander.compute_cartesian_path(
                waypoints, 0.01, 0.0, path_constraints=self.constraints)
            attempts += 1
            rospy.loginfo(f"Attempt {attempts}: Computed fraction {fraction}")

        if fraction >= 0.6:
            rospy.loginfo("Executing planned Cartesian path")
            plan = self.commander.retime_trajectory(
                self.commander.get_current_state(), plan,
                velocity_scaling_factor=vel,
                acceleration_scaling_factor=acc)
            return self.commander.execute(plan, wait=True)
        else:
            rospy.logwarn(f"Failed to compute a complete path, fraction achieved: {fraction}")
            return False

    def move_gripper(self, command="open"):
        """Move the gripper to open or close position."""
        self.move_client.wait_for_server()
        if command == "close":
            goal = MoveGoal(width=-GRASP_WIDTH, speed=GOALSPEED)
        elif command == "open":
            goal = MoveGoal(width=GRASP_WIDTH, speed=GOALSPEED)
        else:
            raise ValueError("Grasp goal should be 'close' or 'open'")
        self.move_client.send_goal(goal)
        if not self.move_client.wait_for_result(rospy.Duration.from_sec(5.0)):
            rospy.logerr(f"Action server timed out while trying to {command} the gripper.")
            return False  # Indicate failure
        result = self.move_client.get_result()
        if result is None:
            rospy.logerr("No result received from the action server.")
            return False  # Indicate failure
        return result.success

    def grasp(self, command):
        """Grasp an object with the gripper."""
        self.grasp_client.wait_for_server()
        if command == "close":
            if self.gripper_is_open:  # Check if the gripper is open before trying to close it.
                goal = GraspGoal(width=GOALWIDTH, speed=GOALSPEED, force=GOALFORCE)
                goal.epsilon.inner = goal.epsilon.outer = 0.1

                self.grasp_client.send_goal(goal)
                self.grasp_client.wait_for_result(rospy.Duration.from_sec(5.0))
                res = self.grasp_client.get_result().success
                if res:
                    self.gripper_is_open = False  # Update state only on successful operation
            else:
                rospy.loginfo("Gripper is already closed.")
        elif command == "open":
            res = self.move_gripper("open")
            if res:
                self.gripper_is_open = True
        else:
            raise ValueError("Grasp goal should be 'close' or 'open'")
        if not self.ignore_grasp_error and not res:
            raise rospy.ROSInterruptException(f"{command} failed")
        return res

    def move_to_pose(self, target_pose):
        """Move the robot to a specific pose."""
        waypoints = [copy.deepcopy(target_pose)]
        if self.execute_path(waypoints):
            rospy.loginfo(
                f"Moved to position: {target_pose.position.x}, {target_pose.position.y}, {target_pose.position.z}")
            return True
        else:
            rospy.logerr("Failed to reach the target position.")
            return False

    def move_to_marker(self, marker_id):
        """Move the robot to the specified ArUco marker."""
        marker_pos = self._markers_world[marker_id]
        target_pose = self.commander.get_current_pose().pose
        target_pose.position.x = marker_pos[0]
        target_pose.position.y = marker_pos[1]
        target_pose.position.z = marker_pos[2]
        return self.move_to_pose(target_pose)

    def pick(self, marker_id):
        """Pick up an object at a specified marker."""
        target_pose = self.commander.get_current_pose().pose
        target_pose.position.z -= 0.07
        if self.move_to_pose(target_pose):
            self.grasp("close")
            rospy.sleep(1)
            target_pose.position.z += 0.15
            self.move_to_pose(target_pose)
            rospy.loginfo(f"Object picked at marker {marker_id}")
        else:
            rospy.logerr("Failed to pick the object at marker {marker_id}")

    def place(self, marker_id):
        """Place an object at a specified marker."""
        target_pose = self.commander.get_current_pose().pose
        target_pose.position.x += 0.045
        self.move_to_pose(target_pose)
        if self.move_to_pose(target_pose):
            self.grasp("close")
            rospy.sleep(1)
            rospy.loginfo(f"Object placed at marker {marker_id}")
        else:
            rospy.logerr("Failed to place the object at marker {marker_id}")

    def push(self, marker_id, directions):
        """Push an object at a specified marker."""
        target_pose = self.commander.get_current_pose().pose
        target_pose.position.z -= 0.15
        if not self.move_to_pose(target_pose):
            rospy.logerr("Failed to lower to the object.")
            return

        self.grasp("close")
        rospy.sleep(1)  # Wait to ensure grip has stabilized

        for direction in directions:
            if direction == 'forward':
                target_pose.position.x += 0.05
            elif direction == 'backward':
                target_pose.position.x -= 0.05
            elif direction == 'left':
                target_pose.position.y += 0.05
            elif direction == 'right':
                target_pose.position.y -= 0.05

            if not self.move_to_pose(target_pose):
                rospy.logerr(f"Failed to push the object {direction} at marker {marker_id}")
                break
            rospy.loginfo(f"Object pushed {direction} at marker {marker_id}")

        self.grasp("open")
        target_pose.position.z += 0.1  # Raise slightly after release
        self.move_to_pose(target_pose)

    def opendrawer(self, marker_id):
        """Open a drawer at a specified marker."""
        self.commander.set_max_velocity_scaling_factor(0.05)
        self.commander.set_max_acceleration_scaling_factor(0.05)

        target_pose = self.commander.get_current_pose().pose
        target_pose.position.z -= 0.03
        if self.move_to_pose(target_pose):
            self.grasp("close")
            rospy.sleep(1)
            rospy.loginfo(f"Drawer opened: {marker_id}")

            target_pose.position.z += 0.1
            target_pose.position.y -= 0.1
            if self.move_to_pose(target_pose):
                rospy.sleep(1)
                self.grasp("open")
            else:
                rospy.logerr("Failed to move up after opening the drawer.")
        else:
            rospy.logerr(f"Failed to open the drawer: {marker_id}")

    def run_task_sequence(self, json_path):
        """
        Execute a sequence of tasks specified in a JSON file using robot motion primitives.

        Parameters:
        - json_path (str): The file path to the JSON file containing the task sequence.
        """
        with open(json_path, 'r') as file:
            data = json.load(file)
            task_sequence = data['task_cohesion']['task_sequence']

        for task in task_sequence:
            match = re.match(r"(\w+)\((.*)\)", task)
            if match:
                function_name, args_str = match.groups()
                args = eval(f"[{args_str}]")

                if hasattr(self, function_name):
                    func = getattr(self, function_name)
                    print(f'Executing {function_name} with arguments {args}...')
                    func(*args)
                else:
                    print(f'Function {function_name} not found in RobotControl.')
            else:
                print(f"Could not parse the task: {task}")

    def run(self):
        """Run the main control loop."""
        self.all_update()
        self.init_constraints()

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--scenario',
            type=str,
            required=True,
            help='Scenario name (see the code for details)')
        args = parser.parse_args()
        scenario_name = args.scenario

        image_path = os.path.join('..', 'out', 'demo', 'detected_markers.jpg')

        aimodel = ChatGPT(credentials, prompt_load_order=prompt_load_order)
        environment_json = aimodel.generate_environment(image_path)

        try:
            environment = json.loads(environment_json)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)

        while True:
            user_feedback = input('Feedback for environment (return empty if satisfied, type "q" to quit): ')
            if user_feedback.lower() == 'q':
                exit()
            elif user_feedback == '':
                break
            else:
                environment_json = aimodel.generate_environment(image_path, is_user_feedback=True,
                                                                feedback_message=user_feedback)
                environment = json.loads(environment_json)
                print("Refined environment:", json.dumps(environment, indent=4))

        if not os.path.exists(os.path.join('./out', scenario_name)):
            os.makedirs(os.path.join('./out', scenario_name))

        instructions = [input("Enter the arm instructions: ")]
        for i, instruction in enumerate(instructions):
            response = aimodel.generate(instruction, environment, is_user_feedback=False)
            while True:
                user_feedback = input('User feedback (return empty if satisfied, type "q" to quit): ')
                if user_feedback.lower() == 'q':
                    exit()
                elif user_feedback == '':
                    break
                else:
                    response = aimodel.generate(user_feedback, environment, is_user_feedback=True)

            aimodel.dump_json(f'./out/{scenario_name}/{i}')

        json_path = os.path.join('..', 'out', 'demo', '0.json')
        self.run_task_sequence(json_path)


if __name__ == "__main__":
    try:
        rospy.init_node("robot_control_node", anonymous=True, log_level=rospy.INFO)
        robo = RobotControl(cam=True)
        rospy.sleep(0.5)
        robo.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
