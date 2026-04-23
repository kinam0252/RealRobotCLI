"""ROS2 robot interface -- Cartesian impedance control, gripper, camera."""

import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from franka_msgs.action import Move as FrankaMove
from franka_msgs.action import Grasp as FrankaGrasp

from config import (JOINT_STATE_TOPIC, JOINT_NAMES,
                    GRIPPER_JOINT_TOPIC, GRIPPER_MOVE_ACTION, GRIPPER_GRASP_ACTION,
                    CAM_BASE_RGB_TOPIC, CAM_BASE_DEPTH_TOPIC, CAM_BASE_INFO_TOPIC,
                    EE_POSE_TOPIC, TARGET_POSE_TOPIC,
                    GRIPPER_OPEN_WIDTH, GRIPPER_CLOSE_WIDTH, GRIPPER_SPEED,
                    GRIPPER_GRASP_FORCE, GRIPPER_GRASP_EPSILON,
                    CART_N_WAYPOINTS, CART_INTERVAL_S)


def _msg_to_numpy(msg):
    """Convert ROS2 Image msg to numpy array."""
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == "bgr8":
        import cv2
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif msg.encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    elif msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")


class RobotNode(Node):
    """ROS2 node: Cartesian impedance control + gripper + camera.

    Motion uses /target_pose (PoseStamped) -- same as main_groot_residual.py.
    The Franka Cartesian impedance controller handles smooth compliance.
    """

    def __init__(self):
        super().__init__("realrobot_cli")
        self._lock = threading.Lock()
        self._joint_pos = None
        self._ee_pos = None         # xyz
        self._ee_quat = None        # xyzw
        self._gripper_width = None
        self._latest_rgb = None
        self._latest_depth = None
        self._cam_K = None

        # Joint state (read-only, for FK/IK reference)
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._cb_joints, 10)

        # EE pose from Franka state broadcaster
        self.create_subscription(PoseStamped, EE_POSE_TOPIC, self._cb_ee_pose, 10)

        # Cartesian target publisher (impedance controller)
        self._target_pose_pub = self.create_publisher(PoseStamped, TARGET_POSE_TOPIC, 10)

        # Gripper state + actions
        self.create_subscription(JointState, GRIPPER_JOINT_TOPIC, self._cb_gripper, 10)
        self._gripper_move_client = ActionClient(self, FrankaMove, GRIPPER_MOVE_ACTION)
        self._gripper_grasp_client = ActionClient(self, FrankaGrasp, GRIPPER_GRASP_ACTION)

        # Camera
        self.create_subscription(Image, CAM_BASE_RGB_TOPIC, self._cb_rgb, 10)
        self.create_subscription(Image, CAM_BASE_DEPTH_TOPIC, self._cb_depth, 10)
        self.create_subscription(CameraInfo, CAM_BASE_INFO_TOPIC, self._cb_cam_info, 10)

        self.get_logger().info("RobotNode ready (Cartesian impedance + gripper + camera)")

    # -- Callbacks --

    def _cb_joints(self, msg):
        idx = {name: i for i, name in enumerate(msg.name)}
        joints = []
        for n in JOINT_NAMES:
            if n in idx:
                joints.append(msg.position[idx[n]])
        if len(joints) == 7:
            with self._lock:
                self._joint_pos = np.array(joints, dtype=np.float64)

    def _cb_ee_pose(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        with self._lock:
            self._ee_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._ee_quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)

    def _cb_gripper(self, msg):
        try:
            positions = np.array(msg.position[:2], dtype=np.float64)
            with self._lock:
                self._gripper_width = float(positions.sum())
        except Exception:
            pass

    def _cb_rgb(self, msg):
        try:
            img = _msg_to_numpy(msg)
            with self._lock:
                self._latest_rgb = img
        except Exception as e:
            self.get_logger().error(f"RGB decode error: {e}")

    def _cb_depth(self, msg):
        try:
            raw = _msg_to_numpy(msg)
            depth = raw.astype(np.float32) * 0.001 if raw.dtype == np.uint16 else raw.astype(np.float32)
            with self._lock:
                self._latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth decode error: {e}")

    def _cb_cam_info(self, msg):
        k = msg.k
        self._cam_K = np.array([
            [k[0], k[1], k[2]],
            [k[3], k[4], k[5]],
            [k[6], k[7], k[8]],
        ], dtype=np.float64)

    # -- Getters --

    def get_joints(self):
        with self._lock:
            return self._joint_pos.copy() if self._joint_pos is not None else None

    def get_ee_pose(self):
        """Returns (pos_xyz, quat_xyzw) or (None, None)."""
        with self._lock:
            if self._ee_pos is not None and self._ee_quat is not None:
                return self._ee_pos.copy(), self._ee_quat.copy()
            return None, None

    def get_gripper_width(self):
        with self._lock:
            return self._gripper_width

    def get_camera_frame(self):
        """Returns (rgb, depth, K) or (None, None, None)."""
        with self._lock:
            rgb = self._latest_rgb.copy() if self._latest_rgb is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
            K = self._cam_K.copy() if self._cam_K is not None else None
        return rgb, depth, K

    # -- Cartesian target publishing --

    def publish_target_pose(self, pos, quat_xyzw):
        """Publish Cartesian target to impedance controller."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self._target_pose_pub.publish(msg)

    def send_stop(self):
        """Freeze robot by republishing current EE pose."""
        pos, quat = self.get_ee_pose()
        if pos is not None:
            self.publish_target_pose(pos, quat)

    def wait_for_convergence(self, target_pos, target_quat_xyzw,
                             pos_thresh_mm=5.0, ori_thresh_deg=5.0,
                             timeout_sec=5.0):
        """Block until EE converges to target or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            pos, quat = self.get_ee_pose()
            if pos is None:
                continue
            pos_err = np.linalg.norm(pos - target_pos) * 1000.0
            q1 = quat / np.linalg.norm(quat)
            q2 = target_quat_xyzw / np.linalg.norm(target_quat_xyzw)
            dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
            ori_err = np.degrees(2.0 * np.arccos(dot))
            if pos_err < pos_thresh_mm and ori_err < ori_thresh_deg:
                return True
        return False

    # -- Gripper commands --

    def open_gripper(self):
        """Open gripper fully using FrankaMove action."""
        goal = FrankaMove.Goal()
        goal.width = GRIPPER_OPEN_WIDTH
        goal.speed = GRIPPER_SPEED
        self._gripper_move_client.wait_for_server(timeout_sec=5.0)
        future = self._gripper_move_client.send_goal_async(goal)
        return future

    def close_gripper(self):
        """Close gripper with force using FrankaGrasp action."""
        goal = FrankaGrasp.Goal()
        goal.width = GRIPPER_CLOSE_WIDTH
        goal.epsilon.inner = GRIPPER_GRASP_EPSILON
        goal.epsilon.outer = GRIPPER_GRASP_EPSILON
        goal.speed = GRIPPER_SPEED
        goal.force = GRIPPER_GRASP_FORCE
        self._gripper_grasp_client.wait_for_server(timeout_sec=5.0)
        future = self._gripper_grasp_client.send_goal_async(goal)
        return future


def wait_for_joint_state(node, timeout_s=10.0):
    """Block until joint state is available. Returns True on success."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_joints() is not None:
            return True
    return False


def wait_for_ee_pose(node, timeout_s=10.0):
    """Block until EE pose is available. Returns True on success."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        pos, _ = node.get_ee_pose()
        if pos is not None:
            return True
    return False


def wait_for_camera(node, timeout_s=10.0):
    """Block until camera data is available. Returns True on success."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
        rgb, depth, K = node.get_camera_frame()
        if rgb is not None and depth is not None and K is not None:
            return True
    return False


def slow_cartesian_move(node, target_pos, target_quat_xyzw,
                        n_waypoints=None, interval=None):
    """Smoothly move EE to target via Cartesian interpolation + SLERP.

    Same approach as slow_move_to() in main_groot_residual.py:
    - Linear interpolation for position
    - SLERP for orientation
    - Publish PoseStamped waypoints to impedance controller
    - Wait for convergence at end
    """
    from scipy.spatial.transform import Rotation, Slerp

    if n_waypoints is None:
        n_waypoints = CART_N_WAYPOINTS
    if interval is None:
        interval = CART_INTERVAL_S

    rclpy.spin_once(node, timeout_sec=0.05)
    start_pos, start_quat = node.get_ee_pose()
    if start_pos is None:
        return False

    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64)

    # Build SLERP interpolator
    rots = Rotation.from_quat([start_quat, target_quat_xyzw])  # xyzw
    slerp = Slerp([0.0, 1.0], rots)

    for i in range(1, n_waypoints + 1):
        t = i / n_waypoints
        wp_pos = start_pos + t * (target_pos - start_pos)
        wp_quat = slerp([t]).as_quat()[0]  # xyzw

        node.publish_target_pose(wp_pos, wp_quat)
        time.sleep(interval)
        rclpy.spin_once(node, timeout_sec=0.001)

    # Wait for convergence
    node.wait_for_convergence(target_pos, target_quat_xyzw, timeout_sec=5.0)
    return True


def move_to_cartesian(node, target_pos, target_quat_xyzw,
                      n_waypoints=None, interval=None):
    """Move to a single Cartesian target. Returns True on success."""
    return slow_cartesian_move(node, target_pos, target_quat_xyzw,
                               n_waypoints, interval)


def move_through_cartesian_waypoints(node, cart_waypoints, n_interp=None,
                                     interval=None):
    """Move through a list of Cartesian waypoints [(pos, quat_xyzw), ...].

    Each waypoint pair is interpolated smoothly.
    """
    for i, (pos, quat) in enumerate(cart_waypoints):
        print(f"  Segment {i+1}/{len(cart_waypoints)}...")
        ok = slow_cartesian_move(node, pos, quat, n_interp, interval)
        if not ok:
            return False, i
    return True, len(cart_waypoints)
