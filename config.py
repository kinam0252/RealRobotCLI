"""RealRobotCLI configuration — paths, safety limits, grasp parameters."""

import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
KINAM_ROOT = os.path.expanduser("~/kinam_dev")
FP_DIR = os.path.join(KINAM_ROOT, "FoundationPose")
CALIB_PATH = os.path.join(KINAM_ROOT, "Mujoco/preproc/data/calib_result_base.calib")
IK_UTILS_DIR = os.path.join(KINAM_ROOT, "Mujoco/utils")

# FoundationPose assets
FP_CUBE_MESH = os.path.join(FP_DIR, "box_4x4x8cm.obj")
FP_VENV = os.path.join(FP_DIR, "venv")
FP_VENV_SAM = os.path.join(FP_DIR, "venv_sam")

# FoundationPose shared memory (for detect skill cache)
FP_SHM_PATH = "/dev/shm/fp_cube_pose.json"
FP_MAX_AGE_S = 2.0  # pose older than this is stale

# ── ROS2 Topics ───────────────────────────────────────────────────────
JOINT_STATE_TOPIC = "/franka/joint_states"
TARGET_POSE_TOPIC = "/target_pose"           # Cartesian impedance target
EE_POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"
GRIPPER_JOINT_TOPIC = "/franka_gripper/joint_states"
GRIPPER_MOVE_ACTION = "/franka_gripper/move"
GRIPPER_GRASP_ACTION = "/franka_gripper/grasp"
CAM_BASE_RGB_TOPIC = "/cam_base/camera/color/image_raw"
CAM_BASE_DEPTH_TOPIC = "/cam_base/camera/aligned_depth_to_color/image_raw"
CAM_BASE_INFO_TOPIC = "/cam_base/camera/color/camera_info"
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# ── Safety workspace (robot base frame, metres) ──────────────────────
WORKSPACE_MIN = np.array([0.15, -0.45, 0.00])
WORKSPACE_MAX = np.array([0.75,  0.45,  0.55])

# ── Grasp parameters ─────────────────────────────────────────────────
PRE_GRASP_Z_OFFSET = 0.05   # 5 cm above cube for pre-grasp
APPROACH_Z_OFFSET = 0.10    # 10 cm above for safe approach waypoint

# Top-down grasp orientation: gripper pointing straight down
TOP_DOWN_ROT = np.array([
    [ 1.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=np.float64)

# ── Gripper parameters ───────────────────────────────────────────────
GRIPPER_OPEN_WIDTH = 0.08     # fully open (metres)
GRIPPER_CLOSE_WIDTH = 0.02    # close target for grasping
GRIPPER_SPEED = 0.1           # m/s
GRIPPER_GRASP_FORCE = 50.0    # Newtons
GRIPPER_GRASP_EPSILON = 0.04  # inner/outer tolerance

# ── Motion parameters (Cartesian impedance) ──────────────────────────
CART_N_WAYPOINTS = 50        # interpolation steps (same as main_groot_residual.py)
CART_INTERVAL_S = 0.08       # seconds between waypoints (smooth compliance)

# ── Home joint position (safe resting pose) ──────────────────────────
HOME_JOINTS = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
