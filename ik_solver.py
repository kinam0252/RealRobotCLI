"""IK solver wrapper — uses existing ik_dh_minchange from Mujoco/utils."""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from config import IK_UTILS_DIR, WORKSPACE_MIN, WORKSPACE_MAX, \
    TOP_DOWN_ROT, PRE_GRASP_Z_OFFSET, APPROACH_Z_OFFSET

# Import IK module from existing codebase (no copy)
if IK_UTILS_DIR not in sys.path:
    sys.path.insert(0, IK_UTILS_DIR)

from ik_dh_minchange import (
    ik_solve_min_change, fk_dh, pose_from_T, rot_to_quat,
    DH_PARAMS_FR3, TOOL_T_HAND,
)


def check_workspace(pos):
    """Check if position is within safe workspace bounds."""
    pos = np.asarray(pos)
    if np.any(pos < WORKSPACE_MIN) or np.any(pos > WORKSPACE_MAX):
        return False
    return True


def get_current_ee_pose(current_joints):
    """Compute current EE pose from joint angles via FK.

    Returns:
        pos (3,), quat_wxyz (4,)
    """
    T = fk_dh(current_joints, DH_PARAMS_FR3, dh_type="modified", tool_T=TOOL_T_HAND)
    pos, quat_wxyz = pose_from_T(T)
    return pos, quat_wxyz


def compute_pregrasp_pose(cube_pos):
    """Compute pre-grasp target: directly above the cube, gripper pointing down.

    Returns:
        target_pos (3,), target_quat_wxyz (4,)
    """
    target_pos = np.array([cube_pos[0], cube_pos[1],
                           cube_pos[2] + PRE_GRASP_Z_OFFSET])
    target_quat_wxyz = rot_to_quat(TOP_DOWN_ROT)
    return target_pos, target_quat_wxyz


def compute_approach_pose(cube_pos):
    """Compute safe approach waypoint: higher above the cube.

    Returns:
        target_pos (3,), target_quat_wxyz (4,)
    """
    target_pos = np.array([cube_pos[0], cube_pos[1],
                           cube_pos[2] + APPROACH_Z_OFFSET])
    target_quat_wxyz = rot_to_quat(TOP_DOWN_ROT)
    return target_pos, target_quat_wxyz


def solve_ik(target_pos, target_quat_wxyz, current_joints,
             max_iters=400, tol_pos=1e-4, tol_ori=5e-4):
    """Solve IK for a target pose.

    Returns:
        IKResult with .q (joint angles), .success, .pos_err, .ori_err
    """
    result = ik_solve_min_change(
        q_current=np.asarray(current_joints, dtype=np.float64),
        p_target=np.asarray(target_pos, dtype=np.float64),
        q_target=np.asarray(target_quat_wxyz, dtype=np.float64),
        dh_params=DH_PARAMS_FR3,
        dh_type="modified",
        tool_T=TOOL_T_HAND,
        max_iters=max_iters,
        tol_pos=tol_pos,
        tol_ori=tol_ori,
        lambda_dls=1e-2,
        k_null=0.08,
    )
    return result


def plan_pregrasp_trajectory(cube_pos, current_joints):
    """Plan a 2-waypoint trajectory: approach → pre-grasp.

    Returns:
        list of (7,) joint arrays, or None on failure.
        error_msg: str or None
    """
    # Safety: check both target positions
    approach_pos, approach_quat = compute_approach_pose(cube_pos)
    pregrasp_pos, pregrasp_quat = compute_pregrasp_pose(cube_pos)

    for label, pos in [("approach", approach_pos), ("pre-grasp", pregrasp_pos)]:
        if not check_workspace(pos):
            return None, f"{label} position {pos.round(3)} is outside safe workspace"

    # Solve approach waypoint
    res1 = solve_ik(approach_pos, approach_quat, current_joints)
    if not res1.success:
        return None, (f"IK failed for approach: "
                      f"pos_err={res1.pos_err:.4e}, ori_err={res1.ori_err:.4e}")

    # Solve pre-grasp (seeded from approach)
    res2 = solve_ik(pregrasp_pos, pregrasp_quat, res1.q)
    if not res2.success:
        return None, (f"IK failed for pre-grasp: "
                      f"pos_err={res2.pos_err:.4e}, ori_err={res2.ori_err:.4e}")

    return [res1.q, res2.q], None


def plan_target_trajectory(target_pos, current_joints):
    """Plan a 2-waypoint trajectory: safe-above → target.

    The safe-above waypoint is at the same XY as target but Z raised by
    APPROACH_Z_OFFSET (relative to target Z), clamped to at least 15cm.

    Returns:
        list of (7,) joint arrays, or None on failure.
        target_pos: the final target position
        error_msg: str or None
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat = rot_to_quat(TOP_DOWN_ROT)

    # Safe approach: above target, at least 15cm high
    safe_z = max(target_pos[2] + APPROACH_Z_OFFSET, 0.15)
    approach_pos = np.array([target_pos[0], target_pos[1], safe_z])

    for label, pos in [("approach", approach_pos), ("target", target_pos)]:
        if not check_workspace(pos):
            return None, target_pos, f"{label} position {pos.round(3)} is outside safe workspace"

    # Solve approach
    res1 = solve_ik(approach_pos, target_quat, current_joints)
    if not res1.success:
        return None, target_pos, (f"IK failed for approach: "
                                  f"pos_err={res1.pos_err:.4e}, ori_err={res1.ori_err:.4e}")

    # Solve target (seeded from approach)
    res2 = solve_ik(target_pos, target_quat, res1.q)
    if not res2.success:
        return None, target_pos, (f"IK failed for target: "
                                  f"pos_err={res2.pos_err:.4e}, ori_err={res2.ori_err:.4e}")

    return [res1.q, res2.q], target_pos, None
