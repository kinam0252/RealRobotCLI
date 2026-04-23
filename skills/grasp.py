"""Skill: grasp -- Full grasp sequence via Cartesian impedance control."""

import os
import time
import numpy as np
import rclpy

from config import (PRE_GRASP_Z_OFFSET, TOP_DOWN_ROT, WORKSPACE_MIN, WORKSPACE_MAX)
from ik_solver import check_workspace, get_current_ee_pose
from robot_mover import move_to_cartesian, move_through_cartesian_waypoints


def _rot_to_quat_xyzw(rot_matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot_matrix).as_quat()


class GraspSkill:
    """Full grasp: move above cube -> descend to cube -> close gripper.

    Uses Cartesian impedance control for smooth, compliant motion.
    """

    name = "grasp"
    description = "Grasp the cube (approach + descend + grip)"

    def execute(self, node, context):
        cube_pos = context.get("cube_pos")
        if cube_pos is None:
            print("  x No cube pose. Run 'detect' first.")
            return False

        age = time.time() - context.get("cube_detect_time", 0)
        print(f"\n  [grasp] Cube: [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}] "
              f"({age:.0f}s ago)")

        cur_pos, cur_quat = node.get_ee_pose()
        if cur_pos is None:
            print("  x No EE pose available.")
            return False

        target_quat_xyzw = _rot_to_quat_xyzw(TOP_DOWN_ROT)

        # Phase 1: pre-grasp (above cube)
        pregrasp_pos = np.array([cube_pos[0], cube_pos[1],
                                 cube_pos[2] + PRE_GRASP_Z_OFFSET])
        print(f"  [1/3] Pre-grasp: [{pregrasp_pos[0]:.4f}, {pregrasp_pos[1]:.4f}, {pregrasp_pos[2]:.4f}]")

        if not check_workspace(pregrasp_pos):
            print("  x Pre-grasp outside workspace!")
            return False

        # Phase 2: descend to cube
        grasp_z = cube_pos[2] + 0.01  # 1cm above cube center
        grasp_pos = np.array([cube_pos[0], cube_pos[1], grasp_z])
        print(f"  [2/3] Grasp pos: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")

        if not check_workspace(grasp_pos):
            print("  x Grasp position outside workspace!")
            return False

        cart_waypoints = [
            (pregrasp_pos, target_quat_xyzw),
            (grasp_pos, target_quat_xyzw),
        ]

        # Preview + confirm
        from preview import preview_and_confirm
        logger = context.get("_trial_logger")
        record_path = None
        if logger and logger.current_trial_dir:
            record_path = os.path.join(logger.current_trial_dir, "preview.mp4")

        cur_joints = node.get_joints()
        if cur_joints is not None:
            from ik_solver import plan_target_trajectory, solve_ik, rot_to_quat
            preview_wps, _, _ = plan_target_trajectory(pregrasp_pos, cur_joints)
            if preview_wps:
                target_quat_wxyz = rot_to_quat(TOP_DOWN_ROT)
                res = solve_ik(grasp_pos, target_quat_wxyz, preview_wps[-1])
                if res.success:
                    preview_wps = preview_wps + [res.q]
                if not preview_and_confirm(node, preview_wps, context,
                                           "Grasp? (approach + descend + grip) [y/N]: ",
                                           record_path=record_path):
                    print("  Cancelled.")
                    return False
            else:
                from preview import _text_confirm
                if not _text_confirm("Grasp? [y/N]: "):
                    return False
        else:
            from preview import _text_confirm
            if not _text_confirm("Grasp? [y/N]: "):
                return False

        # Open gripper first
        print("  Gripper opening...")
        node.open_gripper()
        time.sleep(1.5)
        rclpy.spin_once(node, timeout_sec=0.1)

        # Move: approach -> descend (Cartesian impedance)
        print("  Moving to grasp position (Cartesian impedance)...")
        ok, completed = move_through_cartesian_waypoints(node, cart_waypoints)
        if not ok:
            print(f"  x Move failed at segment {completed + 1}")
            return False

        # Phase 3: close gripper
        print("  [3/3] Closing gripper...")
        node.close_gripper()
        time.sleep(2.0)
        rclpy.spin_once(node, timeout_sec=0.1)

        grip_width = node.get_gripper_width()
        if grip_width is not None:
            print(f"  Gripper width: {grip_width*1000:.1f} mm")
            if grip_width > 0.005:
                print("  Grasp successful! (object detected)")
                context["grasped"] = True
            else:
                print("  Warning: gripper fully closed -- object may not be grasped")
                context["grasped"] = False
        else:
            print("  Grasp command sent (no width feedback)")
            context["grasped"] = True

        return True
