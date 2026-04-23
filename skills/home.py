"""Skill: home -- Move robot to safe home position via Cartesian impedance."""

import os
import numpy as np
from config import HOME_JOINTS, TOP_DOWN_ROT
from ik_solver import get_current_ee_pose
from robot_mover import move_to_cartesian


def _rot_to_quat_xyzw(rot_matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot_matrix).as_quat()


# Pre-compute home EE pose from home joints
_HOME_POS, _ = get_current_ee_pose(HOME_JOINTS)
_HOME_QUAT_XYZW = _rot_to_quat_xyzw(TOP_DOWN_ROT)


class HomeSkill:
    """Move robot to home position via Cartesian impedance control."""

    name = "home"
    description = "Move to home position"

    def execute(self, node, context):
        cur_pos, cur_quat = node.get_ee_pose()
        if cur_pos is None:
            # Fallback to FK
            cur_joints = node.get_joints()
            if cur_joints is not None:
                cur_pos, _ = get_current_ee_pose(cur_joints)
            if cur_pos is None:
                print("  x No EE pose available.")
                return False

        dist = np.linalg.norm(cur_pos - _HOME_POS) * 100
        print(f"\n  [home] Current EE: [{cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}]")
        print(f"  Home EE:    [{_HOME_POS[0]:.4f}, {_HOME_POS[1]:.4f}, {_HOME_POS[2]:.4f}]")
        print(f"  Distance: {dist:.1f} cm")

        if dist < 1.0:
            print("  Already at home position.")
            return True

        # Preview + confirm
        from preview import preview_and_confirm
        logger = context.get("_trial_logger")
        record_path = None
        if logger and logger.current_trial_dir:
            record_path = os.path.join(logger.current_trial_dir, "preview.mp4")

        cur_joints = node.get_joints()
        if cur_joints is not None:
            if not preview_and_confirm(node, [HOME_JOINTS], context, "Execute? [y/N]: ",
                                       record_path=record_path):
                print("  Cancelled.")
                return False
        else:
            from preview import _text_confirm
            if not _text_confirm("Execute? [y/N]: "):
                return False

        print("  Moving to home (Cartesian impedance)...")
        ok = move_to_cartesian(node, _HOME_POS, _HOME_QUAT_XYZW)
        if ok:
            print("  Home position reached.")
            return True
        else:
            print("  x Failed.")
            return False
