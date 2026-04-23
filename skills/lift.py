"""Skill: lift -- Lift the end-effector straight up by a given amount."""

import os
import numpy as np
import re

from config import WORKSPACE_MIN, WORKSPACE_MAX, TOP_DOWN_ROT
from ik_solver import check_workspace
from robot_mover import move_to_cartesian


def _rot_to_quat_xyzw(rot_matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot_matrix).as_quat()


def _parse_lift_amount(text):
    """Parse lift height from text like '5cm', '0.05', '10cm'."""
    if text is None:
        return 0.05  # default 5cm
    text = text.strip().lower()
    m = re.search(r'([\d.]+)\s*(cm|mm|m)?', text)
    if m:
        val = float(m.group(1))
        unit = m.group(2) or 'm'
        if unit == 'cm':
            val /= 100
        elif unit == 'mm':
            val /= 1000
        return val
    return 0.05


class LiftSkill:
    """Lift EE straight up from current position."""

    name = "lift"
    description = "Lift end-effector straight up (default 5cm)"

    def execute(self, node, context):
        cur_pos, cur_quat = node.get_ee_pose()
        if cur_pos is None:
            print("  x No EE pose available.")
            return False

        # Get lift amount from context override or ask
        override = context.pop("_lift_amount", None)
        if override:
            lift_m = _parse_lift_amount(str(override))
        else:
            try:
                user_input = input("  Lift amount [default 5cm]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return False
            lift_m = _parse_lift_amount(user_input) if user_input else 0.05

        target_pos = cur_pos.copy()
        target_pos[2] += lift_m

        print(f"\n  [lift] Current Z: {cur_pos[2]:.4f} m")
        print(f"  [lift] Target Z:  {target_pos[2]:.4f} m (+{lift_m*100:.1f} cm)")

        if not check_workspace(target_pos):
            print("  x Target outside workspace!")
            return False

        target_quat = _rot_to_quat_xyzw(TOP_DOWN_ROT)

        # Preview + confirm
        from preview import preview_and_confirm
        logger = context.get("_trial_logger")
        record_path = None
        if logger and logger.current_trial_dir:
            record_path = os.path.join(logger.current_trial_dir, "preview.mp4")

        cur_joints = node.get_joints()
        if cur_joints is not None:
            from ik_solver import plan_target_trajectory
            preview_wps, _, _ = plan_target_trajectory(target_pos, cur_joints)
            if preview_wps and not preview_and_confirm(node, preview_wps, context,
                                                        "Lift? [y/N]: ",
                                                        record_path=record_path):
                print("  Cancelled.")
                return False
        else:
            from preview import _text_confirm
            if not _text_confirm("Lift? [y/N]: "):
                return False

        print(f"  Lifting +{lift_m*100:.1f} cm...")
        ok = move_to_cartesian(node, target_pos, target_quat)
        if ok:
            print("  Lift complete.")
        else:
            print("  x Lift failed.")
        return ok
