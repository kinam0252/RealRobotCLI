"""Skill: move -- Move robot EE to a user-specified position via Cartesian impedance."""

import os
import re
import time
import numpy as np

from config import (WORKSPACE_MIN, WORKSPACE_MAX, APPROACH_Z_OFFSET,
                    TOP_DOWN_ROT)
from ik_solver import check_workspace, get_current_ee_pose, rot_to_quat
from robot_mover import move_to_cartesian, move_through_cartesian_waypoints


def parse_offset(text, cube_pos):
    """Parse user text into a 3D target position relative to the cube.

    Supports Korean/English direction words + distance in cm/mm/m.
    Examples:
      "위 5cm"         -> cube + [0, 0, +0.05]
      "아래 3cm"       -> cube + [0, 0, -0.03]
      "왼쪽 10cm"      -> cube + [0, +0.10, 0]
      "오른쪽 5cm"     -> cube + [0, -0.05, 0]
      "앞 8cm"         -> cube + [-0.08, 0, 0]
      "뒤 5cm"         -> cube + [+0.05, 0, 0]
      "above 5cm"      -> cube + [0, 0, +0.05]
      "x+3cm y-2cm z+5cm" -> cube + [+0.03, -0.02, +0.05]
      "0.4 0.0 0.12"   -> absolute position [0.4, 0.0, 0.12]
    """
    text = text.strip()
    cube_pos = np.asarray(cube_pos, dtype=np.float64)

    # Absolute coordinates: "0.4 0.0 0.12" or "0.4, 0.0, 0.12"
    abs_match = re.match(
        r'^([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)$',
        text)
    if abs_match:
        pos = np.array([float(abs_match.group(i)) for i in (1, 2, 3)])
        return pos, f"absolute [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

    # Explicit xyz offsets: "x+3cm y-2cm z+5cm"
    xyz_pattern = re.findall(
        r'([xyz])\s*([+-]?\d+\.?\d*)\s*(cm|mm|m)?', text, re.IGNORECASE)
    if xyz_pattern:
        offset = np.zeros(3)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        for axis, val, unit in xyz_pattern:
            d = float(val)
            if unit == 'cm':
                d *= 0.01
            elif unit == 'mm':
                d *= 0.001
            offset[axis_map[axis.lower()]] = d
        target = cube_pos + offset
        return target, f"cube + offset [{offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}]m"

    # Direction-based: Korean/English
    dir_patterns = {
        'up':    (r'(위|above|up|상)', 2, +1),
        'down':  (r'(아래|below|down|하)', 2, -1),
        'left':  (r'(왼쪽?|left|좌)', 1, +1),
        'right': (r'(오른쪽?|right|우)', 1, -1),
        'front': (r'(앞|front|전)', 0, -1),
        'back':  (r'(뒤|back|후)', 0, +1),
    }

    offset = np.zeros(3)
    found_any = False

    for name, (pat, axis, sign) in dir_patterns.items():
        m = re.search(pat + r'\s*([+-]?\d+\.?\d*)\s*(cm|mm|m)?', text, re.IGNORECASE)
        if m:
            d = float(m.group(2))
            unit = m.group(3) or 'cm'
            if unit == 'cm':
                d *= 0.01
            elif unit == 'mm':
                d *= 0.001
            offset[axis] += sign * d
            found_any = True
            continue

        m2 = re.search(r'([+-]?\d+\.?\d*)\s*(cm|mm|m)?\s*' + pat, text, re.IGNORECASE)
        if m2:
            d = float(m2.group(1))
            unit = m2.group(2) or 'cm'
            if unit == 'cm':
                d *= 0.01
            elif unit == 'mm':
                d *= 0.001
            offset[axis] += sign * d
            found_any = True

    if found_any:
        target = cube_pos + offset
        return target, f"cube + [{offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}]m"

    return None, "parse failed"


def _rot_to_quat_xyzw(rot_matrix):
    """Convert 3x3 rotation matrix to quaternion (xyzw)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot_matrix).as_quat()  # xyzw


class MoveSkill:
    """Move robot EE to a user-specified position via Cartesian impedance control."""

    name = "move"
    description = "Move robot to target position (relative/absolute)"

    def execute(self, node, context):
        """Interactive: show cube pose -> ask target -> Cartesian move."""
        # Check cube pose
        cube_pos = context.get("cube_pos")
        if cube_pos is None:
            print("  x No cube pose available. Run 'detect' first.")
            return False

        age = time.time() - context.get("cube_detect_time", 0)
        print(f"\n  Cube position: [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}] "
              f"(detected {age:.0f}s ago)")

        # Check for LLM-provided target override
        override = context.pop("_move_target_override", None)
        if override:
            user_input = override
            print(f"\n  (LLM target: {user_input})")
        else:
            print("\n  Where to move?")
            print("  e.g.: up 5cm / left 3cm / x+3cm z+5cm / 0.4 0.0 0.12")
            print()

            try:
                user_input = input("  Target: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                return False

            if not user_input:
                print("  Cancelled.")
                return False

        target_pos, desc = parse_offset(user_input, cube_pos)
        if target_pos is None:
            print(f"  x Cannot parse: '{user_input}'")
            return False

        print(f"\n  Target: {desc}")
        print(f"  -> Position: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")

        if not check_workspace(target_pos):
            print(f"  x Target outside safety workspace!")
            return False

        # Get current EE pose from Franka state broadcaster
        cur_pos, cur_quat = node.get_ee_pose()
        if cur_pos is None:
            # Fallback to FK from joints
            cur_joints = node.get_joints()
            if cur_joints is not None:
                cur_pos, _ = get_current_ee_pose(cur_joints)
            if cur_pos is None:
                print("  x No EE pose available.")
                return False

        dist = np.linalg.norm(target_pos - cur_pos)
        print(f"  Current EE: [{cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}]")
        print(f"  Distance: {dist*100:.1f} cm")

        # Target orientation: top-down grasp
        target_quat_xyzw = _rot_to_quat_xyzw(TOP_DOWN_ROT)

        # Build Cartesian waypoints: safe-above -> target
        safe_z = max(target_pos[2], cur_pos[2]) + APPROACH_Z_OFFSET
        safe_above = np.array([target_pos[0], target_pos[1], safe_z])

        cart_waypoints = []
        # Only add safe-above if significant horizontal movement
        if dist > 0.05:
            cart_waypoints.append((safe_above, target_quat_xyzw))
        cart_waypoints.append((target_pos, target_quat_xyzw))

        print(f"\n  Plan: {len(cart_waypoints)} Cartesian segments (impedance control)")

        # Preview + confirm
        from preview import preview_and_confirm
        # Get preview recording path from trial logger if available
        logger = context.get("_trial_logger")
        record_path = None
        if logger and logger.current_trial_dir:
            record_path = os.path.join(logger.current_trial_dir, "preview.mp4")

        # For preview, convert to joint waypoints for visualization
        cur_joints = node.get_joints()
        if cur_joints is not None:
            from ik_solver import plan_target_trajectory
            preview_wps, _, _ = plan_target_trajectory(target_pos, cur_joints)
            if preview_wps and not preview_and_confirm(node, preview_wps, context,
                                                        "Execute? [y/N]: ",
                                                        record_path=record_path):
                print("  Cancelled.")
                return False
        else:
            from preview import _text_confirm
            if not _text_confirm("Execute? [y/N]: "):
                print("  Cancelled.")
                return False

        # Execute via Cartesian impedance
        print("  Moving (Cartesian impedance)...")
        ok, completed = move_through_cartesian_waypoints(node, cart_waypoints)
        if ok:
            final_pos, _ = node.get_ee_pose()
            if final_pos is not None:
                err_mm = np.linalg.norm(final_pos - target_pos) * 1000
                print(f"\n  Done!")
                print(f"    Final EE: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
                print(f"    Error: {err_mm:.1f} mm")
                context["last_ee_pos"] = final_pos.copy()
            return True
        else:
            print(f"  x Move failed at segment {completed + 1}")
            return False
