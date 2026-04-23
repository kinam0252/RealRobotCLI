"""Skill: status — Show current robot and detection status."""

import time
import numpy as np
from ik_solver import get_current_ee_pose


class StatusSkill:
    """Display current robot joints, EE pose, gripper, and cube detection status."""

    name = "status"
    description = "Show robot & detection status"

    def execute(self, node, context):
        print("\n  ─── Robot Status ───")
        joints = node.get_joints()
        if joints is not None:
            ee_pos, ee_quat = get_current_ee_pose(joints)
            print(f"  Joints (deg): {np.degrees(joints).round(1).tolist()}")
            print(f"  EE position:  [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
            print(f"  EE orient:    [w={ee_quat[0]:.4f}, x={ee_quat[1]:.4f}, "
                  f"y={ee_quat[2]:.4f}, z={ee_quat[3]:.4f}]")
        else:
            print("  Joints: ✗ not available")

        grip = node.get_gripper_width()
        if grip is not None:
            print(f"  Gripper width: {grip*1000:.1f} mm")
        else:
            print("  Gripper: ✗ no data")

        rgb, depth, K = node.get_camera_frame()
        cam_ok = rgb is not None and depth is not None and K is not None
        print(f"  Camera:  {'✓ OK' if cam_ok else '✗ not available'}"
              + (f" ({rgb.shape[1]}x{rgb.shape[0]})" if cam_ok else ""))

        print("\n  ─── Cube Detection ───")
        cube_pos = context.get("cube_pos")
        if cube_pos is not None:
            age = time.time() - context.get("cube_detect_time", 0)
            quat = context.get("cube_quat")
            print(f"  Position: [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}]")
            if quat is not None:
                print(f"  Orient:   [w={quat[0]:.4f}, x={quat[1]:.4f}, "
                      f"y={quat[2]:.4f}, z={quat[3]:.4f}]")
            print(f"  Age:      {age:.0f}s ago")
        else:
            print("  Cube: not detected yet")

        grasped = context.get("grasped")
        if grasped is not None:
            print(f"  Grasped:  {'Yes' if grasped else 'No'}")

        return True
