"""Skill: release — Open gripper to release object."""

import time
import rclpy


class ReleaseSkill:
    """Open gripper to release held object."""

    name = "release"
    description = "Open gripper (release object)"

    def execute(self, node, context):
        print("\n  [release] Opening gripper...")

        node.open_gripper()
        time.sleep(2.0)
        rclpy.spin_once(node, timeout_sec=0.1)

        grip_width = node.get_gripper_width()
        if grip_width is not None:
            print(f"  Gripper width: {grip_width*1000:.1f} mm")
            if grip_width > 0.06:
                print("  ✓ Gripper fully open.")
            else:
                print(f"  ⚠ Gripper may not be fully open (width={grip_width*1000:.1f}mm)")
        else:
            print("  ✓ Open command sent.")

        context["grasped"] = False
        return True
