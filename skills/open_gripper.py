"""Skill: open_gripper -- Open gripper without any arm movement."""

import time
import rclpy


class OpenGripperSkill:
    """Open gripper. No arm movement."""

    name = "open_gripper"
    description = "Open gripper (no arm movement)"

    def execute(self, node, context):
        print("\n  [open_gripper] Opening gripper...")
        node.open_gripper()
        time.sleep(1.5)
        rclpy.spin_once(node, timeout_sec=0.1)

        grip_width = node.get_gripper_width()
        if grip_width is not None:
            print(f"  Gripper width: {grip_width*1000:.1f} mm")
        else:
            print("  Open command sent.")

        context["grasped"] = False
        return True
