"""Skill: close_gripper -- Close gripper (grasp force) without any arm movement."""

import time
import rclpy


class CloseGripperSkill:
    """Close gripper with grasp force. No arm movement."""

    name = "close_gripper"
    description = "Close gripper (grasp with force, no arm movement)"

    def execute(self, node, context):
        print("\n  [close_gripper] Closing gripper...")
        node.close_gripper()
        time.sleep(2.0)
        rclpy.spin_once(node, timeout_sec=0.1)

        grip_width = node.get_gripper_width()
        if grip_width is not None:
            print(f"  Gripper width: {grip_width*1000:.1f} mm")
            if grip_width > 0.005:
                print("  Object detected in gripper.")
                context["grasped"] = True
            else:
                print("  Gripper fully closed (no object?).")
                context["grasped"] = False
        else:
            print("  Close command sent.")
            context["grasped"] = True

        return True
