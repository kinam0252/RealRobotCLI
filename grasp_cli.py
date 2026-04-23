#!/usr/bin/env python3
"""RealRobotCLI -- Skill-based interactive robot control with LLM chat mode.

Natural language commands are parsed by Copilot CLI into skill sequences.

Each skill execution is logged as a trial with:
  - command.json (skill, params, timing, result)
  - preview.mp4  (MuJoCo preview animation, if applicable)
  - camera.mp4   (cam_base recording during execution)

Output: ~/kinam_dev/RealRobotCLI/output/YYYYMMDD_HHMMSS/
"""

import sys
import os
import argparse
import threading
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from robot_mover import RobotNode, wait_for_joint_state
from skills import SKILLS
from trial_logger import TrialLogger


def print_banner():
    print("\n" + "=" * 58)
    print("  RealRobotCLI -- Skill-based Robot Control")
    print("=" * 58)
    print("  Chat mode -- natural language robot control")
    print("  Copilot CLI parses commands into skill sequences")
    print()
    print('  e.g.: "detect the cube"')
    print('        "move 5cm above cube"')
    print('        "pick up cube and place left 10cm"')
    print('        "show status"')
    print()
    print("  /help   -- help")
    print("  /quit   -- quit")
    print("=" * 58 + "\n")



# -- Camera recording thread --

class CameraFeeder:
    """Background thread that feeds camera frames to the trial logger."""

    def __init__(self, node, logger):
        self.node = node
        self.logger = logger
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while self._running:
            rgb, _, _ = self.node.get_camera_frame()
            if rgb is not None:
                self.logger.feed_camera_frame(rgb)
            time.sleep(1.0 / 15)  # ~15 fps


# -- Skill execution with logging --

def execute_skill(skill_name, skill, node, context, logger, cam_feeder,
                  user_command="", params=None):
    """Execute a single skill with trial logging and camera recording."""
    # Start trial
    logger.start_trial(skill_name, user_command=user_command, params=params)

    # Start camera recording
    logger.start_camera_recording(node)
    cam_feeder.start()

    # Pass logger to context so skills can access trial_dir for preview
    context["_trial_logger"] = logger

    success = False
    result_data = {}
    try:
        success = skill.execute(node, context)
        # Collect result data
        result_data["cube_pos"] = context.get("cube_pos")
        result_data["grasped"] = context.get("grasped")
        ee_pos, _ = node.get_ee_pose()
        if ee_pos is not None:
            result_data["final_ee_pos"] = ee_pos
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        result_data["interrupted"] = True
    except Exception as e:
        print(f"  Error: {e}")
        result_data["error"] = str(e)

    # Stop camera recording
    cam_feeder.stop()
    logger.save_context_snapshot(context)
    logger.end_trial(success=bool(success), result_data=result_data)

    # Cleanup
    context.pop("_trial_logger", None)
    return success


# -- Chat mode --

def execute_plan(plan_data, skill_instances, node, context, logger, cam_feeder,
                 user_command=""):
    """Execute an LLM-generated skill plan step by step."""
    plan = plan_data.get("plan", [])
    reasoning = plan_data.get("reasoning", "")

    if reasoning:
        print(f"  Reasoning: {reasoning}")

    if not plan:
        print("  (No actions in plan)")
        return

    for i, step in enumerate(plan, 1):
        skill_name = step.get("skill", "")
        params = step.get("params", {})
        desc = step.get("description", skill_name)

        print(f"\n  [{i}/{len(plan)}] {desc}")

        if skill_name not in skill_instances:
            print(f"  Unknown skill: {skill_name}")
            break

        if skill_name == "move" and "target" in params:
            context["_move_target_override"] = params["target"]
        if skill_name == "lift" and "amount" in params:
            context["_lift_amount"] = params["amount"]

        skill = skill_instances[skill_name]
        execute_skill(skill_name, skill, node, context, logger, cam_feeder,
                      user_command=user_command, params=params)

    print("\n  Plan complete.")


def chat_loop(node, skill_instances, context, logger, cam_feeder):
    """Natural language chat loop using Copilot CLI as LLM backend."""
    from agent import plan_from_instruction

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return False

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/q"):
                return False
            elif cmd in ("/help", "/?"):
                print_banner()
                continue
            else:
                print(f"  Unknown command: {user_input}")
                continue

        print("  Copilot thinking...")
        try:
            plan_data = plan_from_instruction(user_input, node, context)
            execute_plan(plan_data, skill_instances, node, context,
                         logger, cam_feeder, user_command=user_input)
        except KeyboardInterrupt:
            print("\n  Cancelled.")
        except Exception as e:
            print(f"  LLM error: {e}")
            print("  Tip: try again or rephrase your command")


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="RealRobotCLI")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable MuJoCo preview (text-only confirmation)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable trial logging")
    args = parser.parse_args()

    rclpy.init()
    node = RobotNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Waiting for robot joint state...")
    if not wait_for_joint_state(node, timeout_s=10.0):
        print("ERROR: No joint state after 10s. Is the robot controller running?")
        node.destroy_node()
        rclpy.shutdown()
        return

    print("Robot connected.")

    if args.no_preview:
        import preview as _preview_mod
        _preview_mod.PREVIEW_DISABLED = True

    # Trial logger
    logger = TrialLogger(enabled=not args.no_log)
    cam_feeder = CameraFeeder(node, logger)

    skill_instances = {name: cls() for name, cls in SKILLS.items()}
    context = {}

    print_banner()

    try:
        while True:
            result = chat_loop(node, skill_instances, context,
                               logger, cam_feeder)
            if result is False:
                break
    except KeyboardInterrupt:
        pass

    print("\nShutting down...")
    if logger.session_dir:
        print(f"  Logs: {logger.session_dir}")
    node.destroy_node()
    rclpy.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
