"""LLM Agent — uses local Copilot CLI to parse natural language into skill plans."""

import json
import re
import subprocess
import shutil
import time


def copilot_query(prompt: str, timeout: int = 60) -> str:
    """Send a prompt to the local Copilot CLI and return the response text."""
    binary = shutil.which("copilot")
    if binary is None:
        raise RuntimeError(
            "'copilot' not found in PATH. "
            "Install: gh extension install github/gh-copilot"
        )
    result = subprocess.run(
        [binary, "-p", prompt, "--silent"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"copilot exited with code {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return json.loads(text.strip())


# ── Prompt templates ──────────────────────────────────────────────────

SKILL_REFERENCE = """\
Available SKILLS for a real Franka FR3 robot arm:

1. detect()
   — One-shot: capture camera image → SAM2 mask → FoundationPose → 6D cube pose
   — No parameters needed. Result: cube position [x,y,z] and orientation in robot base frame.

2. move(target)
   — Move robot end-effector to a target position.
   — target: natural language offset relative to cube, e.g. "위 5cm", "above 5cm",
     "왼쪽 3cm", "x+3cm z+5cm", or absolute coords "0.4 0.0 0.12"
   — REQUIRES: detect must have been run first (cube position needed as reference).

3. grasp()
   — FULL grasp sequence: open gripper → approach above cube → descend → close gripper.
   — REQUIRES: detect must have been run first.
   — Use this when you need the complete pick-up sequence from far away.

4. close_gripper()
   — ONLY close the gripper with grasp force. No arm movement at all.
   — Use when the robot is already at the grasp position and you just need to grip.

5. open_gripper()
   — ONLY open the gripper. No arm movement at all.
   — Use for "just open gripper" or before a manual grasp sequence.

6. release()
   — Open gripper and mark object as released. Same as open_gripper but updates state.

7. lift(amount)
   — Lift end-effector straight up from current position.
   — amount: e.g. "5cm", "10cm", "0.05" (default: 5cm)
   — Use after grasping to lift the object.

8. home()
   — Move robot to safe home position.

9. status()
   — Display current robot state (joints, EE pose, gripper, cube detection).

IMPORTANT: Choose the most specific skill. For "just close gripper" use close_gripper(),
NOT grasp(). For "lift up" use lift(), NOT move(). grasp() is only for the full
approach+descend+grip sequence.
"""

PLAN_PROMPT = """\
You are an AI controller for a real Franka FR3 robot arm with a parallel gripper.
You translate natural language instructions into a sequence of skill calls.

{skill_reference}

RULES:
- Output ONLY valid JSON — no markdown fences, no extra text.
- Always detect before move or grasp (cube position needed).
- For pick-and-place: detect → grasp → move(target) → release
- If instruction is ambiguous, use your best judgment.
- Keep plans minimal — don't add unnecessary steps.

CURRENT STATE:
{state}

USER INSTRUCTION: {instruction}

Output format:
{{"plan": [{{"skill": "close_gripper", "params": {{}}}}, {{"skill": "lift", "params": {{"amount": "5cm"}}}}], "reasoning": "..."}}

Generate the skill plan as JSON:"""


def describe_state(node, context):
    """Build a text description of current robot + detection state."""
    lines = []

    joints = node.get_joints()
    if joints is not None:
        import numpy as np
        from ik_solver import get_current_ee_pose
        ee_pos, ee_quat = get_current_ee_pose(joints)
        lines.append(f"EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    else:
        lines.append("EE position: unknown (no joint state)")

    grip = node.get_gripper_width()
    if grip is not None:
        lines.append(f"Gripper width: {grip*1000:.1f} mm ({'open' if grip > 0.06 else 'closed/holding'})")

    cube_pos = context.get("cube_pos")
    if cube_pos is not None:
        age = time.time() - context.get("cube_detect_time", 0)
        lines.append(f"Cube position: [{cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f}] "
                      f"(detected {age:.0f}s ago)")
    else:
        lines.append("Cube: not yet detected")

    grasped = context.get("grasped")
    if grasped is not None:
        lines.append(f"Object grasped: {'yes' if grasped else 'no'}")

    return "\n".join(lines)


def plan_from_instruction(instruction: str, node, context) -> dict:
    """Call Copilot CLI to convert natural language instruction into a skill plan."""
    state = describe_state(node, context)
    prompt = PLAN_PROMPT.format(
        skill_reference=SKILL_REFERENCE,
        state=state,
        instruction=instruction,
    )

    raw = copilot_query(prompt)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}\nRaw: {raw}")
