# RealRobotCLI

Natural language → **Copilot CLI (LLM)** → skill plan → **Franka FR3** robot execution.

## 🚀 Quick Start

```bash
git clone https://github.com/kinam0252/RealRobotCLI.git
cd RealRobotCLI
conda activate env_train
python grasp_cli.py
```

```
>> pick up cube and lift 5cm
```

That's it. The LLM breaks your command into skills (detect → grasp → lift) and runs them on the real robot.

> **Same machine?** All external paths (FoundationPose, IK solver, calibration) are
> already configured in `config.py`. Just clone and run.

### Optional flags

| Flag | Description |
|------|-------------|
| `--no-preview` | Skip MuJoCo trajectory preview |
| `--no-log` | Disable per-trial logging |

---

## What You See

```
==========================================================
  RealRobotCLI -- Skill-based Robot Control
==========================================================
  Chat mode -- natural language robot control
  Copilot CLI parses commands into skill sequences

  e.g.: "detect the cube"
        "move 5cm above cube"
        "pick up cube and place left 10cm"
        "show status"

  /help   -- help
  /quit   -- quit
==========================================================

>>
```

Type any command at the `>>` prompt. The LLM converts it into a skill plan and executes it step by step.

---

## How It Works

```
 User input (natural language)
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  agent.py  —  plan_from_instruction()    │
 │                                          │
 │  1. Collect current robot state          │
 │     • EE position, gripper width         │
 │     • Cube detected? Grasped?            │
 │                                          │
 │  2. Build prompt                         │
 │     • SKILL_REFERENCE (9 skills)         │
 │     • Current state snapshot             │
 │     • User command                       │
 │                                          │
 │  3. Call Copilot CLI                     │
 │     → copilot -p "..." --silent          │
 │     ← Parse JSON response               │
 └──────────────────────────────────────────┘
        │
        ▼
 JSON plan example:
 {
   "plan": [
     {"skill": "detect",         "params": {}},
     {"skill": "move",           "params": {"target": "above 2.5cm"}},
     {"skill": "close_gripper",  "params": {}},
     {"skill": "lift",           "params": {"amount": "5cm"}}
   ],
   "reasoning": "Detect cube, approach from above, grip, then lift"
 }
        │
        ▼
 ┌──────────────────────────────────────────┐
 │  grasp_cli.py  —  execute_plan()         │
 │                                          │
 │  For each step:                          │
 │    • Start camera recording              │
 │    • Run the skill                       │
 │    • Stop recording & save trial log     │
 └──────────────────────────────────────────┘
        │
        ▼
 Logs saved to:  output/YYYYMMDD_HHMMSS/
   trial_001_detect/          command.json + camera.mp4
   trial_002_move/            command.json + camera.mp4 + preview.mp4
   trial_003_close_gripper/   command.json + camera.mp4
   trial_004_lift/            command.json + camera.mp4
```

---

## Usage Examples

### Pick up and lift

```
>> pick up the red cube and lift it 3cm
  Copilot thinking...
  Reasoning: Detect cube, grasp it, then lift 3cm

  [1/3] Detect cube
    ✓ Cube detected at [0.4240, 0.1640, 0.0110]

  [2/3] Full grasp sequence
    [1/3] Pre-grasp  [2/3] Descend  [3/3] Grip
    Gripper width: 38.2 mm — Grasp successful!

  [3/3] Lift 3cm
    Moving up 0.030m... Done!

  Plan complete.
```

### Move to a position

```
>> move to 5cm above the cube
  Copilot thinking...
  [1/2] Detect cube
  [2/2] Move — target: cube + [+0.000, +0.000, +0.050]m
    Execute? [y/N]: y
    Done! Error: 1.2 mm
```

---

## Project Structure

```
RealRobotCLI/
├── grasp_cli.py            # Entry point — chat loop & skill dispatcher
├── agent.py                # LLM agent: natural language → JSON skill plan
├── config.py               # All paths, ROS 2 topics, safety limits, motion params
├── robot_mover.py          # ROS 2 node: joint / gripper / camera pub-sub
├── ik_solver.py            # DH-based inverse kinematics wrapper
├── trial_logger.py         # Per-trial logging (command.json + camera.mp4)
├── preview.py              # Real-time MuJoCo preview before execution
├── fp_oneshot.py           # One-shot detection (SAM2 + FoundationPose)
│
├── skills/                 # One file per skill
│   ├── detect.py           #   SAM2 + FoundationPose → 6D cube pose
│   ├── move.py             #   Move EE relative to cube or absolute coords
│   ├── grasp.py            #   Full grasp: approach → descend → grip
│   ├── close_gripper.py    #   Close gripper (no arm motion)
│   ├── open_gripper.py     #   Open gripper (no arm motion)
│   ├── lift.py             #   Lift straight up from current position
│   ├── release.py          #   Open gripper + update state
│   ├── home.py             #   Return to home position
│   └── status.py           #   Print robot state
│
└── output/                 # Session logs (auto-created)
```

---

## Skills Reference

| Skill | Params | What it does |
|-------|--------|--------------|
| `detect` | — | Camera → SAM2 mask → FoundationPose → 6D cube pose |
| `move` | `target` | Move EE. Accepts: `"above 5cm"`, `"left 3cm"`, `"x+3cm z+2cm"`, `"0.4 0.0 0.12"` |
| `grasp` | — | Open gripper → approach 5 cm above cube → descend → close gripper |
| `close_gripper` | — | Close gripper with grasp force (no arm motion) |
| `open_gripper` | — | Open gripper (no arm motion) |
| `lift` | `amount` (default 5 cm) | Lift EE straight up |
| `release` | — | Open gripper + mark object as released |
| `home` | — | Move to safe home joint configuration |
| `status` | — | Print joints, EE pose, gripper width, cube state |

---

## Safety

| Parameter | Value |
|-----------|-------|
| Workspace bounds | x ∈ [0.15, 0.75], y ∈ [−0.45, 0.45], z ∈ [0.00, 0.55] m |
| Approach waypoint | 10 cm above target (auto-added when horizontal move > 5 cm) |
| Pre-grasp height | 5 cm above cube before descending |
| Grasp force | 50 N |
| User confirmation | MuJoCo preview + `y/N` prompt before every motion |
| IK check | Motion rejected if IK solver fails |

---

## External Dependencies (shared machine)

All external paths are configured in **`config.py`** and point to shared resources on this machine:

| What | Path |
|------|------|
| FoundationPose | `~/kinam_dev/FoundationPose/` |
| SAM2 weights | `~/kinam_dev/FoundationPose/sam_weights/` |
| FP venv | `~/kinam_dev/FoundationPose/venv/` |
| SAM2 venv | `~/kinam_dev/FoundationPose/venv_sam/` |
| Camera calibration | `~/kinam_dev/Mujoco/preproc/data/calib_result_base.calib` |
| IK solver | `~/kinam_dev/Mujoco/utils/ik_dh_minchange.py` |
| MuJoCo viewer env | `~/miniconda3/envs/mujoco_viewer/` |

### System requirements

- **ROS 2 Humble** with Franka ROS 2 driver running
- **Conda env** `env_train` (Python 3.10, PyTorch, scipy, opencv, etc.)
- **Copilot CLI** — `gh extension install github/gh-copilot`
