# RealRobotCLI

A skill-based conversational CLI for controlling a **Franka FR3** robot arm with natural language.

Type a plain-English command → **Copilot CLI (LLM)** converts it into a JSON skill plan → skills execute sequentially on the real robot.

---

## Project Structure

```
RealRobotCLI/
│
├── grasp_cli.py            # Entry point — chat loop & skill dispatcher
├── agent.py                # LLM agent — natural language → JSON skill plan
├── config.py               # Paths, ROS 2 topics, safety limits, motion params
├── robot_mover.py          # ROS 2 node — joint / gripper / camera pub-sub
├── ik_solver.py            # DH-based inverse kinematics wrapper
├── trial_logger.py         # Per-trial logging (command.json + camera.mp4)
├── preview.py              # Real-time MuJoCo preview before execution
├── replay_preview.py       # Offline MuJoCo replay video generator
├── fp_oneshot.py           # One-shot object detection (SAM2 + FoundationPose)
│
├── skills/                 # Skill modules (one file per skill)
│   ├── __init__.py         #   ↳ Skill registry (SKILLS dict)
│   ├── detect.py           #   ↳ SAM2 + FoundationPose → 6D cube pose
│   ├── move.py             #   ↳ Move EE relative to cube / absolute coords
│   ├── grasp.py            #   ↳ Full grasp: approach → descend → grip
│   ├── close_gripper.py    #   ↳ Close gripper only (no arm motion)
│   ├── open_gripper.py     #   ↳ Open gripper only (no arm motion)
│   ├── lift.py             #   ↳ Lift straight up from current position
│   ├── release.py          #   ↳ Open gripper + update state
│   ├── home.py             #   ↳ Return to safe home position
│   └── status.py           #   ↳ Print joint angles, EE pose, gripper, cube info
│
├── output/                 # Session logs (auto-created at runtime)
└── .EXTERNEL/              # Demo videos & comparisons
```

---

## Getting Started

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **ROS 2 Humble** | Franka ROS 2 driver must be running |
| **Conda env** | `env_train` — activate via `source ~/kinam_dev/setup_env.sh` |
| **Copilot CLI** | `gh extension install github/gh-copilot` |
| **FoundationPose + SAM2** | Separate venvs — loaded automatically by `detect` skill |
| **MuJoCo** | For trajectory preview & offline replay |

### Run

```bash
source ~/kinam_dev/setup_env.sh
cd ~/kinam_dev/RealRobotCLI
python grasp_cli.py
```

| Flag | Description |
|------|-------------|
| `--no-preview` | Skip MuJoCo preview (text-only confirmation) |
| `--no-log` | Disable trial logging |

---

## What You See

On launch, the CLI prints a banner and drops you into an interactive prompt:

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

Type any command in natural language at the `>>` prompt.

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
 │     • User's command                     │
 │                                          │
 │  3. Call Copilot CLI                     │
 │     → copilot -p "..." --silent          │
 │     ← Parse JSON response               │
 └──────────────────────────────────────────┘
        │
        ▼
 JSON plan (example):
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
 │    • Run skill                           │
 │    • Stop recording                      │
 │    • Save trial log                      │
 │                                          │
 │  [1/4] detect                            │
 │    Camera → SAM2 mask → FoundationPose   │
 │    → cube 6D pose saved to context       │
 │                                          │
 │  [2/4] move (target: "above 2.5cm")      │
 │    Compute target = cube_pos + z offset  │
 │    Show MuJoCo preview → confirm (y/N)   │
 │    Cartesian impedance control → move    │
 │                                          │
 │  [3/4] close_gripper                     │
 │    Close with grasp force (50 N)         │
 │                                          │
 │  [4/4] lift (amount: 5cm)                │
 │    Move EE straight up by 5 cm           │
 └──────────────────────────────────────────┘
        │
        ▼
 Logs saved to:
 output/YYYYMMDD_HHMMSS/
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

### Move only

```
>> move to 5cm above the cube
  Copilot thinking...
  [1/2] Detect cube
  [2/2] Move — target: cube + [+0.000, +0.000, +0.050]m
    Execute? [y/N]: y
    Done! Error: 1.2 mm
```

### Slash commands

| Command | Action |
|---------|--------|
| `/help` | Show help banner |
| `/quit` or `/q` | Exit CLI |

---

## Skills Reference

| Skill | Params | Description |
|-------|--------|-------------|
| `detect` | — | Camera capture → SAM2 segmentation → FoundationPose 6D pose estimation |
| `move` | `target` | Move EE relative to cube (`"above 5cm"`, `"left 3cm"`, `"x+3cm z+2cm"`) or absolute (`"0.4 0.0 0.12"`) |
| `grasp` | — | Full sequence: open gripper → approach 5 cm above cube → descend → close gripper |
| `close_gripper` | — | Close gripper with grasp force — no arm movement |
| `open_gripper` | — | Open gripper — no arm movement |
| `lift` | `amount` (default 5 cm) | Lift EE straight up from current position |
| `release` | — | Open gripper and mark object as released |
| `home` | — | Move to safe home joint configuration |
| `status` | — | Print joints, EE pose, gripper width, cube detection state |

---

## Safety

| Parameter | Value |
|-----------|-------|
| **Workspace bounds** | x ∈ [0.15, 0.75],  y ∈ [−0.45, 0.45],  z ∈ [0.00, 0.55] m |
| **Approach waypoint** | 10 cm above target (auto-added when horizontal move > 5 cm) |
| **Pre-grasp height** | 5 cm above cube before descending |
| **Grasp force** | 50 N with 4 cm tolerance |
| **User confirmation** | MuJoCo preview + `y/N` prompt before every move |
| **IK check** | Motion rejected if IK solver fails |

---

## Visualization

### MuJoCo Replay

```bash
python replay_preview.py output/YYYYMMDD_HHMMSS/trial_002_move
```

Reads the joint trajectory from `command.json` and renders a `preview_replay.mp4` video.

### HTML Video Comparator

Open `output/video_compare.html` in a browser to view real camera footage and sim replay side by side for each trial.

---

## Dependencies

- **Conda** — `env_train` environment (via `setup_env.sh`)
- **ROS 2 Humble** — with Franka ROS 2 driver
- **Copilot CLI** — `gh extension install github/gh-copilot`
- **FoundationPose + SAM2** — separate Python venvs (auto-activated by `detect`)
- **MuJoCo** — for preview and replay rendering
- **IK solver** — references `~/kinam_dev/Mujoco/utils/ik_dh_minchange.py`
