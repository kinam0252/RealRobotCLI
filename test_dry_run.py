#!/usr/bin/env python3
"""Dry-run test -- verify RealRobotCLI logic without ROS2, MuJoCo, or robot.

Mocks: RobotNode, rclpy, franka_msgs, scipy (for Slerp in robot_mover).
Tests: config, IK, parse_offset, Cartesian control logic, skills, agent, preview.
"""

import sys
import os
import types
import numpy as np

# -- 1. Mock ROS2 and franka_msgs before any imports --

rclpy_mock = types.ModuleType("rclpy")
rclpy_mock.init = lambda *a, **kw: None
rclpy_mock.shutdown = lambda *a, **kw: None
rclpy_mock.spin = lambda *a, **kw: None
rclpy_mock.spin_once = lambda *a, **kw: None

node_mod = types.ModuleType("rclpy.node")
class _FakeNode:
    def __init__(self, *a, **kw): pass
    def create_publisher(self, *a, **kw): return type('P', (), {'publish': lambda s, m: None})()
    def create_subscription(self, *a, **kw): pass
    def get_logger(self): return type('L', (), {'info': lambda s, m: None, 'error': lambda s, m: None})()
    def get_clock(self): return type('C', (), {'now': lambda s: type('T', (), {'to_msg': lambda s: None})()})()
    def destroy_node(self): pass
node_mod.Node = _FakeNode
rclpy_mock.node = node_mod

action_mod = types.ModuleType("rclpy.action")
class _FakeActionClient:
    def __init__(self, *a, **kw): pass
    def wait_for_server(self, *a, **kw): return True
    def send_goal_async(self, *a, **kw): return None
action_mod.ActionClient = _FakeActionClient
rclpy_mock.action = action_mod

sys.modules["rclpy"] = rclpy_mock
sys.modules["rclpy.node"] = node_mod
sys.modules["rclpy.action"] = action_mod

for msg_pkg in ["std_msgs", "std_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
                "geometry_msgs", "geometry_msgs.msg",
                "franka_msgs", "franka_msgs.action"]:
    mod = types.ModuleType(msg_pkg)
    mod.__dict__.setdefault("Float64MultiArray", type("Float64MultiArray", (), {}))
    mod.__dict__.setdefault("JointState", type("JointState", (), {}))
    mod.__dict__.setdefault("Image", type("Image", (), {}))
    mod.__dict__.setdefault("CameraInfo", type("CameraInfo", (), {}))
    # PoseStamped mock with nested attributes
    class _FakePose:
        def __init__(self):
            self.position = type('P', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
            self.orientation = type('Q', (), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0})()
    class _FakePoseStamped:
        def __init__(self):
            self.header = type('H', (), {'stamp': None, 'frame_id': ''})()
            self.pose = _FakePose()
    mod.__dict__["PoseStamped"] = _FakePoseStamped
    _move_goal = type("Goal", (), {"width": 0, "speed": 0})
    _FrankaMove = type("Move", (), {"Goal": _move_goal})
    _grasp_eps = type("Epsilon", (), {"inner": 0, "outer": 0})
    _grasp_goal = type("Goal", (), {"width": 0, "speed": 0, "force": 0,
                                     "epsilon": _grasp_eps()})
    _FrankaGrasp = type("Grasp", (), {"Goal": _grasp_goal})
    mod.__dict__["Move"] = _FrankaMove
    mod.__dict__["Grasp"] = _FrankaGrasp
    sys.modules[msg_pkg] = mod

# -- Setup path --
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CLI_DIR)

# -- 2. Run tests --

passed = 0
failed = 0
errors = []

def test(name):
    def decorator(fn):
        global passed, failed
        try:
            fn()
            print(f"  ok {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
            errors.append((name, e))
        return fn
    return decorator

print("\n" + "=" * 60)
print("  RealRobotCLI Dry-Run Tests (no ROS2 / MuJoCo)")
print("=" * 60 + "\n")

# -- Config --
print("[Config]")

@test("config imports and Cartesian params exist")
def _():
    from config import (KINAM_ROOT, EE_POSE_TOPIC, TARGET_POSE_TOPIC,
                        CART_N_WAYPOINTS, CART_INTERVAL_S,
                        WORKSPACE_MIN, WORKSPACE_MAX, HOME_JOINTS)
    assert EE_POSE_TOPIC == "/franka_robot_state_broadcaster/current_pose"
    assert TARGET_POSE_TOPIC == "/target_pose"
    assert CART_N_WAYPOINTS == 50
    assert CART_INTERVAL_S == 0.08
    assert HOME_JOINTS.shape == (7,)

# -- IK Solver --
print("\n[IK Solver]")

@test("ik_solver imports")
def _():
    from ik_solver import solve_ik, get_current_ee_pose, plan_target_trajectory, check_workspace

@test("FK from home joints")
def _():
    from ik_solver import get_current_ee_pose
    from config import HOME_JOINTS
    pos, quat = get_current_ee_pose(HOME_JOINTS)
    assert pos.shape == (3,)
    assert 0.1 < pos[0] < 0.8
    print(f"      Home EE: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

@test("IK solve round-trip")
def _():
    from ik_solver import solve_ik, get_current_ee_pose, rot_to_quat
    from config import HOME_JOINTS, TOP_DOWN_ROT
    target_pos = np.array([0.4, 0.0, 0.3])
    target_quat = rot_to_quat(TOP_DOWN_ROT)
    result = solve_ik(target_pos, target_quat, HOME_JOINTS)
    assert result.success
    pos, _ = get_current_ee_pose(result.q)
    err = np.linalg.norm(pos - target_pos)
    assert err < 0.001
    print(f"      IK error: {err*1e6:.1f} um")

@test("workspace check")
def _():
    from ik_solver import check_workspace
    assert check_workspace(np.array([0.4, 0.0, 0.3])) == True
    assert check_workspace(np.array([0.0, 0.0, 0.3])) == False

# -- Robot Mover --
print("\n[Robot Mover]")

@test("RobotNode has Cartesian control methods")
def _():
    from robot_mover import RobotNode
    node = RobotNode()
    assert hasattr(node, 'publish_target_pose')
    assert hasattr(node, 'send_stop')
    assert hasattr(node, 'wait_for_convergence')
    assert hasattr(node, 'get_ee_pose')

@test("slow_cartesian_move function exists")
def _():
    from robot_mover import slow_cartesian_move, move_to_cartesian, move_through_cartesian_waypoints

@test("RobotNode EE pose getter (no data)")
def _():
    from robot_mover import RobotNode
    node = RobotNode()
    pos, quat = node.get_ee_pose()
    assert pos is None and quat is None

# -- Parse Offset --
print("\n[Parse Offset]")

@test("Korean directions")
def _():
    from skills.move import parse_offset
    cube = np.array([0.4, 0.0, 0.02])
    cases = [
        ("위 5cm",   [0.4, 0.0, 0.07]),
        ("아래 3cm", [0.4, 0.0, -0.01]),
        ("왼쪽 10cm", [0.4, 0.10, 0.02]),
        ("오른쪽 5cm", [0.4, -0.05, 0.02]),
    ]
    for text, expected in cases:
        pos, desc = parse_offset(text, cube)
        assert pos is not None, f"Failed to parse: {text}"
        err = np.linalg.norm(pos - np.array(expected))
        assert err < 0.001, f"'{text}': expected {expected}, got {pos.tolist()}"

@test("XYZ offsets")
def _():
    from skills.move import parse_offset
    cube = np.array([0.4, 0.0, 0.02])
    pos, _ = parse_offset("x+3cm z+5cm", cube)
    assert pos is not None
    assert np.allclose(pos, [0.43, 0.0, 0.07], atol=0.001)

@test("absolute coordinates")
def _():
    from skills.move import parse_offset
    cube = np.array([0.4, 0.0, 0.02])
    pos, _ = parse_offset("0.35 -0.1 0.15", cube)
    assert pos is not None
    assert np.allclose(pos, [0.35, -0.1, 0.15], atol=0.001)

# -- Preview --
print("\n[Preview]")

@test("expand_trajectory")
def _():
    from preview import expand_trajectory
    cur = np.zeros(7)
    wp1 = np.ones(7) * 0.5
    wp2 = np.ones(7) * 1.0
    frames = expand_trajectory([wp1, wp2], cur, n_per_seg=10)
    assert len(frames) == 20
    assert np.allclose(frames[9], [0.5]*7, atol=1e-6)
    assert np.allclose(frames[-1], [1.0]*7, atol=1e-6)

@test("write and read preview JSON")
def _():
    import json
    from preview import write_preview, cleanup_preview, PREVIEW_SHM
    cur = np.zeros(7)
    wps = [np.ones(7) * 0.5]
    write_preview(cur, wps, cube_pos=[0.4, 0.0, 0.02])
    with open(PREVIEW_SHM) as f:
        data = json.load(f)
    assert "frames" in data
    assert "cube_pos" in data
    assert len(data["frames"]) == 50  # CART_N_WAYPOINTS
    cleanup_preview()

# -- Skills --
print("\n[Skills]")

@test("all 9 skills import")
def _():
    from skills import SKILLS
    assert len(SKILLS) == 9
    for name, cls in SKILLS.items():
        inst = cls()
        assert hasattr(inst, 'execute')

@test("move skill uses Cartesian control")
def _():
    import inspect
    from skills.move import MoveSkill
    src = inspect.getsource(MoveSkill.execute)
    assert "move_through_cartesian_waypoints" in src or "Cartesian" in src

@test("grasp skill uses Cartesian control")
def _():
    import inspect
    from skills.grasp import GraspSkill
    src = inspect.getsource(GraspSkill.execute)
    assert "move_through_cartesian_waypoints" in src or "Cartesian" in src

@test("home skill uses Cartesian control")
def _():
    import inspect
    from skills.home import HomeSkill
    src = inspect.getsource(HomeSkill.execute)
    assert "move_to_cartesian" in src or "Cartesian" in src

# -- Agent --
print("\n[Agent]")

@test("agent imports and JSON parsing")
def _():
    from agent import _parse_json, SKILL_REFERENCE
    raw = '{"plan": [{"skill": "detect"}], "reasoning": "test"}'
    result = _parse_json(raw)
    assert result["plan"][0]["skill"] == "detect"
    for skill in ["detect", "move", "grasp", "release", "home", "status"]:
        assert skill in SKILL_REFERENCE.lower()

# -- CLI --
print("\n[CLI]")

@test("grasp_cli imports and has chat mode")
def _():
    from grasp_cli import print_banner, execute_plan, chat_loop
    # Verify key mode was removed
    import grasp_cli
    assert not hasattr(grasp_cli, 'KEY_TO_SKILL'), "KEY_TO_SKILL should be removed"
    assert not hasattr(grasp_cli, 'key_loop'), "key_loop should be removed"

# -- Trial Logger --
print("\n[Trial Logger]")

@test("trial_logger imports and creates session dir")
def _():
    from trial_logger import TrialLogger
    logger = TrialLogger(enabled=True)
    assert logger.session_dir is not None
    assert os.path.isdir(logger.session_dir)
    # Cleanup
    import shutil
    shutil.rmtree(logger.session_dir)

@test("trial_logger start/end trial creates files")
def _():
    from trial_logger import TrialLogger
    logger = TrialLogger(enabled=True)
    trial_dir = logger.start_trial("detect", user_command="detect cube", params={"mode": "auto"})
    assert trial_dir is not None
    assert os.path.isdir(trial_dir)
    assert logger.current_trial_dir == trial_dir
    logger.end_trial(success=True, result_data={"cube_pos": [0.4, 0.1, 0.03]})
    # Check command.json was created
    cmd_json = os.path.join(trial_dir, "command.json")
    assert os.path.exists(cmd_json)
    import json
    with open(cmd_json) as f:
        data = json.load(f)
    assert data["skill"] == "detect"
    assert data["success"] is True
    # Check session_log.json
    session_log = os.path.join(logger.session_dir, "session_log.json")
    assert os.path.exists(session_log)
    # Cleanup
    import shutil
    shutil.rmtree(logger.session_dir)

@test("trial_logger disabled mode is no-op")
def _():
    from trial_logger import TrialLogger
    logger = TrialLogger(enabled=False)
    assert logger.session_dir is None
    result = logger.start_trial("detect")
    assert result is None

@test("CameraRecorder feed and stop")
def _():
    from trial_logger import CameraRecorder
    import tempfile
    rec = CameraRecorder(resolution=(160, 120), fps=10)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        rec.start(tmp_path)
        assert rec.is_recording
        # Feed some fake frames
        for _ in range(5):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            rec.feed_frame(frame)
        rec.stop()
        assert not rec.is_recording
        assert rec.frame_count == 5
        assert os.path.exists(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@test("grasp_cli CameraFeeder init")
def _():
    from grasp_cli import CameraFeeder
    from trial_logger import TrialLogger
    logger = TrialLogger(enabled=False)
    feeder = CameraFeeder(None, logger)
    assert feeder is not None

# -- Summary --
print("\n" + "=" * 60)
total = passed + failed
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f", {failed} FAILED")
    for name, err in errors:
        print(f"    FAIL {name}: {err}")
else:
    print(" -- ALL PASSED")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
