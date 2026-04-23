#!/usr/bin/env python3
"""MuJoCo preview viewer — animates a planned joint trajectory.

Reads trajectory from /dev/shm/cli_preview.json and plays it back.
Run in mujoco_viewer conda env. No ROS2 required — all data via JSON.

Usage (called automatically by preview.py):
    ~/miniconda3/envs/mujoco_viewer/bin/python preview_viewer.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Path setup — same as mujoco_live_viewer.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_KINAM_ROOT = _SCRIPT_DIR.parent
_MUJOCO_FRANKA_SRC = _KINAM_ROOT / 'Mujoco_Franka' / 'src'
sys.path.insert(0, str(_MUJOCO_FRANKA_SRC))

PREVIEW_SHM = "/dev/shm/cli_preview.json"
CAM_SETTINGS_FILE = str(_KINAM_ROOT / 'execute_real_robot' / '.viewer_cam.json')

DEFAULT_CAM = {
    'lookat': [0.35, 0.0, 0.25],
    'distance': 1.2,
    'azimuth': -60,
    'elevation': -25,
}


def load_cam():
    if os.path.exists(CAM_SETTINGS_FILE):
        try:
            with open(CAM_SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CAM.copy()


def apply_cam(viewer, settings):
    viewer.cam.lookat[:] = settings['lookat']
    viewer.cam.distance = settings['distance']
    viewer.cam.azimuth = settings['azimuth']
    viewer.cam.elevation = settings['elevation']


def load_preview_data():
    """Load trajectory data from shared memory."""
    if not os.path.exists(PREVIEW_SHM):
        print("[Preview] No preview data found")
        return None
    with open(PREVIEW_SHM) as f:
        data = json.load(f)
    age = time.time() - data.get("timestamp", 0)
    if age > 30.0:
        print(f"[Preview] Data too old ({age:.0f}s)")
        return None
    return data


def add_markers(viewer, data, final_joints, model, mj_data):
    """Draw start/end EE markers and cube as user geoms."""
    scn = viewer.user_scn
    scn.ngeom = 0

    # Cube marker (green translucent box)
    cube_pos = data.get("cube_pos")
    if cube_pos is not None and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.02, 0.02, 0.04]),  # half-sizes (4x4x8cm cube)
            np.array(cube_pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.array([0.2, 0.9, 0.2, 0.4], dtype=np.float32))
        scn.ngeom += 1

    # Start EE marker (blue sphere)
    start_joints = data.get("current_joints")
    if start_joints is not None:
        mj_data.qpos[:7] = start_joints
        mujoco.mj_forward(model, mj_data)
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        if hand_id >= 0:
            start_pos = mj_data.xpos[hand_id].copy()
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.015, 0, 0]),
                    start_pos,
                    np.eye(3).flatten(),
                    np.array([0.3, 0.3, 1.0, 0.6], dtype=np.float32))
                scn.ngeom += 1

    # End EE marker (red sphere)
    if final_joints is not None:
        mj_data.qpos[:7] = final_joints
        mujoco.mj_forward(model, mj_data)
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        if hand_id >= 0:
            end_pos = mj_data.xpos[hand_id].copy()
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.015, 0, 0]),
                    end_pos,
                    np.eye(3).flatten(),
                    np.array([1.0, 0.2, 0.2, 0.6], dtype=np.float32))
                scn.ngeom += 1

    # Trajectory path (yellow line segments through waypoint EE positions)
    waypoint_joints = data.get("waypoints", [])
    if len(waypoint_joints) >= 2:
        ee_positions = []
        for wj in waypoint_joints:
            mj_data.qpos[:7] = wj
            mujoco.mj_forward(model, mj_data)
            if hand_id >= 0:
                ee_positions.append(mj_data.xpos[hand_id].copy())

        for i in range(1, len(ee_positions)):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                np.array(ee_positions[i-1], dtype=np.float64),
                np.array(ee_positions[i], dtype=np.float64))
            g.rgba[:] = np.array([1.0, 0.8, 0.0, 0.7], dtype=np.float32)
            scn.ngeom += 1


def main():
    data = load_preview_data()
    if data is None:
        sys.exit(1)

    frames = data.get("frames", [])
    frame_dt = data.get("frame_dt", 0.06)
    current_joints = data.get("current_joints")

    if not frames:
        print("[Preview] No frames to animate")
        sys.exit(1)

    print(f"[Preview] {len(frames)} frames, {len(frames)*frame_dt:.1f}s animation")

    # Load calibration and build model
    from utils import load_calib, make_model
    T_base_cam = load_calib()
    model = make_model(T_base_cam)
    mj_data = mujoco.MjData(model)

    # Find joint indices
    arm_start = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'fr3_joint1')]
    finger1_idx = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')]
    finger2_idx = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')]

    # Set gripper
    gripper_w = data.get("gripper_width", 0.08)
    finger_val = gripper_w / 2.0
    mj_data.qpos[finger1_idx] = finger_val
    mj_data.qpos[finger2_idx] = finger_val

    # Set initial pose
    if current_joints:
        mj_data.qpos[arm_start:arm_start+7] = current_joints
    mujoco.mj_forward(model, mj_data)

    _run_interactive_viewer(model, mj_data, data, frames,
                            current_joints, arm_start, frame_dt)


def _run_interactive_viewer(model, mj_data, data, frames,
                            current_joints, arm_start, frame_dt):
    """Interactive MuJoCo viewer with animation loop."""
    playback_speed = 2.0
    final_joints = frames[-1] if frames else current_joints

    import signal
    _shutdown_requested = [False]
    def _on_sigterm(signum, frame):
        _shutdown_requested[0] = True
    signal.signal(signal.SIGTERM, _on_sigterm)

    with mujoco.viewer.launch_passive(
        model, mj_data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        apply_cam(viewer, load_cam())
        add_markers(viewer, data, final_joints, model, mj_data)

        if current_joints:
            mj_data.qpos[arm_start:arm_start+7] = current_joints
        mujoco.mj_forward(model, mj_data)
        viewer.sync()

        time.sleep(0.5)

        frame_idx = 0
        LOOP_PAUSE = 1.0

        while viewer.is_running() and not _shutdown_requested[0]:
            t0 = time.time()

            if frame_idx < len(frames):
                mj_data.qpos[arm_start:arm_start+7] = frames[frame_idx]
                mujoco.mj_forward(model, mj_data)
                frame_idx += 1
            else:
                time.sleep(LOOP_PAUSE)
                frame_idx = 0
                if current_joints:
                    mj_data.qpos[arm_start:arm_start+7] = current_joints
                    mujoco.mj_forward(model, mj_data)

            viewer.sync()

            elapsed = time.time() - t0
            target_dt = frame_dt / playback_speed
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("[Preview] Viewer closed")


if __name__ == '__main__':
    main()
