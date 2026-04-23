#!/usr/bin/env python3
"""Replay trial data as MuJoCo preview videos matching EXACT real robot trajectory.

Reconstructs the same Cartesian-space linear interpolation that slow_cartesian_move()
does on the real robot, then solves IK per waypoint for rendering.

Usage:
  python replay_preview.py output/20260421_220123 [output/20260421_221804 ...]
"""

import os
import sys
import json
import glob
import subprocess
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (HOME_JOINTS, TOP_DOWN_ROT, CART_N_WAYPOINTS, CART_INTERVAL_S,
                    APPROACH_Z_OFFSET, PRE_GRASP_Z_OFFSET)
from ik_solver import solve_ik, get_current_ee_pose, rot_to_quat
from skills.move import parse_offset

REPLAY_FPS = 15
SEGMENT_DURATION_S = CART_N_WAYPOINTS * CART_INTERVAL_S  # 4.0s
FRAMES_PER_SEGMENT = int(SEGMENT_DURATION_S * REPLAY_FPS)  # 60

# Home EE position (computed once)
_HOME_POS, _ = get_current_ee_pose(HOME_JOINTS)
_TOP_DOWN_QUAT = rot_to_quat(TOP_DOWN_ROT)


def _rot_to_quat_xyzw(rot_matrix):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rot_matrix).as_quat()


_TOP_DOWN_QUAT_XYZW = _rot_to_quat_xyzw(TOP_DOWN_ROT)


def cartesian_interp_to_joints(start_pos, target_pos, n_frames, ref_joints):
    """Replicate slow_cartesian_move: linear Cartesian interp, IK per step.

    Returns list of (7,) joint arrays and the final joints.
    """
    frames = []
    cur_j = ref_joints.copy()
    for i in range(1, n_frames + 1):
        t = i / n_frames
        wp_pos = start_pos + t * (target_pos - start_pos)
        result = solve_ik(wp_pos, _TOP_DOWN_QUAT, cur_j)
        if result.success:
            cur_j = result.q
        frames.append(cur_j.copy())
    return frames, cur_j


def reconstruct_trial(cmd, ctx, prev_final_ee):
    """Reconstruct exact Cartesian waypoints that the real skill code would compute.

    Returns list of (start_ee, target_ee) Cartesian segments, matching the real code.
    """
    skill = cmd["skill"]
    command_text = cmd.get("command", "")
    cube_pos = ctx.get("cube_pos")
    result = cmd.get("result", {})
    final_ee = result.get("final_ee_pos")

    # Start EE: previous trial's final position
    if prev_final_ee is not None:
        start_ee = np.array(prev_final_ee)
    else:
        start_ee = _HOME_POS.copy()

    if skill == "move":
        # Exact same logic as MoveSkill.execute() lines 144-182
        if cube_pos is None:
            return [], final_ee
        target_pos, _ = parse_offset(command_text, cube_pos)
        if target_pos is None:
            return [], final_ee

        safe_z = max(target_pos[2], start_ee[2]) + APPROACH_Z_OFFSET
        safe_above = np.array([target_pos[0], target_pos[1], safe_z])

        dist = np.linalg.norm(target_pos - start_ee)
        segments = []
        if dist > 0.05:
            segments.append((start_ee, safe_above))
        segments.append((safe_above if dist > 0.05 else start_ee, target_pos))
        return segments, final_ee

    elif skill == "lift":
        # Exact same logic as LiftSkill.execute()
        # Parses lift amount from command, adds to current Z
        from skills.lift import _parse_lift_amount
        import re
        # Extract amount from command text
        m = re.search(r'(\d+\.?\d*)\s*(cm|mm|m)?', command_text)
        if m:
            lift_m = _parse_lift_amount(command_text)
        else:
            lift_m = 0.05  # default 5cm

        target_pos = start_ee.copy()
        target_pos[2] += lift_m

        # lift uses move_to_cartesian (single segment, Cartesian interp)
        return [(start_ee, target_pos)], final_ee

    elif skill == "home":
        # home uses move_to_cartesian to _HOME_POS (single segment)
        return [(start_ee, _HOME_POS.copy())], final_ee

    elif skill == "grasp":
        # GraspSkill: move_through_cartesian_waypoints with pregrasp + grasp
        if cube_pos is None:
            return [], final_ee
        cube_pos = np.array(cube_pos)
        pregrasp_pos = np.array([cube_pos[0], cube_pos[1],
                                 cube_pos[2] + PRE_GRASP_Z_OFFSET])
        grasp_pos = cube_pos.copy()
        # Each Cartesian target is a separate slow_cartesian_move call
        # which reads current EE at start of each segment
        return [(start_ee, pregrasp_pos), (pregrasp_pos, grasp_pos)], final_ee

    elif skill in ("close_gripper", "open_gripper"):
        # No arm motion, gripper-only animation
        return "gripper_only", final_ee

    return [], final_ee


def build_session_trajectory(session_dir):
    """Build replay data using exact same trajectory logic as real robot."""
    trials = sorted(glob.glob(os.path.join(session_dir, "trial_*")))
    motion_skills = {"move", "lift", "home", "grasp", "close_gripper", "open_gripper"}
    results = []
    prev_final_ee = None
    gripper_closed = False  # track gripper state across trials

    for td in trials:
        cmd_path = os.path.join(td, "command.json")
        if not os.path.exists(cmd_path):
            continue
        cmd = json.load(open(cmd_path))
        ctx_path = os.path.join(td, "context.json")
        ctx = json.load(open(ctx_path)) if os.path.exists(ctx_path) else {}

        skill = cmd["skill"]
        result = cmd.get("result", {})
        final_ee = result.get("final_ee_pos")

        if skill not in motion_skills:
            if final_ee:
                prev_final_ee = final_ee
            continue

        if final_ee is None:
            continue

        # Reconstruct Cartesian segments exactly as real code would
        cart_segments, _ = reconstruct_trial(cmd, ctx, prev_final_ee)

        if cart_segments == "gripper_only":
            # Gripper-only skill (close_gripper / open_gripper)
            # Hold arm at current pose, animate gripper
            if prev_final_ee is not None:
                seed = solve_ik(np.array(prev_final_ee), _TOP_DOWN_QUAT,
                                np.array(HOME_JOINTS))
                arm_joints = seed.q if seed.success else np.array(HOME_JOINTS)
            else:
                arm_joints = np.array(HOME_JOINTS)

            # 30 frames for gripper animation (~2s at 15fps)
            n_grip_frames = 30
            all_frames = [arm_joints.copy() for _ in range(n_grip_frames)]

            # Camera frame count
            camera_mp4 = os.path.join(td, "camera.mp4")
            cam_frames = 0
            if os.path.exists(camera_mp4):
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-count_frames",
                         "-show_entries", "stream=nb_read_frames",
                         "-of", "csv=p=0", camera_mp4],
                        capture_output=True, text=True, timeout=30)
                    cam_frames = int(probe.stdout.strip())
                except Exception:
                    pass

            start_ee = np.array(prev_final_ee) if prev_final_ee else _HOME_POS.copy()

            results.append({
                "trial_dir": td,
                "trial_name": os.path.basename(td),
                "skill": skill,
                "command": cmd.get("command", ""),
                "frames": all_frames,
                "n_segments": 0,
                "ee_positions": [],
                "start_ee": start_ee,
                "end_ee": np.array(final_ee) if final_ee else start_ee,
                "cube_pos": ctx.get("cube_pos"),
                "motion_duration_s": 2.0,
                "camera_frame_count": cam_frames,
                "gripper_anim": "close" if skill == "close_gripper" else "open",
            })

            gripper_closed = (skill == "close_gripper")
            prev_final_ee = final_ee
            continue

        if not cart_segments:
            print(f"  Skip {os.path.basename(td)}: could not reconstruct trajectory")
            prev_final_ee = final_ee
            continue

        # For each Cartesian segment, do linear Cartesian interp + IK (like real robot)
        # Start IK seed from previous joints or home
        if prev_final_ee is not None:
            seed_j = solve_ik(np.array(prev_final_ee), _TOP_DOWN_QUAT,
                              np.array(HOME_JOINTS))
            ref_joints = seed_j.q if seed_j.success else np.array(HOME_JOINTS)
        else:
            ref_joints = np.array(HOME_JOINTS)

        all_frames = []
        ee_positions = []  # for trajectory visualization

        for seg_start, seg_end in cart_segments:
            seg_frames, ref_joints = cartesian_interp_to_joints(
                seg_start, seg_end, FRAMES_PER_SEGMENT, ref_joints)
            all_frames.extend(seg_frames)

            # Record EE positions for trajectory line (sample every few)
            step = max(1, len(seg_frames) // 15)
            for i in range(0, len(seg_frames), step):
                pos, _ = get_current_ee_pose(seg_frames[i])
                ee_positions.append(pos)

        # Add final position
        pos, _ = get_current_ee_pose(all_frames[-1])
        ee_positions.append(pos)

        start_ee = np.array(prev_final_ee) if prev_final_ee else _HOME_POS.copy()
        end_ee = np.array(final_ee)

        # Get camera.mp4 frame count for sync
        camera_mp4 = os.path.join(td, "camera.mp4")
        cam_frames = 0
        if os.path.exists(camera_mp4):
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-count_frames",
                     "-show_entries", "stream=nb_read_frames",
                     "-of", "csv=p=0", camera_mp4],
                    capture_output=True, text=True, timeout=30)
                cam_frames = int(probe.stdout.strip())
            except Exception:
                pass

        results.append({
            "trial_dir": td,
            "trial_name": os.path.basename(td),
            "skill": skill,
            "command": cmd.get("command", ""),
            "frames": all_frames,
            "n_segments": len(cart_segments),
            "ee_positions": ee_positions,
            "start_ee": start_ee,
            "end_ee": end_ee,
            "cube_pos": ctx.get("cube_pos"),
            "motion_duration_s": len(cart_segments) * SEGMENT_DURATION_S,
            "camera_frame_count": cam_frames,
            "gripper_closed": gripper_closed,
        })

        prev_final_ee = final_ee

    return results


def _add_markers(renderer, start_pos, end_pos, ee_positions):
    """Add trajectory and endpoint markers to renderer scene."""
    import mujoco as mj
    scn = renderer.scene

    # Start EE (blue sphere)
    if start_pos is not None and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0, 0]),
            np.array(start_pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.array([0.3, 0.3, 1.0, 0.6], dtype=np.float32))
        scn.ngeom += 1

    # End EE (green sphere)
    if end_pos is not None and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE,
            np.array([0.015, 0, 0]),
            np.array(end_pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.array([0.2, 0.9, 0.2, 0.6], dtype=np.float32))
        scn.ngeom += 1

    # Trajectory path (yellow sphere dots — capsules crash osmesa)
    for p in ee_positions:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            g, mj.mjtGeom.mjGEOM_SPHERE,
            np.array([0.005, 0, 0]),
            np.ascontiguousarray(p, dtype=np.float64),
            np.eye(3).flatten(),
            np.array([1.0, 0.8, 0.0, 0.7], dtype=np.float32))
        scn.ngeom += 1


def render_trial(trial_data, output_path, model, arm_start,
                 f1_idx, f2_idx, cam, renderer):
    import mujoco as mj
    import cv2

    frames = trial_data["frames"]
    ee_positions = trial_data["ee_positions"]
    start_ee = trial_data["start_ee"]
    end_ee = trial_data["end_ee"]
    camera_frames = trial_data.get("camera_frame_count", 0)

    W, H = renderer.width, renderer.height

    # Gripper state
    gripper_anim = trial_data.get("gripper_anim")  # "close", "open", or None
    grip_open = 0.04
    grip_closed = 0.02  # stops at cube width (4cm full = 0.02 half per finger)

    mj_data = mj.MjData(model)

    # Place cube via qpos — lying flat (90deg X then 90deg Y)
    cube_qadr = trial_data.get("_cube_qadr", -1)
    session_cube = trial_data.get("_session_cube_pos")
    if cube_qadr >= 0 and session_cube is not None:
        mj_data.qpos[cube_qadr:cube_qadr+3] = session_cube
        # Rotate 90deg around X axis so widest face is on table
        # quat wxyz: X90*Y90 = [0.5, 0.5, 0.5, 0.5]
        mj_data.qpos[cube_qadr+3:cube_qadr+7] = [0.5, 0.5, 0.5, 0.5]

    # Initial gripper state depends on context
    if gripper_anim == "close":
        mj_data.qpos[f1_idx] = grip_open
        mj_data.qpos[f2_idx] = grip_open
    elif gripper_anim == "open":
        mj_data.qpos[f1_idx] = grip_closed
        mj_data.qpos[f2_idx] = grip_closed
    elif trial_data.get("gripper_closed"):
        mj_data.qpos[f1_idx] = grip_closed
        mj_data.qpos[f2_idx] = grip_closed
    else:
        mj_data.qpos[f1_idx] = grip_open
        mj_data.qpos[f2_idx] = grip_open

    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, REPLAY_FPS, (W, H))

    # Should cube follow gripper? (grasped state)
    move_cube = trial_data.get("gripper_closed", False) and cube_qadr >= 0 and session_cube is not None
    # Compute initial EE-to-cube offset if cube should move
    cube_z_offset = 0
    if move_cube:
        # Cube center is below EE by some offset; compute from initial state
        mj_data.qpos[arm_start:arm_start+7] = frames[0]
        mj.mj_forward(model, mj_data)
        from ik_solver import get_current_ee_pose
        init_ee, _ = get_current_ee_pose(frames[0])
        cube_offset = np.array(session_cube) - init_ee  # XYZ offset from EE to cube

    def capture(joints, grip_val=None):
        mj_data.qpos[arm_start:arm_start+7] = joints
        if grip_val is not None:
            mj_data.qpos[f1_idx] = grip_val
            mj_data.qpos[f2_idx] = grip_val
        if move_cube:
            from ik_solver import get_current_ee_pose
            ee_pos, _ = get_current_ee_pose(joints)
            mj_data.qpos[cube_qadr:cube_qadr+3] = ee_pos + cube_offset
        mj.mj_forward(model, mj_data)
        renderer.update_scene(mj_data, cam)
        _add_markers(renderer, start_ee, end_ee, ee_positions)
        pixels = renderer.render()
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    motion_frames = len(frames)

    # Precompute gripper values for each motion frame
    grip_vals = [None] * motion_frames
    if gripper_anim:
        for i in range(motion_frames):
            t = i / max(motion_frames - 1, 1)
            if gripper_anim == "close":
                grip_vals[i] = grip_open + t * (grip_closed - grip_open)
            else:
                grip_vals[i] = grip_closed + t * (grip_open - grip_closed)

    if camera_frames > 0:
        if camera_frames >= motion_frames:
            # Normal case: pad start (idle period) + motion + hold end
            pad_end = min(int(REPLAY_FPS * 1.0), max(0, camera_frames - motion_frames))
            pad_start = max(0, camera_frames - motion_frames - pad_end)
        else:
            # Camera shorter than motion (e.g. close_gripper ~2s)
            # Just render motion frames, match camera count
            pad_start = 0
            pad_end = 0

        for _ in range(pad_start):
            capture(frames[0], grip_vals[0])
        for i, f in enumerate(frames):
            capture(f, grip_vals[i])
        for _ in range(pad_end):
            capture(frames[-1], grip_vals[-1])

        total_frames = pad_start + motion_frames + pad_end
    else:
        hold = int(REPLAY_FPS * 0.5)
        for _ in range(hold):
            capture(frames[0], grip_vals[0])
        for i, f in enumerate(frames):
            capture(f, grip_vals[i])
        for _ in range(hold):
            capture(frames[-1], grip_vals[-1])
        total_frames = motion_frames + 2 * hold

    writer.release()

    total_dur = total_frames / REPLAY_FPS

    ret = subprocess.run([
        "ffmpeg", "-y", "-i", tmp_path,
        "-c:v", "libx264", "-tag:v", "avc1",
        "-profile:v", "baseline", "-level", "3.0",
        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-x264-params", "keyint=1:scenecut=0", "-bf", "0",
        "-movflags", "+faststart", "-an",
        output_path
    ], capture_output=True)

    if ret.returncode == 0:
        os.unlink(tmp_path)
        sz = os.path.getsize(output_path) // 1024
        n_segs = trial_data["n_segments"]
        motion_s = trial_data["motion_duration_s"]
        cam_info = f", camera={camera_frames}f" if camera_frames > 0 else ""
        print(f"    {total_frames} frames @ {REPLAY_FPS}fps = {total_dur:.1f}s "
              f"(motion: {motion_frames}f={motion_s:.1f}s{cam_info}) [{sz}KB]")
    else:
        os.rename(tmp_path, output_path)
        print(f"    Saved (mp4v fallback): {output_path}")


def load_camera_settings():
    cam_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "preview_cam.json")
    import mujoco as mj
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.4, 0.0, 0.2]
    cam.distance = 1.3
    cam.azimuth = 145
    cam.elevation = -25
    if os.path.exists(cam_path):
        try:
            saved = json.load(open(cam_path))
            if "lookat" in saved:
                cam.lookat[:] = saved["lookat"]
            for k in ["distance", "azimuth", "elevation"]:
                if k in saved:
                    setattr(cam, k, saved[k])
        except Exception:
            pass
    return cam


def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_preview.py <session_dir> [session_dir2 ...]")
        sys.exit(1)

    import mujoco

    mujoco_src = os.path.expanduser("~/kinam_dev/Mujoco_Franka/src")
    if mujoco_src not in sys.path:
        sys.path.insert(0, mujoco_src)
    from utils import load_calib, make_model_with_cube

    T_base_cam = load_calib()
    cam = load_camera_settings()

    model = make_model_with_cube(T_base_cam, [0.4, 0.0, 0.0], [1, 0, 0, 0],
                                cube_size=(0.02, 0.02, 0.025))

    # Make cube red instead of wood_block brown
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
    if cube_geom_id >= 0:
        model.geom_rgba[cube_geom_id] = [0.85, 0.12, 0.1, 1.0]
        model.geom_matid[cube_geom_id] = -1  # disable material override

    arm_start = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'fr3_joint1')]
    f1 = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')]
    f2 = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')]

    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_joint')
    cube_qadr = model.jnt_qposadr[cube_jnt_id] if cube_jnt_id >= 0 else -1

    renderer = mujoco.Renderer(model, height=600, width=800)

    for session_dir in sys.argv[1:]:
        session_dir = os.path.abspath(session_dir)
        if not os.path.isdir(session_dir):
            continue
        sess_name = os.path.basename(session_dir)
        print(f"\n=== Session: {sess_name} ===")

        trials = build_session_trajectory(session_dir)
        print(f"  {len(trials)} motion trials to render")

        # Find cube position
        session_cube_pos = None
        for td in trials:
            if td.get("cube_pos"):
                session_cube_pos = td["cube_pos"]
                break
        if session_cube_pos is None:
            for trial_dir in sorted(glob.glob(os.path.join(session_dir, "trial_*"))):
                ctx_path = os.path.join(trial_dir, "context.json")
                if os.path.exists(ctx_path):
                    ctx = json.load(open(ctx_path))
                    if ctx.get("cube_pos"):
                        session_cube_pos = ctx["cube_pos"]
                        break

        if session_cube_pos:
            print(f"  Cube at {[round(v,3) for v in session_cube_pos]}")

        for td in trials:
            td["_cube_qadr"] = cube_qadr
            td["_session_cube_pos"] = session_cube_pos
            print(f"\n  {td['trial_name']} ({td['skill']}): \"{td['command']}\"")
            out = os.path.join(td["trial_dir"], "preview_replay.mp4")
            render_trial(td, out, model, arm_start, f1, f2, cam, renderer)

    renderer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
