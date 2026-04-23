"""Preview -- show planned trajectory in MuJoCo viewer before real execution."""

import json
import os
import subprocess
import time
import numpy as np

from config import (KINAM_ROOT, CART_N_WAYPOINTS, CART_INTERVAL_S,
                    CALIB_PATH, HOME_JOINTS)

PREVIEW_SHM = "/dev/shm/cli_preview.json"
PREVIEW_DISABLED = False  # set True via --no-preview
PREVIEW_VIEWER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "preview_viewer.py")
MUJOCO_PYTHON = os.path.expanduser("~/miniconda3/envs/mujoco_viewer/bin/python")


def expand_trajectory(waypoints, current_joints, n_per_seg=None):
    """Expand waypoints into a full joint-space frame list.

    Returns list of (7,) arrays -- one per interpolation step.
    """
    if n_per_seg is None:
        n_per_seg = CART_N_WAYPOINTS

    frames = []
    prev = np.asarray(current_joints, dtype=np.float64)

    for wp in waypoints:
        wp = np.asarray(wp, dtype=np.float64)
        diff = wp - prev
        for i in range(1, n_per_seg + 1):
            t = i / n_per_seg
            frames.append((prev + t * diff).tolist())
        prev = wp

    return frames


def write_preview(current_joints, waypoints, cube_pos=None, cube_quat=None,
                  gripper_width=None):
    """Write preview data to shared memory for the viewer."""
    frames = expand_trajectory(waypoints, current_joints)

    data = {
        "timestamp": time.time(),
        "current_joints": np.asarray(current_joints).tolist(),
        "waypoints": [np.asarray(w).tolist() for w in waypoints],
        "frames": frames,
        "frame_dt": CART_INTERVAL_S,
    }

    if cube_pos is not None:
        data["cube_pos"] = np.asarray(cube_pos).tolist()
    if cube_quat is not None:
        data["cube_quat_wxyz"] = np.asarray(cube_quat).tolist()
    if gripper_width is not None:
        data["gripper_width"] = float(gripper_width)

    tmp = PREVIEW_SHM + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, PREVIEW_SHM)


def launch_preview_viewer():
    """Launch MuJoCo preview viewer as subprocess. Returns Popen or None."""
    if not os.path.exists(MUJOCO_PYTHON):
        return None
    if not os.path.exists(PREVIEW_VIEWER):
        return None

    env = os.environ.copy()
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":1"
    env["MUJOCO_GL"] = "glx"

    cmd = [MUJOCO_PYTHON, PREVIEW_VIEWER]

    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return proc
    except Exception as e:
        print(f"  Warning: Preview viewer launch failed: {e}")
        return None


def _start_ffmpeg_capture(record_path, display=None):
    """Start ffmpeg screen capture of the X11 display. Returns Popen or None."""
    if display is None:
        display = os.environ.get("DISPLAY", ":1")

    cmd = [
        "ffmpeg", "-y",
        "-f", "x11grab",
        "-framerate", "15",
        "-video_size", "1280x720",
        "-i", display,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        record_path,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception:
        return None


def _stop_ffmpeg(proc):
    """Gracefully stop ffmpeg by sending 'q'."""
    if proc is None:
        return
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def cleanup_preview():
    """Remove preview shared memory file."""
    try:
        os.remove(PREVIEW_SHM)
    except FileNotFoundError:
        pass


def preview_and_confirm(node, waypoints, context, prompt_text="Execute? [y/N]: ",
                       record_path=None):
    """Full preview flow: write data -> launch viewer -> ask user -> cleanup.

    Returns True if user confirms, False otherwise.
    Falls back to text-only prompt if viewer unavailable or PREVIEW_DISABLED.
    record_path: if set, captures the viewer window via ffmpeg to this mp4.
    """
    if PREVIEW_DISABLED:
        return _text_confirm(prompt_text)
    cur_joints = node.get_joints()
    if cur_joints is None:
        print("  Warning: No joint state for preview.")
        return _text_confirm(prompt_text)

    # Gather context data
    cube_pos = context.get("cube_pos")
    cube_quat = context.get("cube_quat")
    gripper_width = node.get_gripper_width()

    # Write preview data
    write_preview(cur_joints, waypoints, cube_pos, cube_quat, gripper_width)

    # Launch interactive viewer
    proc = launch_preview_viewer()
    if proc is None:
        print("  Warning: MuJoCo preview unavailable, text-only confirmation.")
        cleanup_preview()
        return _text_confirm(prompt_text)

    # Start screen capture if recording requested
    ffmpeg_proc = None
    if record_path:
        time.sleep(0.5)  # wait for viewer window to appear
        ffmpeg_proc = _start_ffmpeg_capture(record_path)

    print("  Preview viewer launched.")
    print("     (check the viewer, then confirm)")

    # Ask user while viewer is running
    confirmed = _text_confirm(prompt_text)

    # Stop recording first (before killing viewer)
    if ffmpeg_proc is not None:
        _stop_ffmpeg(ffmpeg_proc)
        if os.path.exists(record_path):
            sz = os.path.getsize(record_path)
            print(f"  Preview recorded: {record_path} ({sz//1024}KB)")

    # Terminate viewer
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

    cleanup_preview()
    return confirmed


def _text_confirm(prompt_text):
    """Simple text y/n confirmation."""
    try:
        ans = input(f"\n  {prompt_text}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return False
    return ans == 'y'
