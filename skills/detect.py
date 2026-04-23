"""Skill: detect -- One-shot cube 6D pose via subprocess (SAM2 + FoundationPose).

Runs fp_oneshot.py in the FP venv python as a subprocess to avoid
pytorch3d / torch version conflicts with the CLI environment.
Communication: numpy files + JSON result via /dev/shm.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np

from config import (FP_DIR, FP_VENV, CALIB_PATH, FP_SHM_PATH,
                    WORKSPACE_MIN, WORKSPACE_MAX)

_CLI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP_ONESHOT_SCRIPT = os.path.join(_CLI_DIR, "fp_oneshot.py")
FP_VENV_PYTHON = os.path.join(FP_VENV, "bin", "python3")

# Temp files for IPC
SHM_RGB = "/dev/shm/cli_rgb.npy"
SHM_DEPTH = "/dev/shm/cli_depth.npy"
SHM_K = "/dev/shm/cli_K.npy"
SHM_RESULT = "/dev/shm/cli_fp_result.json"


class DetectSkill:
    """One-shot cube 6D pose detection via FP subprocess."""

    name = "detect"
    description = "Detect red cube 6D pose (SAM2 + FoundationPose)"

    def execute(self, node, context):
        import rclpy

        print("  [detect] Capturing camera frame...")

        # Spin to get fresh frame
        for _ in range(30):
            rclpy.spin_once(node, timeout_sec=0.1)
        rgb, depth, K = node.get_camera_frame()

        if rgb is None or depth is None or K is None:
            print("  x Camera data not available. Is cam_base running?")
            return False

        print(f"  [detect] Camera OK: RGB {rgb.shape}, Depth {depth.shape}")

        # Save to /dev/shm for subprocess
        np.save(SHM_RGB, rgb)
        np.save(SHM_DEPTH, depth)
        np.save(SHM_K, K)

        # Build subprocess command
        if not os.path.exists(FP_VENV_PYTHON):
            print(f"  x FP venv python not found: {FP_VENV_PYTHON}")
            return False

        cmd = [
            FP_VENV_PYTHON, FP_ONESHOT_SCRIPT,
            "--rgb", SHM_RGB,
            "--depth", SHM_DEPTH,
            "--K", SHM_K,
            "--output", SHM_RESULT,
            "--calib", CALIB_PATH,
        ]

        # Environment: need ROS2 libs for the FP venv python
        env = os.environ.copy()
        ld = env.get("LD_LIBRARY_PATH", "")
        ros_libs = "/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu"
        if ros_libs not in ld:
            env["LD_LIBRARY_PATH"] = ros_libs + ":" + ld if ld else ros_libs

        print("  [detect] Running FP subprocess (SAM2 + FoundationPose)...")
        print("           (first run loads models, may take ~30-60s)")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=env,
                capture_output=True, text=True,
                timeout=180,  # 3 min max
            )
        except subprocess.TimeoutExpired:
            print("  x FP subprocess timed out (180s)")
            self._cleanup_shm()
            return False

        elapsed = time.time() - t0

        # Show subprocess output
        if proc.stdout.strip():
            for line in proc.stdout.strip().split("\n"):
                print(f"    {line}")

        if proc.returncode != 0:
            print(f"  x FP subprocess failed (exit code {proc.returncode})")
            if proc.stderr.strip():
                for line in proc.stderr.strip().split("\n")[-10:]:
                    print(f"    {line}")
            self._cleanup_shm()
            return False

        # Read result
        if not os.path.exists(SHM_RESULT):
            print("  x FP result file not found")
            self._cleanup_shm()
            return False

        with open(SHM_RESULT) as f:
            result = json.load(f)

        if result.get("status") != "ok":
            err = result.get("error", "unknown")
            print(f"  x FP detection failed: {err}")
            self._cleanup_shm()
            return False

        pos = np.array(result["cube_pos"])
        quat_wxyz = np.array(result["cube_quat_wxyz"])

        # Write to shared memory for other tools
        fp_data = {
            "status": "tracking",
            "timestamp": time.time(),
            "cube_pos": pos.tolist(),
            "cube_quat_wxyz": quat_wxyz.tolist(),
        }
        tmp = FP_SHM_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(fp_data, f)
        os.replace(tmp, FP_SHM_PATH)

        # Display
        in_ws = bool(np.all(pos >= WORKSPACE_MIN) and np.all(pos <= WORKSPACE_MAX))
        ws_str = "OK" if in_ws else "OUTSIDE!"
        print(f"\n  Cube detected! ({elapsed:.1f}s)")
        print(f"    Position (base): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")
        print(f"    Orientation:     [w={quat_wxyz[0]:.4f}, x={quat_wxyz[1]:.4f}, "
              f"y={quat_wxyz[2]:.4f}, z={quat_wxyz[3]:.4f}]")
        print(f"    Workspace:       {ws_str}")
        print(f"    Mask pixels:     {result.get('mask_pixels', '?')}")

        # Store in context
        context["cube_pos"] = pos.copy()
        context["cube_quat"] = quat_wxyz.copy()
        context["cube_detect_time"] = time.time()

        self._cleanup_shm()
        return True

    def _cleanup_shm(self):
        """Remove temp files from /dev/shm."""
        for p in [SHM_RGB, SHM_DEPTH, SHM_K, SHM_RESULT]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass
