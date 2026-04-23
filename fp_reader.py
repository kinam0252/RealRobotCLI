"""Read FoundationPose cube pose from shared memory."""

import json
import time
import numpy as np
from config import FP_SHM_PATH, FP_MAX_AGE_S


def read_fp_pose(max_age_s=None):
    """Read latest cube pose from FP tracker via /dev/shm.

    Returns:
        (pos, quat_wxyz, age_s) if valid tracking data available.
        (None, None, None) if unavailable/stale/not tracking.
    """
    if max_age_s is None:
        max_age_s = FP_MAX_AGE_S
    try:
        with open(FP_SHM_PATH, "r") as f:
            data = json.load(f)

        status = data.get("status", "unknown")
        if status != "tracking":
            return None, None, None

        age = time.time() - data["timestamp"]
        if age > max_age_s:
            return None, None, None

        pos = np.array(data["cube_pos"], dtype=np.float64)
        quat_wxyz = np.array(data["cube_quat_wxyz"], dtype=np.float64)
        return pos, quat_wxyz, age

    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
        return None, None, None


def read_fp_status():
    """Read FP tracker status string (for display)."""
    try:
        with open(FP_SHM_PATH, "r") as f:
            data = json.load(f)
        status = data.get("status", "unknown")
        age = time.time() - data.get("timestamp", 0)
        return status, age
    except (FileNotFoundError, json.JSONDecodeError):
        return "not_running", -1
