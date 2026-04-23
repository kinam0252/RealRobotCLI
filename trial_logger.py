"""Trial logger -- record commands, preview, and camera video per trial.

Directory structure:
  <repo_root>/output/
    YYYYMMDD_HHMMSS/                   # session
      session_log.json                 # all commands + timestamps
      trial_001_detect/                # per-trial
        command.json                   # skill name, params, result, timing
        camera.mp4                     # cam_base RGB during execution
      trial_002_move/
        command.json
        preview.mp4                    # MuJoCo preview recording
        camera.mp4
      ...
"""

import json
import os
import threading
import time
from datetime import datetime

import numpy as np

_CLI_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(_CLI_DIR, "output")


class CameraRecorder:
    """Record RGB frames from cam_base to an mp4 file (thread-safe)."""

    def __init__(self, resolution=(640, 480), fps=15):
        self.resolution = resolution
        self.fps = fps
        self._writer = None
        self._recording = False
        self._lock = threading.Lock()
        self.frame_count = 0

    def start(self, video_path):
        import cv2
        with self._lock:
            if self._writer:
                self._writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.resolution)
            self.frame_count = 0
            self._recording = True

    def feed_frame(self, rgb_image):
        """Feed an RGB numpy array (H, W, 3). Thread-safe."""
        import cv2
        with self._lock:
            if self._recording and self._writer:
                resized = cv2.resize(rgb_image, self.resolution)
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                self._writer.write(bgr)
                self.frame_count += 1

    def stop(self):
        with self._lock:
            self._recording = False
            if self._writer:
                self._writer.release()
                self._writer = None

    @property
    def is_recording(self):
        with self._lock:
            return self._recording


class TrialLogger:
    """Manages per-session directories and per-trial logging."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.session_dir = None
        self.session_log = []
        self.trial_count = 0
        self._current_trial_dir = None
        self._current_trial_data = None
        self._camera_recorder = CameraRecorder()
        self._trial_start_time = None

        if self.enabled:
            self._init_session()

    def _init_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(OUTPUT_ROOT, ts)
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"  [Logger] Session: {self.session_dir}")

    def start_trial(self, skill_name, user_command="", params=None):
        """Begin a new trial. Returns trial_dir path."""
        if not self.enabled:
            return None

        self.trial_count += 1
        trial_name = f"trial_{self.trial_count:03d}_{skill_name}"
        self._current_trial_dir = os.path.join(self.session_dir, trial_name)
        os.makedirs(self._current_trial_dir, exist_ok=True)

        self._trial_start_time = time.time()
        self._current_trial_data = {
            "trial": self.trial_count,
            "skill": skill_name,
            "command": user_command,
            "params": params or {},
            "start_time": datetime.now().isoformat(),
            "start_epoch": self._trial_start_time,
        }

        return self._current_trial_dir

    def get_trial_dir(self):
        return self._current_trial_dir

    @property
    def current_trial_dir(self):
        return self._current_trial_dir

    def start_camera_recording(self, node):
        """Start recording cam_base RGB to camera.mp4 in current trial dir."""
        if not self.enabled or not self._current_trial_dir:
            return
        video_path = os.path.join(self._current_trial_dir, "camera.mp4")
        self._camera_recorder.start(video_path)
        # Subscribe to camera feed (use node's existing camera callback)
        self._camera_feed_active = True

    def feed_camera_frame(self, rgb_image):
        """Call this with each new camera frame during recording."""
        if self._camera_recorder.is_recording:
            self._camera_recorder.feed_frame(rgb_image)

    def stop_camera_recording(self):
        """Stop camera recording and return frame count."""
        self._camera_recorder.stop()
        return self._camera_recorder.frame_count

    def get_preview_video_path(self):
        """Return path for preview video in current trial dir."""
        if not self.enabled or not self._current_trial_dir:
            return None
        return os.path.join(self._current_trial_dir, "preview.mp4")

    def end_trial(self, success=True, result_data=None):
        """Finalize current trial: stop recording, save command.json."""
        if not self.enabled or not self._current_trial_dir:
            return

        # Stop camera recording
        self.stop_camera_recording()

        # Complete trial data
        end_time = time.time()
        self._current_trial_data["end_time"] = datetime.now().isoformat()
        self._current_trial_data["duration_s"] = round(
            end_time - self._trial_start_time, 2)
        self._current_trial_data["success"] = success
        if result_data:
            self._current_trial_data["result"] = _serialize(result_data)

        # Check what files exist in trial dir
        files = os.listdir(self._current_trial_dir)
        self._current_trial_data["files"] = files
        self._current_trial_data["camera_frames"] = self._camera_recorder.frame_count

        # Save command.json
        cmd_path = os.path.join(self._current_trial_dir, "command.json")
        with open(cmd_path, "w") as f:
            json.dump(self._current_trial_data, f, indent=2, ensure_ascii=False)

        # Append to session log
        self.session_log.append(self._current_trial_data.copy())
        self._save_session_log()

        dur = self._current_trial_data["duration_s"]
        frames = self._camera_recorder.frame_count
        print(f"  [Logger] Trial {self.trial_count} saved "
              f"({dur:.1f}s, {frames} camera frames)")

        self._current_trial_dir = None
        self._current_trial_data = None

    def _save_session_log(self):
        """Save the full session log."""
        if not self.session_dir:
            return
        log_path = os.path.join(self.session_dir, "session_log.json")
        with open(log_path, "w") as f:
            json.dump(self.session_log, f, indent=2, ensure_ascii=False)

    def save_context_snapshot(self, context):
        """Save a snapshot of the shared context to current trial dir."""
        if not self.enabled or not self._current_trial_dir:
            return
        snap = _serialize(context)
        path = os.path.join(self._current_trial_dir, "context.json")
        with open(path, "w") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)


def _serialize(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()
                if not k.startswith("_")}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
