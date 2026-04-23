#!/bin/bash
# Launch FoundationPose tracker server for RealRobotCLI
# Run this in a separate terminal BEFORE starting grasp_cli.py

set -e

export LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH}

FP_DIR=~/kinam_dev/FoundationPose
VENV_PYTHON=${FP_DIR}/venv/bin/python3
SCRIPT=~/kinam_dev/execute_real_robot/fp_tracker_server.py

echo "=== RealRobotCLI — FoundationPose Tracker ==="
echo "  Mesh:  ${FP_DIR}/box_4x4x8cm.obj"
echo "  Calib: ~/kinam_dev/Mujoco/preproc/data/calib_result_base.calib"
echo ""

${VENV_PYTHON} ${SCRIPT} \
    --mesh ${FP_DIR}/box_4x4x8cm.obj \
    --bowl-mesh ${FP_DIR}/bowl_r95_h10mm.obj \
    --calib ~/kinam_dev/Mujoco/preproc/data/calib_result_base.calib \
    --show-vis \
    "$@"
