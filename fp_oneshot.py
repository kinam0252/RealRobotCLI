#!/usr/bin/env python3
"""One-shot FoundationPose detection -- runs in FP venv as subprocess.

Usage (from FP venv python):
  python fp_oneshot.py --rgb /dev/shm/cli_rgb.npy \
                       --depth /dev/shm/cli_depth.npy \
                       --K /dev/shm/cli_K.npy \
                       --output /dev/shm/cli_fp_result.json

Reads numpy arrays, runs SAM2 mask + FP register, writes JSON result.
"""

import os, sys, time, json, argparse, logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s [FP-oneshot] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger()

# ── Paths ──
FP_DIR = os.path.expanduser("~/kinam_dev/FoundationPose")
sys.path.insert(0, FP_DIR)
sys.path.insert(0, os.path.join(FP_DIR, "mycpp", "build"))

CUBE_MESH = os.path.join(FP_DIR, "box_4x4x8cm.obj")
CALIB_PATH = os.path.expanduser("~/kinam_dev/Mujoco/preproc/data/calib_result_base.calib")

SAM_VENV_SITE = os.path.join(FP_DIR, "venv_sam", "lib", "python3.10", "site-packages")
SAM_CKPT = os.path.join(FP_DIR, "sam_weights", "sam2.1_hiera_tiny.pt")


def load_calib(calib_path):
    import yaml
    from scipy.spatial.transform import Rotation as R
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    t = calib["transform"]["translation"]
    q = calib["transform"]["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T


def run_sam_detect(rgb):
    """SAM2 auto-detect red cube mask."""
    import gc, torch

    added = SAM_VENV_SITE not in sys.path
    if added:
        sys.path.insert(0, SAM_VENV_SITE)

    try:
        import sam2
        from sam2.build_sam import _load_checkpoint
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from hydra.core.global_hydra import GlobalHydra
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        cfg_dir = os.path.join(os.path.dirname(sam2.__file__), "configs")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="sam2.1/sam2.1_hiera_t")
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, SAM_CKPT)
        model = model.to("cuda").eval()

        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=16, points_per_batch=32,
            pred_iou_thresh=0.7, stability_score_thresh=0.8,
            min_mask_region_area=200,
        )

        h, w = rgb.shape[:2]
        max_side = 1024
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        else:
            small = rgb
            scale = 1.0

        with torch.inference_mode():
            masks = generator.generate(small)

        del generator, model
        gc.collect()
        torch.cuda.empty_cache()

        total_px = small.shape[0] * small.shape[1]
        best = None
        for m in masks:
            frac = m["area"] / total_px
            if frac > 0.15 or frac < 0.001:
                continue
            seg = m["segmentation"].astype(bool)
            pixels = small[seg]
            if len(pixels) == 0:
                continue
            mr, mg, mb = pixels.mean(axis=0)
            redness = mr - max(mg, mb)
            rows, cols = np.where(seg)
            box_area = (cols.max() - cols.min() + 1) * (rows.max() - rows.min() + 1)
            solidity = m["area"] / max(box_area, 1)
            if redness > 30 and 0.002 < frac < 0.05 and solidity > 0.5:
                if best is None or redness > best[1]:
                    best = (m["segmentation"], redness)

        if best is None:
            return None

        mask = (best[0] * 255).astype(np.uint8)
        if scale != 1.0:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    finally:
        if added and SAM_VENV_SITE in sys.path:
            sys.path.remove(SAM_VENV_SITE)


def run_fp_register(rgb, depth, K, mask):
    """Run FoundationPose register for 6D pose."""
    import trimesh
    from estimater import FoundationPose
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr

    mesh = trimesh.load(CUBE_MESH)
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=scorer, refiner=refiner,
        debug_dir="/tmp/fp_oneshot_debug", debug=0, glctx=glctx,
    )

    pose_cam = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
    return pose_cam, to_origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib", default=CALIB_PATH)
    args = parser.parse_args()

    t0 = time.time()
    result = {"status": "error", "error": "unknown"}

    try:
        rgb = np.load(args.rgb)
        depth = np.load(args.depth)
        K = np.load(args.K)
        log.info(f"Loaded: RGB {rgb.shape}, Depth {depth.shape}, K {K.shape}")

        # SAM2 mask
        log.info("Running SAM2 detection...")
        mask = run_sam_detect(rgb)
        if mask is None or mask.sum() < 100:
            result = {"status": "error", "error": "SAM2: cube not detected"}
            return

        mask_px = int(mask.sum() // 255)
        log.info(f"SAM2 mask: {mask_px} pixels")

        # FP register
        log.info("Running FoundationPose register...")
        pose_cam, to_origin = run_fp_register(rgb, depth, K, mask)

        # Convert to base frame
        from scipy.spatial.transform import Rotation as R
        T_base_cam = load_calib(args.calib).astype(np.float64)
        center_pose = pose_cam @ np.linalg.inv(to_origin)
        pose_base = T_base_cam @ center_pose
        pos = pose_base[:3, 3]
        quat_xyzw = R.from_matrix(pose_base[:3, :3]).as_quat()
        quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                     float(quat_xyzw[1]), float(quat_xyzw[2])]

        result = {
            "status": "ok",
            "cube_pos": [float(x) for x in pos],
            "cube_quat_wxyz": quat_wxyz,
            "mask_pixels": mask_px,
            "elapsed_s": round(time.time() - t0, 2),
        }
        log.info(f"Done: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    except Exception as e:
        import traceback
        result = {"status": "error", "error": str(e),
                  "traceback": traceback.format_exc()}
        log.error(f"Failed: {e}")

    finally:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
