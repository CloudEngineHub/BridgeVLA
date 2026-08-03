# MemoryBench peract_utils. Same scene bounds and camera set as GemBench so a
# BridgeVLA model trained on GemBench transfers without architecture changes.
#
# What IMAGE_SIZE actually drives (verified by reading the codebase):
#   - LIVE  : MemoryBench_Dataset.image_size  -> the resolution at which RGB +
#             PCD are emitted to training. Set to 128 to match the raw saved
#             data (no resize), which is also what sam2act and RLBench use.
#   - LIVE  : client.py default for RLBenchEnv(image_size=[IMAGE_SIZE, IMAGE_SIZE])
#             at eval time -> CoppeliaSim renders observations at this resolution.
#   - DEAD  : the historic call ``RVTAgent(image_resolution=[IMAGE_SIZE, IMAGE_SIZE])``
#             stored the value on ``self._image_resolution`` but never read it
#             back. We've dropped that kwarg from train.py / actioner.py.
#
# Why 128 (the raw saved-data resolution)?
#   MemoryBench RLBench episodes are stored as 128x128 depth PNGs + 128 intrinsics
#   (verified: ``Image.open(front_depth/0.png).size == (128, 128)``). The PCD
#   ``get_stored_demo`` returns is computed by ``pointcloud_from_depth_and_camera_params``
#   from that 128-res depth — so 128-res PCD is the only resolution at which every
#   pixel has a faithful, unique 3D position.
#
#   Earlier we tried IMAGE_SIZE=256 to match GemBench's LMDB (which DOES store
#   native 256x256 PCD because GemBench was re-rendered at 256 by CoppeliaSim with
#   matching intrinsics). On MemoryBench's 128-native depth that's impossible
#   without re-running CoppeliaSim: any upsample fabricates data. Bilinear-
#   upsampling XYZ across a depth discontinuity (gripper edge vs background
#   table) blends a near and a far 3D point and emits a phantom point at
#   intermediate depth, which the MVT renderer splatters as visible "floating"
#   streaks in the front/right-view dumps.
#
#   IMAGE_SIZE=128 sidesteps all that: ``MemoryBench_Dataset._resize_*`` becomes
#   a no-op, samples leave the loader at the raw simulator resolution, and the
#   pipeline matches sam2act / RLBench exactly. The MVT renderer projects to
#   ``mvt.img_size`` (220) regardless of input PCD resolution, so the
#   GemBench-warm-start input distribution downstream of the renderer is
#   essentially unchanged — only the upstream point density per camera drops
#   from 65k (256x256) to 16k (128x128), still well above what the splat radius
#   (~5x5 px on the 220 output) needs for full coverage.

CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max]
IMAGE_SIZE = 128
VOXEL_SIZES = [100]
LOW_DIM_SIZE = 4
ROTATION_RESOLUTION = 5

# 3 MemoryBench tasks. Each ships in sam2act/libs/RLBench/rlbench/tasks/ and
# /data/bridgevla_data/memorybench/data/files/. install_memorybench_tasks.sh
# copies them into the active RLBench install before runs.
MEMORYBENCH_TASKS = ["put_block_back", "rearrange_block", "reopen_drawer"]


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_memorybench(replay_sample, cameras):
    """Mirror of `_preprocess_inputs_gembench`: collect [rgb, pcd] per camera."""
    obs, pcds = [], []
    for n in cameras:
        rgb = replay_sample[n]["rgb"]
        pcd = replay_sample[n]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds


def _preprocess_inputs_memorybench_prefixed(replay_sample, cameras, prefix):
    obs, pcds = [], []
    for n in cameras:
        sub = replay_sample[f"{prefix}{n}"]
        rgb = _norm_rgb(sub["rgb"])
        pcd = sub["pcd"]
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds
