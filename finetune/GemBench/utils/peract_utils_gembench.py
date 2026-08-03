# Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/utils/peract_utils.py
from peract_colab.arm.utils import stack_on_channel

CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 256
VOXEL_SIZES = [100]  # 100x100x100 voxels
LOW_DIM_SIZE = 4  # {left_finger_joint, right_finger_joint, gripper_open, timestep}
ROTATION_RESOLUTION = 5  # degree increments per axis


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_gembench(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = replay_sample[n]["rgb"]
        pcd = replay_sample[n]["pcd"]

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


def _preprocess_inputs_gembench_prefixed(replay_sample, cameras, prefix):
    """Same preprocessing as ``_preprocess_inputs_gembench`` but pulls each
    camera's data from ``replay_sample[f"{prefix}{cam}"]``. Used for the
    episodic memory's anchor (prefix="anchor_") and history slots
    (prefix="hist{k}_").
    """
    obs, pcds = [], []
    for n in cameras:
        sub = replay_sample[f"{prefix}{n}"]
        rgb = _norm_rgb(sub["rgb"])
        pcd = sub["pcd"]
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds
