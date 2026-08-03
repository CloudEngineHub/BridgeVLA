#Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/utils/peract_utils.py
import os

import torch


CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
SCENE_BOUNDS = [
    -0.3,
    -0.5,
    0.6,
    0.7,
    0.5,
    1.6,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
IMAGE_SIZE = 128
# RLBench data root; train.sh/eval.sh export RLBENCH_DATA_FOLDER from
# ${BRIDGEVLA_DATA_ROOT}/RLBench. The fallback derives the same path from
# this file's location so bare `python` runs work.
_BRIDGEVLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_FOLDER = os.environ.get(
    "RLBENCH_DATA_FOLDER",
    os.path.join(_BRIDGEVLA_ROOT, "data", "bridgevla_data", "RLBench"),
)


def _stack_on_channel(x):
    """Local, dependency-free equivalent of peract_colab.arm.utils.stack_on_channel
    (avoids requiring the ``peract_colab`` package, which isn't installed in
    every env that imports this module, e.g. GemBench eval). Collapses the
    temporal dim (dim=1) into the channel dim (dim=2): (B, T, C, ...) -> (B, T*C, ...)."""
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(replay_sample, cameras):
    """Shared by ``bridgevla.models.bridgevla_agent.RVTAgent`` (train ``update``
    and eval ``act``) across benchmarks. Expects flat ``"{cam}_rgb"`` /
    ``"{cam}_point_cloud"`` keys, each shaped (B, T, C, H, W)."""
    obs, pcds = [], []
    for n in cameras:
        rgb = _stack_on_channel(replay_sample["%s_rgb" % n])
        pcd = _stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds
