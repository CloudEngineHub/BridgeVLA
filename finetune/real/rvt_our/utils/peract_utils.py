"""
Constants and preprocessing utilities for real-robot evaluation.

Scene bounds and camera names are configured for the Dobot real-robot setup.
"""

import torch


# Constants
# Keep in sync with real/utils/peract_utils.py (SCENE_BOUNDS_REAL).
# z (height) widened to -0.15..0.55 (was -0.07..0.47) to stop the tighter
# 1.5x-zoom crop from occasionally clipping the tabletop itself.
# y (top-view vertical / left-right) widened to -0.46..0.36 (was -0.42..0.32);
# y_len=0.82 still < x_len=0.86 so render scale / inference zoom unchanged.
SCENE_BOUNDS = [
    -0.88, -0.46, -0.15,
    -0.02,  0.36,  0.55,
]

CAMERAS = ["3rd"]
IMAGE_SIZE = 224
IMG_STRIDE = 4  # ZED downsample stride used by Real_Dataset during training


# Preprocessing
def _norm_rgb(x):
    """Normalise uint8 [0, 255] → float [-1, 1]."""
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(replay_sample, cameras):
    """
    Build (obs, pcds) lists from a real-robot observation dict.

    Args:
        replay_sample: dict with camera name keys, each holding
                       ``"rgb"`` and ``"pcd"`` tensors of shape (1, C, H, W).
        cameras: list of camera name strings, e.g. ``["3rd"]``.

    Returns:
        obs:  list of ``[normed_rgb, pcd]`` per camera
        pcds: list of pcd tensors per camera
    """
    obs, pcds = [], []
    for cam in cameras:
        rgb = replay_sample[cam]["rgb"]
        pcd = replay_sample[cam]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds

