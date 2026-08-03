"""Constants + preprocessing for the Dobot real-robot (ZED) pipeline.

Scene bounds and camera list are carried over from the old BridgeVLA_Real
setup that produced the 20251011 dataset. Image normalization matches the
GemBench/RLBench preprocessing (RGB in [-1, 1]).
"""

CAMERAS_REAL = ["3rd"]

SCENE_BOUNDS_REAL = [
    -0.88, -0.46, -0.15,
    -0.02,  0.36,  0.55,
]  # [x_min, y_min, z_min, x_max, y_max, z_max] in Dobot base frame (meters)
   # 1.5x MVT1 zoom: x/y extents shrunk to 2/3 of original around same centre;
   # scale = 2/max_extent = 2/0.86 ≈ 2.33 (was 2/1.3 ≈ 1.54 → 1.51x).
   # z (height) widened +/-0.08 beyond that 2/3-shrink value (was -0.07..0.47,
   # z_len=0.54) because the tighter z crop occasionally clipped the tabletop
   # itself when the sensed table height jittered near z_min. z_len is now
   # 0.70, still well under x_len=0.86, so the render scale above is unchanged
   # — this only adds vertical margin, it does not re-zoom the image.
   # y (left/right) widened +/-0.04 (was -0.42..0.32, y_len=0.74 → now
   # -0.46..0.36, y_len=0.82): with flip_top_up=True the top-view image
   # vertical axis is world Y, so this loosens the top-view up/down crop.
   # y_len still < x_len → place_pc_in_cube scale (and thus inference zoom)
   # stays pinned to x_len; front-view horizontal also gains the same margin.

IMAGE_SIZE = 224


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_real(replay_sample, cameras):
    """Dict-style preprocessing. Mirrors gembench_utils._preprocess_inputs_gembench.

    Expects replay_sample[n]["rgb"] shape (b, 3, H, W) uint8 and
    replay_sample[n]["pcd"] shape (b, 3, H, W) float32 in base frame.
    """
    obs, pcds = [], []
    for n in cameras:
        rgb = replay_sample[n]["rgb"]
        pcd = replay_sample[n]["pcd"]
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds
