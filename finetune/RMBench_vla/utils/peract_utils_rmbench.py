# RMBench (dual-arm aloha-agilex) peract_utils.
#
# Mirrors GemBench / memoryBench utils so the dual-arm BridgeVLA agent
# (num_arms=2) can reuse the shared training step machinery. The substantive
# RMBench-specific bits live here:
#   * CAMERAS: the 4 cameras saved in data_self — 2 STATIC world cameras
#     (head, front) + the 2 per-arm WRIST cameras (left, right). Verified
#     against RMBench envs/camera/camera.py ("wrist camera" block, whose poses
#     are re-set every render via update_wrist_camera) and against the stored
#     extrinsics (head/front constant over an episode; left/right sweep ~0.36 m
#     with the arms):
#       head  : STATIC world cam @ (-0.032, -0.45, 1.35), LargeView (fovy 50°)
#       front : STATIC world cam @ ( 0.000, -0.45, 0.85), D435      (fovy 37°)
#       left  : WRIST cam on the LEFT arm,  LargeView (fovy 50°), moves per step
#       right : WRIST cam on the RIGHT arm, LargeView (fovy 50°), moves per step
#     All four are unprojected with their OWN per-frame intrinsic/extrinsic
#     (see depth_to_world_pcd), so the fused world point cloud is correct
#     regardless of the wrist motion; only the coverage varies per keyframe.
#     Same 4 cameras at train and at eval (deploy_policy.encode_obs).
#   * SCENE_BOUNDS: a deliberately GENEROUS crop (workspace measured empirically
#     ≈ x∈[-0.85,0.72] y∈[-0.36,0.40] z∈[0.51,1.04]); kept wide so nothing is
#     clipped during debugging. TIGHTEN later once eval is wired up.
#   * GRIPPER_OFFSET / TCP math: the GT heatmap target is the gripper fingertip
#     center, recovered from the stored EE-link endpose (offset 0.12 m along
#     the gripper local x-axis). See RMBench
#     script/extract_key_frames/visualize_keyframes_multiview.py.
#   * depth_to_world_pcd: unproject a (H,W) depth image (millimeters) + OpenCV
#     intrinsic/extrinsic into a world-frame (H,W,3) point cloud. Verified to
#     round-trip the stored TCP to within a few cm.
#   * world_to_render_pos / render_to_world_pos: a FIXED +90° rotation about
#     the gravity (Z) axis that maps the RMBench world frame into the
#     orientation the shared tri-view renderer expects (see below).

import numpy as np

CAMERAS = ["head", "front", "left", "right"]

# World -> render-frame axis fix (RMBench-specific).
#
# The shared point renderer (bridgevla/libs/point-renderer) hard-codes the
# RVT/RLBench world convention for its three fixed views:
#     top   : eye=[0,0, 1]  (looks down  -Z); RMBench sets rotate_top_ccw90
#             so up goes [0,1,0] -> [-1,0,0] (top image rolled CW 90°).
#     front : eye=[1,0, 0]  (looks along  +X)   <- assumes +X is depth (front)
#     right : eye=[0,1, 0]  (looks along  +Y)   <- assumes +Y is left/right
# RMBench's world frame, however, is (measured from real episode data):
#     +X = LEFT/RIGHT  (left arm x<0, right arm x>0)
#     +Y = FRONT/BACK depth (all 4 cameras sit at y~=-0.45 looking toward +Y)
#     +Z = UP
# i.e. RMBench's X and Y are swapped vs. what the renderer expects, so the
# renderer's "front" view actually looks down the left/right axis (a side
# view) and its "right" view looks down the depth axis from the far/back side.
#
# Fix: rotate every POSITION by +90 deg about Z before it reaches the renderer
#     (x, y, z) -> (-y, x, z)
# so RMBench depth (Y) becomes render-X (front) and RMBench left/right (X)
# becomes render-Y (side). The inverse (render -> world) is (x, y, z) ->
# (y, -x, z), applied to the predicted waypoint at eval.
#
# ORIENTATION (quaternions) is deliberately LEFT IN THE WORLD FRAME. The SE3
# training augmentation is yaw-only (rpy=[0,0,45]) about the SAME Z axis, so
# the static Rz90 and the random Rz_aug commute: the offset between the
# render-frame point cloud and the world-frame orientation label is a constant
# Rz90 for every sample, which the rotation head learns once. Keeping quats in
# world frame means the predicted quaternion is already execution-ready (no
# inverse needed) and the verified take_action('ee') path is untouched.
R_WORLD_TO_RENDER = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def world_to_render_pos(p):
    """(..., 3) world XYZ -> render-frame XYZ via +90 deg about Z: (x,y,z)->(-y,x,z)."""
    p = np.asarray(p, dtype=np.float32)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([-y, x, z], axis=-1).astype(np.float32)


def render_to_world_pos(p):
    """(..., 3) render-frame XYZ -> world XYZ via -90 deg about Z: (x,y,z)->(y,-x,z)."""
    p = np.asarray(p, dtype=np.float32)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([y, -x, z], axis=-1).astype(np.float32)


def _bounds_world_to_render(world_bounds):
    """Rotate a [xmin,ymin,zmin, xmax,ymax,zmax] AABB into the render frame.

    Under (x,y,z)->(-y,x,z): render_x = -world_y, render_y = world_x,
    render_z = world_z. Recompute per-axis min/max from the rotated corners.
    """
    lo = np.array(world_bounds[:3], dtype=np.float64)
    hi = np.array(world_bounds[3:], dtype=np.float64)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    r = corners @ R_WORLD_TO_RENDER.T
    return [float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 2].min()),
            float(r[:, 0].max()), float(r[:, 1].max()), float(r[:, 2].max())]


# HDF5 camera-group names (data_self uses the *_camera suffix).
CAMERA_HDF5_NAMES = {
    "head": "head_camera",
    "front": "front_camera",
    "left": "left_camera",
    "right": "right_camera",
}

# Generous workspace crop [x_min, y_min, z_min, x_max, y_max, z_max] (meters).
# Wider than the measured extent on purpose — re-tune when calibrating eval.
#
# Defined in the WORLD frame (measured workspace ~ x in [-0.85,0.72] (L/R),
# y in [-0.36,0.40] (depth), z in [0.51,1.04] (up)). Everything downstream
# (move_pc_in_bound / place_pc_in_cube) operates on RENDER-frame points, so
# the effective crop SCENE_BOUNDS is the world AABB rotated +90 deg about Z.
SCENE_BOUNDS_WORLD = [
    -0.95, -0.55, 0.45,
    0.95, 0.55, 1.20,
]
# Render-frame crop, derived once: (x,y,z)->(-y,x,z) maps the world AABB to
# render_x in [-ymax,-ymin], render_y in [xmin,xmax], render_z unchanged.
# = [-0.55, -0.95, 0.45, 0.55, 0.95, 1.20].
SCENE_BOUNDS = _bounds_world_to_render(SCENE_BOUNDS_WORLD)

# Tri-view renderer outputs at 224 (mvt img_size). The training input image
# resolution (RGB + projected PCD grid) — LargeView is 320x240, so we resize
# to a square IMAGE_SIZE. 224 keeps the per-camera point density high enough
# for the splat renderer while matching the LargeView aspect closely.
IMAGE_SIZE = 224

VOXEL_SIZES = [100]
LOW_DIM_SIZE = 4
ROTATION_RESOLUTION = 5  # degrees per bin -> 72 bins

# EE-link -> gripper-fingertip-center offset along the gripper local x-axis.
# aloha-agilex: gripper_bias == 0.12, so the engine's TCP-frame ee control
# cancels this exactly; the model predicts the TCP center directly.
GRIPPER_OFFSET = 0.12

# 10 RMBench tasks that have extracted keyframe_depth data + instructions.
RMBENCH_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
]


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs_rmbench(replay_sample, cameras):
    """Mirror of ``_preprocess_inputs_gembench``: collect [rgb, pcd] per camera.

    The dual-arm agent reuses ``gembench_utils._preprocess_inputs_gembench``
    directly (the structure is identical), so this exists mainly for parity
    / standalone use.
    """
    obs, pcds = [], []
    for n in cameras:
        rgb = _norm_rgb(replay_sample[n]["rgb"])
        pcd = replay_sample[n]["pcd"]
        obs.append([rgb, pcd])
        pcds.append(pcd)
    return obs, pcds


def endpose_to_tcp_center(endpose_7d):
    """Stored EE-link endpose [x,y,z, qw,qx,qy,qz] (wxyz) -> TCP center xyz.

    TCP = pos + R(quat) @ [GRIPPER_OFFSET, 0, 0]. Rotation is unchanged (the
    TCP shares the EE-link orientation). Matches RMBench's
    ``_endpose_to_gripper_center`` and ``robot.get_left_tcp_pose``.

    Returns (tcp_xyz (3,), quat_xyzw (4,)).
    """
    from scipy.spatial.transform import Rotation

    pos = np.asarray(endpose_7d[:3], dtype=np.float64)
    quat_wxyz = np.asarray(endpose_7d[3:], dtype=np.float64)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    tcp = pos + R @ np.array([GRIPPER_OFFSET, 0.0, 0.0])
    return tcp.astype(np.float32), quat_xyzw.astype(np.float32)


def tcp_center_to_endpose_pos(tcp_xyz, quat_xyzw):
    """Inverse of ``endpose_to_tcp_center`` position: TCP center -> EE-link pos.

    Only needed if a consumer wants the raw EE-link origin. For RMBench
    execution this is NOT used: aloha-agilex gripper_bias == 0.12, so
    ``take_action(action_type='ee')`` already maps the TCP-frame pose back to
    the EE link internally (the 0.12 - gripper_bias offset cancels). The
    deploy policy feeds the predicted TCP pose straight through.
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    return (np.asarray(tcp_xyz, dtype=np.float64)
            - R @ np.array([GRIPPER_OFFSET, 0.0, 0.0])).astype(np.float32)


def depth_to_world_pcd(depth_mm, intrinsic_cv, extrinsic_cv):
    """Unproject an OpenCV depth image to a world-frame point cloud.

    :param depth_mm: (H, W) depth in MILLIMETERS (RMBench stores depth*1000).
    :param intrinsic_cv: (3, 3) pinhole intrinsics.
    :param extrinsic_cv: (3, 4) world->camera [R | t] (OpenCV convention).
    :returns: (H, W, 3) float32 world-frame XYZ. Invalid (<=0) depth pixels
        map to (0, 0, 0) — they are culled later by move_pc_in_bound (the
        scene bounds exclude the origin).
    """
    depth = np.asarray(depth_mm, dtype=np.float64) / 1000.0  # -> meters
    K = np.asarray(intrinsic_cv, dtype=np.float64)
    E = np.asarray(extrinsic_cv, dtype=np.float64)
    H, W = depth.shape

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pc_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    R = E[:, :3]
    t = E[:, 3]
    pc_world = (pc_cam.reshape(-1, 3) - t) @ R
    pc_world = pc_world.reshape(H, W, 3)

    invalid = z <= 0
    pc_world[invalid] = 0.0
    return pc_world.astype(np.float32)
