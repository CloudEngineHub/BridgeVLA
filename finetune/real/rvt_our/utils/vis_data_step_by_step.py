"""
Step through collected data, visualising the point cloud and the coordinate frames.
========================================

Following ``vis_pcd_with_end_pred`` in ``eval_client.py``, each step overlays:


  * the arm base frame (world / base frame, at the origin)
  * the gripper's *current* pose + open/close state (this frame's TCP / claw)
  * the gripper's *next-step (GT)* pose + open/close state (next frame's TCP / claw, i.e. the training target)
  * a motion arrow from current to next

onto that frame's point cloud (already transformed from camera frame into the base frame).

The gripper convention matches training: ``0=OPEN``, ``1=CLOSE``.

Three on-disk formats are supported:

1) Raw collection format — full point cloud (older data_collection):
  * 3rd_cam_pcd/{i}.pkl : (H, W, 3) float32 point cloud in the **camera frame**
  * 3rd_cam_rgb/{i}.pkl : (H, W, 3) uint8 RGB
  * actions/{i}.pkl     : (8,) -> [x, y, z, qw, qx, qy, qz, gripper] in the **base frame**, metres,
                          quaternion in wxyz order
  * extrinsic_matrix.npy: (4, 4) camera-frame -> base-frame extrinsics
  * instruction.txt

2) Raw collection format — depth deprojection (current data_collection_main_single_display_cycle.py):
  * 3rd_cam_depth/{i}.npy : (H, W) uint16 Z depth (metres x depth_scale)
  * 3rd_cam_rgb/{i}.pkl   : (H, W, 3) uint8 RGB
  * intrinsic.pkl         : {fx,fy,cx,cy,depth_scale,...}, used for pinhole deprojection
  * actions/{i}.pkl / extrinsic_matrix.npy / instruction.txt as above

3) Converted dobot format (organize_dataset.py / real_dataset.py):
  * zed_pcd/{i}.pkl / zed_rgb/{i}.pkl
  * pose.pkl            : the whole trajectory as a string (mm + RxRyRz deg + claw)
  * extrinsic_matrix.pkl
  * instruction.pkl

Interaction, inside the Open3D window:
  * N / Space / Right arrow -> next step
  * P / Left arrow          -> previous step
  * Q / Esc                 -> quit

Usage::

    python vis_data_step_by_step.py \
        --episode-dir /abs/path/to/your/real_data/<task_slug>_0

The episode directory can also come from the environment: export REAL_EPISODE_DIR=/abs/path/to/...
"""

import argparse
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# Defaults
# A single episode directory. There is no sensible default — point it at your own absolute path
#     export REAL_EPISODE_DIR=/abs/path/to/your/real_data/<task_slug>_0
# (or override with --episode-dir); with neither given it exits with a hint.
DEFAULT_EPISODE_DIR = os.environ.get("REAL_EPISODE_DIR", "")
# Point-cloud crop range (base frame, metres): x_min, y_min, z_min, x_max, y_max, z_max
DEFAULT_BOUNDS = [-1.3, -1.5, -0.1, 0.4, 0.7, 0.6]

FORMAT_RAW = "raw"
FORMAT_RAW_DEPTH = "raw_depth"
FORMAT_DOBOT = "dobot"
# Default depth quantisation scale, matching the collection script and real_dataset
DEFAULT_DEPTH_SCALE = 4000.0


# IO helpers
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def sorted_numeric_files(
    folder: str, suffix: Optional[str] = ".pkl", suffixes: Optional[Tuple[str, ...]] = None
) -> List[str]:
    """Sort by the number in the filename. ``suffixes`` takes priority over ``suffix``, so .npy/.png can both match."""
    if suffixes is None:
        suffixes = (suffix,) if suffix is not None else (".pkl",)
    files = [f for f in os.listdir(folder) if f.endswith(suffixes)]
    return sorted(
        files,
        key=lambda x: int(os.path.splitext(x)[0])
        if os.path.splitext(x)[0].isdigit()
        else x,
    )


def detect_format(episode_dir: str) -> str:
    """Auto-detect which of the raw / raw_depth / dobot layouts this is."""
    has_dobot = (
        os.path.isdir(os.path.join(episode_dir, "zed_pcd"))
        and os.path.isfile(os.path.join(episode_dir, "pose.pkl"))
    )
    has_raw_pcd = (
        os.path.isdir(os.path.join(episode_dir, "3rd_cam_pcd"))
        and os.path.isdir(os.path.join(episode_dir, "actions"))
    )
    has_raw_depth = (
        os.path.isdir(os.path.join(episode_dir, "3rd_cam_depth"))
        and os.path.isdir(os.path.join(episode_dir, "actions"))
        and os.path.isfile(os.path.join(episode_dir, "intrinsic.pkl"))
    )
    if has_dobot:
        return FORMAT_DOBOT
    # When both pcd and depth are present, prefer the existing point cloud (the older layout)
    if has_raw_pcd:
        return FORMAT_RAW
    if has_raw_depth:
        return FORMAT_RAW_DEPTH
    raise FileNotFoundError(
        f"unrecognised data format: {episode_dir}\n"
        "expected (zed_pcd + pose.pkl) or (3rd_cam_pcd + actions) "
        "or (3rd_cam_depth + actions + intrinsic.pkl)"
    )


def _pcd_rgb_dirs(episode_dir: str, fmt: str) -> Tuple[str, str]:
    if fmt == FORMAT_DOBOT:
        return (
            os.path.join(episode_dir, "zed_pcd"),
            os.path.join(episode_dir, "zed_rgb"),
        )
    if fmt == FORMAT_RAW_DEPTH:
        return (
            os.path.join(episode_dir, "3rd_cam_depth"),
            os.path.join(episode_dir, "3rd_cam_rgb"),
        )
    return (
        os.path.join(episode_dir, "3rd_cam_pcd"),
        os.path.join(episode_dir, "3rd_cam_rgb"),
    )


def load_extrinsic(episode_dir: str, fmt: str) -> np.ndarray:
    if fmt == FORMAT_DOBOT:
        path = os.path.join(episode_dir, "extrinsic_matrix.pkl")
        return np.asarray(load_pickle(path), dtype=np.float64)
    path = os.path.join(episode_dir, "extrinsic_matrix.npy")
    return np.load(path).astype(np.float64)


def load_instruction(episode_dir: str, fmt: str) -> str:
    if fmt == FORMAT_DOBOT:
        path = os.path.join(episode_dir, "instruction.pkl")
        if os.path.isfile(path):
            return str(load_pickle(path)).strip()
    path = os.path.join(episode_dir, "instruction.txt")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return f.read().strip()
    # The raw format sometimes only has instruction.pkl
    pkl_path = os.path.join(episode_dir, "instruction.pkl")
    if os.path.isfile(pkl_path):
        return str(load_pickle(pkl_path)).strip()
    return ""


def load_intrinsic_meta(episode_dir: str) -> dict:
    """Read intrinsic.pkl (fx/fy/cx/cy + depth_scale) for depth deprojection."""
    path = os.path.join(episode_dir, "intrinsic.pkl")
    meta = load_pickle(path)
    if not isinstance(meta, dict):
        raise ValueError(f"intrinsic.pkl should be a dict, got {type(meta)}: {path}")
    for key in ("fx", "fy", "cx", "cy"):
        if key not in meta:
            raise KeyError(f"intrinsic.pkl is missing the field '{key}': {path}")
    return meta


def deproject_depth_to_pcd(
    depth_hw: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = DEFAULT_DEPTH_SCALE,
) -> np.ndarray:
    """uint16 / float depth map -> (H,W,3) point cloud in the camera frame.

    Uses the same pinhole model as real_dataset._load_pcd and the collection script's verify_intrinsic:
      X = (u - cx) / fx * Z,  Y = (v - cy) / fy * Z,  Z = depth / depth_scale
    """
    if depth_hw.ndim == 3:
        depth_hw = depth_hw[:, :, 0]
    h, w = depth_hw.shape
    if np.issubdtype(depth_hw.dtype, np.integer):
        z = depth_hw.astype(np.float64) / float(depth_scale)
    else:
        z = depth_hw.astype(np.float64)
        # Already-metric float depth is not rescaled; collection writes uint16, which takes the branch above
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    x = ((u - float(cx)) / float(fx)) * z
    y = ((v - float(cy)) / float(fy)) * z
    return np.stack([x, y, z], axis=-1)


def parse_pose_pkl_to_actions(pose_pkl_path: str) -> List[np.ndarray]:
    """Parse dobot's pose.pkl into the same array layout as the raw actions.

    Returns per frame: [x, y, z, qw, qx, qy, qz, gripper] in metres, quaternion wxyz.
    Matches real_dataset._parse_pose_string plus the mm->m and euler->quat conversions.
    """
    data_str = load_pickle(pose_pkl_path)
    lines = str(data_str).strip().split("\n")
    actions: List[np.ndarray] = []
    for i, line in enumerate(lines):
        if i == 0:
            continue  # header
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        xyz_m = np.asarray([float(x) for x in parts[1:4]], dtype=np.float64) / 1000.0
        rpy_deg = [float(x) for x in parts[4:7]]
        # scipy's as_quat gives (qx, qy, qz, qw); this visualiser uses wxyz throughout
        qx, qy, qz, qw = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        if len(parts) >= 8:
            claw = float(parts[7])
        else:
            claw = 1.0 if (i in (1, 2, 5)) else 0.0
        actions.append(
            np.asarray([xyz_m[0], xyz_m[1], xyz_m[2], qw, qx, qy, qz, claw], dtype=np.float64)
        )
    return actions


def _load_depth_array(path: str) -> np.ndarray:
    """Read a depth file: the .npy written by collection, or the .png produced by organize."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path))
    if ext in (".png", ".jpg", ".jpeg"):
        # Imported lazily so a headless environment doesn't hard-depend on imageio/cv2
        try:
            import imageio.v2 as imageio
            return np.asarray(imageio.imread(path))
        except Exception:
            from PIL import Image
            return np.asarray(Image.open(path))
    if ext in (".pkl", ".pickle"):
        return np.asarray(load_pickle(path))
    raise ValueError(f"unsupported depth file type: {path}")


def load_frame(
    episode_dir: str,
    idx: int,
    fmt: str,
    intrinsic_meta: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read frame idx's camera-frame point cloud and RGB.

    The ``raw_depth`` format deprojects depth into an (H,W,3) point cloud using the intrinsics.
    """
    geom_dir, rgb_dir = _pcd_rgb_dirs(episode_dir, fmt)
    rgb_files = sorted_numeric_files(rgb_dir)

    if fmt == FORMAT_RAW_DEPTH:
        depth_files = sorted_numeric_files(
            geom_dir, suffixes=(".npy", ".png", ".pkl", ".pickle")
        )
        frame_name = depth_files[idx]
        depth = _load_depth_array(os.path.join(geom_dir, frame_name))
        if intrinsic_meta is None:
            intrinsic_meta = load_intrinsic_meta(episode_dir)
        # Intrinsics may be per frame (convert_pcd_to_depth can write an array of length N);
        # the collection script shares one set for the whole episode, giving length 1.
        i = min(idx, len(np.asarray(intrinsic_meta["fx"]).reshape(-1)) - 1)
        fx = float(np.asarray(intrinsic_meta["fx"]).reshape(-1)[i])
        fy = float(np.asarray(intrinsic_meta["fy"]).reshape(-1)[i])
        cx = float(np.asarray(intrinsic_meta["cx"]).reshape(-1)[i])
        cy = float(np.asarray(intrinsic_meta["cy"]).reshape(-1)[i])
        depth_scale = float(intrinsic_meta.get("depth_scale", DEFAULT_DEPTH_SCALE))
        pcd = deproject_depth_to_pcd(depth, fx, fy, cx, cy, depth_scale)
    else:
        pcd_files = sorted_numeric_files(geom_dir)
        frame_name = pcd_files[idx]
        pcd = np.asarray(load_pickle(os.path.join(geom_dir, frame_name)))[:, :, :3]

    rgb = np.asarray(load_pickle(os.path.join(rgb_dir, rgb_files[idx])))[:, :, :3]
    return pcd.astype(np.float64), rgb, frame_name


def load_action(
    episode_dir: str,
    idx: int,
    fmt: str,
    cached_dobot_actions: Optional[List[np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """Read frame idx's action: [x, y, z, qw, qx, qy, qz, gripper]. Returns None when out of range."""
    if fmt == FORMAT_DOBOT:
        actions = cached_dobot_actions
        if actions is None:
            actions = parse_pose_pkl_to_actions(os.path.join(episode_dir, "pose.pkl"))
        if idx < 0 or idx >= len(actions):
            return None
        return actions[idx]

    actions_dir = os.path.join(episode_dir, "actions")
    files = sorted_numeric_files(actions_dir)
    if idx < 0 or idx >= len(files):
        return None
    return np.asarray(
        load_pickle(os.path.join(actions_dir, files[idx])), dtype=np.float64
    ).reshape(-1)


def count_frames(episode_dir: str, fmt: str) -> int:
    geom_dir, _ = _pcd_rgb_dirs(episode_dir, fmt)
    if fmt == FORMAT_RAW_DEPTH:
        return len(
            sorted_numeric_files(geom_dir, suffixes=(".npy", ".png", ".pkl", ".pickle"))
        )
    return len(sorted_numeric_files(geom_dir))

# Geometry / frame-transform helpers
def convert_pcd_to_base(pcd_cam_hw3: np.ndarray, extrinsic_4x4: np.ndarray) -> np.ndarray:
    """Point cloud: camera frame -> base frame."""
    h, w, _ = pcd_cam_hw3.shape
    pts = pcd_cam_hw3.reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    pts_base = (extrinsic_4x4 @ pts_h.T).T[:, :3]
    return pts_base.reshape(h, w, 3)


def action_to_transform(action: np.ndarray) -> np.ndarray:
    """action [x,y,z,qw,qx,qy,qz,...] -> a 4x4 homogeneous transform (the gripper pose in the base frame)."""
    T = np.eye(4)
    T[:3, 3] = action[:3]
    qw, qx, qy, qz = action[3], action[4], action[5], action[6]
    # scipy expects xyzw order
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


def make_point_cloud(
    pcd_base_hw3: np.ndarray,
    rgb_hw3: np.ndarray,
    stride: int,
    bounds: Optional[List[float]],
    min_norm: float = 1e-6,
    max_points: int = 300000,
) -> o3d.geometry.PointCloud:
    pts = pcd_base_hw3[::stride, ::stride].reshape(-1, 3)
    rgb = rgb_hw3[::stride, ::stride].reshape(-1, 3).astype(np.float64) / 255.0

    mask = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > min_norm)
    if bounds is not None:
        x0, y0, z0, x1, y1, z1 = bounds
        mask &= (
            (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
            & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            & (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
        )
    pts, rgb = pts[mask], rgb[mask]

    if max_points > 0 and pts.shape[0] > max_points:
        sel = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts, rgb = pts[sel], rgb[sel]

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pts)
    geom.colors = o3d.utility.Vector3dVector(rgb)
    return geom


def make_sphere(center, radius: float, color) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    s.translate(np.asarray(center, dtype=np.float64))
    return s


def claw_is_closed(claw: Optional[float]) -> Optional[bool]:
    """Training convention: 0=OPEN, 1=CLOSE. ``None`` means unknown."""
    if claw is None:
        return None
    return float(claw) > 0.5


def claw_state_str(claw: Optional[float]) -> str:
    closed = claw_is_closed(claw)
    if closed is None:
        return "UNKNOWN"
    return "CLOSE" if closed else "OPEN"


def make_gripper_marker(
    T: np.ndarray,
    claw: Optional[float],
    finger_length: float = 0.06,
    max_width: float = 0.05,
    color: Optional[List[float]] = None,
) -> o3d.geometry.LineSet:
    """Draw a simplified \"fork\" marker for the gripper.

    ``claw`` follows the training convention: 0=OPEN (fingertips apart), 1=CLOSE (fingertips together).
    Without ``color``, OPEN is green and CLOSE is red.
    """
    closed = claw_is_closed(claw)
    # openness in [0,1]: 1 -> wide open fingers, 0 -> closed
    if closed is None:
        openness = 0.5
    else:
        openness = 0.0 if closed else 1.0
    half_w = max(max_width / 2.0 * max(openness, 0.15), 0.004)
    pts_local = np.array(
        [
            [0, 0, 0],
            [0, 0, finger_length * 0.8],
            [0, -half_w, 0],
            [0, half_w, 0],
            [0, -half_w, -finger_length],
            [0, half_w, -finger_length],
        ],
        dtype=np.float64,
    )
    pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
    pts_w = (T @ pts_h.T).T[:, :3]
    lines = [[0, 1], [2, 3], [2, 4], [3, 5]]
    if color is None:
        if closed is None:
            color = [1.0, 0.5, 0.0]
        else:
            color = [0.85, 0.0, 0.0] if closed else [0.0, 0.85, 0.0]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_w)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls


def make_motion_arrow(
    p_from: np.ndarray,
    p_to: np.ndarray,
    color: Optional[List[float]] = None,
) -> o3d.geometry.LineSet:
    """Line from the current gripper centre to the predicted next centre."""
    if color is None:
        color = [1.0, 1.0, 0.0]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(
        np.asarray([p_from, p_to], dtype=np.float64)
    )
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


# Step-by-step visualiser
class StepViewer:
    def __init__(self, args):
        self.args = args
        self.episode_dir = os.path.abspath(args.episode_dir)
        self.bounds = None if args.no_bounds else args.bounds
        self.fmt = detect_format(self.episode_dir)
        self.extrinsic = load_extrinsic(self.episode_dir, self.fmt)
        if args.invert_extrinsic:
            self.extrinsic = np.linalg.inv(self.extrinsic)
        self.n_frames = count_frames(self.episode_dir, self.fmt)
        self.idx = max(0, min(args.frame, self.n_frames - 1))
        self.instruction = load_instruction(self.episode_dir, self.fmt)

        # raw_depth: cache the intrinsics so they are not re-read every frame
        self._intrinsic_meta: Optional[dict] = None
        if self.fmt == FORMAT_RAW_DEPTH:
            self._intrinsic_meta = load_intrinsic_meta(self.episode_dir)
            fx = float(np.asarray(self._intrinsic_meta["fx"]).reshape(-1)[0])
            fy = float(np.asarray(self._intrinsic_meta["fy"]).reshape(-1)[0])
            cx = float(np.asarray(self._intrinsic_meta["cx"]).reshape(-1)[0])
            cy = float(np.asarray(self._intrinsic_meta["cy"]).reshape(-1)[0])
            scale = float(self._intrinsic_meta.get("depth_scale", DEFAULT_DEPTH_SCALE))
            print(
                f"[intrinsic] fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} "
                f"depth_scale={scale}"
            )

        # The dobot format caches the whole trajectory in memory, avoiding a pose.pkl re-read per step
        self._dobot_actions: Optional[List[np.ndarray]] = None
        if self.fmt == FORMAT_DOBOT:
            self._dobot_actions = parse_pose_pkl_to_actions(
                os.path.join(self.episode_dir, "pose.pkl")
            )
            n_pose = len(self._dobot_actions)
            if n_pose != self.n_frames:
                print(
                    f"[warning] frame-count mismatch: pcd={self.n_frames}, pose.pkl={n_pose}; "
                    f"showing the gripper for whichever is shorter."
                )

        print(f"[format] {self.fmt}  |  frames={self.n_frames}  |  {self.episode_dir}")

    # -- Build every geometry for frame idx --------------------------------
    def build_geometries(self, idx: int) -> List:
        pcd_cam, rgb, frame_name = load_frame(
            self.episode_dir, idx, self.fmt, intrinsic_meta=self._intrinsic_meta
        )
        pcd_base = convert_pcd_to_base(pcd_cam, self.extrinsic)
        geoms = [
            make_point_cloud(
                pcd_base, rgb, self.args.stride, self.bounds,
                max_points=self.args.max_points,
            )
        ]

        # 1) Arm base frame (origin) — the largest axes
        geoms.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.args.base_frame_size, origin=[0, 0, 0]
            )
        )
        geoms.append(make_sphere([0, 0, 0], self.args.marker_radius, [0, 0, 0]))

        # 2) Gripper "current" — actions[idx] / pose[idx]
        #    cyan sphere + teal gripper fork + axes
        cur = load_action(
            self.episode_dir, idx, self.fmt, cached_dobot_actions=self._dobot_actions
        )
        cur_str = "N/A"
        T_cur = None
        if cur is not None:
            T_cur = action_to_transform(cur)
            f_cur = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.args.cur_frame_size
            )
            f_cur.transform(T_cur)
            geoms.append(f_cur)
            claw_cur = float(cur[7]) if cur.size > 7 else None
            geoms.append(
                make_sphere(T_cur[:3, 3], self.args.marker_radius * 1.2, [0.0, 0.9, 0.9])
            )
            geoms.append(
                make_gripper_marker(T_cur, claw_cur, color=[0.0, 0.85, 0.85])
            )
            cur_str = self._pose_str(T_cur, claw_cur)

        # 3) Gripper "next-step prediction (GT)" — actions[idx+1] / pose[idx+1]
        #    magenta sphere + magenta fork + larger axes (matching the training target)
        nxt = load_action(
            self.episode_dir, idx + 1, self.fmt, cached_dobot_actions=self._dobot_actions
        )
        nxt_str = "N/A (last frame, no prediction target)"
        T_nxt = None
        if nxt is not None:
            T_nxt = action_to_transform(nxt)
            f_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.args.next_frame_size
            )
            f_nxt.transform(T_nxt)
            geoms.append(f_nxt)
            claw_nxt = float(nxt[7]) if nxt.size > 7 else None
            geoms.append(
                make_sphere(T_nxt[:3, 3], self.args.marker_radius * 1.4, [1.0, 0.0, 1.0])
            )
            geoms.append(
                make_gripper_marker(T_nxt, claw_nxt, color=[1.0, 0.0, 1.0])
            )
            nxt_str = self._pose_str(T_nxt, claw_nxt)

        # 4) Motion arrow from current to next
        if T_cur is not None and T_nxt is not None:
            geoms.append(make_motion_arrow(T_cur[:3, 3], T_nxt[:3, 3]))

        self._print_step(idx, frame_name, cur_str, nxt_str)
        return geoms

    @staticmethod
    def _pose_str(T: np.ndarray, claw: Optional[float]) -> str:
        pos = T[:3, 3]
        rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        claw_val = "NA" if claw is None else f"{int(round(float(claw)))}"
        return (
            f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})m  "
            f"rpy=({rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f})deg  "
            f"claw={claw_val}({claw_state_str(claw)})"
        )

    def _print_step(self, idx, frame_name, cur_str, nxt_str):
        print("\n" + "=" * 70)
        print(f"STEP {idx} / {self.n_frames - 1}   (frame={frame_name})")
        print(f"Instruction: {self.instruction}")
        print(f"[base frame]      origin (red X, green Y, blue Z, size={self.args.base_frame_size})")
        print(f"[gripper current] cyan     {cur_str}")
        print(f"[next-step (GT)]  magenta  {nxt_str}")
        print("Legend: fork width = open/close (0 open / 1 close), yellow line = current -> next.  [N/Space/Right] next  [P/Left] prev  [Q/Esc] quit")
        print("=" * 70)

    # -- Run the interactive window ---------------------------------------
    def run(self):
        if self.n_frames == 0:
            print("[error] no point-cloud frame found.")
            return

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Step-by-step visualiser (N/Space = next)", width=1280, height=900)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = self.args.point_size

        state = {"first": True}

        def refresh():
            vis.clear_geometries()
            for g in self.build_geometries(self.idx):
                vis.add_geometry(g, reset_bounding_box=state["first"])
            state["first"] = False

        def go_next(_vis):
            if self.idx < self.n_frames - 1:
                self.idx += 1
                refresh()
            else:
                print("[info] already at the last frame.")
            return False

        def go_prev(_vis):
            if self.idx > 0:
                self.idx -= 1
                refresh()
            else:
                print("[info] already at the first frame.")
            return False

        def quit_vis(_vis):
            _vis.close()
            return False

        # N, Space, Right arrow -> next step
        vis.register_key_callback(ord("N"), go_next)
        vis.register_key_callback(ord(" "), go_next)
        vis.register_key_callback(262, go_next)  # GLFW right arrow
        # P, Left arrow -> previous step
        vis.register_key_callback(ord("P"), go_prev)
        vis.register_key_callback(263, go_prev)  # GLFW left arrow
        # Q, Esc -> quit
        vis.register_key_callback(ord("Q"), quit_vis)
        vis.register_key_callback(256, quit_vis)  # GLFW Esc

        refresh()
        vis.run()
        vis.destroy_window()
        print("[done] exiting the visualiser.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Step through collected data: point cloud + base frame + current/next gripper frames."
    )
    p.add_argument("--episode-dir", default=DEFAULT_EPISODE_DIR, help="a single episode data directory")
    p.add_argument("--frame", type=int, default=0, help="starting frame index")
    p.add_argument("--stride", type=int, default=4, help="point-cloud subsampling stride")
    p.add_argument("--max-points", type=int, default=300000, help="maximum points displayed")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--invert-extrinsic", action="store_true", help="invert the extrinsics")
    p.add_argument("--no-bounds", action="store_true", help="do not crop the point cloud")
    p.add_argument("--bounds", type=float, nargs=6, default=DEFAULT_BOUNDS,
                   metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                   help="point-cloud crop range (base frame, metres)")
    p.add_argument("--base-frame-size", type=float, default=0.12)
    p.add_argument("--cur-frame-size", type=float, default=0.12)
    p.add_argument("--next-frame-size", type=float, default=0.15)
    p.add_argument("--marker-radius", type=float, default=0.02)
    return p.parse_args()


def require_episode_dir(path: str) -> str:
    """Return ``path``, or abort with instructions if it is unset / missing."""
    if not path:
        raise SystemExit(
            "[config] no episode dir configured.\n"
            "         Set it to your own absolute path, either\n"
            "             export REAL_EPISODE_DIR=/abs/path/to/your/real_data/<task_slug>_0\n"
            "         or pass\n"
            "             --episode-dir /abs/path/to/your/real_data/<task_slug>_0"
        )
    if not os.path.isdir(path):
        raise SystemExit(f"[config] episode dir does not exist: {path}")
    return path


def main():
    args = parse_args()
    args.episode_dir = require_episode_dir(args.episode_dir)
    StepViewer(args).run()


if __name__ == "__main__":
    main()
