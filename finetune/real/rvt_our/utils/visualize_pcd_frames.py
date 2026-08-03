import argparse
import json
import os
import pickle
from typing import Iterable, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# Real-robot data root. There is no sensible default — point it at your own absolute path
#     export REAL_COLLECT_DATA_ROOT=/abs/path/to/your/real_data
# (or override with --data-root / --episode-dir); with neither given it exits with a hint.
DEFAULT_DATA_ROOT = os.environ.get("REAL_COLLECT_DATA_ROOT", "")
DEFAULT_BOUNDS = [-1.3, -1.5, -0.2, 0.8, 1.0, 1.0]


def require_path(path: str, what: str, cli_flag: str,
                 env_var: str = "REAL_COLLECT_DATA_ROOT") -> str:
    """Return ``path``, or abort with instructions if it is unset / missing."""
    if not path:
        raise SystemExit(
            f"[config] no {what} configured.\n"
            f"         Set it to your own absolute path, either\n"
            f"             export {env_var}=/abs/path/to/your/real_data\n"
            f"         or pass\n"
            f"             {cli_flag} /abs/path/to/your/real_data"
        )
    if not os.path.exists(path):
        raise SystemExit(
            f"[config] {what} does not exist: {path}\n"
            f"         point {env_var} / {cli_flag} at your own data"
        )
    return path


def sorted_numeric_files(folder: str, suffix: str = ".pkl"):
    files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    return sorted(files, key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)


def find_episode_dirs(data_root: str):
    episodes = []
    for current, dirs, files in os.walk(data_root):
        if "extrinsic_matrix.npy" in files and "3rd_cam_pcd" in dirs:
            episodes.append(current)
    return sorted(episodes)


def resolve_episode_dir(args):
    if args.episode_dir:
        episode_dir = os.path.abspath(args.episode_dir)
        if not os.path.isdir(episode_dir):
            raise FileNotFoundError(f"episode_dir not found: {episode_dir}")
        return episode_dir

    require_path(args.data_root, "real-robot data root", "--data-root")
    episodes = find_episode_dirs(args.data_root)
    if args.task_filter:
        episodes = [p for p in episodes if args.task_filter in p]
    if not episodes:
        raise FileNotFoundError(f"No episode found under {args.data_root}")
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(f"episode_index {args.episode_index} out of range, total={len(episodes)}")
    print(f"[episodes] found {len(episodes)} episode(s)")
    for i, path in enumerate(episodes[: min(len(episodes), args.list_first)]):
        print(f"  [{i}] {path}")
    return episodes[args.episode_index]


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frame(episode_dir: str, frame_idx: int):
    pcd_dir = os.path.join(episode_dir, "3rd_cam_pcd")
    rgb_dir = os.path.join(episode_dir, "3rd_cam_rgb")
    pcd_files = sorted_numeric_files(pcd_dir)
    rgb_files = sorted_numeric_files(rgb_dir)
    if frame_idx < 0 or frame_idx >= len(pcd_files):
        raise IndexError(f"frame {frame_idx} out of range, pcd frames={len(pcd_files)}")
    pcd = np.asarray(load_pickle(os.path.join(pcd_dir, pcd_files[frame_idx])))[:, :, :3].astype(np.float64)
    rgb = np.asarray(load_pickle(os.path.join(rgb_dir, rgb_files[frame_idx])))[:, :, :3]
    return pcd, rgb, pcd_files[frame_idx]


def convert_pcd_to_base(pcd_camera_hw3: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    h, w, _ = pcd_camera_hw3.shape
    pc = pcd_camera_hw3.reshape(-1, 3)
    pc_h = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
    pc_base = (transform_4x4 @ pc_h.T).T[:, :3]
    return pc_base.reshape(h, w, 3)


def parse_bounds(bounds_text: str):
    if bounds_text.lower() in ("none", "off", "false", "0"):
        return None
    values = [float(x.strip()) for x in bounds_text.split(",") if x.strip()]
    if len(values) != 6:
        raise ValueError("bounds must be six comma-separated numbers: x_min,y_min,z_min,x_max,y_max,z_max")
    return values


def finite_point_mask(points: np.ndarray, min_norm: float):
    return np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > min_norm)


def bounds_mask(points: np.ndarray, bounds: Optional[Iterable[float]]):
    if bounds is None:
        return np.ones(points.shape[0], dtype=bool)
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )


def make_point_cloud(pcd_hw3: np.ndarray, rgb_hw3: np.ndarray, stride: int, bounds, min_norm: float, max_points: int):
    pcd = pcd_hw3[::stride, ::stride].reshape(-1, 3)
    rgb = rgb_hw3[::stride, ::stride].reshape(-1, 3).astype(np.float64) / 255.0
    mask_valid = finite_point_mask(pcd, min_norm)
    mask_bounds = bounds_mask(pcd, bounds)
    mask = mask_valid & mask_bounds
    pcd = pcd[mask]
    rgb = rgb[mask]
    if max_points > 0 and pcd.shape[0] > max_points:
        idx = np.linspace(0, pcd.shape[0] - 1, max_points).astype(np.int64)
        pcd = pcd[idx]
        rgb = rgb[idx]
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pcd)
    geom.colors = o3d.utility.Vector3dVector(rgb)
    return geom, mask_valid, mask_bounds


def make_sphere(center, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(np.asarray(center, dtype=np.float64))
    return sphere


def make_bounds_box(bounds):
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=[x_min, y_min, z_min], max_bound=[x_max, y_max, z_max])
    box.color = (1.0, 1.0, 0.0)
    return box


def make_grid(size: float, step: float, z: float):
    values = np.arange(-size, size + 1e-9, step)
    points = []
    lines = []
    colors = []
    for v in values:
        base = len(points)
        points.extend([[-size, v, z], [size, v, z], [v, -size, z], [v, size, z]])
        lines.extend([[base, base + 1], [base + 2, base + 3]])
        colors.extend([[0.45, 0.45, 0.45], [0.45, 0.45, 0.45]])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid


def make_camera_frustum(K: np.ndarray, width: int, height: int, scale: float, transform: np.ndarray):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_px = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    corners = []
    for u, v in corners_px:
        corners.append([(u - cx) / fx * scale, (v - cy) / fy * scale, scale])
    points = np.vstack([[0.0, 0.0, 0.0], np.asarray(corners, dtype=np.float64)])
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_base = (transform @ points_h.T).T[:, :3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_base)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.7, 0.0] for _ in lines])
    return line_set


def load_intrinsic(path: Optional[str]):
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".pkl", ".pickle"):
        arr = load_pickle(path)
    elif ext == ".json":
        with open(path, "r") as f:
            arr = json.load(f)
    else:
        arr = np.loadtxt(path)
    if isinstance(arr, dict):
        for key in ("K", "intrinsic", "intrinsics", "camera_matrix", "intrinsic_matrix"):
            if key in arr:
                arr = arr[key]
                break
    K = np.asarray(arr, dtype=np.float64)
    if K.shape[0] >= 3 and K.shape[1] >= 3:
        return K[:3, :3]
    raise ValueError(f"Unsupported intrinsic shape from {path}: {K.shape}")


def load_depth(episode_dir: str, frame_idx: int):
    for dirname in ("3rd_cam_depth", "zed_depth"):
        depth_dir = os.path.join(episode_dir, dirname)
        if os.path.isdir(depth_dir):
            files = sorted_numeric_files(depth_dir)
            if frame_idx < len(files):
                return np.asarray(load_pickle(os.path.join(depth_dir, files[frame_idx]))).astype(np.float64)
    return None


def pcd_from_depth(depth: np.ndarray, K: np.ndarray):
    h, w = depth.shape[:2]
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    ys, xs = np.indices((h, w), dtype=np.float64)
    z = depth.astype(np.float64)
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def compare_depth_intrinsic_pcd(stored_pcd: np.ndarray, depth: Optional[np.ndarray], K: Optional[np.ndarray]):
    if depth is None or K is None:
        return None
    reconstructed = pcd_from_depth(depth, K)
    if reconstructed.shape != stored_pcd.shape:
        return None
    diff = np.linalg.norm(reconstructed.reshape(-1, 3) - stored_pcd.reshape(-1, 3), axis=1)
    valid = np.isfinite(diff) & (np.linalg.norm(stored_pcd.reshape(-1, 3), axis=1) > 1e-8)
    if not valid.any():
        return None
    return {
        "mean_m": float(np.mean(diff[valid])),
        "median_m": float(np.median(diff[valid])),
        "p95_m": float(np.percentile(diff[valid], 95)),
        "max_m": float(np.max(diff[valid])),
    }


def action_to_transform(action, quat_order: str):
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.size < 7:
        return None
    transform = np.eye(4)
    transform[:3, 3] = arr[:3]
    quat = arr[3:7]
    if quat_order == "wxyz":
        quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    transform[:3, :3] = R.from_quat(quat).as_matrix()
    return transform


def load_action_frame(episode_dir: str, frame_idx: int, quat_order: str):
    actions_dir = os.path.join(episode_dir, "actions")
    if not os.path.isdir(actions_dir):
        return None
    files = sorted_numeric_files(actions_dir)
    if frame_idx >= len(files):
        return None
    return action_to_transform(load_pickle(os.path.join(actions_dir, files[frame_idx])), quat_order)


def load_gripper_pose(episode_dir: str, frame_idx: int, quat_order: str):
    actions_dir = os.path.join(episode_dir, "actions")
    if not os.path.isdir(actions_dir):
        return None, None, None
    files = sorted_numeric_files(actions_dir)
    if frame_idx >= len(files):
        return None, None, None
    raw = np.asarray(load_pickle(os.path.join(actions_dir, files[frame_idx])), dtype=np.float64).reshape(-1)
    transform = action_to_transform(raw, quat_order)
    if transform is None:
        return None, None, raw
    gripper_open = float(raw[7]) if raw.size > 7 else None
    return transform, gripper_open, raw


def make_gripper_marker(transform_4x4: np.ndarray, gripper_open: Optional[float],
                        finger_length: float = 0.06, max_width: float = 0.05):
    if gripper_open is not None:
        half_w = max(max_width / 2.0 * gripper_open, 0.004)
    else:
        half_w = max_width / 4.0
    pts_local = np.array([
        [0,  0,           0],
        [0,  0,           finger_length * 0.8],
        [0, -half_w,      0],
        [0,  half_w,      0],
        [0, -half_w,     -finger_length],
        [0,  half_w,     -finger_length],
    ], dtype=np.float64)
    pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
    pts_w = (transform_4x4 @ pts_h.T).T[:, :3]
    lines = [[0, 1], [2, 3], [2, 4], [3, 5]]
    if gripper_open is not None:
        color = [0.0, 0.85, 0.0] if gripper_open > 0.5 else [0.85, 0.0, 0.0]
    else:
        color = [1.0, 0.5, 0.0]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_w)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    finger_tip_spheres = []
    for tip in [pts_w[4], pts_w[5]]:
        finger_tip_spheres.append(make_sphere(tip, 0.008, color))
    return ls, finger_tip_spheres


def print_stats(episode_dir, frame_name, extrinsic, transform_used, pcd_camera, pcd_base, bounds, mask_valid, mask_bounds, K, intrinsic_check,
                gripper_T=None, gripper_open=None, gripper_raw=None):
    flat = pcd_base.reshape(-1, 3)
    valid = finite_point_mask(flat, 1e-8)
    print(f"[episode] {episode_dir}")
    print(f"[frame] {frame_name}")
    print(f"[pcd_camera] shape={pcd_camera.shape} dtype={pcd_camera.dtype}")
    print("[extrinsic_matrix.npy]")
    print(extrinsic)
    print(f"[extrinsic] det(R)={np.linalg.det(extrinsic[:3, :3]):.8f} orth_err={np.linalg.norm(extrinsic[:3, :3].T @ extrinsic[:3, :3] - np.eye(3)):.8e}")
    print("[transform_used_camera_to_base]")
    print(transform_used)
    print(f"[camera_origin_in_base] {transform_used[:3, 3].tolist()}")
    if valid.any():
        pts = flat[valid]
        print(f"[pcd_base] min={pts.min(axis=0).tolist()} max={pts.max(axis=0).tolist()} mean={pts.mean(axis=0).tolist()}")
    print(f"[valid_ratio_before_bounds] {float(mask_valid.mean()):.6f}")
    if bounds is not None:
        print(f"[bounds] {bounds}")
        print(f"[inside_bounds_ratio_before_valid_filter] {float(mask_bounds.mean()):.6f}")
    if K is not None:
        print("[intrinsic]")
        print(K)
    if intrinsic_check is not None:
        print(f"[depth_intrinsic_vs_saved_pcd] {intrinsic_check}")
    if gripper_T is not None:
        pos = gripper_T[:3, 3]
        rpy = R.from_matrix(gripper_T[:3, :3]).as_euler("xyz", degrees=True)
        state_str = "OPEN" if (gripper_open is not None and gripper_open > 0.5) else "CLOSED"
        print(f"[gripper] pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  rpy=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) deg  state={state_str} (val={gripper_open})")
        if gripper_raw is not None:
            print(f"[gripper_raw] {gripper_raw.tolist()}")


def build_visualization(args):
    episode_dir = resolve_episode_dir(args)
    frame_idx = args.frame
    pcd_camera, rgb, frame_name = load_frame(episode_dir, frame_idx)
    extrinsic = np.load(os.path.join(episode_dir, "extrinsic_matrix.npy")).astype(np.float64)
    transform_used = np.linalg.inv(extrinsic) if args.invert_extrinsic else extrinsic
    K = load_intrinsic(args.intrinsic)
    depth = load_depth(episode_dir, frame_idx)
    intrinsic_check = compare_depth_intrinsic_pcd(pcd_camera, depth, K)
    if args.use_depth_pinhole:
        if K is None:
            raise ValueError("--use-depth-pinhole requires --intrinsic")
        if depth is None:
            raise FileNotFoundError("No depth file found in 3rd_cam_depth or zed_depth")
        pcd_camera = pcd_from_depth(depth, K)
    pcd_base = convert_pcd_to_base(pcd_camera, transform_used)
    bounds = parse_bounds(args.bounds)
    pointcloud, mask_valid, mask_bounds = make_point_cloud(
        pcd_base, rgb, stride=args.stride, bounds=bounds, min_norm=args.min_norm, max_points=args.max_points,
    )
    geoms = [pointcloud]
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.base_frame_size, origin=[0, 0, 0]))
    geoms.append(make_sphere([0, 0, 0], args.marker_radius, [0.0, 0.0, 0.0]))
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.camera_frame_size)
    camera_frame.transform(transform_used)
    geoms.append(camera_frame)
    geoms.append(make_sphere(transform_used[:3, 3], args.marker_radius, [1.0, 0.0, 1.0]))
    if K is not None:
        geoms.append(make_camera_frustum(K, rgb.shape[1], rgb.shape[0], args.frustum_depth, transform_used))
    if bounds is not None:
        geoms.append(make_bounds_box(bounds))
    if args.show_grid:
        geoms.append(make_grid(args.grid_size, args.grid_step, args.grid_z))
    if args.show_action:
        action_T = load_action_frame(episode_dir, frame_idx, args.quat_order)
        if action_T is not None:
            action_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.action_frame_size)
            action_frame.transform(action_T)
            geoms.append(action_frame)
            geoms.append(make_sphere(action_T[:3, 3], args.marker_radius, [1.0, 0.0, 0.0]))
    gripper_T, gripper_open, gripper_raw = None, None, None
    if args.show_gripper:
        gripper_T, gripper_open, gripper_raw = load_gripper_pose(episode_dir, frame_idx, args.quat_order)
        if gripper_T is not None:
            gf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.gripper_frame_size)
            gf.transform(gripper_T)
            geoms.append(gf)
            grip_color = [0.0, 0.85, 0.0] if (gripper_open is not None and gripper_open > 0.5) else [0.85, 0.0, 0.0]
            geoms.append(make_sphere(gripper_T[:3, 3], args.marker_radius, grip_color))
            grip_ls, grip_tip_spheres = make_gripper_marker(
                gripper_T, gripper_open,
                finger_length=args.gripper_finger_len, max_width=args.gripper_max_width,
            )
            geoms.append(grip_ls)
            geoms.extend(grip_tip_spheres)
    print_stats(episode_dir, frame_name, extrinsic, transform_used, pcd_camera, pcd_base, bounds, mask_valid, mask_bounds, K, intrinsic_check, gripper_T, gripper_open, gripper_raw)
    return geoms, episode_dir, frame_name


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 3rd_cam_pcd transformed by extrinsic_matrix.npy in the Dobot base frame.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="Root dir to scan for episodes "
                             "(default: $REAL_COLLECT_DATA_ROOT)")
    parser.add_argument("--episode-dir", default=None,
                        help="Visualize this one episode dir instead of scanning "
                             "--data-root, e.g. "
                             "/abs/path/to/your/real_data/<task_slug>_0")
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--min-norm", type=float, default=1e-8)
    parser.add_argument("--bounds", default=",".join(str(x) for x in DEFAULT_BOUNDS))
    parser.add_argument("--invert-extrinsic", action="store_true")
    parser.add_argument("--intrinsic", default=None)
    parser.add_argument("--use-depth-pinhole", action="store_true")
    parser.add_argument("--show-action", action="store_true")
    parser.add_argument("--show-gripper", action="store_true", default=True)
    parser.add_argument("--no-gripper", dest="show_gripper", action="store_false")
    parser.add_argument("--gripper-frame-size", type=float, default=0.10)
    parser.add_argument("--gripper-finger-len", type=float, default=0.06)
    parser.add_argument("--gripper-max-width", type=float, default=0.05)
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--show-grid", action="store_true", default=True)
    parser.add_argument("--no-grid", dest="show_grid", action="store_false")
    parser.add_argument("--grid-size", type=float, default=1.5)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--grid-z", type=float, default=0.0)
    parser.add_argument("--base-frame-size", type=float, default=0.30)
    parser.add_argument("--camera-frame-size", type=float, default=0.18)
    parser.add_argument("--action-frame-size", type=float, default=0.12)
    parser.add_argument("--marker-radius", type=float, default=0.025)
    parser.add_argument("--frustum-depth", type=float, default=0.35)
    parser.add_argument("--list-first", type=int, default=12)
    return parser.parse_args()


def count_frames(episode_dir: str):
    pcd_dir = os.path.join(episode_dir, "3rd_cam_pcd")
    if not os.path.isdir(pcd_dir):
        return 0
    return len(sorted_numeric_files(pcd_dir))


def main():
    args = parse_args()
    episode_dir = resolve_episode_dir(args)
    total_frames = count_frames(episode_dir)
    if total_frames == 0:
        print("[error] No point cloud frames found.")
        return
    start_frame = args.frame
    print(f"[info] Total frames: {total_frames}, starting from frame {start_frame}")
    for frame_idx in range(start_frame, total_frames):
        args.frame = frame_idx
        geoms, ep_dir, frame_name = build_visualization(args)
        window_name = f"frame {frame_idx}/{total_frames - 1} | {os.path.basename(os.path.dirname(ep_dir))}/{os.path.basename(ep_dir)} | {frame_name}"
        print(f"\n[visualizing] frame {frame_idx}/{total_frames - 1} — close window to advance")
        o3d.visualization.draw_geometries(geoms, window_name=window_name, width=1280, height=900, point_show_normal=False)
    print("[done] All frames visualized.")


if __name__ == "__main__":
    main()
