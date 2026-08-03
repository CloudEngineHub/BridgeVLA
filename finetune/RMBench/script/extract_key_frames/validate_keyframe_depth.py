"""
Validate and visualize keyframe_depth data against original demo_clean HDF5.

Checks:
  1. keyframe_indices match keyframes JSON
  2. joint_action/vector matches original at keyframe frames (threshold 1e-4)
  3. endpose matches original (threshold 1e-3)
  4. RGB mean pixel diff vs original (rendering may differ slightly)
  5. depth sanity (valid range, non-empty)

Visual output (per sampled episode):
  For each keyframe, a 2x2 grid:
    [orig RGB]  [keyframe_depth RGB]
    [abs diff]  [depth colormap]

Usage:
  python data/validate_keyframe_depth.py                     # validate all tasks
  python data/validate_keyframe_depth.py --task battery_try  # single task
  python data/validate_keyframe_depth.py --visualize         # also save images
  python data/validate_keyframe_depth.py --visualize --n_episodes 2 --n_keyframes 3
"""

import argparse
import io
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KEYFRAMES_DIR = "data/keyframes"
ORIG_ROOT = "data/data"
KF_ROOT = "data/keyframe_data"
OUTPUT_ROOT = "data/keyframe_depth_validation"
CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]

JOINT_TOL = 1e-4
ENDPOSE_TOL = 1e-3
RGB_WARN = 3.0
RGB_FAIL = 8.0


def decode_rgb(raw):
    if isinstance(raw, bytes):
        return np.array(Image.open(io.BytesIO(raw)))
    if isinstance(raw, np.ndarray) and raw.dtype.kind == "S":
        return np.array(Image.open(io.BytesIO(raw.tobytes())))
    if isinstance(raw, np.void):
        return np.array(Image.open(io.BytesIO(bytes(raw))))
    return np.array(raw)


def depth_to_colormap(depth, vmin=None, vmax=None):
    """Convert depth (mm) to RGB colormap. Invalid/zero -> black."""
    d = depth.copy().astype(np.float64)
    valid = d > 0
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    if vmin is None:
        vmin = np.percentile(d[valid], 2)
    if vmax is None:
        vmax = np.percentile(d[valid], 98)
    span = max(vmax - vmin, 1e-6)

    norm = np.clip((d - vmin) / span, 0, 1)
    r = (np.clip(1.5 - np.abs(4 * norm - 3), 0, 1) * 255).astype(np.uint8)
    g = (np.clip(1.5 - np.abs(4 * norm - 2), 0, 1) * 255).astype(np.uint8)
    b = (np.clip(1.5 - np.abs(4 * norm - 1), 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    return rgb


def get_font(size=12):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_comparison_panel(orig_rgb, kf_rgb, depth, title_lines):
    """Build 2x2 comparison image for one keyframe."""
    diff = np.abs(orig_rgb.astype(np.float32) - kf_rgb.astype(np.float32))
    diff_vis = np.clip(diff * 8, 0, 255).astype(np.uint8)
    depth_vis = depth_to_colormap(depth)

    h, w = orig_rgb.shape[:2]
    grid = Image.new("RGB", (w * 2, h * 2 + 40), (20, 20, 20))
    panels = [
        (Image.fromarray(orig_rgb), "Original RGB"),
        (Image.fromarray(kf_rgb), "Keyframe Depth RGB"),
        (Image.fromarray(diff_vis), f"Abs Diff (mean={diff.mean():.2f})"),
        (Image.fromarray(depth_vis), "Depth (colormap)"),
    ]
    draw = ImageDraw.Draw(grid)
    font = get_font(11)
    for idx, (img, label) in enumerate(panels):
        x, y = (idx % 2) * w, (idx // 2) * h
        grid.paste(img, (x, y))
        draw.rectangle([x, y, x + w - 1, y + 18], fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 0), font=font)

    y_text = h * 2 + 4
    for line in title_lines:
        draw.text((4, y_text), line, fill=(220, 220, 220), font=font)
        y_text += 14
    return grid


def validate_episode(task, ep_idx, kf_json, visualize=False, out_dir=None, kf_indices_to_vis=None):
    ep_name = f"episode{ep_idx}"
    kf_h5_path = os.path.join(KF_ROOT, task, "keyframe_depth", "data", f"{ep_name}.hdf5")
    orig_h5_path = os.path.join(ORIG_ROOT, task, "demo_clean", "data", f"{ep_name}.hdf5")

    result = {
        "episode": ep_idx,
        "status": "ok",
        "n_keyframes": 0,
        "joint_max_diff": 0.0,
        "endpose_max_diff": 0.0,
        "rgb_mean_max": 0.0,
        "rgb_worst_frame": None,
        "depth_valid_pct_min": 100.0,
        "issues": [],
    }

    if not os.path.exists(kf_h5_path):
        result["status"] = "missing"
        result["issues"].append("keyframe_depth hdf5 not found")
        return result

    if ep_name not in kf_json:
        result["status"] = "error"
        result["issues"].append("episode not in keyframes JSON")
        return result

    expected_kf = sorted(k["frame_idx"] for k in kf_json[ep_name]["keyframes"])
    result["n_keyframes"] = len(expected_kf)

    with h5py.File(kf_h5_path, "r") as kf_f, h5py.File(orig_h5_path, "r") as orig_f:
        stored_kf = list(int(x) for x in kf_f["keyframe_indices"][:])
        if stored_kf != expected_kf:
            result["status"] = "error"
            result["issues"].append(f"keyframe_indices mismatch: stored={stored_kf[:5]}... expected={expected_kf[:5]}...")

        if kf_f["joint_action/vector"].shape[0] != len(stored_kf):
            result["status"] = "error"
            result["issues"].append(
                f"keyframe count mismatch: hdf5 has {kf_f['joint_action/vector'].shape[0]}, "
                f"indices has {len(stored_kf)}"
            )

        for i, frame_idx in enumerate(stored_kf):
            jdiff = np.max(np.abs(kf_f["joint_action/vector"][i] - orig_f["joint_action/vector"][frame_idx]))
            result["joint_max_diff"] = max(result["joint_max_diff"], float(jdiff))
            if jdiff > JOINT_TOL:
                result["status"] = "error"
                result["issues"].append(f"frame {frame_idx}: joint diff={jdiff:.2e}")

            for side in ["left", "right"]:
                ediff = np.max(
                    np.abs(kf_f[f"endpose/{side}_endpose"][i] - orig_f[f"endpose/{side}_endpose"][frame_idx])
                )
                result["endpose_max_diff"] = max(result["endpose_max_diff"], float(ediff))
                if ediff > ENDPOSE_TOL:
                    result["issues"].append(f"frame {frame_idx}: {side}_endpose diff={ediff:.2e}")

            orig_rgb = decode_rgb(orig_f["observation/head_camera/rgb"][frame_idx])
            kf_rgb = decode_rgb(kf_f["observation/head_camera/rgb"][i])
            rgb_diff = np.mean(np.abs(orig_rgb.astype(float) - kf_rgb.astype(float)))
            result["rgb_mean_max"] = max(result["rgb_mean_max"], float(rgb_diff))
            if rgb_diff > result.get("_rgb_worst_val", -1):
                result["_rgb_worst_val"] = rgb_diff
                result["rgb_worst_frame"] = frame_idx

            if rgb_diff > RGB_FAIL:
                result["status"] = "warn"
                result["issues"].append(f"frame {frame_idx}: large RGB diff={rgb_diff:.2f}")
            elif rgb_diff > RGB_WARN and result["status"] == "ok":
                result["status"] = "warn"

            depth = kf_f["observation/head_camera/depth"][i]
            valid_pct = 100.0 * np.count_nonzero(depth > 0) / depth.size
            result["depth_valid_pct_min"] = min(result["depth_valid_pct_min"], valid_pct)
            if valid_pct < 50:
                result["status"] = "warn"
                result["issues"].append(f"frame {frame_idx}: depth valid only {valid_pct:.1f}%")

            if visualize and out_dir and (kf_indices_to_vis is None or i in kf_indices_to_vis):
                title = [
                    f"{task} {ep_name} | kf#{i} frame={frame_idx}",
                    f"joint_max={jdiff:.2e} endpose_max={result['endpose_max_diff']:.2e} rgb_mean={rgb_diff:.2f}",
                ]
                panel = make_comparison_panel(orig_rgb, kf_rgb, depth, title)
                os.makedirs(out_dir, exist_ok=True)
                panel.save(os.path.join(out_dir, f"kf{i:02d}_frame{frame_idx:04d}.png"))

    result.pop("_rgb_worst_val", None)
    if result["status"] == "ok" and result["issues"]:
        result["status"] = "warn"
    return result


def get_tasks(task_filter=None):
    all_tasks = sorted(
        f.replace(".json", "")
        for f in os.listdir(KEYFRAMES_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )
    if task_filter:
        if task_filter not in all_tasks:
            print(f"Unknown task: {task_filter}", file=sys.stderr)
            sys.exit(1)
        return [task_filter]
    return all_tasks


def main():
    parser = argparse.ArgumentParser(description="Validate keyframe_depth against original demo_clean data")
    parser.add_argument("--task", type=str, default=None, help="Single task name")
    parser.add_argument("--visualize", action="store_true", help="Save comparison images")
    parser.add_argument("--n_episodes", type=int, default=3, help="Episodes to visualize per task")
    parser.add_argument("--n_keyframes", type=int, default=4, help="Keyframes to visualize per episode")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT, help="Output directory")
    args = parser.parse_args()

    tasks = get_tasks(args.task)
    os.makedirs(args.output, exist_ok=True)

    full_report = {"tasks": {}, "summary": {}}
    total_ok = total_warn = total_error = total_missing = 0

    for task in tasks:
        kf_json_path = os.path.join(KEYFRAMES_DIR, f"{task}.json")
        kf_json = json.load(open(kf_json_path))
        kf_data_dir = os.path.join(KF_ROOT, task, "keyframe_depth", "data")
        failed_path = os.path.join(KF_ROOT, task, "keyframe_depth", "failed_episodes.json")

        failed_eps = set()
        if os.path.exists(failed_path):
            failed_eps = set(json.load(open(failed_path))["failed"])

        orig_dir = os.path.join(ORIG_ROOT, task, "demo_clean", "data")
        n_orig = len([f for f in os.listdir(orig_dir) if f.endswith(".hdf5")]) if os.path.exists(orig_dir) else 0
        n_kf = len([f for f in os.listdir(kf_data_dir) if f.endswith(".hdf5")]) if os.path.exists(kf_data_dir) else 0

        task_report = {
            "n_orig_episodes": n_orig,
            "n_kf_episodes": n_kf,
            "failed_episodes": sorted(failed_eps),
            "episodes": [],
        }

        print(f"\n{'='*60}")
        print(f"Task: {task}  ({n_kf}/{n_orig} episodes, failed={sorted(failed_eps)})")

        vis_eps = sorted(range(n_orig))[: args.n_episodes] if args.visualize else []

        for ep_idx in range(n_orig):
            if ep_idx in failed_eps:
                task_report["episodes"].append({"episode": ep_idx, "status": "failed_collection"})
                total_missing += 1
                continue

            kf_indices_to_vis = None
            if ep_idx in vis_eps:
                ep_name = f"episode{ep_idx}"
                n_kf_frames = len(kf_json[ep_name]["keyframes"])
                step = max(1, n_kf_frames // args.n_keyframes)
                kf_indices_to_vis = set(range(0, n_kf_frames, step)) | {n_kf_frames - 1}
                out_dir = os.path.join(args.output, task, ep_name)

            ep_result = validate_episode(
                task, ep_idx, kf_json,
                visualize=(ep_idx in vis_eps),
                out_dir=os.path.join(args.output, task, f"episode{ep_idx}") if ep_idx in vis_eps else None,
                kf_indices_to_vis=kf_indices_to_vis,
            )
            task_report["episodes"].append(ep_result)

            st = ep_result["status"]
            if st == "ok":
                total_ok += 1
            elif st == "warn":
                total_warn += 1
            elif st == "missing":
                total_missing += 1
            else:
                total_error += 1

        # Task-level aggregates
        valid_eps = [e for e in task_report["episodes"] if e["status"] not in ("missing", "failed_collection")]
        task_report["aggregate"] = {
            "joint_max_diff": max((e["joint_max_diff"] for e in valid_eps), default=0),
            "endpose_max_diff": max((e["endpose_max_diff"] for e in valid_eps), default=0),
            "rgb_mean_max": max((e["rgb_mean_max"] for e in valid_eps), default=0),
            "depth_valid_pct_min": min((e["depth_valid_pct_min"] for e in valid_eps), default=100),
            "n_ok": sum(1 for e in task_report["episodes"] if e["status"] == "ok"),
            "n_warn": sum(1 for e in task_report["episodes"] if e["status"] == "warn"),
            "n_error": sum(1 for e in task_report["episodes"] if e["status"] == "error"),
        }

        agg = task_report["aggregate"]
        flag = "OK" if agg["n_error"] == 0 and n_kf == n_orig - len(failed_eps) else "CHECK"
        print(f"  [{flag}] joint_max={agg['joint_max_diff']:.2e}  endpose_max={agg['endpose_max_diff']:.2e}  "
              f"rgb_mean_max={agg['rgb_mean_max']:.2f}  depth_valid_min={agg['depth_valid_pct_min']:.1f}%")
        print(f"  ok={agg['n_ok']} warn={agg['n_warn']} error={agg['n_error']} missing/failed={n_orig - n_kf}")

        warn_eps = [e for e in task_report["episodes"] if e["status"] == "warn"]
        if warn_eps:
            print(f"  WARN episodes (RGB diff > {RGB_WARN}):")
            for e in warn_eps[:5]:
                print(f"    ep{e['episode']}: rgb_mean_max={e['rgb_mean_max']:.2f} worst_frame={e['rgb_worst_frame']}")

        full_report["tasks"][task] = task_report

    full_report["summary"] = {
        "total_ok": total_ok,
        "total_warn": total_warn,
        "total_error": total_error,
        "total_missing": total_missing,
    }

    report_path = os.path.join(args.output, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary: ok={total_ok} warn={total_warn} error={total_error} missing={total_missing}")
    print(f"Report saved to {report_path}")
    if args.visualize:
        print(f"Visualizations saved to {args.output}/")


if __name__ == "__main__":
    main()
