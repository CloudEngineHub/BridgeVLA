#!/usr/bin/env python3
"""
Dataset Organizer & Converter (flat episode layout)

Expected raw layout under DATA_ROOT ($REAL_COLLECT_DATA_ROOT, or --data-root):
    put_lids_on_the_blocks_then_uncover_the_blue_block_0/
        instruction.txt  instruction.pkl  extrinsic_matrix.npy  intrinsic.pkl
        3rd_cam_rgb/  3rd_cam_depth/  actions/
    put_lids_on_the_blocks_then_uncover_the_blue_block_1/
    ...

Geometry is carried in one of two layouts. The current collection script writes
3rd_cam_depth/{i}.npy (uint16 depth, uncompressed so the collection path stays
fast) plus intrinsic.pkl; this script is where that gets PNG-compressed for
archival, since the ~150 ms/frame encode is free offline but would dominate the
collection loop. Older trees carry 3rd_cam_pcd/{i}.pkl full-XYZ dumps, which are
copied through as-is — real_dataset.py reads either.

Episode folder name = {instruction_slug}_{episode_idx}  (see dataset_naming.py)

Legacy nested layout (episode group with 0/, 1/, ...) is still discovered
when scanning older data trees.

Output ($REAL_CONVERTED_DATA_ROOT, or --output-root; defaults to
"<DATA_ROOT>_converted"):
    <OUTPUT_ROOT>/
      <task_slug>/
        <task_slug>_{000,001,...,010}/   # converted; episode idx zero-padded to 3 digits
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import transforms3d

from dataset_naming import (
    make_output_name,
    parse_episode_folder_name,
    require_data_root,
)

# Configuration
# Raw collection directory -> the dobot-layout output directory used for training. Neither has a sensible
# default (every machine puts them on a different disk), so point them at your own absolute paths:
#     export REAL_COLLECT_DATA_ROOT=/abs/path/to/your/real_collect
#     export REAL_CONVERTED_DATA_ROOT=/abs/path/to/your/real_collect_converted   # optional
# --data-root / --output-root override them on the CLI.
# With neither set, main() exits with a hint rather than silently writing to the wrong directory.
DATA_ROOT = os.environ.get("REAL_COLLECT_DATA_ROOT", "")
OUTPUT_ROOT = os.environ.get("REAL_CONVERTED_DATA_ROOT", "") or (
    f"{DATA_ROOT}_converted" if DATA_ROOT else ""
)

SKIP_FOLDERS = {
    "20251029_rebuild", "data_process", "lost+found", "test",
    ".Trash-1000", "data_1107",
    "_undo_trash",                    # the collection undo bin, not an episode
}
# Skip the output directory when it is nested inside the input directory, so conversion results are not rescanned.
if OUTPUT_ROOT:
    SKIP_FOLDERS.add(os.path.basename(OUTPUT_ROOT.rstrip("/")))

MIN_FRAMES = 3

# Logging (configured in main)
log = logging.getLogger(__name__)


# Legacy folder name parsing (nested episode groups from older collection)

def parse_legacy_episode_group_name(folder_name: str):
    """
    Old format: {task_name}[_{orig_idx}]_{date}
    Returns (task_name, orig_idx, date_str) or None.
    """
    name = folder_name.strip()
    if not re.match(r"^put[\s_]the[\s_]", name, re.IGNORECASE):
        return None

    parts = re.split(r"[\s_]+", name)
    if not parts or not re.fullmatch(r"\d{4,}", parts[-1]):
        return None
    date_str = parts.pop()
    orig_idx = None
    if parts and re.fullmatch(r"\d{1,2}", parts[-1]):
        orig_idx = parts.pop()
    if not parts:
        return None
    task_name = "_".join(parts)
    return task_name, orig_idx, date_str


def make_legacy_output_name(task_name: str, date_str: str, orig_idx, sub_id: str) -> str:
    if orig_idx is not None:
        return f"{task_name}_{orig_idx}_{date_str}_{sub_id}"
    return f"{task_name}_{date_str}_{sub_id}"


# Discovery

def _has_instruction(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, "instruction.txt"))


def is_dobot_formate(name: str) -> bool:
    return name.startswith("dobot_formate")


def find_flat_episodes(data_root: str) -> List[Tuple[str, str, str]]:
    """
    Discover flat episodes: folders directly under data_root named
    {task_slug}_{idx} with instruction.txt inside.

    Returns list of (abs_path, task_slug, episode_idx).
    """
    results: List[Tuple[str, str, str]] = []
    if not os.path.isdir(data_root):
        return results

    for name in sorted(os.listdir(data_root)):
        if name in SKIP_FOLDERS or is_dobot_formate(name):
            continue
        path = os.path.join(data_root, name)
        if not os.path.isdir(path) or not _has_instruction(path):
            continue
        parsed = parse_episode_folder_name(name)
        if parsed is None:
            log.debug(f"  skip non-episode dir: {name}")
            continue
        task_slug, episode_idx = parsed
        results.append((path, task_slug, episode_idx))
    return results


def find_legacy_episode_groups(scan_dir: str, seen_dobot_names: set) -> list:
    """Recursively find old nested episode groups (children 0/, 1/ have instruction.txt)."""
    results = []
    try:
        entries = sorted(os.listdir(scan_dir))
    except PermissionError:
        return results

    child_dirs = [e for e in entries if os.path.isdir(os.path.join(scan_dir, e))]
    children_with_instruction = [
        d for d in child_dirs if _has_instruction(os.path.join(scan_dir, d))
    ]

    if children_with_instruction:
        folder_name = os.path.basename(scan_dir)
        if is_dobot_formate(folder_name):
            if folder_name in seen_dobot_names:
                return results
            seen_dobot_names.add(folder_name)
            return results  # skip dobot sources
        results.append(scan_dir)
    else:
        for entry in child_dirs:
            if entry in SKIP_FOLDERS:
                continue
            results.extend(find_legacy_episode_groups(os.path.join(scan_dir, entry), seen_dobot_names))
    return results


# Validation

def validate_raw_sub_episode(sub_ep_path: str, min_frames: int) -> tuple:
    # Geometry comes from either layout: the compact 3rd_cam_depth/{i}.npy
    # (uint16 depth + intrinsic.pkl, current collection script) or the legacy
    # 3rd_cam_pcd/{i}.pkl full-XYZ dumps.
    has_depth = os.path.isdir(os.path.join(sub_ep_path, "3rd_cam_depth"))
    has_pcd = os.path.isdir(os.path.join(sub_ep_path, "3rd_cam_pcd"))
    if not (has_depth or has_pcd):
        return False, "missing both 3rd_cam_depth and 3rd_cam_pcd", 0
    geom_dir = "3rd_cam_depth" if has_depth else "3rd_cam_pcd"

    required_dirs = ["3rd_cam_rgb", geom_dir, "actions"]
    required_files = ["extrinsic_matrix.npy", "instruction.pkl"]
    if has_depth:
        # Depth alone cannot rebuild the cloud — the intrinsics are mandatory.
        required_files.append("intrinsic.pkl")

    for f in required_files:
        if not os.path.exists(os.path.join(sub_ep_path, f)):
            return False, f"missing {f}", 0

    frame_counts = {}
    for d in required_dirs:
        dir_path = os.path.join(sub_ep_path, d)
        if not os.path.isdir(dir_path):
            return False, f"missing directory {d}", 0
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        frame_counts[d] = len(files)

    counts = list(frame_counts.values())
    if len(set(counts)) > 1:
        return False, f"inconsistent frame counts: {frame_counts}", 0

    num_frames = counts[0]
    if num_frames < min_frames:
        return False, f"too few frames ({num_frames} < {min_frames})", num_frames

    return True, "ok", num_frames


# Conversion (raw → dobot format)

RGB_JPEG_QUALITY = 95
# Set from --rgb-lossless in main(); when True the raw .pkl frames are copied
# through untouched.
RGB_LOSSLESS = False


def convert_rgb(src_sub_ep: str, dst_sub_ep: str):
    """Copy RGB across, re-encoding the pickled arrays as JPEG.

    6.2 MB/frame of raw uint8 becomes ~0.4 MB at q95 (~47 dB PSNR; the only
    pixels that move meaningfully are isolated sensor hot-pixels, which the
    stride-4 downsample would drop anyway). This is lossy — pass
    ``--rgb-lossless`` to keep the original .pkl instead.
    """
    src_dir = os.path.join(src_sub_ep, "3rd_cam_rgb")
    dst_dir = os.path.join(dst_sub_ep, "zed_rgb")
    os.makedirs(dst_dir, exist_ok=True)
    for fname in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue
        if RGB_LOSSLESS or not fname.endswith(".pkl"):
            shutil.copy2(src, os.path.join(dst_dir, fname))
            continue
        with open(src, "rb") as f:
            rgb = np.asarray(pickle.load(f))[:, :, :3]
        dst = os.path.join(dst_dir, f"{os.path.splitext(fname)[0]}.jpg")
        if not cv2.imwrite(dst, np.ascontiguousarray(rgb),
                           [cv2.IMWRITE_JPEG_QUALITY, RGB_JPEG_QUALITY]):
            raise RuntimeError(f"cv2.imwrite failed: {dst}")


def write_intrinsic(src_sub_ep: str, dst_sub_ep: str, n_frames: int):
    """Copy intrinsic.pkl, broadcasting the constant K to one entry per frame.

    The collection script writes intrinsic.pkl at episode *setup*, before any
    frame exists (so a run that dies mid-episode still has the four numbers it
    needs to deproject). It therefore cannot size the arrays, and stores
    fx/fy/cx/cy as length-1 with ``num_frames = -1``.

    convert_pcd_to_depth.py, converting legacy zed_pcd dumps, fits K per frame
    and writes full-length arrays. real_dataset.py consumes ``meta["fx"][i]``
    via ``min(frame_step, len(fx) - 1)``, so the short form already loads — it
    just silently clamps every frame to index 0. Two shapes for the same field
    is a schema split waiting to trip up anything that indexes without the
    clamp, so this is where the two producers are reconciled: the frame count
    is only known here, once the depth frames have been written.

    K really is constant within an episode (measured spread across the
    per-frame fits: ~1e-6 px), so the broadcast is exact, not an approximation.
    ``source`` and ``verify_err`` are passed through untouched — they record
    how K was obtained and differ legitimately between the two producers.
    """
    src = os.path.join(src_sub_ep, "intrinsic.pkl")
    dst = os.path.join(dst_sub_ep, "intrinsic.pkl")
    with open(src, "rb") as f:
        meta = dict(pickle.load(f))

    for key in ("fx", "fy", "cx", "cy"):
        v = np.asarray(meta[key], dtype=np.float64).ravel()
        if v.size == 1:
            meta[key] = np.full(n_frames, float(v[0]), dtype=np.float64)
        elif v.size == n_frames:
            meta[key] = v
        else:
            raise ValueError(
                f"{src}: intrinsic '{key}' has {v.size} entries, expected 1 "
                f"(constant K) or {n_frames} (one per frame)")
    meta["num_frames"] = n_frames

    with open(dst, "wb") as f:
        pickle.dump(meta, f)


def convert_pcd(src_sub_ep: str, dst_sub_ep: str):
    """Carry the frame geometry across, preferring the compact depth layout.

    New layout: 3rd_cam_depth/{i}.npy (uint16, uncompressed so the collection
    path stays fast) is PNG-compressed here — this is the offline step where
    the ~150 ms/frame encode is free, and it takes 4.15 MB down to ~0.8 MB.
    intrinsic.pkl rides along; without it the depth cannot be deprojected.

    Legacy layout: 3rd_cam_pcd/{i}.pkl full-XYZ dumps are copied verbatim to
    zed_pcd/, which real_dataset.py still reads.
    """
    depth_src = os.path.join(src_sub_ep, "3rd_cam_depth")
    if os.path.isdir(depth_src):
        dst_dir = os.path.join(dst_sub_ep, "zed_depth")
        os.makedirs(dst_dir, exist_ok=True)
        n_frames = 0
        for fname in sorted(os.listdir(depth_src)):
            src = os.path.join(depth_src, fname)
            if not os.path.isfile(src) or not fname.endswith(".npy"):
                continue
            stem = os.path.splitext(fname)[0]
            d16 = np.load(src)
            if d16.dtype != np.uint16:
                raise ValueError(f"{src}: expected uint16 depth, got {d16.dtype}")
            dst = os.path.join(dst_dir, f"{stem}.png")
            if not cv2.imwrite(dst, d16, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"cv2.imwrite failed: {dst}")
            # PNG is the archival format — confirm it decodes back bit-exact
            # before the uncompressed .npy is considered redundant.
            back = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
            if back is None or not np.array_equal(back, d16):
                raise RuntimeError(f"depth PNG round-trip mismatch: {dst}")
            n_frames += 1
        write_intrinsic(src_sub_ep, dst_sub_ep, n_frames)
        return

    src_dir = os.path.join(src_sub_ep, "3rd_cam_pcd")
    dst_dir = os.path.join(dst_sub_ep, "zed_pcd")
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


def convert_pose(src_sub_ep: str, dst_sub_ep: str):
    actions_dir = os.path.join(src_sub_ep, "actions")
    dst_path = os.path.join(dst_sub_ep, "pose.pkl")

    pose_files = sorted(
        [f for f in os.listdir(actions_dir) if f.endswith(".pkl")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    pose_lines = ["Timestamp Position (X, Y, Z) Orientation (Rx, Ry, Rz)"]

    for fname in pose_files:
        src_file = os.path.join(actions_dir, fname)
        with open(src_file, "rb") as f:
            pose = pickle.load(f)
            t = pose[0:3] * 1000.0
            r_quat = pose[3:7]
            gripper_state = pose[7]
            r_euler = transforms3d.euler.quat2euler(r_quat, axes="sxyz")
            r_euler = np.rad2deg(r_euler)
            combined = np.concatenate([t, r_euler])
            result_str = " ".join(str(x) for x in combined)
            if len(pose) == 9:
                arm_flag = pose[8]
                result_str = "202505300000 " + result_str + f" {int(gripper_state)} {int(arm_flag)}"
            else:
                result_str = "202505300000 " + result_str + f" {int(gripper_state)}"
            pose_lines.append(result_str)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump("\n".join(pose_lines), f)


def convert_extrinsic(src_sub_ep: str, dst_sub_ep: str):
    src_path = os.path.join(src_sub_ep, "extrinsic_matrix.npy")
    dst_path = os.path.join(dst_sub_ep, "extrinsic_matrix.pkl")
    extrinsic_matrix = np.load(src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(extrinsic_matrix, f)


def convert_instruction(src_sub_ep: str, dst_sub_ep: str):
    src_path = os.path.join(src_sub_ep, "instruction.pkl")
    dst_path = os.path.join(dst_sub_ep, "instruction.pkl")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def save_image_sample(src_sub_ep: str, dst_sub_ep: str) -> bool:
    dst_img = os.path.join(dst_sub_ep, "image_sample", "0.png")
    if os.path.isfile(dst_img):
        return False
    src_dir = os.path.join(src_sub_ep, "3rd_cam_rgb")
    if not os.path.isdir(src_dir):
        return False
    pngs = sorted(
        [f for f in os.listdir(src_dir) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0,
    )
    if not pngs:
        return False
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    shutil.copy2(os.path.join(src_dir, pngs[0]), dst_img)
    return True


def save_cam_img(src_sub_ep: str, dst_sub_ep: str) -> bool:
    dst_img = os.path.join(dst_sub_ep, "cam_img", "0.png")
    if os.path.isfile(dst_img):
        return False
    src_dir = None
    for candidate in ("3rd_cam_imgs", "3rd_cam_rgb"):
        p = os.path.join(src_sub_ep, candidate)
        if os.path.isdir(p):
            src_dir = p
            break
    if src_dir is None:
        return False
    pngs = sorted(
        [f for f in os.listdir(src_dir) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0,
    )
    if not pngs:
        return False
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    shutil.copy2(os.path.join(src_dir, pngs[0]), dst_img)
    return True


def convert_sub_episode(src_sub_ep: str, dst_sub_ep: str):
    os.makedirs(dst_sub_ep, exist_ok=True)
    convert_rgb(src_sub_ep, dst_sub_ep)
    convert_pcd(src_sub_ep, dst_sub_ep)
    convert_pose(src_sub_ep, dst_sub_ep)
    convert_extrinsic(src_sub_ep, dst_sub_ep)
    convert_instruction(src_sub_ep, dst_sub_ep)
    save_image_sample(src_sub_ep, dst_sub_ep)
    save_cam_img(src_sub_ep, dst_sub_ep)


# Main pipeline

def collect_flat_episodes(data_root: str, min_frames: int):
    """Returns list of (src_path, task_slug, episode_idx, category, nframes)."""
    valid = []
    broken = 0
    for ep_path, task_slug, episode_idx in find_flat_episodes(data_root):
        ok, reason, nframes = validate_raw_sub_episode(ep_path, min_frames)
        if ok:
            valid.append((ep_path, task_slug, episode_idx, task_slug, nframes))
        else:
            broken += 1
            log.warning(f"  BROKEN {ep_path}: {reason}")
    return valid, broken


def collect_legacy_episodes(data_root: str, min_frames: int):
    """Returns same tuple shape from old nested groups."""
    valid = []
    broken = 0
    seen_names: set = set()
    seen_dobot: set = set()

    for ep_path in find_legacy_episode_groups(data_root, seen_dobot):
        folder_name = os.path.basename(ep_path)
        if folder_name in seen_names:
            continue
        seen_names.add(folder_name)

        parsed = parse_legacy_episode_group_name(folder_name)
        if parsed is None:
            log.warning(f"  SKIP legacy (cannot parse): {ep_path}")
            continue
        task_name, orig_idx, date_str = parsed
        category = task_name

        sub_ep_ids = sorted(
            [d for d in os.listdir(ep_path) if d.isdigit() and os.path.isdir(os.path.join(ep_path, d))],
            key=int,
        )
        for sub_id in sub_ep_ids:
            sub_path = os.path.join(ep_path, sub_id)
            ok, reason, nframes = validate_raw_sub_episode(sub_path, min_frames)
            if ok:
                # Use legacy output naming via synthetic episode_idx key
                legacy_key = make_legacy_output_name(task_name, date_str, orig_idx, sub_id)
                valid.append((sub_path, task_name, legacy_key, category, nframes))
            else:
                broken += 1
                log.warning(f"  BROKEN legacy {sub_path}: {reason}")
    return valid, broken


def main():
    parser = argparse.ArgumentParser(description="Organize and convert robot dataset")
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help="Root dir with flat episode folders "
                             "(default: $REAL_COLLECT_DATA_ROOT — required)")
    parser.add_argument("--output-root", default=OUTPUT_ROOT,
                        help="Output directory (default: $REAL_CONVERTED_DATA_ROOT, "
                             "else <data-root>_converted)")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, no writes")
    parser.add_argument("--overwrite", action="store_true", help="Re-convert existing outputs")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also scan for old nested episode groups (put_the_*_MMDD/0/)",
    )
    parser.add_argument(
        "--rgb-lossless",
        action="store_true",
        help="Copy 3rd_cam_rgb/*.pkl verbatim instead of re-encoding to JPEG q95 "
             "(~15x larger on disk)",
    )
    args = parser.parse_args()

    global RGB_LOSSLESS
    RGB_LOSSLESS = args.rgb_lossless

    # Resolve the roots before anything touches the filesystem (the log file
    # below already lands inside data_root).
    args.data_root = require_data_root(args.data_root)
    args.output_root = require_data_root(
        args.output_root or f"{args.data_root}_converted",
        env_var="REAL_CONVERTED_DATA_ROOT",
        cli_flag="--output-root",
        must_exist=False,
    )
    SKIP_FOLDERS.add(os.path.basename(args.output_root.rstrip("/")))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.data_root, "organize_dataset.log"), mode="w"),
        ],
    )

    data_root = args.data_root
    output_root = args.output_root
    dry_run = args.dry_run
    skip_existing = not args.overwrite
    min_frames = args.min_frames

    log.info(f"Data root      : {data_root}")
    log.info(f"Output root    : {output_root}")
    log.info(f"Dry run        : {dry_run}")
    log.info(f"Skip existing  : {skip_existing}")
    log.info(f"Min frames     : {min_frames}")
    log.info(f"Legacy scan    : {args.legacy}")

    log.info("=" * 60)
    log.info("Step 1: Discovering episodes ...")

    flat_valid, flat_broken = collect_flat_episodes(data_root, min_frames)
    log.info(f"  Flat episodes valid : {len(flat_valid)}, broken : {flat_broken}")

    all_valid = list(flat_valid)
    total_broken = flat_broken

    if args.legacy:
        legacy_valid, legacy_broken = collect_legacy_episodes(data_root, min_frames)
        log.info(f"  Legacy episodes valid : {len(legacy_valid)}, broken : {legacy_broken}")
        all_valid.extend(legacy_valid)
        total_broken += legacy_broken

    log.info("=" * 60)
    log.info("Step 2: Building conversion plan ...")

    conversion_plan = []
    category_stats = defaultdict(int)

    for src_path, task_slug, episode_key, category, _nframes in all_valid:
        # Flat layout: episode_key is numeric string; legacy: episode_key is full legacy name
        if episode_key.isdigit():
            out_name = make_output_name(task_slug, episode_key)
        else:
            out_name = episode_key
        dst_ep_dir = os.path.join(output_root, category, out_name)
        conversion_plan.append((src_path, dst_ep_dir))
        category_stats[category] += 1

    for cat in sorted(category_stats.keys()):
        log.info(f"  {cat}: {category_stats[cat]} episodes")
    log.info(f"  Total planned: {len(conversion_plan)}")

    log.info("=" * 60)
    if dry_run:
        log.info("DRY RUN — planned output:")
        for src, dst in conversion_plan:
            log.info(f"  {src}\n    -> {dst}")
        return

    os.makedirs(output_root, exist_ok=True)
    skipped_existing = 0
    log.info("Converting ...")
    for i, (src_sub_path, dst_ep_dir) in enumerate(conversion_plan):
        if skip_existing and os.path.isdir(dst_ep_dir):
            saved = save_cam_img(src_sub_path, dst_ep_dir)
            tag = "SKIP+CAM_IMG" if saved else "SKIP"
            log.info(f"  [{i+1}/{len(conversion_plan)}] {tag} {os.path.basename(dst_ep_dir)}")
            skipped_existing += 1
            continue
        log.info(f"  [{i+1}/{len(conversion_plan)}] {os.path.basename(dst_ep_dir)}")
        try:
            convert_sub_episode(src_sub_path, dst_ep_dir)
        except Exception as e:
            log.error(f"    FAILED {src_sub_path}: {e}")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"  Valid episodes  : {len(all_valid)}")
    log.info(f"  Broken          : {total_broken}")
    log.info(f"  Output planned  : {len(conversion_plan)}")
    if not dry_run:
        log.info(f"  Written under   : {output_root}")
        if skip_existing:
            log.info(f"  Skipped existing: {skipped_existing}")
    log.info("Done.")


if __name__ == "__main__":
    main()
