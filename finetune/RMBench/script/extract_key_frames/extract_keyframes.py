"""
Extract keyframes from RMBench demo data (trajectory heuristic).

Data characteristics (the demos were collected by moving keyframe-to-keyframe):
  * When the robot settles at a target it holds joint positions EXACTLY, so the
    per-frame joint velocity is essentially 0 at every stop.
  * Gripper open/close only happens WHILE the arm is held still (~24 frames);
    the last still frame of such a segment is the fully open/closed + settled
    pose.
  * Genuine motions between stops last >10 frames, while 1-2 frame "gaps" inside
    a stop are just numerical jitter that splits one logical stop in two.

Algorithm:
  1. Compute per-arm joint L2 velocity.
  2. A frame is "still" only when BOTH arms are below vel_thresh (a keyframe is a
     bimanual settle point, not one arm stopping while the other still moves).
  3. Find contiguous both-still segments and merge segments separated by a tiny
     moving gap (<= merge_gap frames) to absorb jitter.
  4. The keyframe of each segment is its LAST frame (target reached / gripper
     fully actuated and settled).
  5. A gripper open/close is attributed to the segment during which it happens
     (gripper value at segment start vs end).

Frame 0 (start) and the last frame (end) are always included.

Usage:
    python data/extract_keyframes.py
    python data/extract_keyframes.py --vel_thresh 1e-4 --merge_gap 2
"""

import argparse
import json
import os
from glob import glob

import h5py
import numpy as np


# Tasks where the arm hovers above a target (e.g. a button) and only twitches a
# few millimetres, producing duplicate stops at the same pose. Only these tasks
# get end-effector displacement de-duplication; all others are left untouched.
DISPLACEMENT_MERGE_TASKS = {"blocks_ranking_try", "press_button"}


def compute_arm_velocity(joint_positions):
    """Per-frame joint L2 velocity; index aligns with frames."""
    if len(joint_positions) < 2:
        return np.zeros(len(joint_positions))
    vel = np.linalg.norm(np.diff(joint_positions, axis=0), axis=1)
    return np.concatenate([[0.0], vel])


def find_still_segments(is_still, min_len=1):
    """Contiguous runs where is_still is True (inclusive indices)."""
    segments = []
    start = None
    for i, still in enumerate(is_still):
        if still:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_len:
                    segments.append((start, i - 1))
                start = None
    if start is not None and len(is_still) - start >= min_len:
        segments.append((start, len(is_still) - 1))
    return segments


def merge_close_segments(segments, merge_gap):
    """Merge still segments separated by <= merge_gap moving frames (jitter)."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for seg_start, seg_end in segments[1:]:
        if seg_start - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = seg_end
        else:
            merged.append([seg_start, seg_end])
    return [tuple(m) for m in merged]


def detect_gripper_change(gripper_vals, seg_start, seg_end, gripper_thresh):
    """Return 'open'/'close'/None for the gripper change within a still segment.

    1.0 = fully open, 0.0 = fully closed: increasing -> open, decreasing -> close.
    """
    delta = float(gripper_vals[seg_end]) - float(gripper_vals[seg_start])
    if abs(delta) < gripper_thresh:
        return None
    return "open" if delta > 0 else "close"


def merge_by_displacement(merged, left_xyz, right_xyz, min_disp):
    """Collapse consecutive keyframes that sit at essentially the same pose.

    Some tasks (e.g. pressing a button) have the arm hover above a target and
    only twitch a few millimetres, which yields a duplicate stop at the same
    pose. When BOTH arms moved less than min_disp (metres) since the previous
    kept keyframe, the two are merged (keep the later, more settled frame and
    union their reasons). The initial "start" frame is never merged away.
    """
    if min_disp <= 0 or len(merged) <= 2:
        return merged
    kept = [merged[0]]
    for kf in merged[1:]:
        prev = kept[-1]
        dl = float(np.linalg.norm(left_xyz[kf["frame_idx"]] - left_xyz[prev["frame_idx"]]))
        dr = float(np.linalg.norm(right_xyz[kf["frame_idx"]] - right_xyz[prev["frame_idx"]]))
        if dl < min_disp and dr < min_disp and prev["frame_idx"] != 0:
            union = prev["reason"].split(";")
            for part in kf["reason"].split(";"):
                if part not in union:
                    union.append(part)
            kept[-1] = {"frame_idx": kf["frame_idx"], "reason": ";".join(union)}
        else:
            kept.append(kf)
    return kept


def extract_keyframes_for_episode(
    hdf5_path,
    vel_thresh=1e-4,
    min_still_len=1,
    merge_gap=2,
    gripper_thresh=0.1,
    min_displacement=0.0,
):
    """Extract keyframes from a single episode HDF5 file."""
    f = h5py.File(hdf5_path, "r")
    left_arm = f["joint_action/left_arm"][:]
    right_arm = f["joint_action/right_arm"][:]
    left_gripper = f["joint_action/left_gripper"][:]
    right_gripper = f["joint_action/right_gripper"][:]
    left_xyz = f["endpose/left_endpose"][:, :3]
    right_xyz = f["endpose/right_endpose"][:, :3]
    n_frames = len(left_gripper)
    f.close()

    left_vel = compute_arm_velocity(left_arm)
    right_vel = compute_arm_velocity(right_arm)

    # A keyframe is a bimanual settle point: BOTH arms must be still.
    both_still = (left_vel < vel_thresh) & (right_vel < vel_thresh)

    segments = find_still_segments(both_still, min_len=1)
    segments = merge_close_segments(segments, merge_gap)
    segments = [s for s in segments if s[1] - s[0] + 1 >= min_still_len]

    keyframes = {}  # frame_idx -> reason

    def add_keyframe(frame_idx, reason):
        if frame_idx in keyframes:
            if reason not in keyframes[frame_idx].split(";"):
                keyframes[frame_idx] = keyframes[frame_idx] + ";" + reason
        else:
            keyframes[frame_idx] = reason

    for seg_start, seg_end in segments:
        # Segment containing frame 0 is the initial home pose -> "start".
        if seg_start == 0:
            continue

        reason_parts = ["vel_zero"]
        lg_change = detect_gripper_change(left_gripper, seg_start, seg_end, gripper_thresh)
        rg_change = detect_gripper_change(right_gripper, seg_start, seg_end, gripper_thresh)
        if lg_change:
            reason_parts.append(f"gripper_{lg_change}_L")
        if rg_change:
            reason_parts.append(f"gripper_{rg_change}_R")

        add_keyframe(seg_end, "+".join(reason_parts))

    keyframes[0] = "start"
    if (n_frames - 1) not in keyframes:
        keyframes[n_frames - 1] = "end"

    merged = [
        {"frame_idx": frame, "reason": keyframes[frame]}
        for frame in sorted(keyframes.keys())
    ]

    # Drop near-stationary duplicate stops (e.g. hovering above a button).
    merged = merge_by_displacement(merged, left_xyz, right_xyz, min_displacement)

    for kf in merged:
        idx = kf["frame_idx"]
        kf["left_gripper"] = float(left_gripper[idx])
        kf["right_gripper"] = float(right_gripper[idx])

    return merged, n_frames


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from RMBench data")
    parser.add_argument("--data_root", type=str, default="data/data")
    parser.add_argument("--output_dir", type=str, default="data/keyframes")
    parser.add_argument(
        "--vel_thresh", type=float, default=1e-4,
        help="Both-arm joint velocity below this counts as 'still'",
    )
    parser.add_argument(
        "--min_still_len", type=int, default=1,
        help="Min length (after gap-merge) of a still segment to keep",
    )
    parser.add_argument(
        "--merge_gap", type=int, default=2,
        help="Merge still segments separated by <= this many moving frames (jitter)",
    )
    parser.add_argument(
        "--gripper_thresh", type=float, default=0.1,
        help="Min gripper value change within a segment to label open/close",
    )
    parser.add_argument(
        "--min_displacement", type=float, default=0.01,
        help=(
            "For DISPLACEMENT_MERGE_TASKS only: collapse consecutive keyframes "
            "whose both-arm end-effector displacement (m) is below this value "
            "(removes near-stationary hover duplicates, e.g. above a button)"
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )

    summary = {}

    for task in tasks:
        task_data_dir = os.path.join(args.data_root, task, "demo_clean", "data")
        if not os.path.isdir(task_data_dir):
            print(f"[SKIP] {task}: no demo_clean/data found")
            continue

        hdf5_files = sorted(glob(os.path.join(task_data_dir, "episode*.hdf5")))
        if not hdf5_files:
            print(f"[SKIP] {task}: no episode HDF5 files")
            continue

        lang_path = os.path.join(args.data_root, task, "demo_clean", "language_annotation.json")
        lang_annotations = {}
        if os.path.exists(lang_path):
            with open(lang_path) as lf:
                lang_annotations = json.load(lf)

        instr_dir = os.path.join(args.data_root, task, "demo_clean", "instructions")

        task_output = {}
        total_kf = 0
        total_frames = 0

        for hdf5_path in hdf5_files:
            ep_name = os.path.splitext(os.path.basename(hdf5_path))[0]
            ep_idx = int(ep_name.replace("episode", ""))

            min_disp = args.min_displacement if task in DISPLACEMENT_MERGE_TASKS else 0.0
            keyframes, n_frames = extract_keyframes_for_episode(
                hdf5_path,
                vel_thresh=args.vel_thresh,
                min_still_len=args.min_still_len,
                merge_gap=args.merge_gap,
                gripper_thresh=args.gripper_thresh,
                min_displacement=min_disp,
            )

            lang_key = f"episode_{ep_idx}"
            lang_segs = lang_annotations.get(lang_key, [])

            if lang_segs:
                cum_frames = 0
                seg_boundaries = []
                for seg_idx, (seg_text, seg_len) in enumerate(lang_segs):
                    seg_boundaries.append(
                        (cum_frames, cum_frames + seg_len - 1, seg_text, seg_idx)
                    )
                    cum_frames += seg_len

                # ``subtask_idx`` = index of the original language segment a
                # keyframe falls in. Unlike ``language_annotation`` (text), it
                # stays distinct across consecutive segments that share the same
                # text, which counting tasks (e.g. press_button: N identical
                # "press one time" segments) rely on to keep each repeated
                # action as its own subtask. See SUBTASK_BY_SEGMENT_TASKS in
                # RMBench_vla/rmbench_dataset.py.
                for kf in keyframes:
                    for seg_start, seg_end, seg_text, seg_idx in seg_boundaries:
                        if seg_start <= kf["frame_idx"] <= seg_end:
                            kf["language_annotation"] = seg_text
                            kf["subtask_idx"] = seg_idx
                            break

            instr_path = os.path.join(instr_dir, f"{ep_name}.json")
            if os.path.exists(instr_path):
                with open(instr_path) as inf:
                    instr = json.load(inf)
                episode_instruction = instr.get("seen", [""])[0]
            else:
                episode_instruction = ""

            task_output[ep_name] = {
                "n_frames": n_frames,
                "n_keyframes": len(keyframes),
                "instruction": episode_instruction,
                "keyframes": keyframes,
            }

            total_kf += len(keyframes)
            total_frames += n_frames

        task_output_path = os.path.join(args.output_dir, f"{task}.json")
        with open(task_output_path, "w") as of:
            json.dump(task_output, of, indent=2)

        avg_kf = total_kf / len(hdf5_files) if hdf5_files else 0
        avg_frames = total_frames / len(hdf5_files) if hdf5_files else 0
        summary[task] = {
            "n_episodes": len(hdf5_files),
            "total_keyframes": total_kf,
            "avg_keyframes_per_episode": round(avg_kf, 1),
            "avg_frames_per_episode": round(avg_frames, 1),
            "compression_ratio": round(avg_kf / avg_frames * 100, 1) if avg_frames > 0 else 0,
        }
        print(
            f"[OK] {task}: {len(hdf5_files)} episodes, {total_kf} keyframes "
            f"(avg {avg_kf:.1f}/ep from {avg_frames:.0f} frames, "
            f"{summary[task]['compression_ratio']}%)"
        )

    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
