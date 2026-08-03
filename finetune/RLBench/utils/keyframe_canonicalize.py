"""
Canonicalize RLBench keyframes so every episode of the same (task, variation)
shares an identical keyframe STRUCTURE.

Why
---
RLBench demos of one variation all execute the SAME scripted waypoint sequence;
per-episode differences in the heuristic ``keypoint_discovery`` output are pure
execution noise:

  * negative-index wrap in ``_is_stopped`` marks frame 1 as a keyframe whenever
    the episode's LAST gripper state matches the first (put_item_in_drawer
    100/100 eps, close_jar 99/100) -> a "predict staying in place" sample;
  * planner micro-adjustments near a goal add extra near-duplicate stop
    keyframes (poses < 5 mm apart);
  * physically failed grasps that the task script retries add a spurious
    close->open gripper-event pair (displacement between the two events is
    tiny, ~50 mm, vs > 300 mm for a real transport cycle);
  * fast blended motion passes a waypoint without max|joint_velocities|
    dropping below the hard 0.1 threshold -> a stop keyframe is MISSED (the
    velocity profile still shows a clear local minimum there).

Approach: majority vote + minimal repair
----------------------------------------
1. Per episode, normalize the heuristic keyframes: split into gripper EVENTS
   (exact, reliable) and intermediate STOPS; drop stops whose pose is within
   ``merge_eps`` of the previous kept keyframe (kills the frame-1 artifact and
   micro-adjust duplicates).
2. Per (task, variation) group, vote: modal gripper-event signature, then
   modal per-segment stop-count tuple among signature-conforming episodes.
   Groups smaller than ``min_group`` fall back to the modal template of other
   variations of the same task that share the event signature (only when they
   all agree; e.g. stack_blocks single-episode variations).
3. Repair ONLY deviating episodes:
     - extra close->open event pairs with small displacement are removed
       (failed grasp retries);
     - segments with too many stops drop the most redundant ones (smallest
       displacement to the previous keyframe);
     - segments with too few stops add frames at the deepest local minima of
       the (lightly smoothed) joint-speed profile.
   Episodes already matching the template keep their ORIGINAL keyframes
   bit-for-bit, so canonical episodes are untouched.
4. Episodes whose event signature cannot be reconciled (e.g. turn_tap demos
   where the recorded gripper never closes, or a truncated demo that never
   releases) are kept as-is (after step-1 merge) and flagged ``conform=False``.

Only numpy is required; observations just need ``gripper_open``,
``joint_velocities`` and ``gripper_pose`` (works with stub-unpickled demos).
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Pose distance below which two same-grip keyframes are considered duplicates.
# BOTH thresholds must hold: rotation-only actions (turn_tap turning the tap,
# reach_and_drag re-orienting the grasped stick) move < 5 mm but are real steps.
MERGE_EPS_M = 0.005
MERGE_EPS_DEG = 10.0
# Max displacement between a close->open event pair for it to be a plausible
# failed-grasp retry (real pick->place cycles move the gripper much farther).
FAILED_GRASP_MAX_DISP_M = 0.08
# Minimum frame separation between an added stop and existing keyframes.
MIN_STOP_SEP_FRAMES = 4
# Groups with fewer episodes than this use the task-level signature-family
# fallback for the template vote.
MIN_GROUP_FOR_VOTE = 3


def episode_arrays(demo) -> dict:
    """Extract the low-dim arrays the canonicalizer needs from a demo
    (a Demo object or a plain list of Observation-like objects)."""
    n = len(demo)
    grip = np.empty((n,), dtype=np.float64)
    speed = np.empty((n,), dtype=np.float64)
    pos = np.empty((n, 3), dtype=np.float64)
    quat = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        o = demo[i]
        grip[i] = float(o.gripper_open)
        speed[i] = float(np.max(np.abs(o.joint_velocities)))
        gp = np.asarray(o.gripper_pose, dtype=np.float64)
        pos[i] = gp[:3]
        q = gp[3:7]
        nq = np.linalg.norm(q)
        quat[i] = q / nq if nq > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
    return {"grip": grip, "speed": speed, "pos": pos, "quat": quat, "n": n}


def _pose_close(arr: dict, i: int, j: int) -> bool:
    """True when frames i and j are duplicates in BOTH position and rotation."""
    if np.linalg.norm(arr["pos"][i] - arr["pos"][j]) >= MERGE_EPS_M:
        return False
    dot = min(1.0, abs(float(np.dot(arr["quat"][i], arr["quat"][j]))))
    return 2.0 * np.degrees(np.arccos(dot)) < MERGE_EPS_DEG


def _split_keyframes(arr: dict, kfs: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split heuristic keyframes into gripper-event frames and stop frames.

    A keyframe is an event iff the gripper state flips relative to the
    PREVIOUS KEYFRAME's state (not the previous frame): the heuristic's final
    dedup can drop the exact flip frame in favour of the adjacent last frame,
    so the flip is only guaranteed to be visible at keyframe granularity.
    The plain last frame (no flip) is excluded — reappended at assembly.
    """
    grip = arr["grip"]
    n = arr["n"]
    events, stops = [], []
    prev = grip[0]
    for k in kfs:
        k = int(k)
        if grip[k] != prev:
            events.append(k)
        elif k != n - 1:
            stops.append(k)
        prev = grip[k]
    return events, stops


def _merge_stops(arr: dict, events: List[int], stops: List[int]) -> List[int]:
    """Drop stops whose pose is within MERGE_EPS_M of the previous kept
    keyframe (segment boundary or previously kept stop). Frame 0 counts as
    the first boundary, which removes the spurious frame-1 keyframe."""
    pos = arr["pos"]
    boundaries = [0] + list(events) + [arr["n"] - 1]
    kept = []
    for s in stops:
        left = 0
        for b in boundaries:
            if b < s:
                left = max(left, b)
        for p in kept:
            if p < s:
                left = max(left, p)
        if _pose_close(arr, s, left):
            continue
        kept.append(s)
    return kept


def _signature(arr: dict, events: List[int]) -> Tuple[int, ...]:
    return tuple(int(arr["grip"][e]) for e in events)


def _segment_bounds(arr: dict, events: List[int]) -> List[Tuple[int, int]]:
    """Segments: start->ev0, ev_i->ev_{i+1}, ev_last->last frame."""
    bounds = [0] + list(events) + [arr["n"] - 1]
    return list(zip(bounds[:-1], bounds[1:]))


def _seg_counts(arr: dict, events: List[int], stops: List[int]) -> Tuple[int, ...]:
    segs = _segment_bounds(arr, events)
    counts = []
    for a, b in segs:
        counts.append(sum(1 for s in stops if a < s < b))
    return tuple(counts)


def _local_minima(speed: np.ndarray, a: int, b: int) -> List[int]:
    """Indices of local minima of the smoothed speed profile strictly inside
    (a, b), sorted by speed ascending (deepest first)."""
    if b - a < 5:
        return []
    sm = speed.astype(np.float64).copy()
    sm[1:-1] = (speed[:-2] + speed[1:-1] + speed[2:]) / 3.0
    cand = []
    for i in range(max(a + 2, 1), min(b - 1, len(sm) - 1)):
        if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1]:
            cand.append(i)
    cand.sort(key=lambda i: (sm[i], i))
    return cand


def _remove_failed_grasp_pairs(arr: dict, events: List[int],
                               target_sig: Tuple[int, ...]) -> Optional[List[int]]:
    """Try to reach target_sig by deleting adjacent close->open event pairs
    whose displacement is < FAILED_GRASP_MAX_DISP_M (failed grasp retries).
    Returns the repaired event list or None if the target can't be reached."""
    grip, pos = arr["grip"], arr["pos"]
    events = list(events)
    while len(events) > len(target_sig):
        cands = []
        for j in range(len(events) - 1):
            e0, e1 = events[j], events[j + 1]
            if grip[e0] == 0.0 and grip[e1] == 1.0:  # close then open
                d = float(np.linalg.norm(pos[e1] - pos[e0]))
                if d < FAILED_GRASP_MAX_DISP_M:
                    cands.append((d, j))
        cands.sort()
        removed = False
        for _, j in cands:
            trial = events[:j] + events[j + 2:]
            if len(trial) >= len(target_sig):
                events = trial
                removed = True
                break
        if not removed:
            return None
    if _signature(arr, events) == target_sig:
        return events
    return None


def _repair_stops(arr: dict, events: List[int], stops: List[int],
                  target: Tuple[int, ...]) -> Tuple[List[int], bool]:
    """Adjust per-segment stop counts to the target template.
    Returns (stops, fully_repaired)."""
    speed, pos = arr["speed"], arr["pos"]
    segs = _segment_bounds(arr, events)
    if len(target) != len(segs):
        return stops, False
    out: List[int] = []
    ok = True
    for (a, b), k in zip(segs, target):
        seg_stops = sorted(s for s in stops if a < s < b)
        while len(seg_stops) > k:
            # Drop the most redundant stop: smallest combined position +
            # rotation distance to its left neighbour keyframe.
            def _left_dist(idx):
                s = seg_stops[idx]
                left = a if idx == 0 else seg_stops[idx - 1]
                d = float(np.linalg.norm(pos[s] - pos[left]))
                dot = min(1.0, abs(float(np.dot(arr["quat"][s], arr["quat"][left]))))
                return d + 0.1 * 2.0 * float(np.arccos(dot))
            drop = min(range(len(seg_stops)), key=_left_dist)
            seg_stops.pop(drop)
        if len(seg_stops) < k:
            existing = [a, b] + seg_stops
            for c in _local_minima(speed, a, b):
                if len(seg_stops) >= k:
                    break
                if any(abs(c - e) < MIN_STOP_SEP_FRAMES for e in existing):
                    continue
                left = max([e for e in existing if e < c], default=a)
                if _pose_close(arr, c, left):
                    continue
                seg_stops.append(c)
                seg_stops.sort()
                existing.append(c)
            if len(seg_stops) < k:
                ok = False
        out.extend(seg_stops)
    return sorted(out), ok


def _assemble(arr: dict, events: List[int], stops: List[int]) -> List[int]:
    """Final keyframe list: events + stops + last frame, mirroring the
    original heuristic's dedup of an event on the second-to-last frame."""
    n = arr["n"]
    kfs = sorted(set(int(x) for x in (list(events) + list(stops))))
    if kfs and kfs[-1] == n - 2:
        kfs.pop()  # matches keypoint_discovery's adjacent-final dedup
    if not kfs or kfs[-1] != n - 1:
        kfs.append(n - 1)
    return kfs


def canonicalize_task(
    episodes: Dict[int, dict],
) -> Dict[int, dict]:
    """Canonicalize all episodes of ONE task.

    Args:
        episodes: {ep_idx: {"arr": episode_arrays(...), "kfs": heuristic
            keyframes (no prepended 0), "variation": int}}

    Returns:
        {ep_idx: {"keyframes": [...], "conform": bool, "changed": bool,
                  "note": str}}
    """
    # ---- pass 1: normalize every episode --------------------------------
    norm = {}
    for ep, e in episodes.items():
        arr = e["arr"]
        events, stops = _split_keyframes(arr, e["kfs"])
        stops = _merge_stops(arr, events, stops)
        # Terminal dedup: the heuristic only merges a keyframe into the final
        # frame when they are EXACTLY adjacent; a release/stop that settles
        # 2-7 frames before the end leaves a pose-identical duplicate pair.
        # Relocate a terminal event onto the last frame (same post-event grip,
        # keeps the signature); drop a terminal stop outright.
        n = arr["n"]
        last_kf = max(events + stops, default=None)
        if (last_kf is not None and last_kf != n - 1
                and arr["grip"][last_kf] == arr["grip"][n - 1]
                and _pose_close(arr, last_kf, n - 1)):
            if events and last_kf == events[-1]:
                events[-1] = n - 1
            else:
                stops.remove(last_kf)
        norm[ep] = {
            "arr": arr,
            "events": events,
            "stops": stops,
            "sig": _signature(arr, events),
            "var": e["variation"],
            "orig": [int(k) for k in e["kfs"]],
        }

    # ---- pass 2: vote templates per variation ----------------------------
    by_var: Dict[int, List[int]] = {}
    for ep, e in norm.items():
        by_var.setdefault(e["var"], []).append(ep)

    templates: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for var, eps in by_var.items():
        sig_mode = Counter(norm[ep]["sig"] for ep in eps).most_common(1)[0][0]
        conf = [ep for ep in eps if norm[ep]["sig"] == sig_mode]
        cnt_mode = Counter(
            _seg_counts(norm[ep]["arr"], norm[ep]["events"], norm[ep]["stops"])
            for ep in conf
        ).most_common(1)[0][0]
        templates[var] = (sig_mode, cnt_mode)

    # Small-group fallback: adopt the unanimous template of same-signature
    # variations that have enough episodes (e.g. stack_blocks 1-episode vars).
    for var, eps in by_var.items():
        if len(eps) >= MIN_GROUP_FOR_VOTE:
            continue
        sig_mode = templates[var][0]
        family = {
            templates[v][1]
            for v, veps in by_var.items()
            if v != var and len(veps) >= MIN_GROUP_FOR_VOTE
            and templates[v][0] == sig_mode
        }
        if len(family) == 1:
            templates[var] = (sig_mode, next(iter(family)))

    # ---- pass 3: repair deviants -----------------------------------------
    out = {}
    for ep, e in norm.items():
        arr = e["arr"]
        sig_t, cnt_t = templates[e["var"]]
        events, note = e["events"], ""
        conform = True

        if e["sig"] != sig_t:
            repaired = None
            if len(e["sig"]) > len(sig_t):
                repaired = _remove_failed_grasp_pairs(arr, events, sig_t)
            if repaired is not None:
                events = repaired
                note = "removed failed-grasp event pair(s)"
            else:
                conform = False
                note = f"signature {e['sig']} != modal {sig_t}, unrepairable"

        stops = e["stops"]
        if conform:
            stops, ok = _repair_stops(arr, events, stops, cnt_t)
            if not ok:
                conform = False
                note = (note + "; " if note else "") + "stop-count repair incomplete"

        kfs = _assemble(arr, events, stops)
        out[ep] = {
            "keyframes": kfs,
            "conform": conform,
            "changed": kfs != e["orig"],
            "note": note,
        }
    return out
