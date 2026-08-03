"""Per-step RMBench eval diagnostics (client-side, no torch / bridgevla).

Written under ``viz/episode{E}/step_{S}/`` alongside gripper.txt:
  * ``action_pose.txt``  — model output + pre-step robot EE/TCP (deploy_policy)
  * ``plan_status.txt``  — Curobo plan result + target vs current (_base_task)
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np


def _fmt_vec3(v) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"[{v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}]"


def _fmt_quat_wxyz(q) -> str:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return f"[w={q[0]:+.6f}, x={q[1]:+.6f}, y={q[2]:+.6f}, z={q[3]:+.6f}]"


def _fmt_quat_xyzw(q) -> str:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return f"[x={q[0]:+.6f}, y={q[1]:+.6f}, z={q[2]:+.6f}, w={q[3]:+.6f}]"


def _pos_delta_m(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3]
                                - np.asarray(b, dtype=np.float64)[:3]))


def write_action_pose_txt(
    step_dir: str,
    *,
    left_server: Sequence[float],
    right_server: Sequence[float],
    action_16d: Sequence[float],
    left_ee_now: Optional[Sequence[float]] = None,
    right_ee_now: Optional[Sequence[float]] = None,
    left_tcp_now: Optional[Sequence[float]] = None,
    right_tcp_now: Optional[Sequence[float]] = None,
) -> None:
    """Log server RPC output and the 16-D ee action sent to ``take_action``."""
    os.makedirs(step_dir, exist_ok=True)
    left_server = np.asarray(left_server, dtype=np.float64).reshape(-1)
    right_server = np.asarray(right_server, dtype=np.float64).reshape(-1)
    action_16d = np.asarray(action_16d, dtype=np.float64).reshape(-1)

    lines = [
        "# This step's model output and the 16-D action handed to take_action('ee') (at decision time, before motion).",
        "# The server returns: EE-link xyz + quat_xyzw + grip (the TCP->EE conversion happens server-side).",
        "# action_16d: [Lxyz, Lquat_wxyz, Lgrip, Rxyz, Rquat_wxyz, Rgrip].",
        "",
    ]

    def _arm_block(name, server_8, ee_now, tcp_now, action_slice):
        lines.append(f"{name}:")
        lines.append(f"  server_xyz:     {_fmt_vec3(server_8[0:3])}")
        lines.append(f"  server_quat_xyzw: {_fmt_quat_xyzw(server_8[3:7])}")
        lines.append(f"  server_grip:    {int(round(server_8[7]))}")
        if action_slice is not None and len(action_slice) >= 8:
            act = np.asarray(action_slice, dtype=np.float64)
            lines.append(f"  action_xyz:     {_fmt_vec3(act[0:3])}")
            lines.append(f"  action_quat_wxyz: {_fmt_quat_wxyz(act[3:7])}")
            lines.append(f"  action_grip:    {int(round(act[7]))}")
        if ee_now is not None and len(ee_now) >= 7:
            ee = np.asarray(ee_now, dtype=np.float64)
            lines.append(f"  current_ee_xyz: {_fmt_vec3(ee[0:3])}")
            lines.append(f"  current_ee_quat_wxyz: {_fmt_quat_wxyz(ee[3:7])}")
            if action_slice is not None and len(action_slice) >= 3:
                lines.append(
                    f"  |action_xyz - current_ee_xyz| = "
                    f"{_pos_delta_m(action_slice[0:3], ee[0:3]):.6f} m"
                )
        if tcp_now is not None and len(tcp_now) >= 7:
            tcp = np.asarray(tcp_now, dtype=np.float64)
            lines.append(f"  current_tcp_xyz: {_fmt_vec3(tcp[0:3])}")
            lines.append(f"  current_tcp_quat_wxyz: {_fmt_quat_wxyz(tcp[3:7])}")
        lines.append("")

    _arm_block(
        "left_arm", left_server, left_ee_now, left_tcp_now,
        action_16d[0:8] if action_16d.size >= 8 else None,
    )
    _arm_block(
        "right_arm", right_server, right_ee_now, right_tcp_now,
        action_16d[8:16] if action_16d.size >= 16 else None,
    )

    path = os.path.join(step_dir, "action_pose.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_plan_status_txt(
    step_dir: str,
    *,
    per_arm: List[dict],
    gripper_after_motion: bool = False,
) -> None:
    """Log Curobo ``plan_path`` outcome per arm.

    Each ``per_arm`` entry should include:
      arm, target_7d (xyz+quat_wxyz), current_ee_7d, current_tcp_7d (optional),
      plan_status, n_step, topp_flag, grip_target, grip_current,
      planned_end_xyz (optional, last waypoint EE xyz if Success).
    """
    os.makedirs(step_dir, exist_ok=True)
    lines = [
        "# Result of Curobo plan_path (inside take_action, before motion).",
        "# plan_status=Fail -> topp_flag=False, and the control loop never calls set_arm_joints (the arm stays put).",
        "# n_step = number of trajectory points on Success; on Fail a fixed 50 physics steps are burned.",
        f"gripper_after_motion: {bool(gripper_after_motion)}",
        "",
    ]
    for entry in per_arm:
        arm = entry["arm"]
        tgt = np.asarray(entry["target_7d"], dtype=np.float64).reshape(-1)
        lines.append(f"{arm}:")
        lines.append(f"  target_xyz:     {_fmt_vec3(tgt[0:3])}")
        lines.append(f"  target_quat_wxyz: {_fmt_quat_wxyz(tgt[3:7])}")
        ee = entry.get("current_ee_7d")
        if ee is not None:
            ee = np.asarray(ee, dtype=np.float64).reshape(-1)
            lines.append(f"  current_ee_xyz: {_fmt_vec3(ee[0:3])}")
            lines.append(f"  current_ee_quat_wxyz: {_fmt_quat_wxyz(ee[3:7])}")
            lines.append(
                f"  |target - current_ee| = {_pos_delta_m(tgt[0:3], ee[0:3]):.6f} m"
            )
        tcp = entry.get("current_tcp_7d")
        if tcp is not None:
            tcp = np.asarray(tcp, dtype=np.float64).reshape(-1)
            lines.append(f"  current_tcp_xyz: {_fmt_vec3(tcp[0:3])}")
        status = str(entry.get("plan_status", "?"))
        n_step = int(entry.get("n_step", -1))
        topp = bool(entry.get("topp_flag", False))
        tag = "EXECUTE" if (status == "Success" and topp and n_step > 0) else "NO_ARM_MOTION"
        if status == "Success" and n_step <= 1:
            tag = "ZERO_OR_TINY_TRAJ"
        if status != "Success":
            tag = "PLAN_FAIL"
        lines.append(f"  plan_status:    {status}")
        lines.append(f"  n_step:         {n_step}")
        lines.append(f"  topp_flag:      {topp}")
        lines.append(f"  tag:            {tag}")
        # Concrete diagnosis when planning failed (from CuroboPlanner.plan_path): curobo's wording +
        # the IK probe category (out of workspace / unreachable gripper orientation / path collision) + residuals, for post-hoc debugging.
        if status != "Success":
            if entry.get("fail_reason") is not None:
                lines.append(f"  fail_reason:    {entry['fail_reason']}")
            if entry.get("ik_diag") is not None:
                lines.append(f"  ik_diag:        {entry['ik_diag']}")
            if entry.get("valid_query") is not None:
                lines.append(f"  valid_query:    {entry['valid_query']}")
            if entry.get("full_ik_pos_err") is not None:
                lines.append(f"  full_ik_pos_err:{entry['full_ik_pos_err']}")
            if entry.get("full_ik_rot_err") is not None:
                lines.append(f"  full_ik_rot_err:{entry['full_ik_rot_err']}")
        if entry.get("planned_end_xyz") is not None:
            lines.append(f"  planned_end_xyz: {_fmt_vec3(entry['planned_end_xyz'])}")
        g_tgt = entry.get("grip_target")
        g_cur = entry.get("grip_current")
        if g_tgt is not None and g_cur is not None:
            lines.append(f"  grip_current:   {float(g_cur):.4f}")
            lines.append(f"  grip_target:    {float(g_tgt):.4f}")
        lines.append("")

    path = os.path.join(step_dir, "plan_status.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
