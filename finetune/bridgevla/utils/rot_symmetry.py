"""Symmetry-aware target for the 6D rotation head.

Some objects look the same after a twist about the gripper's approach axis
(e.g. a cube every 90 deg, a cylinder after any twist), but every demo records
one arbitrary yaw. Visually identical states then carry 6D labels 360/k deg
apart, which makes the regression target ambiguous and hurts training. For a
sample of symmetry order k the loss is therefore taken against the single
equivalent target R_gt @ Rz(2*pi*j/k) whose heading lies in [0, 2*pi/k);
k = "inf" (round objects) supervises only the approach axis.
"""
import math

import torch

INF = "inf"


class RotSymmetryTable:
    """(task, step_idx, variation) -> symmetry order: int k >= 2, "inf", or 1 (none).

    ``rules`` is a list of ``{task, k, steps (optional), variations (optional)}``;
    the first matching rule wins.
    """

    def __init__(self, rules):
        self.rules = []
        for r in rules:
            if not {"task", "k"} <= set(r) <= {"task", "k", "steps", "variations"}:
                raise ValueError(f"bad rot_symmetry rule: {r}")
            k = INF if str(r["k"]).lower() == INF else int(r["k"])
            if k != INF and k < 2:
                raise ValueError(f"rot_symmetry rule {r}: k must be >= 2 or 'inf'")
            steps = None if r.get("steps") is None else frozenset(int(s) for s in r["steps"])
            variations = (None if r.get("variations") is None
                          else frozenset(int(v) for v in r["variations"]))
            self.rules.append((str(r["task"]), k, steps, variations))

    def order(self, task, step_idx, variation):
        for r_task, k, steps, variations in self.rules:
            if (r_task == task and (steps is None or step_idx in steps)
                    and (variations is None or variation in variations)):
                return k
        return 1

    def orders_for_batch(self, tasks, step_idx, variation):
        step_idx = torch.as_tensor(step_idx).reshape(-1).tolist()
        variation = torch.as_tensor(variation).reshape(-1).tolist()
        assert len(tasks) == len(step_idx) == len(variation)
        return [self.order(str(t), int(s), int(v)) for t, s, v in zip(tasks, step_idx, variation)]


def _rot_z(theta, ref):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                        device=ref.device, dtype=ref.dtype)


def symmetric_rotation_loss(R_pred, R_gt, orders):
    """Per-sample Frobenius^2 loss between (bs, 3, 3) rotations, where sample i
    has symmetry order ``orders[i]`` (1, int k or "inf"). Returns (bs,)."""
    if R_pred.dtype in (torch.float16, torch.bfloat16):
        R_pred = R_pred.float()
    R_gt = R_gt.to(R_pred)
    loss = R_pred.new_zeros(R_pred.shape[0])
    for k in dict.fromkeys(orders):
        idx = torch.tensor([i for i, o in enumerate(orders) if o == k], device=R_pred.device)
        Rp, Rg = R_pred[idx], R_gt[idx]
        if k == 1:
            loss[idx] = ((Rp - Rg) ** 2).sum(dim=(-1, -2))
        elif k == INF:
            # Only the approach axis (third column) is defined.
            loss[idx] = ((Rp[:, :, 2] - Rg[:, :, 2]) ** 2).sum(dim=-1)
        else:
            Rz = torch.stack([_rot_z(2.0 * math.pi * j / k, Rp) for j in range(k)], 0)
            cands = torch.einsum("nij,kjl->nkil", Rg, Rz)  # (n, k, 3, 3) = R_gt @ Rz_j
            d = ((Rp[:, None] - cands) ** 2).sum(dim=(-1, -2))  # (n, k)
            # Canonical member: heading atan2(R[1, 0], R[0, 0]) closest to pi/k, i.e. in [0, 2*pi/k).
            yaw = torch.atan2(cands[..., 1, 0], cands[..., 0, 0])
            off = torch.remainder(yaw - math.pi / k + math.pi, 2 * math.pi) - math.pi
            j = off.abs().argmin(dim=1)
            loss[idx] = d.gather(1, j[:, None])[:, 0]
    return loss
