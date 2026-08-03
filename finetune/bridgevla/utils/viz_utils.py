"""Shared training-time visualization helpers used by real/, GemBench/, RLBench/.

The per-bench ``visualize.py`` modules own the data-loading + forward pass
(each bench has its own ``_preprocess_inputs`` and augmentation pattern); the
primitives in this file take the forward's ``out`` / ``q_trans`` / ``action_trans``
tensors and write pred-vs-GT heatmap panels, a per-step meta file, and group
dataset indices into (task, episode) buckets so we can sample one full episode
per task.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


# stage name -> key into MVT.forward's `out` dict (rendered RGB for that stage).
_STAGE_INFO = {
    "mvt1": "mvt1_ori_img",
    "mvt2": "mvt2_ori_img",
}


def _make_overlay(gray_vals: np.ndarray, orig_img: np.ndarray,
                  alpha: float = 0.5,
                  vmin: float = None, vmax: float = None) -> np.ndarray:
    """Blend a single-channel heatmap onto an RGB image. Returns uint8 (H,W,3).

    When vmin/vmax are provided the color mapping uses a fixed scale so that
    multiple images share the same color-to-value correspondence.
    """
    import matplotlib.cm as cm
    hm = gray_vals.astype(np.float64)
    if vmin is None:
        vmin = hm.min()
    if vmax is None:
        vmax = hm.max()
    if vmax - vmin > 1e-8:
        hm = np.clip((hm - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        hm = np.zeros_like(hm)
    heatmap_rgb = (cm.jet(hm) * 255).astype(np.uint8)[..., :3]
    blended = (
        (1 - alpha) * orig_img.astype(np.float64)
        + alpha * heatmap_rgb.astype(np.float64)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _save_stage(
    out: dict,
    q_trans: torch.Tensor,
    action_trans: Optional[torch.Tensor],
    sample_idx: int,
    sample_dir: str,
    stage: str,
    nc: int,
    h: int,
    w: int,
    use_modified_focal_loss: bool = False,
) -> bool:
    """Save pred (and optionally pred|gt) viz for one stage ("mvt1" or "mvt2").

    The pred heatmap activation matches the training objective:
      - ``use_modified_focal_loss=False`` (default, original BridgeVLA):
        per-view softmax over (H*W). The loss is soft-label CE; at
        convergence the overlay should peak at one pixel.
      - ``use_modified_focal_loss=True`` (CenterNet path): per-pixel sigmoid.

    For each of the ``nc`` views saves:
      - ``original_{i}.png`` — rendered RGB (single image)
      - ``overlay_{i}.png``  — pred overlay (1×1) when ``action_trans=None``
        (eval), pred overlay | gt overlay (1×2) when GT is provided (train)
      - ``logits_{i}.png``   — pred logits matplotlib panel; the second
        (GT) panel is added only when ``action_trans`` is provided.

    Eval-mode callers pass ``action_trans=None`` so no GT artefacts are
    written (there is no ground truth at inference time).

    Returns True on success, False if the stage's rendered image is missing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if stage not in _STAGE_INFO:
        raise ValueError(f"unknown stage {stage!r}")
    img_key = _STAGE_INFO[stage]
    if img_key not in out:
        print(f"[viz] {img_key} missing from model output; skip {stage}")
        return False

    # Channel layout: stage_two concat is [mvt1 (nc chans), mvt2 (nc chans)].
    # When stage_two=False q_trans has shape (bs, h*w, nc) and the mvt2
    # ori_img isn't in ``out`` — so this branch is only reached for mvt1.
    if stage == "mvt1":
        sl = slice(0, nc)
    else:  # mvt2
        sl = slice(nc, 2 * nc)

    rgb = out[img_key][sample_idx, :, 3:6].float()  # (nc, 3, H, W) in [0,1]

    pred_raw = q_trans[sample_idx, :, sl].clone().view(h, w, nc).float()
    if use_modified_focal_loss:
        # CenterNet path: per-pixel sigmoid → independent Bernoulli probs.
        pred_hm = torch.sigmoid(pred_raw)
    else:
        # Original BridgeVLA: spatial softmax over (H*W) per view, matching
        # the soft-label CE loss in get_wpt. Output sums to 1 per view, so
        # at convergence the overlay peaks at one pixel.
        _flat = pred_raw.permute(2, 0, 1).reshape(nc, h * w)
        _sm = torch.softmax(_flat, dim=-1)
        pred_hm = _sm.view(nc, h, w).permute(1, 2, 0).contiguous()

    has_gt = action_trans is not None
    if has_gt:
        gt_raw = action_trans[sample_idx, :, sl].clone().view(h, w, nc).float()
        gt_hm = gt_raw

    save_dir = os.path.join(sample_dir, stage)
    os.makedirs(save_dir, exist_ok=True)

    color_imgs = rgb.cpu().numpy().transpose(0, 2, 3, 1)
    pred_gray = pred_hm.cpu().numpy().transpose(2, 0, 1)
    pred_logits = pred_raw.cpu().numpy().transpose(2, 0, 1)
    if has_gt:
        gt_gray = gt_hm.cpu().numpy().transpose(2, 0, 1)
        gt_logits = gt_raw.cpu().numpy().transpose(2, 0, 1)

    # Adaptive logits color scale: share min/max across all nc views so the
    # same color means the same value in every logits_{i}.png. The training
    # loss softmaxes each view independently, so absolute logit magnitudes
    # are NOT comparable across views — but the peak location is what we
    # eyeball, and a shared scale makes a high-confidence view visibly
    # brighter than a flatter one.
    logits_vmin = float(pred_logits.min())
    logits_vmax = float(pred_logits.max())
    if logits_vmax - logits_vmin < 1e-6:
        logits_vmax = logits_vmin + 1e-6

    # Per-view heatmap scale: each view's softmax already sums to 1, but the
    # peak height varies hugely with the distribution's sharpness (a sharp
    # peak ~0.3, a diffuse one ~1/(H*W)). Normalizing each view to its own
    # min/max makes every overlay use the full color range, so the peak
    # location is legible in every view. The "which view is more confident"
    # signal that a shared scale used to encode is dropped — read the
    # logits_{i}.png panels (shared scale) for that.

    for i in range(nc):
        orig = (np.clip(color_imgs[i], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(orig).save(os.path.join(save_dir, f"original_{i}.png"))

        pred_ov = _make_overlay(pred_gray[i], orig)
        if has_gt:
            gt_ov = _make_overlay(gt_gray[i], orig)
            combined_ov = np.concatenate([pred_ov, gt_ov], axis=1)
            Image.fromarray(combined_ov).save(
                os.path.join(save_dir, f"overlay_{i}.png")
            )
        else:
            Image.fromarray(pred_ov).save(
                os.path.join(save_dir, f"overlay_{i}.png")
            )

        p_l = pred_logits[i].astype(np.float64)

        # Title + GT colorbar scale depend on which training objective is
        # active. The default softmax-CE path supervises against a
        # gaussian-normalized-to-sum-1 target (peak ≈ 1/(2π·σ²) ≈ 0.07 for
        # σ=1.5), so a hardcoded ``vmax=1`` colorbar makes the GT look
        # uniformly black. The focal-loss path uses a peak-1 CenterNet
        # target, where ``vmax=1`` is correct.
        if use_modified_focal_loss:
            pred_subtitle = "logits → sigmoid (focal loss)"
            gt_subtitle = "centernet hm, peak=1"
            gt_vmin, gt_vmax = 0.0, 1.0
        else:
            pred_subtitle = "logits → spatial softmax (soft-label CE)"
            gt_subtitle = "softmax-norm gaussian, sum=1"
            gt_vmin = 0.0

        if has_gt:
            g_l = gt_logits[i].astype(np.float64)
            if not use_modified_focal_loss:
                gt_vmax = max(float(g_l.max()), 1e-6)
            fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, ax_p = plt.subplots(1, 1, figsize=(6, 5))

        im_p = ax_p.imshow(p_l, cmap="jet", vmin=logits_vmin, vmax=logits_vmax,
                           interpolation="nearest")
        ax_p.set_title(f"pred view {i} ({pred_subtitle})\n"
                       f"min={p_l.min():.3f}  max={p_l.max():.3f}")
        ax_p.set_axis_off()
        fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
        if has_gt:
            im_g = ax_g.imshow(g_l, cmap="jet", vmin=gt_vmin, vmax=gt_vmax,
                               interpolation="nearest")
            ax_g.set_title(f"gt view {i} ({gt_subtitle})\n"
                           f"min={g_l.min():.3f}  max={g_l.max():.3f}")
            ax_g.set_axis_off()
            fig.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"logits_{i}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    return True


def _save_mvt2_memory_grid(
    anchor_img_one,   # (V, 3, H, W) float in [0,1] or None
    current_img_one,  # (V, 3, H, W) float in [0,1]
    anchor_valid: bool,
    save_path: str,
    view_names: Sequence[str],
    current_hm_one=None,  # (V, H, W) float or None
    step_label: str = "",
) -> None:
    """Stage-2 memory grid: anchor (frame 0) and current rendered at the
    SAME stage-2 zoom-in cameras, so anchor / current alignment under the
    cube transform can be eyeballed view-by-view. Anchor row is blacked
    out when ``anchor_valid`` is False (e.g. step 0).

    When ``current_hm_one`` is given the grid has three rows:
    anchor / current original (no overlay) / current with pred overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = current_img_one.detach().cpu().float().clamp(0, 1).numpy()
    cur = (cur.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    cur_orig = cur.copy()
    if current_hm_one is not None:
        hm = current_hm_one.detach().cpu().float().numpy()
        if hm.ndim == 3 and hm.shape[0] != cur.shape[0] and hm.shape[-1] == cur.shape[0]:
            hm = hm.transpose(2, 0, 1)
        for v in range(min(cur.shape[0], hm.shape[0])):
            cur[v] = _make_overlay(hm[v], cur[v])
    if anchor_img_one is not None:
        anc = anchor_img_one.detach().cpu().float().clamp(0, 1).numpy()
        anc = (anc.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        if not anchor_valid:
            anc = np.zeros_like(anc)
    else:
        anc = np.zeros_like(cur)

    V = cur.shape[0]
    if current_hm_one is not None:
        panel = [anc, cur_orig, cur]
        row_labels = ["anchor (frame 0)", "current (original)",
                      "current (pred overlay)"]
    else:
        panel = [anc, cur]
        row_labels = ["anchor (frame 0)", "current"]
    rows = len(panel)

    fig, axes = plt.subplots(rows, V, figsize=(V * 3.0, rows * 3.0),
                             squeeze=False)
    for r in range(rows):
        for v in range(V):
            ax = axes[r][v]
            ax.imshow(panel[r][v])
            ax.set_xticks([]); ax.set_yticks([])
            if v == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
            if r == 0:
                ax.set_title(view_names[v] if v < len(view_names) else f"view {v}",
                             fontsize=10)
    fig.suptitle(f"mvt2 memory grid {step_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_anchor_frames(
    anchor_img_one,   # (V, 3, H, W) float in [0,1] or None
    anchor_valid: bool,
    save_dir: str,
    view_names: Sequence[str],
) -> None:
    """Save mvt2's per-view anchor (frame 0) render as individual PNGs.

    Companion to ``_save_mvt2_memory_grid``'s anchor row: that helper only
    ever writes the views stitched together into one grid PNG, this writes
    each view separately under ``save_dir`` (e.g. ``step4/anchor_frame/
    {view_name}.png``) so the anchor frame can be inspected view-by-view.
    Blacked out (zeros) when ``anchor_valid`` is False, mirroring the grid's
    gating. No-op when ``anchor_img_one`` is None.
    """
    if anchor_img_one is None:
        return
    anc = anchor_img_one.detach().cpu().float().clamp(0, 1).numpy()
    anc = (anc.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    if not anchor_valid:
        anc = np.zeros_like(anc)
    os.makedirs(save_dir, exist_ok=True)
    for v in range(anc.shape[0]):
        name = view_names[v] if v < len(view_names) else f"view_{v}"
        Image.fromarray(anc[v]).save(os.path.join(save_dir, f"{name}.png"))


def _save_memory_panel(panel_one_sample, save_path, view_names, step_label="",
                       row_labels=None):
    """Save one sample's memory panel as a (R, V) grid PNG.

    ``panel_one_sample`` has shape ``(R, V, H, W, 3)`` uint8. By default the
    rows are interpreted as [anchor, hist_0, ..., hist_{K-1}, current] with
    sliding-window (most-recent-first) labels. Pass ``row_labels`` (length R)
    to override — e.g. the RMBench keyframe-gated memory uses slot 0 = most
    recent admitted subtask-boundary keyframe.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, V, H, W, _ = panel_one_sample.shape
    K = rows - 2
    if row_labels is None:
        row_labels = (
            ["anchor (frame 0)"]
            + [f"hist k={k} (t-{k+1})" for k in range(K)]
            + ["current"]
        )

    fig, axes = plt.subplots(rows, V, figsize=(V * 3.0, rows * 3.0),
                             squeeze=False)
    for r in range(rows):
        for v in range(V):
            ax = axes[r][v]
            ax.imshow(panel_one_sample[r, v])
            ax.set_xticks([]); ax.set_yticks([])
            if v == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
            if r == 0:
                ax.set_title(view_names[v] if v < len(view_names) else f"view {v}",
                             fontsize=10)
    fig.suptitle(f"memory grid {step_label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _load_overlay(path: str) -> Optional[np.ndarray]:
    """Load an ``overlay_{i}.png`` (the pred|gt 1×2 panel) as uint8 (H,W,3).

    Returns None if the file is missing (e.g. that step's forward failed, or
    the stage wasn't rendered) so the caller can drop in a black placeholder.
    """
    if not os.path.isfile(path):
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def stitch_episode_overlays(
    episode_dir: str,
    step_labels: Sequence[str],
    stages: Sequence[str],
    n_views: int,
    view_names: Sequence[str],
    arms: Optional[Sequence[str]] = None,
    instruction: Optional[str] = None,
) -> None:
    """Stitch every step's tri-view overlays into one big grid per (arm, stage).

    Reads the per-step ``overlay_{i}.png`` panels that :func:`_save_stage`
    already wrote and lays them out as a single image:

        - rows  (vertical axis)   = keyframe steps
        - cols  (horizontal axis) = the ``n_views`` rendered views

    Source/output paths depend on whether the bench is dual-arm:

      * ``arms=None`` (single-arm: GemBench / memoryBench / real)::

            in : {episode_dir}/{step}/{stage}/overlay_{i}.png
            out: {episode_dir}/grid_{stage}.png

      * ``arms=["L", "R", ...]`` (dual-arm: RMBench)::

            in : {episode_dir}/{step}/arm_{A}/{stage}/overlay_{i}.png
            out: {episode_dir}/grid_arm_{A}_{stage}.png

    Missing cells are drawn black so a partially-failed episode still produces
    a grid; a (arm, stage) with zero overlays is skipped entirely.

    Uses the object-oriented matplotlib API (``Figure`` directly, no pyplot)
    so it is thread-safe: RMBench eval calls this from a background thread
    while the request-handler thread may be running pyplot-based per-step viz
    for the next episode.
    """
    from matplotlib.figure import Figure

    # Normalize to a list of (arm_or_None) so single/dual-arm share one loop.
    arm_iter = list(arms) if arms is not None else [None]

    for arm in arm_iter:
        for stage in stages:
            # Gather (step_label, [view0, view1, ...]) for every step that has
            # at least one overlay for this (arm, stage).
            rows = []
            for step in step_labels:
                if arm is not None:
                    stage_dir = os.path.join(episode_dir, step, f"arm_{arm}", stage)
                else:
                    stage_dir = os.path.join(episode_dir, step, stage)
                cells = [_load_overlay(os.path.join(stage_dir, f"overlay_{i}.png"))
                         for i in range(n_views)]
                if any(c is not None for c in cells):
                    rows.append((step, cells))
            if not rows:
                continue

            # Fallback shape for missing cells: copy the first real cell.
            ref = next(c for _, cells in rows for c in cells if c is not None)
            black = np.zeros_like(ref)

            n_rows = len(rows)
            fig = Figure(figsize=(n_views * 3.2, n_rows * 1.8))
            axes = fig.subplots(n_rows, n_views, squeeze=False)
            for r, (step, cells) in enumerate(rows):
                for v in range(n_views):
                    ax = axes[r][v]
                    img = cells[v] if cells[v] is not None else black
                    ax.imshow(img)
                    ax.set_xticks([]); ax.set_yticks([])
                    if v == 0:
                        ax.set_ylabel(step, fontsize=9)
                    if r == 0:
                        name = view_names[v] if v < len(view_names) else f"view {v}"
                        ax.set_title(f"{name}\n(pred | gt)", fontsize=9)
            arm_tag = f"arm {arm}  " if arm is not None else ""
            meta_line = (
                f"{os.path.basename(episode_dir)}  {arm_tag}{stage}  "
                f"(rows=steps, cols=views)"
            )
            # Prepend the episode's full instruction at the very top of the
            # caption (wrapped so long instructions don't run off the figure).
            if instruction:
                import textwrap
                instr_wrapped = "\n".join(textwrap.wrap(
                    f"instruction: {instruction}", width=max(40, n_views * 28)))
                title = f"{instr_wrapped}\n{meta_line}"
            else:
                title = meta_line
            fig.suptitle(title, fontsize=11)
            fig.tight_layout()
            if arm is not None:
                out_path = os.path.join(episode_dir, f"grid_arm_{arm}_{stage}.png")
            else:
                out_path = os.path.join(episode_dir, f"grid_{stage}.png")
            # Figure was never registered with pyplot, so no plt.close needed;
            # it is garbage-collected once this scope exits.
            fig.savefig(out_path, dpi=110, bbox_inches="tight")


def _fmt_grip(v) -> str:
    """Format grip label: 0=closed, 1=open (RMBench convention)."""
    i = int(v)
    en = "closed" if i == 0 else "open" if i == 1 else "unknown"
    return f"{i} ({en})"


def write_rmbench_gripper_txt(
    step_dir: str,
    *,
    left_pred: int,
    right_pred: int,
    left_gt: Optional[int] = None,
    right_gt: Optional[int] = None,
) -> None:
    """Write per-step dual-arm gripper GT/pred summary for RMBench viz.

    Eval callers pass only ``*_pred`` (no ground truth at inference).
    Training callers pass both GT and pred (next-step grip supervision
    vs model prediction).
    """
    os.makedirs(step_dir, exist_ok=True)
    lines = [
        "# Gripper open/close for the next action (0=closed, 1=open)",
        "",
    ]
    for arm_name, gt, pred in (
        ("left_arm", left_gt, left_pred),
        ("right_arm", right_gt, right_pred),
    ):
        lines.append(f"{arm_name}:")
        if gt is not None:
            lines.append(f"  gt_grip:   {_fmt_grip(gt)}")
        lines.append(f"  pred_grip: {_fmt_grip(pred)}")
        lines.append("")
    path = os.path.join(step_dir, "gripper.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_rmbench_zoom_pt_count_txt(
    step_dir: str,
    *,
    per_arm: list,
    st_sca: float,
    zoom_half_extent: float,
    threshold: int = 32,
    img_sizes_w=None,
) -> None:
    """Write per-step stage-2 zoom-cube point counts for RMBench eval.

    ``per_arm`` is a list of dicts with keys ``arm`` (str), ``count`` (int),
    ``used_fallback`` (bool). When ``used_fallback`` is True the agent used
    stage-1's waypoint (zoom-cube center) instead of stage-2's heatmap argmax
    for translation because the zoom cube had fewer than ``threshold`` points.
    """
    os.makedirs(step_dir, exist_ok=True)
    lines = [
        "# Number of point-cloud points inside the stage-2 zoom cube (per arm). "
        "FALLBACK_TO_STAGE1 = the translation fell back to the stage-1 waypoint (the zoom centre).",
        "",
        f"st_sca={st_sca}",
        f"renderer_img_sizes_w={img_sizes_w}",
        f"zoom_half_extent_cube={zoom_half_extent}",
        f"threshold={threshold}",
        "",
    ]
    for entry in per_arm:
        arm = entry["arm"]
        cnt = int(entry["count"])
        if entry.get("force_center"):
            tag = "FORCE_CENTER"
        else:
            tag = "FALLBACK_TO_STAGE1" if entry["used_fallback"] else "USE_MVT2"
        lines.append(f"{arm}:")
        lines.append(f"  count={cnt}")
        lines.append(f"  tag={tag}")
        lines.append("")
    path = os.path.join(step_dir, "zoom_pt_count.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_rmbench_memory_trigger_txt(
    step_dir: str,
    *,
    is_first_frame: bool,
    disc_enabled: bool,
    mem_logit: Optional[float],
    trigger_prob: Optional[float],
    threshold: float,
    gate: bool,
    bank_before: int,
    bank_after: int,
) -> None:
    """Write per-step keyframe-discriminator (memory trigger) prediction.

    Records exactly what the trigger predicted this step and whether the
    frame was admitted into temporal memory:
      * ``mem_logit``    — raw logit before sigmoid (None if discriminator off)
      * ``trigger_prob`` — sigmoid(logit), the admit probability
      * ``threshold``    — eval gate threshold (sigmoid > threshold -> admit)
      * ``gate``         — ADMIT / SKIP decision
      * ``bank_before/after`` — temporal memory size around the push, so a
        glance shows whether this step actually grew the memory.
    """
    os.makedirs(step_dir, exist_ok=True)
    _p = "n/a" if trigger_prob is None else f"{trigger_prob:.6f}"
    _l = "n/a" if mem_logit is None else f"{mem_logit:.6f}"
    lines = [
        "# Per-step prediction diagnostics of the memory trigger (keyframe discriminator)",
        "# gate=ADMIT: this frame is written into the temporal memory; gate=SKIP: not written",
        "",
        f"is_first_frame:        {bool(is_first_frame)}",
        f"discriminator_enabled: {bool(disc_enabled)}",
        f"mem_logit:             {_l}",
        f"trigger_prob:          {_p}",
        f"threshold:             {float(threshold):.6f}",
        f"gate:                  {'ADMIT' if gate else 'SKIP'}",
        f"bank_size_before:      {int(bank_before)}",
        f"bank_size_after:       {int(bank_after)}",
    ]
    path = os.path.join(step_dir, "memory_trigger.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _write_meta(sample_dir: str, meta: dict) -> None:
    # Force UTF-8: meta values include Greek/superscript chars (e.g. the
    # "raw log σ²" Kendall-Gal diagnostic key), which crash under the
    # default ASCII locale and abort the whole epoch's viz mid-loop.
    with open(os.path.join(sample_dir, "meta.txt"), "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


def group_dataset_by_episode(
    dataset,
    task_field: str,
    episode_field: str,
    step_field: str = "step_idx",
) -> Dict[str, Dict[str, List[int]]]:
    """Group ``dataset.index`` into {task: {episode: [sorted dataset_idx list]}}.

    Each ``dataset.index[i]`` must carry task / episode / step identifiers.
    The returned list is ordered by ``step_idx`` so callers can replay a
    whole trajectory in order.
    """
    if not hasattr(dataset, "index"):
        raise AttributeError(
            "dataset must expose `.index` (list of per-sample metadata dicts)"
        )
    groups: Dict[str, Dict[str, List[tuple]]] = {}
    for ds_idx, entry in enumerate(dataset.index):
        t = entry[task_field]
        e = entry[episode_field]
        s = int(entry[step_field])
        groups.setdefault(t, {}).setdefault(e, []).append((s, ds_idx))
    out: Dict[str, Dict[str, List[int]]] = {}
    for t, eps in groups.items():
        out[t] = {
            ep: [ds_idx for _, ds_idx in sorted(pairs, key=lambda p: p[0])]
            for ep, pairs in eps.items()
        }
    return out


def pick_one_episode_per_task(
    groups: Dict[str, Dict[str, List[int]]],
    rng: np.random.Generator,
) -> List[tuple]:
    """Given the groups dict from :func:`group_dataset_by_episode`, sample one
    episode per task. Returns ``[(task, episode, [dataset_idx, ...]), ...]``
    with tasks sorted alphabetically for determinism.
    """
    chosen: List[tuple] = []
    for task in sorted(groups.keys()):
        eps = sorted(groups[task].keys())
        if not eps:
            continue
        ep = eps[int(rng.integers(0, len(eps)))]
        chosen.append((task, ep, list(groups[task][ep])))
    return chosen
