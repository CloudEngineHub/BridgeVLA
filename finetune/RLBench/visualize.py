"""Training-time visualization for RLBench.

Mirrors ``GemBench/visualize.py`` 1:1 (same agent, same MVT renderer, same
memory-panel + mvt2 anchor-grid logic) — the only deltas are:

* Samples come from ``RLBench_Keyframe_Dataset`` (the SAME dataset the
  trainer consumes), so the viz forward sees exactly the training sample
  layout, including the episodic-memory bundle (anchor + hist slots).
  The old raw-demo loader (get_stored_demo + keypoint_discovery per viz
  call) was dropped — it duplicated keyframe logic and carried no memory.
* Preprocessing reuses ``_preprocess_inputs_gembench`` — the exact function
  ``agent.update_gembench`` runs at train time (the keyframe dataset emits
  GemBench-layout samples), so viz and training share one code path.
* RLBench's ``dataset.index`` keys episodes by ``ep_idx`` (int) rather than
  ``ep_key`` (str). We pass ``episode_field="ep_idx"`` and format the int
  into a directory name when laying out the per-step viz tree.

Output layout::

    {log_dir}/viz/epoch_{epoch:04d}/
        {task}/episode{ep_idx:04d}/step_{k:03d}/
            memory_grid.png              (stage-1: anchor + K hist + current)
            mvt1/...  mvt2/...           (pred+GT overlays / logits)
            mvt2/anchor_memory_grid.png  (stage-2: anchor + current @ zoom)
        {task}/episode{ep_idx:04d}/grid_{stage}.png   (stitched episode grid)
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from bridgevla.mvt.augmentation import apply_se3_aug_con
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.utils.viz_utils import (
    _save_memory_panel,
    _save_mvt2_memory_grid,
    _save_stage,
    _write_meta,
    group_dataset_by_episode,
    pick_one_episode_per_task,
    stitch_episode_overlays,
)

from utils.peract_utils_rlbench import CAMERAS as RLBENCH_CAMERAS
# The keyframe dataset emits GemBench-layout samples ({cam: {rgb, pcd}} +
# anchor_/hist{k}_ prefixes); use the SAME preprocessor the agent's
# update_gembench / _build_memory_inputs_from_replay run at train time.
from GemBench.utils.peract_utils_gembench import _preprocess_inputs_gembench


# PaliGemma bf16 attention kernel FPEs on H20 with bs < 4. Pad the forward
# batch up to this minimum by duplicating the first sample; duplicates are
# never written to disk.
MIN_FORWARD_BS = 4


# tensor/dict helpers

def _move_to_device(x, device):
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(a))
    if t.dtype == torch.float64:
        t = t.float()
    return t


def _collate_samples(samples: List[dict]) -> dict:
    """Stack a list of RLBench_Keyframe_Dataset samples into a batch dict."""
    assert len(samples) > 0
    keys = samples[0].keys()
    out: dict = {}
    for k in keys:
        v0 = samples[0][k]
        if isinstance(v0, dict):
            out[k] = {
                sub_k: _to_tensor(np.stack([s[k][sub_k] for s in samples], axis=0))
                for sub_k in v0
            }
        elif isinstance(v0, np.ndarray):
            out[k] = _to_tensor(np.stack([s[k] for s in samples], axis=0))
        elif isinstance(v0, (int, float, np.floating, np.integer)):
            out[k] = torch.tensor([float(s[k]) for s in samples])
        else:
            out[k] = [s[k] for s in samples]
    return out


# forward pass

@torch.no_grad()
def _run_viz_forward(
    agent,
    samples: List[dict],
    cameras: Sequence[str],
    apply_aug: bool = False,
) -> Tuple[dict, torch.Tensor, torch.Tensor, int, int, int]:
    """Run a single eval-mode forward through ``agent._net_mod`` and return
    the tensors the visualization layer consumes.
    """
    num_real = len(samples)
    assert num_real > 0, "need at least one sample"

    forward_bs = max(num_real, MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - num_real)

    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)
    batch["lang_goal"] = [[[s["lang_goal"]]] for s in padded]
    batch["tasks"] = [s["tasks"] for s in padded]

    action_gripper_pose = batch["gripper_pose"].float()  # (bs, 8)
    action_trans_con = action_gripper_pose[:, 0:3]
    action_rot = action_gripper_pose[:, 3:7]

    obs, pcd = _preprocess_inputs_gembench(batch, list(cameras))
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)

    # Memory bundle (anchor + history) co-augmented with current pc so
    # stage 1's anchor / hist tokens and stage 2's anchor zoom render
    # land in the same frame as the current pc.
    mem_intermediate = None
    if (getattr(agent, "memory_enabled", False)
            and "anchor_mask" in padded[0]):
        K = int((getattr(agent._net_mod, "memory_cfg", {}) or {})
                .get("k_temporal", 2))
        mem_intermediate = agent._build_memory_inputs_from_replay(
            batch, list(cameras), K=K,
        )

    if apply_aug and getattr(agent, "_transform_augmentation", False):
        if mem_intermediate is not None:
            extras_for_aug = (
                [mem_intermediate["anchor_pc_batched"]]
                + list(mem_intermediate["hist_pc_batched"])
            )
            action_trans_con_np, action_rot_np, pc, perturbed_extras = (
                apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(agent.scene_bounds),
                    trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
                    extra_pcds=extras_for_aug,
                )
            )
            mem_intermediate["anchor_pc_batched"] = perturbed_extras[0]
            mem_intermediate["hist_pc_batched"] = perturbed_extras[1:]
        else:
            action_trans_con_np, action_rot_np, pc = apply_se3_aug_con(
                pcd=pc,
                action_gripper_pose=action_gripper_pose,
                bounds=torch.tensor(agent.scene_bounds),
                trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
                rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
            )
        action_trans_con = torch.tensor(action_trans_con_np).to(pc.device)
        action_rot = torch.tensor(action_rot_np).to(pc.device)

    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound,
    )

    memory_inputs = (
        agent._finalize_memory_inputs(mem_intermediate, pc)
        if mem_intermediate is not None else None
    )
    wpt = [x[:3] for x in action_trans_con]

    wpt_local_list = []
    for _pc, _wpt in zip(pc, wpt):
        a, _ = mvt_utils.place_pc_in_cube(
            _pc, _wpt,
            with_mean_or_bounds=agent._place_with_mean,
            scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
        )
        if getattr(agent, "align_real_frame", False):
            a = a.clone()
            a[..., 0:2] = -a[..., 0:2]
        wpt_local_list.append(a.unsqueeze(0))
    wpt_local = torch.cat(wpt_local_list, dim=0)

    pc = [mvt_utils.place_pc_in_cube(
        _pc,
        with_mean_or_bounds=agent._place_with_mean,
        scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
    )[0] for _pc in pc]
    if getattr(agent, "align_real_frame", False):
        pc = [_pc.clone() for _pc in pc]
        for _pc in pc:
            _pc[..., 0:2] = -_pc[..., 0:2]

    bs = len(pc)
    nc = agent._net_mod.num_img
    h = w = agent._net_mod.img_size

    # viz_force_gt_stage2=True: make mvt2 zoom into the GT waypoint (+noise)
    # the exact same way training does, instead of mvt1's own prediction.
    # This requires passing GT `wpt_local` here even though we're in eval mode.
    # Viz is still fully @torch.no_grad() + agent.eval(), so nothing trains.
    out = agent._net_mod(
        pc=pc,
        img_feat=img_feat,
        lang_emb=None,
        img_aug=0,
        wpt_local=wpt_local,
        rot_x_y=None,
        language_goal=batch["lang_goal"],
        viz_force_gt_stage2=True,
        memory_inputs=memory_inputs,
    )
    if memory_inputs is not None:
        out["_batch_anchor_mask"] = memory_inputs["anchor_mask"].detach()
    q_trans, _, _, _, _, _pts = agent.get_q(out, dims=(bs, nc, h, w))
    action_trans = agent.get_action_trans(
        wpt_local, _pts, out, None, dims=(bs, nc, h, w),
    )

    out["_batch_gripper_pose_orig"] = batch["gripper_pose"].float()
    aug_pose = torch.cat([action_trans_con, action_rot,
                          batch["gripper_pose"][:, 7:].float()], dim=-1)
    out["_batch_gripper_pose"] = aug_pose
    out["_batch_lang_goal"] = [s["lang_goal"] for s in padded]
    out["_batch_tasks"] = [s["tasks"] for s in padded]
    return out, q_trans, action_trans, bs, h, w


# memory-debug panel

@torch.no_grad()
def _render_memory_panel(agent, samples, cameras, apply_aug=True):
    """Render the per-step memory bundle the model attends to at stage 1:
    [anchor (frame 0), hist_0, ..., hist_{K-1}, current], for each sample
    in ``samples``. Uses the SAME pipeline as the real training forward
    inside ``_update_shared``:

      get_pc_img_feat (batched)
      -> apply_se3_aug_con(extra_pcds=[anchor, hist_0, ..., hist_{K-1}])  [if apply_aug]
      -> move_pc_in_bound (batched -> list)
      -> place_pc_in_cube(pc=current, app_pc=anchor)  [cube projection]
      -> mvt1 renderer (fixed cameras)

    Anchor / history are co-rotated with current under the SAME random
    SE3, so what you see in the panel is exactly what PaliGemma sees at
    training time (post-augmentation). Pass ``apply_aug=False`` to render
    the un-augmented bundle for cleaner debug.

    No img_aug (img_aug=0 on every render) — alignment-only debug.

    Returns:
        np.ndarray of shape (num_samples, K+2, num_views, H, W, 3) uint8.
        Slots gated off by anchor_mask / hist_mask (e.g. step 0 anchor,
        step t with t < k+1 history) are filled with black.
    """
    if not getattr(agent, "memory_enabled", False):
        return None
    if "anchor_mask" not in samples[0]:
        return None  # dataset wasn't built with memory_enabled=True

    forward_bs = max(len(samples), MIN_FORWARD_BS)
    padded = samples + [samples[0]] * (forward_bs - len(samples))
    batch = _collate_samples(padded)
    batch = _move_to_device(batch, agent._device)

    # Current preprocessing (batched).
    obs, pcd = _preprocess_inputs_gembench(batch, list(cameras))
    pc_curr, img_feat_curr = rvt_utils.get_pc_img_feat(obs, pcd)

    # Build batched intermediate for memory bundle (matches _update_shared).
    K = int((getattr(agent._net_mod, "memory_cfg", {}) or {}).get("k_temporal", 2))
    mem_intermediate = agent._build_memory_inputs_from_replay(
        batch, list(cameras), K=K,
    )
    if mem_intermediate is None:
        return None

    # SE3 augmentation: same transform applied to current + extras so
    # the panel reflects what the network sees at training time.
    if apply_aug and getattr(agent, "_transform_augmentation", False):
        action_gripper_pose = batch["gripper_pose"].float()
        extras_for_aug = (
            [mem_intermediate["anchor_pc_batched"]]
            + list(mem_intermediate["hist_pc_batched"])
        )
        _atc, _arq, pc_curr, perturbed_extras = apply_se3_aug_con(
            pcd=pc_curr,
            action_gripper_pose=action_gripper_pose,
            bounds=torch.tensor(agent.scene_bounds),
            trans_aug_range=agent._transform_augmentation_xyz.clone().detach(),
            rot_aug_range=torch.tensor(agent._transform_augmentation_rpy),
            extra_pcds=extras_for_aug,
        )
        mem_intermediate["anchor_pc_batched"] = perturbed_extras[0]
        mem_intermediate["hist_pc_batched"] = perturbed_extras[1:]

    # Batched -> per-sample list (move_pc_in_bound), then cube-projection
    # of memory PCs into current's cube (via app_pc).
    pc_curr, img_feat_curr = rvt_utils.move_pc_in_bound(
        pc_curr, img_feat_curr, agent.scene_bounds,
        no_op=not agent.move_pc_in_bound,
    )
    memory_inputs = agent._finalize_memory_inputs(mem_intermediate, pc_curr)

    # Current cube projection (matches MVT.forward).
    pc_curr_cube = [
        mvt_utils.place_pc_in_cube(
            _pc, with_mean_or_bounds=agent._place_with_mean,
            scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
        )[0]
        for _pc in pc_curr
    ]
    if getattr(agent, "align_real_frame", False):
        pc_curr_cube = [_pc.clone() for _pc in pc_curr_cube]
        for _pc in pc_curr_cube:
            _pc[..., 0:2] = -_pc[..., 0:2]

    # Render through the same renderer the model uses (mvt1, fixed cameras).
    mvt = agent._net_mod
    img_curr = mvt.render(
        pc=pc_curr_cube, img_feat=img_feat_curr,
        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
    )
    img_anchor = mvt.render(
        pc=memory_inputs["anchor_pc"],
        img_feat=memory_inputs["anchor_img_feat"],
        img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
    )
    img_hists = [
        mvt.render(
            pc=memory_inputs["hist_pc"][k],
            img_feat=memory_inputs["hist_img_feat"][k],
            img_aug=0, mvt1_or_mvt2=True, dyn_cam_info=None,
        )
        for k in range(K)
    ]

    # Slice RGB (channels 3:6) and zero-out masked rows.
    def _rgb(im):  # (bs, V, channels, H, W) -> (bs, V, 3, H, W)
        return im[:, :, 3:6, :, :].clamp(0, 1)

    anchor_mask = memory_inputs["anchor_mask"].view(-1, 1, 1, 1, 1)
    hist_mask = memory_inputs["hist_mask"]  # (bs, K) bool

    anchor_rgb = _rgb(img_anchor) * anchor_mask.to(torch.float32)
    hist_rgbs = [
        _rgb(img_hists[k]) * hist_mask[:, k].view(-1, 1, 1, 1, 1).to(torch.float32)
        for k in range(K)
    ]
    curr_rgb = _rgb(img_curr)  # current always shown (never masked)

    # (bs, K+2, V, 3, H, W) -> (bs, K+2, V, H, W, 3) uint8
    rows = [anchor_rgb] + hist_rgbs + [curr_rgb]
    panel = torch.stack(rows, dim=1).permute(0, 1, 2, 4, 5, 3).contiguous()
    panel = (panel.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return panel[: len(samples)]


# public entry points

@torch.no_grad()
def visualize_samples(
    agent,
    samples: List[dict],
    save_dirs: List[str],
    stages: Sequence[str] = ("mvt1",),
    cameras: Sequence[str] = RLBENCH_CAMERAS,
    extra_meta: Optional[List[dict]] = None,
    apply_aug: bool = False,
    forward_chunk: int = MIN_FORWARD_BS,
) -> None:
    assert len(samples) == len(save_dirs), "samples and save_dirs length mismatch"
    num_real = len(samples)
    if num_real == 0:
        return

    chunk = max(1, int(forward_chunk))

    was_training = agent._network.training
    agent.eval()
    try:
        for chunk_start in range(0, num_real, chunk):
            chunk_end = min(chunk_start + chunk, num_real)
            chunk_samples = samples[chunk_start:chunk_end]
            chunk_dirs = save_dirs[chunk_start:chunk_end]
            chunk_meta = (
                extra_meta[chunk_start:chunk_end]
                if extra_meta is not None else None
            )

            out, q_trans, action_trans, _bs, h, w = _run_viz_forward(
                agent, chunk_samples, cameras, apply_aug=apply_aug,
            )
            nc = agent._net_mod.num_img

            # Per-step memory debug panel (anchor + K hist + current at
            # stage-1 cameras). Returns None when the dataset / agent
            # were built with memory disabled, in which case we skip.
            # Apply SE3 aug if the surrounding viz forward did, so the
            # panel reflects what the network actually sees at training.
            mem_panel = _render_memory_panel(
                agent, chunk_samples, cameras, apply_aug=apply_aug,
            )
            # Pick view names matching the renderer (fixed three-view
            # convention; oblique mode renames the same three slots).
            _renderer = getattr(agent._net_mod, "renderer", None)
            if getattr(_renderer, "oblique_views", False):
                _view_names = ["oblique_a", "oblique_b", "oblique_c"]
            else:
                _view_names = ["top", "front", "right"]

            # Per-sample stage-1 Kendall-Gal diagnostics (if the network
            # exposes view_logvar — only stage 1 does). s_i is the raw log σ²
            # that enters the training loss; softmax_p is what the eval-time
            # stage-1 fusion uses (see bridgevla_agent.get_pred).
            view_logvar_all = None
            view_softmax_all = None
            if isinstance(out.get("view_logvar"), torch.Tensor):
                _lv = out["view_logvar"].detach().cpu().float()       # (bs, num_img)
                view_logvar_all = _lv.numpy()
                view_softmax_all = torch.softmax(-_lv, dim=-1).numpy()

            for i in range(len(chunk_samples)):
                sample_dir = chunk_dirs[i]
                os.makedirs(sample_dir, exist_ok=True)
                _use_focal = bool(getattr(agent, "use_modified_focal_loss", False))
                for stage in stages:
                    _save_stage(out, q_trans, action_trans, i, sample_dir, stage,
                                nc=nc, h=h, w=w,
                                use_modified_focal_loss=_use_focal)

                # Memory debug grid: anchor + hist + current, all rendered
                # under current's place_pc_in_cube transform via app_pc so
                # alignment can be eyeballed across the rows.
                if mem_panel is not None:
                    _step_lbl = ""
                    if chunk_meta is not None and i < len(chunk_meta):
                        _step_lbl = (
                            f"(step {chunk_meta[i].get('step_idx', i)} / "
                            f"{chunk_meta[i].get('num_steps', '?')})"
                        )
                    _save_memory_panel(
                        mem_panel[i],
                        os.path.join(sample_dir, "memory_grid.png"),
                        view_names=_view_names,
                        step_label=_step_lbl,
                    )

                # Stage-2 memory grid: anchor (frame 0) + current re-rendered
                # at the SAME zoomed-in stage-2 cameras, with the current row
                # overlaid by mvt2's predicted heatmap. Stage 2 has no temporal
                # block, only the spatial anchor — so two rows is the full
                # picture. Saved as {step}/mvt2/anchor_memory_grid.png so it
                # sits next to that step's mvt2 overlay_/logits_ panels,
                # matching the GemBench train-viz layout. Only emitted when
                # memory is active (out carries the anchor render) and stage 2
                # is being visualized.
                if (
                    "mvt2" in stages
                    and "mvt2_ori_img" in out
                    and "mvt2_anchor_ori_img" in out
                ):
                    _step_lbl = ""
                    if chunk_meta is not None and i < len(chunk_meta):
                        _step_lbl = (
                            f"(step {chunk_meta[i].get('step_idx', i)} / "
                            f"{chunk_meta[i].get('num_steps', '?')})"
                        )
                    _anchor_valid = True
                    if "_batch_anchor_mask" in out:
                        _anchor_valid = bool(
                            out["_batch_anchor_mask"][i].item()
                        )
                    # mvt2 predicted heatmap for the current row. Post-
                    # activation matches the training objective (per-view
                    # sigmoid for the focal path, per-view spatial softmax
                    # otherwise), mirroring _save_stage so the overlay is
                    # directly comparable.
                    _mvt2_raw = q_trans[i, :, nc:2 * nc].clone().view(
                        h, w, nc
                    ).float()
                    if _use_focal:
                        _mvt2_hm = torch.sigmoid(_mvt2_raw).permute(
                            2, 0, 1
                        ).contiguous()
                    else:
                        _flat = _mvt2_raw.permute(2, 0, 1).reshape(nc, h * w)
                        _mvt2_hm = torch.softmax(_flat, dim=-1).view(nc, h, w)
                    _mvt2_dir = os.path.join(sample_dir, "mvt2")
                    os.makedirs(_mvt2_dir, exist_ok=True)
                    _save_mvt2_memory_grid(
                        anchor_img_one=out["mvt2_anchor_ori_img"][i, :, 3:6],
                        current_img_one=out["mvt2_ori_img"][i, :, 3:6],
                        current_hm_one=_mvt2_hm,
                        anchor_valid=_anchor_valid,
                        save_path=os.path.join(
                            _mvt2_dir, "anchor_memory_grid.png"
                        ),
                        view_names=_view_names,
                        step_label=_step_lbl,
                    )

                gp = out["_batch_gripper_pose"][i].cpu().tolist()
                gp_orig = out["_batch_gripper_pose_orig"][i].cpu().tolist()
                meta = {
                    "lang_goal":    out["_batch_lang_goal"][i],
                    "task":         out["_batch_tasks"][i],
                    "gt_xyz_m (aug)":     gp[0:3],
                    "gt_quat_xyzw (aug)": gp[3:7],
                    "gt_xyz_m (orig)":     gp_orig[0:3],
                    "gt_quat_xyzw (orig)": gp_orig[3:7],
                    "gt_claw":      gp[7],
                    "augmentation": getattr(agent, "_transform_augmentation", False),
                    "stages":       list(stages),
                }
                if view_logvar_all is not None:
                    meta["mvt1_view_s_i (raw log σ²)"] = [
                        round(float(x), 4) for x in view_logvar_all[i].tolist()
                    ]
                    meta["mvt1_view_softmax_p"] = [
                        round(float(x), 4) for x in view_softmax_all[i].tolist()
                    ]
                if chunk_meta is not None and i < len(chunk_meta):
                    meta.update(chunk_meta[i])
                _write_meta(sample_dir, meta)
    finally:
        if was_training:
            agent.train()


@torch.no_grad()
def visualize_epoch(
    agent,
    dataset,
    epoch: int,
    log_dir: str,
    cameras: Sequence[str] = RLBENCH_CAMERAS,
    seed: Optional[int] = None,
    stages: Sequence[str] = ("mvt1",),
    forward_chunk: int = MIN_FORWARD_BS,
    tasks: Optional[Sequence[str]] = None,
) -> None:
    """Sample ONE episode per task and dump pred+GT viz for every step of it.

    Output layout::

        {log_dir}/viz/epoch_{epoch:04d}/
            {task}/episode{ep_idx:04d}/step_{k:03d}/{stage}/...

    ``tasks`` whitelist:
        None  -> all tasks (default).
        list  -> only those tasks; unknown names are silently dropped.
        []    -> caller should already have skipped this call.

    All forward ops go through ``agent._net_mod`` so only rank 0 should call
    this — other ranks must not enter any NCCL collectives between
    ``agent.eval()`` and the final ``agent.train()``.
    """
    assert len(dataset) > 0, "empty dataset"

    rng = np.random.default_rng(seed if seed is not None else epoch)
    # RLBench's dataset.index keys episodes by integer ``ep_idx``.
    groups = group_dataset_by_episode(
        dataset, task_field="task", episode_field="ep_idx",
    )
    if not groups:
        print(f"[viz] epoch {epoch}: empty dataset index, skipping")
        return

    if tasks is not None:
        whitelist = set(tasks)
        missing = whitelist - set(groups.keys())
        if missing:
            print(f"[viz] epoch {epoch}: viz_tasks contains unknown task(s) "
                  f"{sorted(missing)} — ignoring; available={sorted(groups.keys())}")
        groups = {t: eps for t, eps in groups.items() if t in whitelist}
        if not groups:
            print(f"[viz] epoch {epoch}: no tasks left after viz_tasks filter, skipping")
            return

    chosen = pick_one_episode_per_task(groups, rng)
    viz_root = os.path.join(log_dir, "viz", f"epoch_{epoch:04d}")

    # Tri-view layout for the stitched per-episode grids (rows=steps,
    # cols=views). Oblique renderer renames the same three slots.
    n_views = int(agent._net_mod.num_img)
    _renderer = getattr(agent._net_mod, "renderer", None)
    if getattr(_renderer, "oblique_views", False):
        view_names = ["oblique_a", "oblique_b", "oblique_c"]
    else:
        view_names = ["top", "front", "right"]

    total_steps = 0
    for task, ep_id, ds_indices in chosen:
        ep_name = f"episode{int(ep_id):04d}"
        num_steps = len(ds_indices)
        samples = [dataset[int(i)] for i in ds_indices]
        episode_dir = os.path.join(viz_root, task, ep_name)
        step_labels = [f"step_{k:03d}" for k in range(num_steps)]
        save_dirs = [os.path.join(episode_dir, s) for s in step_labels]
        extra_meta = [
            {
                "dataset_idx": int(ds_indices[k]),
                "task":        task,
                "ep_idx":      int(ep_id),
                "step_idx":    k,
                "num_steps":   num_steps,
                "gt_ignore_collisions": float(samples[k].get("ignore_collisions", -1)),
            }
            for k in range(num_steps)
        ]

        visualize_samples(
            agent, samples, save_dirs,
            stages=stages, cameras=cameras, extra_meta=extra_meta,
            apply_aug=True, forward_chunk=forward_chunk,
        )

        # Stitch the per-step tri-view overlays into one big grid per stage:
        # rows = steps, cols = the 3 views.
        try:
            stitch_episode_overlays(
                episode_dir, step_labels, stages=stages,
                n_views=n_views, view_names=view_names,
            )
        except Exception as e:
            import traceback
            print(f"[viz] stitch failed task={task} ep={ep_name}: {e}")
            traceback.print_exc()

        total_steps += num_steps
        print(f"[viz] epoch {epoch}: task={task} ep={ep_name} "
              f"saved {num_steps} step(s)")

    print(f"[viz] epoch {epoch}: saved {total_steps} step(s) across "
          f"{len(chosen)} task(s) under {viz_root}")
